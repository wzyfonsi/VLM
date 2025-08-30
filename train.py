from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, llm_model_path = 'E:\Python\Workspace\Multi-Model\Model\Qwen2.5-0.5B-Instruct',
                 vision_model_path = 'E:\Python\Workspace\Multi-Model\Model\siglip-base-patch16-224',
                 freeze_vision_model = True,
                 image_pad_num = 49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
        # 利用两个线性层实现视觉向文本对齐 （×4是图片压缩的步骤，减小图片token的序列长度来增加隐藏层的维度）
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        # 预训练中冻结视觉和语言模型的参数，只训练两个全连接层的权重来训练图片向文本对齐
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self,input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        # 像素值输入到视觉模型中得到图片的embedding
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        # b:batch size; s:sequence length, 即每张图片被分成多少个片段，也就是每张图片中被编码的patch数（也可以理解为token数）
        # d: embedding dim, 每个patch被编码后的向量维度
        b, s, d = image_embeds.shape
        # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        # 如果图片的token数太长的话一方面会导致生成的速度下降， 另一方面是训练时文本的长度没有很长，如果图像太长可能会影响模型的指令遵循
        image_embeds = image_embeds.view(b,-1, d*4)
        # 将图片embedding映射到和文本embedding相同的空间
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))

        text_embeds = text_embeds.to(image_features.dtype)

        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask = attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        # 数据集中已经使用占位符给image占据了位置，只需要定位到占位符的位置并替换就行了
        # 定位到占位符的位置
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # 将占位符替换成图像的embedding
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds

class MyDataset(Dataset):
    def __init__(self, image_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r',encoding = 'utf-8') as f:
            self.datas = json.load(f)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            #预训练只取了对话的第一轮，即第一条数据 ？（为什么这样做）
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num) # 将数据集中的图像占位符替换为Qwen中图像占位符的表示
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id]*len(q_input_ids)+a_input_ids
            # 做shift偏移，在计算损失的时候，使用当前token去预测下一个token
            # transformer官方的训练类中已做偏移处理
            input_ids = input_ids[:-1]
            labels = labels[1:]

            image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images = image)['pixel_values']

        except:
            # 将损坏的图片替换为白底的图片，对图片的描述就是内容为空
            default_image = Image.new('RGB', (224,224), color='white')
            pixel_values = self.processor(text = None, images = default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }

# 训练时每个batch的长度要保持一致， 这个类的作用是把一个batch里面所有样本全部填充到batch里长度最长的长度，然后转成tensor
class MyDataCollator:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def __call__(self,features:List[Dict[str,Any]]) -> Dict[str,Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id]*(max_len-len(feature['input_ids'])))
            labels.append(feature['labels']+[self.tokenizer.pad_token_id]*(max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels':torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }


if __name__ == '__main__':
    config = VLMConfig(vision_model_path='E:\Python\Workspace\Multi-Model\Model\siglip-base-patch16-224', image_pad_num=49)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        # 梯度累加
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    # 使用trainer类训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()






