## 数据集

### Pretrain 数据集
- LLaVA-CC3M-Pretrain-595K
- Chinese-LLaVA-Vision-Instructions
### SFT 数据集
- minimind-v_dataset

## 训练备注

由于训练的数据量不多，但是需要训练的参数比较多，因此可能会导致过拟合。
一开始训练的时候只冻结了视觉模型的参数，文本大模型和对齐层的参数是没有冻结的，后来发现过拟合后就冻结文本大模型全连接层和 embedding 层；但还是过拟合，就冻结文本大模型的全部参数。