import torch
from collections import OrderedDict

alpha = 0.8 # 插值权重。
"""
插值权重
alpha = 0 仅使用ResNet网络参数
alpha = 1 仅使用GAN网络参数
"""

model_srgan_path = './results/checkpoint_srgan.pth'
model_srresnet_path = './results/checkpoint_srresnet.pth'
model_model5_path = './results/checkpoint_model5.pth'.format(int(alpha*10))
"""
经过网络插值得到的新网络保存路径如上
"""

model_srgan = torch.load(model_srgan_path)
model_srresnet = torch.load(model_srresnet_path)
model_model5 = OrderedDict()


# 进行网络插值
orderedDictResNetModelValue = model_srresnet['model']
orderedDictGanModelValue = model_srgan['generator']
"""
只需提取模型中的model里的参数
"""

for key in orderedDictResNetModelValue.keys():
    value_ResNet = orderedDictResNetModelValue[key]
    key = 'net.' + key #GAN模型生成器里的Key比ResNet里的model多一个.net
    value_G = orderedDictGanModelValue[key]
    model_model5[key] = (1-alpha)*value_ResNet + alpha*value_G

torch.save(model_model5, model_model5_path)