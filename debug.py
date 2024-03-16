"""
该文件用于调试本项目
2024-3-13
"""
import sys
import torch
from collections import OrderedDict

alpha = 0.8

model_srgan_path = './results/checkpoint_srgan.pth'
model_srresnet_path = './results/checkpoint_srresnet.pth'
model_model5_path = './results/checkpoint_model5.pth'.format(int(alpha*10))


model_srgan = torch.load(model_srgan_path)
model_srresnet = torch.load(model_srresnet_path)
model_model5 = OrderedDict()


# 进行网络插值
orderedDictResNetModelValue = model_srresnet['model']
orderedDictGanModelValue = model_srgan['generator']


for key in orderedDictResNetModelValue.keys():
    value_ResNet = orderedDictResNetModelValue[key]
    key = 'net.' + key
    value_G = orderedDictGanModelValue[key]
    model_model5[key] = (1-alpha)*value_ResNet + alpha*value_G

torch.save(model_model5, model_model5_path)