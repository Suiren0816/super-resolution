"""
该文件用于调试本项目
2024-3-13
"""
import sys
import torch
from collections import OrderedDict

alpha = 0.5

model_srgan_path = './results/checkpoint_srgan.pth'
model_srresnet_path = './results/checkpoint_srresnet.pth'
model_model5_path = './resultes/checkpoint_model5.pth'.format(int(alpha*10))


model_srgan = torch.load(model_srgan_path)
model_srresnet = torch.load(model_srresnet_path)
model_model5 = OrderedDict

#进行网络插值
for k,v in model_srresnet.items():
    v_gan =model_srgan[k]
    model_model5[k] =(1 - alpha) * v + alpha * v_gan
    print("############################")
    # print(str(k) + " " + str(v))

# 保存模型
torch.save(model_model5, model_model5_path)