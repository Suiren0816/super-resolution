"""
该文件用于调试本项目
2024-3-13
"""
import sys
import torch
from collections import OrderedDict

model_srgan_path = './results/checkpoint_srgan.pth'
model_srresnet_path = './results/checkpoint_srresnet.pth'



model_srgan = torch.load(model_srgan_path)
model_srresnet = torch.load(model_srresnet_path)

for k,v in model_srresnet.items():
    print(str(k))
    #print(str(v))
    print("############################")
    # print(str(k) + " " + str(v))
print("-------------------------------------------------------------------")
for k,v in model_srresnet.items():
    print(str(k))
    print("############################")