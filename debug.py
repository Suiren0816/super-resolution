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

model_model5 = OrderedDict()
optimizer_model5 = OrderedDict()


# 进行网络插值
orderedDictResNetModelValue = model_srresnet['model']
orderedDictGanModelValue = model_srgan['generator']

for key, value in orderedDictResNetModelValue:

    value_gan = orderedDictGanModelValue[key]
    model_model5[key] = (1 - alpha) * value + alpha * value_gan

orderedDictResNetOptimizerValue = model_srresnet['optimizer']
orderedDictGanOptimizerValue = model_srgan['optimizer_g']

for key, value in orderedDictResNetOptimizerValue:

    value_gan = orderedDictGanModelValue[key]
    optimizer_model5[key] = (1 - alpha) * value + alpha * value_gan
#你有退烧药吗

torch.save({
            'epoch': 130,
            'model': model_model5.state_dict(),
            'optimizer': optimizer_model5.state_dict()
        }, 'results/checkpoint_model5.pth')
# 保存模型
# torch.save(model_model5, model_model5_path)