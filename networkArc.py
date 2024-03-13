import sys
import torch
from collections import OrderedDict

net_PSNR_path = './models/RRDB_PSNR_x4_old_arch.pth'
net_ESRGAN_path = './models/RRDB_ESRGAN_x4_old_arch.pth'



net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)

for k,v in net_ESRGAN.items():
    print (str(k))
    print("############################")
    # print(str(k) + " " + str(v))
