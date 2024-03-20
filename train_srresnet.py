# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
import time

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import models
from models import SRResNet
from datasets import SRDataset
from utils import *


# 数据集参数
data_folder = './data/'          # 数据存放路径
crop_size = 96      # 高分辨率图像裁剪尺寸
scaling_factor = 4  # 放大比例

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量

# 学习参数
checkpoint = None   # 预训练模型路径，如果不存在则为None
batch_size = 128    # 批大小
start_epoch = 1     # 轮数起始位置
epochs = 128        # 迭代轮数
workers = 4         # 工作线程数
lr = 1e-4           # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 2           # 用来运行的gpu数量

cudnn.benchmark = True # 对卷积进行加速

"""
用tensorboad进行训练结果可视化
"""
writer = SummaryWriter() # 实时监控     使用命令 tensorboard --logdir runs  进行查看

def main():
    """
    训练.
    """
    global checkpoint,start_epoch,writer

    # 初始化
    model = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.SmoothL1Loss.to(device)

    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    # 定制化的dataloaders
    train_dataset = SRDataset(data_folder,split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    start_time = time.time()

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):

        model.train()  # 训练模式：允许使用批样本归一化

        loss_epoch = AverageMeter()  # 统计损失函数

        n_iter = len(train_loader)

        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式

            # 前向传播
            sr_imgs = model(lr_imgs)

            # 计算损失
            loss = criterion(sr_imgs, hr_imgs)  

            # 后向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新模型
            optimizer.step()

            # 记录损失值
            loss_epoch.update(loss.item(), lr_imgs.size(0))

            # 监控图像变化
            if i==(n_iter-2):
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)

            # 打印结果
            print("第 "+str(i)+ " 个batch结束。Time used:{:.3f} 秒".format(time.time()-start_time))
 
        # 手动释放内存              
        del lr_imgs, hr_imgs, sr_imgs

        # 监控损失值变化
        writer.add_scalar('SRResNet/SmoothL1_Loss', loss_epoch.val, epoch)

        # 保存预训练模型
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'results/checkpoint_srresnet.pth')

        progress = float(epoch) / float(epochs)
        progress = progress * 100.0
        print("epoch " + str(epoch) + " train finished! \n " +
              "process " + str(progress) + "% \n"
              "Time used : {:.3f} 秒".format(time.time() - start_time))
    
    # 训练结束关闭监控
    writer.close()


if __name__ == '__main__':
    main()
"""
啥
🐎 屁Wen
I need NVIDIA 4090!!!
echo "NIVDIA RTX 4090 IS not unNECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
echo "NIVDIA RTX 4090 IS NECESSARY!!!"
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
给我4090
就是把你的训练结果可视化
再怎么用
就是把你每一轮的训练结果输出出来，tensorboard显示
4090！!！
用布洛芬跟老陈换4090
"""