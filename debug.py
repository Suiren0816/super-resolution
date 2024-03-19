import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from models import Generator, Discriminator
from datasets import SRDataset
from utils import *

# 数据集参数
data_folder = './data/'
crop_size = 96
scaling_factor = 4

# 模型参数
latent_dim = 100
output_dim = 3 * crop_size * crop_size  # RGB图像

# 学习参数
batch_size = 64
lr = 0.00005
n_critic = 5  # 判别器更新次数
clip_value = 0.01  # 参数截断值
epochs = 50

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 2
cudnn.benchmark = True
writer = SummaryWriter()

def main():
    generator = Generator(latent_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    train_dataset = SRDataset(data_folder, split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    start_time = time.time()

    for epoch in range(epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = generator(z)
                real_validity = discriminator(hr_imgs)
                fake_validity = discriminator(fake_imgs.detach())

                # 计算 WGAN-GP 中的 Wasserstein 距离
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Clip parameters of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # 训练生成器
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)

            # 生成器的损失函数
            g_loss = -torch.mean(fake_validity)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            # 记录生成的图像
            if i % 200 == 0:
                img_grid = make_grid(fake_imgs[:4], normalize=True)
                writer.add_image(f"Generated Images/{epoch}_{i}", img_grid, global_step=epoch * len(train_loader) + i)

    writer.close()

if __name__ == '__main__':
    main()
