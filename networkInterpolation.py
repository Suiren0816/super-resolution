
from collections import OrderedDict
import sys
import time
from PIL import Image
from models import SRResNet,Generator
from utils import *
from torch import nn

# 测试图像
imgPath = './results/test.png'

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generateNetworkInterpolation(alpha,model_path):
    # 预训练模型
    interpolation_checkpoint = model_path

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(interpolation_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint)

    generator.eval()
    model = generator

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.png')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/test_net_interpolation_alpha_' + str(alpha) + '.png')

    print('用时  {:.3f} 秒'.format(time.time() - start))

def main():
    print("0-100")
    alpha = float(sys.argv[1]) / 100.0  # 插值权重。
    """
    插值权重
    alpha = 0 仅使用ResNet网络参数
    alpha = 100 仅使用GAN网络参数
    """

    model_srgan_path = './results/checkpoint_srgan.pth'
    model_srresnet_path = './results/checkpoint_srresnet.pth'
    model_networkInterpolation_path = ('./results/checkpoint_networkInterpolation'
                                       + str(alpha) + '.pth'.format(int(alpha * 10)))
    """
    经过网络插值得到的新网络保存路径如上
    """

    model_srgan = torch.load(model_srgan_path)
    model_srresnet = torch.load(model_srresnet_path)
    model_networkInterpolation = OrderedDict()

    # 进行网络插值
    orderedDictResNetModelValue = model_srresnet['model']
    orderedDictGanModelValue = model_srgan['generator']

    """
    只需提取模型中的model里的参数
    """

    for key in orderedDictResNetModelValue.keys():
        value_ResNet = orderedDictResNetModelValue[key]
        key = 'net.' + key  # GAN模型生成器里的Key比ResNet里的model多一个.net
        value_G = orderedDictGanModelValue[key]
        model_networkInterpolation[key] = (1 - alpha) * value_ResNet + alpha * value_G

    torch.save(model_networkInterpolation, model_networkInterpolation_path)
    # 保存模型

    generateNetworkInterpolation(alpha,model_networkInterpolation_path)
    # 生成测试图像


if __name__ == '__main__':
    main()