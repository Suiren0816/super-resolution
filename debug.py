"""
该文件用于调试本项目
2024-3-13
"""
from PIL import Image, ImageStat
import networkInterpolation

alpha = 0.7

def getBrightness(img_path):
    img = Image.open(img_path).convert('L')
    stat = ImageStat.Stat(img)
    return stat.mean[0]

if __name__ == '__main__':
    img_path = './results/test.png'
    print(getBrightness(img_path))
    img_gan_path = './results/test_srgan.png'
    print(getBrightness(img_gan_path))
    img_interpolation_path = './results/test_net_interpolation_alpha_' + str(alpha) + '.png'
    print(getBrightness(img_interpolation_path))