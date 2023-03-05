import torch
import torch.nn.functional as F
from torch import nn


# x 输入，x_width :结果 尺寸,kernel_width 卷积核尺寸
def zero_mean_norm(x, x_width, kernel_width):
    w = nn.Parameter(torch.full([1, 1, kernel_width, kernel_width], 1 / (kernel_width * kernel_width)))
    x_mean_reduced = F.conv2d(input=x, weight=w, stride=kernel_width)  # 输入通道数，输出通道数
    x_mean_expanded = F.interpolate(x_mean_reduced, scale_factor=(x_width / x_mean_reduced.shape[2]), mode='nearest')

    return x - x_mean_expanded


# def test():
#     x = torch.full([2, 1, 64, 64], 1.)
#     w = nn.Parameter(torch.full([1, 1, 3, 5], 1.2))
#     x_mean_reduced = F.conv2d(input=x, weight=w, stride=1, padding=(1, 2))  # 输入通道数，输出通道数
#     print(x_mean_reduced.shape)


def aver_pool(x, k_width):  # 平均池化，求局部块均值
    out = F.avg_pool2d(x, k_width, k_width)
    return out


def activate(x, acti_mode):
    if acti_mode == 0:
        return x
    elif acti_mode == 1:
        return torch.sigmoid(x)
    elif acti_mode == 3:
        return F.leaky_relu(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder 1
        self.conv1_1 = nn.Conv2d(1, 8, 1, 1)  # 由 1张64*64 到 8 张64*64

        self.conv1_2_1 = nn.Conv2d(8, 16, (3, 7), 1, padding=(1, 3))
        self.conv1_2_2 = nn.Conv2d(8, 16, (7, 3), 1, padding=(3, 1))  # 矩形卷积核

        self.conv1_3 = nn.Conv2d(16, 64, 1, 1)
        self.conv1_4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv1_5 = nn.Conv2d(64, 16, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder 2
        self.conv2_1 = nn.Conv2d(1, 16, 1, 1)  # 由 1张32*32 到 16 张32*32

        self.conv2_2_1 = nn.Conv2d(16, 32, (3, 5), 1, padding=(1, 2))
        self.conv2_2_2 = nn.Conv2d(16, 32, (5, 3), 1, padding=(2, 1))  # 矩形卷积核

        self.conv2_3 = nn.Conv2d(32, 128, 1, 1)
        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv2_5 = nn.Conv2d(128, 32, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder 3
        self.conv3_1 = nn.Conv2d(1, 32, 1, 1)  # 由 1张16*16 到 32 张16*16

        self.conv3_2_1 = nn.Conv2d(32, 64, (1, 3), 1, padding=(0, 1))
        self.conv3_2_2 = nn.Conv2d(32, 64, (3, 1), 1, padding=(1, 0))  # 矩形卷积核

        self.conv3_3 = nn.Conv2d(64, 256, 1, 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv3_5 = nn.Conv2d(256, 64, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder 4
        self.conv4_1 = nn.Conv2d(1, 64, 1, 1)  #
        self.conv4_2 = nn.Conv2d(64, 32, 1, 1)

        self.conv4_3 = nn.Conv2d(32, 128, 1, 1)
        self.conv4_4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4_5 = nn.Conv2d(128, 32, 1, 1)

        self.c_out = nn.Conv2d(32, 36, 1, 1)

    def forward(self, x):
        img64 = zero_mean_norm(x, 64, 8)
        img32 = zero_mean_norm(aver_pool(x, 2), 32, 8)
        img16 = zero_mean_norm(aver_pool(x, 4), 16, 8)
        img8 = zero_mean_norm(aver_pool(x, 8), 8, 8)

        out = self.encoder_1(img64)
        out = self.encoder_2(img32, out)
        out = self.encoder_3(img16, out)
        y = self.encoder_4(img8, out)
        y = F.softmax(y, dim=1)
        y = torch.flatten(y, 2, 3)
        return y

    def encoder_1(self, img64_1):
        act_mode = 3
        x_c1 = self.conv1_1(img64_1)
        x_c1 = activate(x_c1, act_mode)  # 64_8
        x_c2_1 = self.conv1_2_1(x_c1)
        x_c2_1 = activate(x_c2_1, act_mode)

        x_c2_2 = self.conv1_2_2(x_c1)
        x_c2_2 = activate(x_c2_2, act_mode)

        x_c2 = torch.add(x_c2_1, x_c2_2)

        x_c3 = self.conv1_3(x_c2)
        x_c3 = activate(x_c3, act_mode)

        x_c4 = self.conv1_4(x_c3)
        x_c4 = activate(x_c4, act_mode)

        x_c5 = self.conv1_5(x_c4)
        x_c5 = activate(x_c5, act_mode)

        x_a6 = torch.add(x_c2, x_c5)

        x_p = self.pool1(x_a6)

        return x_p  # 32_16

    def encoder_2(self, img32_1, x_3232_16):
        act_mode = 3
        x_c1 = self.conv2_1(img32_1)
        x_c1 = activate(x_c1, act_mode)  # 32_16
        x_c1 = torch.add(x_c1, x_3232_16)

        x_c2_1 = self.conv2_2_1(x_c1)
        x_c2_1 = activate(x_c2_1, act_mode)

        x_c2_2 = self.conv2_2_2(x_c1)
        x_c2_2 = activate(x_c2_2, act_mode)

        x_c2 = torch.add(x_c2_1, x_c2_2)

        x_c3 = self.conv2_3(x_c2)
        x_c3 = activate(x_c3, act_mode)

        x_c4 = self.conv2_4(x_c3)
        x_c4 = activate(x_c4, act_mode)

        x_c5 = self.conv2_5(x_c4)
        x_c5 = activate(x_c5, act_mode)

        x_a6 = torch.add(x_c2, x_c5)

        x_p = self.pool1(x_a6)

        return x_p

    def encoder_3(self, img16_1, x_1616_32):
        act_mode = 3
        x_c1 = self.conv3_1(img16_1)
        x_c1 = activate(x_c1, act_mode)  # 32_16
        x_c1 = torch.add(x_c1, x_1616_32)

        x_c2_1 = self.conv3_2_1(x_c1)
        x_c2_1 = activate(x_c2_1, act_mode)

        x_c2_2 = self.conv3_2_2(x_c1)
        x_c2_2 = activate(x_c2_2, act_mode)

        x_c2 = torch.add(x_c2_1, x_c2_2)

        x_c3 = self.conv3_3(x_c2)
        x_c3 = activate(x_c3, act_mode)

        x_c4 = self.conv3_4(x_c3)
        x_c4 = activate(x_c4, act_mode)

        x_c5 = self.conv3_5(x_c4)
        x_c5 = activate(x_c5, act_mode)

        x_a6 = torch.add(x_c2, x_c5)

        x_p = self.pool1(x_a6)

        return x_p

    def encoder_4(self, img8_1, x_88_64):
        act_mode = 3
        x_c1 = self.conv4_1(img8_1)
        x_c1 = activate(x_c1, act_mode)  # 32_16
        x_c1 = torch.add(x_c1, x_88_64)

        x_c2 = self.conv4_2(x_c1)

        x_c3 = self.conv4_3(x_c2)
        x_c3 = activate(x_c3, act_mode)

        x_c4 = self.conv4_4(x_c3)
        x_c4 = activate(x_c4, act_mode)

        x_c5 = self.conv4_5(x_c4)
        x_c5 = activate(x_c5, act_mode)

        x_a6 = torch.add(x_c2, x_c5)

        out = self.c_out(x_a6)

        return out
