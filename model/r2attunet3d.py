import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import number_of_features_per_level

"""Reference
Recurrent Residual Unet 3D implemented based on https://arxiv.org/pdf/2105.02290.pdf.
"""


class SingleConv(nn.Module):
    # Convolution + Batch Norm + ReLU

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, num_groups=8, activation=True):
        super(SingleConv, self).__init__()

        if out_channels < num_groups:
            num_groups = 1

        if activation:
            # use only one group if the given number of groups is greater than the number of channels
            self.singleconv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ELU(inplace=True)
            )
        else:
            self.singleconv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            )

    def forward(self, x):
        return self.singleconv(x)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()

        self.conv1 = SingleConv(in_channels, out_channels, 1, padding=0, stride=2)
        self.conv2 = SingleConv(in_channels, out_channels, 3, padding=1, stride=2)
        self.conv3 = SingleConv(in_channels, out_channels, 5, padding=2, stride=2)

    def forward(self, x):
        down1 = self.conv1(x) + self.conv2(x) + self.conv3(x)
        # down2 = self.conv1(down1) + self.conv2(down1) + self.conv3(down1)

        return down1


class AttBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttBlock, self).__init__()
        self.W_g = SingleConv(F_g, F_int, kernel_size=1, padding=0, activation=False)

        self.W_x = SingleConv(F_l, F_int, kernel_size=1, padding=0, activation=False)

        self.psi = SingleConv(F_int, 1, kernel_size=1, padding=0, activation=False)

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)

        return x * psi


class RRCU(nn.Module):
    # Recurrent Residual Convolutional Unit

    def __init__(self, out_channels, t=2, kernel_size=3, **kwargs):
        super(RRCU, self).__init__()

        self.t = t

        self.conv = SingleConv(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):

        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)

        return x1


class RRConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2, kernel_size=3):
        super(RRConvBlock, self).__init__()

        self.module = nn.Sequential(
            RRCU(out_channels=out_channels, t=t, kernel_size=kernel_size),
            RRCU(out_channels=out_channels, t=t, kernel_size=kernel_size)
        )

        self.Conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.module(x)

        return x + x1


class Encode(nn.Module):
    def __init__(self, in_channels, out_channels, t=2, conv_kernel_size=3, pool_kernel_size=(1, 2, 2), pooling=True):
        super(Encode, self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.module = RRConvBlock(in_channels=in_channels, out_channels=out_channels, t=t, kernel_size=conv_kernel_size)

    def forward(self, x):

        if self.pooling:
            x = self.maxpool(x)

        x = self.module(x)

        return x


class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(1, 2, 2), padding=1, mode="nearest",
                 **kwargs):
        super(Decode, self).__init__()

        # self.upsample = InterpolateUpsampling(mode)
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            SingleConv(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding)
        )

        self.att = AttBlock(out_channels, out_channels, out_channels // 2)
        self.module = RRConvBlock(in_channels=out_channels, out_channels=out_channels, t=2,
                                  kernel_size=conv_kernel_size)

    def forward(self, encoder_features, x):
        upx = self.upconv(x)
        # upx=1,48,40,14,14
        # encoder_features=1,48,40,14,14
        x = self.att(upx, encoder_features)

        # Summation joining instead of concatenate
        x = encoder_features + x

        x = self.module(x)

        return x


class R2AttUNet3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_class,
                 f_maps=[16, 32, 64, 96],
                 testing=False,
                 **kwargs):
        super(R2AttUNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=4)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing
        self.final_activation = nn.Sigmoid()

        self.first_conv = Encode(in_channels, num_class, pooling=False)

        self.downsample = Downsampling(in_channels, in_channels)

        self.down1 = Encode(in_channels, f_maps[0], pooling=False)
        self.down2 = Encode(f_maps[0], f_maps[1])
        self.down3 = Encode(f_maps[1], f_maps[2])
        self.down4 = Encode(f_maps[2], f_maps[3])

        self.up1 = Decode(f_maps[3], f_maps[2])
        self.up2 = Decode(f_maps[2], f_maps[1])
        self.up3 = Decode(f_maps[1], f_maps[0])

        self.upsample = nn.ConvTranspose3d(f_maps[0], num_class, kernel_size=3, stride=2, padding=1)

        self.final_conv = nn.Conv3d(out_channels, num_class, 1)

    def forward(self, x):
        # print(x.shape)

        x1 = self.first_conv(x)
        # print(x1.shape)
        # 1,1,80,112,112
        x = self.downsample(x)
        # print(x.shape)

        x2 = self.down1(x)
        # print(x2.shape)

        x3 = self.down2(x2)
        # print(x3.shape)

        x4 = self.down3(x3)
        # print(x4.shape)

        x5 = self.down4(x4)
        # print(x5.shape)

        x = self.up1(x4, x5)
        # print(x.shape)

        x = self.up2(x3, x)
        # print(x.shape)

        x = self.up3(x2, x)
        # print(x.shape)
        # 1,16,40,56,56

        x = self.upsample(x, x1.size()[2:])
        # print(x.shape)

        x = nn.Softmax(1)(x)

        return x


if __name__ == '__main__':
    # pass
    # torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    print(device)
    model = R2AttUNet3D(in_channels=1, out_channels=1, num_class=2).to(device)
    x = torch.rand(1, 1, 80, 112, 112)
    x = x.to(device)
    x = model.forward(x)
    print(x)
    # x = torch.ones(1,1,5,5,5)
    # y = torch.zeros(1,4,5,5,5)
    # z = x*y
    # print(z)
