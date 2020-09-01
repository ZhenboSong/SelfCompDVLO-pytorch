import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import conv, vgg_conv, up_conv, init_weights


class RGBNet(nn.Module):
    def __init__(self):
        super(RGBNet, self).__init__()
        # RGB stream simple FPN network  # 346856 parameters
        # pyramid 0
        self.layer0 = nn.Sequential(
            self.conv(3, 16, 3, 1, 1),
            self.conv(16, 16, 3, 1, 2, dilation=2),
            self.conv(16, 16, 3, 1, 2, dilation=2),
        )
        self.lat0 = self.conv(16, 8, 1, 1)
        self.out0 = self.out_conv(8, 8, 3, 1, 1)

        # pyramid 1
        self.layer1 = nn.Sequential(
            self.conv(16, 32, 3, 2, 1),
            self.conv(32, 32, 3, 1, 2, dilation=2),
            self.conv(32, 32, 3, 1, 2, dilation=2),
        )
        self.lat1 = self.conv(32, 8, 1, 1)
        self.out1 = self.out_conv(8, 8, 3, 1, 1)

        # pyramid 2
        self.layer2 = nn.Sequential(
            self.conv(32, 64, 3, 2, 1),
            self.conv(64, 64, 3, 1, 2, dilation=2),
            self.conv(64, 64, 3, 1, 2, dilation=2),
        )
        self.lat2 = self.conv(64, 8, 1, 1)
        self.out2 = self.out_conv(8, 8, 3, 1, 1)

        # pyramid 3
        self.layer3 = nn.Sequential(
            self.conv(64, 96, 3, 2, 1),
            self.conv(96, 96, 3, 1, 2, dilation=2),
            self.conv(96, 96, 3, 1, 2, dilation=2),
        )
        self.out3 = self.out_conv(96, 8, 1, 1)

    @staticmethod
    def out_conv(inc, outc, ker=3, stride=1, pad=0, dilation=1):
        layer = nn.Sequential(
            nn.Conv2d(inc, outc, ker, stride, pad, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.Tanh(),
        )
        for p in layer.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_normal_(p.weight)
                nn.init.constant_(p.bias, 0.01)
        return layer

    @staticmethod
    def conv(inc, outc, ker=3, stride=1, pad=0, dilation=1):
        layer = nn.Sequential(
            nn.Conv2d(inc, outc, ker, stride, pad, dilation=dilation),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
        )
        for p in layer.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_normal_(p.weight)
                nn.init.constant_(p.bias, 0.01)
        return layer

    def forward(self, rgb):
        rgb_0 = self.layer0(rgb)
        rgb_1 = self.layer1(rgb_0)
        rgb_2 = self.layer2(rgb_1)
        rgb_3 = self.layer3(rgb_2)
        x_3 = self.out3(rgb_3)
        x_2 = self.out2(self.lat2(rgb_2) + F.interpolate(x_3, scale_factor=2, mode='bilinear', align_corners=True))
        x_1 = self.out1(self.lat1(rgb_1) + F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=True))
        x_0 = self.out0(self.lat0(rgb_0) + F.interpolate(x_1, scale_factor=2, mode='bilinear', align_corners=True))
        # x_out3 = torch.mean(x_3, dim=1, keepdim=True)
        # x_out2 = torch.mean(x_2, dim=1, keepdim=True)
        # x_out1 = torch.mean(x_1, dim=1, keepdim=True)
        # x_out0 = torch.mean(x_0, dim=1, keepdim=True)
        return [x_3, x_2, x_1, x_0]


class FusionComp(nn.Module):
    def __init__(self):
        super(FusionComp, self).__init__()
        self.d_layer1 = vgg_conv(1, 16, 5, 1)
        self.d_layer2 = vgg_conv(16, 32, 3, 1)
        self.d_layer3 = vgg_conv(32, 64, 3, 2)
        self.d_layer4 = vgg_conv(64, 128, 3, 2)
        self.d_layer5 = vgg_conv(128, 128, 3, 2)

        self.rgb_layer1 = vgg_conv(3, 48, 5, 1)
        self.rgb_layer2 = vgg_conv(48, 96, 3, 1)
        self.rgb_layer3 = vgg_conv(96, 192, 3, 2)
        self.rgb_layer4 = vgg_conv(192, 384, 3, 2)
        self.rgb_layer5 = vgg_conv(384, 384, 3, 2)

        self.up_layer5 = up_conv(512, 256, 3)
        self.conv_layer4 = conv(256+512, 256, 3, 1, 1, 1)
        self.up_layer4 = up_conv(256, 128, 3)
        self.conv_layer3 = conv(128+256, 128, 3, 1, 1, 1)
        self.up_layer3 = up_conv(128, 64, 3)
        self.conv_layer2 = conv(64+128, 64, 3, 1, 1, 1)
        self.up_layer2 = up_conv(64, 32, 3)
        self.conv_layer1 = conv(32+64, 32, 3, 1, 1, 1)
        self.up_layer1 = up_conv(32, 16, 3)
        self.out_layer = self.out_conv(16, 1, 1)

    def out_conv(self, inc, outc, ker):
        layer = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=ker, stride=1, padding=int(ker/2)),
            # nn.BatchNorm2d(outc),
            nn.Sigmoid(),
        )
        init_weights(layer)
        return layer

    def forward(self, rgb, d):
        # rgb stream
        rgb1 = self.rgb_layer1(rgb)
        rgb2 = self.rgb_layer2(rgb1)
        rgb3 = self.rgb_layer3(rgb2)
        rgb4 = self.rgb_layer4(rgb3)
        rgb5 = self.rgb_layer5(rgb4)

        # depth stream
        d1 = self.d_layer1(d)
        d2 = self.d_layer2(d1)
        d3 = self.d_layer3(d2)
        d4 = self.d_layer4(d3)
        d5 = self.d_layer5(d4)

        # fusion stream
        up4 = self.up_layer5(torch.cat((rgb5, d5), dim=1))
        fu4 = self.conv_layer4(torch.cat((rgb4, d4, up4), dim=1))
        up3 = self.up_layer4(fu4)
        fu3 = self.conv_layer3(torch.cat((rgb3, d3, up3), dim=1))
        up2 = self.up_layer3(fu3)
        fu2 = self.conv_layer2(torch.cat((rgb2, d2, up2), dim=1))
        up1 = self.up_layer2(fu2)
        fu1 = self.conv_layer1(torch.cat((rgb1, d1, up1), dim=1))
        up0 = self.up_layer1(fu1)
        out = self.out_layer(up0)

        return out


