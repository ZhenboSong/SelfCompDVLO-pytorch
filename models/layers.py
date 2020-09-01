import torch.nn as nn


def init_weights(layers):
    for p in layers.modules():
        if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
            nn.init.xavier_normal_(p.weight)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0.)
        elif isinstance(p, nn.BatchNorm2d):
            p.weight.data.fill_(1)
            p.bias.data.zero_()


def conv(inc, outc, ker=3, stride=1, pad=0, dilation=1):
    layer = nn.Sequential(
        nn.Conv2d(inc, outc, ker, stride, pad, dilation=dilation),
        nn.BatchNorm2d(outc),
        nn.ReLU(),
    )
    init_weights(layer)
    return layer


def up_conv(inc, outc, ker=3):
    layer = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(inc, outc, kernel_size=ker, stride=1, padding=1),
        nn.BatchNorm2d(outc),
        nn.ReLU(inplace=True),
    )
    init_weights(layer)
    return layer


def vgg_conv(inc, outc, ker=3, n_conv=2):
    layers = list()
    for i in range(n_conv - 1):
        layers.append(nn.Conv2d(inc, outc, kernel_size=ker, stride=1, padding=int(ker / 2)))
        layers.append(nn.BatchNorm2d(outc))
        layers.append(nn.ReLU(inplace=True))
        inc = outc
    layers.append(nn.Conv2d(inc, outc, kernel_size=ker, stride=2, padding=int(ker / 2)))
    layers.append(nn.BatchNorm2d(outc))
    layers.append(nn.ReLU(inplace=True))
    layer = nn.Sequential(*layers)
    init_weights(layer)
    return layer

