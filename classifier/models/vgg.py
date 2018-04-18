from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torch.nn import Parameter


class VGGNet(nn.Module):
    __factory = {
        '11': vgg11,
        '13': vgg13,
        '16': vgg16,
        '19': vgg19,
        '11bn': vgg11_bn,
        '13bn': vgg13_bn,
        '16bn': vgg16_bn,
        '19bn': vgg19_bn
    }

    def __init__(
        self, depth, with_bn=True, pretrained=True,
            num_class=0, dropout=0,
            input_size=(256, 256)):
        super(VGGNet, self).__init__()

        self.depth = depth
        self.with_bn = with_bn
        self.pretrained = pretrained

        # Construct base (pretrained) InceptionNet
        if self.with_bn:
            self.base = VGGNet.__factory['{:d}bn'.format(depth)](pretrained=pretrained)
        else:
            self.base = VGGNet.__factory['{:d}'.format(depth)](pretrained=pretrained)

        self.num_class = num_class
        self.dropout = dropout

        out_planes = int(512*(input_size[0]/32)*(input_size[1]/32))

        # Append new layers
        self.classifier = nn.Linear(out_planes, self.num_class)
        init.kaiming_normal(self.classifier.weight, mode='fan_out')
        init.constant(self.classifier.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base.features._modules.items():
            x = module(x)
        feature = x.view(x.size(0), -1)
        if self.dropout > 0:
            feature = self.drop(feature)
        output = self.classifier(feature)
        return output

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


if __name__ == '__main__':
    model = VGGNet(16, with_bn=False, num_class=102, input_size=(256, 256))
    x = Variable(torch.zeros(30, 3, 256, 256), requires_grad=False)
    output = model(x)
    print(output.data.size())
