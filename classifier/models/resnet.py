from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torch.nn import Parameter


class ResNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101
    }

    def __init__(self, depth, pretrained=True, num_class=102, dropout=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        self.num_class = num_class
        self.dropout = dropout

        out_planes = self.base.fc.in_features

        # Append new layers
        self.classifier = nn.Linear(out_planes, self.num_class)
        init.kaiming_normal(self.classifier.weight, mode='fan_out')
        init.constant(self.classifier.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = F.avg_pool2d(x, x.size()[2:])
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
    model = ResNet(50, num_class=102)
    x = Variable(torch.zeros(30, 3, 256, 256), requires_grad=False)
    output = model(x)
    print(output.data.size())
