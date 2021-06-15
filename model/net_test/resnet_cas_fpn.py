import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from config import CONFIG

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # Figure5(左) Block
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
        # Figure5(右) Block
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_ = nn.Linear(512 * block.expansion, num_classes)

        # 由于要输出到rpn中，所以每一个层都会用于预测
        # 每一层都会向各层采样和自己这层的压缩结果，不需要改变大小
        self.gettop_c2=nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.gettop_c3=nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0)
        self.gettop_c4 = nn.Conv2d(1024,1024, kernel_size=1, stride=1, padding=0)
        self.gettop_c5 = nn.Conv2d(2048,2048, kernel_size=1, stride=1, padding=0)

        # 3*3去除混跌
        self.smooth_c2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth_c3 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.smooth_c4 = nn.Conv2d(1024,1024, kernel_size=3, stride=1, padding=1)
        self.smooth_c5 = nn.Conv2d(2048,2048, kernel_size=3, stride=1, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def upsample_add(self,x1,x2,x3,y):
        _,_,h,w=y.size()
        return F.upsample(x1,size=(h,w),mode='bilinear')+F.upsample(x2,size=(h,w),mode='bilinear')+F.upsample(x3,size=(h,w),mode='bilinear')+y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        # 全部横向压缩
        p5=self.gettop_c5(x3)
        p4=self.gettop_c4(x2)
        p3=self.gettop_c3(x1)
        p2=self.gettop_c2(x)

        # 生成四个重采样之后的特征图
        p5=self.upsample_add(p3,p4,p2,p5)
        p4=self.upsample_add(p5,p3,p2,p4)
        p3=self.upsample_add(p2,p4,p5,p3)
        p2=self.upsample_add(p3,p4,p5,p2)

        #所有层经过3*3去重叠
        p5=self.smooth_c5(p5)
        p4=self.smooth_c5(p4)
        p3=self.smooth_c5(p3)
        p2=self.smooth_c5(p2)

        # 对于每一个
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)

        # 这里牵扯到会生成多个feature map,就会有多个输出，怎么获得这多个输出
        # 输出的大小也不一样，2048-1024-512-256四种情况
        return x


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    cfg = CONFIG()
    model = ResNet(BasicBlock, [3, 4, 6, 3],cfg.num_class, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    cfg=CONFIG()
    model = ResNet(Bottleneck, [3, 4, 6, 3],cfg.num_class, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model