import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3]) 

        self.conv_dim_out = [64,64*4,128*4,256*4,512*4]  # c1,layer1,layer2,layer3,layer4
        self.conv_spatial_scale = [1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32] # c1,layer1,layer2,layer3,layer4

    def forward(self, x):
        c1 = self.model.conv1(x)
        c1_out = self.model.relu(c1)
        # c1_out = self.model.relu(self.model.bn1(c1))
        pool = self.model.maxpool(c1_out)

        l1 = self.model.layer1(pool)
        l2 = self.model.layer2(l1)
        l3 = self.model.layer3(l2)
        l4 = self.model.layer4(l3)

        return l3, l4

    def load_pretrained(self, model_file, verbose=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth

        m = torch.load(model_file)
        mk = m.keys()[:-2] # remove fc layers from model_file
        sd = self.state_dict()
        sdk = sd.keys()

        print("Loading pretrained model %s..."%(model_file))
        for ix,k in enumerate(mk):
            md = m[k]
            sk = sdk[ix]
            d = sd[sk]
            assert d.shape == md.shape
            if verbose:
                print("%s -> %s [%s]"%(k, sk, str(d.shape)))
            sd[sk] = md
        self.load_state_dict(sd)
        print("Loaded pretrained model %s"%(model_file))

if __name__ == '__main__':
    resnet = ResNet50()
    model_file = "/data/models/resnet50.pth"

    # m = torch.load(model_file)
    resnet.load_pretrained(model_file, False)
    m = resnet.model

    x = torch.FloatTensor(2,3,480,640)
    l3, l4 = resnet(x)
