import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv_dim_out = [64,128,256,512,512]
        self.conv_spatial_scale = [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/16]
        c_dims = self.conv_dim_out

        inplace = True

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(64, c_dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dims[0], 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(128, c_dims[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dims[1], 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(256, c_dims[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dims[2], 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, c_dims[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(c_dims[3], 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, c_dims[4], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.max_pool2d(conv1))
        conv3 = self.conv3(self.max_pool2d(conv2))
        conv4 = self.conv4(self.max_pool2d(conv3))
        conv5 = self.conv5(self.max_pool2d(conv4))
        return conv4, conv5

    def load_pretrained(self, model_file, verbose=True):
        # https://download.pytorch.org/models/vgg16-397923af.pth
        
        m = torch.load(model_file)
        mk = list(m.keys())[:26]
        sd = self.state_dict()
        sdk = list(sd.keys())

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
