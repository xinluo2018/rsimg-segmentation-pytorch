## author: Dual-Star
## create: 2024.3.30;  
## modify by xin luo: 2024.5.20
## des: pytorch version watnet, 
## note: the watnet is very similar to the deeplabv3plus_mobilev2 in model_seg/deeplabv3plus_mobilev2.py; 
##       and the deeplabv3plus_mobilev2 have more parameters. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seg.deeplabv3plus as deeplabv3plus
import model_backbone.mobilenet as mobilenet

def conv1x1_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv3x3_bn_relu(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, \
                            kernel_size=3, stride=2, padding=1, output_padding=1),    ### in tensorflow version watnet, kernel_size=3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class mobilenet_feat(nn.Module):
    ## Get the 3rd, 9th, 39th-layers feature. 
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = mobilenet.MobileNetV2(num_bands=self.in_channels, num_classes=2)

    def forward(self, input):
        x = self.backbone.head(input)
        # Extract low-dimensional features
        x = self.backbone.body.inverted_0(x)      # channel -> 16, size -> 1/2,
        low_feat = x
        # Extract mid-dimensional features
        x = self.backbone.body.inverted_1(x)      # channel -> 24, size -> 1/4
        mid_feat = x
        # Extract high-dimensional features
        x = self.backbone.body.inverted_2(x)
        x = self.backbone.body.inverted_3(x)
        x = self.backbone.body.inverted_4(x)      #   channel -> 96, size -> 1/16
        high_feat = x
        return low_feat, mid_feat, high_feat

class watnet(nn.Module):
    def __init__(self, num_bands, 
                num_classes=2,
                aspp_atrous_rates=(6, 12, 18)):
        super().__init__()
        self.name = 'watnet'
        self.in_channels = num_bands
        self.channels_feas_mobilenet = [16, 24, 96]   ## the channels of low, mid, and high-level features.
        self.atrous_rates = aspp_atrous_rates
        # get multiscale features. 
        self.backbone = mobilenet_feat(self.in_channels)
        self.aspp = deeplabv3plus.aspp(in_channels=self.channels_feas_mobilenet[2], atrous_rates=self.atrous_rates)
    
        self.mid_layer = conv1x1_bn_relu(self.channels_feas_mobilenet[1], 48)
        self.high_mid_layer = nn.Sequential(
                        conv3x3_bn_relu(48+self.aspp.out_channels, 128),
                        conv3x3_bn_relu(128, 128)
                        )
        self.low_layer = conv1x1_bn_relu(self.channels_feas_mobilenet[0], 48)
        self.high_mid_low_layer = nn.Sequential(
                        deconv3x3_bn_relu(128+48, 256),
                        nn.Dropout(0.5),
                        conv1x1_bn_relu(256, 128),
                        conv3x3_bn_relu(128, 128),
                        nn.Dropout(0.1),
                        )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                    nn.Sigmoid())
        else: 
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1),
                    nn.Softmax(dim=1))
        # Initialize model parameters.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    def forward(self, input):
        fea_low, fea_mid, fea_high = self.backbone(input)
        ### ------high level feature
        x_high = self.aspp(fea_high)            # output channels:256
        x_high = F.interpolate(x_high, fea_mid.size()[-2:], mode='bilinear', align_corners=True)
        ### ------ mid-level feature, and concatenate high level feature.
        x_mid = self.mid_layer(fea_mid)
        x_high_mid = torch.cat([x_high, x_mid], dim=1)
        x_high_mid = self.high_mid_layer(x_high_mid)
        x_high_mid = F.interpolate(x_high_mid, fea_low.size()[-2:], mode='bilinear', align_corners=True)
        ### ------low-level feature, and concatenate high and mid level features.
        x_low = self.low_layer(fea_low)
        x_high_mid_low = torch.cat([x_high_mid, x_low], dim=1)
        x_high_mid_low = self.high_mid_low_layer(x_high_mid_low)
        ### output layer
        out_prob = self.outp_layer(x_high_mid_low)
        return out_prob

