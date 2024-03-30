## author: Dual-Star
## creat: 2024.3.30
## des: deeplabv3plus model with Mobilev2 backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.deeplabv3plus as deeplabv3plus
import model.mobilenet as mobilenet

# Get feature of different depths extracted through mobilenet
class mobilenet_get_feat(nn.Module):

    def __init__(self, in_ch_num):
        super().__init__()
        self.in_ch_num = in_ch_num
        self.backbone = mobilenet.MobileNetV2(num_bands=self.in_ch_num, num_classes=2)

    def forward(self, input):
        head = self.backbone.head(input)

        # Extract low-dimensional features
        low_feat = self.backbone.body.inverted_0(head)                     # low_feat_ch=16
        _low_feat = self.backbone.body.inverted_1[0].conv[0](low_feat)
        _low_feat = self.backbone.body.inverted_1[0].conv[1](_low_feat)
        _low_feat = self.backbone.body.inverted_1[0].conv[2](_low_feat)    # _low_feat_ch=96

        # Extract mid-dimensional features
        mid_feat = self.backbone.body.inverted_1(low_feat)                 # mid_feat_ch=24
        _mid_feat = self.backbone.body.inverted_2[0].conv[0](mid_feat)
        _mid_feat = self.backbone.body.inverted_2[0].conv[1](_mid_feat)
        _mid_feat = self.backbone.body.inverted_2[0].conv[2](_mid_feat)    # _mid_feat_ch=144

        # Extract high-dimensional features
        high_feat = self.backbone.body.inverted_4(
            self.backbone.body.inverted_3(self.backbone.body.inverted_2(mid_feat)))    # high_feat_ch=96
        _hig_feat = self.backbone.body.inverted_5[0].conv[0](high_feat)
        _hig_feat = self.backbone.body.inverted_5[0].conv[1](_hig_feat)
        _hig_feat = self.backbone.body.inverted_5[0].conv[2](_hig_feat)                # _high_feat_ch=576

        # print(f"{_high_feat.size()=}")
        # print(f"{_mid_feat.shape=}")
        # print(f"{_low_feat.shape=}")
        return _low_feat, _mid_feat, _hig_feat


class watnet(nn.Module):
    def __init__(self, patch_size, in_channel, atrous_rates=(6, 12, 18), out_channel=1):
        super().__init__()
        self.name = 'watnet'
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        # dilation rate
        self.atrous_rates = atrous_rates
        # Obtain high, mid and low dimensional features from mobilenet network
        self.backbone = mobilenet_get_feat(self.in_channel)
        # Get the characteristics of the aspp layer of deeplabv3plus, aspp_ch=128
        self.aspp = deeplabv3plus.aspp(in_channels=576, atrous_rates=self.atrous_rates)
        # High-dimensional feature upsampling
        self.high_up = nn.Upsample(size=self.patch_size // 4, mode='bilinear', align_corners=True)
        # High-Mid concat feature upsampling
        self.mid_up = nn.Upsample(size=self.patch_size // 2, mode='bilinear', align_corners=True)

        # The features extracted by MobileNet and the processing block before merging the high-dimensional features.
        # input mid_feat, ch 144
        self.mid_treat = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=48, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(48),  # num_feature = 48
            nn.ReLU()
        )

        # The mid features extracted by MobileNet and the processing block before merging the high-dimensional features.
        # input low_feat, ch 96
        self.low_treat = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(48),  # num_feature = 48
            nn.ReLU()
        )

        # The processing block after fusion of mid and high-dimensional features.
        # input mid_high_cat, ch 128+48
        self.mid_high_cat_treat = nn.Sequential(
            nn.Conv2d(in_channels=176, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # The processing block after fusion of high_mid_cat and low-dimensional features.
        # input high_mid_low_cat, ch 128+48
        self.low_mid_cat_treat = nn.Sequential(
            nn.ConvTranspose2d(in_channels=176, out_channels=128, kernel_size=3, stride=(2, 2), padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # The final output
        self.last_output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1, padding='same'),
            nn.Sigmoid()
        )

        # Initialize model parameters.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        low_feat, mid_feat, high_feat = self.backbone(input)
        high_aspp = self.aspp(high_feat)
        high_upsample = self.high_up(high_aspp)
        mid_treated = self.mid_treat(mid_feat)
        low_treated = self.low_treat(low_feat)

        # Concatenate mid and high-dimensional features.
        mid_concat = torch.cat((mid_treated, high_upsample), dim=1)
        mid_high_cat_treated = self.mid_high_cat_treat(mid_concat)
        mid_upsample = self.mid_up(mid_high_cat_treated)

        # Concatenate mid-high and low-dimensional features.
        low_concat = torch.cat((mid_upsample, low_treated), dim=1)
        low_mid_cat_treated = self.low_mid_cat_treat(low_concat)
        model = self.last_output(low_mid_cat_treated)
        # print(f'{model.shape=}')
        return model

# model = watnet(patch_size=512, in_channel=6,out_channel=1,aspp_channel=32,atrous_rates=[6,12,18]).to(device)
# print(model)
