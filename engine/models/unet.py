'''
    U-Net implementation (Ronneberger et al., 2015) for Detectron2.

    Adapted from 2019 Joris van Vugt (source: https://raw.githubusercontent.com/jvanvugt/pytorch-unet/master/unet.py).
    
    2021 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY

from engine import util


@META_ARCH_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()

        self.num_classes = cfg.INPUT.get('NUM_CLASSES', 10)
        self.in_channels = cfg.INPUT.get('NUM_INPUT_CHANNELS', 3)
        self.depth = cfg.MODEL.get('DEPTH', 5)
        self.num_features_exponent = cfg.MODEL.get('NUM_FEATURES_EXPONENT', 6)
        self.batch_norm = cfg.MODEL.get('BATCH_NORM', False)
        self.padding = True     # hard-coded to True to assert equal input and output dimensions
        self.upsampling_mode = cfg.MODEL.get('UPSAMPLING_MODE', 'upsample')
        assert self.upsampling_mode in ('upconv', 'upsample')
        self.register_buffer('pixel_mean', torch.Tensor(cfg.MODEL.get('PIXEL_MEAN')).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(cfg.MODEL.get('PIXEL_STD')).view(-1, 1, 1), False)

        lossWeights = cfg.MODEL.get('LOSS_WEIGHTS', None)
        if lossWeights is not None:
            lossWeights = torch.tensor(lossWeights).to(self.device)
        self.loss = nn.CrossEntropyLoss(weight=lossWeights)

        # build model
        prev_channels = self.in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (self.num_features_exponent + i), self.padding, self.batch_norm)
            )
            prev_channels = 2 ** (self.num_features_exponent + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (self.num_features_exponent + i), self.upsampling_mode, self.padding, self.batch_norm)
            )
            prev_channels = 2 ** (self.num_features_exponent + i)

        self.last = nn.Conv2d(prev_channels, self.num_classes, kernel_size=1)
    

    @property
    def device(self):
        return self.pixel_mean.device


    def preprocess_image(self, x):
        return x - self.pixel_mean / self.pixel_std


    
    def _forward_image(self, x):
        sz = x.size()
        x = x.to(self.device)
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)

        #TODO: interpolate if output size is not equal to input size (crude hack)
        sz_o = x.size()
        if sz[2] != sz_o[2] or sz[3] != sz_o[3]:
            x = F.interpolate(x, (sz[2], sz[3]))

        return x



    def forward(self, items):
        if self.training:
            loss = torch.tensor(0.0, dtype=self.pixel_mean.dtype, device=self.device)
            for item in items:
                img = item['image']
                sz = img.size()
                segmask = util.instances_to_segmask(item['instances'], (sz[1], sz[2]), class_offset=1)      # zero-class = negative
                pred = self._forward_image(item['image'].unsqueeze(0))
                loss += self.loss(pred, segmask.unsqueeze(0).to(self.device))
            loss /= len(items)
            return {
                'CE': loss
            }

        else:
            # inference
            pred = []
            for item in items:
                pred.append(self._forward_image(item['image'].unsqueeze(0)))
            return pred



class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)


    def forward(self, x):
        out = self.block(x)
        return out



class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)


    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]


    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out