import torch
import torch.nn as nn

def make_conv2d(in_channel, out_channel, kernel_size, strides, padding, drop_out=0., act='relu', batch_norm = False):
    layers = []

    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    if act == 'relu':
        layers.append(nn.ReLU())
    elif act == 'sigmoid':
        layers.append(nn.Sigmoid())

    if drop_out > .0:
        layers.append(nn.Dropout(p=drop_out))

    layers = nn.Sequential(*layers)    
    return layers


def make_ConvTranspose2d(in_channel, out_channel, kernel_size, strides, padding=0, output_padding = 0, drop_out=0., act='relu', batch_norm = False):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding, output_padding=output_padding))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    if act == 'relu':
        layers.append(nn.ReLU())

    if drop_out > .0:
        layers.append(nn.Dropout(p=drop_out))

    layers = nn.Sequential(*layers)    
    return layers


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, drop_rate, pooling = True, **kwargs):
        super(EncoderBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rate = drop_rate
        self.pooling = pooling
        
        self.conv1 = make_conv2d(self.in_channel , self.out_channel, kernel_size=3, strides=1, padding=1, act='relu', drop_out=self.rate)
        self.conv2 = make_conv2d(self.out_channel, self.out_channel, kernel_size=3, strides=1, padding=1, act='relu')
        if self.pooling:
            self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x
    
    
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, drop_rate, **kwargs):
        super(DecoderBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rate = drop_rate
        
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv_t1 = make_ConvTranspose2d(in_channel, in_channel, kernel_size=3, strides=2, padding=1, output_padding=1, act='relu') 
        self.encoder = EncoderBlock(in_channel*2, out_channel, drop_rate, pooling = False)
        
    def forward(self, x, skip_x):
        x = self.bn1(x)
        x = self.conv_t1(x)
        x = torch.cat([x, skip_x], axis=1)
        x = self.encoder(x)
        return x
    
    
class AttentionGate(nn.Module):
    
    def __init__(self, in_channel, out_channel, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.normal = make_conv2d(self.in_channel, self.out_channel, kernel_size=3, strides=1, padding=1, act='relu')
        self.down = make_conv2d(self.in_channel, self.out_channel, kernel_size=3, strides=2, padding=1, act='relu')
        self.conv = make_conv2d(self.in_channel, 1, kernel_size=1, strides=1, padding=0, act='sigmoid')
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
    def forward(self, x, skip_x):
        x = self.normal(x)
        down_skip_x = self.down(skip_x)
        x = x + down_skip_x
        x = self.conv(x)
        x = self.upsample(x)
        
        x = x * skip_x
        return x

    
class Unet(nn.Module):
    
    def __init__(self, **kwargs):
        super(Unet, self).__init__(**kwargs)
        
        self.encoder1 = EncoderBlock(3, 64, 0.1)
        self.encoder2 = EncoderBlock(64, 128, 0.1)
        self.encoder3 = EncoderBlock(128, 256, 0.2)
        self.encoder4 = EncoderBlock(256, 512, 0.2)
        
        self.center = EncoderBlock(512, 512, 0.3, pooling=False)
        
        self.attention4 = AttentionGate(512, 512) 
        self.decoder4 = DecoderBlock(512, 256, 0.2) 
        
        self.attention3 = AttentionGate(256, 256) 
        self.decoder3 = DecoderBlock(256, 128, 0.2)  
        
        self.attention2 = AttentionGate(128, 128)
        self.decoder2 = DecoderBlock(128, 64, 0.1)
        
        self.attention1 = AttentionGate(64, 64)
        self.decoder1 = DecoderBlock(64, 64, 0.1)
        
        self.final = make_conv2d(64, 3, kernel_size=3, strides=1, padding=1, act='sigmoid')
        
        
    def forward(self, x):
        x, e1 = self.encoder1(x)
        x, e2 = self.encoder2(x)
        x, e3 = self.encoder3(x)
        x, e4 = self.encoder4(x)
        
        x = self.center(x)
        
        a4 = self.attention4(x, e4)
        x = self.decoder4(x, a4)
        
        a3 = self.attention3(x, e3)
        x = self.decoder3(x, a3)
        
        a2 = self.attention2(x, e2)
        x = self.decoder2(x, a2)
        
        a1 = self.attention1(x, e1)
        x = self.decoder1(x, a1)
        
        x = self.final(x)
        return x