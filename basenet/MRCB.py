import torch
from math import sqrt
from torch import nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate, scale


ROTATION_SCALE = {0 : 1, 
                  30 : 2/(1+sqrt(3)), 
                  45 : 1/sqrt(2), 
                  60 : 2/(1+sqrt(3)),
                  90 : 1,
                 }

def to_half(_list):
    return [tensor.half() for tensor in _list]

def to_float(_list):
    return [tensor.float() for tensor in _list]


'''
def conv1x1_bn_relu(in_planes, inter_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(inter_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3_bn_relu(in_planes, inter_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(inter_planes),
        nn.ReLU(inplace=True),
    )
'''


class conv1x1_bn_relu(nn.Module):
    def __init__(self, in_planes, inter_planes):
        super(conv1x1_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0)
        self.bn   = nn.BatchNorm2d(inter_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
    
class conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, inter_planes):
        super(conv3x3_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(inter_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class Multi_Rotation_Convolution_Block(nn.Module):
    def __init__(self, inplanes):
        super(Multi_Rotation_Convolution_Block, self).__init__()
        
        angles = [0, 30, 45, 60]
        scales = [ROTATION_SCALE[angle] for angle in angles]
        
        self.angles = [torch.tensor(angle).float() for angle in angles]
        self.scale_factors = [torch.tensor(scale).float() for scale in scales]
        
        self.conv_1 = conv3x3_bn_relu(inplanes, inplanes)
        self.conv_2 = conv3x3_bn_relu(inplanes, inplanes)
        self.conv_3 = conv3x3_bn_relu(inplanes, inplanes)
        self.conv_4 = conv3x3_bn_relu(inplanes, inplanes)
        
        self.conv_last = conv3x3_bn_relu(inplanes, inplanes)

        self.reverse = torch.tensor(360).float()
        
        
    def forward(self, x):
        
        """ smoothing """
        x = torch.nn.functional.avg_pool2d(x, (3, 3), stride=1, padding=1)
        
        
        """ scale down & rotate & conv3x3 """
        outs = [ scale(x.clone(), scale_factor.to(x.device)) for scale_factor in self.scale_factors ]
        outs = [ rotate(out, angle.to(out.device)) for out, angle in zip(outs, self.angles) ]
        
        outs = [self.conv_1(outs[0]), self.conv_2(outs[1]), self.conv_3(outs[2]), self.conv_4(outs[3])]
        
        
        """ recovery """
        outs = [ rotate(out, self.reverse.to(out.device)-angle.to(out.device)) for out, angle in zip(outs, self.angles) ]
        outs = [ scale(out, 1. / scale_factor.to(out.device)) for out, scale_factor in zip(outs, self.scale_factors) ]
        
        outs, _ = torch.max(torch.stack(outs), 0)
        #outs = torch.cat(outs, 1)
        
        outs = self.conv_last(outs)

        return outs
    
    
    
class Multi_Rotation_Convolution_Block_light(nn.Module):
    def __init__(self, in_planes):
        super(Multi_Rotation_Convolution_Block_light, self).__init__()
        
        angles = [0, 30, 45, 60]
        scales = [ROTATION_SCALE[angle] for angle in angles]
        
        self.angles = [torch.tensor(angle).float() for angle in angles]
        self.scale_factors = [torch.tensor(scale).float() for scale in scales]
        
        inter_planes = in_planes // 4
        inter_planes_2 = in_planes - 3*inter_planes
        
        self.split_1 = conv1x1_bn_relu(in_planes, inter_planes)
        self.split_2 = conv1x1_bn_relu(in_planes, inter_planes)
        self.split_3 = conv1x1_bn_relu(in_planes, inter_planes)
        self.split_4 = conv1x1_bn_relu(in_planes, inter_planes)
        
        self.conv_1 = conv3x3_bn_relu(inter_planes, inter_planes)
        self.conv_2 = conv3x3_bn_relu(inter_planes, inter_planes)
        self.conv_3 = conv3x3_bn_relu(inter_planes, inter_planes)
        self.conv_4 = conv3x3_bn_relu(inter_planes, inter_planes)
        
        self.conv_final = conv3x3_bn_relu(in_planes, in_planes)

        self.reverse = torch.tensor(360).float()
        
        
    def forward(self, x):

        is_amp = (x.dtype == torch.half)
        
        """ smoothing """
        x = torch.nn.functional.avg_pool2d(x, (3, 3), stride=1, padding=1)
        
        """ split 1/4 channels """
        outs = [self.split_1(x), self.split_2(x), self.split_3(x), self.split_4(x)]
        
        """ scale down & rotate & conv3x3 """
        if is_amp: outs = to_float(outs)
        outs = [ scale(out, scale_factor.to(out.device)) for out, scale_factor in zip(outs, self.scale_factors) ]
        outs = [ rotate(out, angle.to(out.device)) for out, angle in zip(outs, self.angles) ]
        if is_amp: outs = to_half(outs)
        
        outs = [self.conv_1(outs[0]), self.conv_2(outs[1]), self.conv_3(outs[2]), self.conv_4(outs[3])]
        
        """ recovery """
        if is_amp: outs = to_float(outs)
        outs = [ rotate(out, self.reverse.to(out.device)-angle.to(out.device)) for out, angle in zip(outs, self.angles) ]
        outs = [ scale(out, 1. / scale_factor.to(out.device)) for out, scale_factor in zip(outs, self.scale_factors) ]
        if is_amp: outs = to_half(outs)

        outs = torch.cat(outs, 1)
        outs = self.conv_final(outs)
        
        return outs
    

if __name__ == '__main__':
    device = 'cpu'
    
    MRCB = Multi_Rotation_Convolution_Block(256).to(device)
    print(MRCB)
    
    feature = torch.rand(2, 256, 64, 64).to(device)

    out = MRCB(feature)
    print(out.shape)
    