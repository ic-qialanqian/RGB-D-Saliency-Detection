import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, Parameter, Softmax
import torch.nn.init as init
from wavelet_transform.DWT_IDWT_layer import DWT_2D, IDWT_2D

class SpatialSELayer1(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer1, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dwt = DWT_2D(wavename='haar')
        self.idwt = IDWT_2D(wavename='haar')
        
    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        
        LL, LH, HL, HH = self.dwt(input_tensor)
        batch_size, channel, a, b = LL.size()
        
        
        out1 = self.conv1(LL)
            
        out1 = self.relu(out1)
        squeeze_tensor1 = self.sigmoid(self.conv2(out1))

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        #squeeze_tensor1 = squeeze_tensor1.view(batch_size, 1, a, b)
        
        output_tensor1 = LL * squeeze_tensor1
        #output_tensor1 = torch.mul(LL, squeeze_tensor1)
        
        
        output_tensor = self.idwt(output_tensor1,LH,HL,HH)
        return output_tensor
        
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dwt = DWT_2D(wavename='haar')
        #self.idwt = IDWT_2D(wavename='haar')
        
    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        
        LL, LH, HL, HH = self.dwt(input_tensor)
        
        LL = F.upsample(LL, size=input_tensor.size()[2:], mode='bilinear')
        batch_size, channel, a, b = LL.size()
        out1 = self.conv1(LL)
            
        out1 = self.relu(out1)
        squeeze_tensor1 = self.sigmoid(self.conv2(out1))

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        #squeeze_tensor1 = squeeze_tensor1.view(batch_size, 1, a, b)
        
        output_tensor = input_tensor * squeeze_tensor1 + self.conv3(input_tensor)
        #output_tensor1 = torch.mul(LL, squeeze_tensor1)
        
        
        #output_tensor = self.idwt(output_tensor1,LH,HL,HH)
        return output_tensor
        
class SpatialSELayer_orig(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer_orig, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor
        
class SpatialSELayer1(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer1, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dwt = DWT_2D(wavename='haar')
        self.idwt = IDWT_2D(wavename='haar')
        
    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        
        LL, LH, HL, HH = self.dwt(input_tensor)
        batch_size, channel, a, b = LL.size()
        
        
        out1 = self.conv1(LL)
            
        out1 = self.relu(out1)
        squeeze_tensor1 = self.sigmoid(self.conv2(out1))

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        #squeeze_tensor1 = squeeze_tensor1.view(batch_size, 1, a, b)
        
        output_tensor1 = LL * squeeze_tensor1
        #output_tensor1 = torch.mul(LL, squeeze_tensor1)
        
        
        output_tensor = self.idwt(output_tensor1,LH,HL,HH)
        return output_tensor
        
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        self.fc3 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc4 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        self.fc5 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc6 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        self.fc7 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc8 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dwt = DWT_2D(wavename='haar')
        self.idwt = IDWT_2D(wavename='haar')

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        LL, LH, HL, HH = self.dwt(input_tensor)
        batch_size, num_channels, H, W = LL.size()
        # Average along each channel
        squeeze_tensor_LL = LL.view(batch_size, num_channels, -1).mean(dim=2)
        squeeze_tensor_LH = LH.view(batch_size, num_channels, -1).mean(dim=2)
        squeeze_tensor_HL = HL.view(batch_size, num_channels, -1).mean(dim=2)
        squeeze_tensor_HH = HH.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor_LL))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        fc_out_3 = self.relu(self.fc3(squeeze_tensor_LH))
        fc_out_4 = self.sigmoid(self.fc4(fc_out_3))
        
        fc_out_5 = self.relu(self.fc5(squeeze_tensor_HL))
        fc_out_6 = self.sigmoid(self.fc6(fc_out_5))
        
        fc_out_7 = self.relu(self.fc7(squeeze_tensor_HH))
        fc_out_8 = self.sigmoid(self.fc8(fc_out_7))

        a, b = squeeze_tensor_LL.size()
        output_tensor_LL = torch.mul(LL, fc_out_2.view(a, b, 1, 1))
        output_tensor_LH = torch.mul(LH, fc_out_4.view(a, b, 1, 1))
        output_tensor_HL = torch.mul(HL, fc_out_6.view(a, b, 1, 1))
        output_tensor_HH = torch.mul(HH, fc_out_8.view(a, b, 1, 1))
        
        output_tensor = self.idwt(output_tensor_LL,output_tensor_LH,output_tensor_HL,output_tensor_HH)
        return output_tensor
        
class maxpool_attention(nn.Module): # pspnet
    def __init__(self, inchannel,down_dim):
        super(maxpool_attention, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(inchannel, down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(3, 3)),nn.Conv2d(inchannel, down_dim, kernel_size=1), AttentionLayer(down_dim, down_dim, 3, stride=1, padding=1, groups=1, bias=False),nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(5, 5)), nn.Conv2d(inchannel, down_dim, kernel_size=1),AttentionLayer(down_dim, down_dim, 3, stride=1, padding=1, groups=1, bias=False),nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(7, 7)),nn.Conv2d(inchannel, down_dim, kernel_size=1), AttentionLayer(down_dim, down_dim, 3, stride=1, padding=1, groups=1, bias=False),nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        #self.conv4 = nn.Sequential(
            #nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            #nn.BatchNorm2d(down_dim), nn.PReLU()
        #)

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

    def forward(self, x):
        conv0 = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        #conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear')
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear')
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear')
        #conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear')

        return self.fuse(torch.cat((conv0,conv1_up, conv2_up, conv3_up), 1))
        
class scale_aware_attention(nn.Module): # deeplab

    def __init__(self, in_channels):
        super(scale_aware_attention, self).__init__()
        #self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             #nn.PReLU())
        #down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding=0), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0), nn.PReLU()
        )
        
        self.merge1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=1),nn.PReLU()
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=1),nn.PReLU()
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding=0), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0), nn.PReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0), nn.PReLU()
        )
        

    def forward(self, x):
        #x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        #conv4 = self.conv3(x)
        #conv5 = self.conv3(x)
        
        merge1 = self.conv4(self.merge1(conv1))
        merge2 = self.conv5(self.merge2(conv2))
        merge3 = self.conv6(self.merge3(conv3))
        #merge4 = self.merge3(conv4)
        #merge5 = self.merge3(conv5)
        #output = self.fuse(merge1+merge2+merge3)
        return merge1,merge2,merge3
        
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out
        
class DEP(nn.Module): # deeplab

    def __init__(self):
        super(DEP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(256),nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(128),nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(64),nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  nn.AdaptiveAvgPool2d(32),nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(16),nn.PReLU()
        )
        

    def forward(self, x):
        #x, LH,HL,HH = self.dwt_depth0(x)
        conv1 = self.conv1(x)
        
        conv2 = self.conv2(conv1)
        
        conv3 = self.conv3(conv2)
        
        conv4 = self.conv4(conv3)
        
        conv5 = self.conv5(conv4)
        return conv5
        
        
class merge(nn.Module):
    def __init__(self,down_dim):
        super(merge, self).__init__()
        #self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=1), AttentionLayer(down_dim, down_dim, 3, stride=1, padding=1, groups=1, bias=False),nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.dwt = DWT_2D(wavename='haar')
        self.idwt = IDWT_2D(wavename='haar')
        '''
        self.rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.depth = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.merge = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, kernel_size=1, padding=0), nn.BatchNorm2d(in_channels),nn.PReLU())
        #self.merge1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        '''


    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        
        out1 = self.conv1(LL)
        
        
        output = self.idwt(out1,LH,HL,HH)
        
        return output

        
        
###CVPR2017 Pyramid Scene Parsing Network
class PPM(nn.Module): # pspnet
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(2048,down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear')
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear')
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear')
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear')

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))

###TPAMI2017 Deeplabv2
class ASPP(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1))

###CVPR2019 AFNet: Attentive Feedback Network for Boundary-aware Salient Object Detection
class GPM(nn.Module): # cvpr19 AFNet -rgb_sod

    def __init__(self, in_dim):
        super(GPM, self).__init__()
        down_dim = 512
        n1, n2, n3 = 2, 4, 6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(down_dim * n1 * n1, down_dim * n1 * n1, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n1 * n1), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim * n2 * n2, down_dim * n2 * n2, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n2 * n2), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim * n3 * n3, down_dim * n3 * n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n3 * n3), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        ###########################################################################
        gm_2_a = torch.chunk(conv1, 2, 2)
        c = []
        for i in range(len(gm_2_a)):
            b = torch.chunk(gm_2_a[i], 2, 3)
            c.append(torch.cat((b[0], b[1]), 1))
        gm1 = torch.cat((c[0], c[1]), 1)
        gm1 = self.conv2(gm1)
        gm1 = torch.chunk(gm1, 2 * 2, 1)
        d = []
        for i in range(2):
            d.append(torch.cat((gm1[2 * i], gm1[2 * i + 1]), 3))
        gm1 = torch.cat((d[0], d[1]), 2)
        ###########################################################################
        gm_4_a = torch.chunk(conv1, 4, 2)
        e = []
        for i in range(len(gm_4_a)):
            f = torch.chunk(gm_4_a[i], 4, 3)
            e.append(torch.cat((f[0], f[1], f[2], f[3]), 1))
        gm2 = torch.cat((e[0], e[1], e[2], e[3]), 1)
        gm2 = self.conv3(gm2)
        gm2 = torch.chunk(gm2, 4 * 4, 1)
        g = []
        for i in range(4):
            g.append(torch.cat((gm2[4 * i], gm2[4 * i + 1], gm2[4 * i + 2], gm2[4 * i + 3]), 3))
        gm2 = torch.cat((g[0], g[1], g[2], g[3]), 2)
        ###########################################################################
        gm_6_a = torch.chunk(conv1, 6, 2)
        h = []
        for i in range(len(gm_6_a)):
            k = torch.chunk(gm_6_a[i], 6, 3)
            h.append(torch.cat((k[0], k[1], k[2], k[3], k[4], k[5]), 1))

        gm3 = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), 1)
        gm3 = self.conv4(gm3)
        gm3 = torch.chunk(gm3, 6 * 6, 1)
        j = []
        for i in range(6):
            j.append(
                torch.cat((gm3[6 * i], gm3[6 * i + 1], gm3[6 * i + 2], gm3[6 * i + 3], gm3[6 * i + 4], gm3[6 * i + 5]),
                          3))
        gm3 = torch.cat((j[0], j[1], j[2], j[3], j[4], j[5]), 2)
        ###########################################################################

        return self.fuse(torch.cat((gm1, gm2, gm3), 1))


###ECCV2020 A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection
class PAFEM(nn.Module):
    def __init__(self, dim,in_dim):
        super(PAFEM, self).__init__()
        
        self.dwt = DWT_2D(wavename='haar')
        
        
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = Parameter(torch.zeros(1))

        #self.conv5 = nn.Sequential(
            #nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU())

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        LL = self.down_conv(LL)
        conv1 = self.conv1(x)
        
        conv2 = self.conv2(LL)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2* out2 + conv2
        
        conv3 = self.conv3(LL)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        
        conv4 = self.conv4(LL)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        #conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear') 
        
        final_LL = self.fuse(torch.cat((out2, out3,out4), 1))
        attented = conv1 + F.upsample(final_LL, size=conv1.size()[2:], mode='bilinear')
        return attented


###ECCV2020 Suppress and Balance: A Simple Gated Network for Salient Object Detection
class FoldConv_aspp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(FoldConv_aspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.PReLU())
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU())

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.PReLU()
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)


        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=2, dilation=1, padding=0, stride=2)
        in_feature = self.up_conv(in_feature)
        return in_feature


class AttentionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionLayer, self).__init__()
        # define the params
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.stride = stride
        # define the query key and value
        self.key_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        # define the relative position embedding
        # rel_col, rel_row = self._generate_rel_pos()
        # make it an parameters - we fix it and don't need the gradients
        # self.rel_col = nn.Parameter(torch.tensor(rel_col, dtype=torch.float32).unsqueeze(0), requires_grad=False)
        # self.rel_row = nn.Parameter(torch.tensor(rel_row, dtype=torch.float32).unsqueeze(0), requires_grad=False)

    def _generate_rel_pos(self):
        # use mesh grid to generate the relative position matrix
        rel_col, rel_row = np.meshgrid(np.arange(self.kernel_size), np.arange(self.kernel_size))
        rel_col = rel_col - (self.kernel_size - 1) * 0.5
        rel_row = rel_row - (self.kernel_size - 1) * 0.5
        # repeat
        rel_col = np.repeat(np.expand_dims(rel_col, 2), int(self.out_planes / 2), axis=2)
        rel_row = np.repeat(np.expand_dims(rel_row, 2), int(self.out_planes / 2), axis=2)
        # transpose the maps
        rel_col = np.transpose(rel_col, (2, 0, 1))
        rel_row = np.transpose(rel_row, (2, 0, 1))
        return rel_col, rel_row

    def forward(self, x):
        batch, channels, height, width = x.size()
        # padding the inputs - but not for the feature map send into the q
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # generate the query, keys and the value
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        # start the next steps - unfold the output into windwos for easy sliding window multiplication
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # split the v for relative position
        #v_out_col, v_out_row = v_out.split(int(self.out_planes / 2), dim=1)
        #v_out = torch.cat((v_out_col + self.rel_col, v_out_row + self.rel_row), dim=1)
        #
        k_out = k_out.contiguous().view(batch, self.groups, int(self.out_planes / self.groups), height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, int(self.out_planes / self.groups), height, width, -1)
        q_out = q_out.view(batch, self.groups, int(self.out_planes / self.groups), height, width, 1)
        # sum
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # the final output
        return out

