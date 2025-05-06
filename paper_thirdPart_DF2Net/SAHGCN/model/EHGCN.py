import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    Mx_min = Mx.min()
    if Mx_min < 0:
        Mx +=abs(Mx_min)
        Mx_min = Mx.min()

    if axis == 1:
        M_min = np.amin(Mx, axis=1)
        M_max = np.amax(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.amin(Mx, axis=0)
        M_max = np.amax(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx
class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        # LeakyReLU是ReLU函数的一个变体，解决了ReLU函数存在的问题，α的默认往往是非常小的，比如0.01，这样就保证了Dead Neurons的问题。
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))  # 初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        G = G.to(torch.float32)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G.to_sparse(), x)
        return x
class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,Q,A, dropout=0.):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_dim, hidden_dim)
        self.hgc2 = HGNN_conv(hidden_dim, 64)
        # self.computeG = compute_G(W)
        self.batch_normalzation1 = nn.BatchNorm1d(input_dim)
        self.batch_normalzation2 = nn.BatchNorm1d(hidden_dim)
        self.batch_normalzation3 = nn.BatchNorm1d(64)
        self.A = A

        self.lin = nn.Linear(64, num_classes)
        self.lin2 = nn.Linear(input_dim, input_dim)
        self.Q=Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
    def forward(self, x):
        (h, w, c) = x.shape
        x_flaten = x.reshape([h * w, -1])
        # x_flaten = self.batch_normalzation1(x_flaten)
        # x_flaten=self.lin2(x_flaten)
        # x_flaten = self.batch_normalzation1(x_flaten)
        supX = torch.mm(self.norm_col_Q.t(), x_flaten)
        supX = self.hgc1(supX, self.A)
        supX = self.batch_normalzation2(supX)
        supX = F.relu(supX)
        supX = F.dropout(supX, self.dropout)
        supX = self.hgc2(supX, self.A)
        supX = self.batch_normalzation3(supX)
        supX = F.relu(supX)
        supX=self.lin(supX)
        Y = torch.matmul(self.Q, supX)
        return F.softmax(Y, -1)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        print(in_dim)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class DFEF(nn.Module):
    def __init__(self, input_channels):
        super(DFEF, self).__init__()

        # Assuming subtract_feature is a custom function
        self.subtract_feature = lambda x: x[0] - x[1]

        # Assuming input_channels is the number of channels in the input
        self.global_fc1 = nn.Linear(input_channels, 1)
        self.global_fc2 = nn.Linear(input_channels, 1)
        self.batch_normalzation = nn.BatchNorm1d(input_channels)
        self.activation = nn.Tanh()

    def forward(self, x_hgcn,x_cnn ):

        subtracted = self.subtract_feature([x_cnn, x_hgcn])
        # subtracted_weight = self.global_fc1(subtracted.view(subtracted.size(0), -1))
        subtracted_weight = torch.mean(subtracted.view(subtracted.size(0), -1))
        excitation_weight = self.activation(subtracted_weight)

        subtracted2 = self.subtract_feature([x_hgcn, x_cnn])
        subtracted_weight2 = torch.mean(subtracted2.view(subtracted2.size(0), -1))
        excitation_weight2 = self.activation(subtracted_weight2)

        # Apply excitation weights
        x_cnn_weight = x_cnn * excitation_weight
        x_hgcn_weight = x_hgcn * excitation_weight2

        # Fuse features
        x_cnn_mix = x_hgcn_weight + x_cnn
        x_hgcn_mix = x_hgcn+ x_cnn_weight

        return x_hgcn_mix,x_cnn_mix
class DFEF2(nn.Module):
    def __init__(self, input_channels):
        super(DFEF2, self).__init__()

        # Assuming subtract_feature is a custom function
        self.subtract_feature = lambda x: x[0] - x[1]

        # Assuming input_channels is the number of channels in the input
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_normalzation = nn.BatchNorm1d(input_channels)
        self.activation = nn.Tanh()

    def forward(self, x_hgcn,x_cnn ):

        subtracted = self.subtract_feature([x_cnn, x_hgcn])
        subtracted_weight = self.global_avg_pool(subtracted)
        excitation_weight = self.activation(subtracted_weight)

        subtracted2 = self.subtract_feature([x_hgcn, x_cnn])
        subtracted_weight2 = self.global_avg_pool(subtracted2)
        excitation_weight2 = self.activation(subtracted_weight2)

        # Apply excitation weights
        x_cnn_weight = x_cnn * excitation_weight
        x_hgcn_weight = x_hgcn * excitation_weight2

        # Fuse features
        x_cnn_mix = 0.1*x_hgcn_weight + x_cnn
        x_hgcn_mix = x_hgcn+ 0.1*x_cnn_weight

        return x_hgcn_mix,x_cnn_mix

class EHGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, alph,
                 ):
        super(EHGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.alph = alph
        layers_count = 2


        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                # self.CNN_Branch.add_module('CrissCrossAttention' + str(i), CrissCrossAttention(self.channel))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(self.channel , 128, kernel_size=3))
                # self.CNN_Branch.add_module('CrissCrossAttention' + str(i), CrissCrossAttention(128))
            else:
                # self.CNN_Branch.add_module('CrissCrossAttention' + str(i), CrissCrossAttention(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
                # self.CNN_Branch.add_module('CrissCrossAttention' + str(i), CrissCrossAttention(64))

        self.HGNN_Branch = HGCN(self.channel,128, 64,self.Q,self.A,dropout=0.)
        # Softmax layer
        # self.DFEF_fusion=DFEF(input_channels=64)
        self.DFEF_fusion2=DFEF2(input_channels=64)
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))  # 128

    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape
        clean_x = x
        hx = clean_x
        # CNN与GCN分两条支路
        hx2=torch.unsqueeze(hx.permute([2, 0, 1]), 0)
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q

        H = clean_x
        HGNN_result= self.HGNN_Branch(H)
        HGNN_result,CNN_result=self.DFEF_fusion2(torch.unsqueeze(HGNN_result.reshape([h,w,-1]).permute([2, 0, 1]), 0),torch.unsqueeze(CNN_result.reshape([h,w,-1]).permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        HGNN_result = torch.squeeze(HGNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        Y = self.alph * CNN_result + (1 - self.alph) * HGNN_result
        # Y=CNN_result

        # Y = torch.cat([HGNN_result, CNN_result], dim=-1)
        # Y=HGNN_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y

