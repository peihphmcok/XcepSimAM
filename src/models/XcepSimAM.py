import torch
import torch.nn as nn
from torch.nn import init

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class XcepBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, use_simam=False):
        super(XcepBlock, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)
        self.simam = SimAM() if use_simam else None

    def forward(self, inp):
        x = self.rep(inp)
        if self.simam is not None:
            x = self.simam(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        x += skip
        return x

class XceptionSimAM(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(XceptionSimAM, self).__init__()
        
        # Entry Flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XcepBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XcepBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XcepBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle Flow (3 Parallel Branches)
        self.mid_branch1 = nn.Sequential(
            XcepBlock(728, 728, 3, 1, use_simam=True)
        )
        self.mid_branch2 = nn.Sequential(
            XcepBlock(728, 728, 3, 1, use_simam=True),
            XcepBlock(728, 728, 3, 1, use_simam=True)
        )
        self.mid_branch3 = nn.Sequential(
            XcepBlock(728, 728, 3, 1, use_simam=True),
            XcepBlock(728, 728, 3, 1, use_simam=True),
            XcepBlock(728, 728, 3, 1, use_simam=True)
        )

        self.mid_fusion_bn = nn.BatchNorm2d(728 * 3)
        self.mid_fusion_conv = nn.Conv2d(728 * 3, 728, 1, 1, 0, bias=False)

        # Exit Flow (Dual Attention)
        self.exit_block_resid = XcepBlock(728, 1024, 2, 1, start_with_relu=True, grow_first=False)
        self.exit_sep_conv_1 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.exit_bn_1 = nn.BatchNorm2d(1536)

        self.exit_sep_conv_2 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.exit_bn_2 = nn.BatchNorm2d(2048)

        self.exit_att_proj = nn.Conv2d(1536, 2048, 1, 1, 0)
        self.exit_self_att = SelfAttention(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(2048, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1); init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)

    def forward(self, x):
        # Entry Flow
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)

        # Middle Flow
        x1 = self.mid_branch1(x)
        x2 = self.mid_branch2(x)
        x3 = self.mid_branch3(x)
        
        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.mid_fusion_bn(x_fused)
        x_fused = self.relu(x_fused)
        x = self.mid_fusion_conv(x_fused)

        # Exit Flow
        x = self.exit_block_resid(x)
        x = self.exit_sep_conv_1(x); x = self.exit_bn_1(x); x = self.relu(x)

        out_local = self.exit_sep_conv_2(x); out_local = self.exit_bn_2(out_local); out_local = self.relu(out_local)
        out_global = self.exit_att_proj(x); out_global = self.exit_self_att(out_global)

        x_final = out_local + out_global
        x_final = self.global_avg_pool(x_final)
        x_final = x_final.view(x_final.size(0), -1)

        x_final = self.dropout(x_final)
        x_final = self.fc(x_final)
        return x_final