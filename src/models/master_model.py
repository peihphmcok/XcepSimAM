# Lưu với tên: src/models/master_model.py
import torch
import torch.nn as nn
from torch.nn import init

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Cảnh báo: Thư viện 'mamba-ssm' chưa được cài đặt. Vui lòng chạy: pip install mamba-ssm")


# --- MODULES ATTENTION ---
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        ca = self.channel_gate(x)
        x_ca = x * self.sigmoid(ca)
        # Spatial Attention
        sa_in = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        sa = self.spatial_gate(sa_in)
        x_sa = x_ca * sa  # Áp dụng spatial attention sau channel
        return x_sa


# --- CÁC KHỐI CƠ BẢN ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x);
        x = self.pointwise(x);
        return x


class XcepBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True,
                 attention_type='none'):
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
            rep.append(self.relu);
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False));
            rep.append(nn.BatchNorm2d(out_filters));
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu);
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1, bias=False));
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu);
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False));
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1: rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

        # Khởi tạo module Attention
        if attention_type.lower() == 'simam':
            self.attention = SimAM()
        elif attention_type.lower() == 'cbam':
            # CBAM cần biết số kênh đầu ra của block
            self.attention = CBAM(out_filters)
        else:
            self.attention = None

    def forward(self, inp):
        x = self.rep(inp)
        if self.attention is not None:
            x = self.attention(x)

        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp

        x += skip
        return x


class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model=728, n_head=8, dim_ff=2048):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_ff,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x_seq):
        return self.encoder(x_seq)


# --- MÔ HÌNH MASTER ---
class ConfigurableXcepMamba(nn.Module):
    def __init__(self, num_classes=1,
                 middle_flow_type='hierarchical',
                 attention_type='simam',
                 encoder_type='mamba'):

        super(ConfigurableXcepMamba, self).__init__()

        self.num_classes = num_classes
        self.middle_flow_type = middle_flow_type
        self.attention_type = attention_type
        self.encoder_type = encoder_type

        # --- 1. ENTRY FLOW (Luôn cố định) ---
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False);
        self.bn1 = nn.BatchNorm2d(32);
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False);
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = XcepBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True, attention_type='none')
        self.block2 = XcepBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True, attention_type='none')
        self.block3 = XcepBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True, attention_type='none')

        reps = 3  # 3 reps cho mỗi block ở Middle Flow

        # --- 2. MIDDLE FLOW (Đa-cấu-hình) ---

        if self.middle_flow_type == 'hierarchical':
            # Cấu trúc của BẠN (2-4-2 blocks)
            self.middle_stem_1 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.middle_stem_2 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.parallel_a_1 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.parallel_a_2 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.parallel_b_1 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.parallel_b_2 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.middle_fusion_bn = nn.BatchNorm2d(728 * 2)
            self.middle_fusion_relu = nn.ReLU(inplace=True)
            self.middle_fusion_conv = nn.Conv2d(728 * 2, 728, 1, 1, 0, bias=False)
            self.middle_tail_1 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            self.middle_tail_2 = XcepBlock(728, 728, reps, 1, attention_type=attention_type)

        elif self.middle_flow_type == 'sequential':
            # Cấu trúc 8-block nối tiếp (Baseline)
            self.middle_seq = nn.Sequential(
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type),
                XcepBlock(728, 728, reps, 1, attention_type=attention_type)
            )

        elif self.middle_flow_type == 'parallel_3_branch':
            # Cấu trúc 3-nhánh song song (Của bài báo gốc, 1+2+3=6 blocks)
            self.branch1 = XcepBlock(728, 728, 1, 1, attention_type=attention_type)  # 1 rep
            self.branch2 = nn.Sequential(
                XcepBlock(728, 728, 2, 1, attention_type=attention_type),  # 2 reps
                XcepBlock(728, 728, 2, 1, attention_type=attention_type)
            )
            self.branch3 = nn.Sequential(
                XcepBlock(728, 728, 3, 1, attention_type=attention_type),  # 3 reps
                XcepBlock(728, 728, 3, 1, attention_type=attention_type),
                XcepBlock(728, 728, 3, 1, attention_type=attention_type)
            )
            self.branch_fusion_bn = nn.BatchNorm2d(728 * 3)
            self.branch_fusion_relu = nn.ReLU(inplace=True)
            self.branch_fusion_conv = nn.Conv2d(728 * 3, 728, 1, 1, 0, bias=False)

        # --- 3. ENCODER HEAD (Đa-cấu-hình) ---

        if self.encoder_type == 'mamba':
            if Mamba is None: raise ImportError("Mamba is not available. Please install 'mamba-ssm'.")
            self.encoder = Mamba(d_model=728, d_state=16, d_conv=4, expand=2)
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.last_linear = nn.Linear(728, num_classes)

        elif self.encoder_type == 'self_attention':
            self.encoder = SelfAttentionEncoder(d_model=728)
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.last_linear = nn.Linear(728, num_classes)

        elif self.encoder_type == 'gap':
            # Xóa bỏ Exit Flow gốc và thay bằng GAP
            self.encoder = nn.AdaptiveAvgPool2d((1, 1))
            self.last_linear = nn.Linear(728, num_classes)

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1); init.constant_(m.bias, 0)

    def forward_middle_flow(self, x):
        if self.middle_flow_type == 'hierarchical':
            x_stem = self.middle_stem_1(x)
            x_stem = self.middle_stem_2(x_stem)
            x_a = self.parallel_a_1(x_stem);
            x_a = self.parallel_a_2(x_a)
            x_b = self.parallel_b_1(x_stem);
            x_b = self.parallel_b_2(x_b)
            x_fused = torch.cat([x_a, x_b], dim=1)
            x_fused = self.middle_fusion_bn(x_fused)
            x_fused = self.middle_fusion_relu(x_fused)
            x_fused = self.middle_fusion_conv(x_fused)
            x_out = self.middle_tail_1(x_fused)
            x_out = self.middle_tail_2(x_out)
            return x_out

        elif self.middle_flow_type == 'sequential':
            return self.middle_seq(x)

        elif self.middle_flow_type == 'parallel_3_branch':
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            x_fused = torch.cat([x1, x2, x3], dim=1)
            x_fused = self.branch_fusion_bn(x_fused)
            x_fused = self.branch_fusion_relu(x_fused)
            x_fused = self.branch_fusion_conv(x_fused)
            return x_fused

    def forward_encoder(self, x):
        if self.encoder_type == 'gap':
            x_pooled = self.encoder(x)  # GAP
            x_pooled = x_pooled.view(x_pooled.size(0), -1)
            return self.last_linear(x_pooled)
        else:
            b, c, h, w = x.shape
            x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, 728)
            x_encoded = self.encoder(x_seq)
            x_pooled = self.pooling(x_encoded.transpose(1, 2)).squeeze(-1)  # (B, 728)
            return self.last_linear(x_pooled)

    def forward(self, x):
        # 1. Entry Flow
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x)
        x = self.conv2(x);
        x = self.bn2(x);
        x = self.relu(x)
        x = self.block1(x);
        x = self.block2(x);
        x = self.block3(x)

        # 2. Middle Flow
        x = self.forward_middle_flow(x)

        # 3. Encoder Head
        x = self.forward_encoder(x)

        return x