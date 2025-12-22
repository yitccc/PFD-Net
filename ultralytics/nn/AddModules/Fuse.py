import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseAwareFusionModule(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3):
        super(PhaseAwareFusionModule, self).__init__()
        self.kernel_size = kernel_size  # 用于计算注意力图的卷积核大小

    def forward(self, Z):
        # Step 1: 展开输入
        x, y = Z
        X = self.flatten_input(x)
        Y = self.flatten_input(y)

        # Step 2: 对输入进行傅里叶变换
        A_x, P_x = self.fft_transform(X)
        A_y, P_y = self.fft_transform(Y)

        # Step 3: 计算相位差
        delta_phi = self.compute_phase_difference(P_x, P_y)

        # Step 4: 计算相位差的注意力权重
        attn_map = self.compute_attention_map(delta_phi)

        # Step 5: 用注意力权重加权幅值
        enhanced_A_x, enhanced_A_y = self.apply_attention_to_amplitude(A_x, A_y, attn_map)

        # Step 6: 进行逆傅里叶变换
        img_x_enhanced, img_y_enhanced = self.ifft_transform(enhanced_A_x, P_x, enhanced_A_y, P_y)

        # Step 7: 还原为输入尺寸
        img_x_enhanced, img_y_enhanced = self.restore_to_input_size(img_x_enhanced, img_y_enhanced, x.shape[2:])

        return img_x_enhanced, img_y_enhanced

    def flatten_input(self, x):
        unfold = torch.nn.Unfold(kernel_size=(10, 10), stride=10)
        patches = unfold(x)
        return patches

    def fft_transform(self, x):
        F_x = torch.fft.fft2(x)  # 进行傅里叶变换
        A_x = torch.abs(F_x)  # 幅值
        P_x = torch.angle(F_x)  # 相位
        return A_x, P_x

    def compute_phase_difference(self, P_x, P_y):
        delta_phi = torch.abs(P_x - P_y)  # 计算相位差
        return delta_phi

    def compute_attention_map(self, delta_phi):
        attn_map = torch.sigmoid(F.avg_pool2d(delta_phi, self.kernel_size, stride=1, padding=self.kernel_size // 2))
        return attn_map

    def apply_attention_to_amplitude(self, A_x, A_y, attn_map):
        enhanced_A_x = A_x * attn_map  # 用相位差加权幅值
        enhanced_A_y = A_y * (1 - attn_map)  # 对另一个模态做反向加权
        return enhanced_A_x, enhanced_A_y

    def ifft_transform(self, A_x, P_x, A_y, P_y):
        F_x_enhanced = A_x * torch.exp(1j * P_x)  # 使用增强幅值和原始相位
        F_y_enhanced = A_y * torch.exp(1j * P_y)  # 使用增强幅值和原始相位
        img_x_enhanced = torch.fft.ifft2(F_x_enhanced).real  # 逆傅里叶变换
        img_y_enhanced = torch.fft.ifft2(F_y_enhanced).real  # 逆傅里叶变换
        return img_x_enhanced, img_y_enhanced

    def restore_to_input_size(self, img_x_enhanced, img_y_enhanced, original_size):
        fold = torch.nn.Fold(original_size, kernel_size=(10, 10), stride=10)
        foldx = fold(img_x_enhanced)
        foldy = fold(img_y_enhanced)
        return foldx, foldy

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SpatialDualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cbr1 = CBR(channels, channels)
        self.cbr2 = CBR(channels, channels)

        self.conv_r = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_t = nn.Conv2d(channels, channels, kernel_size=1)

        self.global_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.global_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.kk = PhaseAwareFusionModule()

    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.spatial_conv(x_cat))
        return attn

    def forward(self, X):
        # 通道注意力
        km = self.kk(X)
        R_s, T_s = km
        attn_r = self.global_r(self.conv_r(R_s))
        attn_t = self.global_t(self.conv_t(T_s))
        R_s_attn = R_s * attn_r
        T_s_attn = T_s * attn_t

        # 多尺度 CBR 融合
        f = self.cbr1(R_s_attn) + self.cbr2(T_s_attn)

        # 空间注意力
        S_attn = self.spatial_attention(f)
        R_s_enhanced = R_s_attn * S_attn
        T_s_enhanced = T_s_attn * S_attn

        # 输出三路
        F_cr = R_s + R_s_enhanced
        F_ct = T_s + T_s_enhanced
        F_cf = F_cr + F_ct
        return F_cf

# B, C, H, W = 4, 128, 80, 80  # 将高度和宽度从64改为32
# x = torch.randn(B, C, H, W)  # RGB图像
# y = torch.randn(B, C, H, W)  # IR图像
# z = SpatialDualAttention(128)
# M = z([x,y])[2]
# print(M.shape)