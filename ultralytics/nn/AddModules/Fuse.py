import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PhaseAwareFusionModule(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3):
        super(PhaseAwareFusionModule, self).__init__()
        self.kernel_size = kernel_size
        self.eps = 1e-8

    def forward(self, Z):
        x, y = Z

        H_dim, W_dim = x.shape[2], x.shape[3]

        X = self.flatten_input(x)
        Y = self.flatten_input(y)

        A_x, P_x = self.fft_transform(X)
        A_y, P_y = self.fft_transform(Y)

        diff = P_x - P_y
        delta_phi = torch.atan2(torch.sin(diff), torch.cos(diff))
        abs_delta_phi = torch.abs(delta_phi)

        attn_map = torch.sigmoid(F.avg_pool2d(abs_delta_phi, self.kernel_size, stride=1, padding=self.kernel_size // 2))

        enhanced_P_x, enhanced_P_y = self.apply_attention_to_phase(P_x, P_y, attn_map)

        img_x_enhanced, img_y_enhanced = self.ifft_transform(A_x, enhanced_P_x, A_y, enhanced_P_y)

        img_x_enhanced, img_y_enhanced = self.restore_to_input_size(img_x_enhanced, img_y_enhanced, (H_dim, W_dim))

        return img_x_enhanced, img_y_enhanced

    def flatten_input(self, x):
        unfold = torch.nn.Unfold(kernel_size=(10, 10), stride=10)
        return unfold(x)

    def fft_transform(self, x):
        F_x = torch.fft.fft2(x)
        return torch.abs(F_x), torch.angle(F_x)

    def apply_attention_to_phase(self, P_x, P_y, W):
        Ux_real, Ux_imag = torch.cos(P_x), torch.sin(P_x)
        Uy_real, Uy_imag = torch.cos(P_y), torch.sin(P_y)
        U_tilde_x_real = (1 - W) * Ux_real + W * Uy_real
        U_tilde_x_imag = (1 - W) * Ux_imag + W * Uy_imag
        U_tilde_y_real = (1 - W) * Uy_real + W * Ux_real
        U_tilde_y_imag = (1 - W) * Uy_imag + W * Ux_imag
        enhanced_P_x = torch.atan2(U_tilde_x_imag, U_tilde_x_real)
        enhanced_P_y = torch.atan2(U_tilde_y_imag, U_tilde_y_real)
        return enhanced_P_x, enhanced_P_y

    def ifft_transform(self, A_x, P_x_enh, A_y, P_y_enh):
        F_x_enhanced = A_x * torch.exp(1j * P_x_enh)
        F_y_enhanced = A_y * torch.exp(1j * P_y_enh)
        return torch.fft.ifft2(F_x_enhanced).real, torch.fft.ifft2(F_y_enhanced).real

    def restore_to_input_size(self, img_x_enhanced, img_y_enhanced, original_size):
        fold = torch.nn.Fold(original_size, kernel_size=(10, 10), stride=10)
        return fold(img_x_enhanced), fold(img_y_enhanced)

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
        km = self.kk(X)
        R_s, T_s = km
        attn_r = self.global_r(self.conv_r(R_s))
        attn_t = self.global_t(self.conv_t(T_s))
        R_s_attn = R_s * attn_r
        T_s_attn = T_s * attn_t

        f = self.cbr1(R_s_attn) + self.cbr2(T_s_attn)

        S_attn = self.spatial_attention(f)
        R_s_enhanced = R_s_attn * S_attn
        T_s_enhanced = T_s_attn * S_attn

        F_cr = R_s + R_s_enhanced
        F_ct = T_s + T_s_enhanced
        F_cf = F_cr + F_ct
        return F_cf
