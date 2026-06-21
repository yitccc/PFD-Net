import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PhaseAwareFusionModule(nn.Module):
    """
    Phase-Differentiated Modal Interaction Module (PDMIM).
    Operates patch-wise in the frequency domain to transfer structural information 
    across modalities via phase discrepancy while preserving amplitude statistics.
    """
    def __init__(self, in_channels=3, kernel_size=3):
        super(PhaseAwareFusionModule, self).__init__()
        self.kernel_size = kernel_size
        self.eps = 1e-8

    def forward(self, Z):
        x, y = Z

        H_dim, W_dim = x.shape[2], x.shape[3]

        # Extract local patches via Unfold operation
        X = self.flatten_input(x)
        Y = self.flatten_input(y)

        # Factorize into magnitudes and phases via 2D FFT (Equation 14)
        A_x, P_x = self.fft_transform(X)
        A_y, P_y = self.fft_transform(Y)

        # Compute wrapped phase discrepancy avoiding 2pi discontinuities (Equation 15)
        diff = P_x - P_y
        delta_phi = torch.atan2(torch.sin(diff), torch.cos(diff))
        abs_delta_phi = torch.abs(delta_phi)

        # Construct adaptive structural dissonance weight map W (Equation 16)
        attn_map = torch.sigmoid(F.avg_pool2d(abs_delta_phi, self.kernel_size, stride=1, padding=self.kernel_size // 2))

        # Phase interpolation strictly on the unit complex circle (Equation 17 & 18)
        enhanced_P_x, enhanced_P_y = self.apply_attention_to_phase(P_x, P_y, attn_map)

        # Reconstruct spatial patches via Inverse FFT (Equation 19)
        # Amplitudes (A_x, A_y) remain invariant to preserve semantic strength
        img_x_enhanced, img_y_enhanced = self.ifft_transform(A_x, enhanced_P_x, A_y, enhanced_P_y)

        # Restore the full-resolution feature maps via Fold operation
        img_x_enhanced, img_y_enhanced = self.restore_to_input_size(img_x_enhanced, img_y_enhanced, (H_dim, W_dim))

        return img_x_enhanced, img_y_enhanced

    def flatten_input(self, x):
        # Extract patches with window size PxP and stride P (P=10)
        unfold = torch.nn.Unfold(kernel_size=(10, 10), stride=10)
        return unfold(x)

    def fft_transform(self, x):
        # Apply 2D FFT and decouple amplitude and phase
        F_x = torch.fft.fft2(x)
        return torch.abs(F_x), torch.angle(F_x)

    def apply_attention_to_phase(self, P_x, P_y, W):
        # Project phases onto the unit complex circle
        Ux_real, Ux_imag = torch.cos(P_x), torch.sin(P_x)
        Uy_real, Uy_imag = torch.cos(P_y), torch.sin(P_y)
        
        # Weighted interpolation of complex vectors
        U_tilde_x_real = (1 - W) * Ux_real + W * Uy_real
        U_tilde_x_imag = (1 - W) * Ux_imag + W * Uy_imag
        U_tilde_y_real = (1 - W) * Uy_real + W * Ux_real
        U_tilde_y_imag = (1 - W) * Uy_imag + W * Ux_imag
        
        # Extract the refined phases from the unit vectors
        enhanced_P_x = torch.atan2(U_tilde_x_imag, U_tilde_x_real)
        enhanced_P_y = torch.atan2(U_tilde_y_imag, U_tilde_y_real)
        return enhanced_P_x, enhanced_P_y

    def ifft_transform(self, A_x, P_x_enh, A_y, P_y_enh):
        # Recombine original amplitudes with enhanced phases
        F_x_enhanced = A_x * torch.exp(1j * P_x_enh)
        F_y_enhanced = A_y * torch.exp(1j * P_y_enh)
        return torch.fft.ifft2(F_x_enhanced).real, torch.fft.ifft2(F_y_enhanced).real

    def restore_to_input_size(self, img_x_enhanced, img_y_enhanced, original_size):
        # Fold spatial patches back to the original feature map grid
        fold = torch.nn.Fold(original_size, kernel_size=(10, 10), stride=10)
        return fold(img_x_enhanced), fold(img_y_enhanced)

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard Convolution-BatchNorm-ReLU block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SpatialDualAttention(nn.Module):
    """
    Wraps the PDMIM (PhaseAwareFusionModule) with dual-path global context 
    and spatial attention for robust cross-modal interaction.
    """
    def __init__(self, channels):
        super().__init__()
        self.cbr1 = CBR(channels, channels)
        self.cbr2 = CBR(channels, channels)

        self.conv_r = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_t = nn.Conv2d(channels, channels, kernel_size=1)

        # Global context attention branches
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

        # Spatial attention fusion head
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
        # Instantiate PDMIM
        self.kk = PhaseAwareFusionModule()

    def spatial_attention(self, x):
        # Generate spatial attention map via max and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.spatial_conv(x_cat))
        return attn

    def forward(self, X):
        # Apply Phase-Differentiated Modal Interaction Module
        km = self.kk(X)
        R_s, T_s = km
        
        # Apply global context recalibration
        attn_r = self.global_r(self.conv_r(R_s))
        attn_t = self.global_t(self.conv_t(T_s))
        R_s_attn = R_s * attn_r
        T_s_attn = T_s * attn_t

        # Feature aggregation and spatial attention refinement
        f = self.cbr1(R_s_attn) + self.cbr2(T_s_attn)
        S_attn = self.spatial_attention(f)
        
        R_s_enhanced = R_s_attn * S_attn
        T_s_enhanced = T_s_attn * S_attn

        # Residual connections
        F_cr = R_s + R_s_enhanced
        F_ct = T_s + T_s_enhanced
        F_cf = F_cr + F_ct
        return F_cf
