import torch
import torch.nn as nn
import torch.nn.functional as F

class BAA_Module(nn.Module):
    """
    Bidirectional Alignment Attention (BAA) Module for dual-resolution feature fusion.
    输入:
        FH: [B, C, Hh, Wh] 高分辨率特征
        FL: [B, C, Hl, Wl] 低分辨率特征
    输出:
        Fi: [B, C, Hl, Wl] 增强后的低分辨率特征
    """
    def __init__(self, in_channels, reduction=16, pool_size=4):
        super(BAA_Module, self).__init__()
        self.C = in_channels
        self.r = reduction
        self.k = pool_size  # 全局池化后的空间尺寸 (k x k)

        # Spatial Attention: Q^td_s, K^td_s → Wo_s
        self.Wq_s = nn.Linear(in_channels, in_channels // reduction)
        self.Wk_s = nn.Linear(in_channels, in_channels // reduction)
        self.Wo_s = nn.Linear(self.k ** 2, 1)

        # Channel Attention: Q^td_c, K^td_c → Wo_c
        self.Wq_c = nn.Linear(in_channels, in_channels // reduction)
        self.Wk_c = nn.Linear(in_channels, in_channels)
        self.Wo_c = nn.Linear(in_channels // reduction, 1)

        self.norm_s = nn.LayerNorm([in_channels // reduction])
        self.norm_c = nn.LayerNorm([in_channels])

    def forward(self, X):
        FH, FL = X
        B, C, Hh, Wh = FH.shape
        _, _, Hl, Wl = FL.shape
        k2 = self.k ** 2

        # === [1] 池化 + 拉平为序列 ===
        FH_pool = F.adaptive_avg_pool2d(FH, (self.k, self.k))  # [B, C, k, k]
        FL_pool = F.adaptive_avg_pool2d(FL, (self.k, self.k))  # [B, C, k, k]
        Xh = FH_pool.flatten(2).transpose(1, 2)
        Xl = FL_pool.flatten(2).transpose(1, 2)  # [B, k2, C]
        XL = FL.flatten(2).transpose(1, 2)

        # === [2] Spatial Attention ===
        Qs = self.Wq_s(Xh)
        Ks = self.Wk_s(XL)  # [B, Hl*Wl, C//r]
        Attn_s = torch.matmul(Ks, Qs.transpose(1, 2))  # [B, Hl*Wl, k2]
        Attn_s = self.Wo_s(Attn_s)
        S_s = torch.sigmoid(Attn_s.view(B, 1, Hl, Wl)) # [B, 1, Hl, Wl]
        S_s = S_s * FL


        # === [3] Channel Attention ===
        Qc = self.Wq_c(Xh).transpose(1, 2)  # [B, k2, C//r]
        Kc = self.Wk_c(Xh).transpose(1, 2)               # [B, k2, C]
        Attn_c = torch.matmul(Kc, Qc.transpose(1, 2)) # [B, C//r]
        Attn_c = self.Wo_c(Attn_c)                     # [B, 1]
        S_c = torch.sigmoid(Attn_c.view(B, C, 1, 1))   # [B, C, 1, 1]
        S_c = S_c * FL

        # === [4] Feature Interaction ==
        Fi = S_s + S_c + FL  # [B, C, Hl, Wl]
        return Fi