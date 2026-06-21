import torch
import torch.nn as nn
import torch.nn.functional as F

class BAA_Module(nn.Module):
    """
    Bidirectional Alignment Attention (BAA) Module, serving as the core implementation 
    for the Bidirectional Dual-Alignment Module (BDAM).
    
    Inputs:
        FH: [B, C, Hh, Wh] High-resolution guiding feature (e.g., Prior Map A).
        FL: [B, C, Hl, Wl] Low-resolution guided feature (e.g., VIS feature B).
    Outputs:
        Fi: [B, C, Hl, Wl] Enhanced/aligned guided feature.
    """
    def __init__(self, in_channels, reduction=16, pool_size=4):
        super(BAA_Module, self).__init__()
        self.C = in_channels
        self.r = reduction
        self.k = pool_size  # Spatial dimension after adaptive average pooling (k x k)

        # Spatial Attention Projections: Q_s, K_s -> W_os
        self.Wq_s = nn.Linear(in_channels, in_channels // reduction)
        self.Wk_s = nn.Linear(in_channels, in_channels // reduction)
        self.Wo_s = nn.Linear(self.k ** 2, 1)

        # Channel Attention Projections: Q_c, K_c -> W_oc
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

        # === [1] Adaptive Pooling and Flattening into Sequences ===
        FH_pool = F.adaptive_avg_pool2d(FH, (self.k, self.k))  # [B, C, k, k]
        FL_pool = F.adaptive_avg_pool2d(FL, (self.k, self.k))  # [B, C, k, k]
        
        # Reshape guiding feature (A) into sequence
        Xh = FH_pool.flatten(2).transpose(1, 2)
        # Reshape guided feature (B) into sequence
        Xl = FL_pool.flatten(2).transpose(1, 2)  # [B, k2, C]
        
        # Flatten the full-resolution guided feature
        XL = FL.flatten(2).transpose(1, 2)

        # === [2] Spatial Attention (Cross-Domain Alignment) ===
        Qs = self.Wq_s(Xh)  # Query generated from the guiding feature (A)
        Ks = self.Wk_s(XL)  # Key generated from the guided feature (B) [B, Hl*Wl, C//r]
        
        # Compute cross-domain spatial attention matrix
        Attn_s = torch.matmul(Ks, Qs.transpose(1, 2))  # [B, Hl*Wl, k2]
        Attn_s = self.Wo_s(Attn_s)
        S_s = torch.sigmoid(Attn_s.view(B, 1, Hl, Wl)) # [B, 1, Hl, Wl]
        
        # Modulate guided feature B with the spatial attention map
        S_s = S_s * FL

        # === [3] Channel Attention ===
        Qc = self.Wq_c(Xh).transpose(1, 2)  # Query from guiding feature (A) [B, k2, C//r]
        Kc = self.Wk_c(Xh).transpose(1, 2)  # Key from guiding feature (A) [B, k2, C]
        
        # Compute channel attention weights
        Attn_c = torch.matmul(Kc, Qc.transpose(1, 2))  # [B, C//r]
        Attn_c = self.Wo_c(Attn_c)                     # [B, 1]
        S_c = torch.sigmoid(Attn_c.view(B, C, 1, 1))   # [B, C, 1, 1]
        
        # Modulate guided feature B with the channel attention weights
        S_c = S_c * FL

        # === [4] Feature Interaction (Residual Connection) ===
        # Combine spatial-modulated and channel-modulated features with original feature B
        Fi = S_s + S_c + FL  # [B, C, Hl, Wl]
        return Fi
