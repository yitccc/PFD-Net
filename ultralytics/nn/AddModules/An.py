import torch
import torch.nn as nn
import torch.nn.functional as F

class SmokeAlpha(nn.Module):
    """
    YOLO-friendly 模块，用于从输入RGB图像计算浓度感知烟雾引导图 α(x)
    输入: Tensor [B, 3, H, W]
    输出: Tensor [B, 1, H, W]，归一化浓度图，可直接用于模态融合
    """
    def __init__(self, kernel_size=15):
        super(SmokeAlpha, self).__init__()
        self.kernel_size = kernel_size
        self.eps = 1e-6

    def forward(self, x):
        B, C, H, W = x.shape
        if x.max() > 1.0:
            x = x / 255.0

        # HSV (V-S)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        v = max_rgb
        s = (max_rgb - min_rgb) / (max_rgb + self.eps)
        vs_diff = (v - s).clamp(0, 1)

        # Dark Channel
        dark = torch.min(x, dim=1, keepdim=True)[0]
        pad = self.kernel_size // 2
        dark_eroded = -F.max_pool2d(-dark, self.kernel_size, stride=1, padding=pad)
        dark_inv = 1.0 - dark_eroded

        # α(x) = (vs_diff + dark_inv) / 2
        alpha = 0.5 * vs_diff + 0.5 * dark_inv
        alpha = F.avg_pool2d(alpha, kernel_size=7, stride=1, padding=3)
        alpha = alpha.clamp(0, 1)
        return alpha  # [B, 1, H, W]

class BasicBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels1=64, activation=nn.ReLU):
        """
        初始化一个基础模块，包含两个3x3卷积层和一个激活函数层。

        参数：
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数
        - activation (nn.Module): 激活函数，默认使用ReLU
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1 // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.act = activation()
        self.conv2 = nn.Conv2d(out_channels1 // 2, out_channels1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels1 // 2)  # 可选，增加归一化
        self.bn2 = nn.BatchNorm2d(out_channels1)  # 可选，增加归一化
        self.mm = SmokeAlpha()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.mm(x)
        x = self.conv1(x)
        x = self.bn1(x)  # 如果不需要BatchNorm，可以直接去掉这两行
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)  # 如果不需要BatchNorm，可以直接去掉这两行
        x = self.pool(x)
        return x