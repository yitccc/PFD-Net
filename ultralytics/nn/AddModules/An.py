import torch
import torch.nn as nn
import torch.nn.functional as F

class SmokeAlpha(nn.Module):
    """
    A YOLO-friendly module designed to compute the concentration-aware smoke guidance map alpha(x) 
    from an input RGB image.
    
    Inputs:
        x: Tensor of shape [B, 3, H, W] representing the input RGB image.
    Outputs:
        alpha: Tensor of shape [B, 1, H, W] representing the normalized concentration map, 
               which can be directly utilized for cross-modal fusion.
    """
    def __init__(self, kernel_size=15):
        super(SmokeAlpha, self).__init__()
        self.kernel_size = kernel_size
        self.eps = 1e-6

    def forward(self, x):
        B, C, H, W = x.shape
        # Normalize the input tensor to the [0, 1] range if not already normalized
        if x.max() > 1.0:
            x = x / 255.0

        # Extract RGB channels
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        
        # Compute the Value-Saturation (V-S) difference from the HSV color space
        v = max_rgb
        s = (max_rgb - min_rgb) / (max_rgb + self.eps)
        vs_diff = (v - s).clamp(0, 1)

        # Compute the Dark Channel Prior (DCP)
        dark = torch.min(x, dim=1, keepdim=True)[0]
        pad = self.kernel_size // 2
        
        # Perform morphological erosion via max-pooling on the negative dark channel
        dark_eroded = -F.max_pool2d(-dark, self.kernel_size, stride=1, padding=pad)
        dark_inv = 1.0 - dark_eroded

        # Formulate the atmospheric scattering prior alpha(x)
        alpha = 0.5 * vs_diff + 0.5 * dark_inv
        
        # Apply average pooling for local smoothing
        alpha = F.avg_pool2d(alpha, kernel_size=7, stride=1, padding=3)
        alpha = alpha.clamp(0, 1)
        return alpha  # [B, 1, H, W]

class BasicBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels1=64, activation=nn.ReLU):
        """
        Initializes a basic block containing two 3x3 convolutional layers 
        and an activation function layer.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels1 (int): Number of output channels.
        - activation (nn.Module): Activation function, defaults to nn.ReLU.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1 // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.act = activation()
        self.conv2 = nn.Conv2d(out_channels1 // 2, out_channels1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels1 // 2)  # Optional batch normalization
        self.bn2 = nn.BatchNorm2d(out_channels1)       # Optional batch normalization
        
        # Initialize the physical prior extraction module (SmokeAlpha)
        self.mm = SmokeAlpha()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Extract the concentration-aware smoke prior map alpha(x)
        x = self.mm(x)
        
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)  # These two batch norm lines can be removed if not needed
        
        # Activation function
        x = self.act(x)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)  # These two batch norm lines can be removed if not needed
        
        # Max pooling layer
        x = self.pool(x)
        
        return x
