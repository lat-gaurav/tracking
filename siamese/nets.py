"""
SiamFC backbone and cross-correlation head.

Architecture
------------
  - AlexNet-style fully-convolutional backbone (no FC layers, no padding in
    conv layers so spatial info is preserved)
  - Depth-wise cross-correlation: the template feature map is used as a
    convolutional kernel sliding over the search feature map → response map

Reference: Bertinetto et al., "Fully-Convolutional Siamese Networks for
Object Tracking", ECCV 2016.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFCBackbone(nn.Module):
    """
    5-layer AlexNet-style backbone, zero-padding disabled so the output shrinks
    spatially.  Both the template (z) and search (x) branches share weights.

    Input size  →  Output size
    127×127  z  →  6×6   feature map
    255×255  x  →  22×22 feature map
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, stride=2),   # (N,3,H,W) → (N,96,...)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            # conv2
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            # conv3
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SiamFC(nn.Module):
    """
    Full SiamFC model.

    forward(z, x):
        z – template patch (B, 3, 127, 127)
        x – search  patch  (B, 3, 255, 255)
      → score map          (B, 1, H_score, W_score)  typically (B,1,17,17)

    The score map peak gives the displacement of the target from the search
    centre, scaled by the network's total stride (8).
    """

    def __init__(self):
        super().__init__()
        self.backbone = SiamFCBackbone()
        self.bn_adjust = nn.BatchNorm2d(1)   # learnable scale/bias on scores

    # ---- cross-correlation ------------------------------------------------
    @staticmethod
    def _xcorr(z_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """
        Depth-wise cross-correlation.
        z_feat: (B, C, hz, wz)  – template features used as kernel
        x_feat: (B, C, hx, wx)  – search features
        Returns: (B, 1, hx-hz+1, wx-wz+1)
        """
        B, C, hz, wz = z_feat.shape
        # merge batch into channels so a single grouped conv does the job
        x_flat = x_feat.view(1, B * C, x_feat.shape[2], x_feat.shape[3])
        z_flat = z_feat.view(B * C, 1, hz, wz)
        out    = F.conv2d(x_flat, z_flat, groups=B * C)         # (1, B*C, ...)
        out    = out.view(B, C, out.shape[2], out.shape[3])     # (B, C, ...)
        return out.mean(dim=1, keepdim=True)                     # (B, 1, ...)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_feat = self.backbone(z)
        x_feat = self.backbone(x)
        score  = self._xcorr(z_feat, x_feat)
        score  = self.bn_adjust(score)
        return score

    # ---- convenience: feature-only (used by tracker to cache template) ------
    def extract(self, patch: torch.Tensor) -> torch.Tensor:
        return self.backbone(patch)

    def score_from_features(self, z_feat: torch.Tensor,
                             x_feat: torch.Tensor) -> torch.Tensor:
        score = self._xcorr(z_feat, x_feat)
        score = self.bn_adjust(score)
        return score
