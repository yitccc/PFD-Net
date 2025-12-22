# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules (optimized)."""

import copy
import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import matplotlib.pyplot as plt

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "DeFE"


# ------------------------------------------------------------------------------------------------
# Basic Detect family
# ------------------------------------------------------------------------------------------------
class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """Dual-branch (one2many/one2one) forward."""
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multi-level feature maps."""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """Top-k class-first selection."""
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    """Head for integrating YOLO detection with text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


# ------------------------------------------------------------------------------------------------
# DeFE: Density-based Fast Estimator (GPU-only optimized)
# ------------------------------------------------------------------------------------------------
class DeFE(nn.Module):
    """
    轻量密度图 + 纯 GPU 查询数估计：
      - 三层 depthwise 膨胀卷积 + 通道注意力 + 1×1 投影为密度图
      - 通过下采样二值图 + max/avg-pool 的“近似连通块计数”，完全 GPU 上完成
    """

    def __init__(self, in_channels=256, out_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        self.out_size = out_size

        # 多尺度感受野 depthwise dilated conv
        self.dilated_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels),
            nn.ReLU(inplace=True),
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 投影为密度图
        self.density_head = nn.Conv2d(in_channels, 1, 3, padding=1)



    @torch.no_grad()
    def _estimate_queries_gpu(
            self,
            D: torch.Tensor,  # [B,1,H,W] in [0,1]
            K_N: int = 10, K_M: int = 300,
            q: float = 0.7,
            down: int = 4,
            grid_eps: int = 20,
            beta: float = 1.0
    ):
        """
        近似聚类计数（按原文描述）：
          1) 平滑密度图
          2) 聚类（使用DBSCAN或8邻域连通域算法）
          3) 分位数阈值筛选
          4) 连通域分析，统计有效查询区域
        """
        B, _, H, W = D.shape
        device = D.device

        # 1) 密度平滑（减少噪声）
        X = F.avg_pool2d(D, kernel_size=down, stride=down) if down > 1 else D  # [B,1,h,w]
        Bh, Bw = X.shape[2], X.shape[3]  # 下采样后的高度与宽度

        # 2) 聚类
        # 将密度图X展平为二维
        flat = X.view(B, -1)
        X_flat = flat.cpu().numpy()
        # 使用0替换NaN
        X_flat = np.nan_to_num(X_flat, nan=0.0)  # 使用 numpy 的 nan_to_num

        # 使用DBSCAN进行聚类，假设是使用欧氏距离
        db = DBSCAN(eps=grid_eps, min_samples=20).fit(X_flat)  # DBSCAN聚类
        labels = db.labels_  # 获取每个点的簇标签

        # 获取每个簇的响应数目
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # 3) 分位数阈值二值化
        thr = torch.quantile(flat, q, dim=1, keepdim=True).view(B, 1, 1, 1)
        Mb = (X >= thr).float()  # 小于阈值的部分置为0，大于阈值的部分置为1

        # 4) 连通域分析
        # 将二值化后的图像展平
        Mb_flat = Mb.view(B, -1).cpu().numpy()

        n_clusters_list = []
        for b in range(B):
            m = Mb_flat[b].reshape(Bh, Bw).astype(np.uint8)  # ✅ 改为下采样后的形状
            num_labels, labels = cv2.connectedComponents(m)
            n_clusters_list.append(num_labels - 1)

            # # 可视化每一张图像的密度图 X 和 聚类结果
            # if b == 0:  # 只在第一个批次中显示
            #     plt.figure(figsize=(12, 6))
            #
            #     # 可视化密度图 X
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(X[b, 0].cpu().numpy(), cmap='viridis')  # 选择第一张图像，去掉通道维度
            #     plt.title('Density Map (X)')
            #
            #     # 可视化聚类结果
            #     plt.subplot(1, 2, 2)
            #     cluster_img = labels.reshape(Bh, Bw)
            #     plt.imshow(cluster_img, cmap='tab20', interpolation='nearest')  # 使用不同的颜色区分簇
            #     plt.title('Clustering Result')
            #
            #     plt.show()

        # 计算每个图像的查询数K
        n_clusters_tensor = torch.tensor(n_clusters_list, device=device)
        K = (K_N + beta * n_clusters_tensor).clamp(K_N, K_M).long()
        return K, n_clusters_tensor

    def forward(
        self,
        x: torch.Tensor,
        q: float = 0.7,
        K_N: int = 100,
        K_M: int = 300,
        beta: float = 2.0,
        down: int = 4,
        eps: float = 6.0
    ):
        """
        返回：
          D_pred: [B,1,H,W] 归一化密度图 (H,W = out_size)
          K:      [B]       基于密度图的查询数估计（GPU-only）
        """
        # 密度图
        x = self.dilated_convs(x)
        x = x * self.channel_attn(x)
        x = self.density_head(x)
        D_pred = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False).sigmoid()

        # 估计查询数
        K, _ = self._estimate_queries_gpu(D_pred, K_N=K_N, K_M=K_M, q=q, down=down, grid_eps=int(round(eps)), beta=beta)
        return D_pred, K

# ------------------------------------------------------------------------------------------------
# RTDETR Decoder (optimized)
# ------------------------------------------------------------------------------------------------
class RTDETRDecoder(nn.Module):
    export = False  # export mode   ← 新增这一行
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6,
                 d_ffn=1024, dropout=0.0, act=nn.ReLU(), eval_idx=-1,
                 nd=100, label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.nhead = nh
        self.nl = len(ch)

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
        self._anchor_cache: Dict[Tuple[Tuple[int, int], ...], Tuple[torch.Tensor, torch.Tensor]] = {}

        self._reset_parameters()

    def forward(self, x, batch=None, K: Optional[torch.Tensor] = None):
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        embed, refer_bbox, enc_bboxes, enc_scores, attn_mask = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox, K=K
        )
        dec_bboxes, dec_scores = self.decoder(
            embed, refer_bbox, feats, shapes,
            self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            attn_mask=attn_mask
        )

        # 训练：固定 5 元素（不要多、不要少）
        x = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        if self.training:
            return x

        # 验证/推理：返回 (y, x)，其中 x 仍然是 5 元素
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _get_encoder_input(self, x):
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats, shapes = [], []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append((h, w))
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _generate_anchors_cached(self, shapes, dtype, device, grid_size=0.05, eps=1e-2):
        key = tuple(shapes)
        if key in self._anchor_cache:
            anchors, valid = self._anchor_cache[key]
            return anchors.to(device), valid.to(device)
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy, sx = torch.arange(h, device=device), torch.arange(w, device=device)
            gy, gx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([gx, gy], -1)
            valid_WH = torch.tensor([w, h], device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        self._anchor_cache[key] = (anchors.cpu(), valid_mask.cpu())
        return anchors, valid_mask

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, K=None):
        bs = feats.shape[0]
        device, dtype = feats.device, feats.dtype
        anchors, valid_mask = self._generate_anchors_cached(shapes, dtype, device)
        features = self.enc_output(valid_mask * feats)
        enc_scores = self.enc_score_head(features)
        enc_obj = enc_scores.max(-1).values

        # 动态 query 数
        if K is not None:
            K = K.clamp_min(1)
            k_max = int(K.max().item())
        else:
            k_max = self.num_queries

        topk_idx = torch.topk(enc_obj, k=k_max, dim=1).indices
        b_idx = torch.arange(bs, device=device)[:, None].expand_as(topk_idx)
        top_feat = features[b_idx, topk_idx]
        top_anchor = anchors[:, topk_idx].view(bs, k_max, -1)
        refer_bbox = self.enc_bbox_head(top_feat) + top_anchor
        enc_bboxes = refer_bbox.sigmoid()
        enc_scores = enc_scores[b_idx, topk_idx]

        # 构造 attn_mask 屏蔽超出 K 的位置
        attn_mask = None
        if K is not None:
            mask = torch.arange(k_max, device=device)[None, :].expand(bs, -1) >= K[:, None]
            # 构造 attn_mask 屏蔽超出 K 的位置
            attn_mask = None
            if K is not None:
                mask = torch.arange(k_max, device=device)[None, :].expand(bs, -1) >= K[:, None]
                # [B, 1, 1, k_max] -> [B, heads, k_max, k_max]
                mask = mask[:, None, None, :].expand(bs, self.nhead, k_max, k_max)
                attn_mask = mask.reshape(bs * self.nhead, k_max, k_max)

        embeddings = top_feat if not self.learnt_init_query else self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores, attn_mask

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0)
            constant_(reg_.layers[-1].bias, 0)
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


# ------------------------------------------------------------------------------------------------
# v10Detect
# ------------------------------------------------------------------------------------------------
class v10Detect(Detect):
    """v10 detection head."""
    end2end = True

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
