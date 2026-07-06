from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGA(nn.Module):
    """
    Cascaded Group Attention.

    This is a lightweight, detection-friendly approximation of the CGA mechanism
    described in the paper. It keeps channel count unchanged and is safe for
    long-running training on limited GPU memory.
    """

    def __init__(self, c1: int, groups: int = 4):
        super().__init__()
        g = max(1, min(groups, c1))
        while c1 % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.group_channels = c1 // g

        self.attn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.group_channels,
                        self.group_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=self.group_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.group_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, stride=1, bias=False),
                    nn.Sigmoid(),
                )
                for _ in range(self.groups)
            ]
        )
        self.proj = nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.groups, dim=1)
        outs: list[torch.Tensor] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = chunk + outs[-1]
            attn = self.attn_blocks[i](chunk)
            outs.append(chunk * attn)
        y = torch.cat(outs, dim=1)
        return self.act(self.bn(self.proj(y)))


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention.

    EMA is a channel-preserving attention block used to replace CGA in the
    CEMA experiment. It groups channels, models horizontal/vertical coordinate
    context, and mixes local depthwise features. The output shape is identical
    to the input shape, so it is safe for Ultralytics YAML parsing as a custom
    module.
    """

    def __init__(self, c1: int, groups: int = 4, reduction: int = 4):
        super().__init__()
        g = max(1, min(int(groups), c1))
        while c1 % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.group_channels = c1 // g
        hidden = max(4, self.group_channels // max(1, int(reduction)))

        self.coord_fuse = nn.Sequential(
            nn.Conv2d(self.group_channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.group_channels, kernel_size=1, stride=1, bias=True),
        )
        self.local_mix = nn.Sequential(
            nn.Conv2d(
                self.group_channels,
                self.group_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.group_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.group_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.group_channels),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.group_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.group_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xg = x.reshape(b * self.groups, self.group_channels, h, w)

        x_h = F.adaptive_avg_pool2d(xg, (h, 1))
        x_w = F.adaptive_avg_pool2d(xg, (1, w)).transpose(2, 3)
        coord = self.coord_fuse(torch.cat([x_h, x_w], dim=2))
        attn_h, attn_w = torch.split(coord, [h, w], dim=2)
        coord_gate = attn_h.sigmoid() * attn_w.transpose(2, 3).sigmoid()

        local = self.local_mix(xg)
        y = local * coord_gate * self.channel_gate(local)
        y = y.reshape(b, c, h, w)
        return self.act(x + self.out(y))


class HAT(nn.Module):
    """
    Hybrid Attention Transformer.

    A lightweight HAT-style block with:
    1) local depthwise-conv branch
    2) pooled global MHSA branch
    3) residual fusion
    """

    def __init__(self, c1: int, heads: int = 4, pool_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.c1 = c1
        self.pool_size = max(2, int(pool_size))
        self.local = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
        )
        self.norm1 = nn.LayerNorm(c1)
        self.attn = nn.MultiheadAttention(embed_dim=c1, num_heads=max(1, heads), batch_first=True)
        self.norm2 = nn.LayerNorm(c1)
        hidden = int(c1 * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(c1, hidden),
            nn.GELU(),
            nn.Linear(hidden, c1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local_feat = self.local(x)

        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        tokens = pooled.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.norm1(tokens)
        attn_tokens, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_tokens
        tokens = tokens + self.mlp(self.norm2(tokens))
        global_feat = tokens.transpose(1, 2).reshape(b, c, self.pool_size, self.pool_size)
        global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([local_feat, global_feat], dim=1))
        return x + fused


class HATLite(nn.Module):
    """
    GFLOPS-aligned HAT approximation for the paper reproduction.

    It enhances the small-object P3 feature map with repeated local mixing and
    channel attention while avoiding the costly MHSA implementation that pushed
    the previous reproduction far above the paper's reported complexity.
    """

    def __init__(self, c1: int, repeats: int = 4, reduction: int = 4):
        super().__init__()
        repeats = max(1, int(repeats))
        hidden = max(8, c1 // max(1, int(reduction)))
        self.local = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(c1),
                    nn.SiLU(inplace=True),
                )
                for _ in range(repeats)
            ]
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)
        return x + y * self.channel_attn(y)


class MSEA(nn.Module):
    """
    Multi-Scale Edge Attention for coral detection.

    The paper's target scenes contain small coral regions, uneven illumination,
    and seabed textures that are visually close to the target classes. MSEA keeps
    the feature map size unchanged while emphasizing local contrast, multi-scale
    texture, and edge-like responses before prediction.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.dw3 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.dw5 = nn.Conv2d(c1, c1, kernel_size=5, stride=1, padding=2, groups=c1, bias=False)
        self.dw7 = nn.Conv2d(c1, c1, kernel_size=7, stride=1, padding=3, groups=c1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 3, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.fuse(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        contrast = x - local_mean

        avg_map = ms.mean(dim=1, keepdim=True)
        max_map = ms.amax(dim=1, keepdim=True)
        contrast_map = contrast.abs().mean(dim=1, keepdim=True)
        spatial = self.spatial_attn(torch.cat([avg_map, max_map, contrast_map], dim=1))
        channel = self.channel_attn(ms + contrast)

        y = ms * channel * (1.0 + spatial)
        return self.act(x + self.out(y))


class RLICBFA(nn.Module):
    """
    Reduced-Redundancy Lightweight Inverted Block with Concatenation-Based
    Feature Aggregation.

    This channel-preserving block keeps the YOLOv11n-CBFA/CIB idea lightweight:
    a reduced inverted branch extracts non-redundant local features, then
    concatenation-based aggregation fuses them back into the main stream.
    """

    def __init__(self, c1: int, reduction: int = 2):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.reduce = nn.Sequential(
            nn.Conv2d(c1, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 + hidden, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(self.reduce(x))
        fused = self.fuse(torch.cat([x, y], dim=1))
        return self.act(x + fused * self.gate(fused))


class MEFA(nn.Module):
    """
    Multi-scale Edge-aware Feature Aggregation.

    MEFA replaces the previous MSP-ELA-style local enhancer. It aggregates
    3x3, 5x5, and dilated local contexts, then injects an edge-aware spatial
    attention signal based on local contrast to better separate coral boundaries
    from visually similar seabed backgrounds.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.branch3 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=5, stride=1, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.aggregate = nn.Sequential(
            nn.Conv2d(c1 * 3, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = self.aggregate(torch.cat([self.branch3(x), self.branch5(x), self.branch_dilated(x)], dim=1))
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = (x - local_mean).abs()
        edge_map = torch.cat([edge.mean(dim=1, keepdim=True), edge.amax(dim=1, keepdim=True)], dim=1)
        y = multi_scale * self.channel_gate(multi_scale) * (1.0 + self.edge_gate(edge_map))
        return self.act(x + self.out(y))


class LECA(nn.Module):
    """
    Lightweight Edge-aware Coordinate Attention.

    LECA is the second-stage replacement for MEFA. It keeps the edge-aware idea
    but constrains it with coordinate attention, so coral boundaries are
    enhanced without amplifying every seabed texture equally. The block is
    channel-preserving and safe to insert at the same YAML positions as MEFA.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.edge_mix = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.coord_reduce = nn.Sequential(
            nn.Conv2d(c1, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.coord_h = nn.Conv2d(hidden, c1, kernel_size=1, stride=1, bias=True)
        self.coord_w = nn.Conv2d(hidden, c1, kernel_size=1, stride=1, bias=True)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        edge_feat = self.edge_mix(x)
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        contrast = (x - local_mean).abs()
        edge_map = torch.cat([contrast.mean(dim=1, keepdim=True), contrast.amax(dim=1, keepdim=True)], dim=1)

        x_h = F.adaptive_avg_pool2d(edge_feat, (h, 1))
        x_w = F.adaptive_avg_pool2d(edge_feat, (1, w)).transpose(2, 3)
        coord = self.coord_reduce(torch.cat([x_h, x_w], dim=2))
        attn_h, attn_w = torch.split(coord, [h, w], dim=2)
        coord_gate = self.coord_h(attn_h).sigmoid() * self.coord_w(attn_w).transpose(2, 3).sigmoid()

        y = edge_feat * coord_gate * self.channel_gate(edge_feat) * (1.0 + self.edge_gate(edge_map))
        return self.act(x + self.out(y))


class HGA(nn.Module):
    """
    Hierarchical Group-wise Attention.

    HGA keeps the successful group-attention idea but adds global context to
    each group and cascades information from shallow groups to deeper groups,
    improving responses on key coral regions with limited extra computation.
    """

    def __init__(self, c1: int, groups: int = 4, reduction: int = 4):
        super().__init__()
        g = max(1, min(int(groups), c1))
        while c1 % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.group_channels = c1 // g
        hidden = max(4, self.group_channels // max(1, int(reduction)))
        self.group_attn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.group_channels, hidden, kernel_size=1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, self.group_channels, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
                for _ in range(self.groups)
            ]
        )
        self.local_mix = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.group_channels,
                        self.group_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=self.group_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.group_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in range(self.groups)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.groups, dim=1)
        outputs: list[torch.Tensor] = []
        carry: torch.Tensor | None = None
        for i, chunk in enumerate(chunks):
            if carry is not None:
                chunk = chunk + carry
            mixed = self.local_mix[i](chunk)
            out = mixed * self.group_attn[i](mixed)
            outputs.append(out)
            carry = out
        return self.act(x + self.proj(torch.cat(outputs, dim=1)))


class CBGS(nn.Module):
    """
    Coral Boundary-Guided Small-object head block.

    CBGS is designed for the final detection features. It preserves the channel
    count while using fixed Sobel-like boundary cues and lightweight context
    reweighting to improve small coral localization in cluttered seabed scenes.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.detail_mix = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=5, stride=1, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        sx = self.sobel_x.to(dtype=gray.dtype, device=gray.device)
        sy = self.sobel_y.to(dtype=gray.dtype, device=gray.device)
        edge_x = F.conv2d(gray, sx, padding=1)
        edge_y = F.conv2d(gray, sy, padding=1)
        edge = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
        contrast = (gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)).abs()

        detail = self.detail_mix(x)
        boundary = self.boundary_gate(torch.cat([edge, contrast], dim=1))
        y = detail * self.context_gate(detail) * (1.0 + boundary)
        return self.act(x + self.out(y))


class WFPN(nn.Module):
    """
    Weighted Feature Pyramid fusion refinement.

    Ultralytics only passes channel arguments automatically for built-in modules,
    so this block is inserted after a normal Concat layer. It splits the
    concatenated tensor back into its source-scale channel groups, learns a
    non-negative weight for each group, and refines the weighted multi-scale
    feature without changing the channel count.
    """

    def __init__(self, c1: int, splits: list[int] | tuple[int, ...], reduction: int = 4):
        super().__init__()
        self.c1 = int(c1)
        self.splits = [int(x) for x in splits]
        if sum(self.splits) != self.c1:
            raise ValueError(f"WFPN splits {self.splits} must sum to c1={self.c1}.")
        hidden = max(8, self.c1 // max(1, int(reduction)))
        self.weights = nn.Parameter(torch.ones(len(self.splits), dtype=torch.float32))
        self.local_refine = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c1, self.c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.c1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.splits, dim=1)
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-6)
        fused = torch.cat([part * weights[i] * len(parts) for i, part in enumerate(parts)], dim=1)
        refined = self.local_refine(fused)
        return self.act(x + refined * self.channel_gate(refined))


class CAFM(nn.Module):
    """
    Class-Aware Feature Modulation.

    CAFM is inserted after CBGS detection features. It predicts lightweight
    class-response maps from the current feature, converts the class prior into
    channel modulation, and combines it with spatial saliency. The block keeps
    channels and feature-map size unchanged, so it can be placed before Detect
    without changing the P2/P3/P4/P5 wiring.
    """

    def __init__(self, c1: int, reduction: int = 4, classes: int = 3):
        super().__init__()
        classes = max(1, int(classes))
        hidden = max(8, c1 // max(1, int(reduction)))
        self.classes = classes
        self.local_context = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.class_logits = nn.Conv2d(c1, classes, kernel_size=1, stride=1, bias=True)
        self.class_to_channel = nn.Sequential(
            nn.Conv2d(classes, hidden, kernel_size=1, stride=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.local_context(x)
        probs = F.softmax(self.class_logits(context), dim=1)
        class_prior = F.adaptive_avg_pool2d(probs, 1)
        class_gate = self.class_to_channel(class_prior)

        confidence = probs.amax(dim=1, keepdim=True)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)
        if self.classes > 1:
            entropy = entropy / torch.log(torch.tensor(float(self.classes), dtype=entropy.dtype, device=entropy.device))
        class_saliency = confidence * (1.0 - entropy.clamp(0.0, 1.0))
        spatial = self.spatial_gate(
            torch.cat(
                [
                    context.mean(dim=1, keepdim=True),
                    context.amax(dim=1, keepdim=True),
                    class_saliency,
                ],
                dim=1,
            )
        )

        y = context * self.channel_gate(context) * (1.0 + 0.5 * class_gate) * (1.0 + spatial)
        return self.act(x + self.out(y))


class MFEM(nn.Module):
    """
    Multi-scale Frequency Edge Module.

    MFEM strengthens coral texture cues by combining multi-scale depthwise
    context, local high-frequency residuals, and a fixed Laplacian response.
    It is channel-preserving so it can replace MEFA/MSP-ELA positions safely.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.local3 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.local5 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=5, stride=1, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.local_dilated = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 4, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.freq_gate = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplacian", lap, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        gray = x.mean(dim=1, keepdim=True)
        lap = self.laplacian.to(dtype=gray.dtype, device=gray.device)
        lap_edge = F.conv2d(gray, lap, padding=1).abs()
        contrast = (gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)).abs()

        multi = self.fuse(torch.cat([self.local3(x), self.local5(x), self.local_dilated(x), high], dim=1))
        gate = self.freq_gate(torch.cat([lap_edge, contrast, high.abs().mean(dim=1, keepdim=True)], dim=1))
        y = multi * self.channel_gate(multi + high) * (1.0 + gate)
        return self.act(x + self.out(y))


class BFCA(nn.Module):
    """
    Boundary-Frequency Collaborative Attention.

    BFCA is a detection-head enhancement for coral targets. It couples Sobel
    boundaries, Laplacian high-frequency responses, and lightweight semantic
    context to emphasize real coral contours while suppressing seabed texture.
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, c1 // max(1, int(reduction)))
        self.detail = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=2, dilation=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.freq_mix = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=5, stride=1, padding=2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU(inplace=True)

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplacian", lap, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        sx = self.sobel_x.to(dtype=gray.dtype, device=gray.device)
        sy = self.sobel_y.to(dtype=gray.dtype, device=gray.device)
        lap = self.laplacian.to(dtype=gray.dtype, device=gray.device)
        edge_x = F.conv2d(gray, sx, padding=1)
        edge_y = F.conv2d(gray, sy, padding=1)
        edge = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
        lap_edge = F.conv2d(gray, lap, padding=1).abs()
        contrast = (gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)).abs()

        detail = self.detail(x)
        freq = self.freq_mix(x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2))
        spatial = self.spatial_gate(torch.cat([edge, lap_edge, contrast, detail.mean(dim=1, keepdim=True)], dim=1))
        y = (detail + freq) * self.semantic_gate(detail) * (1.0 + spatial)
        return self.act(x + self.out(y))


def register_cshc_modules() -> None:
    """Inject custom modules into ultralytics runtime namespace."""
    import ultralytics.nn.modules as u_modules
    import ultralytics.nn.tasks as u_tasks

    setattr(u_modules, "BFCA", BFCA)
    setattr(u_modules, "CAFM", CAFM)
    setattr(u_modules, "CGA", CGA)
    setattr(u_modules, "EMA", EMA)
    setattr(u_modules, "HAT", HAT)
    setattr(u_modules, "HATLite", HATLite)
    setattr(u_modules, "MFEM", MFEM)
    setattr(u_modules, "MSEA", MSEA)
    setattr(u_modules, "RLICBFA", RLICBFA)
    setattr(u_modules, "MEFA", MEFA)
    setattr(u_modules, "LECA", LECA)
    setattr(u_modules, "HGA", HGA)
    setattr(u_modules, "CBGS", CBGS)
    setattr(u_modules, "WFPN", WFPN)
    setattr(u_tasks, "BFCA", BFCA)
    setattr(u_tasks, "CAFM", CAFM)
    setattr(u_tasks, "CGA", CGA)
    setattr(u_tasks, "EMA", EMA)
    setattr(u_tasks, "HAT", HAT)
    setattr(u_tasks, "HATLite", HATLite)
    setattr(u_tasks, "MFEM", MFEM)
    setattr(u_tasks, "MSEA", MSEA)
    setattr(u_tasks, "RLICBFA", RLICBFA)
    setattr(u_tasks, "MEFA", MEFA)
    setattr(u_tasks, "LECA", LECA)
    setattr(u_tasks, "HGA", HGA)
    setattr(u_tasks, "CBGS", CBGS)
    setattr(u_tasks, "WFPN", WFPN)
