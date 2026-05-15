#!/usr/bin/env python3
"""
Paper-inspired multi-camera triplane tokenizer.

Implements the core mechanics from:
  "Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving"

This is not the authors' exact code. The paper uses learned per-image and
cross-image deformable attention over projected 3D queries. This script keeps
the same geometry-aware contract, but uses differentiable projection +
grid_sample feature lifting as a compact, runnable reference implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PlaneDict = Dict[str, Tensor]


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


@dataclass(frozen=True)
class PaperTriplaneConfig:
    image_height: int = 128
    image_width: int = 224
    feature_channels: int = 64
    plane_channels: int = 64
    volume_resolution: int = 24
    token_dim: int = 256
    patch_xy: int = 8
    patch_xz: int = 8
    patch_yz: int = 8
    x_range: Tuple[float, float] = (-45.0, 45.0)
    y_range: Tuple[float, float] = (-45.0, 45.0)
    z_range: Tuple[float, float] = (-3.0, 12.0)

    def __post_init__(self) -> None:
        for name, patch in (("xy", self.patch_xy), ("xz", self.patch_xz), ("yz", self.patch_yz)):
            if patch <= 0:
                raise ValueError(f"patch_{name} must be positive")
            if patch > self.volume_resolution:
                raise ValueError(f"patch_{name} cannot exceed volume_resolution")


class ImageEncoder(nn.Module):
    """Small per-camera image encoder. Replace with DINOv2/ResNet in production."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            group_norm(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            group_norm(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        batch, cameras, channels, height, width = images.shape
        flat = images.view(batch * cameras, channels, height, width)
        features = self.net(flat)
        _, feat_channels, feat_height, feat_width = features.shape
        return features.view(batch, cameras, feat_channels, feat_height, feat_width)


class GeometryLifter(nn.Module):
    """
    Lifts multi-camera image features into an ego-centered 3D feature volume.

    Args:
        features: B x N x C x Hf x Wf image features.
        intrinsics: B x N x 3 x 3 camera intrinsics in original image pixels.
        camera_from_ego: B x N x 4 x 4 transforms from ego/world to camera frame.
    """

    def __init__(self, config: PaperTriplaneConfig) -> None:
        super().__init__()
        self.config = config
        self.fusion = nn.Sequential(
            nn.Linear(config.feature_channels + 1, config.plane_channels),
            nn.GELU(),
            nn.Linear(config.plane_channels, config.plane_channels),
        )

    def forward(self, features: Tensor, intrinsics: Tensor, camera_from_ego: Tensor) -> Tensor:
        cfg = self.config
        points = ego_grid(cfg, features.device, features.dtype)
        batch, cameras, channels, feat_height, feat_width = features.shape

        lifted_features = []
        lifted_masks = []
        for camera_index in range(cameras):
            sampled, mask = project_and_sample(
                features[:, camera_index],
                points,
                intrinsics[:, camera_index],
                camera_from_ego[:, camera_index],
                cfg.image_height,
                cfg.image_width,
            )
            lifted_features.append(sampled)
            lifted_masks.append(mask)

        per_camera = torch.stack(lifted_features, dim=1)
        masks = torch.stack(lifted_masks, dim=1)
        weighted = per_camera * masks.unsqueeze(-1)
        visibility = masks.sum(dim=1, keepdim=False).clamp_min(1.0)
        fused = weighted.sum(dim=1) / visibility.unsqueeze(-1)

        visibility_ratio = masks.mean(dim=1, keepdim=False).unsqueeze(-1)
        fused = self.fusion(torch.cat([fused, visibility_ratio], dim=-1))
        return fused.transpose(1, 2).view(
            batch,
            cfg.plane_channels,
            cfg.volume_resolution,
            cfg.volume_resolution,
            cfg.volume_resolution,
        )


class TriplaneProjector(nn.Module):
    """Averages the lifted feature volume into XY, XZ, and YZ planes."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, volume: Tensor) -> PlaneDict:
        return {
            "xy": volume.mean(dim=2),
            "xz": volume.mean(dim=3),
            "yz": volume.mean(dim=4),
        }


class PlanePatchTokenizer(nn.Module):
    """Patchifies each plane and maps plane patches to downstream token width."""

    def __init__(self, config: PaperTriplaneConfig) -> None:
        super().__init__()
        self.config = config
        max_patch = max(config.patch_xy, config.patch_xz, config.patch_yz)
        self.max_patch_values = config.plane_channels * max_patch * max_patch
        self.proj = nn.Sequential(
            nn.LayerNorm(self.max_patch_values),
            nn.Linear(self.max_patch_values, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.plane_embedding = nn.Parameter(torch.randn(3, config.token_dim) * 0.02)

    def forward(self, planes: PlaneDict, front_half_only: bool = False) -> Tensor:
        specs = [("xy", self.config.patch_xy), ("xz", self.config.patch_xz), ("yz", self.config.patch_yz)]
        token_groups = []
        for plane_index, (name, patch_size) in enumerate(specs):
            plane = crop_front_half(planes[name], name) if front_half_only and name in {"xy", "xz"} else planes[name]
            patches = F.unfold(plane, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
            patches = F.pad(patches, (0, self.max_patch_values - patches.shape[-1]))
            tokens = self.proj(patches) + self.plane_embedding[plane_index].view(1, 1, -1)
            token_groups.append(tokens)
        return torch.cat(token_groups, dim=1)


class MultiCameraTriplaneTokenizer(nn.Module):
    """End-to-end paper-inspired tokenizer: cameras -> triplanes -> tokens."""

    def __init__(self, config: PaperTriplaneConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config.feature_channels)
        self.lifter = GeometryLifter(config)
        self.projector = TriplaneProjector()
        self.tokenizer = PlanePatchTokenizer(config)

    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        camera_from_ego: Tensor,
        front_half_only: bool = False,
    ) -> Tuple[PlaneDict, Tensor]:
        features = self.image_encoder(images)
        volume = self.lifter(features, intrinsics, camera_from_ego)
        planes = self.projector(volume)
        tokens = self.tokenizer(planes, front_half_only=front_half_only)
        return planes, tokens


class RadianceHead(nn.Module):
    """Small decoder for offline volumetric reconstruction training."""

    def __init__(self, plane_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(plane_channels, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, triplane_features: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.net(triplane_features)
        rgb = torch.sigmoid(out[..., :3])
        sigma = F.softplus(out[..., 3:4])
        return rgb, sigma


def ego_grid(config: PaperTriplaneConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    xs = torch.linspace(*config.x_range, config.volume_resolution, device=device, dtype=dtype)
    ys = torch.linspace(*config.y_range, config.volume_resolution, device=device, dtype=dtype)
    zs = torch.linspace(*config.z_range, config.volume_resolution, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([x, y, z], dim=-1).view(-1, 3)


def project_and_sample(
    feature: Tensor,
    ego_points: Tensor,
    intrinsic: Tensor,
    camera_from_ego: Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[Tensor, Tensor]:
    batch, channels, feat_height, feat_width = feature.shape
    points_h = torch.cat([ego_points, torch.ones_like(ego_points[:, :1])], dim=-1)
    points_h = points_h.unsqueeze(0).expand(batch, -1, -1)
    camera_points = torch.bmm(points_h, camera_from_ego.transpose(1, 2))[..., :3]

    depth = camera_points[..., 2].clamp_min(1e-4)
    x = camera_points[..., 0] / depth
    y = camera_points[..., 1] / depth

    fx = intrinsic[:, 0, 0].view(batch, 1) * (feat_width / image_width)
    fy = intrinsic[:, 1, 1].view(batch, 1) * (feat_height / image_height)
    cx = intrinsic[:, 0, 2].view(batch, 1) * (feat_width / image_width)
    cy = intrinsic[:, 1, 2].view(batch, 1) * (feat_height / image_height)

    u = fx * x + cx
    v = fy * y + cy
    grid_x = (u / max(feat_width - 1, 1)) * 2.0 - 1.0
    grid_y = (v / max(feat_height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(batch, 1, ego_points.shape[0], 2)

    visible = (
        (camera_points[..., 2] > 0.0)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    sampled = F.grid_sample(feature, grid, align_corners=True, padding_mode="zeros")
    sampled = sampled.squeeze(2).transpose(1, 2)
    return sampled, visible.to(sampled.dtype)


def sample_triplanes(planes: PlaneDict, xyz_normalized: Tensor) -> Tensor:
    """Samples triplane features and aggregates with elementwise product, as in the paper."""
    grids = {
        "xy": xyz_normalized[..., [0, 1]],
        "xz": xyz_normalized[..., [0, 2]],
        "yz": xyz_normalized[..., [1, 2]],
    }
    samples = []
    for name, grid in grids.items():
        grid = grid.view(grid.shape[0], 1, grid.shape[1], 2)
        sampled = F.grid_sample(planes[name], grid, align_corners=True)
        samples.append(sampled.squeeze(2).transpose(1, 2))
    return samples[0] * samples[1] * samples[2]


def crop_front_half(plane: Tensor, name: str) -> Tensor:
    midpoint = plane.shape[-1] // 2
    if name == "xy":
        return plane[..., midpoint:]
    if name == "xz":
        return plane[..., midpoint:]
    return plane


def synthetic_calibration(batch: int, cameras: int, config: PaperTriplaneConfig) -> Tuple[Tensor, Tensor]:
    intrinsics = torch.eye(3).repeat(batch, cameras, 1, 1)
    intrinsics[:, :, 0, 0] = 120.0
    intrinsics[:, :, 1, 1] = 120.0
    intrinsics[:, :, 0, 2] = config.image_width / 2.0
    intrinsics[:, :, 1, 2] = config.image_height / 2.0

    camera_from_ego = torch.eye(4).repeat(batch, cameras, 1, 1)
    yaws = torch.linspace(-0.9, 0.9, cameras)
    for camera_index, yaw in enumerate(yaws):
        c, s = torch.cos(yaw), torch.sin(yaw)
        # Camera looks along ego x; convert ego forward into camera z.
        rotation = torch.tensor([[s, -c, 0.0], [0.0, 0.0, -1.0], [c, s, 0.0]])
        camera_from_ego[:, camera_index, :3, :3] = rotation
        camera_from_ego[:, camera_index, :3, 3] = torch.tensor([0.0, -1.2, -1.5])
    return intrinsics, camera_from_ego


def run_demo(args: argparse.Namespace) -> None:
    config = PaperTriplaneConfig(
        volume_resolution=args.volume_resolution,
        patch_xy=args.patch_xy,
        patch_xz=args.patch_xz,
        patch_yz=args.patch_yz,
    )
    model = MultiCameraTriplaneTokenizer(config).eval()
    images = torch.randn(args.batch_size, args.cameras, 3, config.image_height, config.image_width)
    intrinsics, camera_from_ego = synthetic_calibration(args.batch_size, args.cameras, config)

    with torch.no_grad():
        planes, tokens = model(
            images,
            intrinsics,
            camera_from_ego,
            front_half_only=args.front_half_only,
        )
        points = torch.rand(args.batch_size, 256, 3) * 2.0 - 1.0
        triplane_features = sample_triplanes(planes, points)
        rgb, sigma = RadianceHead(config.plane_channels)(triplane_features)

    print(f"images:  {tuple(images.shape)}")
    print(f"xy:      {tuple(planes['xy'].shape)}")
    print(f"xz:      {tuple(planes['xz'].shape)}")
    print(f"yz:      {tuple(planes['yz'].shape)}")
    print(f"tokens:  {tuple(tokens.shape)}")
    print(f"render head rgb/sigma: {tuple(rgb.shape)} / {tuple(sigma.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-inspired multi-camera triplane tokenizer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--volume-resolution", type=int, default=24)
    parser.add_argument("--patch-xy", type=int, default=8)
    parser.add_argument("--patch-xz", type=int, default=8)
    parser.add_argument("--patch-yz", type=int, default=8)
    parser.add_argument("--front-half-only", action="store_true")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
