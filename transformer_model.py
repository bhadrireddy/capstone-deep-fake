import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm


class FFTBranchCNN(nn.Module):
    """
    Simple CNN branch for FFT (frequency) input.
    Input: (B, 1, H, W) or (B, 3, H, W)
    Output: embedding vector per image.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class RGBFFT_ViT(nn.Module):
    """
    Dual-branch model:
      - RGB branch: ViT backbone (pretrained on ImageNet)
      - FFT branch: small CNN on frequency-domain representation
    Fusion: concatenation + MLP → single logit for binary classification.
    """

    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        vit_embed_dim: int = 768,
        fft_embed_dim: int = 256,
        fusion_hidden_dim: int = 512,
        fft_in_channels: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()

        # RGB branch: Vision Transformer without classification head (num_classes=0 → features)
        self.rgb_backbone = timm.create_model(
            vit_model_name, pretrained=pretrained, num_classes=0
        )

        # FFT branch: lightweight CNN
        self.fft_branch = FFTBranchCNN(in_channels=fft_in_channels, embed_dim=fft_embed_dim)

        fusion_in_dim = vit_embed_dim + fft_embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, rgb: torch.Tensor, fft_img: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, 3, H, W)
        fft_img: (B, C_fft, H, W)
        returns logits: (B, 1)
        """
        rgb_embed = self.rgb_backbone(rgb)          # (B, vit_embed_dim)
        fft_embed = self.fft_branch(fft_img)        # (B, fft_embed_dim)

        fused = torch.cat([rgb_embed, fft_embed], dim=1)
        logits = self.fusion_mlp(fused)
        return logits.squeeze(1)


def _compute_fft_image(pil_img: Image.Image, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Compute normalized log-magnitude FFT image from a PIL image.
    Returns: float32 array in [0,1] of shape (H, W).
    """
    # Convert to grayscale
    img_gray = pil_img.convert("L")
    img_gray = img_gray.resize(size, Image.BILINEAR)
    arr = np.array(img_gray).astype(np.float32)

    # 2D FFT
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Log magnitude to compress dynamic range
    magnitude = np.log1p(magnitude)

    # Normalize to [0,1]
    magnitude -= magnitude.min()
    if magnitude.max() > 0:
        magnitude /= magnitude.max()

    return magnitude.astype(np.float32)


def _prepare_rgb_and_fft_tensors(
    pil_img: Image.Image,
    size: Tuple[int, int] = (224, 224),
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes a PIL image and returns:
      - rgb_tensor: (1, 3, H, W)
      - fft_tensor: (1, 1, H, W)
    """
    pil_rgb = pil_img.convert("RGB").resize(size, Image.BILINEAR)
    rgb_np = np.array(pil_rgb).astype(np.float32) / 255.0
    rgb_np = rgb_np.transpose(2, 0, 1)  # HWC → CHW

    fft_np = _compute_fft_image(pil_img, size=size)  # (H, W)
    fft_np = fft_np[None, ...]  # add channel → (1, H, W)

    rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device)   # (1, 3, H, W)
    fft_tensor = torch.from_numpy(fft_np).unsqueeze(0).to(device)   # (1, 1, H, W)

    return rgb_tensor, fft_tensor


def vit_fft_image_pred(
    image_path: str,
    threshold: float = 0.5,
    img_size: int = 224,
    device: torch.device = None,
) -> Tuple[str, float]:
    """
    High-level wrapper for single-image prediction with RGB+FFT ViT model.
    Returns:
      - label: 'real' or 'fake' (AI-generated)
      - prob_fake: sigmoid probability of being AI-generated

    NOTE: threshold is only used to derive the label; the model itself does NOT
    hardcode 0.5 and always returns the raw sigmoid probability.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pil_img = Image.open(image_path).convert("RGB")
    rgb_tensor, fft_tensor = _prepare_rgb_and_fft_tensors(
        pil_img, size=(img_size, img_size), device=device
    )

    # Instantiate model; in a production setting you should load trained weights here.
    model = RGBFFT_ViT(pretrained=True).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(rgb_tensor, fft_tensor)  # (1,)
        prob_fake = torch.sigmoid(logits).item()

    # Use provided threshold for label, but DO NOT hardcode 0.5 inside model.
    label = "fake" if prob_fake >= threshold else "real"
    return label, float(prob_fake)


def vit_fft_video_frame_preds(
    frames_rgb: torch.Tensor,
    frames_fft: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Run RGB+FFT ViT model on a batch of video frames.

    frames_rgb: (N, 3, H, W)
    frames_fft: (N, 1, H, W)
    Returns:
      probs_fake: (N,) sigmoid probabilities for each frame.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RGBFFT_ViT(pretrained=True).to(device)
    model.eval()

    frames_rgb = frames_rgb.to(device)
    frames_fft = frames_fft.to(device)

    with torch.no_grad():
        logits = model(frames_rgb, frames_fft)  # (N,)
        probs_fake = torch.sigmoid(logits)

    return probs_fake.cpu()


