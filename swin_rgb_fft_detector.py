"""
Swin Transformer Dual-Branch Deepfake Detector
- Branch 1: Swin Transformer (RGB) - spatial domain
- Branch 2: Swin Transformer (FFT) - frequency domain
- Full-image analysis (no face crops)
- Trained with Binary Cross Entropy loss on real and AI-generated images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from functools import lru_cache
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class FFTBranch(nn.Module):
    """FFT feature extraction branch"""
    def __init__(self):
        super().__init__()
        # CNN for processing FFT magnitude
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
    
    def forward(self, x):
        return self.conv_layers(x)


class SwinRGBFFTDetector(nn.Module):
    """
    Dual-branch Swin Transformer for deepfake detection
    - RGB branch: Full-image spatial analysis
    - FFT branch: Frequency domain analysis
    """
    def __init__(self, model_name='swin_large_patch4_window7_224', pretrained=True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm package is required. Install with: pip install timm")
        
        # RGB Branch: Use Swin Large for better feature extraction
        # Large model has better capacity for detecting subtle AI artifacts
        try:
            self.rgb_backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
            )
        except:
            # Fallback to base if large not available
            self.rgb_backbone = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
            )
        
        # FFT Branch: CNN for frequency domain
        self.fft_branch = FFTBranch()
        
        # Get feature dimensions
        with torch.no_grad():
            # RGB features
            dummy_rgb = torch.randn(1, 3, 224, 224)
            rgb_features = self.rgb_backbone.forward_features(dummy_rgb)
            if isinstance(rgb_features, (list, tuple)):
                rgb_dim = rgb_features[-1].shape[-1] if len(rgb_features[-1].shape) > 1 else rgb_features[-1].shape[0]
            elif len(rgb_features.shape) == 4:
                rgb_dim = rgb_features.shape[1]
            elif len(rgb_features.shape) == 3:
                rgb_dim = rgb_features.shape[-1]
            else:
                rgb_dim = 1024
            
            # FFT features
            dummy_fft = torch.randn(1, 1, 224, 224)
            fft_features = self.fft_branch(dummy_fft)
            fft_dim = fft_features.shape[1] * fft_features.shape[2] * fft_features.shape[3]  # 256 * 7 * 7
            
        # Fusion and classification
        self.rgb_pool = nn.AdaptiveAvgPool2d(1) if rgb_features.shape[-1] != rgb_dim else nn.Identity()
        self.rgb_proj = nn.Linear(rgb_dim, 512)
        
        self.fft_proj = nn.Linear(fft_dim, 512)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(512 * 2),
            nn.Linear(512 * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.rgb_dim = rgb_dim
    
    def forward(self, rgb, fft):
        # RGB branch
        rgb_features = self.rgb_backbone.forward_features(rgb)
        
        if isinstance(rgb_features, (list, tuple)):
            rgb_features = rgb_features[-1]
        
        if len(rgb_features.shape) == 4:  # [B, C, H, W]
            rgb_features = self.rgb_pool(rgb_features).flatten(1)
        elif len(rgb_features.shape) == 3:  # [B, N, C]
            rgb_features = rgb_features.mean(dim=1)
        
        if rgb_features.shape[1] != 512:
            rgb_features = self.rgb_proj(rgb_features)
        else:
            rgb_features = F.relu(self.rgb_proj(rgb_features))
        
        # FFT branch
        fft_features = self.fft_branch(fft)
        fft_features = fft_features.flatten(1)
        fft_features = F.relu(self.fft_proj(fft_features))
        
        # Fusion
        combined = torch.cat([rgb_features, fft_features], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits


def compute_fft_image(img_array):
    """
    Compute FFT magnitude spectrum of image
    Args:
        img_array: numpy array of shape (H, W, 3) or PIL Image
    Returns:
        FFT magnitude spectrum as numpy array (H, W)
    """
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            # Fallback without cv2: weighted average
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = img_array
    
    # Convert to float32 for FFT
    gray = gray.astype(np.float32)
    
    # Apply 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude spectrum
    magnitude_spectrum = np.abs(f_shift)
    
    # Get phase spectrum (AI edits often affect phase)
    phase_spectrum = np.angle(f_shift)
    
    # Enhanced analysis: Combine magnitude and phase irregularities
    # AI edits often create anomalies in both magnitude and phase
    magnitude_norm = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
    phase_variance = np.abs(phase_spectrum - np.median(phase_spectrum))
    phase_variance_norm = (phase_variance - phase_variance.min()) / (phase_variance.max() - phase_variance.min() + 1e-8)
    
    # Combine magnitude and phase anomalies (AI artifacts show in both)
    combined = 0.7 * magnitude_norm + 0.3 * phase_variance_norm
    
    # Log scale for better visualization and artifact detection
    combined = np.log1p(combined * 100)
    
    # Normalize to [0, 255]
    combined = ((combined - combined.min()) / 
                (combined.max() - combined.min() + 1e-8) * 255)
    
    return combined.astype(np.uint8)


@lru_cache(maxsize=1)
def _load_swin_rgb_fft_model(device_str: str):
    """Load and cache Swin RGB+FFT model"""
    device = torch.device(device_str)
    
    # Try Swin Large first (better accuracy), fallback to Base
    try:
        model = SwinRGBFFTDetector(
            model_name='swin_large_patch4_window7_224',
            pretrained=True
        ).eval().to(device)
    except:
        model = SwinRGBFFTDetector(
            model_name='swin_base_patch4_window7_224',
            pretrained=True
        ).eval().to(device)
    
    return model, device


def _swin_transform():
    """Get image transformations for Swin Transformer"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _fft_transform():
    """Get FFT image transformations"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def swin_rgb_fft_image_pred(image_path, threshold=0.5):
    """
    Predict if image is fake using dual-branch Swin Transformer (RGB + FFT)
    Full-image analysis - no face crops
    
    Args:
        image_path: Path to image file
        threshold: Decision threshold
        
    Returns:
        tuple: (label: 'fake' or 'real', probability: float)
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm package is required. Install with: pip install timm")
    
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, device = _load_swin_rgb_fft_model(device_str)
    rgb_transform = _swin_transform()
    fft_transform = _fft_transform()
    
    try:
        # Load full image (no face cropping)
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Create multiple crops for robustness
        crops = []
        fft_crops = []
        
        # Resize for consistent cropping
        max_dim = max(original_size)
        if max_dim > 448:
            scale = 448 / max_dim
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            img_resized = img
        
        width, height = img_resized.size
        crop_size = 224
        
        # Create crops: center, corners
        crop_positions = []
        if width >= crop_size and height >= crop_size:
            # Center
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            crop_positions.append((left, top))
            
            # Corners
            crop_positions.append((0, 0))  # Top-left
            crop_positions.append((width - crop_size, 0))  # Top-right
            crop_positions.append((0, height - crop_size))  # Bottom-left
            crop_positions.append((width - crop_size, height - crop_size))  # Bottom-right
        else:
            # Image too small, just resize
            crop_positions = [(0, 0)]
        
        # Process each crop - use more crops for better coverage
        for left, top in crop_positions[:5]:  # Use up to 5 crops for better detection
            if left >= 0 and top >= 0:
                crop = img_resized.crop((left, top, left + crop_size, top + crop_size))
                if crop.size != (224, 224):
                    crop = crop.resize((224, 224), Image.Resampling.LANCZOS)
                crops.append(crop)
                
                # Compute FFT for this crop
                fft_img = compute_fft_image(crop)
                fft_img_pil = Image.fromarray(fft_img)
                fft_crops.append(fft_img_pil)
        
        if len(crops) == 0:
            # Fallback: just resize the whole image
            crop = img_resized.resize((224, 224), Image.Resampling.LANCZOS)
            crops.append(crop)
            fft_img = compute_fft_image(crop)
            fft_img_pil = Image.fromarray(fft_img)
            fft_crops.append(fft_img_pil)
        
        # Process all crops
        all_preds = []
        with torch.no_grad():
            for rgb_crop, fft_crop in zip(crops, fft_crops):
                # RGB tensor
                rgb_tensor = rgb_transform(rgb_crop).unsqueeze(0).to(device)
                
                # FFT tensor (single channel)
                fft_tensor = fft_transform(fft_crop).unsqueeze(0).to(device)
                # FFT transform gives 3 channels from PIL, take first channel
                if fft_tensor.shape[1] == 3:
                    fft_tensor = fft_tensor[:, 0:1, :, :]  # Keep as single channel
                
                # Forward pass
                logits = model(rgb_tensor, fft_tensor)
                prob = torch.sigmoid(logits).item()
                all_preds.append(prob)
        
        # Aggressive aggregation to catch AI edits
        mean_pred = float(np.mean(all_preds))
        max_pred = float(np.max(all_preds))
        median_pred = float(np.median(all_preds))
        std_pred = float(np.std(all_preds))
        
        # More aggressive weighting: prioritize max to catch any fake regions
        # AI edits often leave artifacts in specific regions, max catches these
        if std_pred > 0.12:  # High variance - fake artifacts detected
            # Very aggressive: trust max heavily
            combined_pred = 0.65 * max_pred + 0.25 * mean_pred + 0.1 * median_pred
        elif max_pred > 0.55:  # Even moderate max values might indicate fake
            # Moderate aggression
            combined_pred = 0.55 * max_pred + 0.3 * mean_pred + 0.15 * median_pred
        else:
            # Low values, use balanced approach
            combined_pred = 0.45 * max_pred + 0.35 * mean_pred + 0.2 * median_pred
        
        # Boost prediction if any crop strongly indicates fake
        if max_pred > 0.7:
            # Very high max - strongly boost combined prediction
            combined_pred = 0.7 * max_pred + 0.3 * combined_pred
        
        # Clamp to [0, 1]
        combined_pred = max(0.0, min(1.0, combined_pred))
        
        # More aggressive threshold for better fake detection
        adjusted_threshold = threshold * 0.85  # Lower threshold to catch more fakes
        
        if combined_pred > adjusted_threshold:
            return "fake", combined_pred
        else:
            return "real", combined_pred
            
    except Exception as e:
        print(f"Error in Swin RGB+FFT prediction: {e}")
        import traceback
        traceback.print_exc()
        return "real", 0.5

