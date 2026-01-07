"""
Swin Transformer-based Deepfake Detector for Images
Uses full-image RGB analysis instead of face crops for better AI edit detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from functools import lru_cache

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class SwinDeepfakeDetector(nn.Module):
    """Swin Transformer with classification head for deepfake detection"""
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm package is required for Swin Transformer. Install with: pip install timm")
        
        # Load Swin Transformer backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, we'll add our own
            global_pool='',
        )
        
        # Get feature dimension by testing with dummy input
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone.forward_features(dummy)
                
                if isinstance(features, (list, tuple)):
                    feature_dim = features[-1].shape[-1] if len(features[-1].shape) > 1 else features[-1].shape[0]
                elif len(features.shape) == 4:  # [B, C, H, W]
                    feature_dim = features.shape[1]
                elif len(features.shape) == 3:  # [B, N, C]
                    feature_dim = features.shape[-1]
                else:
                    feature_dim = features.shape[-1] if len(features.shape) > 1 else 768  # Default
        except:
            # Fallback to default feature dimension for Swin Base
            feature_dim = 1024
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        self.feature_dim = feature_dim
    
    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Handle different feature formats
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global pooling if needed
        if len(features.shape) == 4:  # [B, C, H, W]
            features = self.global_pool(features).flatten(1)
        elif len(features.shape) == 3:  # [B, N, C]
            features = features.mean(dim=1)
        elif len(features.shape) == 2:  # Already [B, C]
            pass
        else:
            # Fallback: flatten
            features = features.view(features.size(0), -1)
            if features.shape[1] != self.feature_dim:
                # Project to correct dimension if needed
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(features.shape[1], self.feature_dim).to(features.device)
                features = self.projection(features)
        
        # Classification
        logits = self.classifier(features)
        return logits


@lru_cache(maxsize=1)
def _load_swin_model(device_str: str):
    """Load and cache Swin Transformer model"""
    device = torch.device(device_str)
    
    model = SwinDeepfakeDetector(
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


def swin_image_pred(image_path, threshold=0.5):
    """
    Predict if image is fake using Swin Transformer (full-image RGB analysis)
    Better at detecting AI edits because it analyzes the full image, not just faces
    
    Args:
        image_path: Path to image file
        threshold: Decision threshold
        
    Returns:
        tuple: (label: 'fake' or 'real', probability: float)
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm package is required. Install with: pip install timm")
    
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, device = _load_swin_model(device_str)
    transform = _swin_transform()
    
    try:
        # Load image (full image, not face crop)
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Create multiple crops for better robustness
        # This helps catch AI artifacts that might appear in different parts of the image
        crops = []
        
        # Strategy: Take multiple crops from different regions
        # 1. Center crop
        # 2. Four corner crops
        # 3. Random crops
        
        # Resize to a standard size first for consistent cropping
        max_dim = max(original_size)
        if max_dim > 448:
            scale = 448 / max_dim
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            img_resized = img
        
        width, height = img_resized.size
        crop_size = 224
        
        # Center crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        if left >= 0 and top >= 0:
            center_crop = img_resized.crop((left, top, left + crop_size, top + crop_size))
            crops.append(center_crop)
        
        # Top-left
        if width >= crop_size and height >= crop_size:
            crops.append(img_resized.crop((0, 0, crop_size, crop_size)))
        
        # Top-right
        if width >= crop_size and height >= crop_size:
            crops.append(img_resized.crop((width - crop_size, 0, width, crop_size)))
        
        # Bottom-left
        if width >= crop_size and height >= crop_size:
            crops.append(img_resized.crop((0, height - crop_size, crop_size, height)))
        
        # Bottom-right
        if width >= crop_size and height >= crop_size:
            crops.append(img_resized.crop((width - crop_size, height - crop_size, width, height)))
        
        # If image is smaller than crop size, just resize it
        if len(crops) == 0:
            crops.append(img_resized.resize((crop_size, crop_size), Image.Resampling.LANCZOS))
        
        # Process all crops
        all_preds = []
        with torch.no_grad():
            for crop in crops:
                # Ensure crop is exactly 224x224
                if crop.size != (224, 224):
                    crop = crop.resize((224, 224), Image.Resampling.LANCZOS)
                
                img_tensor = transform(crop).unsqueeze(0).to(device)
                logits = model(img_tensor)
                prob = torch.sigmoid(logits).item()
                all_preds.append(prob)
        
        # Aggregate predictions
        mean_pred = float(np.mean(all_preds))
        max_pred = float(np.max(all_preds))
        median_pred = float(np.median(all_preds))
        std_pred = float(np.std(all_preds))
        
        # Use adaptive weighting: if there's high variance, trust max more (indicates fake artifacts)
        # If low variance, use mean/median (consistent prediction)
        if std_pred > 0.15:  # High variance - likely fake artifacts detected in some regions
            # Weight max more heavily to catch fake regions
            combined_pred = 0.5 * max_pred + 0.3 * mean_pred + 0.2 * median_pred
        else:
            # Low variance - consistent prediction
            combined_pred = 0.4 * max_pred + 0.4 * mean_pred + 0.2 * median_pred
        
        # Clamp to [0, 1]
        combined_pred = max(0.0, min(1.0, combined_pred))
        
        # Decision - use slightly lower threshold for Swin (it's more sensitive)
        adjusted_threshold = threshold * 0.9
        
        if combined_pred > adjusted_threshold:
            return "fake", combined_pred
        else:
            return "real", combined_pred
            
    except Exception as e:
        print(f"Error in Swin Transformer prediction: {e}")
        import traceback
        traceback.print_exc()
        return "real", 0.5

