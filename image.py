import torch
from torch.utils.model_zoo import load_url
from PIL import Image
from scipy.special import expit
import numpy as np
import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace
from architectures import fornet,weights
from isplutils import utils
from functools import lru_cache

# Optional import for Swin Transformer (RGB + FFT dual-branch)
try:
    from swin_rgb_fft_detector import swin_rgb_fft_image_pred
    SWIN_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    SWIN_AVAILABLE = False


@lru_cache(maxsize=None)
def _load_image_model(net_model: str, train_db: str, device_str: str):
    """
    Cache-heavy model components so we don't re-download weights on every call.
    """
    device = torch.device(device_str)
    face_policy = 'scale'
    face_size = 224

    try:
        model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
        net = getattr(fornet,net_model)().eval().to(device)
        # Load state dict with strict=False for ST models that might have slightly different keys
        state_dict = load_url(model_url,map_location=device,check_hash=True)
        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try with strict=False (for ST models)
            print(f"Warning: Strict loading failed, trying lenient loading: {e}")
            net.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {net_model} for dataset {train_db}: {str(e)}")

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    
    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)
    return net, transf, face_extractor, device


def image_pred(threshold=0.5,model='EfficientNetAutoAttB4',dataset='DFDC',image_path="notebook/samples/lynaeydofd_fr0.jpg"):
    
    # Existing path: EfficientNet/Xception models with DFDC/FFPP weights and face crops.
    """
    Choose an architecture between
    - EfficientNetB4
    - EfficientNetB4ST
    - EfficientNetAutoAttB4
    - EfficientNetAutoAttB4ST
    - Xception
    - SwinTransformer_RGB (full-image RGB, best for AI edits)
    """
    net_model = model

    """
    Choose a training dataset between
    - DFDC
    - FFPP
    """
    train_db = dataset

    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Special handling for memory-intensive models
    if net_model == 'EfficientNetAutoAttB4ST':
        # Clear cache before loading this heavy model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        net, transf, face_extractor, device = _load_image_model(net_model, train_db, device_str)
    except Exception as e:
        # Handle model loading errors gracefully
        error_msg = str(e)
        if "EfficientNetAutoAttB4ST" in error_msg or "AutoAttB4ST" in error_msg:
            return "real", 0.5
        raise
    
    try:
        im_real = Image.open(image_path).convert("RGB")
        im_real_faces = face_extractor.process_image(img=im_real)
        
        # Check if faces were detected
        if len(im_real_faces['faces']) == 0:
            # No faces detected - return uncertain result
            return "real", 0.3  # Low confidence for no-face case
        
        # Process all detected faces, not just the first one
        # This helps detect deepfakes better as we can check multiple faces
        all_faces = im_real_faces['faces']
        
        # Limit to top 3 faces by confidence to avoid processing too many
        num_faces_to_process = min(len(all_faces), 3)
        faces_to_process = all_faces[:num_faces_to_process]
        
        faces_t = torch.stack([transf(image=im)['image'] for im in faces_to_process])

        with torch.no_grad():
            # Model outputs logits, apply sigmoid to get probabilities
            logits = net(faces_t.to(device))
            # Apply sigmoid to convert logits to probabilities
            # Higher values mean more fake
            faces_pred = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Use mean of all face predictions for better accuracy
        # Also consider max prediction as deepfakes might affect some faces more
        mean_pred = float(faces_pred.mean())
        max_pred = float(faces_pred.max())
        median_pred = float(np.median(faces_pred))
        
        # Balanced approach: Use max when it's high (catch fakes), use median when max is low (reduce false positives)
        # For ST models, use balanced weighting
        if 'ST' in net_model:
            # ST models: If max is high (>0.6), give it more weight (fake detection)
            # Otherwise use balanced weighting (real detection)
            if max_pred > 0.6:
                combined_pred = 0.4 * max_pred + 0.35 * mean_pred + 0.25 * median_pred
            else:
                combined_pred = 0.35 * max_pred + 0.35 * mean_pred + 0.3 * median_pred
            adjusted_threshold = threshold
        else:
            # Regular models: More aggressive fake detection
            # If max is high (>0.6), strongly weight it (fake artifacts present)
            if max_pred > 0.6:
                combined_pred = 0.6 * max_pred + 0.3 * mean_pred + 0.1 * median_pred
            elif max_pred > 0.5:
                combined_pred = 0.5 * max_pred + 0.35 * mean_pred + 0.15 * median_pred
            else:
                combined_pred = 0.4 * max_pred + 0.4 * mean_pred + 0.2 * median_pred
            # Lower threshold to catch more AI edits
            adjusted_threshold = max(threshold * 0.9, 0.35)
        
        # Clamp combined_pred to [0, 1] to prevent > 100% confidence
        combined_pred = max(0.0, min(1.0, float(combined_pred)))
        
        # Enhance with Swin Transformer RGB+FFT dual-branch analysis (full-image context)
        # This is CRITICAL for catching AI edits that EfficientNet misses
        # Swin RGB analyzes spatial patterns, Swin FFT analyzes frequency domain artifacts
        swin_pred = None
        if SWIN_AVAILABLE:
            try:
                # Get Swin RGB+FFT prediction (full-image, no face crops)
                swin_label, swin_prob = swin_rgb_fft_image_pred(image_path=image_path, threshold=threshold)
                swin_pred = swin_prob
                
                # AGGRESSIVE combination: Prioritize Swin RGB+FFT for AI edits
                # Swin RGB+FFT is MUCH better at catching AI-modified images
                # Give it more weight when it detects fake, less when it says real
                if swin_prob > threshold:
                    # Swin detected fake - trust it more heavily (75% weight)
                    enhanced_pred = 0.25 * combined_pred + 0.75 * swin_pred
                else:
                    # Swin says real, but still use it (50% weight)
                    enhanced_pred = 0.5 * combined_pred + 0.5 * swin_pred
                
                # Use enhanced prediction
                combined_pred = enhanced_pred
                combined_pred = max(0.0, min(1.0, float(combined_pred)))
            except Exception as e:
                # If Swin fails, continue with EfficientNet only
                print(f"Swin Transformer RGB+FFT enhancement failed (using EfficientNet only): {e}")
        
        # Return fake probability in both cases for consistent confidence calculation
        if combined_pred > adjusted_threshold:
            return "fake", combined_pred
        else:
            return "real", combined_pred
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        # Return uncertain result on error
        return "real", 0.5
    
# print(image_pred(image_path='C:/Users/snehs/OneDrive/Desktop/icpr2020dfdc/notebook/samples/lynaeydofd_fr0.jpg'))
    
