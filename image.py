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

# Optional import for Transformer models (RGB + FFT dual-branch)
try:
    from transformer_models import vit_fft_image_pred, _prepare_rgb_and_fft_tensors, RGBFFT_ViT
    TRANSFORMER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    TRANSFORMER_AVAILABLE = False


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
            return "Suspicious", 0.5  # Uncertain result on error
        raise
    
    try:
        im_real = Image.open(image_path).convert("RGB")
        im_real_faces = face_extractor.process_image(img=im_real)
        
        # Check if faces were detected
        if len(im_real_faces['faces']) == 0:
            # No faces detected - use full-image transformer only if available
            if TRANSFORMER_AVAILABLE:
                try:
                    transformer_label, transformer_prob = vit_fft_image_pred(
                        image_path=image_path, 
                        threshold=threshold
                    )
                    # Use transformer prediction directly for no-face case
                    combined_pred = transformer_prob
                    # Apply three-class logic
                    fake_threshold = 0.65
                    suspicious_lower = 0.35
                    if combined_pred > fake_threshold:
                        return "Fake", combined_pred
                    elif combined_pred >= suspicious_lower:
                        return "Suspicious", combined_pred
                    else:
                        # Lower confidence for real when no faces detected
                        return "Likely Real", combined_pred * 0.7
                except Exception as e:
                    print(f"Transformer prediction failed for no-face case: {e}")
                    # Fall through to uncertain result
            # No faces and no transformer - return uncertain/suspicious
            return "Suspicious", 0.5  # Uncertain result for no-face case
        
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
        face_pred = max(0.0, min(1.0, float(combined_pred)))
        
        # Enhance with Transformer RGB+FFT dual-branch analysis (full-image context)
        # This is CRITICAL for catching AI edits that EfficientNet might miss
        # ViT RGB analyzes spatial patterns, FFT analyzes frequency domain artifacts
        # Transformer is the PRIMARY decision maker for full-image analysis
        transformer_pred = None
        if TRANSFORMER_AVAILABLE:
            try:
                # Get ViT RGB+FFT prediction (full-image, no face crops)
                transformer_label, transformer_prob = vit_fft_image_pred(
                    image_path=image_path, 
                    threshold=threshold
                )
                transformer_pred = transformer_prob
            except Exception as e:
                # If Transformer fails, continue with EfficientNet only
                print(f"Transformer RGB+FFT enhancement failed (using EfficientNet only): {e}")
        
        # AGGRESSIVE COMBINATION: If EITHER face model OR transformer is suspicious, increase fake probability
        # This prevents averaging that would make AI images appear real
        if transformer_pred is not None:
            # Both models available - use OR logic: if either is suspicious, boost fake probability
            face_suspicious = face_pred > threshold
            transformer_suspicious = transformer_pred > threshold
            
            if face_suspicious or transformer_suspicious:
                # If EITHER model is suspicious, prioritize the higher prediction
                # Use weighted max: take the higher prediction and boost it
                max_pred = max(face_pred, transformer_pred)
                # Boost by taking 60% of max and 40% of average (but only when suspicious)
                if max_pred > 0.6:
                    # Very suspicious - heavily weight the max
                    combined_pred = 0.7 * max_pred + 0.2 * ((face_pred + transformer_pred) / 2) + 0.1 * min(face_pred, transformer_pred)
                else:
                    # Moderately suspicious - balanced boost
                    combined_pred = 0.5 * max_pred + 0.3 * ((face_pred + transformer_pred) / 2) + 0.2 * min(face_pred, transformer_pred)
            else:
                # Both say real, but use transformer more (it's better at full-image analysis)
                combined_pred = 0.4 * face_pred + 0.6 * transformer_pred
        else:
            # Only face model available
            combined_pred = face_pred
        
        # Clamp combined_pred to [0, 1]
        combined_pred = max(0.0, min(1.0, float(combined_pred)))
        
        # THREE-CLASS OUTPUT: Fake, Suspicious, Likely Real
        # Lower confidence for 'real' predictions when model is unsure
        # Thresholds:
        # - Fake: > 0.65 (high confidence fake)
        # - Suspicious: 0.35 - 0.65 (uncertain, could be fake)
        # - Likely Real: < 0.35 (but with lower confidence if close to suspicious threshold)
        
        fake_threshold = 0.65
        suspicious_lower = 0.35
        
        if combined_pred > fake_threshold:
            # FAKE - High confidence
            return "Fake", combined_pred
        elif combined_pred >= suspicious_lower:
            # SUSPICIOUS - Uncertain, could be fake or real
            return "Suspicious", combined_pred
        else:
            # LIKELY REAL - But with lower confidence if close to suspicious threshold
            # The closer to suspicious threshold, the lower the confidence for "real"
            # This prevents confidently labeling AI images as real
            if combined_pred > 0.25:
                # Close to suspicious - very low confidence
                confidence_penalty = (combined_pred - 0.25) / 0.1  # Penalty factor [0, 1]
                adjusted_confidence = combined_pred * (1 - 0.5 * confidence_penalty)
                return "Likely Real", max(0.05, adjusted_confidence)  # Min 5% confidence
            else:
                # Far from suspicious - moderate confidence, but still lower than fake
                return "Likely Real", combined_pred * 0.8  # Scale down confidence for real
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        # Return uncertain result on error
        return "Suspicious", 0.5
    
# print(image_pred(image_path='C:/Users/snehs/OneDrive/Desktop/icpr2020dfdc/notebook/samples/lynaeydofd_fr0.jpg'))
    
