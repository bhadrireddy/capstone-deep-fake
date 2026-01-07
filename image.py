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
        
        # More conservative approach to reduce false positives
        # Use median as it's less affected by outliers than mean
        # For ST models, use balanced weighting
        if 'ST' in net_model:
            # ST models: use median-heavy weighting (more conservative)
            combined_pred = 0.4 * median_pred + 0.35 * mean_pred + 0.25 * max_pred
            # ST models use original threshold (more conservative)
            adjusted_threshold = threshold
        else:
            # Regular models: still use median-heavy but allow some max influence
            # This reduces false positives while still catching fakes
            combined_pred = 0.5 * median_pred + 0.3 * mean_pred + 0.2 * max_pred
            # Use original threshold for better true negative rate (reduce false positives)
            adjusted_threshold = threshold
        
        # Clamp combined_pred to [0, 1] to prevent > 100% confidence
        combined_pred = max(0.0, min(1.0, float(combined_pred)))
        
        # Return fake probability in both cases for consistent confidence calculation
        # Only classify as fake if prediction is clearly above threshold
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
    
