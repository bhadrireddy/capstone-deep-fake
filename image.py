import torch
from torch.utils.model_zoo import load_url
from PIL import Image
from scipy.special import expit
import sys
import streamlit as st
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace
from architectures import fornet,weights
from isplutils import utils

def image_pred(threshold=0.5,model='EfficientNetAutoAttB4',dataset='DFDC',image_path="notebook/samples/lynaeydofd_fr0.jpg"):
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

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224

    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    
    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    
    facedet.load_anchors("blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)
    
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
        
        # Weighted combination: give more weight to max prediction
        # This helps catch cases where only some faces are fake
        combined_pred = 0.6 * max_pred + 0.4 * mean_pred
        
        # Adjust threshold slightly lower to be more sensitive to AI-generated content
        adjusted_threshold = threshold * 0.9
        
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
    
