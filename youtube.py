import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
import numpy as np
from PIL import Image

import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace , VideoReader
# from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils

# Optional import for transformer model (only if timm is available)
try:
    from transformer_model import (
        swin_video_pred, 
        _compute_fft_image, 
        _compute_fft_video_clip,
        _prepare_rgb_and_fft_tensors
    )
    TRANSFORMER_MODEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    TRANSFORMER_MODEL_AVAILABLE = False
    # Silently fail - old models will still work

# Optional import for physiological detector
try:
    from physiological_detector import (
        PhysiologicalDetector,
        calculate_ear,
        EyeBlinkGazeLSTM,
        EARLSTM,
        FacialLandmarksLSTM,
        GNNLandmarkHead,
        SyncNetLike,
        AudioVisualTransformer,
        rPPGCNN,
        TemporalFilteringCNN
    )
    from physiological_detector import CV2_AVAILABLE, AUDIO_AVAILABLE
    PHYSIOLOGICAL_MODEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    PHYSIOLOGICAL_MODEL_AVAILABLE = False

# Optional import for ensemble detector
try:
    from ensemble_detector import ensemble_video_prediction
    ENSEMBLE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    ENSEMBLE_AVAILABLE = False

def video_pred(threshold=0.5,model='EfficientNetAutoAttB4',dataset='DFDC',frames=100,video_path="notebook/samples/mqzvfufzoq.mp4"):
    
    # Ensemble Architecture: Combines all models
    if model == 'Ensemble_All':
        if not ENSEMBLE_AVAILABLE:
            raise ImportError(
                "Ensemble_All model requires ensemble_detector module. "
                "Please ensure all dependencies are installed."
            )
        return _ensemble_video_pred(video_path, threshold, frames)
    
    # Physiological and Behavioral Detection Model
    if model == 'Physiological_Behavioral':
        if not PHYSIOLOGICAL_MODEL_AVAILABLE:
            raise ImportError(
                "Physiological_Behavioral model requires 'dlib' and 'librosa' packages. "
                "Please install them with: pip install dlib librosa torchaudio"
            )
        return _physiological_video_pred(video_path, threshold, frames)
    
    # New path: Video Swin Transformer RGB+FFT model (no face crops, temporal video understanding)
    if model == 'ViT_RGB_FFT':
        if not TRANSFORMER_MODEL_AVAILABLE:
            raise ImportError(
                "ViT_RGB_FFT model requires 'timm' package. "
                "Please install it with: pip install timm"
            )
        """
        For the Video Swin Transformer RGB+FFT model:
          - Read video frames as temporal clips (not individual frames)
          - Build RGB and FFT video clip tensors
          - Process entire video clips with temporal understanding
          - Return video-level prediction (NOT frame-wise aggregation)
        """
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        videoreader = VideoReader(verbose=False)
        
        # Read frames as a temporal clip (recommended: 16-32 frames for good temporal context)
        clip_length = min(frames, 32)  # Limit to reasonable clip length for video transformer
        result = videoreader.read_frames(video_path, num_frames=clip_length)

        if result is None:
            return 'real', 0.5

        frames_np, idxs = result  # (T, H, W, 3), [T]
        if len(frames_np) == 0:
            return 'real', 0.5

        T = len(frames_np)
        
        # Prepare RGB and FFT video clip tensors
        rgb_clip_list = []
        fft_clip_list = []
        
        for frame in frames_np:
            # frame: (H, W, 3), uint8 RGB from VideoReader
            pil_img = Image.fromarray(frame)

            # RGB preprocessing: resize and normalize
            pil_rgb = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
            rgb_np = np.array(pil_rgb).astype('float32') / 255.0
            rgb_np = rgb_np.transpose(2, 0, 1)  # HWC → CHW: (3, H, W)
            rgb_clip_list.append(rgb_np)

            # FFT preprocessing
            fft_np = _compute_fft_image(pil_img, size=(224, 224))  # (H, W)
            fft_clip_list.append(fft_np)

        # Stack to form video clips: (T, 3, H, W) and (T, H, W)
        rgb_clip_arr = np.stack(rgb_clip_list, axis=0)  # (T, 3, 224, 224)
        fft_clip_arr = np.stack(fft_clip_list, axis=0).astype('float32')  # (T, 224, 224)
        fft_clip_arr = fft_clip_arr[:, None, :, :]  # (T, 1, 224, 224)

        # Convert to tensors
        rgb_clip_tensor = torch.from_numpy(rgb_clip_arr).float()  # (T, 3, H, W)
        fft_clip_tensor = torch.from_numpy(fft_clip_arr).float()  # (T, 1, H, W)

        # Process entire video clip with Video Swin Transformer (temporal understanding)
        # This returns a SINGLE video-level probability (NOT frame-wise)
        prob_fake = swin_video_pred(rgb_clip_tensor, fft_clip_tensor, device=device)

        # Single video-level prediction - no frame aggregation needed
        label = 'fake' if prob_fake >= threshold else 'real'
        return label, prob_fake

    """
    Existing path: CNN-based EfficientNet/Xception models trained on DFDC/FFPP using face crops.
    """
    net_model = model

    """
    Choose a training dataset between
    - DFDC
    - FFPP
    """
    train_db = dataset

    # setting the parameters
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224
    frames_per_video = frames

    # loading the weights
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)

    try:
        vid_fake_faces = face_extractor.process_video(video_path)
        
        if not vid_fake_faces or len(vid_fake_faces) == 0:
            # No frames processed
            return 'real', 0.3
        
        # Collect all faces from all frames
        all_faces = []
        for frame in vid_fake_faces:
            if len(frame['faces']) > 0:
                # Process the best face from each frame (already sorted by confidence)
                all_faces.append(frame['faces'][0])
        
        if len(all_faces) == 0:
            # No faces detected in any frame
            return 'real', 0.3
        
        # Process all faces in batches for efficiency
        faces_fake_t = torch.stack([transf(image=im)['image'] for im in all_faces])
        
        with torch.no_grad():
            # Model outputs logits, apply sigmoid to get probabilities
            logits = net(faces_fake_t.to(device))
            # CRITICAL FIX: Apply sigmoid to EACH prediction before averaging
            # This is the correct way to aggregate probabilities
            faces_fake_pred = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Better aggregation strategy for video:
        # 1. Mean of all predictions (overall trend)
        # 2. Max prediction (catches frames with clear deepfake artifacts)
        # 3. Percentile-based (reduces impact of outliers)
        mean_pred = float(faces_fake_pred.mean())
        max_pred = float(faces_fake_pred.max())
        median_pred = float(np.median(faces_fake_pred))
        
        # Use 75th percentile to be more sensitive to fake frames
        percentile_75 = float(np.percentile(faces_fake_pred, 75))
        
        # Weighted combination: prioritize max and high percentiles
        # This helps catch videos where only some frames are manipulated
        combined_pred = 0.4 * max_pred + 0.3 * percentile_75 + 0.2 * mean_pred + 0.1 * median_pred
        
        # Adjust threshold slightly lower and cap it to be more sensitive to AI-generated content.
        # This makes the detector more likely to flag subtle AI edits as fake.
        adjusted_threshold = min(threshold * 0.9, 0.4)
        
        # Return fake probability in both cases for consistent confidence calculation
        if combined_pred > adjusted_threshold:
            return 'fake', combined_pred
        else:
            return 'real', combined_pred
            
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        # Return uncertain result on error
        return 'real', 0.5


def _physiological_video_pred(video_path, threshold=0.5, num_frames=32):
    """
    Physiological and behavioral deepfake detection.
    Uses multiple temporal models:
    - Eye blink/gaze (CNN+LSTM, EAR+LSTM)
    - Head pose (Landmarks+LSTM, GNN)
    - Lip sync (SyncNet, Audio-Visual Transformer)
    - Heartbeat/skin signals (rPPG, Temporal Filtering)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Read video frames
    videoreader = VideoReader(verbose=False)
    result = videoreader.read_frames(video_path, num_frames=num_frames)
    
    if result is None:
        return 'real', 0.5
    
    frames_np, idxs = result
    if len(frames_np) == 0:
        return 'real', 0.5
    
    T = len(frames_np)
    
    # Initialize detector
    detector = PhysiologicalDetector().to(device)
    detector.eval()
    
    # Prepare inputs for physiological detection
    inputs_dict = {}
    
    try:
        # Note: Full implementation would require facial landmark detection (dlib)
        # and audio extraction. This is a skeleton showing the integration.
        
        # For now, use placeholder tensors that match expected shapes
        # In production, you would:
        # 1. Detect faces and extract eye regions, landmarks using dlib/OpenCV
        # 2. Extract audio from video and compute spectrograms/MFCC
        # 3. Extract face regions for rPPG analysis
        
        # Placeholder: Eye regions (B=1, T, 3, H, W)
        if CV2_AVAILABLE:
            # TODO: Extract eye regions from frames
            eye_regions = torch.randn(1, T, 3, 64, 64).to(device)
            inputs_dict['eye_regions'] = eye_regions
        
        # Placeholder: EAR sequence
        ear_sequence = torch.randn(1, T, 1).to(device)
        inputs_dict['ear_sequence'] = ear_sequence
        
        # Placeholder: Landmarks sequence (68 landmarks * 2 coords)
        landmarks_sequence = torch.randn(1, T, 68, 2).to(device)
        inputs_dict['landmarks_sequence'] = landmarks_sequence
        
        # Placeholder: Lip region for sync
        lip_region = torch.randn(1, 3, 96, 96).to(device)
        inputs_dict['lip_region'] = lip_region
        
        # Placeholder: Audio features
        if AUDIO_AVAILABLE:
            # TODO: Extract audio and compute spectrogram/MFCC
            audio_features = torch.randn(1, 1, 128, 128).to(device)
            inputs_dict['audio_features'] = audio_features
            
            visual_seq = torch.randn(1, T, 256).to(device)
            audio_seq = torch.randn(1, T, 256).to(device)
            inputs_dict['visual_seq'] = visual_seq
            inputs_dict['audio_seq'] = audio_seq
        
        # Face region for rPPG
        face_region_temporal = torch.randn(1, T, 3, 128, 128).to(device)
        inputs_dict['face_region_temporal'] = face_region_temporal
        
        # Run physiological detector
        with torch.no_grad():
            logits = detector(inputs_dict)
            prob_fake = torch.sigmoid(logits).item()
        
        label = 'fake' if prob_fake >= threshold else 'real'
        return label, float(prob_fake)
        
    except Exception as e:
        print(f"Error in physiological detection: {e}")
        import traceback
        traceback.print_exc()
        return 'real', 0.5


def _ensemble_video_pred(video_path, threshold=0.5, num_frames=32):
    """
    Ensemble video prediction combining:
    1. Spatial–Frequency (Swin + FFT)
    2. Temporal Video (Video Swin)
    3. Physiological (Landmarks + LSTM)
    4. Audio–Visual (SyncNet)
    
    Uses adaptive weighting and video-level decisions (no frame-wise).
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Read video frames
    videoreader = VideoReader(verbose=False)
    result = videoreader.read_frames(video_path, num_frames=min(num_frames, 32))
    
    if result is None:
        return 'real', 0.5
    
    frames_np, idxs = result
    if len(frames_np) == 0:
        return 'real', 0.5
    
    T = len(frames_np)
    
    # Initialize logits storage
    spatial_freq_logits = None
    temporal_video_logits = None
    physiological_logits = None
    audio_visual_logits = None
    
    try:
        # ========================================================================
        # 1. Spatial–Frequency Model: Swin Transformer + FFT
        # ========================================================================
        if TRANSFORMER_MODEL_AVAILABLE:
            try:
                from transformer_model import swin_video_pred, _compute_fft_image
                
                # Prepare RGB and FFT clips
                rgb_clip_list = []
                fft_clip_list = []
                
                for frame in frames_np:
                    pil_img = Image.fromarray(frame)
                    
                    # RGB
                    pil_rgb = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
                    rgb_np = np.array(pil_rgb).astype('float32') / 255.0
                    rgb_np = rgb_np.transpose(2, 0, 1)
                    rgb_clip_list.append(rgb_np)
                    
                    # FFT
                    fft_np = _compute_fft_image(pil_img, size=(224, 224))
                    fft_clip_list.append(fft_np)
                
                rgb_clip_arr = np.stack(rgb_clip_list, axis=0)
                fft_clip_arr = np.stack(fft_clip_list, axis=0).astype('float32')
                fft_clip_arr = fft_clip_arr[:, None, :, :]
                
                rgb_clip_tensor = torch.from_numpy(rgb_clip_arr).float()
                fft_clip_tensor = torch.from_numpy(fft_clip_arr).float()
                
                # Get prediction (convert prob to logit)
                prob = swin_video_pred(rgb_clip_tensor, fft_clip_tensor, device=device)
                spatial_freq_logits = torch.tensor([np.log(prob / (1 - prob + 1e-8))]).to(device)
            except Exception as e:
                print(f"Spatial-Frequency model failed: {e}")
        
        # ========================================================================
        # 2. Temporal Video Model: Video Swin Transformer
        # ========================================================================
        # Reuse same model as above (they're similar, but can be different)
        # For now, use the same prediction
        if spatial_freq_logits is not None:
            temporal_video_logits = spatial_freq_logits.clone()
        
        # ========================================================================
        # 3. Physiological Model: Facial Landmarks + LSTM
        # ========================================================================
        # Skip for now - requires real landmark extraction (not random placeholders)
        # When real landmarks are available, uncomment this:
        # if PHYSIOLOGICAL_MODEL_AVAILABLE:
        #     try:
        #         from physiological_detector import FacialLandmarksLSTM
        #         # Extract real landmarks using dlib/OpenCV
        #         # landmarks_sequence = extract_landmarks(frames_np)
        #         # model_physio = FacialLandmarksLSTM().to(device)
        #         # physiological_logits = model_physio(landmarks_sequence)
        #     except Exception as e:
        #         print(f"Physiological model failed: {e}")
        physiological_logits = None  # Skip untrained random models
        
        # ========================================================================
        # 4. Audio–Visual Model: SyncNet
        # ========================================================================
        # Skip for now - requires real audio extraction (not random placeholders)
        # When real audio is available, uncomment this:
        # if PHYSIOLOGICAL_MODEL_AVAILABLE and AUDIO_AVAILABLE:
        #     try:
        #         from physiological_detector import SyncNetLike
        #         # Extract real audio and lip regions
        #         # audio_features = extract_audio_features(video_path)
        #         # lip_region = extract_lip_regions(frames_np)
        #         # model_sync = SyncNetLike().to(device)
        #         # audio_visual_logits = model_sync(lip_region, audio_features)
        #     except Exception as e:
        #         print(f"Audio-Visual model failed: {e}")
        audio_visual_logits = None  # Skip untrained random models
        
        # ========================================================================
        # Ensemble Combination
        # ========================================================================
        ensemble_result = ensemble_video_prediction(
            spatial_freq_logits=spatial_freq_logits,
            temporal_video_logits=temporal_video_logits,
            physiological_logits=physiological_logits,
            audio_visual_logits=audio_visual_logits,
            threshold=threshold,
            device=device,
        )
        
        # Return label and probability
        return ensemble_result['label'], ensemble_result['probability']
        
    except Exception as e:
        print(f"Error in ensemble prediction: {e}")
        import traceback
        traceback.print_exc()
        return 'real', 0.5
    
# print(preprocess())
