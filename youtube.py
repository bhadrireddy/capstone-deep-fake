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
from functools import lru_cache

@lru_cache(maxsize=None)
def _load_video_model(net_model: str, train_db: str, device_str: str):
    """
    Cache model and transformer to avoid re-loading weights on every video.
    """
    device = torch.device(device_str)
    face_policy = 'scale'
    face_size = 224

    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    return net, transf, device


def video_pred(threshold=0.5,model='EfficientNetAutoAttB4',dataset='DFDC',frames=100,video_path="notebook/samples/mqzvfufzoq.mp4"):
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

    frames_per_video = frames
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net, transf, device = _load_video_model(net_model, train_db, device_str)

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
        cnn_prob = 0.4 * max_pred + 0.3 * percentile_75 + 0.2 * mean_pred + 0.1 * median_pred
        
        # Physiological heuristics from detections (no heavy deps)
        physio_prob = _physio_heuristics_from_detections(vid_fake_faces)
        
        # Fuse CNN probability with physiological score
        combined_pred = 0.85 * cnn_prob + 0.15 * physio_prob
        
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
    
def _physio_heuristics_from_detections(vid_frames) -> float:
    """
    Lightweight physiological/behavioral proxy using only BlazeFace detections.
    Produces a score in [0,1] where higher = more likely fake.
    Heuristics:
      - Detection dropout ratio
      - Bounding box center/area jerk (second derivative magnitude)
      - Keypoint temporal instability (using 6 kpts from detections columns 4:16)
    """
    centers = []
    areas = []
    kpt_series = []
    num_frames = len(vid_frames)
    noface_count = 0
    
    for frame in vid_frames:
        dets = frame.get('detections')
        if dets is None or len(dets) == 0:
            noface_count += 1
            continue
        det = dets[0]  # best face
        ymin, xmin, ymax, xmax = det[:4]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        area = max(1.0, (xmax - xmin) * (ymax - ymin))
        centers.append([cx, cy])
        areas.append(area)
        # 6 keypoints (x,y) in columns 4..15
        kpts = det[4:16].reshape(6, 2)
        kpt_series.append(kpts)
    
    if num_frames == 0:
        return 0.5
    
    dropout_ratio = noface_count / max(1, num_frames)
    
    def jerk(seq):
        if len(seq) < 3:
            return 0.0
        seq = np.asarray(seq, dtype=np.float32)
        d1 = np.diff(seq, axis=0)
        d2 = np.diff(d1, axis=0)
        return float(np.mean(np.linalg.norm(d2, axis=-1)))
    
    # Normalize centers by sqrt(area) to be scale-invariant
    if len(centers) >= 3:
        centers = np.asarray(centers, dtype=np.float32)
        areas = np.asarray(areas, dtype=np.float32)
        scale = np.sqrt(np.maximum(areas, 1.0))[:, None]
        norm_centers = centers / scale
        center_jerk = jerk(norm_centers)
        # Normalize jerk into [0,1]
        center_score = float(np.tanh(center_jerk / 5.0))
    else:
        center_score = 0.0
    
    # Keypoint instability
    if len(kpt_series) >= 3:
        kpt_arr = np.stack(kpt_series, axis=0)  # (T, 6, 2)
        # Normalize by bbox scale approximately using median area
        med_scale = np.sqrt(np.median(areas)) if len(areas) > 0 else 1.0
        kpt_arr = kpt_arr / max(1.0, med_scale)
        kpt_jerk = jerk(kpt_arr.reshape(len(kpt_series), -1))
        kpt_score = float(np.tanh(kpt_jerk / 5.0))
    else:
        kpt_score = 0.0
    
    # Combine heuristics
    physio_score = 0.5 * dropout_ratio + 0.25 * center_score + 0.25 * kpt_score
    return float(np.clip(physio_score, 0.0, 1.0))

# print(preprocess())
