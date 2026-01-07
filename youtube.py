import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
import numpy as np

import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace , VideoReader
# from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils
from transformer_model import vit_fft_video_frame_preds, _compute_fft_image

def video_pred(threshold=0.5,model='EfficientNetAutoAttB4',dataset='DFDC',frames=100,video_path="notebook/samples/mqzvfufzoq.mp4"):
    
    # New path: full-frame RGB + FFT ViT model (no face crops, no DFDC/FFPP assumptions)
    if model == 'ViT_RGB_FFT':
        """
        For the transformer-based RGB+FFT model, we:
          - read multiple frames from the video using VideoReader,
          - build RGB and FFT tensors for each frame,
          - run the ViT model on all frames,
          - aggregate sigmoid probabilities over frames,
          - return the video-level probability of being AI-generated.
        """
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        videoreader = VideoReader(verbose=False)
        result = videoreader.read_frames(video_path, num_frames=frames)

        if result is None:
            return 'real', 0.5

        frames_np, idxs = result  # (N, H, W, 3), [N]
        if len(frames_np) == 0:
            return 'real', 0.5

        # Prepare RGB and FFT tensors for all frames
        rgb_list = []
        fft_list = []
        for frame in frames_np:
            # frame: (H, W, 3), uint8 BGR converted to RGB in VideoReader
            pil_img = Image.fromarray(frame)

            # Reuse the same FFT computation as for images
            fft_np = _compute_fft_image(pil_img, size=(224, 224))  # (H, W)
            fft_np = fft_np.astype('float32')

            # RGB preprocessing
            pil_rgb = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
            rgb_np = np.array(pil_rgb).astype('float32') / 255.0
            rgb_np = rgb_np.transpose(2, 0, 1)  # HWC → CHW

            rgb_list.append(rgb_np)
            fft_list.append(fft_np)

        rgb_arr = np.stack(rgb_list, axis=0)  # (N, 3, H, W)
        fft_arr = np.stack(fft_list, axis=0)  # (N, H, W)
        fft_arr = fft_arr[:, None, :, :]      # (N, 1, H, W)

        frames_rgb = torch.from_numpy(rgb_arr)
        frames_fft = torch.from_numpy(fft_arr)

        probs_fake = vit_fft_video_frame_preds(frames_rgb, frames_fft, device=device)  # (N,)

        # Aggregate over frames without hardcoding threshold; just compute a robust statistic.
        mean_pred = float(probs_fake.mean())
        max_pred = float(probs_fake.max())
        perc75 = float(np.percentile(probs_fake.numpy(), 75))

        combined_pred = 0.4 * max_pred + 0.3 * perc75 + 0.3 * mean_pred

        label = 'fake' if combined_pred >= threshold else 'real'
        return label, combined_pred

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
    
# print(preprocess())
