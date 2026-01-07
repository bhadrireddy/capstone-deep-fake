"""
Physiological and Behavioral Deepfake Detection Models

This module implements multiple temporal models for detecting deepfakes based on:
1. Eye blink and gaze patterns (CNN+LSTM, EAR+LSTM)
2. Head pose and micro-movements (Facial Landmarks+LSTM, GNN)
3. Lip sync (SyncNet, Audio-Visual Transformer)
4. Heartbeat/skin signals (rPPG-based CNN, Temporal Filtering)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from collections import deque

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None

try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    librosa = None


# ============================================================================
# 1. Eye Blink & Gaze Detection
# ============================================================================

class EyeRegionCNN(nn.Module):
    """CNN for extracting features from eye region."""
    
    def __init__(self, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, embed_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class EyeBlinkGazeLSTM(nn.Module):
    """CNN + LSTM for eye blink and gaze detection."""
    
    def __init__(self, lstm_hidden=128, num_layers=2):
        super().__init__()
        self.eye_cnn = EyeRegionCNN(embed_dim=lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, eye_regions):
        """
        eye_regions: (B, T, 3, H, W) - batch of temporal eye region sequences
        Returns: (B, 1) logits
        """
        B, T, C, H, W = eye_regions.shape
        eye_regions = eye_regions.view(B * T, C, H, W)
        eye_features = self.eye_cnn(eye_regions)  # (B*T, hidden)
        eye_features = eye_features.view(B, T, -1)  # (B, T, hidden)
        
        lstm_out, _ = self.lstm(eye_features)  # (B, T, hidden*2)
        # Use last timestep output
        final_output = lstm_out[:, -1, :]  # (B, hidden*2)
        return self.classifier(final_output).squeeze(1)


def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR).
    landmarks: facial landmark points
    eye_indices: indices for left/right eye landmarks
    """
    if landmarks is None or len(landmarks) < max(eye_indices):
        return None
    
    eye_points = landmarks[eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    
    if h == 0:
        return None
    ear = (v1 + v2) / (2.0 * h)
    return ear


class EARLSTM(nn.Module):
    """EAR (Eye Aspect Ratio) + LSTM for blink detection."""
    
    def __init__(self, lstm_hidden=64, num_layers=2):
        super().__init__()
        # Input: EAR value (scalar) per frame
        self.lstm = nn.LSTM(
            input_size=1,  # Single EAR value
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, ear_sequence):
        """
        ear_sequence: (B, T, 1) - EAR values over time
        Returns: (B, 1) logits
        """
        lstm_out, _ = self.lstm(ear_sequence)  # (B, T, hidden*2)
        final_output = lstm_out[:, -1, :]  # (B, hidden*2)
        return self.classifier(final_output).squeeze(1)


# ============================================================================
# 2. Head Pose & Micro-Movements
# ============================================================================

class FacialLandmarksLSTM(nn.Module):
    """Facial Landmarks + LSTM for head pose and micro-movements."""
    
    def __init__(self, num_landmarks=68, landmark_dim=2, lstm_hidden=128, num_layers=2):
        super().__init__()
        # Project landmarks to feature space
        self.landmark_encoder = nn.Sequential(
            nn.Linear(num_landmarks * landmark_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, landmarks_sequence):
        """
        landmarks_sequence: (B, T, num_landmarks, 2) or (B, T, num_landmarks*2)
        Returns: (B, 1) logits
        """
        B, T = landmarks_sequence.shape[:2]
        if landmarks_sequence.dim() == 4:
            landmarks_sequence = landmarks_sequence.view(B, T, -1)
        
        # Encode landmarks
        encoded = self.landmark_encoder(landmarks_sequence)  # (B, T, 64)
        
        lstm_out, _ = self.lstm(encoded)  # (B, T, hidden*2)
        final_output = lstm_out[:, -1, :]  # (B, hidden*2)
        return self.classifier(final_output).squeeze(1)


class GNNLandmarkHead(nn.Module):
    """Graph Neural Network on facial landmarks for head pose."""
    
    def __init__(self, num_landmarks=68, feature_dim=64, num_layers=3):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.feature_dim = feature_dim
        
        # Node feature projection
        self.node_encoder = nn.Linear(2, feature_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Build adjacency matrix (connect nearby landmarks)
        self.register_buffer('adjacency', self._build_adjacency(num_landmarks))
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _build_adjacency(self, n):
        """Build adjacency matrix connecting nearby landmarks."""
        adj = torch.zeros(n, n)
        # Connect landmarks in same facial region (simplified)
        for i in range(n):
            if i > 0:
                adj[i, i-1] = 1
            if i < n-1:
                adj[i, i+1] = 1
        return adj
    
    def forward(self, landmarks_sequence):
        """
        landmarks_sequence: (B, T, num_landmarks, 2)
        Returns: (B, 1) logits per timestep, averaged
        """
        B, T, N, D = landmarks_sequence.shape
        outputs = []
        
        for t in range(T):
            landmarks_t = landmarks_sequence[:, t, :, :]  # (B, N, 2)
            
            # Encode node features
            node_features = self.node_encoder(landmarks_t)  # (B, N, feature_dim)
            
            # Apply GNN layers
            x = node_features
            for gnn_layer in self.gnn_layers:
                # Graph convolution: A @ X @ W
                aggregated = torch.matmul(self.adjacency.unsqueeze(0), x)
                x = gnn_layer(aggregated) + x  # Residual connection
            
            # Global pooling
            x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, feature_dim)
            output = self.classifier(x_pooled)  # (B, 1)
            outputs.append(output)
        
        # Average over time
        final_output = torch.stack(outputs, dim=1).mean(dim=1)  # (B, 1)
        return final_output.squeeze(1)


# ============================================================================
# 3. Lip Sync (Video + Audio)
# ============================================================================

class SyncNetLike(nn.Module):
    """SyncNet-like model for audio-visual lip sync detection."""
    
    def __init__(self, visual_embed_dim=256, audio_embed_dim=256):
        super().__init__()
        # Visual branch (lip region CNN)
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Audio branch (MFCC/spectrogram CNN)
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Sync detection
        self.sync_classifier = nn.Sequential(
            nn.Linear(visual_embed_dim + audio_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, lip_region, audio_features):
        """
        lip_region: (B, 3, H, W)
        audio_features: (B, 1, H_audio, W_audio) - spectrogram/MFCC
        Returns: (B, 1) sync score logits
        """
        visual_embed = self.visual_cnn(lip_region).flatten(1)  # (B, 256)
        audio_embed = self.audio_cnn(audio_features).flatten(1)  # (B, 256)
        
        combined = torch.cat([visual_embed, audio_embed], dim=1)
        return self.sync_classifier(combined).squeeze(1)


class AudioVisualTransformer(nn.Module):
    """Audio-Visual Transformer for lip sync."""
    
    def __init__(self, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        # Visual encoder
        self.visual_proj = nn.Linear(256, d_model)
        
        # Audio encoder
        self.audio_proj = nn.Linear(256, d_model)
        
        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1)
    
    def forward(self, visual_seq, audio_seq):
        """
        visual_seq: (B, T, 256)
        audio_seq: (B, T, 256)
        Returns: (B, 1) logits
        """
        visual_embed = self.visual_proj(visual_seq)
        audio_embed = self.audio_proj(audio_seq)
        
        # Concatenate visual and audio tokens
        combined = torch.cat([visual_embed, audio_embed], dim=1)  # (B, 2*T, d_model)
        
        # Transformer encoding
        encoded = self.transformer(combined)
        
        # Use CLS token (first token after pooling)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled).squeeze(1)


# ============================================================================
# 4. Heartbeat / Skin Signal (rPPG)
# ============================================================================

class rPPGCNN(nn.Module):
    """rPPG-based CNN for heartbeat/skin signal detection."""
    
    def __init__(self, temporal_filter_size=5):
        super().__init__()
        # Temporal filtering layer
        self.temporal_filter = nn.Conv1d(3, 32, kernel_size=temporal_filter_size, padding=temporal_filter_size//2)
        
        # Spatial CNN
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Frequency domain analysis (for heartbeat detection)
        self.freq_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.classifier = nn.Linear(32, 1)
    
    def forward(self, face_region_temporal):
        """
        face_region_temporal: (B, T, 3, H, W) - temporal sequence of face region
        Returns: (B, 1) logits
        """
        B, T, C, H, W = face_region_temporal.shape
        
        # Reshape for temporal filtering: (B*H*W, C, T)
        face_flat = face_region_temporal.permute(0, 3, 4, 2, 1).contiguous()
        face_flat = face_flat.view(B * H * W, C, T)
        
        # Temporal filtering (rPPG signal extraction)
        filtered = self.temporal_filter(face_flat)  # (B*H*W, 32, T)
        
        # Reshape back: (B, 32, H, W, T) -> average over T -> (B, 32, H, W)
        filtered = filtered.view(B, H, W, 32, T).permute(0, 3, 4, 1, 2)
        filtered_mean = filtered.mean(dim=2)  # (B, 32, H, W)
        
        # Spatial CNN
        spatial_features = self.spatial_cnn(filtered_mean).flatten(1)  # (B, 128)
        
        # Frequency analysis
        freq_features = self.freq_analyzer(spatial_features)  # (B, 32)
        
        return self.classifier(freq_features).squeeze(1)


class TemporalFilteringCNN(nn.Module):
    """CNN with temporal filtering for skin signal detection."""
    
    def __init__(self):
        super().__init__()
        # Multi-scale temporal filters
        self.temporal_filters = nn.ModuleList([
            nn.Conv1d(3, 16, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        self.fusion = nn.Conv1d(48, 64, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, face_region_temporal):
        """
        face_region_temporal: (B, T, 3, H, W)
        Returns: (B, 1) logits
        """
        B, T, C, H, W = face_region_temporal.shape
        
        # Average over spatial dimensions: (B, T, 3)
        temporal_signal = face_region_temporal.mean(dim=(3, 4))
        
        # Apply temporal filters: (B, 3, T) -> (B, 16, T)
        filtered_outputs = []
        temporal_signal_t = temporal_signal.transpose(1, 2)  # (B, 3, T)
        for filter_layer in self.temporal_filters:
            filtered_outputs.append(filter_layer(temporal_signal_t))
        
        # Concatenate multi-scale features
        combined = torch.cat(filtered_outputs, dim=1)  # (B, 48, T)
        fused = self.fusion(combined)  # (B, 64, T)
        
        # Global pooling
        pooled = fused.mean(dim=2)  # (B, 64)
        return self.classifier(pooled).squeeze(1)


# ============================================================================
# Combined Physiological Detector
# ============================================================================

class PhysiologicalDetector(nn.Module):
    """
    Combined physiological and behavioral deepfake detector.
    Integrates all detection methods with weighted fusion.
    """
    
    def __init__(self):
        super().__init__()
        self.eye_blink_gaze = EyeBlinkGazeLSTM()
        self.ear_lstm = EARLSTM()
        self.landmark_lstm = FacialLandmarksLSTM()
        self.landmark_gnn = GNNLandmarkHead()
        self.syncnet = SyncNetLike()
        self.audio_visual_transformer = AudioVisualTransformer()
        self.rppg_cnn = rPPGCNN()
        self.temporal_filter_cnn = TemporalFilteringCNN()
        
        # Weighted fusion (learnable or fixed)
        self.fusion_weights = nn.Parameter(torch.ones(8) / 8)
        self.final_classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, inputs_dict):
        """
        inputs_dict: dictionary containing all necessary inputs
        Returns: (B, 1) logits
        """
        predictions = []
        
        # Eye blink/gaze
        if 'eye_regions' in inputs_dict:
            pred_eye = self.eye_blink_gaze(inputs_dict['eye_regions'])
            predictions.append(pred_eye)
        
        # EAR
        if 'ear_sequence' in inputs_dict:
            pred_ear = self.ear_lstm(inputs_dict['ear_sequence'])
            predictions.append(pred_ear)
        
        # Landmarks LSTM
        if 'landmarks_sequence' in inputs_dict:
            pred_landmark_lstm = self.landmark_lstm(inputs_dict['landmarks_sequence'])
            predictions.append(pred_landmark_lstm)
        
        # Landmarks GNN
        if 'landmarks_sequence' in inputs_dict:
            pred_landmark_gnn = self.landmark_gnn(inputs_dict['landmarks_sequence'])
            predictions.append(pred_landmark_gnn)
        
        # SyncNet
        if 'lip_region' in inputs_dict and 'audio_features' in inputs_dict:
            pred_sync = self.syncnet(inputs_dict['lip_region'], inputs_dict['audio_features'])
            predictions.append(pred_sync)
        
        # Audio-Visual Transformer
        if 'visual_seq' in inputs_dict and 'audio_seq' in inputs_dict:
            pred_av_transformer = self.audio_visual_transformer(
                inputs_dict['visual_seq'], inputs_dict['audio_seq']
            )
            predictions.append(pred_av_transformer)
        
        # rPPG
        if 'face_region_temporal' in inputs_dict:
            pred_rppg = self.rppg_cnn(inputs_dict['face_region_temporal'])
            predictions.append(pred_rppg)
        
        # Temporal Filtering
        if 'face_region_temporal' in inputs_dict:
            pred_temp = self.temporal_filter_cnn(inputs_dict['face_region_temporal'])
            predictions.append(pred_temp)
        
        # Combine predictions
        if len(predictions) == 0:
            return torch.zeros(1, 1)
        
        stacked = torch.stack(predictions, dim=1)  # (B, num_methods)
        # Apply learned weights
        weighted = stacked * self.fusion_weights.unsqueeze(0)
        return self.final_classifier(weighted).squeeze(1)

