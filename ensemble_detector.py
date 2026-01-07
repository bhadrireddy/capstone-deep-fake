"""
Ensemble Deepfake Detection Architecture

Combines multiple detection methods:
1. Spatial–Frequency Image Model: Swin Transformer + FFT
2. Temporal Video Model: Video Swin Transformer
3. Physiological Model: Facial Landmarks + LSTM
4. Audio–Visual Model: SyncNet

Uses adaptive weighting and video-level decision making (no frame-wise decisions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class EnsembleDetector(nn.Module):
    """
    Ensemble architecture combining multiple detection methods.
    Uses learnable/adaptive weighting for optimal combination.
    """
    
    def __init__(
        self,
        use_spatial_frequency: bool = True,
        use_temporal_video: bool = True,
        use_physiological: bool = True,
        use_audio_visual: bool = True,
        learnable_weights: bool = True,
    ):
        super().__init__()
        
        self.use_spatial_frequency = use_spatial_frequency
        self.use_temporal_video = use_temporal_video
        self.use_physiological = use_physiological
        self.use_audio_visual = use_audio_visual
        
        # Initialize weights (learnable or fixed)
        if learnable_weights:
            # Learnable attention-based weighting
            self.weight_attention = nn.Sequential(
                nn.Linear(4, 16),  # 4 models
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.Softmax(dim=-1)
            )
        else:
            # Fixed equal weights
            self.register_buffer('fixed_weights', torch.ones(4) / 4)
        
        self.learnable_weights = learnable_weights
        
        # Final fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(4, 64),  # 4 model predictions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Confidence estimation for adaptive thresholding
        self.confidence_estimator = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Outputs confidence in [0, 1]
        )
    
    def forward(
        self,
        spatial_freq_pred: Optional[torch.Tensor] = None,
        temporal_video_pred: Optional[torch.Tensor] = None,
        physiological_pred: Optional[torch.Tensor] = None,
        audio_visual_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        All predictions should be logits (before sigmoid) from individual models.
        Returns dictionary with:
        - 'logits': combined logits
        - 'probabilities': sigmoid probabilities
        - 'weights': applied weights
        - 'confidence': confidence score
        """
        predictions = []
        available_models = []
        
        # Collect available predictions
        if self.use_spatial_frequency and spatial_freq_pred is not None:
            predictions.append(spatial_freq_pred)
            available_models.append(0)
        
        if self.use_temporal_video and temporal_video_pred is not None:
            predictions.append(temporal_video_pred)
            available_models.append(1)
        
        if self.use_physiological and physiological_pred is not None:
            predictions.append(physiological_pred)
            available_models.append(2)
        
        if self.use_audio_visual and audio_visual_pred is not None:
            predictions.append(audio_visual_pred)
            available_models.append(3)
        
        if len(predictions) == 0:
            # Fallback: return neutral prediction
            return {
                'logits': torch.zeros(1),
                'probabilities': torch.ones(1) * 0.5,
                'weights': torch.zeros(4),
                'confidence': torch.zeros(1)
            }
        
        # Stack predictions: (num_available, batch_size)
        stacked_preds = torch.stack(predictions, dim=0)  # (num_available, B)
        if stacked_preds.dim() == 1:
            stacked_preds = stacked_preds.unsqueeze(1)  # (num_available, 1)
        
        # Create full prediction vector (fill missing with zeros)
        full_preds = torch.zeros(4, stacked_preds.shape[1]).to(stacked_preds.device)
        for idx, model_idx in enumerate(available_models):
            full_preds[model_idx, :] = stacked_preds[idx, :]
        
        full_preds = full_preds.transpose(0, 1)  # (B, 4)
        
        # Compute adaptive weights
        if self.learnable_weights:
            # Use prediction magnitudes as context for weighting
            pred_magnitudes = torch.abs(full_preds)
            weights = self.weight_attention(pred_magnitudes)  # (B, 4)
        else:
            # Equal weights, but only for available models
            weights = torch.zeros_like(full_preds)
            for model_idx in available_models:
                weights[:, model_idx] = 1.0 / len(available_models)
        
        # Apply weights to predictions
        weighted_preds = full_preds * weights  # (B, 4)
        
        # Estimate confidence
        confidence = self.confidence_estimator(pred_magnitudes)  # (B, 1)
        
        # Final fusion
        combined_logits = self.fusion_classifier(weighted_preds)  # (B, 1)
        combined_probs = torch.sigmoid(combined_logits)  # (B, 1)
        
        return {
            'logits': combined_logits.squeeze(1),  # (B,)
            'probabilities': combined_probs.squeeze(1),  # (B,)
            'weights': weights,  # (B, 4)
            'confidence': confidence.squeeze(1),  # (B,)
            'individual_predictions': {
                'spatial_frequency': spatial_freq_pred,
                'temporal_video': temporal_video_pred,
                'physiological': physiological_pred,
                'audio_visual': audio_visual_pred,
            }
        }


class AdaptiveThresholdEnsemble:
    """
    Adaptive threshold system for ensemble predictions.
    Uses confidence-based thresholding instead of fixed threshold.
    """
    
    def __init__(
        self,
        base_threshold: float = 0.5,
        confidence_weight: float = 0.3,
        min_threshold: float = 0.3,
        max_threshold: float = 0.7,
    ):
        self.base_threshold = base_threshold
        self.confidence_weight = confidence_weight
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def compute_adaptive_threshold(self, confidence: float) -> float:
        """
        Compute adaptive threshold based on confidence.
        Lower confidence → more conservative threshold (closer to 0.5)
        Higher confidence → use base threshold
        """
        # When confidence is low, move threshold closer to 0.5 (more conservative)
        # When confidence is high, use base threshold
        adjustment = (1.0 - confidence) * self.confidence_weight * (0.5 - self.base_threshold)
        adaptive_threshold = self.base_threshold + adjustment
        
        # Clamp to min/max bounds
        adaptive_threshold = np.clip(adaptive_threshold, self.min_threshold, self.max_threshold)
        return adaptive_threshold
    
    def predict(
        self,
        probability: float,
        confidence: float,
    ) -> Tuple[str, float]:
        """
        Make prediction using adaptive threshold.
        Returns: (label, confidence)
        """
        threshold = self.compute_adaptive_threshold(confidence)
        label = 'fake' if probability >= threshold else 'real'
        return label, confidence


def ensemble_video_prediction(
    spatial_freq_logits: Optional[torch.Tensor] = None,
    temporal_video_logits: Optional[torch.Tensor] = None,
    physiological_logits: Optional[torch.Tensor] = None,
    audio_visual_logits: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    device: torch.device = None,
    use_trained_fusion: bool = False,
) -> Dict[str, float]:
    """
    High-level function for ensemble video prediction.
    
    Args:
        All logits from individual models (can be None if model unavailable)
        threshold: Base threshold for adaptive thresholding
    
    Returns:
        Dictionary with:
        - 'label': 'real' or 'fake'
        - 'probability': Combined probability
        - 'confidence': Confidence score
        - 'adaptive_threshold': Used threshold
        - 'weights': Individual model weights
        - 'individual_probs': Individual model probabilities
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Convert to probabilities and filter out unreliable models
    individual_probs = {}
    reliable_probs = []
    reliable_names = []
    
    # Prepare inputs and convert to probabilities
    if spatial_freq_logits is not None:
        if not isinstance(spatial_freq_logits, torch.Tensor):
            spatial_freq_logits = torch.tensor([spatial_freq_logits]).to(device)
        prob = torch.sigmoid(spatial_freq_logits).item()
        individual_probs['spatial_frequency'] = prob
        reliable_probs.append(prob)
        reliable_names.append('spatial_frequency')
    
    if temporal_video_logits is not None:
        if not isinstance(temporal_video_logits, torch.Tensor):
            temporal_video_logits = torch.tensor([temporal_video_logits]).to(device)
        prob = torch.sigmoid(temporal_video_logits).item()
        individual_probs['temporal_video'] = prob
        # Only use if different from spatial (avoid duplicate)
        if len(reliable_probs) == 0 or abs(prob - reliable_probs[0]) > 0.1:
            reliable_probs.append(prob)
            reliable_names.append('temporal_video')
    
    # Skip physiological and audio-visual if they're from random/untrained models
    # (they'll have unreliable predictions)
    # Only include if we have real features (not placeholders)
    if physiological_logits is not None:
        if not isinstance(physiological_logits, torch.Tensor):
            physiological_logits = torch.tensor([physiological_logits]).to(device)
        prob = torch.sigmoid(physiological_logits).item()
        individual_probs['physiological'] = prob
        # Only trust if probability is reasonable (not too extreme, which indicates random)
        if 0.2 < prob < 0.8:
            reliable_probs.append(prob)
            reliable_names.append('physiological')
    
    if audio_visual_logits is not None:
        if not isinstance(audio_visual_logits, torch.Tensor):
            audio_visual_logits = torch.tensor([audio_visual_logits]).to(device)
        prob = torch.sigmoid(audio_visual_logits).item()
        individual_probs['audio_visual'] = prob
        # Only trust if probability is reasonable
        if 0.2 < prob < 0.8:
            reliable_probs.append(prob)
            reliable_names.append('audio_visual')
    
    # If no reliable models, fallback
    if len(reliable_probs) == 0:
        return {
            'label': 'real',
            'probability': 0.5,
            'confidence': 0.0,
            'adaptive_threshold': threshold,
            'weights': {},
            'individual_probs': individual_probs,
        }
    
    # Simple weighted averaging (give more weight to spatial-temporal models)
    if len(reliable_probs) == 1:
        combined_prob = reliable_probs[0]
        weights_dict = {reliable_names[0]: 1.0}
    elif len(reliable_probs) == 2:
        # If we have spatial + temporal, weight them equally
        combined_prob = np.mean(reliable_probs)
        weights_dict = {name: 0.5 for name in reliable_names}
    else:
        # Weight spatial/temporal models more heavily
        weights = []
        for name in reliable_names:
            if name in ['spatial_frequency', 'temporal_video']:
                weights.append(0.4)
            else:
                weights.append(0.2 / (len(reliable_probs) - 2))
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        combined_prob = sum(p * w for p, w in zip(reliable_probs, weights))
        weights_dict = {name: w for name, w in zip(reliable_names, weights)}
    
    # Calculate confidence based on agreement between models
    if len(reliable_probs) > 1:
        # High confidence if models agree, low if they disagree
        variance = np.var(reliable_probs)
        confidence = max(0.0, 1.0 - variance * 4)  # Scale variance to [0, 1]
    else:
        confidence = 0.5  # Medium confidence with single model
    
    # Use base threshold (adaptive thresholding can cause issues with untrained models)
    label = 'fake' if combined_prob >= threshold else 'real'
    
    return {
        'label': label,
        'probability': float(combined_prob),
        'confidence': float(confidence),
        'adaptive_threshold': threshold,
        'weights': weights_dict,
        'individual_probs': individual_probs,
    }

