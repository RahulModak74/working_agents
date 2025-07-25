#!/usr/bin/env python3
"""
Production-Ready Pyro Security Analyzer with Model Persistence
Supports CrowdStrike, Cloudflare, and Defender log types
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle
import os
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Pyro imports with fallback
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
    from pyro.optim import Adam, ClippedAdam
    from pyro.infer.autoguide import AutoDiagonalNormal
    PYRO_AVAILABLE = True
    pyro.set_rng_seed(42)
except ImportError:
    PYRO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
def get_optimal_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("🔧 Using CPU")
    return device

device = get_optimal_device()

@dataclass
class SecurityFeatures:
    """Universal security features for all log types"""
    temporal_anomaly: float
    multi_vector: float
    privilege_escalation: float
    lateral_movement: float
    data_access: float
    persistence: float
    entropy_score: float
    behavioral_deviation: float
    network_centrality: float
    file_system_anomaly: float
    process_chain_depth: float
    credential_access: float

class UniversalHMM(nn.Module):
    """Universal HMM that adapts to different log types"""
    
    def __init__(self, log_type="crowdstrike", num_states=6, feature_dim=12, device=None):
        super().__init__()
        self.log_type = log_type
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.device = device or get_optimal_device()
        
        # Universal state names
        self.state_names = [
            "normal_operations",
            "reconnaissance", 
            "initial_access",
            "privilege_escalation",
            "lateral_movement",
            "data_exfiltration"
        ]
        
        # Log-type specific attack weights
        if log_type == "crowdstrike":
            self.state_attack_weights = {
                0: 0.0, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.85, 5: 0.95
            }
        elif log_type == "cloudflare":
            self.state_attack_weights = {
                0: 0.0, 1: 0.30, 2: 0.60, 3: 0.40, 4: 0.70, 5: 0.90
            }
        elif log_type == "defender":
            self.state_attack_weights = {
                0: 0.0, 1: 0.20, 2: 0.50, 3: 0.75, 4: 0.80, 5: 0.95
            }
        else:
            self.state_attack_weights = {
                0: 0.0, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.85, 5: 0.95
            }
        
        self._init_parameters()
        self.trained = False
        self.training_history = []
    
    def _init_parameters(self):
        """Initialize HMM parameters"""
        trans_init = torch.eye(self.num_states) * 0.7
        trans_init += torch.randn(self.num_states, self.num_states) * 0.1
        self.transition_logits = nn.Parameter(trans_init)
        
        initial_bias = torch.full((self.num_states,), -2.0)
        initial_bias[0] = 1.5
        self.initial_logits = nn.Parameter(initial_bias)
        
        self.emission_means = nn.Parameter(torch.randn(self.num_states, self.feature_dim) * 0.3)
        self.emission_log_scales = nn.Parameter(torch.full((self.num_states, self.feature_dim), -0.5))
        
        self.to(self.device)
    
    def get_transition_probs(self):
        return F.softmax(self.transition_logits, dim=-1)
    
    def get_initial_probs(self):
        return F.softmax(self.initial_logits, dim=-1)
    
    def get_emission_params(self):
        means = self.emission_means
        scales = F.softplus(self.emission_log_scales) + 1e-6
        return means, scales
    
    def predict_attack_probability(self, observations):
        """Predict attack probability for a sequence"""
        if not self.trained or len(observations) == 0:
            return 0.0
        
        try:
            predicted_states = self.viterbi_decode(observations)
            attack_prob = 0.0
            for state in predicted_states:
                state_weight = self.state_attack_weights.get(int(state), 0.0)
                attack_prob += state_weight
            
            return min(max(attack_prob / len(predicted_states), 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Attack probability prediction failed: {e}")
            return 0.5
    
    def viterbi_decode(self, observations):
        """Viterbi decoding for most likely state sequence"""
        if not self.trained or len(observations) == 0:
            return torch.zeros(len(observations), dtype=torch.long)
        
        try:
            seq_len = len(observations)
            observations = observations.to(self.device)
            
            transition_probs = self.get_transition_probs()
            initial_probs = self.get_initial_probs()
            emission_means, emission_scales = self.get_emission_params()
            
            viterbi_log_probs = torch.full((seq_len, self.num_states), -float('inf'), device=self.device)
            viterbi_path = torch.zeros((seq_len, self.num_states), dtype=torch.long, device=self.device)
            
            # Initialize first timestep
            for s in range(self.num_states):
                emission_logp = self._compute_emission_logp(observations[0], emission_means[s], emission_scales[s])
                viterbi_log_probs[0, s] = torch.log(initial_probs[s] + 1e-8) + emission_logp
            
            # Forward pass
            for t in range(1, seq_len):
                for s in range(self.num_states):
                    trans_scores = viterbi_log_probs[t-1] + torch.log(transition_probs[:, s] + 1e-8)
                    best_prev_state = torch.argmax(trans_scores)
                    
                    viterbi_path[t, s] = best_prev_state
                    emission_logp = self._compute_emission_logp(observations[t], emission_means[s], emission_scales[s])
                    viterbi_log_probs[t, s] = trans_scores[best_prev_state] + emission_logp
            
            # Backward pass
            path = torch.zeros(seq_len, dtype=torch.long, device=self.device)
            path[-1] = torch.argmax(viterbi_log_probs[-1])
            
            for t in range(seq_len - 2, -1, -1):
                path[t] = viterbi_path[t + 1, path[t + 1]]
            
            return path.cpu()
            
        except Exception as e:
            logger.warning(f"Viterbi decoding failed: {e}")
            return torch.zeros(len(observations), dtype=torch.long)
    
    def _compute_emission_logp(self, obs, mean, scale):
        """Compute emission log probability"""
        diff = obs - mean
        return -0.5 * ((diff / (scale + 1e-6)) ** 2).sum() - scale.log().sum()
    
    def fit_pytorch(self, sequences, num_iterations=200, learning_rate=0.01):
        """PyTorch-based training"""
        logger.info(f"🔧 Training {self.log_type} HMM with PyTorch")
        
        processed_sequences = self._prepare_sequences(sequences)
        if not processed_sequences:
            return []
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
        
        losses = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            total_loss = 0.0
            for seq in processed_sequences:
                loss = self._compute_loss(seq)
                total_loss += loss
            
            if total_loss > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            
            losses.append(total_loss.item())
            scheduler.step(total_loss)
            
            if iteration % 50 == 0:
                logger.info(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")
        
        self.trained = True
        self.training_history = losses
        return losses
    
    def _prepare_sequences(self, sequences, max_sequences=75, max_length=20):
        """Prepare sequences for training"""
        processed = []
        
        for i, seq in enumerate(sequences[:max_sequences]):
            if torch.is_tensor(seq):
                seq_tensor = seq.to(self.device)
            else:
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            
            if len(seq_tensor) > max_length:
                seq_tensor = seq_tensor[:max_length]
            
            if len(seq_tensor) >= 2 and torch.isfinite(seq_tensor).all():
                processed.append(seq_tensor)
        
        return processed
    
    def _compute_loss(self, seq):
        """Compute sequence loss"""
        seq_len = len(seq)
        
        transition_probs = self.get_transition_probs()
        initial_probs = self.get_initial_probs()
        emission_means, emission_scales = self.get_emission_params()
        
        log_alpha = torch.full((seq_len, self.num_states), -float('inf'), device=self.device)
        
        # Initialize
        for s in range(self.num_states):
            emission_logp = self._compute_emission_logp(seq[0], emission_means[s], emission_scales[s])
            log_alpha[0, s] = torch.log(initial_probs[s] + 1e-8) + emission_logp
        
        # Forward pass
        for t in range(1, seq_len):
            for s in range(self.num_states):
                trans_scores = log_alpha[t-1] + torch.log(transition_probs[:, s] + 1e-8)
                log_alpha[t, s] = torch.logsumexp(trans_scores, dim=0)
                
                emission_logp = self._compute_emission_logp(seq[t], emission_means[s], emission_scales[s])
                log_alpha[t, s] += emission_logp
        
        total_logp = torch.logsumexp(log_alpha[-1], dim=0)
        reg_loss = 0.01 * (transition_probs.var() + emission_scales.mean())
        
        return -(total_logp - reg_loss)

class LogTypeProcessor:
    """Process different log types into universal features"""
    
    def __init__(self, log_type="crowdstrike"):
        self.log_type = log_type
        
        # Common suspicious indicators
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 'psexec.exe',
            'rundll32.exe', 'regsvr32.exe', 'certutil.exe', 'bitsadmin.exe'
        }
        self.suspicious_ports = {135, 139, 445, 3389, 22, 1433, 1521, 5985, 5986}
        self.internal_networks = ['10.', '192.168.', '172.16.', '172.17.', '172.18.']
    
    def extract_features_from_session(self, session_data: Dict) -> SecurityFeatures:
        """Extract universal features from any log type"""
        
        if self.log_type == "crowdstrike":
            return self._extract_crowdstrike_features(session_data)
        elif self.log_type == "cloudflare":
            return self._extract_cloudflare_features(session_data)
        elif self.log_type == "defender":
            return self._extract_defender_features(session_data)
        else:
            return self._extract_generic_features(session_data)
    
    def _extract_crowdstrike_features(self, session_data: Dict) -> SecurityFeatures:
        """Extract features from CrowdStrike logs"""
        risk_prob = session_data.get('risk_probability', 0.0)
        alert_count = session_data.get('alert_count', 0)
        diversity = session_data.get('event_type_diversity', 1)
        multi_vector = session_data.get('multi_vector', False)
        attack_intensity = session_data.get('attack_intensity', 0.0)
        
        # CrowdStrike-specific feature extraction
        temporal_anomaly = min(risk_prob * 1.2, 1.0)
        multi_vector_score = 1.0 if multi_vector else min(diversity / 3.0, 1.0)
        privilege_escalation = min(attack_intensity * 1.5, 1.0)
        lateral_movement = min(risk_prob * attack_intensity, 1.0)
        data_access = min(alert_count / 50.0, 1.0)
        persistence = min(attack_intensity * 0.8, 1.0)
        
        # Advanced features
        entropy_score = min(diversity * risk_prob, 1.0)
        behavioral_deviation = min(attack_intensity * 1.2, 1.0)
        network_centrality = min(risk_prob * 0.7, 1.0)
        file_system_anomaly = min(attack_intensity * 0.9, 1.0)
        process_chain_depth = min(alert_count / 30.0, 1.0)
        credential_access = min(attack_intensity * 1.1, 1.0)
        
        return SecurityFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector_score,
            privilege_escalation=privilege_escalation,
            lateral_movement=lateral_movement,
            data_access=data_access,
            persistence=persistence,
            entropy_score=entropy_score,
            behavioral_deviation=behavioral_deviation,
            network_centrality=network_centrality,
            file_system_anomaly=file_system_anomaly,
            process_chain_depth=process_chain_depth,
            credential_access=credential_access
        )
    
    def _extract_cloudflare_features(self, session_data: Dict) -> SecurityFeatures:
        """Extract features from Cloudflare logs"""
        risk_prob = session_data.get('risk_probability', 0.0)
        alert_count = session_data.get('alert_count', 0)
        attack_intensity = session_data.get('attack_intensity', 0.0)
        
        # Cloudflare focuses more on network patterns
        temporal_anomaly = min(risk_prob * 0.8, 1.0)
        multi_vector_score = min(alert_count / 10.0, 1.0)
        privilege_escalation = min(attack_intensity * 0.6, 1.0)  # Less relevant for network logs
        lateral_movement = min(risk_prob * 1.3, 1.0)  # More relevant for network
        data_access = min(attack_intensity * 1.2, 1.0)
        persistence = min(risk_prob * 0.5, 1.0)  # Less relevant for network logs
        
        # Network-focused features
        entropy_score = min(risk_prob * attack_intensity * 1.5, 1.0)
        behavioral_deviation = min(attack_intensity * 1.4, 1.0)
        network_centrality = min(risk_prob * 1.5, 1.0)  # Most important for Cloudflare
        file_system_anomaly = min(attack_intensity * 0.3, 1.0)  # Less relevant
        process_chain_depth = min(attack_intensity * 0.4, 1.0)  # Less relevant
        credential_access = min(attack_intensity * 0.7, 1.0)
        
        return SecurityFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector_score,
            privilege_escalation=privilege_escalation,
            lateral_movement=lateral_movement,
            data_access=data_access,
            persistence=persistence,
            entropy_score=entropy_score,
            behavioral_deviation=behavioral_deviation,
            network_centrality=network_centrality,
            file_system_anomaly=file_system_anomaly,
            process_chain_depth=process_chain_depth,
            credential_access=credential_access
        )
    
    def _extract_defender_features(self, session_data: Dict) -> SecurityFeatures:
        """Extract features from Windows Defender logs"""
        risk_prob = session_data.get('risk_probability', 0.0)
        alert_count = session_data.get('alert_count', 0)
        diversity = session_data.get('event_type_diversity', 1)
        attack_intensity = session_data.get('attack_intensity', 0.0)
        
        # Defender focuses on endpoint and process behavior
        temporal_anomaly = min(risk_prob * 1.1, 1.0)
        multi_vector_score = min(diversity / 2.5, 1.0)
        privilege_escalation = min(attack_intensity * 1.8, 1.0)  # Very important for Defender
        lateral_movement = min(risk_prob * 0.9, 1.0)
        data_access = min(alert_count / 40.0, 1.0)
        persistence = min(attack_intensity * 1.6, 1.0)  # Very important for Defender
        
        # Windows-focused features
        entropy_score = min(diversity * attack_intensity, 1.0)
        behavioral_deviation = min(attack_intensity * 1.3, 1.0)
        network_centrality = min(risk_prob * 0.6, 1.0)  # Less relevant
        file_system_anomaly = min(attack_intensity * 1.4, 1.0)  # Very important
        process_chain_depth = min(alert_count / 20.0, 1.0)  # Very important
        credential_access = min(attack_intensity * 1.7, 1.0)  # Very important
        
        return SecurityFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector_score,
            privilege_escalation=privilege_escalation,
            lateral_movement=lateral_movement,
            data_access=data_access,
            persistence=persistence,
            entropy_score=entropy_score,
            behavioral_deviation=behavioral_deviation,
            network_centrality=network_centrality,
            file_system_anomaly=file_system_anomaly,
            process_chain_depth=process_chain_depth,
            credential_access=credential_access
        )
    
    def _extract_generic_features(self, session_data: Dict) -> SecurityFeatures:
        """Extract generic features for unknown log types"""
        risk_prob = session_data.get('risk_probability', 0.0)
        alert_count = session_data.get('alert_count', 0)
        diversity = session_data.get('event_type_diversity', 1)
        attack_intensity = session_data.get('attack_intensity', 0.0)
        
        return SecurityFeatures(
            temporal_anomaly=min(risk_prob, 1.0),
            multi_vector=min(diversity / 3.0, 1.0),
            privilege_escalation=min(attack_intensity, 1.0),
            lateral_movement=min(risk_prob * 0.8, 1.0),
            data_access=min(alert_count / 30.0, 1.0),
            persistence=min(attack_intensity * 0.9, 1.0),
            entropy_score=min(risk_prob * diversity / 3.0, 1.0),
            behavioral_deviation=min(attack_intensity, 1.0),
            network_centrality=min(risk_prob * 0.7, 1.0),
            file_system_anomaly=min(attack_intensity * 0.8, 1.0),
            process_chain_depth=min(alert_count / 25.0, 1.0),
            credential_access=min(attack_intensity * 0.9, 1.0)
        )

class ProductionSecurityAnalyzer:
    """Production-ready security analyzer with model persistence"""
    
    def __init__(self, log_type="crowdstrike", model_dir="models"):
        self.log_type = log_type
        self.model_dir = model_dir
        self.processor = LogTypeProcessor(log_type)
        self.hmm = UniversalHMM(log_type=log_type, device=device)
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_path = os.path.join(model_dir, f"{log_type}_hmm_model.pth")
        self.metadata_path = os.path.join(model_dir, f"{log_type}_metadata.json")
        
        self.trained = False
        self.feature_sequences = []
        self.session_metadata = []
    
    def load_training_data(self, json_file: str, max_sessions: int = 500):
        """Load training data from JSON file"""
        logger.info(f"📁 Loading {self.log_type} training data from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                sessions = data[:max_sessions]
            else:
                sessions = list(data.values())[:max_sessions]
            
            logger.info(f"✅ Loaded {len(sessions)} sessions")
            
            # Process sessions into feature sequences
            feature_sequences = []
            session_metadata = []
            
            for session in sessions:
                features = self.processor.extract_features_from_session(session)
                
                # Convert to feature vector
                feature_vector = [
                    features.temporal_anomaly, features.multi_vector,
                    features.privilege_escalation, features.lateral_movement,
                    features.data_access, features.persistence,
                    features.entropy_score, features.behavioral_deviation,
                    features.network_centrality, features.file_system_anomaly,
                    features.process_chain_depth, features.credential_access
                ]
                
                # Create time windows (simplified for production)
                num_windows = max(2, min(10, session.get('alert_count', 5) // 3))
                sequence = []
                
                for i in range(num_windows):
                    # Add noise for temporal variation
                    noise = np.random.normal(0, 0.05, len(feature_vector))
                    windowed_features = [max(0, min(1, f + n)) for f, n in zip(feature_vector, noise)]
                    sequence.append(windowed_features)
                
                feature_tensor = torch.tensor(sequence, dtype=torch.float32)
                feature_sequences.append(feature_tensor)
                
                session_metadata.append({
                    'session_id': session.get('session_id', f'session_{len(session_metadata)}'),
                    'is_attack': session.get('is_attack', False),
                    'risk_probability': session.get('risk_probability', 0.0),
                    'attack_intensity': session.get('attack_intensity', 0.0)
                })
            
            self.feature_sequences = feature_sequences
            self.session_metadata = session_metadata
            
            logger.info(f"✅ Prepared {len(feature_sequences)} feature sequences")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            return False
    
    def train_model(self, num_iterations=200, learning_rate=0.01):
        """Train the HMM model"""
        if not self.feature_sequences:
            raise ValueError("No training data loaded")
        
        logger.info(f"🚀 Training {self.log_type} model...")
        
        losses = self.hmm.fit_pytorch(self.feature_sequences, num_iterations, learning_rate)
        
        if self.hmm.trained:
            logger.info("✅ Model training completed")
            self.trained = True
            return losses
        else:
            logger.error("❌ Model training failed")
            return []
    
    def save_model(self):
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Save model state
            torch.save({
                'model_state_dict': self.hmm.state_dict(),
                'log_type': self.log_type,
                'num_states': self.hmm.num_states,
                'feature_dim': self.hmm.feature_dim,
                'state_names': self.hmm.state_names,
                'state_attack_weights': self.hmm.state_attack_weights,
                'training_history': self.hmm.training_history
            }, self.model_path)
            
            # Save metadata
            metadata = {
                'log_type': self.log_type,
                'model_path': self.model_path,
                'trained_at': datetime.now().isoformat(),
                'training_sessions': len(self.session_metadata),
                'device': str(self.hmm.device)
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Model saved: {self.model_path}")
            logger.info(f"💾 Metadata saved: {self.metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.hmm.device)
            
            # Recreate model with saved parameters
            self.hmm = UniversalHMM(
                log_type=checkpoint['log_type'],
                num_states=checkpoint['num_states'],
                feature_dim=checkpoint['feature_dim'],
                device=self.hmm.device
            )
            
            self.hmm.load_state_dict(checkpoint['model_state_dict'])
            self.hmm.state_names = checkpoint['state_names']
            self.hmm.state_attack_weights = checkpoint['state_attack_weights']
            self.hmm.training_history = checkpoint.get('training_history', [])
            self.hmm.trained = True
            self.trained = True
            
            logger.info(f"✅ Model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict_session(self, session_data: Dict) -> Dict:
        """Predict attack probability for a single session"""
        if not self.trained:
            raise ValueError("Model must be trained or loaded before prediction")
        
        try:
            # Extract features
            features = self.processor.extract_features_from_session(session_data)
            
            feature_vector = [
                features.temporal_anomaly, features.multi_vector,
                features.privilege_escalation, features.lateral_movement,
                features.data_access, features.persistence,
                features.entropy_score, features.behavioral_deviation,
                features.network_centrality, features.file_system_anomaly,
                features.process_chain_depth, features.credential_access
            ]
            
            # Create sequence (simplified for single prediction)
            num_windows = max(2, min(8, session_data.get('alert_count', 5) // 2))
            sequence = []
            
            for i in range(num_windows):
                # Slight variations for temporal modeling
                variation = 0.02 * (i - num_windows/2) / (num_windows/2)
                windowed_features = [max(0, min(1, f + variation)) for f in feature_vector]
                sequence.append(windowed_features)
            
            observations = torch.tensor(sequence, dtype=torch.float32)
            
            # Get predictions
            attack_probability = self.hmm.predict_attack_probability(observations)
            predicted_states = self.hmm.viterbi_decode(observations)
            state_sequence = [self.hmm.state_names[state] for state in predicted_states]
            
            # Risk assessment
            risk_level = self._assess_risk_level(attack_probability)
            
            return {
                'session_id': session_data.get('session_id', 'unknown'),
                'log_type': self.log_type,
                'attack_probability': float(attack_probability),
                'predicted_states': predicted_states.tolist(),
                'state_sequence': state_sequence,
                'risk_level': risk_level['level'],
                'priority': risk_level['priority'],
                'confidence': min(0.9, max(0.6, attack_probability + 0.3)),
                'feature_summary': {
                    'temporal_anomaly': features.temporal_anomaly,
                    'multi_vector': features.multi_vector,
                    'privilege_escalation': features.privilege_escalation,
                    'lateral_movement': features.lateral_movement,
                    'data_access': features.data_access,
                    'persistence': features.persistence,
                    'behavioral_deviation': features.behavioral_deviation,
                    'network_centrality': features.network_centrality
                },
                'recommended_action': risk_level['action'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'session_id': session_data.get('session_id', 'unknown'),
                'log_type': self.log_type,
                'error': str(e),
                'attack_probability': 0.0,
                'risk_level': 'ERROR',
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_risk_level(self, attack_probability: float) -> Dict:
        """Assess risk level based on attack probability"""
        if attack_probability >= 0.8:
            return {
                'level': 'CRITICAL',
                'priority': 1,
                'action': 'Immediate investigation and containment required'
            }
        elif attack_probability >= 0.6:
            return {
                'level': 'HIGH',
                'priority': 2,
                'action': 'Urgent investigation recommended'
            }
        elif attack_probability >= 0.4:
            return {
                'level': 'MEDIUM',
                'priority': 3,
                'action': 'Monitor closely and investigate when resources available'
            }
        elif attack_probability >= 0.2:
            return {
                'level': 'LOW',
                'priority': 4,
                'action': 'Low priority monitoring'
            }
        else:
            return {
                'level': 'MINIMAL',
                'priority': 5,
                'action': 'Standard monitoring sufficient'
            }
    
    def batch_predict(self, sessions: List[Dict]) -> List[Dict]:
        """Predict attack probabilities for multiple sessions"""
        if not self.trained:
            raise ValueError("Model must be trained or loaded before prediction")
        
        results = []
        for session in sessions:
            try:
                result = self.predict_session(session)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for session {session.get('session_id', 'unknown')}: {e}")
                results.append({
                    'session_id': session.get('session_id', 'unknown'),
                    'log_type': self.log_type,
                    'error': str(e),
                    'attack_probability': 0.0,
                    'risk_level': 'ERROR',
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.trained:
            return {'status': 'not_trained', 'log_type': self.log_type}
        
        return {
            'status': 'trained',
            'log_type': self.log_type,
            'model_path': self.model_path,
            'num_states': self.hmm.num_states,
            'feature_dim': self.hmm.feature_dim,
            'state_names': self.hmm.state_names,
            'device': str(self.hmm.device),
            'training_sessions': len(self.session_metadata) if self.session_metadata else 0
        }

# Factory function to create analyzers for different log types
def create_analyzer(log_type: str, model_dir: str = "models") -> ProductionSecurityAnalyzer:
    """Factory function to create analyzer for specific log type"""
    supported_types = ["crowdstrike", "cloudflare", "defender"]
    
    if log_type not in supported_types:
        logger.warning(f"Unknown log type {log_type}, using generic analyzer")
        log_type = "generic"
    
    return ProductionSecurityAnalyzer(log_type=log_type, model_dir=model_dir)

# Training script for production deployment
def train_production_models():
    """Train all three models for production"""
    log_types = [
        ("crowdstrike", "synthetic_crowdstrike_logs_hmm_training.json"),
        ("cloudflare", "synthetic_cloudflare_logs_hmm_training.json"),
        ("defender", "synthetic_defender_logs_hmm_training.json")
    ]
    
    results = {}
    
    for log_type, data_file in log_types:
        logger.info(f"\n🚀 Training {log_type} model...")
        
        try:
            analyzer = create_analyzer(log_type)
            
            # Load training data
            if analyzer.load_training_data(data_file, max_sessions=500):
                # Train model
                losses = analyzer.train_model(num_iterations=150, learning_rate=0.01)
                
                if losses:
                    # Save model
                    if analyzer.save_model():
                        results[log_type] = {
                            'status': 'success',
                            'final_loss': losses[-1] if losses else 0,
                            'training_sessions': len(analyzer.session_metadata),
                            'model_path': analyzer.model_path
                        }
                        logger.info(f"✅ {log_type} model training completed")
                    else:
                        results[log_type] = {'status': 'save_failed'}
                else:
                    results[log_type] = {'status': 'training_failed'}
            else:
                results[log_type] = {'status': 'data_load_failed'}
                
        except Exception as e:
            logger.error(f"❌ {log_type} model training failed: {e}")
            results[log_type] = {'status': 'error', 'error': str(e)}
    
    # Save training summary
    with open('training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n📊 Training Summary:")
    for log_type, result in results.items():
        status = result.get('status', 'unknown')
        logger.info(f"  {log_type}: {status}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Production Security Models")
    parser.add_argument("--crowdstrike", default="synthetic_crowdstrike_logs_hmm_training.json",
                       help="CrowdStrike training data file")
    parser.add_argument("--cloudflare", default="synthetic_cloudflare_logs_hmm_training.json", 
                       help="Cloudflare training data file")
    parser.add_argument("--defender", default="synthetic_defender_logs_hmm_training.json",
                       help="Defender training data file")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--max-sessions", type=int, default=500, help="Maximum sessions per model")
    parser.add_argument("--iterations", type=int, default=150, help="Training iterations")
    parser.add_argument("--single", choices=["crowdstrike", "cloudflare", "defender"],
                       help="Train only a single model type")
    
    args = parser.parse_args()
    
    print("🚀 Training Production Security Models")
    print("=" * 50)
    print(f"📁 Model Directory: {args.model_dir}")
    print(f"📊 Max Sessions: {args.max_sessions}")
    print(f"🔄 Iterations: {args.iterations}")
    
    if args.single:
        # Train single model
        print(f"\n🎯 Training single model: {args.single}")
        
        data_file = getattr(args, args.single)
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            sys.exit(1)
        
        try:
            analyzer = create_analyzer(args.single, args.model_dir)
            
            if analyzer.load_training_data(data_file, args.max_sessions):
                losses = analyzer.train_model(args.iterations, 0.01)
                
                if losses and analyzer.save_model():
                    print(f"✅ {args.single} model training completed")
                    print(f"💾 Model saved: {analyzer.model_path}")
                else:
                    print(f"❌ {args.single} model training failed")
                    sys.exit(1)
            else:
                print(f"❌ Failed to load data: {data_file}")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)
    
    else:
        # Train all models
        log_types = [
            ("crowdstrike", args.crowdstrike),
            ("cloudflare", args.cloudflare),
            ("defender", args.defender)
        ]
        
        # Check if all data files exist
        missing_files = []
        for log_type, data_file in log_types:
            if not os.path.exists(data_file):
                missing_files.append(data_file)
        
        if missing_files:
            print(f"❌ Missing data files: {missing_files}")
            print("💡 Available files:")
            for f in os.listdir("."):
                if f.endswith(".json"):
                    print(f"   • {f}")
            sys.exit(1)
        
        results = {}
        
        for log_type, data_file in log_types:
            print(f"\n🚀 Training {log_type} model with {data_file}...")
            
            try:
                analyzer = create_analyzer(log_type, args.model_dir)
                
                if analyzer.load_training_data(data_file, args.max_sessions):
                    losses = analyzer.train_model(args.iterations, 0.01)
                    
                    if losses:
                        if analyzer.save_model():
                            results[log_type] = {
                                'status': 'success',
                                'data_file': data_file,
                                'final_loss': losses[-1] if losses else 0,
                                'training_sessions': len(analyzer.session_metadata),
                                'model_path': analyzer.model_path
                            }
                            print(f"✅ {log_type} model completed")
                        else:
                            results[log_type] = {'status': 'save_failed', 'data_file': data_file}
                    else:
                        results[log_type] = {'status': 'training_failed', 'data_file': data_file}
                else:
                    results[log_type] = {'status': 'data_load_failed', 'data_file': data_file}
                    
            except Exception as e:
                print(f"❌ {log_type} model training failed: {e}")
                results[log_type] = {'status': 'error', 'error': str(e), 'data_file': data_file}
        
        # Save training summary
        summary_file = 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Training Summary:")
        successful = 0
        for log_type, result in results.items():
            status = result.get('status', 'unknown')
            data_file = result.get('data_file', 'unknown')
            print(f"  {log_type}: {status} (data: {data_file})")
            if status == 'success':
                successful += 1
        
        print(f"\n✅ Training completed: {successful}/{len(results)} successful")
        print(f"💾 Models saved in '{args.model_dir}/' directory")
        print(f"📋 Training summary saved in '{summary_file}'")