#!/usr/bin/env python3
"""
Enhanced Pyro-based Security Alert Analysis with Bayesian Regression
Combines HMM and Bayesian regression for robust cybersecurity analysis
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
from dataclasses import dataclass
from collections import defaultdict
import warnings
import time
import signal
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# Check Pyro availability with better error handling
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
    from pyro.optim import Adam, ClippedAdam
    from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
    PYRO_AVAILABLE = True
    pyro.set_rng_seed(42)
except ImportError as e:
    print(f"⚠️ Pyro not available: {e}")
    print("💡 Install with: pip install pyro-ppl")
    PYRO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device with better detection
def get_optimal_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("🔧 Using CPU")
    return device

device = get_optimal_device()

@contextmanager
def timeout_context(seconds):
    """Context manager for timeouts"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

@dataclass
class EnhancedAlertFeatures:
    """Enhanced structured representation of alert features"""
    # Original features
    temporal_anomaly: float
    multi_vector: float
    privilege_escalation: float
    lateral_movement: float
    data_access: float
    persistence: float
    
    # New advanced features
    entropy_score: float
    behavioral_deviation: float
    network_centrality: float
    file_system_anomaly: float
    process_chain_depth: float
    credential_access: float

class BayesianThreatRegressor(nn.Module):
    """Bayesian regression model for threat assessment"""
    
    def __init__(self, feature_dim=12, hidden_dim=64, output_dim=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural network layers
        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return torch.sigmoid(x)  # Attack probability between 0 and 1

class EnhancedPyroHMM(nn.Module):
    """Enhanced Pyro HMM with better numerical stability"""
    
    def __init__(self, num_states=6, feature_dim=12, device=None):
        super().__init__()
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.device = device or get_optimal_device()
        
        self.state_names = [
            "normal_operations",
            "reconnaissance", 
            "initial_access",
            "privilege_escalation",
            "lateral_movement",
            "data_exfiltration"
        ]
        
        # Enhanced state attack weights with more granular scoring
        self.state_attack_weights = {
            0: 0.0,    # normal_operations
            1: 0.25,   # reconnaissance
            2: 0.45,   # initial_access
            3: 0.65,   # privilege_escalation
            4: 0.85,   # lateral_movement
            5: 0.95    # data_exfiltration
        }
        
        self._init_parameters()
        self.trained = False
        self.training_history = []
    
    def _init_parameters(self):
        """Initialize HMM parameters with better priors"""
        # Transition matrix with domain-aware initialization
        trans_init = torch.eye(self.num_states) * 0.7  # Self-transition bias
        trans_init += torch.randn(self.num_states, self.num_states) * 0.1
        self.transition_logits = nn.Parameter(trans_init)
        
        # Initial state with strong normal bias
        initial_bias = torch.full((self.num_states,), -2.0)
        initial_bias[0] = 1.5  # Strong bias towards normal
        self.initial_logits = nn.Parameter(initial_bias)
        
        # Emission parameters with better initialization
        self.emission_means = nn.Parameter(torch.randn(self.num_states, self.feature_dim) * 0.3)
        self.emission_log_scales = nn.Parameter(torch.full((self.num_states, self.feature_dim), -0.5))
        
        # Add learnable concentration parameters for robustness
        self.concentration = nn.Parameter(torch.ones(self.num_states) * 2.0)
        
        self.to(self.device)
    
    def get_transition_probs(self):
        """Get normalized transition probabilities with temperature"""
        return F.softmax(self.transition_logits / 1.0, dim=-1)
    
    def get_initial_probs(self):
        """Get normalized initial state probabilities"""
        return F.softmax(self.initial_logits, dim=-1)
    
    def get_emission_params(self):
        """Get emission parameters with numerical stability"""
        means = self.emission_means
        scales = F.softplus(self.emission_log_scales) + 1e-6  # Ensure positive and stable
        return means, scales
    
    def get_learned_parameters(self):
        """Extract learned parameters for analysis"""
        if not self.trained:
            return {}
        
        transition_probs = self.get_transition_probs().detach().cpu().numpy()
        initial_probs = self.get_initial_probs().detach().cpu().numpy()
        emission_means, emission_scales = self.get_emission_params()
        
        return {
            'transitions': transition_probs.tolist(),
            'initial_probs': initial_probs.tolist(),
            'emission_means': emission_means.detach().cpu().numpy().tolist(),
            'emission_scales': emission_scales.detach().cpu().numpy().tolist()
        }
    
    def predict_attack_probability(self, observations):
        """Predict attack probability for a sequence"""
        if not self.trained:
            logger.warning("HMM not trained, returning default probability")
            return 0.5
        
        try:
            if len(observations) == 0:
                return 0.0
            
            # Get Viterbi path
            predicted_states = self.viterbi_decode(observations)
            
            # Calculate weighted attack probability
            attack_prob = 0.0
            for state in predicted_states:
                state_weight = self.state_attack_weights.get(int(state), 0.0)
                attack_prob += state_weight
            
            # Normalize by sequence length
            attack_prob = attack_prob / len(predicted_states)
            
            return min(max(attack_prob, 0.0), 1.0)
            
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
            
            # Get parameters
            transition_probs = self.get_transition_probs()
            initial_probs = self.get_initial_probs()
            emission_means, emission_scales = self.get_emission_params()
            
            # Initialize Viterbi tables
            viterbi_log_probs = torch.full((seq_len, self.num_states), -float('inf'), device=self.device)
            viterbi_path = torch.zeros((seq_len, self.num_states), dtype=torch.long, device=self.device)
            
            # Initialize first timestep
            for s in range(self.num_states):
                emission_logp = self._compute_emission_logp(observations[0], emission_means[s], emission_scales[s])
                viterbi_log_probs[0, s] = torch.log(initial_probs[s] + 1e-8) + emission_logp
            
            # Forward pass
            for t in range(1, seq_len):
                for s in range(self.num_states):
                    # Transition scores from all previous states
                    trans_scores = viterbi_log_probs[t-1] + torch.log(transition_probs[:, s] + 1e-8)
                    best_prev_state = torch.argmax(trans_scores)
                    
                    # Store best path and probability
                    viterbi_path[t, s] = best_prev_state
                    emission_logp = self._compute_emission_logp(observations[t], emission_means[s], emission_scales[s])
                    viterbi_log_probs[t, s] = trans_scores[best_prev_state] + emission_logp
            
            # Backward pass to reconstruct best path
            path = torch.zeros(seq_len, dtype=torch.long, device=self.device)
            path[-1] = torch.argmax(viterbi_log_probs[-1])
            
            for t in range(seq_len - 2, -1, -1):
                path[t] = viterbi_path[t + 1, path[t + 1]]
            
            return path.cpu()
            
        except Exception as e:
            logger.warning(f"Viterbi decoding failed: {e}")
            return torch.zeros(len(observations), dtype=torch.long)
    
    def model(self, sequences_batch, obs_mask=None):
        """Enhanced Pyro model with better numerical stability"""
        if not PYRO_AVAILABLE:
            raise RuntimeError("Pyro not available")
        
        batch_size = len(sequences_batch)
        max_seq_len = max(len(seq) for seq in sequences_batch)
        
        # Hierarchical priors for better generalization
        transition_concentration = pyro.param("trans_conc", 
                                            torch.ones(self.num_states, self.num_states) * 2.0,
                                            constraint=dist.constraints.positive)
        
        initial_concentration = pyro.param("init_conc", 
                                         torch.ones(self.num_states) * 1.0,
                                         constraint=dist.constraints.positive)
        
        # Sample global parameters from hierarchical priors
        with pyro.plate("state_plate", self.num_states):
            transition_probs = pyro.sample("transition_probs",
                                         dist.Dirichlet(transition_concentration))
            
        initial_probs = pyro.sample("initial_probs",
                                  dist.Dirichlet(initial_concentration))
        
        # Emission parameters with hierarchical priors
        emission_means = pyro.sample("emission_means",
                                   dist.Normal(0, 1).expand([self.num_states, self.feature_dim]).to_event(2))
        
        emission_scales = pyro.sample("emission_scales",
                                    dist.LogNormal(-1, 0.5).expand([self.num_states, self.feature_dim]).to_event(2))
        
        # Process sequences with proper masking
        with pyro.plate("batch", batch_size):
            for seq_idx in range(batch_size):
                seq = sequences_batch[seq_idx]
                seq_len = len(seq)
                
                if seq_len == 0:
                    continue
                
                # Hidden state sequence
                state_seq = []
                
                # Initial state
                state = pyro.sample(f"state_{seq_idx}_0", 
                                  dist.Categorical(initial_probs))
                state_seq.append(state)
                
                # Observations and transitions
                for t in range(seq_len):
                    # Transition (except for first timestep)
                    if t > 0:
                        state = pyro.sample(f"state_{seq_idx}_{t}",
                                          dist.Categorical(transition_probs[state_seq[-1]]))
                        state_seq.append(state)
                    
                    # Emission
                    obs_dist = dist.Normal(emission_means[state], emission_scales[state])
                    pyro.sample(f"obs_{seq_idx}_{t}",
                              obs_dist.to_event(1),
                              obs=seq[t])
    
    def guide(self, sequences_batch, obs_mask=None):
        """Enhanced mean-field variational guide"""
        if not PYRO_AVAILABLE:
            raise RuntimeError("Pyro not available")
        
        batch_size = len(sequences_batch)
        
        # Variational parameters for transition probabilities
        trans_alpha = pyro.param("trans_alpha",
                               torch.ones(self.num_states, self.num_states) * 2.0,
                               constraint=dist.constraints.positive)
        
        init_alpha = pyro.param("init_alpha",
                              torch.ones(self.num_states) * 1.0,
                              constraint=dist.constraints.positive)
        
        # Sample from variational distributions
        with pyro.plate("state_plate", self.num_states):
            pyro.sample("transition_probs", dist.Dirichlet(trans_alpha))
        
        pyro.sample("initial_probs", dist.Dirichlet(init_alpha))
        
        # Emission parameters
        emission_mean_loc = pyro.param("emission_mean_loc",
                                     torch.zeros(self.num_states, self.feature_dim))
        emission_mean_scale = pyro.param("emission_mean_scale",
                                       torch.ones(self.num_states, self.feature_dim) * 0.5,
                                       constraint=dist.constraints.positive)
        
        emission_scale_loc = pyro.param("emission_scale_loc",
                                      torch.zeros(self.num_states, self.feature_dim))
        emission_scale_scale = pyro.param("emission_scale_scale",
                                        torch.ones(self.num_states, self.feature_dim) * 0.3,
                                        constraint=dist.constraints.positive)
        
        pyro.sample("emission_means",
                   dist.Normal(emission_mean_loc, emission_mean_scale).to_event(2))
        pyro.sample("emission_scales",
                   dist.LogNormal(emission_scale_loc, emission_scale_scale).to_event(2))
    
    def fit_enhanced_pyro(self, sequences, num_iterations=300, learning_rate=0.005, patience=20):
        """Enhanced Pyro training with better convergence"""
        logger.info(f"🚀 Enhanced Pyro training on {self.device}")
        
        pyro.clear_param_store()
        
        # Prepare sequences with better preprocessing
        processed_sequences = self._prepare_sequences(sequences, max_sequences=75, max_length=20)
        
        if not processed_sequences:
            logger.warning("No valid sequences for training")
            return []
        
        # Enhanced optimizer with gradient clipping
        optimizer = ClippedAdam({"lr": learning_rate, "clip_norm": 2.0})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
        
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        try:
            with timeout_context(300):  # 5 minute timeout
                for iteration in range(num_iterations):
                    loss = svi.step(processed_sequences)
                    losses.append(loss)
                    
                    # Enhanced convergence monitoring
                    if iteration % 25 == 0:
                        avg_recent_loss = np.mean(losses[-10:]) if len(losses) >= 10 else loss
                        logger.info(f"Iteration {iteration:3d}, Loss: {loss:8.2f}, Avg: {avg_recent_loss:8.2f}")
                    
                    # Improved early stopping
                    if loss < best_loss - 1e-4:  # Require meaningful improvement
                        best_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at iteration {iteration}")
                            break
                    
                    # Check for convergence
                    if len(losses) >= 20:
                        recent_losses = losses[-20:]
                        loss_std = np.std(recent_losses)
                        if loss_std < 1e-3:
                            logger.info(f"Converged at iteration {iteration}")
                            break
                            
        except TimeoutError:
            logger.warning("Training timed out")
        except Exception as e:
            logger.warning(f"Enhanced Pyro training failed: {e}")
            return self.fit_pytorch_fallback(sequences, num_iterations, learning_rate)
        
        self.trained = True
        self.training_history = losses
        logger.info("✅ Enhanced Pyro training completed")
        return losses
    
    def _prepare_sequences(self, sequences, max_sequences=75, max_length=20):
        """Enhanced sequence preparation with better filtering"""
        processed = []
        
        # Sort by length and take diverse sample
        seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
        seq_lengths.sort(key=lambda x: x[1])
        
        # Take sequences of different lengths for diversity
        selected_indices = []
        for i in range(0, len(seq_lengths), max(1, len(seq_lengths) // max_sequences)):
            selected_indices.append(seq_lengths[i][0])
            if len(selected_indices) >= max_sequences:
                break
        
        for idx in selected_indices:
            seq = sequences[idx]
            if torch.is_tensor(seq):
                seq_tensor = seq.to(self.device)
            else:
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            
            # Truncate if too long
            if len(seq_tensor) > max_length:
                seq_tensor = seq_tensor[:max_length]
            
            # Skip if too short or contains invalid values
            if len(seq_tensor) >= 2 and torch.isfinite(seq_tensor).all():
                processed.append(seq_tensor)
        
        logger.info(f"Prepared {len(processed)} sequences for training")
        return processed
    
    def fit_pytorch_fallback(self, sequences, num_iterations=200, learning_rate=0.01):
        """Enhanced PyTorch fallback with better optimization"""
        logger.info("🔧 Using enhanced PyTorch fallback")
        
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
                loss = self._compute_enhanced_loss(seq)
                total_loss += loss
            
            if total_loss > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            
            losses.append(total_loss.item())
            scheduler.step(total_loss)
            
            if iteration % 50 == 0:
                logger.info(f"PyTorch Iteration {iteration}, Loss: {total_loss.item():.4f}")
        
        self.trained = True
        return losses
    
    def _compute_enhanced_loss(self, seq):
        """Enhanced loss computation with regularization"""
        seq_len = len(seq)
        
        transition_probs = self.get_transition_probs()
        initial_probs = self.get_initial_probs()
        emission_means, emission_scales = self.get_emission_params()
        
        # Forward algorithm approximation with numerical stability
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
        
        # Total likelihood
        total_logp = torch.logsumexp(log_alpha[-1], dim=0)
        
        # Add regularization
        reg_loss = 0.01 * (transition_probs.var() + emission_scales.mean())
        
        return -(total_logp - reg_loss)
    
    def _compute_emission_logp(self, obs, mean, scale):
        """Compute emission log probability with numerical stability"""
        diff = obs - mean
        return -0.5 * ((diff / (scale + 1e-6)) ** 2).sum() - scale.log().sum()
    
    def fit(self, sequences, num_iterations=300, learning_rate=0.005):
        """Enhanced training dispatcher"""
        if not PYRO_AVAILABLE:
            return self.fit_pytorch_fallback(sequences, num_iterations, learning_rate)
        
        try:
            return self.fit_enhanced_pyro(sequences, num_iterations, learning_rate)
        except Exception as e:
            logger.warning(f"Pyro training failed: {e}, using PyTorch fallback")
            return self.fit_pytorch_fallback(sequences, num_iterations, learning_rate)

class AdvancedAlertProcessor:
    """Enhanced alert processor with more sophisticated feature extraction"""
    
    def __init__(self):
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 'psexec.exe',
            'rundll32.exe', 'regsvr32.exe', 'certutil.exe', 'bitsadmin.exe',
            'mshta.exe', 'cscript.exe', 'wscript.exe'
        }
        self.suspicious_ports = {135, 139, 445, 3389, 22, 1433, 1521, 5985, 5986}
        self.internal_networks = ['10.', '192.168.', '172.16.', '172.17.', '172.18.']
        self.sensitive_extensions = {'.docx', '.pdf', '.xlsx', '.csv', '.txt', '.db', '.sql', '.key', '.pem'}
        
    def extract_enhanced_features(self, alerts: List[Dict]) -> EnhancedAlertFeatures:
        """Extract enhanced features with more sophisticated analysis"""
        if not alerts:
            return EnhancedAlertFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Original features
        temporal_anomaly = self._analyze_temporal_patterns(alerts)
        multi_vector = self._calculate_multi_vector_score(alerts)
        privilege_escalation = self._detect_privilege_escalation(alerts)
        lateral_movement = self._detect_lateral_movement(alerts)
        data_access = self._detect_data_access(alerts)
        persistence = self._detect_persistence(alerts)
        
        # New enhanced features
        entropy_score = self._calculate_entropy_score(alerts)
        behavioral_deviation = self._calculate_behavioral_deviation(alerts)
        network_centrality = self._calculate_network_centrality(alerts)
        file_system_anomaly = self._detect_file_system_anomalies(alerts)
        process_chain_depth = self._analyze_process_chains(alerts)
        credential_access = self._detect_credential_access(alerts)
        
        return EnhancedAlertFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector,
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
    
    def _analyze_temporal_patterns(self, alerts: List[Dict]) -> float:
        """Enhanced temporal analysis"""
        timestamps = []
        for alert in alerts:
            try:
                ts = datetime.fromisoformat(alert['timestamp'])
                timestamps.append(ts)
            except (ValueError, KeyError):
                continue
        
        if len(timestamps) < 2:
            return 0.0
        
        timestamps.sort()
        
        # Multiple temporal features
        off_hours_score = sum(1 for ts in timestamps 
                             if ts.weekday() >= 5 or ts.hour >= 22 or ts.hour <= 6) / len(timestamps)
        
        # Burst detection
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        burst_score = sum(1 for diff in time_diffs if diff < 30) / max(len(time_diffs), 1)
        
        # Regularity analysis
        if len(time_diffs) > 5:
            regularity_score = 1.0 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-6))
            regularity_score = max(0, min(1, regularity_score))
        else:
            regularity_score = 0.0
        
        return min((off_hours_score * 0.4 + burst_score * 0.4 + regularity_score * 0.2), 1.0)
    
    def _calculate_entropy_score(self, alerts: List[Dict]) -> float:
        """Calculate information entropy of alert patterns"""
        if len(alerts) < 2:
            return 0.0
        
        # Event type entropy
        event_types = [alert.get('event_type', 'unknown') for alert in alerts]
        type_counts = {}
        for et in event_types:
            type_counts[et] = type_counts.get(et, 0) + 1
        
        type_entropy = 0.0
        total = len(event_types)
        for count in type_counts.values():
            p = count / total
            if p > 0:
                type_entropy -= p * np.log2(p)
        
        # Severity entropy
        severities = [alert.get('severity', 'unknown') for alert in alerts]
        sev_counts = {}
        for sev in severities:
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        
        sev_entropy = 0.0
        for count in sev_counts.values():
            p = count / total
            if p > 0:
                sev_entropy -= p * np.log2(p)
        
        # Normalize entropies
        max_type_entropy = np.log2(min(len(type_counts), 8))  # Cap at reasonable max
        max_sev_entropy = np.log2(min(len(sev_counts), 4))
        
        normalized_entropy = (type_entropy / max(max_type_entropy, 1e-6) + 
                            sev_entropy / max(max_sev_entropy, 1e-6)) / 2
        
        return min(normalized_entropy, 1.0)
    
    def _calculate_behavioral_deviation(self, alerts: List[Dict]) -> float:
        """Calculate deviation from normal behavior patterns"""
        if len(alerts) < 3:
            return 0.0
        
        deviation_indicators = 0
        total_checks = 0
        
        # Process behavior analysis
        process_alerts = [a for a in alerts if a.get('event_type') == 'Process']
        if process_alerts:
            process_files = [a.get('details', {}).get('file_name', '').lower() 
                           for a in process_alerts]
            suspicious_ratio = sum(1 for pf in process_files 
                                 if any(sp in pf for sp in self.suspicious_processes))
            deviation_indicators += suspicious_ratio / len(process_files)
            total_checks += 1
        
        # Network behavior analysis
        network_alerts = [a for a in alerts if a.get('event_type') == 'Network']
        if network_alerts:
            unusual_ports = sum(1 for na in network_alerts
                              if na.get('details', {}).get('remote_port', 0) in self.suspicious_ports)
            deviation_indicators += unusual_ports / len(network_alerts)
            total_checks += 1
        
        # File access patterns
        file_alerts = [a for a in alerts if a.get('event_type') == 'File']
        if file_alerts:
            sensitive_files = sum(1 for fa in file_alerts
                                if any(ext in fa.get('details', {}).get('file_name', '').lower()
                                     for ext in self.sensitive_extensions))
            deviation_indicators += sensitive_files / len(file_alerts)
            total_checks += 1
        
        return min(deviation_indicators / max(total_checks, 1), 1.0)
    
    def _calculate_network_centrality(self, alerts: List[Dict]) -> float:
        """Calculate network centrality score"""
        network_alerts = [a for a in alerts if a.get('event_type') == 'Network']
        if not network_alerts:
            return 0.0
        
        # Analyze unique connections
        connections = set()
        internal_connections = 0
        
        for alert in network_alerts:
            details = alert.get('details', {})
            remote_ip = details.get('remote_ip', '')
            remote_port = details.get('remote_port', 0)
            
            if remote_ip and remote_port:
                connections.add((remote_ip, remote_port))
                
                # Check if internal connection
                if any(remote_ip.startswith(net) for net in self.internal_networks):
                    internal_connections += 1
        
        # Calculate centrality metrics
        unique_ratio = len(connections) / len(network_alerts)
        internal_ratio = internal_connections / len(network_alerts)
        
        # High centrality indicates potential lateral movement
        centrality_score = (unique_ratio * 0.6 + internal_ratio * 0.4)
        return min(centrality_score, 1.0)
    
    def _detect_file_system_anomalies(self, alerts: List[Dict]) -> float:
        """Detect file system anomalies"""
        file_alerts = [a for a in alerts if a.get('event_type') == 'File']
        if not file_alerts:
            return 0.0
        
        anomaly_score = 0.0
        
        # Check for suspicious file operations
        for alert in file_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            file_path = details.get('file_path', '').lower()
            action = details.get('action', '')
            
            # Temporary directory operations
            if any(temp in file_path for temp in ['temp', 'tmp', 'appdata']):
                anomaly_score += 0.3
            
            # System directory modifications
            if any(sys in file_path for sys in ['system32', 'syswow64', 'windows']):
                anomaly_score += 0.5
            
            # Suspicious file extensions
            if any(ext in file_name for ext in ['.exe', '.dll', '.bat', '.ps1', '.vbs']):
                anomaly_score += 0.2
            
            # File deletion patterns
            if action in ['FileDeleted', 'FileShredded']:
                anomaly_score += 0.4
        
        return min(anomaly_score / len(file_alerts), 1.0)
    
    def _analyze_process_chains(self, alerts: List[Dict]) -> float:
        """Analyze process execution chains"""
        process_alerts = [a for a in alerts if a.get('event_type') == 'Process']
        if not process_alerts:
            return 0.0
        
        # Build process tree
        process_tree = {}
        max_depth = 0
        
        for alert in process_alerts:
            details = alert.get('details', {})
            process_id = details.get('process_id', 0)
            parent_id = details.get('parent_process_id', 0)
            
            if process_id:
                process_tree[process_id] = parent_id
        
        # Calculate maximum depth
        def get_depth(pid, visited=None):
            if visited is None:
                visited = set()
            if pid in visited or pid == 0:
                return 0
            visited.add(pid)
            parent = process_tree.get(pid, 0)
            return 1 + get_depth(parent, visited)
        
        for pid in process_tree:
            depth = get_depth(pid)
            max_depth = max(max_depth, depth)
        
        # Normalize depth score
        return min(max_depth / 10.0, 1.0)  # Cap at depth 10
    
    def _detect_credential_access(self, alerts: List[Dict]) -> float:
        """Detect credential access attempts"""
        credential_indicators = 0
        total_alerts = len(alerts)
        
        credential_keywords = [
            'lsass', 'sam', 'security', 'password', 'credential', 'token',
            'mimikatz', 'sekurlsa', 'logonpasswords', 'dcsync'
        ]
        
        for alert in alerts:
            details = alert.get('details', {})
            command_line = details.get('command_line', '').lower()
            file_name = details.get('file_name', '').lower()
            
            # Check for credential-related activities
            if any(keyword in command_line for keyword in credential_keywords):
                credential_indicators += 1
            
            if any(keyword in file_name for keyword in credential_keywords):
                credential_indicators += 0.5
            
            # Registry access to credential stores
            if 'registry' in command_line and any(cred in command_line for cred in ['sam', 'security', 'software']):
                credential_indicators += 0.7
        
        return min(credential_indicators / max(total_alerts, 1), 1.0)
    
    def _calculate_multi_vector_score(self, alerts: List[Dict]) -> float:
        """Enhanced multi-vector attack detection"""
        event_types = set(alert.get('event_type', 'unknown') for alert in alerts)
        severities = set(alert.get('severity', 'unknown') for alert in alerts)
        
        # Weighted scoring based on attack complexity
        type_score = min(len(event_types) / 4.0, 1.0)  # Normalize to 4 types
        severity_score = min(len(severities) / 3.0, 1.0)  # Normalize to 3 severities
        
        return (type_score * 0.7 + severity_score * 0.3)
    
    def _detect_privilege_escalation(self, alerts: List[Dict]) -> float:
        """Enhanced privilege escalation detection"""
        escalation_score = 0.0
        process_alerts = [a for a in alerts if a.get('event_type') == 'Process']
        
        if not process_alerts:
            return 0.0
        
        for alert in process_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            command_line = details.get('command_line', '').lower()
            
            # Enhanced detection patterns
            if any(proc in file_name for proc in self.suspicious_processes):
                escalation_score += 0.4
            
            escalation_keywords = [
                'runas', 'administrator', 'elevated', 'uac', 'bypass',
                'token', 'impersonate', 'privilege', 'system'
            ]
            
            if any(keyword in command_line for keyword in escalation_keywords):
                escalation_score += 0.6
            
            # Service creation/modification
            if any(svc in command_line for svc in ['sc create', 'sc config', 'net localgroup']):
                escalation_score += 0.8
        
        return min(escalation_score / len(process_alerts), 1.0)
    
    def _detect_lateral_movement(self, alerts: List[Dict]) -> float:
        """Enhanced lateral movement detection"""
        network_alerts = [a for a in alerts if a.get('event_type') == 'Network']
        if not network_alerts:
            return 0.0
        
        lateral_score = 0.0
        internal_connections = 0
        admin_ports = 0
        
        for alert in network_alerts:
            details = alert.get('details', {})
            remote_ip = details.get('remote_ip', '')
            remote_port = details.get('remote_port', 0)
            direction = details.get('direction', '')
            
            # Internal network connections
            if any(remote_ip.startswith(net) for net in self.internal_networks):
                internal_connections += 1
                
                # Administrative ports
                if remote_port in {135, 139, 445, 3389, 5985, 5986}:
                    admin_ports += 1
                
                # Multiple internal targets
                if direction == 'outbound':
                    lateral_score += 0.3
        
        if len(network_alerts) > 0:
            internal_ratio = internal_connections / len(network_alerts)
            admin_ratio = admin_ports / len(network_alerts)
            
            lateral_score = (internal_ratio * 0.5 + admin_ratio * 0.5)
        
        return min(lateral_score, 1.0)
    
    def _detect_data_access(self, alerts: List[Dict]) -> float:
        """Enhanced data access detection"""
        file_alerts = [a for a in alerts if a.get('event_type') == 'File']
        if not file_alerts:
            return 0.0
        
        data_score = 0.0
        
        for alert in file_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            file_path = details.get('file_path', '').lower()
            action = details.get('action', '')
            
            # Sensitive file extensions
            if any(ext in file_name for ext in self.sensitive_extensions):
                data_score += 0.5
            
            # Sensitive directories
            sensitive_dirs = ['documents', 'desktop', 'downloads', 'pictures']
            if any(dir_name in file_path for dir_name in sensitive_dirs):
                data_score += 0.3
            
            # Data exfiltration indicators
            if action in ['FileCopied', 'FileAccessed', 'FileRead']:
                data_score += 0.2
            
            # Large file operations
            file_size = details.get('file_size', 0)
            if file_size > 1024 * 1024:  # Files > 1MB
                data_score += 0.2
        
        return min(data_score / len(file_alerts), 1.0)
    
    def _detect_persistence(self, alerts: List[Dict]) -> float:
        """Enhanced persistence detection"""
        persistence_score = 0.0
        total_alerts = len(alerts)
        
        persistence_indicators = [
            # Registry persistence
            'run', 'runonce', 'startup', 'logon', 'winlogon',
            # Service persistence  
            'service', 'schtasks', 'at.exe',
            # File system persistence
            'autostart', 'autorun', 'autoexec'
        ]
        
        for alert in alerts:
            event_type = alert.get('event_type', '')
            details = alert.get('details', {})
            
            if event_type == 'Process':
                command_line = details.get('command_line', '').lower()
                if any(indicator in command_line for indicator in persistence_indicators):
                    persistence_score += 0.8
            
            elif event_type == 'File':
                file_path = details.get('file_path', '').lower()
                if any(indicator in file_path for indicator in persistence_indicators):
                    persistence_score += 0.6
            
            elif event_type == 'Registry':
                reg_key = details.get('registry_key', '').lower()
                if any(indicator in reg_key for indicator in persistence_indicators):
                    persistence_score += 0.9
        
        return min(persistence_score / max(total_alerts, 1), 1.0)

class EnhancedBayesianAnalyzer:
    """Enhanced Bayesian analyzer combining HMM and regression"""
    
    def __init__(self, device=None):
        self.device = device or get_optimal_device()
        self.processor = AdvancedAlertProcessor()
        self.hmm = EnhancedPyroHMM(device=self.device)
        self.regressor = BayesianThreatRegressor().to(self.device)
        
        self.sessions_data = None
        self.alerts_data = None
        self.feature_sequences = []
        self.session_metadata = []
        self.regression_trained = False
    
    def load_json_data(self, json_file: str, max_sessions: int = 300):
        """Load real session data from JSON file - fixed based on working version"""
        logger.info(f"📁 Loading real session data from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"🔍 JSON structure: {type(data).__name__}")
            
            # Handle session dictionary format (your structure)
            if isinstance(data, dict):
                logger.info(f"🗝️ Object with {len(data)} keys")
                
                # Check if this is a dictionary of sessions
                sample_keys = list(data.keys())[:3]
                is_session_dict = False
                
                for key in sample_keys:
                    value = data[key]
                    if isinstance(value, dict) and ('session_id' in value or 'alerts' in value):
                        is_session_dict = True
                        break
                
                if is_session_dict:
                    logger.info(f"✅ Detected session dictionary format with {len(data)} sessions")
                    sessions = list(data.values())
                else:
                    sessions = [data]
            elif isinstance(data, list):
                logger.info(f"📋 Direct array with {len(data)} elements")
                sessions = data
            else:
                raise ValueError(f"Unsupported JSON root type: {type(data)}")
            
            logger.info(f"✅ Found {len(sessions)} potential sessions")
            
            # Limit sessions for memory management
            if len(sessions) > max_sessions:
                logger.info(f"🔄 Limiting to {max_sessions} sessions for memory management")
                sessions = sessions[:max_sessions]
            
            # Initialize storage
            self.sessions_data = {}
            self.alerts_data = []
            
            processed_count = 0
            failed_count = 0
            
            for i, session in enumerate(sessions):
                try:
                    # Extract session ID
                    session_id = session.get('session_id', f"session_{i:04d}")
                    
                    # Get alerts directly from session
                    alerts = session.get('alerts', [])
                    
                    if not alerts or len(alerts) == 0:
                        logger.debug(f"⚠️ No alerts found in session {session_id}")
                        failed_count += 1
                        continue
                    
                    # Use pre-calculated values from your data structure
                    is_attack = session.get('is_attack_session', False)
                    risk_score = session.get('risk_score', 10)
                    avg_attack_intensity = session.get('avg_attack_intensity', 0.0)
                    
                    # Convert risk score to attack probability
                    attack_probability = min(risk_score / 100.0, 1.0)
                    if avg_attack_intensity > 0:
                        attack_probability = min(attack_probability + (avg_attack_intensity * 0.2), 1.0)
                    
                    # Store session data
                    self.sessions_data[session_id] = {
                        'alerts': alerts,
                        'is_attack_session': is_attack,
                        'risk_score': risk_score,
                        'alert_count': len(alerts),
                        'ground_truth_attack_prob': attack_probability,
                        'original_session': session
                    }
                    
                    self.alerts_data.extend(alerts)
                    processed_count += 1
                    
                    # Progress feedback
                    if processed_count % 100 == 0:
                        logger.info(f"📊 Processed {processed_count} sessions...")
                    
                    # Early success logging for debugging
                    if processed_count == 1:
                        logger.info(f"✅ Successfully processed first session: {session_id}")
                        logger.info(f"   • Alerts: {len(alerts)}")
                        logger.info(f"   • Attack: {is_attack}")
                        logger.info(f"   • Risk Score: {risk_score}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process session {i}: {e}")
                    failed_count += 1
                    
                    # For first few failures, show more detail
                    if failed_count <= 3:
                        logger.warning(f"   Session keys: {list(session.keys()) if isinstance(session, dict) else 'Not a dict'}")
                        logger.warning(f"   Error details: {str(e)}")
                    
                    continue
            
            logger.info(f"🔄 Successfully processed {processed_count} real sessions")
            if failed_count > 0:
                logger.info(f"⚠️ Failed to process {failed_count} sessions")
            
            if processed_count == 0:
                logger.error("❌ No sessions could be processed successfully")
                raise ValueError("No sessions could be processed from the JSON file")
            
        except Exception as e:
            logger.error(f"❌ Failed to load JSON data: {e}")
            raise
    
    def _extract_alerts_from_session(self, session: Dict) -> List[Dict]:
        """Extract and normalize alerts from real session data with better error handling"""
        alerts = []
        
        # For your specific format, alerts are directly in the 'alerts' key
        if 'alerts' in session and isinstance(session['alerts'], list):
            raw_alerts = session['alerts']
            logger.debug(f"Found {len(raw_alerts)} alerts in 'alerts' key")
        else:
            # Fallback to other strategies if alerts not in expected location
            alert_containers = ['events', 'logs', 'data', 'items', 'records']
            
            raw_alerts = None
            container_used = None
            
            for container in alert_containers:
                if container in session:
                    candidate = session[container]
                    if isinstance(candidate, list) and len(candidate) > 0:
                        raw_alerts = candidate
                        container_used = container
                        break
            
            if not raw_alerts:
                # Look for any list that might contain alerts
                for key, value in session.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict):
                            # Check if dict has alert-like fields
                            sample_keys = set(value[0].keys())
                            alert_indicators = {'timestamp', 'time', 'event', 'event_type', 'type', 'message', 'severity', 'level'}
                            if any(indicator in sample_keys for indicator in alert_indicators):
                                raw_alerts = value
                                container_used = key
                                break
            
            if not raw_alerts:
                logger.debug(f"No alerts found in session. Available keys: {list(session.keys()) if isinstance(session, dict) else 'Not a dict'}")
                return []
            
            logger.debug(f"Found {len(raw_alerts)} alerts in container '{container_used}'")
        
        # Process and normalize alerts
        for i, alert in enumerate(raw_alerts):
            try:
                # Your alerts are already in good format, just need minimal normalization
                normalized_alert = self._normalize_alert_structure(alert)
                if normalized_alert:
                    alerts.append(normalized_alert)
            except Exception as e:
                logger.debug(f"Failed to normalize alert {i}: {e}")
                continue
        
        logger.debug(f"Successfully normalized {len(alerts)} alerts")
        return alerts
    
    def _normalize_alert_structure(self, alert: Dict) -> Dict:
        """Normalize alert structure to standard format - optimized for your format"""
        normalized = {}
        
        # Your format already has good structure, just pass through with minimal changes
        normalized['timestamp'] = alert.get('timestamp', datetime.now().isoformat())
        normalized['event_type'] = alert.get('event_type', 'Unknown')
        
        # Handle severity - your format may not have it, so derive from event_type
        if 'severity' in alert:
            normalized['severity'] = alert['severity']
        else:
            # Derive severity from event type and details
            event_type = normalized['event_type'].lower()
            details = alert.get('details', {})
            
            # Check for suspicious indicators to assign higher severity
            if event_type == 'process':
                file_name = details.get('file_name', '').lower()
                cmd_line = details.get('command_line', '').lower()
                if any(proc in file_name for proc in ['powershell', 'cmd', 'wmic', 'net.exe']):
                    normalized['severity'] = 'High'
                elif any(sus in cmd_line for sus in ['bypass', 'hidden', 'encoded']):
                    normalized['severity'] = 'Critical'
                else:
                    normalized['severity'] = 'Medium'
            elif event_type == 'network':
                remote_port = details.get('remote_port', 0)
                if remote_port in [445, 3389, 22, 135]:  # Suspicious ports
                    normalized['severity'] = 'High'
                else:
                    normalized['severity'] = 'Medium'
            elif event_type == 'file':
                action = details.get('action', '')
                if action in ['FileDeleted', 'FileModified']:
                    normalized['severity'] = 'Medium'
                else:
                    normalized['severity'] = 'Low'
            else:
                normalized['severity'] = 'Medium'
        
        # Copy details directly - your format is already good
        normalized['details'] = alert.get('details', {})
        
        # Copy additional useful fields
        for field in ['device_name', 'user_name']:
            if field in alert:
                normalized['details'][field] = alert[field]
        
        return normalized
    
    def _extract_alerts_from_session(self, session: Dict) -> List[Dict]:
        """Extract and normalize alerts from real session data"""
        alerts = []
        
        # Handle different session structures
        if 'alerts' in session:
            raw_alerts = session['alerts']
        elif 'events' in session:
            raw_alerts = session['events']
        elif 'logs' in session:
            raw_alerts = session['logs']
        else:
            # Try to find alert-like data in the session
            for key, value in session.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        raw_alerts = value
                        break
            else:
                return []
        
        for alert in raw_alerts:
            try:
                # Normalize alert structure
                normalized_alert = self._normalize_alert_structure(alert)
                if normalized_alert:
                    alerts.append(normalized_alert)
            except Exception as e:
                logger.debug(f"Failed to normalize alert: {e}")
                continue
        
        return alerts
    
    def _normalize_alert_structure(self, alert: Dict) -> Dict:
        """Normalize alert structure to standard format"""
        normalized = {}
        
        # Extract timestamp
        timestamp_fields = ['timestamp', 'time', 'event_time', 'created_at', 'date']
        for field in timestamp_fields:
            if field in alert:
                try:
                    if isinstance(alert[field], str):
                        # Try to parse timestamp
                        normalized['timestamp'] = alert[field]
                    else:
                        normalized['timestamp'] = str(alert[field])
                    break
                except:
                    continue
        else:
            # Default timestamp if none found
            normalized['timestamp'] = datetime.now().isoformat()
        
        # Extract event type
        event_type_fields = ['event_type', 'type', 'category', 'log_type']
        for field in event_type_fields:
            if field in alert:
                event_type = str(alert[field])
                # Map common event types
                if any(proc in event_type.lower() for proc in ['process', 'exec', 'command']):
                    normalized['event_type'] = 'Process'
                elif any(net in event_type.lower() for net in ['network', 'connection', 'traffic']):
                    normalized['event_type'] = 'Network'
                elif any(file in event_type.lower() for file in ['file', 'filesystem', 'io']):
                    normalized['event_type'] = 'File'
                elif any(reg in event_type.lower() for reg in ['registry', 'reg']):
                    normalized['event_type'] = 'Registry'
                else:
                    normalized['event_type'] = event_type
                break
        else:
            normalized['event_type'] = 'Unknown'
        
        # Extract severity
        severity_fields = ['severity', 'level', 'priority', 'criticality']
        for field in severity_fields:
            if field in alert:
                severity = str(alert[field])
                # Normalize severity levels
                if any(crit in severity.lower() for crit in ['critical', 'high', 'severe']):
                    normalized['severity'] = 'Critical'
                elif any(med in severity.lower() for med in ['medium', 'moderate', 'warning']):
                    normalized['severity'] = 'Medium'
                elif any(low in severity.lower() for low in ['low', 'info', 'informational']):
                    normalized['severity'] = 'Low'
                else:
                    normalized['severity'] = severity
                break
        else:
            normalized['severity'] = 'Medium'
        
        # Extract details
        details = {}
        detail_fields = ['details', 'data', 'metadata', 'fields', 'properties']
        for field in detail_fields:
            if field in alert and isinstance(alert[field], dict):
                details.update(alert[field])
                break
        
        # Extract common detail fields directly from alert
        common_fields = [
            'file_name', 'filename', 'process_name', 'command_line', 'cmd',
            'remote_ip', 'src_ip', 'dst_ip', 'remote_port', 'src_port', 'dst_port',
            'file_path', 'path', 'registry_key', 'key', 'action', 'operation'
        ]
        
        for field in common_fields:
            if field in alert:
                details[field] = alert[field]
        
        normalized['details'] = details
        
        return normalized
    
    def _analyze_real_session_indicators(self, session: Dict, alerts: List[Dict]) -> Dict:
        """Analyze real session data to determine attack indicators - optimized for your pre-analyzed data"""
        # Your data already has excellent attack analysis! Use it directly
        if 'is_attack_session' in session and 'risk_score' in session:
            # Use the pre-calculated values from your data
            is_attack = session.get('is_attack_session', False)
            risk_score = session.get('risk_score', 10)
            
            # Convert risk_score to attack_probability
            attack_probability = min(risk_score / 100.0, 1.0)
            
            # Use additional indicators if available
            if 'avg_attack_intensity' in session:
                intensity = session.get('avg_attack_intensity', 0.0)
                # Boost probability based on intensity
                attack_probability = min(attack_probability + (intensity * 0.2), 1.0)
            
            return {
                'is_attack': is_attack,
                'attack_probability': attack_probability,
                'risk_score': risk_score
            }
        
        # Fallback to manual analysis if pre-calculated values not available
        indicators = {
            'is_attack': False,
            'attack_probability': 0.0,
            'risk_score': 10
        }
        
        if not alerts:
            return indicators
        
        try:
            # Count suspicious indicators
            suspicious_count = 0
            total_alerts = len(alerts)
            
            # Analyze event types
            event_types = [alert.get('event_type', '') for alert in alerts]
            process_events = sum(1 for et in event_types if 'process' in et.lower())
            network_events = sum(1 for et in event_types if 'network' in et.lower())
            file_events = sum(1 for et in event_types if 'file' in et.lower())
            
            # Multi-vector indicator
            if len(set(event_types)) >= 3:
                suspicious_count += 2
            
            # High severity alerts
            high_severity = sum(1 for alert in alerts 
                               if alert.get('severity', '').lower() in ['critical', 'high'])
            if high_severity > total_alerts * 0.3:
                suspicious_count += 3
            
            # Suspicious processes
            suspicious_processes = {
                'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 'psexec.exe',
                'rundll32.exe', 'regsvr32.exe', 'certutil.exe', 'mimikatz'
            }
            
            for alert in alerts:
                details = alert.get('details', {})
                file_name = details.get('file_name', '').lower()
                command_line = details.get('command_line', '').lower()
                
                if any(proc in file_name for proc in suspicious_processes):
                    suspicious_count += 1
                
                if any(proc in command_line for proc in suspicious_processes):
                    suspicious_count += 1
            
            # Network indicators
            internal_networks = ['10.', '192.168.', '172.16.']
            suspicious_ports = {3389, 445, 135, 139, 22, 1433}
            
            for alert in alerts:
                if alert.get('event_type') == 'Network':
                    details = alert.get('details', {})
                    remote_ip = details.get('remote_ip', '')
                    remote_port = details.get('remote_port', 0)
                    
                    if any(remote_ip.startswith(net) for net in internal_networks):
                        suspicious_count += 0.5
                    
                    if remote_port in suspicious_ports:
                        suspicious_count += 1
            
            # Calculate scores
            suspicion_ratio = suspicious_count / max(total_alerts, 1)
            
            indicators['attack_probability'] = min(suspicion_ratio * 0.4, 1.0)
            indicators['risk_score'] = min(10 + (suspicious_count * 15), 100)
            indicators['is_attack'] = indicators['attack_probability'] > 0.3 or indicators['risk_score'] > 60
            
        except Exception as e:
            logger.debug(f"Error in manual attack analysis: {e}")
            # Return safe defaults
            pass
        
        return indicators
    
    def load_csv_data(self, csv_file: str):
        """Enhanced CSV data loading (legacy support)"""
        logger.info(f"📁 Loading enhanced data from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"✅ Loaded {len(df)} sessions from CSV")
            
            self.sessions_data = {}
            self.alerts_data = []
            
            for _, row in df.iterrows():
                session_id = row['session_id']
                
                # Enhanced synthetic alert generation
                alerts = self._create_enhanced_synthetic_alerts(row)
                
                # Determine attack status with better logic
                attack_prob = float(row.get('attack_probability', 0))
                is_attack = (attack_prob > 0.5 or 
                           row.get('original_risk_score', 0) > 80)
                
                self.sessions_data[session_id] = {
                    'alerts': alerts,
                    'is_attack_session': is_attack,
                    'risk_score': row.get('original_risk_score', 0),
                    'alert_count': row.get('alert_count', len(alerts)),
                    'ground_truth_attack_prob': attack_prob
                }
                
                self.alerts_data.extend(alerts)
            
            logger.info(f"🔄 Converted to {len(self.sessions_data)} sessions with enhanced features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV data: {e}")
            raise
    
    def _create_enhanced_synthetic_alerts(self, row) -> List[Dict]:
        """Create enhanced synthetic alerts with realistic patterns"""
        alerts = []
        session_id = row['session_id']
        alert_count = min(int(row.get('alert_count', 8)), 20)
        attack_prob = float(row.get('attack_probability', 0))
        risk_score = int(row.get('original_risk_score', 50))
        
        base_time = datetime.now()
        
        # Determine attack sophistication
        if attack_prob > 0.8:
            attack_type = "advanced_persistent"
        elif attack_prob > 0.5:
            attack_type = "targeted_attack"
        elif attack_prob > 0.2:
            attack_type = "opportunistic"
        else:
            attack_type = "normal"
        
        for i in range(alert_count):
            # Realistic time distribution
            time_offset = np.random.exponential(300) if attack_type != "normal" else np.random.uniform(60, 3600)
            alert_time = base_time + pd.Timedelta(seconds=i * time_offset)
            
            alert = self._generate_alert_by_type(alert_time, i, session_id, attack_type, risk_score)
            alerts.append(alert)
        
        return alerts
    
    def _generate_alert_by_type(self, timestamp: datetime, index: int, session_id: str, 
                               attack_type: str, risk_score: int) -> Dict:
        """Generate realistic alerts based on attack type"""
        
        if attack_type == "advanced_persistent":
            return self._create_apt_alert(timestamp, index, session_id)
        elif attack_type == "targeted_attack":
            return self._create_targeted_alert(timestamp, index, session_id)
        elif attack_type == "opportunistic":
            return self._create_opportunistic_alert(timestamp, index, session_id)
        else:
            return self._create_normal_alert(timestamp, index, session_id)
    
    def _create_apt_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create APT-style alerts"""
        apt_patterns = [
            {
                'event_type': 'Process',
                'severity': 'Critical',
                'details': {
                    'file_name': 'powershell.exe',
                    'command_line': 'powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -EncodedCommand',
                    'process_id': 2000 + index,
                    'parent_process_id': 1500,
                    'integrity_level': 'High'
                }
            },
            {
                'event_type': 'Network',
                'severity': 'High',
                'details': {
                    'remote_ip': f'10.0.{index % 50}.{100 + index % 155}',
                    'remote_port': 445,
                    'direction': 'outbound',
                    'protocol': 'SMB',
                    'bytes_transferred': 1024 * (50 + index * 10)
                }
            },
            {
                'event_type': 'File',
                'severity': 'High',
                'details': {
                    'file_name': f'sensitive_data_{index}.docx',
                    'action': 'FileAccessed',
                    'file_path': 'C:\\Users\\admin\\Documents\\Confidential\\',
                    'file_size': 1024 * 1024 * 2  # 2MB
                }
            },
            {
                'event_type': 'Registry',
                'severity': 'High',
                'details': {
                    'registry_key': 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run',
                    'action': 'RegistryValueSet',
                    'value_name': f'SecurityUpdate{index}',
                    'value_data': 'C:\\Windows\\Temp\\update.exe'
                }
            }
        ]
        
        pattern = apt_patterns[index % len(apt_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        
        return pattern
    
    def _create_targeted_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create targeted attack alerts"""
        targeted_patterns = [
            {
                'event_type': 'Process',
                'severity': 'High',
                'details': {
                    'file_name': 'cmd.exe',
                    'command_line': 'cmd.exe /c net user administrator /active:yes',
                    'process_id': 3000 + index,
                    'parent_process_id': 2500
                }
            },
            {
                'event_type': 'Network',
                'severity': 'Medium',
                'details': {
                    'remote_ip': f'192.168.1.{50 + index % 50}',
                    'remote_port': 3389,
                    'direction': 'inbound',
                    'protocol': 'RDP'
                }
            }
        ]
        
        pattern = targeted_patterns[index % len(targeted_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        
        return pattern
    
    def _create_opportunistic_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create opportunistic attack alerts"""
        opportunistic_patterns = [
            {
                'event_type': 'Process',
                'severity': 'Medium',
                'details': {
                    'file_name': 'rundll32.exe',
                    'command_line': 'rundll32.exe shell32.dll,ShellExec_RunDLL',
                    'process_id': 4000 + index
                }
            },
            {
                'event_type': 'File',
                'severity': 'Medium',
                'details': {
                    'file_name': f'temp_file_{index}.exe',
                    'action': 'FileCreated',
                    'file_path': 'C:\\Windows\\Temp\\'
                }
            }
        ]
        
        pattern = opportunistic_patterns[index % len(opportunistic_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        
        return pattern
    
    def _create_normal_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create normal operational alerts"""
        normal_patterns = [
            {
                'event_type': 'Process',
                'severity': 'Low',
                'details': {
                    'file_name': 'notepad.exe',
                    'command_line': 'notepad.exe document.txt',
                    'process_id': 5000 + index
                }
            },
            {
                'event_type': 'File',
                'severity': 'Informational',
                'details': {
                    'file_name': f'report_{index}.pdf',
                    'action': 'FileOpened',
                    'file_path': 'C:\\Users\\user\\Documents\\'
                }
            },
            {
                'event_type': 'Network',
                'severity': 'Informational',
                'details': {
                    'remote_ip': '8.8.8.8',
                    'remote_port': 80,
                    'direction': 'outbound',
                    'protocol': 'HTTP'
                }
            }
        ]
        
        pattern = normal_patterns[index % len(normal_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        
        return pattern
    
    def preprocess_enhanced_sessions(self, min_alerts=2):
        """Enhanced session preprocessing with better feature extraction"""
        logger.info("🔄 Enhanced preprocessing with 12-dimensional features...")
        
        feature_sequences = []
        session_metadata = []
        
        # Process all sessions but limit for memory
        limited_sessions = dict(list(self.sessions_data.items())[:150])
        
        for session_id, session in limited_sessions.items():
            alerts = session['alerts']
            
            if len(alerts) < min_alerts:
                continue
            
            try:
                # Enhanced time windowing
                alert_windows = self._create_adaptive_windows(alerts)
                
                # Extract enhanced features
                window_features = []
                for window_alerts in alert_windows:
                    if window_alerts:
                        features = self.processor.extract_enhanced_features(window_alerts)
                        feature_vector = [
                            features.temporal_anomaly,
                            features.multi_vector,
                            features.privilege_escalation,
                            features.lateral_movement,
                            features.data_access,
                            features.persistence,
                            features.entropy_score,
                            features.behavioral_deviation,
                            features.network_centrality,
                            features.file_system_anomaly,
                            features.process_chain_depth,
                            features.credential_access
                        ]
                        
                        # Validate and normalize features
                        if all(isinstance(f, (int, float)) and not np.isnan(f) for f in feature_vector):
                            # Apply feature scaling
                            normalized_features = self._normalize_features(feature_vector)
                            window_features.append(normalized_features)
                
                # Quality control for sequences
                if len(window_features) >= 2:
                    max_windows = min(len(window_features), 25)  # Increased limit
                    feature_tensor = torch.tensor(window_features[:max_windows], dtype=torch.float32)
                    
                    feature_sequences.append(feature_tensor)
                    session_metadata.append({
                        'session_id': session_id,
                        'is_attack_session': session.get('is_attack_session', False),
                        'risk_score': session.get('risk_score', 0),
                        'alert_count': len(alerts),
                        'window_count': len(window_features),
                        'ground_truth_attack_prob': session.get('ground_truth_attack_prob', 0.0)
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process session {session_id}: {e}")
                continue
        
        self.feature_sequences = feature_sequences
        self.session_metadata = session_metadata
        
        logger.info(f"✅ Created {len(feature_sequences)} enhanced feature sequences")
        logger.info(f"📊 Average sequence length: {np.mean([len(seq) for seq in feature_sequences]):.1f}")
        
        return feature_sequences, session_metadata
    
    def _create_adaptive_windows(self, alerts: List[Dict], base_window_minutes: int = 3) -> List[List[Dict]]:
        """Create adaptive time windows based on alert density"""
        if not alerts:
            return []
        
        # Parse timestamps
        valid_alerts = []
        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                valid_alerts.append((alert_time, alert))
            except (ValueError, KeyError, TypeError):
                continue
        
        if not valid_alerts:
            return []
        
        valid_alerts.sort(key=lambda x: x[0])
        
        # Adaptive windowing based on alert density
        windows = []
        current_window = []
        current_window_start = None
        
        for i, (alert_time, alert) in enumerate(valid_alerts):
            if current_window_start is None:
                current_window_start = alert_time
                current_window = [alert]
            else:
                # Adaptive window size based on alert frequency
                time_diff = (alert_time - current_window_start).total_seconds()
                
                # Calculate dynamic window size
                if len(current_window) > 5:  # High frequency
                    window_seconds = base_window_minutes * 60 * 0.5
                elif len(current_window) > 2:  # Medium frequency
                    window_seconds = base_window_minutes * 60
                else:  # Low frequency
                    window_seconds = base_window_minutes * 60 * 2
                
                if time_diff <= window_seconds:
                    current_window.append(alert)
                else:
                    if current_window:
                        windows.append(current_window)
                    current_window = [alert]
                    current_window_start = alert_time
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _normalize_features(self, feature_vector: List[float]) -> List[float]:
        """Normalize features to [0, 1] range with outlier handling"""
        normalized = []
        for feature in feature_vector:
            # Clip outliers and normalize
            clipped = max(0.0, min(1.0, feature))
            normalized.append(clipped)
        return normalized
    
    def train_bayesian_regressor(self, num_iterations=500, learning_rate=0.001):
        """Train Bayesian regression model"""
        if not self.feature_sequences:
            raise ValueError("Must preprocess sessions before training regressor")
        
        logger.info("🧠 Training Bayesian Threat Regressor...")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for i, seq in enumerate(self.feature_sequences):
            metadata = self.session_metadata[i]
            
            # Use mean features as input
            mean_features = seq.mean(dim=0)
            X_train.append(mean_features)
            
            # Use ground truth attack probability as target
            attack_prob = metadata.get('ground_truth_attack_prob', 0.0)
            y_train.append(attack_prob)
        
        X_tensor = torch.stack(X_train).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Train with Bayesian approach if Pyro available
        if PYRO_AVAILABLE:
            try:
                losses = self._train_bayesian_pyro(X_tensor, y_tensor, num_iterations, learning_rate)
            except Exception as e:
                logger.warning(f"Pyro Bayesian training failed: {e}, using PyTorch")
                losses = self._train_regressor_pytorch(X_tensor, y_tensor, num_iterations, learning_rate)
        else:
            losses = self._train_regressor_pytorch(X_tensor, y_tensor, num_iterations, learning_rate)
        
        self.regression_trained = True
        logger.info("✅ Bayesian regressor training completed")
        return losses
    
    def _train_bayesian_pyro(self, X, y, num_iterations, learning_rate):
        """Train using Pyro's Bayesian inference"""
        
        def model(X, y):
            # Priors for weights and bias
            w_prior = dist.Normal(0., 1.).expand([X.shape[1], 1]).to_event(2)
            b_prior = dist.Normal(0., 1.).expand([1]).to_event(1)
            
            weights = pyro.sample("weights", w_prior)
            bias = pyro.sample("bias", b_prior)
            
            # Likelihood
            mean = torch.sigmoid(torch.matmul(X, weights) + bias)
            sigma = pyro.sample("sigma", dist.Uniform(0.01, 0.5))
            
            with pyro.plate("data", X.shape[0]):
                pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        
        def guide(X, y):
            # Variational parameters
            w_loc = pyro.param("w_loc", torch.randn(X.shape[1], 1))
            w_scale = pyro.param("w_scale", torch.ones(X.shape[1], 1), 
                                constraint=dist.constraints.positive)
            
            b_loc = pyro.param("b_loc", torch.randn(1))
            b_scale = pyro.param("b_scale", torch.ones(1), 
                                constraint=dist.constraints.positive)
            
            # Sample from variational distribution
            pyro.sample("weights", dist.Normal(w_loc, w_scale).to_event(2))
            pyro.sample("bias", dist.Normal(b_loc, b_scale).to_event(1))
            
            sigma_loc = pyro.param("sigma_loc", torch.tensor(0.1))
            sigma_scale = pyro.param("sigma_scale", torch.tensor(0.05),
                                   constraint=dist.constraints.positive)
            pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
        
        # Clear parameter store and setup SVI
        pyro.clear_param_store()
        optimizer = ClippedAdam({"lr": learning_rate})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        losses = []
        for iteration in range(num_iterations):
            loss = svi.step(X, y)
            losses.append(loss)
            
            if iteration % 100 == 0:
                logger.info(f"Bayesian Iteration {iteration}, Loss: {loss:.4f}")
        
        return losses
    
    def _train_regressor_pytorch(self, X, y, num_iterations, learning_rate):
        """Fallback PyTorch training for regressor"""
        optimizer = torch.optim.AdamW(self.regressor.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        losses = []
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            predictions = self.regressor(X)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if iteration % 100 == 0:
                logger.info(f"PyTorch Regressor Iteration {iteration}, Loss: {loss.item():.4f}")
        
        return losses
    
    def analyze_session_enhanced(self, session_id: str) -> Dict:
        """Enhanced session analysis combining HMM and Bayesian regression"""
        if not self.hmm.trained:
            raise ValueError("HMM must be trained before analysis")
        
        try:
            session = self.sessions_data.get(session_id)
            if not session:
                return {'session_id': session_id, 'error': 'Session not found'}
            
            alerts = session.get('alerts', [])
            if not alerts:
                return {
                    'session_id': session_id,
                    'hmm_attack_probability': 0.0,
                    'bayesian_attack_probability': 0.0,
                    'combined_score': 0.0,
                    'confidence': 0.0,
                    'risk_assessment': {'risk_level': 'MINIMAL', 'priority': 5, 'adjusted_score': 0.0, 'recommended_action': 'No action needed'}
                }
            
            # HMM Analysis
            alert_windows = self._create_adaptive_windows(alerts)
            window_features = []
            
            for window_alerts in alert_windows:
                if window_alerts:
                    features = self.processor.extract_enhanced_features(window_alerts)
                    feature_vector = [
                        features.temporal_anomaly, features.multi_vector,
                        features.privilege_escalation, features.lateral_movement,
                        features.data_access, features.persistence,
                        features.entropy_score, features.behavioral_deviation,
                        features.network_centrality, features.file_system_anomaly,
                        features.process_chain_depth, features.credential_access
                    ]
                    
                    if all(isinstance(f, (int, float)) and not np.isnan(f) for f in feature_vector):
                        normalized_features = self._normalize_features(feature_vector)
                        window_features.append(normalized_features)
            
            if not window_features:
                return {
                    'session_id': session_id,
                    'hmm_attack_probability': 0.0,
                    'bayesian_attack_probability': 0.0,
                    'combined_score': 0.0,
                    'confidence': 0.0,
                    'risk_assessment': {'risk_level': 'MINIMAL', 'priority': 5, 'adjusted_score': 0.0, 'recommended_action': 'No action needed'}
                }
            
            # Limit sequence length
            max_windows = min(len(window_features), 25)
            observations = torch.tensor(window_features[:max_windows], dtype=torch.float32)
            
            # HMM predictions
            hmm_attack_prob = self.hmm.predict_attack_probability(observations)
            predicted_states = self.hmm.viterbi_decode(observations)
            state_sequence = [self.hmm.state_names[state] for state in predicted_states]
            
            # Bayesian regression prediction
            bayesian_attack_prob = 0.0
            if self.regression_trained:
                try:
                    mean_features = observations.mean(dim=0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        bayesian_pred = self.regressor(mean_features)
                        bayesian_attack_prob = float(bayesian_pred.item())
                except Exception as e:
                    logger.warning(f"Bayesian prediction failed: {e}")
            
            # Combine predictions with intelligent weighting
            if self.regression_trained:
                # Weight based on confidence and agreement
                agreement = 1.0 - abs(hmm_attack_prob - bayesian_attack_prob)
                hmm_weight = 0.6 if agreement > 0.5 else 0.4
                bayesian_weight = 1.0 - hmm_weight
                
                combined_score = (hmm_attack_prob * hmm_weight + 
                                bayesian_attack_prob * bayesian_weight)
                confidence = min(agreement + 0.3, 1.0)
            else:
                combined_score = hmm_attack_prob
                confidence = 0.7
            
            # Enhanced attack progression analysis
            attack_progression = self._analyze_enhanced_progression(state_sequence, window_features)
            
            return {
                'session_id': session_id,
                'hmm_attack_probability': float(hmm_attack_prob),
                'bayesian_attack_probability': float(bayesian_attack_prob),
                'combined_score': float(combined_score),
                'confidence': float(confidence),
                'predicted_states': predicted_states.tolist() if len(predicted_states) > 0 else [],
                'state_sequence': state_sequence,
                'window_count': len(window_features),
                'attack_progression': attack_progression,
                'feature_summary': self._summarize_features(window_features),
                'threat_indicators': self._extract_threat_indicators(alerts),
                'risk_assessment': self._assess_risk_level(combined_score, confidence)
            }
            
        except Exception as e:
            logger.warning(f"Enhanced session analysis failed for {session_id}: {e}")
            return {
                'session_id': session_id,
                'hmm_attack_probability': 0.0,
                'bayesian_attack_probability': 0.0,
                'combined_score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'risk_assessment': {'risk_level': 'MINIMAL', 'priority': 5, 'adjusted_score': 0.0, 'recommended_action': 'Error in analysis'}
            }
    
    def _analyze_enhanced_progression(self, state_sequence: List[str], features: List[List[float]]) -> Dict:
        """Enhanced attack progression analysis"""
        progression = {
            'stages_detected': list(set(state_sequence)),
            'kill_chain_coverage': 0.0,
            'progression_timeline': [],
            'persistence_detected': False,
            'escalation_detected': False,
            'data_access_detected': False,
            'lateral_movement_detected': False,
            'attack_sophistication': 'low'
        }
        
        # Kill chain analysis
        attack_stages = ['reconnaissance', 'initial_access', 'privilege_escalation', 
                        'lateral_movement', 'data_exfiltration']
        detected_attack_stages = [s for s in progression['stages_detected'] if s in attack_stages]
        progression['kill_chain_coverage'] = len(detected_attack_stages) / len(attack_stages)
        
        # Timeline analysis
        for i, state in enumerate(state_sequence):
            if state != 'normal_operations':
                window_features = features[i] if i < len(features) else [0] * 12
                progression['progression_timeline'].append({
                    'window': i,
                    'stage': state,
                    'feature_intensity': sum(window_features) / len(window_features)
                })
        
        # Specific detections based on features
        if features:
            avg_features = [sum(f[i] for f in features) / len(features) for i in range(12)]
            
            progression['persistence_detected'] = avg_features[5] > 0.4  # persistence feature
            progression['escalation_detected'] = avg_features[2] > 0.5   # privilege_escalation
            progression['data_access_detected'] = avg_features[4] > 0.3  # data_access
            progression['lateral_movement_detected'] = avg_features[3] > 0.4  # lateral_movement
            
            # Assess sophistication
            sophistication_score = (avg_features[6] +   # entropy_score
                                  avg_features[7] +   # behavioral_deviation
                                  avg_features[10]) / 3  # process_chain_depth
            
            if sophistication_score > 0.7:
                progression['attack_sophistication'] = 'high'
            elif sophistication_score > 0.4:
                progression['attack_sophistication'] = 'medium'
        
        return progression
    
    def _summarize_features(self, features: List[List[float]]) -> Dict:
        """Summarize feature statistics"""
        if not features:
            return {}
        
        feature_names = [
            'temporal_anomaly', 'multi_vector', 'privilege_escalation',
            'lateral_movement', 'data_access', 'persistence',
            'entropy_score', 'behavioral_deviation', 'network_centrality',
            'file_system_anomaly', 'process_chain_depth', 'credential_access'
        ]
        
        summary = {}
        for i, name in enumerate(feature_names):
            values = [f[i] for f in features]
            summary[name] = {
                'mean': float(np.mean(values)),
                'max': float(np.max(values)),
                'std': float(np.std(values))
            }
        
        return summary
    
    def _extract_threat_indicators(self, alerts: List[Dict]) -> Dict:
        """Extract specific threat indicators"""
        indicators = {
            'suspicious_processes': [],
            'network_anomalies': [],
            'file_operations': [],
            'registry_modifications': [],
            'credential_access_attempts': []
        }
        
        for alert in alerts:
            event_type = alert.get('event_type', '')
            details = alert.get('details', {})
            
            if event_type == 'Process':
                file_name = details.get('file_name', '').lower()
                if any(proc in file_name for proc in self.processor.suspicious_processes):
                    indicators['suspicious_processes'].append({
                        'process': file_name,
                        'command': details.get('command_line', ''),
                        'timestamp': alert.get('timestamp', '')
                    })
            
            elif event_type == 'Network':
                remote_ip = details.get('remote_ip', '')
                remote_port = details.get('remote_port', 0)
                if remote_port in self.processor.suspicious_ports:
                    indicators['network_anomalies'].append({
                        'destination': f"{remote_ip}:{remote_port}",
                        'protocol': details.get('protocol', ''),
                        'timestamp': alert.get('timestamp', '')
                    })
            
            elif event_type == 'File':
                action = details.get('action', '')
                if action in ['FileDeleted', 'FileAccessed', 'FileCopied']:
                    indicators['file_operations'].append({
                        'action': action,
                        'file': details.get('file_name', ''),
                        'path': details.get('file_path', ''),
                        'timestamp': alert.get('timestamp', '')
                    })
            
            elif event_type == 'Registry':
                indicators['registry_modifications'].append({
                    'key': details.get('registry_key', ''),
                    'action': details.get('action', ''),
                    'timestamp': alert.get('timestamp', '')
                })
        
        return indicators
    
    def _assess_risk_level(self, combined_score: float, confidence: float) -> Dict:
        """Assess overall risk level"""
        # Adjust score based on confidence
        adjusted_score = combined_score * confidence
        
        if adjusted_score >= 0.8:
            risk_level = 'CRITICAL'
            priority = 1
            recommended_action = 'Immediate investigation and containment required'
        elif adjusted_score >= 0.6:
            risk_level = 'HIGH'
            priority = 2
            recommended_action = 'Urgent investigation recommended'
        elif adjusted_score >= 0.4:
            risk_level = 'MEDIUM'
            priority = 3
            recommended_action = 'Monitor closely and investigate when resources available'
        elif adjusted_score >= 0.2:
            risk_level = 'LOW'
            priority = 4
            recommended_action = 'Low priority monitoring'
        else:
            risk_level = 'MINIMAL'
            priority = 5
            recommended_action = 'Standard monitoring sufficient'
        
        return {
            'risk_level': risk_level,
            'priority': priority,
            'adjusted_score': float(adjusted_score),
            'recommended_action': recommended_action
        }
    
    def train_enhanced_models(self, hmm_iterations=200, reg_iterations=300):
        """Train both HMM and Bayesian regressor"""
        if not self.feature_sequences:
            raise ValueError("Must preprocess sessions before training")
        
        logger.info("🚀 Training Enhanced Pyro Models...")
        
        # Train HMM
        logger.info("📊 Training Enhanced HMM...")
        hmm_losses = self.hmm.fit(self.feature_sequences, 
                                 num_iterations=hmm_iterations, 
                                 learning_rate=0.005)
        
        # Train Bayesian regressor
        logger.info("🧠 Training Bayesian Regressor...")
        reg_losses = self.train_bayesian_regressor(num_iterations=reg_iterations,
                                                  learning_rate=0.001)
        
        logger.info("✅ Enhanced model training completed!")
        
        return {
            'hmm_losses': hmm_losses,
            'regressor_losses': reg_losses,
            'hmm_trained': self.hmm.trained,
            'regressor_trained': self.regression_trained
        }
    
    def visualize_enhanced_results(self, save_results=True):
        """Create enhanced visualizations and save results"""
        try:
            print("\n🚀 ENHANCED PYRO + BAYESIAN ANALYSIS RESULTS")
            print("=" * 70)
            print(f"🔥 Device: {self.device}")
            print(f"🔥 Pyro Available: {PYRO_AVAILABLE}")
            print(f"🧠 Models Trained: HMM={self.hmm.trained}, Bayesian={self.regression_trained}")
            
            # Model parameters
            hmm_params = self.hmm.get_learned_parameters()
            
            if hmm_params:
                print("\n🔄 ENHANCED HMM TRANSITION MATRIX:")
                transition_df = pd.DataFrame(
                    hmm_params['transitions'],
                    index=self.hmm.state_names,
                    columns=self.hmm.state_names
                )
                print(transition_df.round(3).to_string())
                
                print("\n📊 INITIAL STATE PROBABILITIES:")
                for i, state in enumerate(self.hmm.state_names):
                    prob = hmm_params['initial_probs'][i]
                    print(f"   {state:20s}: {prob:.3f}")
            
            # Analyze sessions with both models
            session_results = []
            hmm_scores = []
            bayesian_scores = []
            combined_scores = []
            
            # Process subset to prevent hanging
            limited_metadata = self.session_metadata[:min(len(self.session_metadata), 150)]
            
            for metadata in limited_metadata:
                session_id = metadata['session_id']
                try:
                    analysis = self.analyze_session_enhanced(session_id)
                    
                    hmm_score = analysis['hmm_attack_probability']
                    bayesian_score = analysis['bayesian_attack_probability']
                    combined_score = analysis['combined_score']
                    
                    hmm_scores.append(hmm_score)
                    bayesian_scores.append(bayesian_score)
                    combined_scores.append(combined_score)
                    
                    session_results.append({
                        'session_id': session_id,
                        'hmm_attack_probability': hmm_score,
                        'bayesian_attack_probability': bayesian_score,
                        'combined_score': combined_score,
                        'confidence': analysis['confidence'],
                        'risk_level': analysis['risk_assessment']['risk_level'],
                        'original_risk_score': metadata['risk_score'],
                        'alert_count': metadata['alert_count'],
                        'is_attack': metadata.get('is_attack_session', False)
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Visualization analysis failed for {session_id}: {e}")
                    continue
            
            results_df = pd.DataFrame(session_results)
            
            if len(session_results) > 0:
                print(f"\n🎯 ENHANCED DUAL-MODEL ANALYSIS:")
                print(f"   Sessions Analyzed: {len(session_results)}")
                print(f"   HMM Mean Score: {np.mean(hmm_scores):.1%}")
                print(f"   Bayesian Mean Score: {np.mean(bayesian_scores):.1%}")
                print(f"   Combined Mean Score: {np.mean(combined_scores):.1%}")
                print(f"   Critical Threats: {len([s for s in session_results if s['risk_level'] == 'CRITICAL'])}")
                print(f"   High Risk: {len([s for s in session_results if s['risk_level'] == 'HIGH'])}")
                
                # Model comparison
                print(f"\n📊 MODEL COMPARISON:")
                if self.regression_trained:
                    correlation = np.corrcoef(hmm_scores, bayesian_scores)[0, 1]
                    print(f"   HMM-Bayesian Correlation: {correlation:.3f}")
                    
                    agreement_rate = sum(1 for h, b in zip(hmm_scores, bayesian_scores) 
                                       if abs(h - b) < 0.2) / len(hmm_scores)
                    print(f"   Model Agreement Rate: {agreement_rate:.1%}")
                
                # Risk distribution
                print(f"\n🚨 RISK DISTRIBUTION:")
                risk_counts = defaultdict(int)
                for result in session_results:
                    risk_counts[result['risk_level']] += 1
                
                for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
                    count = risk_counts[risk_level]
                    percentage = count / len(session_results) * 100 if session_results else 0
                    bar = '█' * max(1, int(percentage / 5))
                    print(f"   {risk_level:>8}: {count:3d} sessions {bar} ({percentage:.1f}%)")
            
            # Save enhanced results
            if save_results and len(results_df) > 0:
                results_df.to_csv('enhanced_pyro_bayesian_analysis.csv', index=False)
                
                if hmm_params:
                    transition_df.to_csv('enhanced_pyro_transition_matrix.csv')
                
                print(f"\n💾 Enhanced Results Saved:")
                print(f"   • enhanced_pyro_bayesian_analysis.csv")
                print(f"   • enhanced_pyro_transition_matrix.csv")
            
            return results_df
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced visualization failed: {e}")
            return None
    
    def export_enhanced_results(self, output_file='enhanced_pyro_intelligence_report.json'):
        """Export comprehensive enhanced results"""
        logger.info("📊 Exporting Enhanced Intelligence Report...")
        
        # Simple report generation for now
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model_type': 'Enhanced_Pyro_HMM_Bayesian',
                'total_sessions_processed': len(self.session_metadata),
                'device_used': str(self.device),
                'pyro_available': PYRO_AVAILABLE,
                'models_trained': {
                    'hmm': self.hmm.trained,
                    'bayesian_regressor': self.regression_trained
                }
            },
            'summary': f"Processed {len(self.session_metadata)} sessions with enhanced Pyro+Bayesian analysis"
        }
        
        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Enhanced report exported: {output_file}")
        
        return report

def main():
    """Enhanced main execution function"""
    print("🚀 Enhanced Pyro + Bayesian Cybersecurity Analysis")
    print("=" * 70)
    
    # System information
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("🔧 Using CPU")
    
    print(f"🔥 Pyro Available: {PYRO_AVAILABLE}")
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedBayesianAnalyzer()
        
        # Check for your real session data first
        real_data_file = "synthetic_sessions_sample.json"
        use_real_data = False
        
        try:
            import os
            if os.path.exists(real_data_file):
                file_size = os.path.getsize(real_data_file)
                print(f"📁 Found real session data: {real_data_file} ({file_size / 1024 / 1024:.1f} MB)")
                use_real_data = True
            else:
                print(f"📁 Real session data not found: {real_data_file}")
        except:
            print(f"📁 Could not access real session data file")
        
        if use_real_data:
            # Load and preprocess real data
            print(f"\n📊 Loading REAL session data from {real_data_file}...")
            analyzer.load_json_data(real_data_file, max_sessions=300)  # Limit for memory
            
            print(f"✅ Loaded {len(analyzer.sessions_data)} real sessions")
            
            if len(analyzer.sessions_data) == 0:
                print("❌ No sessions were successfully loaded from the JSON file")
                print("🔍 This might be due to:")
                print("   • Unexpected JSON structure")
                print("   • Missing required fields in sessions")
                print("   • All sessions filtered out due to missing alerts")
                print("\n🔧 Running JSON diagnostic...")
                
                # Run diagnostic to understand the issue
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, "-c", """
import json
with open('synthetic_sessions_sample.json', 'r') as f:
    data = json.load(f)
print(f'Root type: {type(data).__name__}')
if isinstance(data, list):
    print(f'Array length: {len(data)}')
    if data: print(f'First element keys: {list(data[0].keys()) if isinstance(data[0], dict) else "Not a dict"}')
elif isinstance(data, dict):
    print(f'Object keys: {list(data.keys())}')
    for key in ['sessions', 'data', 'events', 'alerts']:
        if key in data: print(f'Found container "{key}": {type(data[key]).__name__} with {len(data[key]) if hasattr(data[key], "__len__") else "unknown"} items')
                    """], capture_output=True, text=True, timeout=10)
                    if result.stdout:
                        print("📊 JSON Structure Analysis:")
                        print(result.stdout)
                except:
                    pass
                
                print("\n🔄 Falling back to synthetic data...")
                use_real_data = False
            else:
                # Show data statistics
                attack_sessions = sum(1 for s in analyzer.sessions_data.values() if s['is_attack_session'])
                total_alerts = sum(len(s['alerts']) for s in analyzer.sessions_data.values())
                avg_alerts = total_alerts / len(analyzer.sessions_data) if analyzer.sessions_data else 0
                
                print(f"📊 Data Statistics:")
                print(f"   • Total Sessions: {len(analyzer.sessions_data)}")
                print(f"   • Attack Sessions: {attack_sessions} ({attack_sessions/len(analyzer.sessions_data)*100:.1f}%)")
                print(f"   • Total Alerts: {total_alerts}")
                print(f"   • Average Alerts/Session: {avg_alerts:.1f}")
        
        if not use_real_data or len(analyzer.sessions_data) == 0:
            # Fallback to synthetic data
            print("\n📊 Creating synthetic data (real data not available)...")
            sample_data = create_sample_data()
            sample_csv = "sample_security_data.csv"
            sample_data.to_csv(sample_csv, index=False)
            print(f"📊 Created sample dataset: {sample_csv}")
            analyzer.load_csv_data(sample_csv)
        
        # Preprocess sessions
        print("\n🔄 Preprocessing Sessions...")
        feature_sequences, session_metadata = analyzer.preprocess_enhanced_sessions()
        
        if not feature_sequences:
            print("❌ No valid feature sequences generated")
            return
        
        print(f"✅ Generated {len(feature_sequences)} feature sequences")
        print(f"📊 Average sequence length: {np.mean([len(seq) for seq in feature_sequences]):.1f}")
        
        # Train models
        print("\n🚀 Training Enhanced Models...")
        training_results = analyzer.train_enhanced_models(hmm_iterations=100, reg_iterations=150)
        
        print(f"✅ HMM Training: {'Success' if training_results['hmm_trained'] else 'Failed'}")
        print(f"✅ Bayesian Training: {'Success' if training_results['regressor_trained'] else 'Failed'}")
        
        if training_results['hmm_losses']:
            print(f"📉 HMM Final Loss: {training_results['hmm_losses'][-1]:.4f}")
        if training_results['regressor_losses']:
            print(f"📉 Bayesian Final Loss: {training_results['regressor_losses'][-1]:.4f}")
        
        # Analyze sessions
        print("\n🔍 Analyzing Sessions...")
        results_df = analyzer.visualize_enhanced_results(save_results=True)
        
        # Generate intelligence report
        print("\n📋 Generating Intelligence Report...")
        report = analyzer.export_enhanced_results()
        
        # Display summary
        print("\n" + "="*70)
        print("🎯 ANALYSIS COMPLETE")
        print("="*70)
        
        if results_df is not None and len(results_df) > 0:
            critical_count = len([r for r in results_df.to_dict('records') if r.get('risk_level') == 'CRITICAL'])
            high_count = len([r for r in results_df.to_dict('records') if r.get('risk_level') == 'HIGH'])
            
            print(f"📊 Sessions Analyzed: {len(results_df)}")
            print(f"🚨 Critical Threats: {critical_count}")
            print(f"🔥 High Risk: {high_count}")
            print(f"📈 Detection Rate: {(critical_count + high_count) / len(results_df) * 100:.1f}%")
            
            if use_real_data:
                print(f"🎯 REAL DATA INSIGHTS:")
                print(f"   • Training on {len(analyzer.sessions_data)} real sessions")
                print(f"   • {total_alerts} real security alerts processed")
                print(f"   • Enhanced 12D feature extraction applied")
        
        print(f"\n💾 Results saved to:")
        print(f"   • enhanced_pyro_bayesian_analysis.csv")
        print(f"   • enhanced_pyro_intelligence_report.json")
        if use_real_data:
            print(f"   • Trained on REAL session data from {real_data_file}")
        print("\n✅ Enhanced Pyro + Bayesian Analysis Complete!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        print(f"❌ Error: {e}")
        raise

def create_sample_data():
    """Create sample security data for demonstration"""
    print("🏗️ Creating sample security dataset...")
    
    np.random.seed(42)
    num_sessions = 100
    
    data = []
    for i in range(num_sessions):
        # Create realistic attack probability distribution
        attack_prob = np.random.beta(2, 8)  # Skewed towards lower probabilities
        
        # Determine session characteristics based on attack probability
        if attack_prob > 0.8:
            # Advanced persistent threat
            alert_count = np.random.randint(15, 25)
            risk_score = np.random.randint(85, 100)
            session_type = "APT"
        elif attack_prob > 0.5:
            # Targeted attack
            alert_count = np.random.randint(8, 18)
            risk_score = np.random.randint(70, 90)
            session_type = "Targeted"
        elif attack_prob > 0.2:
            # Opportunistic attack
            alert_count = np.random.randint(5, 12)
            risk_score = np.random.randint(50, 75)
            session_type = "Opportunistic"
        else:
            # Normal operations
            alert_count = np.random.randint(3, 8)
            risk_score = np.random.randint(10, 45)
            session_type = "Normal"
        
        # Add some noise to make it more realistic
        risk_score += np.random.randint(-5, 6)
        risk_score = max(0, min(100, risk_score))
        
        data.append({
            'session_id': f"session_{i:04d}",
            'alert_count': alert_count,
            'attack_probability': round(attack_prob, 3),
            'original_risk_score': risk_score,
            'session_type': session_type,
            'timestamp': (datetime.now() - pd.Timedelta(days=np.random.randint(0, 30))).isoformat()
        })
    
    df = pd.DataFrame(data)
    logger.info(f"📊 Created {len(df)} sample sessions")
    logger.info(f"   • APT sessions: {len(df[df.session_type == 'APT'])}")
    logger.info(f"   • Targeted attacks: {len(df[df.session_type == 'Targeted'])}")
    logger.info(f"   • Opportunistic: {len(df[df.session_type == 'Opportunistic'])}")
    logger.info(f"   • Normal operations: {len(df[df.session_type == 'Normal'])}")
    
    return df

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n💥 Fatal error: {e}")
        print("Please check the logs for more details.")