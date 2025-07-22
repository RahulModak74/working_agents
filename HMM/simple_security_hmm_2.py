#!/usr/bin/env python3
"""
FIXED Security Alert HMM Analysis
Comprehensive fixes for probability calibration, data processing, and model accuracy
"""

import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AlertFeatures:
    """Structured representation of alert features for HMM"""
    temporal_anomaly: float      # 0-1: unusual timing
    multi_vector: float          # 0-1: multiple event types
    privilege_escalation: float  # 0-1: admin/privilege indicators
    lateral_movement: float      # 0-1: network traversal indicators
    data_access: float          # 0-1: file/data manipulation
    persistence: float          # 0-1: system persistence indicators

class FixedSecurityHMM:
    """Fixed Hidden Markov Model for security alert sequence analysis"""
    
    def __init__(self, num_states=5):
        """
        Initialize the Fixed Security HMM
        
        Args:
            num_states: Number of hidden attack states
        """
        self.num_states = num_states
        
        # Define attack states based on cyber kill chain
        self.state_names = [
            "normal_operations",     # State 0: Normal activity
            "reconnaissance",        # State 1: Info gathering
            "initial_access",        # State 2: Entry point
            "lateral_movement",      # State 3: Network traversal  
            "objective_execution"    # State 4: Final objectives
        ]
        
        # Attack severity weights for each state
        self.state_attack_weights = {
            0: 0.0,    # normal_operations - no attack
            1: 0.15,   # reconnaissance - low severity
            2: 0.35,   # initial_access - medium severity
            3: 0.65,   # lateral_movement - high severity
            4: 0.90    # objective_execution - critical severity
        }
        
        # HMM parameters
        self.transition_probs = None
        self.emission_models = None
        self.initial_probs = None
        self.trained = False
        self.feature_scaler = StandardScaler()
        
        # Numerical stability constants
        self.MIN_PROB = 1e-10
        self.LOG_MIN_PROB = np.log(self.MIN_PROB)
        self.MAX_LOG_PROB = 700  # Prevent overflow
    
    def _safe_log(self, x):
        """Safe logarithm that avoids NaN values"""
        x_safe = np.maximum(x, self.MIN_PROB)
        return np.clip(np.log(x_safe), self.LOG_MIN_PROB, self.MAX_LOG_PROB)
    
    def _safe_exp(self, x):
        """Safe exponential that clips extreme values"""
        x_clipped = np.clip(x, -self.MAX_LOG_PROB, self.MAX_LOG_PROB)
        return np.exp(x_clipped)
    
    def _normalize_features(self, features):
        """Properly normalize features to [0,1] range"""
        if isinstance(features, list):
            features = np.array(features)
        
        # Clip extreme values and normalize
        features_clipped = np.clip(features, 0, 3)  # Allow some overflow
        return features_clipped / 3.0  # Normalize to [0,1]
    
    def fit(self, sequences, ground_truth_labels=None, num_iterations=100):
        """
        Train the HMM with proper feature normalization and validation
        
        Args:
            sequences: List of feature sequences
            ground_truth_labels: Optional ground truth labels for validation
            num_iterations: Number of training iterations
        """
        logger.info(f"Training Fixed HMM with {num_iterations} iterations...")
        
        if not sequences:
            raise ValueError("No sequences provided for training")
        
        # Convert and normalize all features
        normalized_sequences = []
        all_features = []
        
        for seq_idx, seq in enumerate(sequences):
            if torch.is_tensor(seq):
                seq_array = seq.numpy()
            else:
                seq_array = np.array(seq)
            
            if seq_array.size == 0:
                continue
                
            # Normalize features
            normalized_seq = []
            for features in seq_array:
                normalized_features = self._normalize_features(features)
                normalized_seq.append(normalized_features)
                all_features.append(normalized_features)
            
            if normalized_seq:
                normalized_sequences.append(np.array(normalized_seq))
        
        if not normalized_sequences:
            raise ValueError("No valid sequences after normalization")
        
        all_features = np.array(all_features)
        
        # Fit feature scaler for consistent normalization
        self.feature_scaler.fit(all_features)
        
        # Apply additional scaling
        scaled_features = self.feature_scaler.transform(all_features)
        
        # Initialize with improved K-means clustering
        logger.info("Initializing states with K-means clustering...")
        kmeans = KMeans(n_clusters=self.num_states, random_state=42, n_init=20, max_iter=300)
        initial_states = kmeans.fit_predict(scaled_features)
        
        # Initialize emission models with better regularization
        self.emission_models = {}
        for state in range(self.num_states):
            state_features = scaled_features[initial_states == state]
            
            if len(state_features) > 5:  # Need sufficient samples
                try:
                    # Use regularized covariance
                    mean = np.mean(state_features, axis=0)
                    cov = np.cov(state_features.T)
                    
                    # Add strong regularization
                    reg_strength = 0.1
                    cov += np.eye(len(mean)) * reg_strength
                    
                    self.emission_models[state] = {
                        'mean': mean,
                        'cov': cov,
                        'type': 'gaussian'
                    }
                except Exception as e:
                    logger.warning(f"Failed to create emission model for state {state}: {e}")
                    # Fallback to uniform model
                    self.emission_models[state] = {
                        'mean': np.mean(scaled_features, axis=0),
                        'cov': np.eye(scaled_features.shape[1]) * 0.5,
                        'type': 'gaussian'
                    }
            else:
                # Not enough samples - use global statistics
                self.emission_models[state] = {
                    'mean': np.mean(scaled_features, axis=0),
                    'cov': np.eye(scaled_features.shape[1]) * 0.5,
                    'type': 'gaussian'
                }
        
        # Initialize transition probabilities with domain knowledge
        self.transition_probs = self._initialize_transition_matrix()
        
        # Initialize state probabilities (bias towards normal)
        self.initial_probs = np.array([0.7, 0.1, 0.1, 0.05, 0.05])  # Favor normal state
        
        # Training with early stopping
        prev_likelihood = -np.inf
        patience = 10
        patience_counter = 0
        
        for iteration in range(num_iterations):
            if iteration % 20 == 0:
                logger.info(f"Training iteration {iteration}/{num_iterations}")
            
            # E-step: Estimate state probabilities
            state_probs, likelihood = self._estimate_states_with_likelihood(normalized_sequences)
            
            # Check for convergence
            if likelihood - prev_likelihood < 1e-6:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
            else:
                patience_counter = 0
            
            prev_likelihood = likelihood
            
            # M-step: Update parameters
            self._update_parameters(normalized_sequences, state_probs)
            
            # Ensure numerical stability
            self._stabilize_parameters()
        
        self.trained = True
        logger.info("Training completed successfully!")
        
        # Validate if ground truth is available
        if ground_truth_labels:
            self._validate_model(normalized_sequences, ground_truth_labels)
    
    def _initialize_transition_matrix(self):
        """Initialize transition matrix with domain knowledge"""
        trans = np.ones((self.num_states, self.num_states)) * 0.01  # Small baseline
        
        # Domain knowledge: normal operations transitions
        trans[0, 0] = 0.85  # Stay normal
        trans[0, 1] = 0.10  # Normal -> reconnaissance
        trans[0, 2] = 0.03  # Normal -> initial access (rare)
        trans[0, 3] = 0.01  # Normal -> lateral (very rare)
        trans[0, 4] = 0.01  # Normal -> objective (very rare)
        
        # Reconnaissance transitions
        trans[1, 0] = 0.30  # Back to normal
        trans[1, 1] = 0.40  # Stay in recon
        trans[1, 2] = 0.25  # Recon -> initial access
        trans[1, 3] = 0.04  # Skip to lateral
        trans[1, 4] = 0.01  # Skip to objective
        
        # Initial access transitions
        trans[2, 0] = 0.20  # Back to normal (failed attack)
        trans[2, 1] = 0.05  # Back to recon
        trans[2, 2] = 0.30  # Stay in initial access
        trans[2, 3] = 0.40  # Progress to lateral movement
        trans[2, 4] = 0.05  # Skip to objective
        
        # Lateral movement transitions
        trans[3, 0] = 0.15  # Back to normal
        trans[3, 1] = 0.05  # Back to recon
        trans[3, 2] = 0.10  # Back to initial access
        trans[3, 3] = 0.50  # Stay in lateral movement
        trans[3, 4] = 0.20  # Progress to objective
        
        # Objective execution transitions
        trans[4, 0] = 0.40  # Back to normal (mission complete)
        trans[4, 1] = 0.05  # New reconnaissance
        trans[4, 2] = 0.05  # New initial access
        trans[4, 3] = 0.20  # Back to lateral movement
        trans[4, 4] = 0.30  # Stay in objective execution
        
        # Normalize each row
        for i in range(self.num_states):
            trans[i] = trans[i] / trans[i].sum()
        
        return trans
    
    def _estimate_states_with_likelihood(self, sequences):
        """Estimate state probabilities and calculate likelihood"""
        state_probs = []
        total_likelihood = 0.0
        
        for seq in sequences:
            if len(seq) == 0:
                continue
                
            seq_array = np.array(seq)
            seq_state_probs = []
            seq_likelihood = 0.0
            
            for features in seq_array:
                # Scale features consistently
                scaled_features = self.feature_scaler.transform([features])[0]
                
                # Calculate likelihood for each state
                likelihoods = []
                for state in range(self.num_states):
                    likelihood = self._emission_probability(scaled_features, state)
                    likelihoods.append(max(likelihood, self.MIN_PROB))
                
                # Normalize to get probabilities
                likelihoods = np.array(likelihoods)
                total = likelihoods.sum()
                
                if total > self.MIN_PROB:
                    probs = likelihoods / total
                    seq_likelihood += np.log(total)
                else:
                    probs = np.ones(self.num_states) / self.num_states
                
                seq_state_probs.append(probs)
            
            if seq_state_probs:
                state_probs.append(np.array(seq_state_probs))
                total_likelihood += seq_likelihood
        
        return state_probs, total_likelihood
    
    def _emission_probability(self, features, state):
        """Calculate emission probability with improved numerical stability"""
        try:
            model = self.emission_models[state]
            mean = model['mean']
            cov = model['cov']
            
            diff = features - mean
            
            # Improved numerical computation
            try:
                # Use scipy's multivariate normal if available
                try:
                    from scipy.stats import multivariate_normal
                    prob = multivariate_normal.pdf(features, mean, cov)
                    return max(prob, self.MIN_PROB)
                except ImportError:
                    pass
                
                # Fallback to manual computation with better stability
                cov_det = np.linalg.det(cov)
                if cov_det <= 0:
                    return self.MIN_PROB
                
                cov_inv = np.linalg.inv(cov)
                
                # Mahalanobis distance
                mahal_dist = diff.T @ cov_inv @ diff
                
                # Log-space computation for numerical stability
                log_prob = -0.5 * (len(features) * np.log(2 * np.pi) + 
                                  np.log(cov_det) + mahal_dist)
                
                prob = self._safe_exp(log_prob)
                return max(prob, self.MIN_PROB)
                
            except (np.linalg.LinAlgError, ValueError) as e:
                # Fallback to simple distance-based probability
                dist = np.linalg.norm(diff)
                prob = self._safe_exp(-dist)
                return max(prob, self.MIN_PROB)
                
        except Exception as e:
            logger.debug(f"Emission probability failed for state {state}: {e}")
            return self.MIN_PROB
    
    def _update_parameters(self, sequences, state_probs):
        """Update HMM parameters with improved estimation"""
        # Update emission parameters
        for state in range(self.num_states):
            state_features = []
            state_weights = []
            
            for seq_idx, seq in enumerate(sequences):
                if seq_idx >= len(state_probs):
                    continue
                    
                seq_array = np.array(seq)
                seq_state_probs = state_probs[seq_idx]
                
                for t, features in enumerate(seq_array):
                    if t < len(seq_state_probs):
                        weight = seq_state_probs[t][state]
                        if weight > 0.001:  # Only include if reasonably probable
                            # Scale features consistently
                            scaled_features = self.feature_scaler.transform([features])[0]
                            state_features.append(scaled_features)
                            state_weights.append(weight)
            
            # Update emission model if we have enough data
            if len(state_features) > 2:
                state_features = np.array(state_features)
                state_weights = np.array(state_weights)
                
                # Weighted mean
                weighted_mean = np.average(state_features, weights=state_weights, axis=0)
                
                # Weighted covariance with regularization
                if len(state_features) > 1:
                    # Compute weighted covariance
                    centered = state_features - weighted_mean
                    weighted_cov = np.cov(centered.T, aweights=state_weights)
                    
                    # Add regularization
                    reg_strength = 0.1
                    weighted_cov += np.eye(len(weighted_mean)) * reg_strength
                    
                    self.emission_models[state]['mean'] = weighted_mean
                    self.emission_models[state]['cov'] = weighted_cov
        
        # Update transition probabilities
        transition_counts = np.ones((self.num_states, self.num_states)) * 0.1
        
        for seq_idx, seq in enumerate(sequences):
            if seq_idx >= len(state_probs):
                continue
                
            seq_state_probs = state_probs[seq_idx]
            
            for t in range(len(seq_state_probs) - 1):
                curr_probs = seq_state_probs[t]
                next_probs = seq_state_probs[t + 1]
                
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        transition_counts[i, j] += curr_probs[i] * next_probs[j]
        
        # Normalize transition probabilities
        for i in range(self.num_states):
            total = transition_counts[i].sum()
            if total > 0:
                self.transition_probs[i] = transition_counts[i] / total
    
    def _stabilize_parameters(self):
        """Ensure all parameters are numerically stable"""
        # Stabilize transition probabilities
        self.transition_probs = np.maximum(self.transition_probs, self.MIN_PROB)
        for i in range(self.num_states):
            row_sum = self.transition_probs[i].sum()
            if row_sum > 0:
                self.transition_probs[i] = self.transition_probs[i] / row_sum
        
        # Stabilize initial probabilities
        self.initial_probs = np.maximum(self.initial_probs, self.MIN_PROB)
        self.initial_probs = self.initial_probs / self.initial_probs.sum()
        
        # Stabilize emission models
        for state in range(self.num_states):
            if state in self.emission_models:
                cov = self.emission_models[state]['cov']
                
                # Ensure positive definite covariance
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    eigenvals = np.maximum(eigenvals, 1e-6)
                    self.emission_models[state]['cov'] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                except:
                    # Fallback to identity matrix
                    dim = len(self.emission_models[state]['mean'])
                    self.emission_models[state]['cov'] = np.eye(dim) * 0.1
    
    def viterbi_decode(self, observations):
        """Enhanced Viterbi algorithm with better numerical stability"""
        if not self.trained:
            raise ValueError("Model must be trained before decoding")
        
        if torch.is_tensor(observations):
            obs_array = observations.numpy()
        else:
            obs_array = np.array(observations)
        
        if obs_array.size == 0:
            return np.array([])
        
        seq_len = len(obs_array)
        
        # Normalize observations
        normalized_obs = []
        for features in obs_array:
            normalized_features = self._normalize_features(features)
            scaled_features = self.feature_scaler.transform([normalized_features])[0]
            normalized_obs.append(scaled_features)
        
        normalized_obs = np.array(normalized_obs)
        
        # Initialize log probability matrix
        log_delta = np.full((seq_len, self.num_states), self.LOG_MIN_PROB)
        psi = np.zeros((seq_len, self.num_states), dtype=int)
        
        # Initialization step
        for s in range(self.num_states):
            emission_prob = self._emission_probability(normalized_obs[0], s)
            log_delta[0, s] = self._safe_log(self.initial_probs[s]) + self._safe_log(emission_prob)
        
        # Forward pass
        for t in range(1, seq_len):
            for s in range(self.num_states):
                # Calculate transition scores
                trans_scores = []
                for prev_s in range(self.num_states):
                    if log_delta[t-1, prev_s] > self.LOG_MIN_PROB:
                        trans_prob = max(self.transition_probs[prev_s, s], self.MIN_PROB)
                        score = log_delta[t-1, prev_s] + self._safe_log(trans_prob)
                        trans_scores.append((score, prev_s))
                    else:
                        trans_scores.append((self.LOG_MIN_PROB, prev_s))
                
                # Find best transition
                if trans_scores:
                    best_score, best_prev = max(trans_scores, key=lambda x: x[0])
                    psi[t, s] = best_prev
                    
                    # Add emission probability
                    emission_prob = self._emission_probability(normalized_obs[t], s)
                    log_delta[t, s] = best_score + self._safe_log(emission_prob)
                else:
                    psi[t, s] = 0
                    log_delta[t, s] = self.LOG_MIN_PROB
        
        # Backward pass (traceback)
        states = np.zeros(seq_len, dtype=int)
        
        # Find best final state
        if np.any(log_delta[-1] > self.LOG_MIN_PROB):
            states[-1] = np.argmax(log_delta[-1])
        else:
            states[-1] = 0  # Default to normal state
        
        # Trace back optimal path
        for t in range(seq_len - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def predict_attack_probability(self, observations):
        """Calculate attack probability with proper calibration"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            states = self.viterbi_decode(observations)
            
            if len(states) == 0:
                return 0.0
            
            # Calculate weighted attack probability using state weights
            total_weighted_score = 0.0
            for state in states:
                total_weighted_score += self.state_attack_weights.get(state, 0.0)
            
            # Average and apply calibration
            avg_score = total_weighted_score / len(states)
            
            # Apply sigmoid-like calibration for more realistic probabilities
            calibrated_prob = self._calibrate_probability(avg_score)
            
            return min(max(calibrated_prob, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Attack probability prediction failed: {e}")
            return 0.0
    
    def _calibrate_probability(self, raw_score):
        """Apply calibration to raw attack scores"""
        # Sigmoid-like calibration
        # Maps raw scores to more realistic probability ranges
        
        if raw_score <= 0.1:
            return raw_score * 0.5  # Very low scores -> even lower probabilities
        elif raw_score <= 0.3:
            return 0.05 + (raw_score - 0.1) * 1.5  # Low scores -> moderate scaling
        elif raw_score <= 0.6:
            return 0.35 + (raw_score - 0.3) * 1.5  # Medium scores -> higher scaling
        else:
            return 0.8 + (raw_score - 0.6) * 0.5   # High scores -> cap at realistic levels
    
    def _validate_model(self, sequences, ground_truth_labels):
        """Validate model performance against ground truth"""
        logger.info("Validating model against ground truth...")
        
        predictions = []
        actuals = []
        
        for i, seq in enumerate(sequences):
            if i < len(ground_truth_labels):
                prob = self.predict_attack_probability(seq)
                pred = prob > 0.5
                
                predictions.append(pred)
                actuals.append(ground_truth_labels[i])
        
        if predictions and actuals:
            precision = precision_score(actuals, predictions, zero_division=0)
            recall = recall_score(actuals, predictions, zero_division=0)
            f1 = f1_score(actuals, predictions, zero_division=0)
            
            logger.info(f"Validation Results - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

class EnhancedAlertProcessor:
    """Enhanced processor with better feature extraction"""
    
    def __init__(self):
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 
            'psexec.exe', 'rundll32.exe', 'regsvr32.exe', 'sc.exe',
            'at.exe', 'schtasks.exe', 'whoami.exe', 'systeminfo.exe'
        }
        
        self.suspicious_ports = {135, 139, 445, 3389, 22, 1433, 1521, 5985, 5986}
        self.internal_ip_patterns = ['10.', '192.168.', '172.16.', '172.17.', '172.18.', '172.19.']
        
        # Time-based anomaly detection
        self.normal_hours = list(range(8, 18))  # 8 AM to 6 PM
        self.normal_days = list(range(0, 5))    # Monday to Friday
    
    def extract_features(self, alerts: List[Dict]) -> AlertFeatures:
        """Enhanced feature extraction with better normalization"""
        if not alerts:
            return AlertFeatures(0, 0, 0, 0, 0, 0)
        
        # Extract valid timestamps
        timestamps = self._extract_timestamps(alerts)
        
        # Feature calculations with improved logic
        temporal_anomaly = self._analyze_temporal_patterns(timestamps)
        multi_vector = self._calculate_multi_vector_score(alerts)
        privilege_escalation = self._detect_privilege_escalation(alerts)
        lateral_movement = self._detect_lateral_movement(alerts)
        data_access = self._detect_data_access(alerts)
        persistence = self._detect_persistence(alerts)
        
        return AlertFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector,
            privilege_escalation=privilege_escalation,
            lateral_movement=lateral_movement,
            data_access=data_access,
            persistence=persistence
        )
    
    def _extract_timestamps(self, alerts: List[Dict]) -> List[datetime]:
        """Extract and validate timestamps"""
        timestamps = []
        for alert in alerts:
            try:
                if 'timestamp' in alert:
                    ts = datetime.fromisoformat(alert['timestamp'])
                    timestamps.append(ts)
            except (ValueError, TypeError):
                continue
        return timestamps
    
    def _analyze_temporal_patterns(self, timestamps: List[datetime]) -> float:
        """Enhanced temporal anomaly detection"""
        if len(timestamps) < 2:
            return 0.0
        
        try:
            off_hours_score = 0.0
            rapid_sequence_score = 0.0
            
            # Off-hours analysis
            off_hours_count = 0
            for ts in timestamps:
                is_off_hours = (ts.weekday() not in self.normal_days or 
                               ts.hour not in self.normal_hours)
                if is_off_hours:
                    off_hours_count += 1
            
            off_hours_score = off_hours_count / len(timestamps)
            
            # Rapid sequence analysis
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps)-1)]
                
                # Count alerts within rapid windows
                rapid_count = sum(1 for diff in time_diffs if diff < 30)  # < 30 seconds
                burst_count = sum(1 for diff in time_diffs if diff < 5)   # < 5 seconds
                
                rapid_sequence_score = (rapid_count * 0.3 + burst_count * 0.7) / max(len(time_diffs), 1)
            
            # Combined temporal anomaly score
            temporal_score = off_hours_score * 0.4 + rapid_sequence_score * 0.6
            
            return min(temporal_score, 1.0)
            
        except Exception as e:
            logger.debug(f"Temporal analysis failed: {e}")
            return 0.0
    
    def _calculate_multi_vector_score(self, alerts: List[Dict]) -> float:
        """Calculate multi-vector attack score"""
        event_types = set()
        attack_techniques = set()
        
        for alert in alerts:
            # Event type diversity
            event_type = alert.get('event_type', 'unknown')
            event_types.add(event_type)
            
            # Attack technique diversity
            details = alert.get('details', {})
            if 'command_line' in details:
                attack_techniques.add('command_execution')
            if 'remote_ip' in details:
                attack_techniques.add('network_activity')
            if 'file_name' in details:
                attack_techniques.add('file_activity')
        
        # Score based on diversity (more vectors = higher score)
        type_diversity = min(len(event_types) / 4.0, 1.0)  # Normalize by expected max
        technique_diversity = min(len(attack_techniques) / 3.0, 1.0)
        
        return (type_diversity + technique_diversity) / 2.0
    
    def _detect_privilege_escalation(self, alerts: List[Dict]) -> float:
        """Enhanced privilege escalation detection"""
        escalation_score = 0.0
        process_alerts = [a for a in alerts if a.get('event_type') == 'Process']
        
        if not process_alerts:
            return 0.0
        
        high_risk_indicators = 0
        medium_risk_indicators = 0
        
        for alert in process_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            command_line = details.get('command_line', '').lower()
            
            # High-risk indicators
            if any(proc in file_name for proc in ['psexec', 'wmic', 'sc.exe', 'at.exe']):
                high_risk_indicators += 1
            
            if any(cmd in command_line for cmd in ['runas', 'administrator', 'elevated', 'bypass', 'uac']):
                high_risk_indicators += 1
            
            # Medium-risk indicators
            if any(proc in file_name for proc in self.suspicious_processes):
                medium_risk_indicators += 1
            
            if any(cmd in command_line for cmd in ['net user', 'net group', 'whoami', 'systeminfo']):
                medium_risk_indicators += 1
        
        # Calculate escalation score
        total_indicators = high_risk_indicators * 2 + medium_risk_indicators
        escalation_score = min(total_indicators / (len(process_alerts) * 2), 1.0)
        
        return escalation_score
    
    def _detect_lateral_movement(self, alerts: List[Dict]) -> float:
        """Enhanced lateral movement detection"""
        network_alerts = [a for a in alerts if a.get('event_type') == 'Network']
        if not network_alerts:
            return 0.0
        
        lateral_score = 0.0
        internal_connections = 0
        suspicious_port_connections = 0
        unique_internal_ips = set()
        
        for alert in network_alerts:
            details = alert.get('details', {})
            remote_ip = details.get('remote_ip', '')
            remote_port = details.get('remote_port', 0)
            
            # Check for internal network communication
            is_internal = any(remote_ip.startswith(pattern) for pattern in self.internal_ip_patterns)
            
            if is_internal:
                internal_connections += 1
                unique_internal_ips.add(remote_ip)
                
                # Check for suspicious ports on internal networks
                if remote_port in self.suspicious_ports:
                    suspicious_port_connections += 1
        
        if len(network_alerts) > 0:
            internal_ratio = internal_connections / len(network_alerts)
            suspicious_port_ratio = suspicious_port_connections / len(network_alerts)
            ip_diversity = min(len(unique_internal_ips) / 5.0, 1.0)  # More IPs = more lateral movement
            
            lateral_score = (internal_ratio * 0.3 + suspicious_port_ratio * 0.4 + ip_diversity * 0.3)
        
        return min(lateral_score, 1.0)
    
    def _detect_data_access(self, alerts: List[Dict]) -> float:
        """Enhanced data access and exfiltration detection"""
        file_alerts = [a for a in alerts if a.get('event_type') == 'File']
        if not file_alerts:
            return 0.0
        
        data_score = 0.0
        sensitive_file_access = 0
        bulk_operations = 0
        sensitive_extensions = {'.docx', '.pdf', '.xlsx', '.csv', '.txt', '.db', '.sql', '.config'}
        sensitive_paths = {'documents', 'desktop', 'downloads', 'temp', 'system32'}
        
        for alert in file_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            action = details.get('action', '')
            file_path = details.get('file_path', '').lower()
            
            # Sensitive file types
            if any(ext in file_name for ext in sensitive_extensions):
                sensitive_file_access += 1
            
            # Sensitive paths
            if any(path in file_path for path in sensitive_paths):
                sensitive_file_access += 0.5
            
            # Bulk operations (rapid file access)
            if action in ['FileDeleted', 'FileAccessed', 'FileCopied']:
                bulk_operations += 1
        
        if len(file_alerts) > 0:
            sensitive_ratio = sensitive_file_access / len(file_alerts)
            bulk_ratio = bulk_operations / len(file_alerts)
            
            data_score = (sensitive_ratio * 0.6 + bulk_ratio * 0.4)
        
        return min(data_score, 1.0)
    
    def _detect_persistence(self, alerts: List[Dict]) -> float:
        """Enhanced persistence mechanism detection"""
        persistence_score = 0.0
        total_alerts = len(alerts)
        
        if total_alerts == 0:
            return 0.0
        
        persistence_indicators = 0
        high_persistence_indicators = 0
        
        for alert in alerts:
            event_type = alert.get('event_type')
            details = alert.get('details', {})
            
            if event_type == 'Process':
                command_line = details.get('command_line', '').lower()
                file_name = details.get('file_name', '').lower()
                
                # High-persistence indicators
                if any(cmd in command_line for cmd in ['schtasks /create', 'sc create', 'reg add']):
                    high_persistence_indicators += 1
                
                # Medium-persistence indicators
                if any(cmd in command_line for cmd in ['schtasks', 'startup', 'registry', 'service']):
                    persistence_indicators += 1
                
                if 'startup' in file_name or 'autorun' in file_name:
                    persistence_indicators += 1
            
            elif event_type == 'File':
                file_name = details.get('file_name', '').lower()
                file_path = details.get('file_path', '').lower()
                
                # Persistence-related file locations
                if any(path in file_path for path in ['startup', 'system32', 'autorun', 'run']):
                    persistence_indicators += 1
        
        # Calculate persistence score
        total_persistence = high_persistence_indicators * 2 + persistence_indicators
        persistence_score = min(total_persistence / max(total_alerts, 1), 1.0)
        
        return persistence_score

class FixedSecurityAnalyzer:
    """Fixed main analyzer with proper data handling and validation"""
    
    def __init__(self):
        self.processor = EnhancedAlertProcessor()
        self.hmm = FixedSecurityHMM()
        self.sessions_data = None
        self.feature_sequences = []
        self.session_metadata = []
    
    def load_data(self, alerts_file: str, sessions_file: str):
        """Load and validate alerts and sessions data"""
        logger.info("Loading security data...")
        
        try:
            with open(alerts_file, 'r') as f:
                self.alerts_data = json.load(f)
            
            with open(sessions_file, 'r') as f:
                self.sessions_data = json.load(f)
            
            logger.info(f"Loaded {len(self.alerts_data)} alerts and {len(self.sessions_data)} sessions")
            
            # Validate data quality
            self._validate_data_quality()
            
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def _validate_data_quality(self):
        """Validate the quality of loaded data"""
        logger.info("Validating data quality...")
        
        # Check sessions data
        valid_sessions = 0
        attack_sessions = 0
        
        for session_id, session in self.sessions_data.items():
            if 'alerts' in session and len(session['alerts']) > 0:
                valid_sessions += 1
                if session.get('is_attack_session', False):
                    attack_sessions += 1
        
        logger.info(f"Valid sessions: {valid_sessions}/{len(self.sessions_data)}")
        logger.info(f"Attack sessions: {attack_sessions}/{valid_sessions}")
        
        if attack_sessions == 0:
            logger.warning("No attack sessions found in data - model may have limited attack detection capability")
    
    def load_csv_data(self, csv_file: str):
        """Load data from CSV file and convert to expected format"""
        logger.info(f"Loading data from CSV: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} sessions from CSV")
            
            # Convert CSV to expected format
            self.sessions_data = {}
            self.alerts_data = []
            
            for _, row in df.iterrows():
                session_id = row['session_id']
                
                # Create synthetic alerts based on CSV data
                alerts = self._create_synthetic_alerts_from_csv(row)
                
                # Determine if this is an attack session based on attack_probability
                is_attack = row.get('attack_probability', 0) > 0.7
                
                self.sessions_data[session_id] = {
                    'alerts': alerts,
                    'is_attack_session': is_attack,
                    'risk_score': row.get('original_risk_score', 0),
                    'alert_count': row.get('alert_count', len(alerts))
                }
                
                self.alerts_data.extend(alerts)
            
            logger.info(f"Converted to {len(self.sessions_data)} sessions with {len(self.alerts_data)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def _create_synthetic_alerts_from_csv(self, row) -> List[Dict]:
        """Create synthetic alerts based on CSV row data"""
        alerts = []
        session_id = row['session_id']
        alert_count = int(row.get('alert_count', 10))
        attack_prob = float(row.get('attack_probability', 0))
        
        base_time = datetime.now()
        
        for i in range(alert_count):
            # Create alert with characteristics based on attack probability
            alert_time = base_time.replace(minute=i % 60, second=(i * 10) % 60)
            
            if attack_prob > 0.7:
                # High-risk session - create attack-like alerts
                alert = self._create_attack_alert(alert_time, i, session_id)
            elif attack_prob > 0.3:
                # Medium-risk session - create suspicious alerts
                alert = self._create_suspicious_alert(alert_time, i, session_id)
            else:
                # Low-risk session - create normal alerts
                alert = self._create_normal_alert(alert_time, i, session_id)
            
            alerts.append(alert)
        
        return alerts
    
    def _create_attack_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create an attack-like alert"""
        attack_patterns = [
            {
                'event_type': 'Process',
                'details': {
                    'file_name': 'powershell.exe',
                    'command_line': 'powershell.exe -ExecutionPolicy Bypass -EncodedCommand',
                    'process_id': 1234 + index
                }
            },
            {
                'event_type': 'Network',
                'details': {
                    'remote_ip': f'10.0.0.{50 + index % 50}',
                    'remote_port': 445,
                    'direction': 'outbound'
                }
            },
            {
                'event_type': 'File',
                'details': {
                    'file_name': f'confidential_document_{index}.docx',
                    'action': 'FileAccessed',
                    'file_path': 'C:\\Users\\admin\\Documents\\'
                }
            }
        ]
        
        pattern = attack_patterns[index % len(attack_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        pattern['severity'] = 'High'
        
        return pattern
    
    def _create_suspicious_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create a suspicious alert"""
        suspicious_patterns = [
            {
                'event_type': 'Process',
                'details': {
                    'file_name': 'cmd.exe',
                    'command_line': 'cmd.exe /c whoami',
                    'process_id': 2000 + index
                }
            },
            {
                'event_type': 'Network',
                'details': {
                    'remote_ip': f'192.168.1.{100 + index % 50}',
                    'remote_port': 3389,
                    'direction': 'inbound'
                }
            },
            {
                'event_type': 'File',
                'details': {
                    'file_name': f'temp_file_{index}.txt',
                    'action': 'FileCreated',
                    'file_path': 'C:\\Windows\\Temp\\'
                }
            }
        ]
        
        pattern = suspicious_patterns[index % len(suspicious_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        pattern['severity'] = 'Medium'
        
        return pattern
    
    def _create_normal_alert(self, timestamp: datetime, index: int, session_id: str) -> Dict:
        """Create a normal alert"""
        normal_patterns = [
            {
                'event_type': 'Process',
                'details': {
                    'file_name': 'notepad.exe',
                    'command_line': 'notepad.exe document.txt',
                    'process_id': 3000 + index
                }
            },
            {
                'event_type': 'Network',
                'details': {
                    'remote_ip': '8.8.8.8',
                    'remote_port': 80,
                    'direction': 'outbound'
                }
            },
            {
                'event_type': 'File',
                'details': {
                    'file_name': f'report_{index}.pdf',
                    'action': 'FileOpened',
                    'file_path': 'C:\\Users\\user\\Documents\\'
                }
            }
        ]
        
        pattern = normal_patterns[index % len(normal_patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        pattern['severity'] = 'Low'
        
        return pattern
    
    def preprocess_sessions(self, min_alerts=3):
        """Enhanced session preprocessing with proper validation"""
        logger.info("Preprocessing sessions into feature sequences...")
        
        feature_sequences = []
        session_metadata = []
        
        for session_id, session in self.sessions_data.items():
            alerts = session.get('alerts', [])
            
            # Skip sessions with too few alerts
            if len(alerts) < min_alerts:
                continue
            
            try:
                # Create time windows
                alert_windows = self._create_time_windows(alerts, window_minutes=5)
                
                if not alert_windows:
                    continue
                
                # Extract features for each window
                window_features = []
                for window_alerts in alert_windows:
                    if window_alerts:
                        features = self.processor.extract_features(window_alerts)
                        feature_vector = [
                            features.temporal_anomaly,
                            features.multi_vector,
                            features.privilege_escalation,
                            features.lateral_movement,
                            features.data_access,
                            features.persistence
                        ]
                        
                        # Validate feature vector
                        if all(isinstance(f, (int, float)) and not np.isnan(f) for f in feature_vector):
                            window_features.append(feature_vector)
                
                # Only include sequences with enough windows
                if len(window_features) >= 2:
                    feature_sequences.append(torch.tensor(window_features, dtype=torch.float32))
                    session_metadata.append({
                        'session_id': session_id,
                        'is_attack_session': session.get('is_attack_session', False),
                        'risk_score': session.get('risk_score', 0),
                        'alert_count': len(alerts),
                        'window_count': len(window_features)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process session {session_id}: {e}")
                continue
        
        self.feature_sequences = feature_sequences
        self.session_metadata = session_metadata
        
        logger.info(f"Created {len(feature_sequences)} valid feature sequences")
        
        # Log data distribution
        attack_count = sum(1 for m in session_metadata if m['is_attack_session'])
        logger.info(f"Attack sessions: {attack_count}/{len(session_metadata)} ({attack_count/len(session_metadata)*100:.1f}%)")
        
        return feature_sequences, session_metadata
    
    def _create_time_windows(self, alerts: List[Dict], window_minutes: int = 5) -> List[List[Dict]]:
        """Enhanced time window creation with better error handling"""
        if not alerts:
            return []
        
        # Sort alerts by timestamp
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
        
        windows = []
        current_window = []
        current_window_start = None
        
        for alert_time, alert in valid_alerts:
            if current_window_start is None:
                current_window_start = alert_time
                current_window = [alert]
            elif (alert_time - current_window_start).total_seconds() <= window_minutes * 60:
                current_window.append(alert)
            else:
                # Start new window
                if current_window:
                    windows.append(current_window)
                current_window = [alert]
                current_window_start = alert_time
        
        # Add final window
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def train_hmm(self, num_iterations=100):
        """Train the HMM with proper validation"""
        if not self.feature_sequences:
            raise ValueError("Must preprocess sessions before training")
        
        logger.info("Training Enhanced HMM...")
        
        # Extract ground truth labels for validation
        ground_truth = [m['is_attack_session'] for m in self.session_metadata]
        
        # Train the model
        self.hmm.fit(self.feature_sequences, ground_truth_labels=ground_truth, num_iterations=num_iterations)
        
        return [0]  # Dummy loss for compatibility
    
    def analyze_session(self, session_id: str) -> Dict:
        """Enhanced session analysis with proper error handling"""
        if not self.hmm.trained:
            raise ValueError("HMM must be trained before analysis")
        
        try:
            session = self.sessions_data.get(session_id)
            if not session:
                return {
                    'session_id': session_id,
                    'attack_probability': 0.0,
                    'error': 'Session not found'
                }
            
            alerts = session.get('alerts', [])
            if not alerts:
                return {
                    'session_id': session_id,
                    'attack_probability': 0.0,
                    'predicted_states': [],
                    'state_sequence': [],
                    'confidence': 0.0
                }
            
            # Create time windows and extract features
            alert_windows = self._create_time_windows(alerts, window_minutes=5)
            window_features = []
            
            for window_alerts in alert_windows:
                if window_alerts:
                    features = self.processor.extract_features(window_alerts)
                    feature_vector = [
                        features.temporal_anomaly,
                        features.multi_vector,
                        features.privilege_escalation,
                        features.lateral_movement,
                        features.data_access,
                        features.persistence
                    ]
                    
                    # Validate features
                    if all(isinstance(f, (int, float)) and not np.isnan(f) for f in feature_vector):
                        window_features.append(feature_vector)
            
            if not window_features:
                return {
                    'session_id': session_id,
                    'attack_probability': 0.0,
                    'predicted_states': [],
                    'state_sequence': [],
                    'confidence': 0.0
                }
            
            # Convert to tensor and analyze
            observations = torch.tensor(window_features, dtype=torch.float32)
            
            # Get attack probability (this is the key fix)
            attack_probability = self.hmm.predict_attack_probability(observations)
            
            # Get state sequence
            predicted_states = self.hmm.viterbi_decode(observations)
            state_sequence = [self.hmm.state_names[state] for state in predicted_states]
            
            # Calculate confidence based on consistency
            confidence = self._calculate_confidence(predicted_states, attack_probability)
            
            return {
                'session_id': session_id,
                'attack_probability': float(attack_probability),
                'predicted_states': predicted_states.tolist() if len(predicted_states) > 0 else [],
                'state_sequence': state_sequence,
                'confidence': float(confidence),
                'window_count': len(window_features),
                'attack_progression': self._analyze_attack_progression(state_sequence)
            }
            
        except Exception as e:
            logger.warning(f"Session analysis failed for {session_id}: {e}")
            return {
                'session_id': session_id,
                'attack_probability': 0.0,
                'predicted_states': [],
                'state_sequence': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, predicted_states, attack_probability):
        """Calculate confidence score based on state consistency"""
        if len(predicted_states) == 0:
            return 0.0
        
        # Confidence based on state consistency and probability
        state_consistency = len(set(predicted_states)) / len(predicted_states)  # Lower is more consistent
        confidence = (1 - state_consistency) * 0.7 + min(attack_probability * 2, 1.0) * 0.3
        
        return min(confidence, 1.0)
    
    def _analyze_attack_progression(self, state_sequence: List[str]) -> Dict:
        """Enhanced attack progression analysis"""
        progression = {
            'stages_detected': list(set(state_sequence)),
            'kill_chain_coverage': 0.0,
            'progression_timeline': [],
            'persistence': False,
            'escalation_detected': False
        }
        
        attack_stages = ['reconnaissance', 'initial_access', 'lateral_movement', 'objective_execution']
        detected_attack_stages = [s for s in progression['stages_detected'] if s in attack_stages]
        
        progression['kill_chain_coverage'] = len(detected_attack_stages) / len(attack_stages)
        
        # Track stage transitions and detect escalation
        prev_stage_level = 0
        for i, state in enumerate(state_sequence):
            stage_level = attack_stages.index(state) if state in attack_stages else -1
            
            if state != 'normal_operations':
                progression['progression_timeline'].append({
                    'window': i,
                    'stage': state,
                    'severity': stage_level
                })
                
                # Check for escalation
                if stage_level > prev_stage_level:
                    progression['escalation_detected'] = True
                
                prev_stage_level = max(prev_stage_level, stage_level)
        
        # Check for persistence (returning to attack states after normal)
        for i in range(1, len(state_sequence)):
            if (state_sequence[i-1] == 'normal_operations' and 
                state_sequence[i] in attack_stages):
                progression['persistence'] = True
                break
        
        return progression
    
    def generate_intelligence_report(self, top_n: int = 10) -> Dict:
        """Generate comprehensive intelligence report with proper validation"""
        logger.info("Generating enhanced intelligence report...")
        
        attack_sessions = []
        normal_sessions = []
        analysis_errors = []
        
        for metadata in self.session_metadata:
            session_id = metadata['session_id']
            try:
                analysis = self.analyze_session(session_id)
                analysis.update(metadata)
                
                if 'error' in analysis:
                    analysis_errors.append(analysis)
                elif analysis['attack_probability'] > 0.15:  # Lower threshold for initial filtering
                    attack_sessions.append(analysis)
                else:
                    normal_sessions.append(analysis)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze session {session_id}: {e}")
                analysis_errors.append({
                    'session_id': session_id,
                    'error': str(e)
                })
        
        # Sort by attack probability
        attack_sessions.sort(key=lambda x: x['attack_probability'], reverse=True)
        
        # Generate summary statistics
        total_sessions = len(self.session_metadata)
        high_risk_sessions = len([s for s in attack_sessions if s['attack_probability'] > 0.7])
        medium_risk_sessions = len([s for s in attack_sessions if 0.3 < s['attack_probability'] <= 0.7])
        low_risk_sessions = len([s for s in attack_sessions if 0.15 < s['attack_probability'] <= 0.3])
        
        # Calculate model performance if we have ground truth
        model_performance = self._calculate_model_performance(attack_sessions + normal_sessions)
        
        report = {
            'summary': {
                'total_sessions_analyzed': total_sessions,
                'analysis_errors': len(analysis_errors),
                'suspicious_sessions': len(attack_sessions),
                'high_risk_sessions': high_risk_sessions,
                'medium_risk_sessions': medium_risk_sessions,
                'low_risk_sessions': low_risk_sessions,
                'normal_sessions': len(normal_sessions),
                'detection_rate': len(attack_sessions) / total_sessions if total_sessions > 0 else 0
            },
            'model_performance': model_performance,
            'top_threats': attack_sessions[:top_n],
            'attack_patterns': self._analyze_attack_patterns(attack_sessions),
            'recommendations': self._generate_recommendations(attack_sessions),
            'analysis_errors': analysis_errors[:5]  # Include first 5 errors for debugging
        }
        
        return report
    
    def _calculate_model_performance(self, analyzed_sessions: List[Dict]) -> Dict:
        """Calculate model performance metrics"""
        if not analyzed_sessions:
            return {}
        
        y_true = []
        y_pred = []
        y_prob = []
        
        for session in analyzed_sessions:
            if 'is_attack_session' in session and 'attack_probability' in session:
                y_true.append(session['is_attack_session'])
                y_prob.append(session['attack_probability'])
                y_pred.append(session['attack_probability'] > 0.5)
        
        if not y_true:
            return {}
        
        try:
            performance = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'accuracy': sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
            }
            
            # Add AUC if we have both classes
            if len(set(y_true)) > 1:
                performance['auc_roc'] = roc_auc_score(y_true, y_prob)
            
            return performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def _analyze_attack_patterns(self, attack_sessions: List[Dict]) -> Dict:
        """Enhanced attack pattern analysis"""
        patterns = {
            'common_progressions': defaultdict(int),
            'avg_attack_duration': 0.0,
            'most_common_stages': defaultdict(int),
            'persistence_rate': 0.0,
            'escalation_rate': 0.0,
            'attack_probability_distribution': {}
        }
        
        if not attack_sessions:
            return patterns
        
        total_duration = 0
        persistence_count = 0
        escalation_count = 0
        prob_ranges = {'0.15-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
        
        for session in attack_sessions:
            # Probability distribution
            prob = session['attack_probability']
            if prob < 0.3:
                prob_ranges['0.15-0.3'] += 1
            elif prob < 0.5:
                prob_ranges['0.3-0.5'] += 1
            elif prob < 0.7:
                prob_ranges['0.5-0.7'] += 1
            elif prob < 0.9:
                prob_ranges['0.7-0.9'] += 1
            else:
                prob_ranges['0.9-1.0'] += 1
            
            # Progression analysis
            if 'state_sequence' in session:
                state_seq = session['state_sequence']
                progression_key = ' → '.join(set(state_seq))
                patterns['common_progressions'][progression_key] += 1
                
                # Count individual stages
                for stage in state_seq:
                    patterns['most_common_stages'][stage] += 1
            
            # Attack characteristics
            if 'attack_progression' in session:
                progression = session['attack_progression']
                if progression.get('persistence', False):
                    persistence_count += 1
                if progression.get('escalation_detected', False):
                    escalation_count += 1
            
            # Duration
            total_duration += session.get('window_count', 0)
        
        patterns['avg_attack_duration'] = total_duration / len(attack_sessions)
        patterns['persistence_rate'] = persistence_count / len(attack_sessions)
        patterns['escalation_rate'] = escalation_count / len(attack_sessions)
        patterns['attack_probability_distribution'] = prob_ranges
        
        return patterns
    
    def _generate_recommendations(self, attack_sessions: List[Dict]) -> List[str]:
        """Generate enhanced security recommendations"""
        recommendations = []
        
        if not attack_sessions:
            return ["✅ No significant threats detected. Continue monitoring with current security posture."]
        
        high_risk_count = len([s for s in attack_sessions if s['attack_probability'] > 0.7])
        medium_risk_count = len([s for s in attack_sessions if 0.3 < s['attack_probability'] <= 0.7])
        
        # Critical alerts
        if high_risk_count > 0:
            recommendations.append(
                f"🚨 CRITICAL: Immediate investigation required for {high_risk_count} high-risk sessions (>70% attack probability)"
            )
        
        if medium_risk_count > 0:
            recommendations.append(
                f"⚠️ HIGH PRIORITY: Review {medium_risk_count} medium-risk sessions (30-70% attack probability)"
            )
        
        # Pattern-based recommendations
        all_stages = []
        persistence_count = 0
        escalation_count = 0
        
        for session in attack_sessions:
            if 'state_sequence' in session:
                all_stages.extend([s for s in session['state_sequence'] if s != 'normal_operations'])
            
            if 'attack_progression' in session:
                if session['attack_progression'].get('persistence', False):
                    persistence_count += 1
                if session['attack_progression'].get('escalation_detected', False):
                    escalation_count += 1
        
        # Stage-specific recommendations
        if all_stages:
            stage_counts = defaultdict(int)
            for stage in all_stages:
                stage_counts[stage] += 1
            
            total_sessions = len(attack_sessions)
            
            if stage_counts['lateral_movement'] > total_sessions * 0.4:
                recommendations.append(
                    "🔒 NETWORK: Implement network segmentation - lateral movement detected in >40% of suspicious sessions"
                )
            
            if stage_counts['reconnaissance'] > total_sessions * 0.3:
                recommendations.append(
                    "👁️ MONITORING: Enhance reconnaissance detection - suspicious scanning activities identified"
                )
            
            if stage_counts['objective_execution'] > total_sessions * 0.2:
                recommendations.append(
                    "🎯 RESPONSE: Active attack objectives detected - review data access controls and backup integrity"
                )
        
        # Behavioral recommendations
        if persistence_count > len(attack_sessions) * 0.25:
            recommendations.append(
                "🛡️ HARDENING: High persistence rate detected - review startup processes, scheduled tasks, and registry entries"
            )
        
        if escalation_count > len(attack_sessions) * 0.3:
            recommendations.append(
                "🔐 PRIVILEGE: Implement stricter privilege controls - privilege escalation detected in multiple sessions"
            )
        
        # General recommendations
        recommendations.extend([
            f"📊 ANALYSIS: Monitored {len(attack_sessions)} suspicious sessions out of total analyzed",
            "🔄 CONTINUOUS: Maintain enhanced monitoring and update threat intelligence regularly"
        ])
        
        return recommendations
    
    def export_results(self, output_file='enhanced_security_intelligence_report.json'):
        """Export comprehensive analysis results"""
        logger.info("Generating and exporting enhanced intelligence report...")
        
        report = self.generate_intelligence_report()
        
        # Add model parameters and metadata
        if self.hmm.trained:
            report['model_metadata'] = {
                'model_type': 'FixedSecurityHMM_Enhanced',
                'state_names': self.hmm.state_names,
                'state_attack_weights': self.hmm.state_attack_weights,
                'feature_names': [
                    'temporal_anomaly', 'multi_vector', 'privilege_escalation',
                    'lateral_movement', 'data_access', 'persistence'
                ],
                'numerical_improvements': [
                    'Calibrated attack probability calculation',
                    'Enhanced feature normalization',
                    'Improved Viterbi algorithm stability',
                    'Domain-knowledge transition matrix',
                    'Regularized emission models'
                ]
            }
            
            if self.hmm.transition_probs is not None:
                report['model_metadata']['transition_probabilities'] = self.hmm.transition_probs.tolist()
        
        # Add detailed session analyses
        detailed_analyses = []
        for metadata in self.session_metadata:
            session_id = metadata['session_id']
            analysis = self.analyze_session(session_id)
            analysis.update(metadata)
            detailed_analyses.append(analysis)
        
        report['detailed_session_analyses'] = detailed_analyses
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Enhanced intelligence report exported to {output_file}")
        return report
    
    def visualize_results(self, save_results=True):
        """Create comprehensive pandas-based analysis and visualizations"""
        try:
            print("\n" + "="*80)
            print("📊 ENHANCED SECURITY HMM ANALYSIS RESULTS")
            print("="*80)
            
            # Analyze all sessions
            session_results = []
            for metadata in self.session_metadata:
                session_id = metadata['session_id']
                analysis = self.analyze_session(session_id)
                
                session_results.append({
                    'session_id': session_id,
                    'attack_probability': analysis['attack_probability'],
                    'hmm_prediction': analysis['attack_probability'] > 0.5,
                    'original_risk_score': metadata['risk_score'],
                    'ground_truth': metadata['is_attack_session'],
                    'alert_count': metadata['alert_count'],
                    'window_count': analysis.get('window_count', 0),
                    'confidence': analysis.get('confidence', 0),
                    'dominant_state': max(set(analysis.get('state_sequence', ['normal_operations'])), 
                                        key=analysis.get('state_sequence', ['normal_operations']).count)
                })
            
            results_df = pd.DataFrame(session_results)
            
            # 1. Model Performance Analysis
            print("\n🎯 MODEL PERFORMANCE METRICS:")
            print("-" * 40)
            
            if 'ground_truth' in results_df.columns and results_df['ground_truth'].notna().any():
                # Calculate performance metrics
                y_true = results_df['ground_truth'].astype(bool)
                y_pred = results_df['hmm_prediction'].astype(bool)
                y_prob = results_df['attack_probability']
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                accuracy = (y_true == y_pred).mean()
                
                print(f"   Precision: {precision:.3f}")
                print(f"   Recall: {recall:.3f}")
                print(f"   F1-Score: {f1:.3f}")
                print(f"   Accuracy: {accuracy:.3f}")
                
                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                print(f"\n   Confusion Matrix:")
                print(f"   True Negatives:  {tn:4d}  False Positives: {fp:4d}")
                print(f"   False Negatives: {fn:4d}  True Positives:  {tp:4d}")
                
                # AUC if we have both classes
                if len(set(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_prob)
                    print(f"   AUC-ROC: {auc:.3f}")
            
            # 2. Attack Probability Distribution
            print(f"\n📈 ATTACK PROBABILITY DISTRIBUTION:")
            print("-" * 40)
            
            prob_stats = results_df['attack_probability'].describe()
            print(f"   Mean:   {prob_stats['mean']:.3f}")
            print(f"   Median: {prob_stats['50%']:.3f}")
            print(f"   Std:    {prob_stats['std']:.3f}")
            print(f"   Min:    {prob_stats['min']:.3f}")
            print(f"   Max:    {prob_stats['max']:.3f}")
            
            # Distribution visualization
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            bin_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
            
            print(f"\n   Distribution by Risk Level:")
            for i in range(len(bins)-1):
                mask = (results_df['attack_probability'] >= bins[i]) & (results_df['attack_probability'] < bins[i+1])
                count = mask.sum()
                percentage = count / len(results_df) * 100
                bar = '█' * max(1, int(percentage / 3))
                print(f"   {bin_labels[i]:>8}: {count:3d} sessions {bar:20s} ({percentage:.1f}%)")
            
            # 3. State Analysis
            print(f"\n🔄 ATTACK STATE ANALYSIS:")
            print("-" * 40)
            
            state_counts = results_df['dominant_state'].value_counts()
            print("   Most Common States:")
            for state, count in state_counts.head().items():
                percentage = count / len(results_df) * 100
                print(f"     • {state:20s}: {count:3d} sessions ({percentage:.1f}%)")
            
            # 4. Correlation Analysis
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print("-" * 40)
            
            # HMM vs Original Risk Score
            results_df['normalized_risk'] = results_df['original_risk_score'] / 100.0
            correlation = results_df['attack_probability'].corr(results_df['normalized_risk'])
            print(f"   HMM vs Original Risk Score: {correlation:.3f}")
            
            # Alert count vs Attack probability
            alert_corr = results_df['attack_probability'].corr(results_df['alert_count'])
            print(f"   Alert Count vs Attack Prob: {alert_corr:.3f}")
            
            # 5. Model Calibration Analysis
            print(f"\n⚖️ MODEL CALIBRATION ANALYSIS:")
            print("-" * 40)
            
            # Check for over-confident predictions
            high_conf_high_prob = ((results_df['attack_probability'] > 0.8) & 
                                  (results_df['confidence'] > 0.8)).sum()
            high_conf_low_prob = ((results_df['attack_probability'] < 0.2) & 
                                 (results_df['confidence'] > 0.8)).sum()
            
            print(f"   High confidence + High probability: {high_conf_high_prob} sessions")
            print(f"   High confidence + Low probability:  {high_conf_low_prob} sessions")
            
            # 6. Top Threats Summary
            print(f"\n🚨 TOP THREAT SESSIONS:")
            print("-" * 40)
            
            top_threats = results_df.nlargest(5, 'attack_probability')
            for idx, row in top_threats.iterrows():
                print(f"   {row['session_id']:20s}: {row['attack_probability']:.1%} "
                      f"(Confidence: {row['confidence']:.1%}, State: {row['dominant_state']})")
            
            # 7. Feature Importance (if transition matrix available)
            if self.hmm.trained and self.hmm.transition_probs is not None:
                print(f"\n🔄 TRANSITION MATRIX ANALYSIS:")
                print("-" * 40)
                
                transition_df = pd.DataFrame(
                    self.hmm.transition_probs,
                    index=self.hmm.state_names,
                    columns=self.hmm.state_names
                )
                
                print("   Key Transitions (>30% probability):")
                for i, from_state in enumerate(self.hmm.state_names):
                    for j, to_state in enumerate(self.hmm.state_names):
                        prob = transition_df.iloc[i, j]
                        if prob > 0.3 and from_state != to_state:
                            print(f"     {from_state:20s} → {to_state:20s}: {prob:.1%}")
            
            # 8. Save results
            if save_results:
                # Save detailed results
                results_df.to_csv('enhanced_hmm_session_analysis.csv', index=False)
                
                # Save summary statistics
                summary_stats = {
                    'total_sessions': len(results_df),
                    'high_risk_sessions': (results_df['attack_probability'] > 0.7).sum(),
                    'medium_risk_sessions': ((results_df['attack_probability'] > 0.3) & 
                                            (results_df['attack_probability'] <= 0.7)).sum(),
                    'low_risk_sessions': (results_df['attack_probability'] <= 0.3).sum(),
                    'mean_attack_probability': results_df['attack_probability'].mean(),
                    'mean_confidence': results_df['confidence'].mean()
                }
                
                with open('enhanced_hmm_summary_stats.json', 'w') as f:
                    json.dump(summary_stats, f, indent=2)
                
                if self.hmm.trained and self.hmm.transition_probs is not None:
                    transition_df.to_csv('enhanced_hmm_transition_matrix.csv')
                
                print(f"\n💾 RESULTS SAVED:")
                print(f"   • enhanced_hmm_session_analysis.csv")
                print(f"   • enhanced_hmm_summary_stats.json")
                print(f"   • enhanced_hmm_transition_matrix.csv")
            
            print(f"\n✅ Analysis complete! {len(results_df)} sessions processed.")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function with comprehensive testing"""
    print("🔒 ENHANCED Security Alert HMM Analysis")
    print("=" * 60)
    print("🎯 Key Improvements:")
    print("   ✅ Fixed probability calibration")
    print("   ✅ Enhanced feature extraction") 
    print("   ✅ Proper data validation")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Model performance metrics")
    print("   ✅ CSV data loading support")
    print()
    
    # Initialize analyzer
    analyzer = FixedSecurityAnalyzer()
    
    try:
        # Try to load JSON data first, fallback to CSV
        try:
            analyzer.load_data('synthetic_alerts_sample.json', 'synthetic_sessions_sample.json')
        except FileNotFoundError:
            logger.info("JSON files not found, attempting to load from CSV...")
            try:
                analyzer.load_csv_data('paste.txt')  # Load from the provided CSV
            except FileNotFoundError:
                logger.error("No data files found. Please provide either:")
                logger.error("  - synthetic_alerts_sample.json + synthetic_sessions_sample.json")
                logger.error("  - paste.txt (CSV format)")
                return
        
        # Preprocess sessions into feature sequences
        feature_sequences, session_metadata = analyzer.preprocess_sessions(min_alerts=2)
        
        if not feature_sequences:
            print("❌ No valid sessions found for analysis")
            return
        
        print(f"✅ Preprocessed {len(feature_sequences)} sessions")
        print(f"📊 Average sequence length: {np.mean([len(seq) for seq in feature_sequences]):.1f}")
        
        # Check data balance
        attack_count = sum(1 for m in session_metadata if m['is_attack_session'])
        normal_count = len(session_metadata) - attack_count
        print(f"📈 Data distribution: {attack_count} attacks, {normal_count} normal ({attack_count/(attack_count+normal_count)*100:.1f}% attacks)")
        
        # Train HMM
        print("\n🧠 Training Enhanced Hidden Markov Model...")
        losses = analyzer.train_hmm(num_iterations=75)
        
        print(f"✅ Training completed successfully!")
        
        # Test individual session analysis
        print(f"\n🔍 TESTING SESSION ANALYSIS:")
        sample_session = analyzer.session_metadata[0]['session_id']
        analysis = analyzer.analyze_session(sample_session)
        
        print(f"   Sample Session: {sample_session}")
        print(f"   Attack Probability: {analysis['attack_probability']:.1%}")
        print(f"   State Sequence: {' → '.join(analysis['state_sequence'])}")
        print(f"   Confidence: {analysis['confidence']:.1%}")
        
        # Generate intelligence report
        print("\n📋 Generating Enhanced Intelligence Report...")
        report = analyzer.generate_intelligence_report(top_n=5)
        
        # Display summary
        summary = report['summary']
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   Total Sessions: {summary['total_sessions_analyzed']}")
        print(f"   Analysis Errors: {summary['analysis_errors']}")
        print(f"   Suspicious Sessions: {summary['suspicious_sessions']}")
        print(f"   High Risk: {summary['high_risk_sessions']}")
        print(f"   Medium Risk: {summary['medium_risk_sessions']}")
        print(f"   Low Risk: {summary['low_risk_sessions']}")
        print(f"   Detection Rate: {summary['detection_rate']:.1%}")
        
        # Display model performance
        if 'model_performance' in report and report['model_performance']:
            perf = report['model_performance']
            print(f"\n🎯 MODEL PERFORMANCE:")
            for metric, value in perf.items():
                print(f"   {metric.title()}: {value:.3f}")
        
        # Display top threats
        print(f"\n🚨 TOP THREATS:")
        for i, threat in enumerate(report['top_threats'][:3], 1):
            print(f"   {i}. {threat['session_id']}")
            print(f"      Attack Probability: {threat['attack_probability']:.1%}")
            print(f"      Ground Truth: {'ATTACK' if threat.get('is_attack_session', False) else 'NORMAL'}")
            print(f"      Confidence: {threat['confidence']:.1%}")
            print()
        
        # Display recommendations
        print(f"💡 RECOMMENDATIONS:")
        for rec in report['recommendations'][:5]:
            print(f"   • {rec}")
        
        # Export results
        analyzer.export_results('enhanced_security_intelligence_report.json')
        
        # Create comprehensive visualizations
        print(f"\n📊 Generating Comprehensive Analysis...")
        results_df = analyzer.visualize_results(save_results=True)
        
        if results_df is not None:
            print(f"\n🎉 SUCCESS: Enhanced HMM analysis completed!")
            print(f"   • Proper probability calibration implemented")
            print(f"   • Model performance metrics calculated")
            print(f"   • Comprehensive error handling added")
            print(f"   • Results exported for further analysis")
        
        # Validation check
        print(f"\n🔍 VALIDATION CHECK:")
        high_prob_sessions = results_df[results_df['attack_probability'] > 0.7]
        actual_attacks = high_prob_sessions['ground_truth'].sum() if 'ground_truth' in high_prob_sessions.columns else 0
        
        print(f"   Sessions with >70% attack probability: {len(high_prob_sessions)}")
        print(f"   Of these, actual attacks: {actual_attacks}")
        
        if len(high_prob_sessions) > 0:
            accuracy_high_conf = actual_attacks / len(high_prob_sessions)
            print(f"   High-confidence accuracy: {accuracy_high_conf:.1%}")
            
            if accuracy_high_conf > 0.8:
                print("   ✅ Model shows good calibration for high-confidence predictions")
            elif accuracy_high_conf > 0.5:
                print("   ⚠️ Model calibration is moderate - consider threshold adjustment")
            else:
                print("   ❌ Model may need recalibration - high false positive rate")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 TROUBLESHOOTING TIPS:")
        print(f"   • Check data file format and structure")
        print(f"   • Ensure sufficient data for training")
        print(f"   • Verify ground truth labels are available")
        print(f"   • Check for data quality issues")

if __name__ == "__main__":
    main()