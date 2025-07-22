#!/usr/bin/env python3
"""
Simple Security Alert HMM Analysis - ROBUST VERSION
Converts alert noise into actionable attack intelligence using a simplified HMM approach
Based on the AI Cyber Security approach detailed in the article
"""

import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class SimpleSecurityHMM:
    """Simplified Hidden Markov Model for security alert sequence analysis"""
    
    def __init__(self, num_states=5):
        """
        Initialize the Security HMM
        
        Args:
            num_states: Number of hidden attack states
        """
        self.num_states = num_states
        
        # Define attack states based on cyber kill chain
        self.state_names = [
            "normal_operations",
            "reconnaissance", 
            "initial_access",
            "lateral_movement",
            "objective_execution"
        ]
        
        # HMM parameters
        self.transition_probs = None
        self.emission_models = None
        self.initial_probs = None
        self.trained = False
    
    def fit(self, sequences, num_iterations=50):
        """
        Train the HMM using a simplified EM-like approach
        
        Args:
            sequences: List of feature sequences (each sequence is a 2D array)
            num_iterations: Number of training iterations
        """
        logger.info(f"Training Simple HMM with {num_iterations} iterations...")
        
        # Convert sequences to numpy arrays
        all_features = []
        sequence_info = []
        
        for seq_idx, seq in enumerate(sequences):
            seq_array = seq.numpy() if torch.is_tensor(seq) else np.array(seq)
            for t, features in enumerate(seq_array):
                all_features.append(features)
                sequence_info.append((seq_idx, t))
        
        all_features = np.array(all_features)
        
        # Initialize with K-means clustering to get initial state assignments
        logger.info("Initializing states with K-means clustering...")
        kmeans = KMeans(n_clusters=self.num_states, random_state=42, n_init=10)
        initial_states = kmeans.fit_predict(all_features)
        
        # Initialize emission models (Gaussian Mixture for each state)
        self.emission_models = {}
        for state in range(self.num_states):
            state_features = all_features[initial_states == state]
            if len(state_features) > 1:
                # Use GMM for this state
                gmm = GaussianMixture(n_components=1, random_state=42)
                gmm.fit(state_features)
                self.emission_models[state] = gmm
            else:
                # Fallback for states with too few samples
                self.emission_models[state] = {
                    'mean': np.mean(all_features, axis=0),
                    'cov': np.eye(all_features.shape[1]) * 0.1
                }
        
        # Initialize transition probabilities
        self.transition_probs = np.ones((self.num_states, self.num_states)) / self.num_states
        self.initial_probs = np.ones(self.num_states) / self.num_states
        
        # Simple EM-like training
        for iteration in range(num_iterations):
            if iteration % 10 == 0:
                logger.info(f"Training iteration {iteration}/{num_iterations}")
            
            # E-step: Estimate state probabilities for each observation
            state_probs = self._estimate_states(sequences)
            
            # M-step: Update parameters
            self._update_parameters(sequences, state_probs)
        
        self.trained = True
        logger.info("Training completed!")
    
    def _estimate_states(self, sequences):
        """Estimate state probabilities for each observation"""
        state_probs = []
        
        for seq in sequences:
            seq_array = seq.numpy() if torch.is_tensor(seq) else np.array(seq)
            seq_state_probs = []
            
            for features in seq_array:
                # Calculate likelihood for each state
                likelihoods = []
                for state in range(self.num_states):
                    likelihood = self._emission_probability(features, state)
                    likelihoods.append(likelihood)
                
                # Normalize to get probabilities
                likelihoods = np.array(likelihoods)
                if likelihoods.sum() > 0:
                    probs = likelihoods / likelihoods.sum()
                else:
                    probs = np.ones(self.num_states) / self.num_states
                
                seq_state_probs.append(probs)
            
            state_probs.append(np.array(seq_state_probs))
        
        return state_probs
    
    def _emission_probability(self, features, state):
        """Calculate emission probability for features given state"""
        if isinstance(self.emission_models[state], dict):
            # Simple Gaussian
            mean = self.emission_models[state]['mean']
            cov = self.emission_models[state]['cov']
            diff = features - mean
            try:
                prob = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
                return prob / np.sqrt((2 * np.pi) ** len(features) * np.linalg.det(cov))
            except:
                return 1e-6
        else:
            # Gaussian Mixture Model
            try:
                return self.emission_models[state].score_samples([features])[0]
            except:
                return 1e-6
    
    def _update_parameters(self, sequences, state_probs):
        """Update HMM parameters based on current state estimates"""
        # Update emission parameters
        for state in range(self.num_states):
            state_features = []
            state_weights = []
            
            for seq_idx, seq in enumerate(sequences):
                seq_array = seq.numpy() if torch.is_tensor(seq) else np.array(seq)
                seq_state_probs = state_probs[seq_idx]
                
                for t, features in enumerate(seq_array):
                    weight = seq_state_probs[t][state]
                    if weight > 0.1:  # Only include if reasonably probable
                        state_features.append(features)
                        state_weights.append(weight)
            
            if len(state_features) > 1:
                state_features = np.array(state_features)
                state_weights = np.array(state_weights)
                
                # Update emission model
                if isinstance(self.emission_models[state], dict):
                    # Simple Gaussian update
                    weighted_mean = np.average(state_features, weights=state_weights, axis=0)
                    self.emission_models[state]['mean'] = weighted_mean
                    
                    # Simple covariance update
                    cov = np.cov(state_features.T) + np.eye(len(weighted_mean)) * 0.01
                    self.emission_models[state]['cov'] = cov
        
        # Update transition probabilities (simplified)
        transition_counts = np.ones((self.num_states, self.num_states))
        
        for seq_idx, seq in enumerate(sequences):
            seq_state_probs = state_probs[seq_idx]
            
            for t in range(len(seq_state_probs) - 1):
                curr_state_probs = seq_state_probs[t]
                next_state_probs = seq_state_probs[t + 1]
                
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        transition_counts[i, j] += curr_state_probs[i] * next_state_probs[j]
        
        # Normalize transition probabilities
        for i in range(self.num_states):
            if transition_counts[i].sum() > 0:
                self.transition_probs[i] = transition_counts[i] / transition_counts[i].sum()
    
    def viterbi_decode(self, observations):
        """Find most likely state sequence using Viterbi algorithm"""
        if not self.trained:
            raise ValueError("Model must be trained before decoding")
        
        obs_array = observations.numpy() if torch.is_tensor(observations) else np.array(observations)
        seq_len = len(obs_array)
        
        # Initialize
        delta = np.zeros((seq_len, self.num_states))
        psi = np.zeros((seq_len, self.num_states), dtype=int)
        
        # First step
        for s in range(self.num_states):
            delta[0, s] = np.log(self.initial_probs[s]) + np.log(self._emission_probability(obs_array[0], s) + 1e-6)
        
        # Forward pass
        for t in range(1, seq_len):
            for s in range(self.num_states):
                trans_scores = delta[t-1] + np.log(self.transition_probs[:, s] + 1e-6)
                psi[t, s] = np.argmax(trans_scores)
                delta[t, s] = np.max(trans_scores) + np.log(self._emission_probability(obs_array[t], s) + 1e-6)
        
        # Backward pass
        states = np.zeros(seq_len, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(seq_len - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def predict_attack_probability(self, observations):
        """Calculate attack probability for a sequence"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        states = self.viterbi_decode(observations)
        
        # Attack states are 1, 2, 3, 4 (not 0 which is normal)
        attack_states = [1, 2, 3, 4]
        attack_windows = sum(1 for state in states if state in attack_states)
        
        return attack_windows / len(states) if len(states) > 0 else 0.0

class AlertProcessor:
    """Process raw security alerts into HMM-ready features"""
    
    def __init__(self):
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 
            'psexec.exe', 'rundll32.exe', 'regsvr32.exe'
        }
        
        self.suspicious_ports = {135, 139, 445, 3389, 22, 1433, 1521}
        self.internal_ip_patterns = ['10.', '192.168.', '172.16.']
    
    def extract_features(self, alerts: List[Dict]) -> AlertFeatures:
        """Extract features from a sequence of alerts"""
        if not alerts:
            return AlertFeatures(0, 0, 0, 0, 0, 0)
        
        # Temporal analysis
        timestamps = [datetime.fromisoformat(alert['timestamp']) for alert in alerts]
        temporal_anomaly = self._analyze_temporal_patterns(timestamps)
        
        # Multi-vector analysis
        event_types = set(alert['event_type'] for alert in alerts)
        multi_vector = min(len(event_types) / 3.0, 1.0)  # Normalize to 0-1
        
        # Privilege escalation indicators
        privilege_escalation = self._detect_privilege_escalation(alerts)
        
        # Lateral movement indicators
        lateral_movement = self._detect_lateral_movement(alerts)
        
        # Data access patterns
        data_access = self._detect_data_access(alerts)
        
        # Persistence indicators
        persistence = self._detect_persistence(alerts)
        
        return AlertFeatures(
            temporal_anomaly=temporal_anomaly,
            multi_vector=multi_vector,
            privilege_escalation=privilege_escalation,
            lateral_movement=lateral_movement,
            data_access=data_access,
            persistence=persistence
        )
    
    def _analyze_temporal_patterns(self, timestamps: List[datetime]) -> float:
        """Analyze temporal anomalies in alert timing"""
        if len(timestamps) < 2:
            return 0.0
        
        # Check for off-hours activity (weekends, late nights)
        off_hours_count = 0
        for ts in timestamps:
            # Weekend or late night (10 PM - 6 AM)
            if ts.weekday() >= 5 or ts.hour >= 22 or ts.hour <= 6:
                off_hours_count += 1
        
        # Check for rapid sequences (many alerts in short time)
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        rapid_sequences = sum(1 for diff in time_diffs if diff < 60)  # < 1 minute
        
        # Combine factors
        temporal_score = (off_hours_count / len(timestamps)) * 0.5 + \
                        (rapid_sequences / len(time_diffs)) * 0.5
        
        return min(temporal_score, 1.0)
    
    def _detect_privilege_escalation(self, alerts: List[Dict]) -> float:
        """Detect privilege escalation indicators"""
        escalation_indicators = 0
        total_process_alerts = 0
        
        for alert in alerts:
            if alert['event_type'] == 'Process':
                total_process_alerts += 1
                details = alert.get('details', {})
                
                # Check for suspicious processes
                file_name = details.get('file_name', '').lower()
                command_line = details.get('command_line', '').lower()
                
                if any(proc in file_name for proc in self.suspicious_processes):
                    escalation_indicators += 1
                
                # Check for privilege-related commands
                if any(cmd in command_line for cmd in ['runas', 'administrator', 'elevated']):
                    escalation_indicators += 1
        
        return min(escalation_indicators / max(total_process_alerts, 1), 1.0)
    
    def _detect_lateral_movement(self, alerts: List[Dict]) -> float:
        """Detect lateral movement indicators"""
        network_alerts = [a for a in alerts if a['event_type'] == 'Network']
        if not network_alerts:
            return 0.0
        
        lateral_indicators = 0
        internal_connections = 0
        suspicious_ports = 0
        
        for alert in network_alerts:
            details = alert.get('details', {})
            remote_ip = details.get('remote_ip', '')
            remote_port = details.get('remote_port', 0)
            
            # Internal network communication
            if any(remote_ip.startswith(pattern) for pattern in self.internal_ip_patterns):
                internal_connections += 1
                
                # Suspicious ports on internal networks
                if remote_port in self.suspicious_ports:
                    suspicious_ports += 1
        
        if len(network_alerts) == 0:
            return 0.0
        
        # Score based on internal connections and suspicious ports
        lateral_score = (internal_connections / len(network_alerts)) * 0.6 + \
                       (suspicious_ports / len(network_alerts)) * 0.4
        
        return min(lateral_score, 1.0)
    
    def _detect_data_access(self, alerts: List[Dict]) -> float:
        """Detect data access and exfiltration indicators"""
        file_alerts = [a for a in alerts if a['event_type'] == 'File']
        if not file_alerts:
            return 0.0
        
        data_indicators = 0
        sensitive_extensions = {'.docx', '.pdf', '.xlsx', '.csv', '.txt', '.db'}
        
        for alert in file_alerts:
            details = alert.get('details', {})
            file_name = details.get('file_name', '').lower()
            action = details.get('action', '')
            
            # Sensitive file types
            if any(ext in file_name for ext in sensitive_extensions):
                data_indicators += 1
            
            # Bulk file operations
            if action in ['FileDeleted', 'FileAccessed'] and 'document_' in file_name:
                data_indicators += 0.5
        
        return min(data_indicators / len(file_alerts), 1.0)
    
    def _detect_persistence(self, alerts: List[Dict]) -> float:
        """Detect persistence mechanism indicators"""
        persistence_indicators = 0
        total_alerts = len(alerts)
        
        for alert in alerts:
            if alert['event_type'] == 'Process':
                details = alert.get('details', {})
                command_line = details.get('command_line', '').lower()
                
                # Persistence-related commands
                if any(cmd in command_line for cmd in ['schtasks', 'startup', 'registry', 'service']):
                    persistence_indicators += 1
            
            elif alert['event_type'] == 'File':
                details = alert.get('details', {})
                file_name = details.get('file_name', '').lower()
                
                # System/startup related files
                if any(path in file_name for path in ['startup', 'system32', 'temp']):
                    persistence_indicators += 0.5
        
        return min(persistence_indicators / max(total_alerts, 1), 1.0)

class SecurityAnalyzer:
    """Main analyzer that orchestrates HMM analysis of security alerts"""
    
    def __init__(self):
        self.processor = AlertProcessor()
        self.hmm = SimpleSecurityHMM()
        self.sessions_data = None
        self.feature_sequences = []
    
    def load_data(self, alerts_file: str, sessions_file: str):
        """Load alerts and sessions data"""
        logger.info("Loading security data...")
        
        with open(alerts_file, 'r') as f:
            self.alerts_data = json.load(f)
        
        with open(sessions_file, 'r') as f:
            self.sessions_data = json.load(f)
        
        logger.info(f"Loaded {len(self.alerts_data)} alerts and {len(self.sessions_data)} sessions")
    
    def preprocess_sessions(self, min_alerts=5):
        """Convert sessions into feature sequences for HMM training"""
        logger.info("Preprocessing sessions into feature sequences...")
        
        feature_sequences = []
        session_metadata = []
        
        for session_id, session in self.sessions_data.items():
            alerts = session['alerts']
            
            # Skip sessions with too few alerts
            if len(alerts) < min_alerts:
                continue
            
            # Group alerts into time windows (e.g., 5-minute windows)
            alert_windows = self._create_time_windows(alerts, window_minutes=5)
            
            # Extract features for each window
            window_features = []
            for window_alerts in alert_windows:
                if window_alerts:  # Skip empty windows
                    features = self.processor.extract_features(window_alerts)
                    window_features.append([
                        features.temporal_anomaly,
                        features.multi_vector,
                        features.privilege_escalation,
                        features.lateral_movement,
                        features.data_access,
                        features.persistence
                    ])
            
            if len(window_features) >= 3:  # Minimum sequence length
                feature_sequences.append(torch.tensor(window_features, dtype=torch.float32))
                session_metadata.append({
                    'session_id': session_id,
                    'is_attack': session.get('is_attack_session', False),
                    'risk_score': session.get('risk_score', 0),
                    'alert_count': len(alerts),
                    'window_count': len(window_features)
                })
        
        self.feature_sequences = feature_sequences
        self.session_metadata = session_metadata
        
        logger.info(f"Created {len(feature_sequences)} feature sequences")
        return feature_sequences, session_metadata
    
    def _create_time_windows(self, alerts: List[Dict], window_minutes: int = 5) -> List[List[Dict]]:
        """Group alerts into time windows"""
        if not alerts:
            return []
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(alerts, key=lambda x: x['timestamp'])
        
        windows = []
        current_window = []
        current_window_start = None
        
        for alert in sorted_alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            
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
        """Train the HMM on preprocessed data"""
        if not self.feature_sequences:
            raise ValueError("Must preprocess sessions before training")
        
        logger.info("Training Simple HMM...")
        self.hmm.fit(self.feature_sequences, num_iterations=num_iterations)
        return [0]  # Dummy loss for compatibility
    
    def analyze_session(self, session_id: str) -> Dict:
        """Analyze a specific session using trained HMM"""
        if not self.hmm.trained:
            raise ValueError("HMM must be trained before analysis")
        
        session = self.sessions_data[session_id]
        alerts = session['alerts']
        
        # Create time windows and extract features
        alert_windows = self._create_time_windows(alerts, window_minutes=5)
        window_features = []
        
        for window_alerts in alert_windows:
            if window_alerts:
                features = self.processor.extract_features(window_alerts)
                window_features.append([
                    features.temporal_anomaly,
                    features.multi_vector,
                    features.privilege_escalation,
                    features.lateral_movement,
                    features.data_access,
                    features.persistence
                ])
        
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
        
        # Get attack probability
        attack_probability = self.hmm.predict_attack_probability(observations)
        
        # Get state sequence
        predicted_states = self.hmm.viterbi_decode(observations)
        state_sequence = [self.hmm.state_names[state] for state in predicted_states]
        
        # Simple confidence calculation
        confidence = min(attack_probability + 0.5, 1.0)  # Simplified
        
        return {
            'session_id': session_id,
            'attack_probability': attack_probability,
            'predicted_states': predicted_states.tolist(),
            'state_sequence': state_sequence,
            'confidence': confidence,
            'window_count': len(window_features),
            'attack_progression': self._analyze_attack_progression(state_sequence)
        }
    
    def _analyze_attack_progression(self, state_sequence: List[str]) -> Dict:
        """Analyze attack progression through states"""
        progression = {
            'stages_detected': list(set(state_sequence)),
            'kill_chain_coverage': 0.0,
            'progression_timeline': [],
            'persistence': False
        }
        
        attack_stages = ['reconnaissance', 'initial_access', 'lateral_movement', 'objective_execution']
        detected_attack_stages = [s for s in progression['stages_detected'] if s in attack_stages]
        
        progression['kill_chain_coverage'] = len(detected_attack_stages) / len(attack_stages)
        
        # Track stage transitions
        for i, state in enumerate(state_sequence):
            if state != 'normal_operations':
                progression['progression_timeline'].append({
                    'window': i,
                    'stage': state
                })
        
        # Check for persistence (returning to attack states after normal)
        for i in range(1, len(state_sequence)):
            if (state_sequence[i-1] == 'normal_operations' and 
                state_sequence[i] in attack_stages):
                progression['persistence'] = True
                break
        
        return progression
    
    def generate_intelligence_report(self, top_n: int = 10) -> Dict:
        """Generate actionable intelligence report from all sessions"""
        logger.info("Generating intelligence report...")
        
        attack_sessions = []
        normal_sessions = []
        
        for i, metadata in enumerate(self.session_metadata):
            session_id = metadata['session_id']
            analysis = self.analyze_session(session_id)
            
            analysis.update(metadata)
            
            if analysis['attack_probability'] > 0.3:  # Threshold for suspicious
                attack_sessions.append(analysis)
            else:
                normal_sessions.append(analysis)
        
        # Sort by attack probability
        attack_sessions.sort(key=lambda x: x['attack_probability'], reverse=True)
        
        # Generate summary statistics
        total_sessions = len(self.session_metadata)
        high_risk_sessions = len([s for s in attack_sessions if s['attack_probability'] > 0.7])
        medium_risk_sessions = len([s for s in attack_sessions if 0.3 < s['attack_probability'] <= 0.7])
        
        report = {
            'summary': {
                'total_sessions_analyzed': total_sessions,
                'suspicious_sessions': len(attack_sessions),
                'high_risk_sessions': high_risk_sessions,
                'medium_risk_sessions': medium_risk_sessions,
                'normal_sessions': len(normal_sessions),
                'detection_rate': len(attack_sessions) / total_sessions if total_sessions > 0 else 0
            },
            'top_threats': attack_sessions[:top_n],
            'attack_patterns': self._analyze_attack_patterns(attack_sessions),
            'recommendations': self._generate_recommendations(attack_sessions)
        }
        
        return report
    
    def _analyze_attack_patterns(self, attack_sessions: List[Dict]) -> Dict:
        """Analyze common attack patterns"""
        patterns = {
            'common_progressions': defaultdict(int),
            'avg_attack_duration': 0.0,
            'most_common_stages': defaultdict(int),
            'persistence_rate': 0.0
        }
        
        if not attack_sessions:
            return patterns
        
        total_duration = 0
        persistence_count = 0
        
        for session in attack_sessions:
            # Count state progressions
            state_seq = session['state_sequence']
            progression_key = ' → '.join(set(state_seq))
            patterns['common_progressions'][progression_key] += 1
            
            # Count individual stages
            for stage in state_seq:
                patterns['most_common_stages'][stage] += 1
            
            # Track persistence
            if session['attack_progression']['persistence']:
                persistence_count += 1
            
            # Duration (window count as proxy)
            total_duration += session['window_count']
        
        patterns['avg_attack_duration'] = total_duration / len(attack_sessions)
        patterns['persistence_rate'] = persistence_count / len(attack_sessions)
        
        return patterns
    
    def _generate_recommendations(self, attack_sessions: List[Dict]) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        
        if not attack_sessions:
            return ["✅ No significant threats detected. Continue monitoring."]
        
        high_risk_count = len([s for s in attack_sessions if s['attack_probability'] > 0.7])
        
        if high_risk_count > 0:
            recommendations.append(
                f"🚨 IMMEDIATE: Investigate {high_risk_count} high-risk sessions with attack probability > 70%"
            )
        
        # Analyze common attack stages
        all_stages = []
        for session in attack_sessions:
            all_stages.extend(session['state_sequence'])
        
        stage_counts = defaultdict(int)
        for stage in all_stages:
            stage_counts[stage] += 1
        
        if stage_counts['lateral_movement'] > len(attack_sessions) * 0.5:
            recommendations.append(
                "🔒 Implement network segmentation - lateral movement detected in >50% of attacks"
            )
        
        if stage_counts['reconnaissance'] > len(attack_sessions) * 0.3:
            recommendations.append(
                "👁️ Enhance monitoring for reconnaissance activities"
            )
        
        persistence_rate = sum(1 for s in attack_sessions 
                             if s['attack_progression']['persistence']) / len(attack_sessions)
        
        if persistence_rate > 0.2:
            recommendations.append(
                "🛡️ Review system integrity - high persistence rate detected"
            )
        
        recommendations.append(
            f"📊 Continue monitoring - analyzed {len(attack_sessions)} suspicious sessions"
        )
        
        return recommendations
    
    def visualize_results(self, save_results=True):
        """Create pandas-based visualizations and summaries of the analysis results"""
        try:
            print("\n📊 VISUALIZATION RESULTS")
            print("=" * 60)
            
            # 1. State transition matrix as DataFrame
            if self.hmm.trained and self.hmm.transition_probs is not None:
                print("\n🔄 STATE TRANSITION PROBABILITIES:")
                transition_df = pd.DataFrame(
                    self.hmm.transition_probs,
                    index=[f"From_{state}" for state in self.hmm.state_names],
                    columns=[f"To_{state}" for state in self.hmm.state_names]
                )
                print(transition_df.round(3).to_string())
                
                # Find most likely transitions
                print("\n📈 Most Likely State Transitions:")
                for i, from_state in enumerate(self.hmm.state_names):
                    max_prob_idx = np.argmax(transition_df.iloc[i])
                    max_prob = transition_df.iloc[i, max_prob_idx]
                    to_state = self.hmm.state_names[max_prob_idx]
                    if max_prob > 0.3 and from_state != to_state:  # Only show significant transitions
                        print(f"   {from_state} → {to_state}: {max_prob:.1%}")
            
            # 2. Attack probability distribution
            print("\n🎯 ATTACK PROBABILITY ANALYSIS:")
            attack_probs = []
            session_data = []
            
            for metadata in self.session_metadata:
                session_id = metadata['session_id']
                analysis = self.analyze_session(session_id)
                attack_probs.append(analysis['attack_probability'])
                session_data.append({
                    'session_id': session_id,
                    'attack_probability': analysis['attack_probability'],
                    'original_risk_score': metadata['risk_score'],
                    'alert_count': metadata['alert_count'],
                    'is_attack': metadata.get('is_attack_session', False)
                })
            
            # Create analysis DataFrame
            results_df = pd.DataFrame(session_data)
            
            # Attack probability statistics
            print(f"   Mean Attack Probability: {np.mean(attack_probs):.1%}")
            print(f"   Median Attack Probability: {np.median(attack_probs):.1%}")
            print(f"   Max Attack Probability: {np.max(attack_probs):.1%}")
            print(f"   Sessions > 30% probability: {sum(1 for p in attack_probs if p > 0.3)}")
            print(f"   Sessions > 70% probability: {sum(1 for p in attack_probs if p > 0.7)}")
            
            # Distribution bins
            print("\n📊 Attack Probability Distribution:")
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            bin_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
            
            for i in range(len(bins)-1):
                count = sum(1 for p in attack_probs if bins[i] <= p < bins[i+1])
                percentage = count / len(attack_probs) * 100
                bar = '█' * int(percentage / 5)  # Simple text bar chart
                print(f"   {bin_labels[i]:>8}: {count:3d} sessions {bar} ({percentage:.1f}%)")
            
            # 3. Model vs Original Risk Score Correlation
            print("\n🔗 HMM VALIDATION vs ORIGINAL RISK SCORES:")
            correlation_df = results_df[['attack_probability', 'original_risk_score']].copy()
            correlation_df['normalized_risk'] = correlation_df['original_risk_score'] / 100.0
            
            correlation = correlation_df['attack_probability'].corr(correlation_df['normalized_risk'])
            print(f"   Correlation Coefficient: {correlation:.3f}")
            
            # Show top discrepancies (where HMM disagrees with original scoring)
            correlation_df['difference'] = abs(correlation_df['attack_probability'] - correlation_df['normalized_risk'])
            top_discrepancies = correlation_df.nlargest(5, 'difference')
            
            print("\n📋 Top Discrepancies (HMM vs Original Risk):")
            for idx in top_discrepancies.index:
                session_id = results_df.loc[idx, 'session_id']
                hmm_prob = results_df.loc[idx, 'attack_probability']
                orig_risk = results_df.loc[idx, 'original_risk_score']
                print(f"   {session_id}: HMM={hmm_prob:.1%}, Original={orig_risk}%, Diff={abs(hmm_prob - orig_risk/100):.1%}")
            
            # 4. Attack Pattern Summary
            print("\n🛡️ ATTACK PATTERN ANALYSIS:")
            attack_sessions = results_df[results_df['attack_probability'] > 0.3]
            
            if len(attack_sessions) > 0:
                all_progressions = []
                all_stages = []
                
                for _, row in attack_sessions.iterrows():
                    session_id = row['session_id']
                    analysis = self.analyze_session(session_id)
                    progression = analysis['attack_progression']
                    
                    # Collect stages
                    stages = progression['stages_detected']
                    all_stages.extend([s for s in stages if s != 'normal_operations'])
                    
                    # Create progression signature
                    attack_stages_only = [s for s in stages if s != 'normal_operations']
                    if attack_stages_only:
                        progression_sig = ' → '.join(sorted(set(attack_stages_only)))
                        all_progressions.append(progression_sig)
                
                # Stage frequency analysis
                if all_stages:
                    stage_counts = pd.Series(all_stages).value_counts()
                    print("   Most Common Attack Stages:")
                    for stage, count in stage_counts.head().items():
                        percentage = count / len(attack_sessions) * 100
                        print(f"     • {stage}: {count} sessions ({percentage:.1f}%)")
                
                # Progression patterns
                if all_progressions:
                    progression_counts = pd.Series(all_progressions).value_counts()
                    print("\n   Common Attack Progressions:")
                    for progression, count in progression_counts.head(3).items():
                        print(f"     • {progression}: {count} sessions")
            
            # 5. Save detailed results
            if save_results:
                # Save comprehensive results
                results_df.to_csv('simple_hmm_session_analysis.csv', index=False)
                
                if self.hmm.trained and self.hmm.transition_probs is not None:
                    transition_df.to_csv('simple_hmm_transition_matrix.csv')
                
                print(f"\n💾 Results saved to:")
                print(f"   • simple_hmm_session_analysis.csv")
                print(f"   • simple_hmm_transition_matrix.csv")
            
            return results_df
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            logger.info("Continuing without detailed visualizations")
            return None
    
    def export_results(self, output_file='simple_security_intelligence_report.json'):
        """Export analysis results to JSON file"""
        logger.info("Generating and exporting intelligence report...")
        
        report = self.generate_intelligence_report()
        
        # Add model parameters for transparency
        if self.hmm.trained and self.hmm.transition_probs is not None:
            report['model_parameters'] = {
                'transition_probabilities': self.hmm.transition_probs.tolist(),
                'state_names': self.hmm.state_names,
                'model_type': 'SimpleHMM'
            }
        
        # Add individual session analyses
        detailed_analyses = []
        for metadata in self.session_metadata:
            session_id = metadata['session_id']
            analysis = self.analyze_session(session_id)
            analysis.update(metadata)
            detailed_analyses.append(analysis)
        
        report['detailed_session_analyses'] = detailed_analyses
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Intelligence report exported to {output_file}")
        return report

def main():
    """Main execution function demonstrating the Simple HMM security analysis"""
    print("🔒 Simple Security Alert HMM Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SecurityAnalyzer()
    
    try:
        # Load data (adjust file paths as needed)
        analyzer.load_data('synthetic_alerts_sample.json', 'synthetic_sessions_sample.json')
        
        # Preprocess sessions into feature sequences
        feature_sequences, session_metadata = analyzer.preprocess_sessions(min_alerts=3)
        
        if not feature_sequences:
            print("❌ No valid sessions found for analysis")
            return
        
        print(f"✅ Preprocessed {len(feature_sequences)} sessions")
        print(f"📊 Average sequence length: {np.mean([len(seq) for seq in feature_sequences]):.1f}")
        
        # Train HMM
        print("\n🧠 Training Simple Hidden Markov Model...")
        losses = analyzer.train_hmm(num_iterations=50)
        
        print(f"✅ Training completed successfully!")
        
        # Generate intelligence report
        print("\n📋 Generating Intelligence Report...")
        report = analyzer.generate_intelligence_report(top_n=5)
        
        # Display summary
        summary = report['summary']
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   Total Sessions: {summary['total_sessions_analyzed']}")
        print(f"   Suspicious Sessions: {summary['suspicious_sessions']}")
        print(f"   High Risk: {summary['high_risk_sessions']}")
        print(f"   Medium Risk: {summary['medium_risk_sessions']}")
        print(f"   Detection Rate: {summary['detection_rate']:.1%}")
        
        # Display top threats
        print(f"\n🚨 TOP THREATS:")
        for i, threat in enumerate(report['top_threats'][:3], 1):
            print(f"   {i}. Session: {threat['session_id']}")
            print(f"      Attack Probability: {threat['attack_probability']:.1%}")
            print(f"      Attack Stages: {', '.join(threat['attack_progression']['stages_detected'])}")
            print(f"      Confidence: {threat['confidence']:.1%}")
            print()
        
        # Display recommendations
        print(f"💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        # Export results
        analyzer.export_results('simple_security_intelligence_report.json')
        
        # Create visualizations
        print(f"\n📊 Generating pandas-based analysis...")
        results_df = analyzer.visualize_results(save_results=True)
        
        print(f"\n✅ Analysis complete! Check simple_security_intelligence_report.json for detailed results.")
        
        # Demonstrate individual session analysis
        if analyzer.session_metadata:
            print(f"\n🔍 EXAMPLE SESSION ANALYSIS:")
            sample_session = analyzer.session_metadata[0]['session_id']
            analysis = analyzer.analyze_session(sample_session)
            
            print(f"   Session: {sample_session}")
            print(f"   Attack Probability: {analysis['attack_probability']:.1%}")
            print(f"   State Sequence: {' → '.join(analysis['state_sequence'])}")
            print(f"   Kill Chain Coverage: {analysis['attack_progression']['kill_chain_coverage']:.1%}")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("💡 Make sure synthetic_alerts_sample.json and synthetic_sessions_sample.json are available")
        print("   You can generate them using the synthetic data generator from your article")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()