#!/usr/bin/env python3
"""
Standalone Attack Chain Progression Analyzer
Works with output from fixed_pyro_2.py and original JSON data
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AttackChainEvent:
    """Represents a single event in an attack chain"""
    timestamp: datetime
    stage: str
    event_type: str
    technique: str
    confidence: float
    details: Dict
    window_index: int
    feature_intensity: float

@dataclass
class AttackProgression:
    """Represents the complete progression of an attack"""
    session_id: str
    attack_probability: float
    total_stages: int
    kill_chain_coverage: float
    sophistication_level: str
    persistence_detected: bool
    lateral_movement_detected: bool
    data_exfiltration_detected: bool
    credential_access_detected: bool
    privilege_escalation_detected: bool
    events: List[AttackChainEvent]
    timeline_duration_minutes: float
    attack_velocity: float
    stealth_score: float

class StandaloneAttackChainAnalyzer:
    """Standalone attack chain analyzer"""
    
    def __init__(self):
        # MITRE ATT&CK technique mapping
        self.mitre_techniques = {
            'reconnaissance': {
                'T1590': 'Gather Victim Network Information',
                'T1589': 'Gather Victim Identity Information', 
                'T1595': 'Active Scanning',
                'T1087': 'Account Discovery'
            },
            'initial_access': {
                'T1566': 'Phishing',
                'T1190': 'Exploit Public-Facing Application',
                'T1078': 'Valid Accounts',
                'T1133': 'External Remote Services'
            },
            'privilege_escalation': {
                'T1548': 'Abuse Elevation Control Mechanism',
                'T1134': 'Access Token Manipulation',
                'T1055': 'Process Injection',
                'T1068': 'Exploitation for Privilege Escalation'
            },
            'lateral_movement': {
                'T1021': 'Remote Services',
                'T1570': 'Lateral Tool Transfer',
                'T1210': 'Exploitation of Remote Services',
                'T1563': 'Remote Service Session Hijacking'
            },
            'data_exfiltration': {
                'T1041': 'Exfiltration Over C2 Channel',
                'T1048': 'Exfiltration Over Alternative Protocol',
                'T1567': 'Exfiltration Over Web Service',
                'T1020': 'Automated Exfiltration'
            }
        }
        
        # Technique indicators for automatic classification
        self.technique_indicators = {
            'T1087': ['whoami', 'net user', 'net group', 'net localgroup'],
            'T1055': ['process injection', 'dll injection', 'hollowing'],
            'T1021.002': ['psexec', 'smb', 'admin$', 'c$'],
            'T1078': ['runas', 'logon', 'authentication'],
            'T1566.001': ['attachment', 'email', 'macro', 'document'],
            'T1041': ['beacon', 'c2', 'command control', 'exfil'],
            'T1134': ['token', 'impersonation', 'delegation'],
            'T1570': ['copy', 'xcopy', 'robocopy', 'scp'],
            'T1048': ['dns tunneling', 'ftp', 'http post'],
            'T1590': ['nslookup', 'dig', 'nmap', 'port scan']
        }
        
        # Kill chain stages
        self.kill_chain_stages = [
            'reconnaissance',
            'initial_access', 
            'privilege_escalation',
            'lateral_movement',
            'data_exfiltration'
        ]
        
        self.stage_weights = {
            'normal_operations': 0.0,
            'reconnaissance': 0.2,
            'initial_access': 0.4,
            'privilege_escalation': 0.6,
            'lateral_movement': 0.8,
            'data_exfiltration': 1.0
        }
        
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe', 'psexec.exe',
            'rundll32.exe', 'regsvr32.exe', 'certutil.exe', 'bitsadmin.exe',
            'mshta.exe', 'cscript.exe', 'wscript.exe'
        }
    
    def inspect_csv_structure(self, csv_file: str) -> None:
        """Inspect CSV structure for debugging"""
        logger.info(f" Inspecting CSV structure: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f" CSV Info:")
            logger.info(f"   • Rows: {len(df)}")
            logger.info(f"   • Columns: {len(df.columns)}")
            logger.info(f"   • Column names: {list(df.columns)}")
            
            # Show first few rows
            logger.info(f" First 3 rows:")
            for i, (_, row) in enumerate(df.head(3).iterrows()):
                logger.info(f"   Row {i+1}: {dict(row)}")
            
            # Check for common column patterns
            prob_cols = [col for col in df.columns if 'prob' in col.lower() or 'score' in col.lower()]
            if prob_cols:
                logger.info(f" Probability/Score columns found: {prob_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f" Failed to inspect CSV: {e}")
            raise
    
    def load_pyro_results(self, pyro_csv_file: str) -> pd.DataFrame:
        """Load results from fixed_pyro_2.py output"""
        logger.info(f" Loading Pyro analysis results from: {pyro_csv_file}")
        
        try:
            # First inspect the structure
            df = self.inspect_csv_structure(pyro_csv_file)
            logger.info(f" Loaded {len(df)} sessions from Pyro results")
            return df
        except Exception as e:
            logger.error(f" Failed to load Pyro results: {e}")
            raise
    
    def load_original_data(self, json_file: str) -> Dict:
        """Load original session data from JSON"""
        logger.info(f" Loading original session data from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f" JSON structure: {type(data).__name__}")
            
            sessions_dict = {}
            
            # Handle different JSON structures
            if isinstance(data, list):
                logger.info(f" Processing list with {len(data)} items")
                for i, session in enumerate(data):
                    if isinstance(session, dict):
                        # Your JSON structure has session_id directly
                        session_id = session.get('session_id', f"session_{i:04d}")
                        sessions_dict[session_id] = session
                        logger.debug(f"   Added session: {session_id}")
            
            elif isinstance(data, dict):
                # Check if this is a dictionary of sessions
                sample_keys = list(data.keys())[:3]
                is_session_dict = False
                
                for key in sample_keys:
                    value = data[key]
                    if isinstance(value, dict) and ('session_id' in value or 'alerts' in value):
                        is_session_dict = True
                        break
                
                if is_session_dict:
                    sessions_dict = {k: v for k, v in data.items()}
                else:
                    # Single session object
                    session_id = data.get('session_id', 'session_0000')
                    sessions_dict[session_id] = data
            
            else:
                raise ValueError(f"Unsupported JSON root type: {type(data)}")
            
            logger.info(f" Loaded {len(sessions_dict)} sessions from original data")
            
            # Log sample session structure
            if sessions_dict:
                sample_session = list(sessions_dict.values())[0]
                logger.info(f" Sample session keys: {list(sample_session.keys())}")
            
            return sessions_dict
            
        except Exception as e:
            logger.error(f" Failed to load original data: {e}")
            raise
    
    def map_attack_probability_to_stage(self, attack_prob: float, risk_score: int) -> str:
        """Map attack probability to likely attack stage"""
        if attack_prob < 0.1:
            return 'normal_operations'
        elif attack_prob < 0.3:
            return 'reconnaissance'
        elif attack_prob < 0.5:
            return 'initial_access'
        elif attack_prob < 0.7:
            return 'privilege_escalation'
        elif attack_prob < 0.9:
            return 'lateral_movement'
        else:
            return 'data_exfiltration'
    
    def create_synthetic_alerts_from_metadata(self, session_data: Dict, attack_prob: float, 
                                            alert_count: int, risk_score: int) -> List[Dict]:
        """Create synthetic alerts based on session metadata"""
        alerts = []
        base_time = datetime.now()
        
        # Get session_id
        session_id = session_data.get('session_id', 'unknown')
        
        # Check if session already has alerts
        if 'alerts' in session_data and session_data['alerts']:
            logger.debug(f"Using existing alerts for {session_id}")
            return session_data['alerts']
        
        # Use metadata from your JSON structure to inform alert generation
        is_attack = session_data.get('is_attack', False)
        attack_intensity = session_data.get('attack_intensity', 0.0)
        multi_vector = session_data.get('multi_vector', False)
        event_type_diversity = session_data.get('event_type_diversity', 1)
        
        # Determine attack sophistication based on available metadata
        if is_attack and attack_intensity > 0.5:
            attack_type = "advanced_persistent"
        elif is_attack and multi_vector:
            attack_type = "targeted_attack"
        elif attack_prob > 0.3 or risk_score > 60:
            attack_type = "opportunistic"
        else:
            attack_type = "normal"
        
        logger.debug(f"Generating {alert_count} alerts for {session_id} (type: {attack_type})")
        
        # Ensure we have reasonable alert count
        effective_alert_count = max(min(alert_count, 25), 2)
        
        for i in range(effective_alert_count):
            # Realistic time distribution
            if attack_type != "normal":
                time_offset = np.random.exponential(300)  # Clustered in time
            else:
                time_offset = np.random.uniform(60, 3600)  # Spread out
                
            alert_time = base_time + timedelta(seconds=i * time_offset)
            
            alert = self._generate_alert_by_type(alert_time, i, session_id, attack_type, risk_score)
            alerts.append(alert)
        
        return alerts
    
    def _generate_alert_by_type(self, timestamp: datetime, index: int, session_id: str, 
                               attack_type: str, risk_score: int) -> Dict:
        """Generate realistic alerts based on attack type"""
        
        if attack_type == "advanced_persistent":
            patterns = [
                {
                    'event_type': 'Process',
                    'severity': 'Critical',
                    'details': {
                        'file_name': 'powershell.exe',
                        'command_line': 'powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass',
                        'process_id': 2000 + index,
                        'parent_process_id': 1500
                    }
                },
                {
                    'event_type': 'Network',
                    'severity': 'High',
                    'details': {
                        'remote_ip': f'10.0.{index % 50}.{100 + index % 155}',
                        'remote_port': 445,
                        'direction': 'outbound',
                        'protocol': 'SMB'
                    }
                },
                {
                    'event_type': 'File',
                    'severity': 'High',
                    'details': {
                        'file_name': f'sensitive_data_{index}.docx',
                        'action': 'FileAccessed',
                        'file_path': 'C:\\Users\\admin\\Documents\\Confidential\\'
                    }
                }
            ]
        elif attack_type == "targeted_attack":
            patterns = [
                {
                    'event_type': 'Process',
                    'severity': 'High',
                    'details': {
                        'file_name': 'cmd.exe',
                        'command_line': 'cmd.exe /c net user administrator /active:yes',
                        'process_id': 3000 + index
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
        elif attack_type == "opportunistic":
            patterns = [
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
        else:  # normal
            patterns = [
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
                }
            ]
        
        pattern = patterns[index % len(patterns)]
        pattern['timestamp'] = timestamp.isoformat()
        pattern['session_id'] = session_id
        
        return pattern
    
    def analyze_attack_progression(self, session_id: str, session_data: Dict, 
                                 attack_prob: float, alert_count: int, 
                                 risk_score: int) -> Optional[AttackProgression]:
        """Analyze attack progression for a single session"""
        
        # Get or create alerts for this session
        if 'alerts' in session_data and session_data['alerts']:
            alerts = session_data['alerts']
        else:
            # Create synthetic alerts based on the analysis
            alerts = self.create_synthetic_alerts_from_metadata(
                session_data, attack_prob, alert_count, risk_score
            )
        
        if not alerts or attack_prob < 0.1:
            return None
        
        # Create attack events
        events = self._create_attack_events(session_id, alerts, attack_prob, risk_score)
        
        if not events:
            return None
        
        # Calculate metrics
        timeline_duration = self._calculate_timeline_duration(events)
        attack_velocity = self._calculate_attack_velocity(events, timeline_duration)
        stealth_score = self._calculate_stealth_score(events, attack_prob)
        sophistication = self._determine_sophistication_level(attack_prob, risk_score, len(events))
        
        # Detect specific attack types
        persistence_detected = self._detect_persistence(events)
        lateral_movement_detected = self._detect_lateral_movement(events)
        data_exfiltration_detected = self._detect_data_exfiltration(events)
        credential_access_detected = self._detect_credential_access(events)
        privilege_escalation_detected = self._detect_privilege_escalation(events)
        
        # Calculate kill chain coverage
        unique_stages = set(event.stage for event in events if event.stage != 'normal_operations')
        kill_chain_coverage = len(unique_stages) / len(self.kill_chain_stages)
        
        return AttackProgression(
            session_id=session_id,
            attack_probability=attack_prob,
            total_stages=len(unique_stages),
            kill_chain_coverage=kill_chain_coverage,
            sophistication_level=sophistication,
            persistence_detected=persistence_detected,
            lateral_movement_detected=lateral_movement_detected,
            data_exfiltration_detected=data_exfiltration_detected,
            credential_access_detected=credential_access_detected,
            privilege_escalation_detected=privilege_escalation_detected,
            events=events,
            timeline_duration_minutes=timeline_duration,
            attack_velocity=attack_velocity,
            stealth_score=stealth_score
        )
    
    def _create_attack_events(self, session_id: str, alerts: List[Dict], 
                            attack_prob: float, risk_score: int) -> List[AttackChainEvent]:
        """Create attack events from alerts"""
        events = []
        
        # Determine primary attack stage based on probability
        primary_stage = self.map_attack_probability_to_stage(attack_prob, risk_score)
        
        # Create time windows
        alert_windows = self._create_time_windows(alerts)
        
        for i, window_alerts in enumerate(alert_windows):
            if not window_alerts:
                continue
            
            # Determine stage for this window
            if i == 0:
                stage = 'reconnaissance' if attack_prob > 0.2 else 'normal_operations'
            elif i < len(alert_windows) // 2:
                stage = 'initial_access' if attack_prob > 0.3 else primary_stage
            else:
                stage = primary_stage
            
            # Estimate timestamp
            timestamp = self._get_window_timestamp(window_alerts)
            
            # Classify technique
            technique = self._classify_technique_from_alerts(stage, window_alerts)
            
            # Determine event type
            event_type = self._get_primary_event_type(window_alerts)
            
            # Calculate confidence and intensity
            confidence = min(attack_prob + (risk_score / 100.0) * 0.3, 1.0)
            feature_intensity = min(len(window_alerts) / 10.0, 1.0)
            
            event = AttackChainEvent(
                timestamp=timestamp,
                stage=stage,
                event_type=event_type,
                technique=technique,
                confidence=confidence,
                details=self._extract_window_details(window_alerts),
                window_index=i,
                feature_intensity=feature_intensity
            )
            
            events.append(event)
        
        return events
    
    def _create_time_windows(self, alerts: List[Dict], window_minutes: int = 5) -> List[List[Dict]]:
        """Create time windows from alerts"""
        if not alerts:
            return []
        
        # Parse timestamps
        timestamped_alerts = []
        for alert in alerts:
            try:
                timestamp = datetime.fromisoformat(alert['timestamp'])
                timestamped_alerts.append((timestamp, alert))
            except:
                # Use current time if timestamp parsing fails
                timestamped_alerts.append((datetime.now(), alert))
        
        timestamped_alerts.sort(key=lambda x: x[0])
        
        # Create windows
        windows = []
        current_window = []
        current_window_start = None
        
        for timestamp, alert in timestamped_alerts:
            if current_window_start is None:
                current_window_start = timestamp
                current_window = [alert]
            else:
                time_diff = (timestamp - current_window_start).total_seconds() / 60
                
                if time_diff <= window_minutes:
                    current_window.append(alert)
                else:
                    if current_window:
                        windows.append(current_window)
                    current_window = [alert]
                    current_window_start = timestamp
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _get_window_timestamp(self, window_alerts: List[Dict]) -> datetime:
        """Get representative timestamp for window"""
        timestamps = []
        for alert in window_alerts:
            try:
                timestamps.append(datetime.fromisoformat(alert['timestamp']))
            except:
                timestamps.append(datetime.now())
        
        return min(timestamps) if timestamps else datetime.now()
    
    def _classify_technique_from_alerts(self, stage: str, alerts: List[Dict]) -> str:
        """Classify MITRE technique from alerts"""
        # Get stage techniques
        stage_techniques = self.mitre_techniques.get(stage, {})
        
        if not stage_techniques:
            return 'Unknown'
        
        # Analyze alert content
        alert_content = ' '.join([
            str(alert.get('details', {})).lower() 
            for alert in alerts
        ])
        
        # Score techniques based on indicators
        technique_scores = {}
        for tech_id, indicators in self.technique_indicators.items():
            score = sum(1 for indicator in indicators if indicator in alert_content)
            if score > 0:
                technique_scores[tech_id] = score
        
        # Return best matching technique
        if technique_scores:
            best_technique = max(technique_scores.items(), key=lambda x: x[1])[0]
            # Find technique name
            for techniques in self.mitre_techniques.values():
                if best_technique in techniques:
                    return f"{best_technique}: {techniques[best_technique]}"
        
        # Default to first technique in stage
        if stage_techniques:
            first_tech = list(stage_techniques.items())[0]
            return f"{first_tech[0]}: {first_tech[1]}"
        
        return 'Unknown'
    
    def _get_primary_event_type(self, alerts: List[Dict]) -> str:
        """Get primary event type from alerts"""
        event_types = [alert.get('event_type', 'Unknown') for alert in alerts]
        if event_types:
            return Counter(event_types).most_common(1)[0][0]
        return 'Unknown'
    
    def _extract_window_details(self, alerts: List[Dict]) -> Dict:
        """Extract details from window alerts"""
        return {
            'alert_count': len(alerts),
            'event_types': list(set(alert.get('event_type', 'Unknown') for alert in alerts)),
            'severities': list(set(alert.get('severity', 'Unknown') for alert in alerts))
        }
    
    def _calculate_timeline_duration(self, events: List[AttackChainEvent]) -> float:
        """Calculate timeline duration in minutes"""
        if len(events) < 2:
            return 0.0
        
        timestamps = [event.timestamp for event in events]
        return (max(timestamps) - min(timestamps)).total_seconds() / 60.0
    
    def _calculate_attack_velocity(self, events: List[AttackChainEvent], duration: float) -> float:
        """Calculate attack velocity (stages per hour)"""
        if duration == 0:
            return 0.0
        
        unique_stages = len(set(event.stage for event in events if event.stage != 'normal_operations'))
        return (unique_stages / duration) * 60.0
    
    def _calculate_stealth_score(self, events: List[AttackChainEvent], attack_prob: float) -> float:
        """Calculate stealth score"""
        if not events:
            return 0.0
        
        # Lower intensity events = more stealthy
        avg_intensity = np.mean([event.feature_intensity for event in events])
        return 1.0 - min(avg_intensity, attack_prob)
    
    def _determine_sophistication_level(self, attack_prob: float, risk_score: int, event_count: int) -> str:
        """Determine sophistication level"""
        sophistication_score = (attack_prob * 0.4 + 
                              (risk_score / 100.0) * 0.4 + 
                              min(event_count / 10.0, 1.0) * 0.2)
        
        if sophistication_score >= 0.8:
            return 'Advanced Persistent Threat'
        elif sophistication_score >= 0.6:
            return 'Advanced'
        elif sophistication_score >= 0.4:
            return 'Intermediate'
        else:
            return 'Basic'
    
    def _detect_persistence(self, events: List[AttackChainEvent]) -> bool:
        """Detect persistence indicators"""
        persistence_keywords = ['registry', 'service', 'startup', 'scheduled']
        
        for event in events:
            if any(keyword in event.technique.lower() for keyword in persistence_keywords):
                return True
            if any(keyword in str(event.details).lower() for keyword in persistence_keywords):
                return True
        
        return False
    
    def _detect_lateral_movement(self, events: List[AttackChainEvent]) -> bool:
        """Detect lateral movement"""
        return any(event.stage == 'lateral_movement' for event in events)
    
    def _detect_data_exfiltration(self, events: List[AttackChainEvent]) -> bool:
        """Detect data exfiltration"""
        return any(event.stage == 'data_exfiltration' for event in events)
    
    def _detect_credential_access(self, events: List[AttackChainEvent]) -> bool:
        """Detect credential access"""
        credential_keywords = ['credential', 'password', 'token', 'lsass']
        
        for event in events:
            if any(keyword in event.technique.lower() for keyword in credential_keywords):
                return True
        
        return False
    
    def _detect_privilege_escalation(self, events: List[AttackChainEvent]) -> bool:
        """Detect privilege escalation"""
        return any(event.stage == 'privilege_escalation' for event in events)
    
    def analyze_all_chains(self, pyro_results: pd.DataFrame, 
                          original_sessions: Dict) -> List[AttackProgression]:
        """Analyze attack chains for all sessions"""
        logger.info(" Starting attack chain analysis...")
        
        # Check available columns and adapt
        logger.info(f" Available columns: {list(pyro_results.columns)}")
        
        attack_chains = []
        
        for _, row in pyro_results.iterrows():
            try:
                # Try different column name variations
                session_id = row.get('session_id', f"session_{len(attack_chains)}")
                
                # Try different attack probability column names
                attack_prob = 0.0
                for col in ['attack_probability', 'combined_score', 'hmm_attack_probability', 'risk_probability']:
                    if col in row and pd.notna(row[col]):
                        attack_prob = float(row[col])
                        break
                
                # Try different risk score column names
                risk_score = 50  # default
                for col in ['original_risk_score', 'risk_score', 'risk_probability']:
                    if col in row and pd.notna(row[col]):
                        if col == 'risk_probability':
                            risk_score = int(float(row[col]) * 100)  # Convert probability to score
                        else:
                            risk_score = int(row[col])
                        break
                
                # Try different alert count column names
                alert_count = 5  # default
                for col in ['alert_count', 'total_alerts', 'event_count']:
                    if col in row and pd.notna(row[col]):
                        alert_count = int(row[col])
                        break
                
                logger.debug(f"Processing {session_id}: prob={attack_prob}, risk={risk_score}, alerts={alert_count}")
                
                # Only analyze sessions with reasonable attack probability
                if attack_prob > 0.05:  # Lower threshold
                    session_data = original_sessions.get(session_id, {'session_id': session_id})
                    
                    chain = self.analyze_attack_progression(
                        session_id, session_data, attack_prob, alert_count, risk_score
                    )
                    
                    if chain:
                        attack_chains.append(chain)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze chain for session {session_id if 'session_id' in locals() else 'unknown'}: {e}")
                continue
        
        logger.info(f" Analyzed {len(attack_chains)} attack chains")
        return attack_chains
    
    def export_attack_chains(self, attack_chains: List[AttackProgression], 
                           output_prefix: str = 'attack_chains') -> None:
        """Export attack chains to CSV files"""
        if not attack_chains:
            logger.warning("No attack chains to export")
            return
        
        # Create summary dataframe
        summary_data = []
        detailed_data = []
        
        for chain in attack_chains:
            # Summary row
            summary_row = {
                'session_id': chain.session_id,
                'attack_probability': chain.attack_probability,
                'total_stages': chain.total_stages,
                'kill_chain_coverage': chain.kill_chain_coverage,
                'sophistication_level': chain.sophistication_level,
                'timeline_duration_minutes': chain.timeline_duration_minutes,
                'attack_velocity_stages_per_hour': chain.attack_velocity,
                'stealth_score': chain.stealth_score,
                'persistence_detected': chain.persistence_detected,
                'lateral_movement_detected': chain.lateral_movement_detected,
                'data_exfiltration_detected': chain.data_exfiltration_detected,
                'credential_access_detected': chain.credential_access_detected,
                'privilege_escalation_detected': chain.privilege_escalation_detected,
                'total_events': len(chain.events),
                'attack_progression': ' → '.join([event.stage for event in chain.events if event.stage != 'normal_operations']),
                'primary_techniques': ' | '.join([event.technique.split(':')[0] for event in chain.events[:3]])
            }
            summary_data.append(summary_row)
            
            # Detailed rows
            for i, event in enumerate(chain.events):
                detailed_row = summary_row.copy()
                detailed_row.update({
                    'event_sequence': i + 1,
                    'event_timestamp': event.timestamp.isoformat(),
                    'event_stage': event.stage,
                    'event_type': event.event_type,
                    'mitre_technique': event.technique,
                    'event_confidence': event.confidence,
                    'feature_intensity': event.feature_intensity,
                    'window_index': event.window_index,
                    'event_details_json': json.dumps(event.details)
                })
                detailed_data.append(detailed_row)
        
        # Export files
        summary_df = pd.DataFrame(summary_data)
        detailed_df = pd.DataFrame(detailed_data)
        
        summary_file = f'{output_prefix}_summary.csv'
        detailed_file = f'{output_prefix}_detailed.csv'
        
        summary_df.to_csv(summary_file, index=False)
        detailed_df.to_csv(detailed_file, index=False)
        
        logger.info(f" Attack chains exported to:")
        logger.info(f"   • {summary_file}")
        logger.info(f"   • {detailed_file}")
        
        # Generate report
        self._generate_attack_chain_report(attack_chains, f'{output_prefix}_report.json')
    
    def _generate_attack_chain_report(self, attack_chains: List[AttackProgression], 
                                    output_file: str) -> None:
        """Generate attack chain analysis report"""
        
        # Calculate statistics
        total_chains = len(attack_chains)
        if total_chains == 0:
            return
        
        sophistication_dist = Counter(chain.sophistication_level for chain in attack_chains)
        avg_duration = np.mean([chain.timeline_duration_minutes for chain in attack_chains])
        avg_velocity = np.mean([chain.attack_velocity for chain in attack_chains])
        avg_stealth = np.mean([chain.stealth_score for chain in attack_chains])
        
        # Technique analysis
        all_techniques = []
        for chain in attack_chains:
            for event in chain.events:
                technique_id = event.technique.split(':')[0]
                all_techniques.append(technique_id)
        
        technique_frequency = Counter(all_techniques).most_common(10)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_attack_chains': total_chains,
                'analysis_type': 'Standalone Attack Chain Analysis'
            },
            'summary': {
                'average_duration_minutes': round(avg_duration, 2),
                'average_attack_velocity': round(avg_velocity, 2),
                'average_stealth_score': round(avg_stealth, 2)
            },
            'sophistication_distribution': dict(sophistication_dist),
            'top_techniques': dict(technique_frequency),
            'behavioral_patterns': {
                'persistence_rate': sum(1 for c in attack_chains if c.persistence_detected) / total_chains,
                'lateral_movement_rate': sum(1 for c in attack_chains if c.lateral_movement_detected) / total_chains,
                'data_exfiltration_rate': sum(1 for c in attack_chains if c.data_exfiltration_detected) / total_chains,
                'credential_access_rate': sum(1 for c in attack_chains if c.credential_access_detected) / total_chains
            }
        }
        
        # Export report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f" Attack chain report exported: {output_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Standalone Attack Chain Progression Analyzer')
    parser.add_argument('--pyro-results', required=True, help='CSV file with Pyro analysis results')
    parser.add_argument('--original-data', required=True, help='JSON file with original session data')
    parser.add_argument('--output-prefix', default='attack_chains', help='Output file prefix')
    
    args = parser.parse_args()
    
    print(" Standalone Attack Chain Progression Analysis")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = StandaloneAttackChainAnalyzer()
        
        # Load data
        print(f" Loading Pyro results from: {args.pyro_results}")
        pyro_results = analyzer.load_pyro_results(args.pyro_results)
        
        print(f" Loading original data from: {args.original_data}")
        original_sessions = analyzer.load_original_data(args.original_data)
        
        # Analyze attack chains
        print(" Analyzing attack progression chains...")
        attack_chains = analyzer.analyze_all_chains(pyro_results, original_sessions)
        
        if not attack_chains:
            print(" No attack chains found to analyze")
            return
        
        # Export results
        print(" Exporting attack chain analysis...")
        analyzer.export_attack_chains(attack_chains, args.output_prefix)
        
        # Display summary
        print("\n" + "="*60)
        print(" ATTACK CHAIN ANALYSIS COMPLETE")
        print("="*60)
        
        total_chains = len(attack_chains)
        critical_chains = sum(1 for c in attack_chains if c.sophistication_level == 'Advanced Persistent Threat')
        advanced_chains = sum(1 for c in attack_chains if c.sophistication_level == 'Advanced')
        
        print(f"Total Attack Chains Analyzed: {total_chains}")
        print(f" Critical (APT) Chains: {critical_chains}")
        print(f" Advanced Chains: {advanced_chains}")
        
        if total_chains > 0:
            avg_stages = np.mean([c.total_stages for c in attack_chains])
            avg_coverage = np.mean([c.kill_chain_coverage for c in attack_chains])
            
            print(f" Average Kill Chain Stages: {avg_stages:.1f}")
            print(f" Average Kill Chain Coverage: {avg_coverage:.1%}")
            
            ###Show sophistication distribution
            print(f"\n SOPHISTICATION DISTRIBUTION:")
            sophistication_counts = Counter(c.sophistication_level for c in attack_chains)
            #for level, count in sophistication_counts.most_common():
             #   percentage = count / total_chains * 100
             ##  print(f"   {level:>25}: {count:3d} chains {bar} ({percentage:.1f}%)")
            
            # Show attack pattern detection
            print(f"\n ATTACK PATTERN DETECTION:")
            persistence_rate = sum(1 for c in attack_chains if c.persistence_detected) / total_chains
            lateral_rate = sum(1 for c in attack_chains if c.lateral_movement_detected) / total_chains
            exfil_rate = sum(1 for c in attack_chains if c.data_exfiltration_detected) / total_chains
            cred_rate = sum(1 for c in attack_chains if c.credential_access_detected) / total_chains
            
            print(f"   Persistence Detected:     {persistence_rate:.1%}")
            print(f"   Lateral Movement:         {lateral_rate:.1%}")
            print(f"   Data Exfiltration:        {exfil_rate:.1%}")
            print(f"   Credential Access:        {cred_rate:.1%}")
        
        print(f"\n Results exported to:")
        print(f"   • {args.output_prefix}_summary.csv")
        print(f"   • {args.output_prefix}_detailed.csv")
        print(f"   • {args.output_prefix}_report.json")
        
        print("\n Attack Chain Analysis Complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f" Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())