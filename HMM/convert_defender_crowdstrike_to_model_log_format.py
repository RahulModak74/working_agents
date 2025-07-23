#!/usr/bin/env python3
"""
Multi-Format Security Log Converter
Converts enhanced security logs (Defender/CrowdStrike/Cloudflare) to format expected by existing models
Converts out put of create_detailed_defender_crowdstrike_synethetic_data.py to the format created by create_synethetic_data.py so that fixed_pyro_2.py
can run
"""

import json
import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
import random
from pathlib import Path

class LogFormatConverter:
    """Converts multi-format logs to standardized model input format"""
    
    def __init__(self):
        self.session_counter = 0
        self.user_mapping = {}
        self.machine_mapping = {}
    
    def convert_logs_to_model_format(self, logs: List[Dict], source_format: str) -> Tuple[List[Dict], Dict[str, Dict], Dict]:
        """
        Convert enhanced logs to format expected by existing models
        
        Args:
            logs: List of logs from enhanced generator
            source_format: 'defender', 'crowdstrike', or 'cloudflare'
            
        Returns:
            Tuple of (converted_alerts, sessions, analysis_results)
        """
        print(f"Converting {len(logs)} {source_format} logs to model format...")
        
        # Convert individual logs to alert format
        converted_alerts = []
        for i, log in enumerate(logs):
            alert = self._convert_single_log(log, source_format, i)
            if alert:
                converted_alerts.append(alert)
        
        # Group into sessions
        sessions = self._group_alerts_into_sessions(converted_alerts)
        
        # Calculate analysis results
        results = self._calculate_analysis_results(sessions)
        
        print(f"Converted to {len(converted_alerts)} alerts in {len(sessions)} sessions")
        
        return converted_alerts, sessions, results
    
    def _convert_single_log(self, log: Dict, source_format: str, index: int) -> Optional[Dict]:
        """Convert a single log entry to alert format"""
        
        try:
            if source_format == 'defender':
                return self._convert_defender_log(log, index)
            elif source_format == 'crowdstrike':
                return self._convert_crowdstrike_log(log, index)
            elif source_format == 'cloudflare':
                return self._convert_cloudflare_log(log, index)
            else:
                raise ValueError(f"Unknown source format: {source_format}")
        except Exception as e:
            print(f"Warning: Failed to convert log {index}: {str(e)}")
            return None
    
    def _convert_defender_log(self, log: Dict, index: int) -> Dict:
        """Convert Microsoft Defender log to alert format"""
        
        # Extract timestamp
        timestamp = log.get('Timestamp', datetime.datetime.now().isoformat())
        
        # Determine event type based on ActionType
        action_type = log.get('ActionType', 'Unknown')
        if 'Process' in action_type:
            event_type = 'Process'
            details = {
                'file_name': log.get('FileName', 'unknown.exe'),
                'command_line': log.get('ProcessCommandLine', ''),
                'process_id': log.get('ProcessId', random.randint(1000, 9999))
            }
        elif 'File' in action_type:
            event_type = 'File'
            details = {
                'file_name': log.get('FileName', 'unknown.txt'),
                'action': action_type,
                'hash': log.get('SHA256', '').lower() or self._generate_hash(log.get('FileName', ''))
            }
        elif 'Network' in action_type:
            event_type = 'Network'
            details = {
                'remote_ip': log.get('RemoteIP', '0.0.0.0'),
                'remote_port': log.get('RemotePort', 80),
                'protocol': log.get('Protocol', 'TCP')
            }
        else:
            # Default to Process for unknown types
            event_type = 'Process'
            details = {
                'file_name': log.get('FileName', log.get('InitiatingProcessFileName', 'unknown.exe')),
                'command_line': log.get('ProcessCommandLine', log.get('InitiatingProcessCommandLine', '')),
                'process_id': log.get('ProcessId', random.randint(1000, 9999))
            }
        
        # Check for attack indicators
        attack_indicator = self._detect_attack_indicators(log, details)
        
        return {
            'timestamp': timestamp,
            'event_type': event_type,
            'device_name': log.get('DeviceName', 'UNKNOWN-PC'),
            'user_name': f"{log.get('AccountDomain', 'DOMAIN')}\\{log.get('AccountName', 'user')}",
            'details': details,
            'raw': {
                'source': 'defender_converted',
                'session_id': self._generate_session_id(log.get('DeviceName', ''), log.get('AccountName', '')),
                'alert_index': index,
                'attack_indicator': attack_indicator,
                'original_log': log
            }
        }
    
    def _convert_crowdstrike_log(self, log: Dict, index: int) -> Dict:
        """Convert CrowdStrike log to alert format"""
        
        # Extract metadata and event data
        metadata = log.get('metadata', {})
        event = log.get('event', {})
        
        # Extract timestamp
        event_time = metadata.get('eventCreationTime', datetime.datetime.now().timestamp() * 1000)
        timestamp = datetime.datetime.fromtimestamp(int(event_time) / 1000).isoformat()
        
        # Determine event type
        event_type_raw = metadata.get('eventType', 'Unknown')
        if 'Process' in event_type_raw:
            event_type = 'Process'
            details = {
                'file_name': event.get('ImageFileName', '').split('\\')[-1] or 'unknown.exe',
                'command_line': event.get('CommandLine', ''),
                'process_id': event.get('ProcessId', random.randint(1000, 9999))
            }
        elif 'File' in event_type_raw:
            event_type = 'File'
            details = {
                'file_name': event.get('FileName', event.get('FilePath', '').split('\\')[-1]) or 'unknown.txt',
                'action': 'FileCreated' if 'Written' in event_type_raw else 'FileAccessed',
                'hash': event.get('SHA256HashData', '').lower() or self._generate_hash(event.get('FileName', ''))
            }
        elif 'Network' in event_type_raw:
            event_type = 'Network'
            details = {
                'remote_ip': event.get('RemoteAddressIP4', '0.0.0.0'),
                'remote_port': event.get('RemotePort', 80),
                'protocol': event.get('Protocol', 'TCP')
            }
        else:
            event_type = 'Process'
            details = {
                'file_name': 'unknown.exe',
                'command_line': '',
                'process_id': random.randint(1000, 9999)
            }
        
        # Check for attack indicators
        attack_indicator = self._detect_attack_indicators(log, details)
        
        return {
            'timestamp': timestamp,
            'event_type': event_type,
            'device_name': event.get('ComputerName', 'UNKNOWN-PC'),
            'user_name': event.get('UserName', 'DOMAIN\\user'),
            'details': details,
            'raw': {
                'source': 'crowdstrike_converted',
                'session_id': self._generate_session_id(event.get('ComputerName', ''), event.get('UserName', '')),
                'alert_index': index,
                'attack_indicator': attack_indicator,
                'original_log': log
            }
        }
    
    def _convert_cloudflare_log(self, log: Dict, index: int) -> Dict:
        """Convert Cloudflare log to alert format"""
        
        # Extract timestamp
        timestamp = log.get('EdgeEndTimestamp', log.get('Timestamp', datetime.datetime.now().isoformat()))
        
        # Determine event type based on available fields
        if log.get('ClientRequestMethod'):
            # HTTP request
            event_type = 'Network'
            details = {
                'remote_ip': log.get('ClientIP', '0.0.0.0'),
                'remote_port': 80 if log.get('ClientRequestMethod') == 'GET' else 443,
                'protocol': 'HTTP'
            }
        elif log.get('QueryName'):
            # DNS query
            event_type = 'Network'
            details = {
                'remote_ip': log.get('SourceIP', log.get('ClientIP', '0.0.0.0')),
                'remote_port': 53,
                'protocol': 'DNS'
            }
        else:
            # Default to network event
            event_type = 'Network'
            details = {
                'remote_ip': log.get('ClientIP', '0.0.0.0'),
                'remote_port': 80,
                'protocol': 'TCP'
            }
        
        # Extract device/user info (simulated for Cloudflare)
        client_ip = log.get('ClientIP', '0.0.0.0')
        host = log.get('ClientRequestHost', log.get('EdgeRequestHost', 'unknown.com'))
        
        # Check for attack indicators
        attack_indicator = self._detect_attack_indicators(log, details)
        
        return {
            'timestamp': timestamp,
            'event_type': event_type,
            'device_name': f"client-{client_ip.replace('.', '-')}",
            'user_name': f"web\\{host.split('.')[0]}",
            'details': details,
            'raw': {
                'source': 'cloudflare_converted',
                'session_id': self._generate_session_id(client_ip, host),
                'alert_index': index,
                'attack_indicator': attack_indicator,
                'original_log': log
            }
        }
    
    def _detect_attack_indicators(self, log: Dict, details: Dict) -> bool:
        """Detect if log contains attack indicators"""
        attack_patterns = [
            'powershell.exe -executionpolicy bypass',
            'cmd.exe /c',
            'wmic process',
            'net user',
            'whoami',
            'passwords.txt',
            'credentials.db',
            'keylogger.exe',
            'malware',
            'phishing',
            'suspicious',
            'command injection',
            'directory traversal',
            'shell.php',
            '../../'
        ]
        
        # Convert log to string for pattern matching
        log_str = json.dumps(log).lower()
        details_str = json.dumps(details).lower()
        
        # Check for attack patterns
        for pattern in attack_patterns:
            if pattern in log_str or pattern in details_str:
                return True
        
        # Check specific fields for Cloudflare WAF blocks
        if log.get('WAFAction') == 'block' or log.get('FirewallMatchesActions', []):
            return True
        
        # Check CrowdStrike threat intelligence
        if isinstance(log.get('event'), dict):
            threat_intel = log['event'].get('ThreatIntel', {})
            if isinstance(threat_intel, dict) and threat_intel.get('ReputationScore', 100) < 50:
                return True
        
        return False
    
    def _generate_session_id(self, device: str, user: str) -> str:
        """Generate consistent session ID"""
        device_clean = device.replace('-', '_').replace('.', '_').replace('\\', '_').lower()
        user_clean = user.replace('\\', '_').replace('.', '_').lower()
        return f"{user_clean}_{device_clean}"
    
    def _generate_hash(self, filename: str) -> str:
        """Generate consistent hash for filename"""
        return hashlib.sha256(filename.encode()).hexdigest()
    
    def _group_alerts_into_sessions(self, alerts: List[Dict]) -> Dict[str, Dict]:
        """Group alerts into sessions (matches original format)"""
        sessions = {}
        
        for alert in alerts:
            session_id = alert['raw']['session_id']
            
            if session_id not in sessions:
                sessions[session_id] = {
                    'session_id': session_id,
                    'user_name': alert['user_name'],
                    'device_name': alert['device_name'],
                    'alerts': [],
                    'first_alert': alert['timestamp'],
                    'last_alert': alert['timestamp'],
                    'event_types': set(),
                    'attack_indicators': []
                }
            
            # Add alert to session
            sessions[session_id]['alerts'].append(alert)
            sessions[session_id]['last_alert'] = alert['timestamp']
            sessions[session_id]['event_types'].add(alert['event_type'])
            
            # Track attack indicators
            if alert['raw'].get('attack_indicator'):
                sessions[session_id]['attack_indicators'].append({
                    'stage': self._determine_attack_stage(alert),
                    'intensity': self._calculate_attack_intensity(alert)
                })
        
        # Convert sets to lists and add metadata
        for session in sessions.values():
            session['event_types'] = list(session['event_types'])
            session['alert_count'] = len(session['alerts'])
            session['is_attack_session'] = len(session['attack_indicators']) > 0
            
            if session['is_attack_session']:
                session['avg_attack_intensity'] = sum(ai['intensity'] for ai in session['attack_indicators']) / len(session['attack_indicators'])
            else:
                session['avg_attack_intensity'] = 0
        
        return sessions
    
    def _determine_attack_stage(self, alert: Dict) -> str:
        """Determine attack stage based on alert content"""
        details = alert.get('details', {})
        log_content = json.dumps(alert).lower()
        
        # Stage classification based on indicators
        if any(x in log_content for x in ['reconnaissance', 'whoami', 'net user']):
            return 'reconnaissance'
        elif any(x in log_content for x in ['privilege', 'admin', 'runas']):
            return 'privilege_escalation'
        elif any(x in log_content for x in ['lateral', 'wmic', 'psexec']):
            return 'lateral_movement'
        elif any(x in log_content for x in ['password', 'credential', 'keylog']):
            return 'data_collection'
        elif any(x in log_content for x in ['exfil', 'upload', 'ftp']):
            return 'exfiltration'
        elif any(x in log_content for x in ['encrypt', 'ransom', 'lock']):
            return 'ransomware_execution'
        else:
            return 'initial_foothold'
    
    def _calculate_attack_intensity(self, alert: Dict) -> float:
        """Calculate attack intensity score (0.0 to 1.0)"""
        log_content = json.dumps(alert).lower()
        intensity = 0.0
        
        # Base intensity on attack indicators
        high_intensity_patterns = ['powershell -executionpolicy bypass', 'wmic process call create', 'net user administrator']
        medium_intensity_patterns = ['cmd.exe /c', 'whoami', 'net user']
        low_intensity_patterns = ['suspicious', 'unusual']
        
        if any(pattern in log_content for pattern in high_intensity_patterns):
            intensity = 0.8
        elif any(pattern in log_content for pattern in medium_intensity_patterns):
            intensity = 0.5
        elif any(pattern in log_content for pattern in low_intensity_patterns):
            intensity = 0.3
        else:
            intensity = 0.1
        
        # Add randomness to make it more realistic
        intensity += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, intensity))
    
    def _calculate_analysis_results(self, sessions: Dict[str, Dict]) -> Dict:
        """Calculate analysis results (matches original format)"""
        total_sessions = len(sessions)
        attack_sessions = [s for s in sessions.values() if s['is_attack_session']]
        
        # Calculate risk scores for sessions
        for session in sessions.values():
            risk_score = self._calculate_session_risk_score(session)
            session['risk_score'] = risk_score
            session['is_suspicious'] = risk_score >= 60  # Threshold
        
        actually_suspicious = len([s for s in sessions.values() if s['is_suspicious']])
        
        results = {
            'total_alerts': sum(s['alert_count'] for s in sessions.values()),
            'total_sessions': total_sessions,
            'suspicious_sessions': actually_suspicious,
            'llm_analyzed': actually_suspicious,
            'attack_sessions_embedded': len(attack_sessions),
            'detection_accuracy': len(attack_sessions) / max(1, actually_suspicious) if actually_suspicious > 0 else 0,
            'session_breakdown': {
                'safe': total_sessions - actually_suspicious,
                'medium_risk': len([s for s in sessions.values() if 60 <= s['risk_score'] < 80]),
                'high_risk': len([s for s in sessions.values() if 80 <= s['risk_score'] < 90]),
                'critical': len([s for s in sessions.values() if s['risk_score'] >= 90])
            }
        }
        
        return results
    
    def _calculate_session_risk_score(self, session: Dict) -> int:
        """Calculate risk score based on session characteristics"""
        score = 0
        alert_count = session['alert_count']
        event_types = session['event_types']
        
        # Rule 1: High alert volume
        if alert_count > 10:
            score += 30
        
        # Rule 2: Multi-vector attack
        if len(event_types) >= 3:
            score += 40
        
        # Rule 3: Rapid sequence (approximated by high alert count)
        if alert_count >= 8:
            score += 35
        
        # Rule 4: Attack indicators (if embedded)
        if session['is_attack_session']:
            score += int(session['avg_attack_intensity'] * 50)
        
        # Rule 5: Time-based anomalies (simulated)
        if random.random() < 0.1:
            score += 25
        
        return min(score, 100)

def load_enhanced_logs(file_path: str) -> Tuple[List[Dict], str]:
    """Load logs from enhanced generator and detect format"""
    
    with open(file_path, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        raise ValueError("No logs found in file")
    
    # Detect format based on log structure
    sample_log = logs[0]
    
    if 'DeviceName' in sample_log and 'ActionType' in sample_log:
        return logs, 'defender'
    elif 'metadata' in sample_log and 'event' in sample_log:
        return logs, 'crowdstrike'
    elif 'ClientIP' in sample_log or 'QueryName' in sample_log:
        return logs, 'cloudflare'
    else:
        raise ValueError("Unknown log format - cannot detect format automatically")

def convert_and_save(input_file: str, output_prefix: str = None):
    """Convert logs and save in model-ready format"""
    
    print(f"Loading logs from {input_file}...")
    logs, detected_format = load_enhanced_logs(input_file)
    
    print(f"Detected format: {detected_format}")
    print(f"Found {len(logs)} logs to convert")
    
    # Convert logs
    converter = LogFormatConverter()
    alerts, sessions, results = converter.convert_logs_to_model_format(logs, detected_format)
    
    # Generate output filenames
    if output_prefix is None:
        output_prefix = f"converted_{detected_format}"
    
    alerts_file = f"{output_prefix}_alerts.json"
    sessions_file = f"{output_prefix}_sessions.json"
    results_file = f"{output_prefix}_results.json"
    
    # Save converted data
    print(f"\nSaving converted data...")
    
    with open(alerts_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    print(f"✅ Alerts saved to {alerts_file}")
    
    with open(sessions_file, 'w') as f:
        json.dump(sessions, f, indent=2, default=str)
    print(f"✅ Sessions saved to {sessions_file}")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {results_file}")
    
    # Print summary
    print(f"\n📊 CONVERSION SUMMARY:")
    print(f"   Original logs: {len(logs)} ({detected_format} format)")
    print(f"   Converted alerts: {len(alerts)}")
    print(f"   Sessions created: {len(sessions)}")
    print(f"   Suspicious sessions: {results['suspicious_sessions']}")
    print(f"   Attack sessions: {results['attack_sessions_embedded']}")
    
    return alerts, sessions, results

def export_for_hmm_training(sessions: Dict[str, Dict], output_file: str = 'converted_hmm_training_data.json'):
    """Export converted sessions for HMM training (matches original format)"""
    training_data = []
    
    for session_id, session in sessions.items():
        # Extract features for HMM observations (matching original format)
        features = {
            'session_id': session_id,
            'risk_probability': session['risk_score'] / 100.0,
            'alert_count': session['alert_count'],
            'event_type_diversity': len(session['event_types']),
            'temporal_span_minutes': 60,  # Simplified
            'multi_vector': len(session['event_types']) >= 3,
            'is_attack': session['is_attack_session'],
            'attack_intensity': session.get('avg_attack_intensity', 0.0)
        }
        
        # Add ground truth labels for validation
        if session['is_attack_session'] and session['attack_indicators']:
            features['true_attack_stage'] = session['attack_indicators'][0]['stage']
        else:
            features['true_attack_stage'] = 'normal_operations'
        
        training_data.append(features)
    
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ HMM training data exported to {output_file}")
    return training_data

def main():
    """Main conversion workflow"""
    print("🔄 Multi-Format Security Log Converter")
    print("=" * 50)
    
    # Example usage with different formats
    example_files = [
        'synthetic_defender_logs.json',
        'synthetic_crowdstrike_logs.json', 
        'synthetic_cloudflare_logs.json'
    ]
    
    print("This script converts enhanced security logs to your model format.")
    print("\nUsage examples:")
    print("1. python converter.py --input synthetic_defender_logs.json")
    print("2. python converter.py --input synthetic_crowdstrike_logs.json --output crowdstrike_converted")
    print("3. Auto-detect format and convert all files")
    
    # Check for existing files and convert them
    converted_data = {}
    
    for example_file in example_files:
        if Path(example_file).exists():
            print(f"\n🔧 Converting {example_file}...")
            try:
                alerts, sessions, results = convert_and_save(example_file)
                
                # Export HMM training data
                hmm_file = example_file.replace('.json', '_hmm_training.json')
                export_for_hmm_training(sessions, hmm_file)
                
                converted_data[example_file] = {
                    'alerts': len(alerts),
                    'sessions': len(sessions),
                    'suspicious': results['suspicious_sessions'],
                    'accuracy': results['detection_accuracy']
                }
                
            except Exception as e:
                print(f"❌ Failed to convert {example_file}: {str(e)}")
    
    if converted_data:
        print(f"\n🎉 CONVERSION COMPLETE!")
        print(f"Successfully converted {len(converted_data)} files:")
        for file, stats in converted_data.items():
            print(f"   {file}: {stats['alerts']} alerts → {stats['sessions']} sessions")
            print(f"      Suspicious: {stats['suspicious']}, Accuracy: {stats['accuracy']:.2%}")
    else:
        print("\n⚠️  No enhanced log files found to convert.")
        print("Run the enhanced generator first to create log files.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert enhanced security logs to model format')
    parser.add_argument('--input', '-i', help='Input log file to convert')
    parser.add_argument('--output', '-o', help='Output prefix for converted files')
    parser.add_argument('--format', '-f', choices=['defender', 'crowdstrike', 'cloudflare'], 
                       help='Force specific format (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    if args.input:
        if Path(args.input).exists():
            convert_and_save(args.input, args.output)
        else:
            print(f"❌ Input file {args.input} not found")
    else:
        main()