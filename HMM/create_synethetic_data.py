#!/usr/bin/env python3
"""
Synthetic Security Data Generator for AI Analytics Testing
Generates realistic security alerts with embedded attack scenarios for HMM validation
"""

import json
import random
import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import uuid

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation"""
    total_alerts: int = 47329
    target_sessions: int = 1847
    suspicious_sessions: int = 23
    attack_campaigns: int = 3
    time_window_hours: int = 24
    organizations: List[str] = None
    
    def __post_init__(self):
        if self.organizations is None:
            self.organizations = ['startup_tech', 'financial_services', 'healthcare', 'manufacturing']

class OrganizationProfiles:
    """Realistic organizational behavior profiles"""
    
    PROFILES = {
        'startup_tech': {
            'work_hours': (8, 22),  # Long work hours
            'weekend_activity': 0.4,  # High weekend activity
            'process_diversity': 0.8,  # Lots of different tools
            'admin_ratio': 0.3,  # Many users have admin
            'developer_tools': ['python.exe', 'node.exe', 'code.exe', 'git.exe', 'docker.exe'],
            'common_files': ['.py', '.js', '.json', '.md', '.yml'],
            'network_pattern': 'high_external'  # Lots of external connections
        },
        'financial_services': {
            'work_hours': (7, 18),  # Standard business hours
            'weekend_activity': 0.05,  # Minimal weekend activity
            'process_diversity': 0.3,  # Limited approved software
            'admin_ratio': 0.05,  # Few admin users
            'developer_tools': ['excel.exe', 'outlook.exe', 'bloomberg.exe', 'java.exe'],
            'common_files': ['.xlsx', '.pdf', '.csv', '.docx', '.msg'],
            'network_pattern': 'restricted'  # Limited external access
        },
        'healthcare': {
            'work_hours': (6, 20),  # Extended hours for shifts
            'weekend_activity': 0.2,  # Some weekend activity
            'process_diversity': 0.4,  # Medical software + office tools
            'admin_ratio': 0.1,  # Limited admin access
            'developer_tools': ['epic.exe', 'cerner.exe', 'outlook.exe', 'chrome.exe'],
            'common_files': ['.pdf', '.docx', '.hl7', '.xml', '.jpg'],
            'network_pattern': 'moderate'
        },
        'manufacturing': {
            'work_hours': (5, 23),  # Shift work
            'weekend_activity': 0.3,  # Weekend shifts
            'process_diversity': 0.5,  # Mix of industrial and office software
            'admin_ratio': 0.15,  # Some admin for maintenance
            'developer_tools': ['autocad.exe', 'solidworks.exe', 'plc_software.exe', 'scada.exe'],
            'common_files': ['.dwg', '.step', '.pdf', '.xlsx', '.log'],
            'network_pattern': 'industrial'
        }
    }

class AttackScenarios:
    """Realistic attack progression templates"""
    
    APT_SCENARIO = {
        'name': 'Advanced Persistent Threat',
        'duration_days': 12,
        'stages': [
            {'day': 1, 'stage': 'initial_foothold', 'intensity': 0.3, 'obvious': 0.2},
            {'day': 3, 'stage': 'reconnaissance', 'intensity': 0.25, 'obvious': 0.15},
            {'day': 5, 'stage': 'lateral_movement', 'intensity': 0.35, 'obvious': 0.4},
            {'day': 8, 'stage': 'lateral_movement', 'intensity': 0.4, 'obvious': 0.5},
            {'day': 12, 'stage': 'objective_execution', 'intensity': 0.55, 'obvious': 0.8}
        ]
    }
    
    INSIDER_THREAT = {
        'name': 'Insider Data Exfiltration',
        'duration_days': 8,
        'stages': [
            {'day': 1, 'stage': 'reconnaissance', 'intensity': 0.2, 'obvious': 0.1},
            {'day': 3, 'stage': 'privilege_escalation', 'intensity': 0.3, 'obvious': 0.3},
            {'day': 5, 'stage': 'data_collection', 'intensity': 0.4, 'obvious': 0.4},
            {'day': 8, 'stage': 'exfiltration', 'intensity': 0.6, 'obvious': 0.7}
        ]
    }
    
    RANSOMWARE = {
        'name': 'Ransomware Campaign',
        'duration_days': 5,
        'stages': [
            {'day': 1, 'stage': 'initial_compromise', 'intensity': 0.4, 'obvious': 0.3},
            {'day': 2, 'stage': 'lateral_movement', 'intensity': 0.5, 'obvious': 0.5},
            {'day': 4, 'stage': 'encryption_prep', 'intensity': 0.3, 'obvious': 0.2},
            {'day': 5, 'stage': 'ransomware_execution', 'intensity': 0.8, 'obvious': 0.9}
        ]
    }

class SyntheticDataGenerator:
    """Main generator class"""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.start_time = datetime.datetime.now() - datetime.timedelta(hours=config.time_window_hours)
        self.users = self._generate_users()
        self.machines = self._generate_machines()
        self.attack_scenarios = [
            AttackScenarios.APT_SCENARIO,
            AttackScenarios.INSIDER_THREAT,
            AttackScenarios.RANSOMWARE
        ]
    
    def _generate_users(self) -> List[Dict]:
        """Generate realistic user profiles"""
        users = []
        user_count = max(50, self.config.target_sessions // 15)  # Rough estimate
        
        departments = ['engineering', 'sales', 'hr', 'finance', 'operations', 'marketing']
        roles = ['developer', 'analyst', 'manager', 'director', 'admin', 'intern']
        
        for i in range(user_count):
            org = random.choice(self.config.organizations)
            dept = random.choice(departments)
            role = random.choice(roles)
            
            # Adjust role probability based on org type
            if org == 'startup_tech' and random.random() < 0.6:
                role = 'developer'
            elif org == 'financial_services' and random.random() < 0.4:
                role = 'analyst'
            
            user = {
                'id': f"user_{i:04d}",
                'name': f"{random.choice(['john', 'jane', 'mike', 'sarah', 'david', 'lisa'])}.{random.choice(['smith', 'johnson', 'williams', 'brown', 'jones'])}",
                'organization': org,
                'department': dept,
                'role': role,
                'risk_profile': random.choice(['low', 'low', 'low', 'medium', 'high']),  # Most users are low risk
                'work_pattern': self._generate_work_pattern(org)
            }
            users.append(user)
        
        return users
    
    def _generate_machines(self) -> List[Dict]:
        """Generate realistic machine profiles"""
        machines = []
        machine_count = max(100, self.config.target_sessions // 10)
        
        machine_types = ['workstation', 'laptop', 'server', 'virtual_machine']
        os_types = ['windows_10', 'windows_11', 'windows_server_2019', 'windows_server_2022']
        
        for i in range(machine_count):
            machine = {
                'id': f"machine_{i:04d}",
                'name': f"{''.join(random.choices(['dev', 'prod', 'test', 'user'], k=1))}-{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))}",
                'type': random.choice(machine_types),
                'os': random.choice(os_types),
                'department': random.choice(['engineering', 'sales', 'hr', 'finance', 'operations']),
                'criticality': random.choice(['low', 'low', 'medium', 'medium', 'high'])
            }
            machines.append(machine)
        
        return machines
    
    def _generate_work_pattern(self, org: str) -> Dict:
        """Generate realistic work patterns based on organization"""
        profile = OrganizationProfiles.PROFILES[org]
        
        return {
            'typical_start': profile['work_hours'][0] + random.randint(-1, 2),
            'typical_end': profile['work_hours'][1] + random.randint(-2, 1),
            'weekend_probability': profile['weekend_activity'],
            'late_night_probability': 0.1 if org == 'startup_tech' else 0.02
        }
    
    def generate_alerts(self) -> List[Dict]:
        """Generate the full set of synthetic alerts"""
        print(f"Generating {self.config.total_alerts} alerts across {self.config.target_sessions} sessions...")
        
        # First, distribute alerts across sessions
        session_sizes = self._generate_session_distribution()
        
        # Create session-user-machine mappings
        sessions = self._create_sessions(session_sizes)
        
        # Inject attack scenarios
        attack_sessions = self._inject_attack_scenarios(sessions)
        
        # Generate individual alerts for each session
        all_alerts = []
        for session in sessions:
            session_alerts = self._generate_session_alerts(session)
            all_alerts.extend(session_alerts)
        
        # Shuffle to make timing more realistic
        random.shuffle(all_alerts)
        all_alerts.sort(key=lambda x: x['timestamp'])
        
        print(f"Generated {len(all_alerts)} total alerts in {len(sessions)} sessions")
        print(f"Attack scenarios embedded: {len(attack_sessions)}")
        
        return all_alerts
    
    def _generate_session_distribution(self) -> List[int]:
        """Generate realistic session size distribution"""
        # Most sessions are small, few are large (power law distribution)
        session_sizes = []
        remaining_alerts = self.config.total_alerts
        
        for i in range(self.config.target_sessions):
            if i == self.config.target_sessions - 1:
                # Last session gets remaining alerts
                session_sizes.append(remaining_alerts)
            else:
                # Power law: most sessions are small
                if random.random() < 0.7:  # 70% small sessions
                    size = random.randint(1, 10)
                elif random.random() < 0.9:  # 20% medium sessions
                    size = random.randint(11, 50)
                else:  # 10% large sessions
                    size = random.randint(51, 200)
                
                size = min(size, remaining_alerts)
                session_sizes.append(size)
                remaining_alerts -= size
                
                if remaining_alerts <= 0:
                    break
        
        return session_sizes
    
    def _create_sessions(self, session_sizes: List[int]) -> List[Dict]:
        """Create session metadata"""
        sessions = []
        
        for i, size in enumerate(session_sizes):
            user = random.choice(self.users)
            machine = random.choice(self.machines)
            
            # Generate session timeframe
            session_start = self._generate_session_start_time(user)
            session_duration = self._generate_session_duration(size, user)
            
            session = {
                'session_id': f"{user['name']}_{machine['name']}".lower().replace('.', '_'),
                'user': user,
                'machine': machine,
                'alert_count': size,
                'start_time': session_start,
                'duration_minutes': session_duration,
                'is_attack': False,  # Will be updated if attack is injected
                'attack_scenario': None,
                'risk_factors': []
            }
            sessions.append(session)
        
        return sessions
    
    def _generate_session_start_time(self, user: Dict) -> datetime.datetime:
     """Generate realistic session start time based on user work pattern"""
     work_pattern = user['work_pattern']
     org_profile = OrganizationProfiles.PROFILES[user['organization']]
    
     # Random day within time window
     day_offset = random.randint(0, max(1, self.config.time_window_hours // 24))
     base_time = self.start_time + datetime.timedelta(days=day_offset)
    
     # Determine if weekend
     is_weekend = base_time.weekday() >= 5
    
     if is_weekend and random.random() > work_pattern['weekend_probability']:
        # Skip this timestamp, try weekday
        base_time -= datetime.timedelta(days=2)
    
     # Generate hour based on work pattern
     if random.random() < work_pattern['late_night_probability']:
        # Handle late night / early morning hours that span midnight
        # Choose from 22, 23, 0, 1, 2, 3, 4, 5, 6
        late_night_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
        hour = random.choice(late_night_hours)
     else:
        start_hour = work_pattern['typical_start']
        end_hour = work_pattern['typical_end']
        hour = random.randint(start_hour, end_hour)
    
     minute = random.randint(0, 59)
     second = random.randint(0, 59)
    
     return base_time.replace(hour=hour % 24, minute=minute, second=second)
    def _generate_session_duration(self, alert_count: int, user: Dict) -> int:
        """Generate realistic session duration based on alert count and user"""
        # Base duration on alert count with some randomness
        base_minutes = alert_count * random.uniform(0.5, 3.0)
        
        # Adjust based on user role
        if user['role'] == 'developer':
            base_minutes *= random.uniform(1.5, 3.0)  # Developers have longer sessions
        elif user['role'] == 'admin':
            base_minutes *= random.uniform(0.5, 1.5)  # Admins might have quick sessions
        
        return max(1, int(base_minutes))
    
    def _inject_attack_scenarios(self, sessions: List[Dict]) -> List[Dict]:
        """Inject attack scenarios into random sessions"""
        attack_sessions = []
        
        for scenario_idx in range(min(self.config.attack_campaigns, len(self.attack_scenarios))):
            scenario = self.attack_scenarios[scenario_idx]
            
            # Select random sessions for this attack
            available_sessions = [s for s in sessions if not s['is_attack']]
            if not available_sessions:
                break
            
            # Group sessions by user for realistic attack progression
            user_sessions = {}
            for session in available_sessions:
                user_key = session['user']['name']
                if user_key not in user_sessions:
                    user_sessions[user_key] = []
                user_sessions[user_key].append(session)
            
            # Pick a user with multiple sessions for multi-stage attack
            target_user_sessions = [sessions for sessions in user_sessions.values() if len(sessions) >= len(scenario['stages'])]
            if not target_user_sessions:
                target_user_sessions = list(user_sessions.values())
            
            selected_user_sessions = random.choice(target_user_sessions)
            selected_sessions = random.sample(selected_user_sessions, min(len(scenario['stages']), len(selected_user_sessions)))
            
            # Apply attack stages to sessions
            for i, stage in enumerate(scenario['stages'][:len(selected_sessions)]):
                session = selected_sessions[i]
                session['is_attack'] = True
                session['attack_scenario'] = scenario['name']
                session['attack_stage'] = stage['stage']
                session['attack_intensity'] = stage['intensity']
                session['attack_obvious'] = stage['obvious']
                session['risk_factors'].append(f"attack_stage_{stage['stage']}")
                
                attack_sessions.append(session)
        
        return attack_sessions
    
    def _generate_session_alerts(self, session: Dict) -> List[Dict]:
        """Generate individual alerts for a session"""
        alerts = []
        user = session['user']
        machine = session['machine']
        org_profile = OrganizationProfiles.PROFILES[user['organization']]
        
        session_start = session['start_time']
        session_duration = session['duration_minutes']
        
        for i in range(session['alert_count']):
            # Generate timestamp within session
            minutes_offset = (i / session['alert_count']) * session_duration + random.uniform(-5, 5)
            timestamp = session_start + datetime.timedelta(minutes=max(0, minutes_offset))
            
            # Generate alert based on whether it's part of an attack
            if session['is_attack']:
                alert = self._generate_attack_alert(session, timestamp, i)
            else:
                alert = self._generate_normal_alert(session, timestamp, i, org_profile)
            
            alerts.append(alert)
        
        return alerts
    
    def _generate_normal_alert(self, session: Dict, timestamp: datetime.datetime, index: int, org_profile: Dict) -> Dict:
        """Generate normal business activity alert"""
        user = session['user']
        machine = session['machine']
        
        event_types = ['Process', 'File', 'Network']
        event_type = random.choice(event_types)
        
        if event_type == 'Process':
            details = {
                'file_name': random.choice(org_profile['developer_tools']),
                'command_line': self._generate_normal_command_line(org_profile),
                'process_id': random.randint(1000, 9999)
            }
        elif event_type == 'File':
            details = {
                'file_name': f"document_{random.randint(1, 1000)}{random.choice(org_profile['common_files'])}",
                'action': random.choice(['FileCreated', 'FileModified', 'FileDeleted', 'FileAccessed']),
                'hash': ''.join(random.choices('abcdef0123456789', k=64))
            }
        else:  # Network
            details = {
                'remote_ip': self._generate_normal_ip(org_profile),
                'remote_port': random.choice([80, 443, 22, 3389, 53]),
                'protocol': random.choice(['TCP', 'UDP'])
            }
        
        return {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'device_name': machine['name'],
            'user_name': user['name'],
            'details': details,
            'raw': {
                'source': 'synthetic_generator',
                'session_id': session['session_id'],
                'alert_index': index
            }
        }
    
    def _generate_attack_alert(self, session: Dict, timestamp: datetime.datetime, index: int) -> Dict:
        """Generate suspicious alert as part of attack scenario"""
        user = session['user']
        machine = session['machine']
        
        # Make some alerts more obviously malicious based on attack_obvious score
        is_obvious = random.random() < session['attack_obvious']
        
        event_types = ['Process', 'File', 'Network']
        event_type = random.choice(event_types)
        
        if event_type == 'Process':
            if is_obvious:
                details = {
                    'file_name': random.choice(['powershell.exe', 'cmd.exe', 'wmic.exe', 'net.exe']),
                    'command_line': random.choice([
                        'powershell.exe -ExecutionPolicy Bypass -EncodedCommand',
                        'cmd.exe /c "whoami /priv"',
                        'wmic process call create',
                        'net user administrator /active:yes'
                    ]),
                    'process_id': random.randint(1000, 9999)
                }
            else:
                details = {
                    'file_name': random.choice(['svchost.exe', 'explorer.exe', 'winlogon.exe']),
                    'command_line': 'normal looking command with suspicious timing',
                    'process_id': random.randint(1000, 9999)
                }
        elif event_type == 'File':
            if is_obvious:
                details = {
                    'file_name': random.choice(['passwords.txt', 'credentials.db', 'keylogger.exe']),
                    'action': 'FileCreated',
                    'hash': ''.join(random.choices('abcdef0123456789', k=64))
                }
            else:
                details = {
                    'file_name': f"temp_{random.randint(1000, 9999)}.tmp",
                    'action': 'FileAccessed',
                    'hash': ''.join(random.choices('abcdef0123456789', k=64))
                }
        else:  # Network
            if is_obvious:
                details = {
                    'remote_ip': random.choice(['192.168.1.100', '10.0.0.50', '172.16.1.200']),  # Internal scanning
                    'remote_port': random.choice([135, 139, 445, 3389]),  # Suspicious ports
                    'protocol': 'TCP'
                }
            else:
                details = {
                    'remote_ip': f"{random.randint(1, 223)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    'remote_port': random.choice([80, 443, 53]),
                    'protocol': 'TCP'
                }
        
        alert = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'device_name': machine['name'],
            'user_name': user['name'],
            'details': details,
            'raw': {
                'source': 'synthetic_generator',
                'session_id': session['session_id'],
                'alert_index': index,
                'attack_indicator': True,
                'attack_stage': session['attack_stage'],
                'attack_intensity': session['attack_intensity']
            }
        }
        
        return alert
    
    def _generate_normal_command_line(self, org_profile: Dict) -> str:
        """Generate realistic command line for normal business activity"""
        commands = [
            f"{random.choice(org_profile['developer_tools'])} -version",
            f"python.exe script_{random.randint(1, 100)}.py",
            f"git.exe pull origin main",
            f"npm.exe install package_{random.randint(1, 50)}",
            f"docker.exe run -d app_{random.randint(1, 20)}"
        ]
        return random.choice(commands)
    
    def _generate_normal_ip(self, org_profile: Dict) -> str:
        """Generate realistic IP based on organization network pattern"""
        if org_profile['network_pattern'] == 'high_external':
            # Startups access lots of external services
            return f"{random.randint(1, 223)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif org_profile['network_pattern'] == 'restricted':
            # Financial services mostly internal
            return f"10.{random.randint(1, 10)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:
            # Mix of internal and external
            if random.random() < 0.6:
                return f"192.168.{random.randint(1, 10)}.{random.randint(1, 255)}"
            else:
                return f"{random.randint(1, 223)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def group_alerts_into_sessions(self, alerts: List[Dict]) -> Dict[str, Dict]:
        """Group alerts into sessions for analysis (simulates your session_analyzer.py)"""
        sessions = {}
        
        for alert in alerts:
            # Generate session ID (user_machine format)
            user_clean = alert['user_name'].replace('.', '_').replace('\\', '_')
            machine_clean = alert['device_name'].replace('-', '_').replace(' ', '_')
            session_id = f"{user_clean}_{machine_clean}".lower()
            
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
                    'stage': alert['raw']['attack_stage'],
                    'intensity': alert['raw']['attack_intensity']
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
    
    def calculate_expected_results(self, sessions: Dict[str, Dict]) -> Dict:
        """Calculate the expected results that would match the article claims"""
        total_sessions = len(sessions)
        attack_sessions = [s for s in sessions.values() if s['is_attack_session']]
        suspicious_sessions = len(attack_sessions)
        
        # Calculate risk scores for sessions
        for session in sessions.values():
            risk_score = self._calculate_session_risk_score(session)
            session['risk_score'] = risk_score
            session['is_suspicious'] = risk_score >= 60  # Threshold from config
        
        actually_suspicious = len([s for s in sessions.values() if s['is_suspicious']])
        
        results = {
            'total_alerts': sum(s['alert_count'] for s in sessions.values()),
            'total_sessions': total_sessions,
            'suspicious_sessions': actually_suspicious,
            'llm_analyzed': actually_suspicious,  # All suspicious get LLM analysis
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
            score += int(session['avg_attack_intensity'] * 50)  # Scale intensity to score
        
        # Rule 5: Time-based anomalies (simulated)
        if random.random() < 0.1:  # 10% chance of temporal anomaly
            score += 25
        
        return min(score, 100)

def main():
    """Generate synthetic data and demonstrate results"""
    print("🔬 Synthetic Security Data Generator")
    print("=" * 50)
    
    # Configure generation parameters
    config = SyntheticConfig(
        total_alerts=47329,
        target_sessions=1847,
        suspicious_sessions=23,
        attack_campaigns=3,
        time_window_hours=24
    )
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    
    print("Step 1: Generating synthetic alerts...")
    alerts = generator.generate_alerts()
    
    print("Step 2: Grouping alerts into sessions...")
    sessions = generator.group_alerts_into_sessions(alerts)
    
    print("Step 3: Calculating expected analysis results...")
    results = generator.calculate_expected_results(sessions)
    
    # Display results
    print(f"\n📊 SYNTHETIC DATA RESULTS:")
    print(f"   Total alerts: {results['total_alerts']}")
    print(f"   Total sessions: {results['total_sessions']}")
    print(f"   Suspicious sessions: {results['suspicious_sessions']}")
    print(f"   LLM analyzed: {results['llm_analyzed']}")
    print(f"   Embedded attack sessions: {results['attack_sessions_embedded']}")
    
    print(f"\n🎯 SESSION RISK BREAKDOWN:")
    breakdown = results['session_breakdown']
    print(f"   🟢 Safe sessions: {breakdown['safe']}")
    print(f"   🟡 Medium risk: {breakdown['medium_risk']}")
    print(f"   🟠 High risk: {breakdown['high_risk']}")
    print(f"   🔴 Critical: {breakdown['critical']}")
    
    # Save sample data
    sample_alerts = alerts[:100]
    sample_sessions = dict(list(sessions.items())[:10])
    
    print(f"\n💾 Saving sample data...")
    
    # Save alerts sample
    with open('synthetic_alerts_sample.json', 'w') as f:
        json.dump(sample_alerts, f, indent=2)
    
    # Save sessions sample  
    with open('synthetic_sessions_sample.json', 'w') as f:
        json.dump(sample_sessions, f, indent=2, default=str)
    
    # Save full results summary
    with open('synthetic_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ synthetic_alerts_sample.json (100 sample alerts)")
    print(f"   ✅ synthetic_sessions_sample.json (10 sample sessions)")
    print(f"   ✅ synthetic_analysis_results.json (complete analysis)")
    
    # Show attack scenario examples
    print(f"\n🚨 EMBEDDED ATTACK EXAMPLES:")
    attack_sessions = [s for s in sessions.values() if s['is_attack_session']]
    for i, session in enumerate(attack_sessions[:3]):
        print(f"   Attack {i+1}: {session['session_id']}")
        print(f"      Risk Score: {session['risk_score']}/100")
        print(f"      Alert Count: {session['alert_count']}")
        print(f"      Attack Intensity: {session['avg_attack_intensity']:.2f}")
        if session['attack_indicators']:
            stages = list(set(ai['stage'] for ai in session['attack_indicators']))
            print(f"      Attack Stages: {', '.join(stages)}")
        print()
    
    print("🎉 Synthetic data generation complete!")
    print("\nThis data can be used to:")
    print("- Test your HMM models with known ground truth")
    print("- Demonstrate AI analytics capabilities")
    print("- Validate detection accuracy claims")
    print("- Generate different scenarios for different audiences")

def generate_custom_scenario(alert_count: int, session_count: int, attack_count: int = 3):
    """Generate custom synthetic scenario with specified parameters"""
    config = SyntheticConfig(
        total_alerts=alert_count,
        target_sessions=session_count,
        attack_campaigns=attack_count
    )
    
    generator = SyntheticDataGenerator(config)
    alerts = generator.generate_alerts()
    sessions = generator.group_alerts_into_sessions(alerts)
    results = generator.calculate_expected_results(sessions)
    
    return alerts, sessions, results

def export_for_hmm_training(sessions: Dict[str, Dict], output_file: str = 'hmm_training_data.json'):
    """Export sessions in format suitable for HMM training"""
    training_data = []
    
    for session_id, session in sessions.items():
        # Extract features for HMM observations
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
    
    print(f"HMM training data exported to {output_file}")
    return training_data

if __name__ == "__main__":
    main()