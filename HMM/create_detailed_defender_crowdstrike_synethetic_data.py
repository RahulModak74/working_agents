#!/usr/bin/env python3
"""
Enhanced Pluggable Security Log Generator
Generates realistic security logs in exact formats for Defender, CrowdStrike, and Cloudflare
"""

import json
import random
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
import base64
import hashlib
from abc import ABC, abstractmethod

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation"""
    total_alerts: int = 47329
    target_sessions: int = 1847
    suspicious_sessions: int = 23
    attack_campaigns: int = 3
    time_window_hours: int = 24
    organizations: List[str] = None
    log_format: str = "defender"  # defender, crowdstrike, cloudflare
    
    def __post_init__(self):
        if self.organizations is None:
            self.organizations = ['startup_tech', 'financial_services', 'healthcare', 'manufacturing']

class LogFormatProvider(ABC):
    """Abstract base class for log format providers"""
    
    @abstractmethod
    def format_process_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        pass
    
    @abstractmethod
    def format_file_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        pass
    
    @abstractmethod
    def format_network_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        pass

class MicrosoftDefenderProvider(LogFormatProvider):
    """Microsoft Defender for Endpoint log format"""
    
    def __init__(self):
        self.machine_groups = ['Domain Controllers', 'Workstations', 'Servers', 'Test Machines']
        self.threat_types = ['Malware', 'PUA', 'Suspicious', 'Clean']
    
    def format_process_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        return {
            "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "DeviceId": f"device-{hashlib.md5(machine['name'].encode()).hexdigest()[:16]}",
            "DeviceName": machine['name'].upper(),
            "ActionType": "ProcessCreated",
            "FileName": details['file_name'],
            "FolderPath": f"C:\\{'Windows\\System32' if 'system' in details['file_name'].lower() else 'Program Files\\Application'}\\",
            "SHA256": hashlib.sha256(details['file_name'].encode()).hexdigest(),
            "ProcessCommandLine": details['command_line'],
            "ProcessId": details['process_id'],
            "ProcessCreationTime": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "AccountDomain": user['organization'].upper(),
            "AccountName": user['name'].split('.')[0],
            "AccountSid": f"S-1-5-21-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-{random.randint(1000, 9999)}",
            "LogonId": f"{random.randint(100000, 999999)}",
            "InitiatingProcessFileName": random.choice(['explorer.exe', 'cmd.exe', 'powershell.exe', 'services.exe']),
            "InitiatingProcessCommandLine": "N/A",
            "InitiatingProcessId": random.randint(1000, 9999),
            "InitiatingProcessParentFileName": "winlogon.exe",
            "InitiatingProcessParentId": random.randint(500, 1500),
            "InitiatingProcessAccountName": user['name'].split('.')[0],
            "InitiatingProcessAccountDomain": user['organization'].upper(),
            "InitiatingProcessLogonId": f"{random.randint(100000, 999999)}",
            "ReportId": random.randint(10000, 99999),
            "AppGuardContainerId": "",
            "AdditionalFields": json.dumps({
                "ThreatFamily": random.choice(self.threat_types) if session.get('is_attack') else "Clean",
                "DetectionSource": "AMSI" if 'powershell' in details['file_name'].lower() else "Behavior",
                "Severity": "High" if session.get('is_attack') and session.get('attack_obvious', 0) > 0.7 else "Medium"
            })
        }
    
    def format_file_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        return {
            "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "DeviceId": f"device-{hashlib.md5(machine['name'].encode()).hexdigest()[:16]}",
            "DeviceName": machine['name'].upper(),
            "ActionType": details['action'],
            "FileName": details['file_name'],
            "FolderPath": self._generate_realistic_path(details['file_name'], user),
            "SHA256": details['hash'],
            "FileSize": random.randint(1024, 10485760),  # 1KB to 10MB
            "AccountDomain": user['organization'].upper(),
            "AccountName": user['name'].split('.')[0],
            "ProcessName": random.choice(['explorer.exe', 'notepad.exe', 'winword.exe', 'excel.exe']),
            "ProcessId": random.randint(1000, 9999),
            "ProcessCommandLine": f"{random.choice(['explorer.exe', 'notepad.exe'])} \"{details['file_name']}\"",
            "InitiatingProcessFileName": "explorer.exe",
            "InitiatingProcessId": random.randint(1000, 9999),
            "RequestSource": "User",
            "RequestSourceInfo": json.dumps({"RequestSource": "User"}),
            "ReportId": random.randint(10000, 99999),
            "AdditionalFields": json.dumps({
                "FileOriginUrl": "" if not session.get('is_attack') else "http://suspicious-domain.com/payload.exe",
                "FileOriginIP": "" if not session.get('is_attack') else f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                "ZoneIdentifier": "3" if session.get('is_attack') else "2"  # Internet zone vs Trusted zone
            })
        }
    
    def format_network_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        return {
            "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "DeviceId": f"device-{hashlib.md5(machine['name'].encode()).hexdigest()[:16]}",
            "DeviceName": machine['name'].upper(),
            "ActionType": "NetworkConnectionEvents",
            "RemoteIP": details['remote_ip'],
            "RemotePort": details['remote_port'],
            "LocalIP": f"192.168.{random.randint(1,10)}.{random.randint(1,255)}",
            "LocalPort": random.randint(49152, 65535),  # Ephemeral port range
            "Protocol": details['protocol'],
            "InitiatingProcessFileName": random.choice(['chrome.exe', 'firefox.exe', 'outlook.exe', 'teams.exe']),
            "InitiatingProcessId": random.randint(1000, 9999),
            "InitiatingProcessCommandLine": "\"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\"",
            "InitiatingProcessAccountName": user['name'].split('.')[0],
            "InitiatingProcessAccountDomain": user['organization'].upper(),
            "ReportId": random.randint(10000, 99999),
            "AdditionalFields": json.dumps({
                "IsConnectionSuccess": "true" if not session.get('is_attack') or random.random() > 0.3 else "false",
                "ConnectTime": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "ThreatIntel": {
                    "IndicatorValue": details['remote_ip'],
                    "ThreatType": "Malicious" if session.get('is_attack') and session.get('attack_obvious', 0) > 0.5 else "Unknown",
                    "Confidence": random.randint(70, 95) if session.get('is_attack') else random.randint(1, 30)
                }
            })
        }
    
    def _generate_realistic_path(self, filename: str, user: Dict) -> str:
        """Generate realistic Windows file paths"""
        if any(ext in filename for ext in ['.exe', '.dll', '.sys']):
            return random.choice([
                "C:\\Windows\\System32\\",
                "C:\\Program Files\\",
                f"C:\\Users\\{user['name'].split('.')[0]}\\AppData\\Local\\Temp\\",
                "C:\\Program Files (x86)\\Microsoft Office\\Office16\\"
            ])
        elif any(ext in filename for ext in ['.docx', '.xlsx', '.pdf']):
            return f"C:\\Users\\{user['name'].split('.')[0]}\\Documents\\"
        else:
            return f"C:\\Users\\{user['name'].split('.')[0]}\\Desktop\\"
    
    def get_provider_name(self) -> str:
        return "Microsoft Defender for Endpoint"

class CrowdStrikeProvider(LogFormatProvider):
    """CrowdStrike Falcon log format"""
    
    def __init__(self):
        self.cs_event_types = ['DetectionSummaryEvent', 'ProcessRollup2', 'NetworkConnectIP4', 'DnsRequest']
        self.severity_levels = ['Critical', 'High', 'Medium', 'Low', 'Informational']
    
    def format_process_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        return {
            "metadata": {
                "customerIDString": f"cid-{random.randint(100000000000, 999999999999)}",
                "offset": random.randint(10000, 99999),
                "eventType": "ProcessRollup2",
                "eventCreationTime": int(timestamp.timestamp() * 1000),
                "version": "1.0"
            },
            "event": {
                "ProcessStartTime": int(timestamp.timestamp() * 1000),
                "ProcessEndTime": int((timestamp + datetime.timedelta(minutes=random.randint(1, 60))).timestamp() * 1000),
                "ProcessId": details['process_id'],
                "ParentProcessId": random.randint(1000, 9999),
                "ComputerName": machine['name'].upper(),
                "UserName": user['name'],
                "UserSid": f"S-1-5-21-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-{random.randint(1000, 9999)}",
                "ImageFileName": f"\\Device\\HarddiskVolume2\\Windows\\System32\\{details['file_name']}",
                "CommandLine": details['command_line'],
                "RawProcessId": f"{details['process_id']}:{random.randint(100000000000, 999999999999)}",
                "SHA256HashData": hashlib.sha256(details['file_name'].encode()).hexdigest(),
                "MD5HashData": hashlib.md5(details['file_name'].encode()).hexdigest(),
                "ProcessGroupId": f"{random.randint(100000000000, 999999999999)}",
                "ParentAuthenticationId": f"{random.randint(100000, 999999)}",
                "Tags": ["ARCHIVE_FILE_WRITTEN", "SCRIPT_INCLUDE_FILE"] if session.get('is_attack') else ["NORMAL_OPERATION"],
                "ContextTimeStamp": int(timestamp.timestamp() * 1000),
                "Entitlements": "15",
                "SourceProcessId": random.randint(1000, 9999),
                "SourceThreadId": random.randint(1000, 9999),
                "TargetProcessId": details['process_id'],
                "ConfigBuild": f"1007.4.0016304.11-{random.choice(['main', 'hotfix', 'release'])}.{random.randint(200000, 300000)}",
                "ConfigStateHash": f"{random.randint(100000000, 999999999)}",
                "TreeId": f"{random.randint(100000000000, 999999999999)}",
                "TreeId_decimal": str(random.randint(100000000000, 999999999999))
            }
        }
    
    def format_file_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        action_mapping = {
            'FileCreated': 'written',
            'FileModified': 'written', 
            'FileDeleted': 'deleted',
            'FileAccessed': 'read'
        }
        
        return {
            "metadata": {
                "customerIDString": f"cid-{random.randint(100000000000, 999999999999)}",
                "offset": random.randint(10000, 99999),
                "eventType": "FileWritten" if details['action'] in ['FileCreated', 'FileModified'] else "FileRead",
                "eventCreationTime": int(timestamp.timestamp() * 1000),
                "version": "1.0"
            },
            "event": {
                "FilePath": f"\\Device\\HarddiskVolume2\\Users\\{user['name'].split('.')[0]}\\Documents\\{details['file_name']}",
                "FileName": details['file_name'],
                "Size": random.randint(1024, 10485760),
                "MD5HashData": hashlib.md5(details['file_name'].encode()).hexdigest(),
                "SHA256HashData": details['hash'],
                "ProcessId": random.randint(1000, 9999),
                "ProcessName": random.choice(['notepad.exe', 'winword.exe', 'excel.exe', 'powershell.exe']),
                "ComputerName": machine['name'].upper(),
                "UserName": user['name'],
                "ContextTimeStamp": int(timestamp.timestamp() * 1000),
                "EventType": action_mapping.get(details['action'], 'unknown'),
                "TreeId": f"{random.randint(100000000000, 999999999999)}",
                "RawProcessId": f"{random.randint(1000, 9999)}:{random.randint(100000000000, 999999999999)}",
                "Tags": [],
                "ConfigBuild": f"1007.4.0016304.11-main.{random.randint(200000, 300000)}",
                "ConfigStateHash": f"{random.randint(100000000, 999999999999)}",
                "Entitlements": "15",
                "MachineOSBuild": "19041",
                "MachineOSVersion": "Windows 10",
                "PatternDisposition": random.choice([0, 2048, 4096]) if session.get('is_attack') else 0,
                "ExecutablesWritten": [{
                    "SHA256HashData": details['hash'],
                    "FilePath": f"\\Device\\HarddiskVolume2\\Users\\{user['name'].split('.')[0]}\\Documents\\{details['file_name']}",
                    "Timestamp": int(timestamp.timestamp() * 1000)
                }] if details['file_name'].endswith('.exe') else []
            }
        }
    
    def format_network_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        user = session['user']
        machine = session['machine']
        
        return {
            "metadata": {
                "customerIDString": f"cid-{random.randint(100000000000, 999999999999)}",
                "offset": random.randint(10000, 99999),
                "eventType": "NetworkConnectIP4",
                "eventCreationTime": int(timestamp.timestamp() * 1000),
                "version": "1.0"
            },
            "event": {
                "LocalAddressIP4": f"192.168.{random.randint(1,10)}.{random.randint(1,255)}",
                "LocalPort": random.randint(49152, 65535),
                "RemoteAddressIP4": details['remote_ip'],
                "RemotePort": details['remote_port'],
                "Protocol": details['protocol'],
                "ConnectionDirection": random.choice(["outbound", "inbound"]),
                "ProcessId": random.randint(1000, 9999),
                "ImageFileName": f"\\Device\\HarddiskVolume2\\Windows\\System32\\{random.choice(['chrome.exe', 'firefox.exe', 'outlook.exe'])}",
                "ComputerName": machine['name'].upper(),
                "UserName": user['name'],
                "ContextTimeStamp": int(timestamp.timestamp() * 1000),
                "ConfigBuild": f"1007.4.0016304.11-main.{random.randint(200000, 300000)}",
                "ConfigStateHash": f"{random.randint(100000000, 999999999999)}",
                "Entitlements": "15",
                "TreeId": f"{random.randint(100000000000, 999999999999)}",
                "RawProcessId": f"{random.randint(1000, 9999)}:{random.randint(100000000000, 999999999999)}",
                "Tags": ["NETWORK_CONNECT_OUTBOUND"] if details['remote_port'] in [80, 443] else ["NETWORK_CONNECT_SUSPICIOUS"] if session.get('is_attack') else [],
                "ConnectionFlags": random.choice([0, 1, 2]),
                "NetworkProfile": random.choice(["Domain", "Public", "Private"]),
                "ICMPType": "",
                "ICMPCode": "",
                "ConnectionEndTime": int((timestamp + datetime.timedelta(seconds=random.randint(1, 300))).timestamp() * 1000),
                "PatternDisposition": 0 if not session.get('is_attack') else random.choice([2048, 4096, 8192]),
                "ThreatIntel": {
                    "IndicatorMatchCount": random.randint(1, 5) if session.get('is_attack') else 0,
                    "ReputationScore": random.randint(1, 40) if session.get('is_attack') else random.randint(80, 100)
                }
            }
        }
    
    def get_provider_name(self) -> str:
        return "CrowdStrike Falcon"

class CloudflareProvider(LogFormatProvider):
    """Cloudflare log format (HTTP/DNS/Security events)"""
    
    def __init__(self):
        self.cf_actions = ['allow', 'block', 'challenge', 'jschallenge', 'log', 'bypass']
        self.cf_sources = ['firewall', 'ratelimit', 'securitylevel', 'bic', 'hot', 'l7ddos', 'validation']
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'curl/7.68.0',
            'python-requests/2.25.1',
            'PostmanRuntime/7.28.0'
        ]
    
    def format_process_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        # For Cloudflare, process events don't directly map, so we'll create HTTP requests that indicate process behavior
        user = session['user']
        machine = session['machine']
        
        # Simulate web application calls that might indicate process execution
        endpoint = random.choice(['/api/execute', '/admin/commands', '/system/process', '/tools/run'])
        if session.get('is_attack'):
            endpoint = random.choice(['/shell.php', '/cmd.exe', '/admin/exec', '/../../../etc/passwd'])
        
        return {
            "ClientIP": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            "ClientRequestHost": f"{user['organization'].replace('_', '-')}.example.com",
            "ClientRequestMethod": "POST" if session.get('is_attack') else random.choice(['GET', 'POST']),
            "ClientRequestURI": endpoint,
            "ClientRequestUserAgent": random.choice(self.user_agents) if not session.get('is_attack') else "curl/7.68.0",
            "EdgeEndTimestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EdgeRequestHost": f"{user['organization'].replace('_', '-')}.example.com",
            "EdgeResponseBytes": random.randint(500, 5000),
            "EdgeResponseStatus": 200 if not session.get('is_attack') else random.choice([403, 404, 500]),
            "EdgeStartTimestamp": (timestamp - datetime.timedelta(milliseconds=random.randint(10, 500))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "FirewallMatchesActions": ["block"] if session.get('is_attack') and session.get('attack_obvious', 0) > 0.6 else ["allow"],
            "FirewallMatchesRuleIDs": [f"rule_{random.randint(100000, 999999)}"] if session.get('is_attack') else [],
            "FirewallMatchesSources": ["firewall"] if session.get('is_attack') else [],
            "OriginIP": f"10.0.{random.randint(1,10)}.{random.randint(1,255)}",
            "OriginResponseStatus": 200,
            "OriginResponseTime": random.randint(50, 500),
            "RayID": f"{random.randint(100000000000000000, 999999999999999999):016x}",
            "SecurityLevel": "medium",
            "WAFAction": "block" if session.get('is_attack') and session.get('attack_obvious', 0) > 0.5 else "allow",
            "WAFFlags": "0",
            "WAFMatchedVar": "args:cmd" if session.get('is_attack') else "",
            "WAFProfile": "normal",
            "WAFRuleID": f"100{random.randint(100, 999)}" if session.get('is_attack') else "",
            "WAFRuleMessage": "Command injection attack detected" if session.get('is_attack') else "",
            "WorkerCPUTime": random.randint(1, 50),
            "WorkerStatus": "ok",
            "WorkerSubrequest": False,
            "ZoneID": random.randint(100000000000000000, 999999999999999999)
        }
    
    def format_file_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        # File events as HTTP requests to file endpoints
        user = session['user']
        machine = session['machine']
        
        action_to_method = {
            'FileCreated': 'PUT',
            'FileModified': 'PUT',
            'FileDeleted': 'DELETE',
            'FileAccessed': 'GET'
        }
        
        endpoint = f"/files/{details['file_name']}"
        if session.get('is_attack'):
            endpoint = random.choice([
                f"/uploads/../../../{details['file_name']}",
                f"/files/../../../../etc/passwd",
                f"/download.php?file=../../../{details['file_name']}"
            ])
        
        return {
            "ClientIP": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            "ClientRequestHost": f"{user['organization'].replace('_', '-')}.example.com",
            "ClientRequestMethod": action_to_method.get(details['action'], 'GET'),
            "ClientRequestURI": endpoint,
            "ClientRequestUserAgent": random.choice(self.user_agents),
            "EdgeEndTimestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EdgeRequestHost": f"{user['organization'].replace('_', '-')}.example.com",
            "EdgeResponseBytes": random.randint(len(details['file_name']) * 100, len(details['file_name']) * 1000),
            "EdgeResponseStatus": 200 if not session.get('is_attack') else random.choice([403, 404, 500]),
            "EdgeStartTimestamp": (timestamp - datetime.timedelta(milliseconds=random.randint(10, 500))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "FirewallMatchesActions": ["block"] if session.get('is_attack') and "../../" in endpoint else ["allow"],
            "FirewallMatchesRuleIDs": [f"rule_{random.randint(100000, 999999)}"] if session.get('is_attack') else [],
            "FirewallMatchesSources": ["firewall"] if session.get('is_attack') else [],
            "OriginIP": f"10.0.{random.randint(1,10)}.{random.randint(1,255)}",
            "OriginResponseStatus": 200 if not session.get('is_attack') else 403,
            "OriginResponseTime": random.randint(50, 2000),
            "RayID": f"{random.randint(100000000000000000, 999999999999999999):016x}",
            "SecurityLevel": "high" if session.get('is_attack') else "medium",
            "WAFAction": "block" if session.get('is_attack') and "../../" in endpoint else "allow",
            "WAFFlags": "0",
            "WAFMatchedVar": "args:file" if session.get('is_attack') and "../../" in endpoint else "",
            "WAFProfile": "normal",
            "WAFRuleID": f"100{random.randint(200, 299)}" if session.get('is_attack') else "",
            "WAFRuleMessage": "Directory traversal attack detected" if session.get('is_attack') and "../../" in endpoint else "",
            "WorkerCPUTime": random.randint(1, 100),
            "WorkerStatus": "ok",
            "WorkerSubrequest": False,
            "ZoneID": random.randint(100000000000000000, 999999999999999999)
        }
    
    def format_network_event(self, session: Dict, timestamp: datetime.datetime, details: Dict) -> Dict:
        # Network events as DNS requests or HTTP connections
        user = session['user']
        machine = session['machine']
        
        # Decide between DNS or HTTP event
        if random.choice([True, False]):
            # DNS Event
            query_name = f"api.{user['organization'].replace('_', '-')}.com"
            if session.get('is_attack'):
                query_name = random.choice([
                    f"malware-c2-{random.randint(1000,9999)}.tk",
                    f"phishing-{random.randint(100,999)}.ru",
                    f"suspicious.{random.randint(1000,9999)}.info"
                ])
            
            return {
                "ColoCode": random.choice(["SJC", "LAX", "DFW", "ATL", "JFK", "LHR"]),
                "EDNSSubnet": f"{random.randint(1,255)}.{random.randint(1,255)}.0.0/16",
                "EDNSSubnetLength": 16,
                "QueryName": query_name,
                "QueryType": random.choice([1, 28, 5, 15]),  # A, AAAA, CNAME, MX
                "ResponseCached": random.choice([True, False]),
                "ResponseCode": 0 if not session.get('is_attack') else random.choice([0, 3, 2]),  # NOERROR, NXDOMAIN, SERVFAIL
                "SourceIP": details['remote_ip'],
                "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "QueryTypeName": random.choice(["A", "AAAA", "CNAME", "MX"]),
                "ResponseCodeName": "NOERROR" if not session.get('is_attack') else random.choice(["NOERROR", "NXDOMAIN", "SERVFAIL"]),
                "ClientSubnet": f"{random.randint(1,255)}.{random.randint(1,255)}.0.0",
                "ClientSubnetLength": 16,
                "ClientCountry": random.choice(["US", "GB", "DE", "FR", "CA"]),
                "ClientASN": random.randint(1000, 50000),
                "ClientASNDescription": f"AS{random.randint(1000, 50000)} ISP Corp",
                "SecurityCategory": ["malware", "phishing"] if session.get('is_attack') else [],
                "SecurityCategoryName": ["Malware", "Phishing"] if session.get('is_attack') else []
            }
        else:
            # HTTP Connection Event (similar to process_event but network focused)
            return self.format_process_event(session, timestamp, details)
    
    def get_provider_name(self) -> str:
        return "Cloudflare"

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

class LogFormatRegistry:
    """Registry of available log format providers"""
    
    _providers = {
        'defender': MicrosoftDefenderProvider,
        'crowdstrike': CrowdStrikeProvider,
        'cloudflare': CloudflareProvider
    }
    
    @classmethod
    def get_provider(cls, format_name: str) -> LogFormatProvider:
        """Get a provider instance by name"""
        if format_name not in cls._providers:
            raise ValueError(f"Unknown log format: {format_name}. Available formats: {list(cls._providers.keys())}")
        return cls._providers[format_name]()
    
    @classmethod
    def list_formats(cls) -> List[str]:
        """List all available log formats"""
        return list(cls._providers.keys())

class EnhancedSyntheticDataGenerator:
    """Enhanced generator with pluggable log formats"""
    
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
        self.log_provider = LogFormatRegistry.get_provider(config.log_format)
        
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
    
    def generate_logs(self) -> List[Dict]:
        """Generate logs in the specified format"""
        print(f"Generating {self.config.total_alerts} logs using {self.log_provider.get_provider_name()} format...")
        
        # First, distribute alerts across sessions
        session_sizes = self._generate_session_distribution()
        
        # Create session-user-machine mappings
        sessions = self._create_sessions(session_sizes)
        
        # Inject attack scenarios
        attack_sessions = self._inject_attack_scenarios(sessions)
        
        # Generate individual logs for each session
        all_logs = []
        for session in sessions:
            session_logs = self._generate_session_logs(session)
            all_logs.extend(session_logs)
        
        # Shuffle to make timing more realistic
        random.shuffle(all_logs)
        all_logs.sort(key=lambda x: self._get_timestamp_from_log(x))
        
        print(f"Generated {len(all_logs)} total logs in {len(sessions)} sessions")
        print(f"Attack scenarios embedded: {len(attack_sessions)}")
        
        return all_logs
    
    def _get_timestamp_from_log(self, log: Dict) -> str:
        """Extract timestamp from log regardless of format"""
        # Different providers use different timestamp fields
        timestamp_fields = ['Timestamp', 'EdgeEndTimestamp', 'timestamp']
        for field in timestamp_fields:
            if field in log:
                return log[field]
            # Check nested metadata for CrowdStrike
            if 'metadata' in log and 'eventCreationTime' in log['metadata']:
                return str(log['metadata']['eventCreationTime'])
        return datetime.datetime.now().isoformat()
    
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
    
    def _generate_session_logs(self, session: Dict) -> List[Dict]:
        """Generate individual logs for a session using the configured provider"""
        logs = []
        user = session['user']
        machine = session['machine']
        org_profile = OrganizationProfiles.PROFILES[user['organization']]
        
        session_start = session['start_time']
        session_duration = session['duration_minutes']
        
        for i in range(session['alert_count']):
            # Generate timestamp within session
            minutes_offset = (i / session['alert_count']) * session_duration + random.uniform(-5, 5)
            timestamp = session_start + datetime.timedelta(minutes=max(0, minutes_offset))
            
            # Generate log based on whether it's part of an attack
            if session['is_attack']:
                log = self._generate_attack_log(session, timestamp, i, org_profile)
            else:
                log = self._generate_normal_log(session, timestamp, i, org_profile)
            
            logs.append(log)
        
        return logs
    
    def _generate_normal_log(self, session: Dict, timestamp: datetime.datetime, index: int, org_profile: Dict) -> Dict:
        """Generate normal business activity log"""
        event_types = ['Process', 'File', 'Network']
        event_type = random.choice(event_types)
        
        if event_type == 'Process':
            details = {
                'file_name': random.choice(org_profile['developer_tools']),
                'command_line': self._generate_normal_command_line(org_profile),
                'process_id': random.randint(1000, 9999)
            }
            return self.log_provider.format_process_event(session, timestamp, details)
        elif event_type == 'File':
            details = {
                'file_name': f"document_{random.randint(1, 1000)}{random.choice(org_profile['common_files'])}",
                'action': random.choice(['FileCreated', 'FileModified', 'FileDeleted', 'FileAccessed']),
                'hash': ''.join(random.choices('abcdef0123456789', k=64))
            }
            return self.log_provider.format_file_event(session, timestamp, details)
        else:  # Network
            details = {
                'remote_ip': self._generate_normal_ip(org_profile),
                'remote_port': random.choice([80, 443, 22, 3389, 53]),
                'protocol': random.choice(['TCP', 'UDP'])
            }
            return self.log_provider.format_network_event(session, timestamp, details)
    
    def _generate_attack_log(self, session: Dict, timestamp: datetime.datetime, index: int, org_profile: Dict) -> Dict:
        """Generate suspicious log as part of attack scenario"""
        # Make some logs more obviously malicious based on attack_obvious score
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
            return self.log_provider.format_process_event(session, timestamp, details)
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
            return self.log_provider.format_file_event(session, timestamp, details)
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
            return self.log_provider.format_network_event(session, timestamp, details)
    
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

def generate_multi_format_logs(alert_count: int, session_count: int, attack_count: int = 3):
    """Generate logs in all supported formats"""
    results = {}
    
    for format_name in LogFormatRegistry.list_formats():
        print(f"\n🔧 Generating {format_name} format logs...")
        
        config = SyntheticConfig(
            total_alerts=alert_count,
            target_sessions=session_count,
            attack_campaigns=attack_count,
            log_format=format_name
        )
        
        generator = EnhancedSyntheticDataGenerator(config)
        logs = generator.generate_logs()
        
        # Save to format-specific file
        output_file = f"synthetic_{format_name}_logs.json"
        with open(output_file, 'w') as f:
            json.dump(logs[:100], f, indent=2)  # Save first 100 for inspection
        
        results[format_name] = {
            'logs': logs,
            'count': len(logs),
            'file': output_file,
            'provider': generator.log_provider.get_provider_name()
        }
        
        print(f"   ✅ Generated {len(logs)} {format_name} logs → {output_file}")
    
    return results

# Utility functions for session analysis and HMM training

def extract_session_id(log: Dict) -> str:
    """Extract session ID from log regardless of format"""
    # Microsoft Defender
    if 'DeviceName' in log and 'AccountName' in log:
        return f"{log['AccountName']}_{log['DeviceName']}".lower()
    
    # CrowdStrike
    elif 'event' in log and 'ComputerName' in log['event'] and 'UserName' in log['event']:
        return f"{log['event']['UserName']}_{log['event']['ComputerName']}".lower().replace('.', '_')
    
    # Cloudflare
    elif 'ClientIP' in log and 'ClientRequestHost' in log:
        return f"cloudflare_{log['ClientIP']}_{log['ClientRequestHost']}".lower().replace('.', '_')
    
    # Fallback
    return f"unknown_session_{random.randint(1000, 9999)}"

def get_timestamp_from_log(log: Dict) -> str:
    """Extract timestamp from log regardless of format"""
    timestamp_fields = ['Timestamp', 'EdgeEndTimestamp', 'timestamp']
    for field in timestamp_fields:
        if field in log:
            return log[field]
        # Check nested metadata for CrowdStrike
        if 'metadata' in log and 'eventCreationTime' in log['metadata']:
            return str(log['metadata']['eventCreationTime'])
    return datetime.datetime.now().isoformat()

def extract_event_type(log: Dict) -> Optional[str]:
    """Extract event type from log regardless of format"""
    # Microsoft Defender
    if 'ActionType' in log:
        return log['ActionType']
    
    # CrowdStrike
    elif 'metadata' in log and 'eventType' in log['metadata']:
        return log['metadata']['eventType']
    
    # Cloudflare
    elif 'ClientRequestMethod' in log:
        return f"HTTP_{log['ClientRequestMethod']}"
    elif 'QueryType' in log:
        return f"DNS_{log.get('QueryTypeName', 'UNKNOWN')}"
    
    return None

def is_attack_indicator(log: Dict) -> bool:
    """Check if a log contains attack indicators"""
    attack_indicators = [
        # Process indicators
        'powershell.exe -ExecutionPolicy Bypass',
        'cmd.exe /c "whoami',
        'net user administrator',
        'wmic process call create',
        
        # File indicators
        'passwords.txt',
        'credentials.db',
        'keylogger.exe',
        
        # Network indicators
        'malware-c2',
        'phishing',
        'suspicious',
        
        # Cloudflare WAF blocks
        'Command injection attack detected',
        'Directory traversal attack detected'
    ]
    
    log_str = json.dumps(log).lower()
    return any(indicator.lower() in log_str for indicator in attack_indicators)

def group_logs_into_sessions(logs: List[Dict]) -> Dict[str, Dict]:
    """Group logs into sessions for analysis"""
    sessions = {}
    
    for log in logs:
        # Extract session ID based on log format
        session_id = extract_session_id(log)
        
        if session_id not in sessions:
            sessions[session_id] = {
                'session_id': session_id,
                'logs': [],
                'first_log': get_timestamp_from_log(log),
                'last_log': get_timestamp_from_log(log),
                'event_types': set(),
                'attack_indicators': []
            }
        
        # Add log to session
        sessions[session_id]['logs'].append(log)
        sessions[session_id]['last_log'] = get_timestamp_from_log(log)
        
        # Extract event type based on log format
        event_type = extract_event_type(log)
        if event_type:
            sessions[session_id]['event_types'].add(event_type)
        
        # Check for attack indicators
        if is_attack_indicator(log):
            sessions[session_id]['attack_indicators'].append(log)
    
    # Convert sets to lists and add metadata
    for session in sessions.values():
        session['event_types'] = list(session['event_types'])
        session['log_count'] = len(session['logs'])
        session['is_attack_session'] = len(session['attack_indicators']) > 0
        session['attack_score'] = len(session['attack_indicators']) / session['log_count']
    
    return sessions

def export_for_hmm_training(sessions: Dict[str, Dict]) -> Dict:
    """Export session data for HMM training"""
    training_data = {
        'sessions': [],
        'feature_vectors': [],
        'labels': [],
        'metadata': {
            'total_sessions': len(sessions),
            'attack_sessions': sum(1 for s in sessions.values() if s['is_attack_session']),
            'generation_time': datetime.datetime.now().isoformat(),
            'feature_descriptions': {
                'log_count': 'Total number of logs in session',
                'unique_event_types': 'Number of unique event types',
                'attack_indicators': 'Number of detected attack indicators',
                'session_duration': 'Duration of session in minutes',
                'time_variance': 'Variance in timing between events',
                'event_type_entropy': 'Shannon entropy of event type distribution'
            }
        }
    }
    
    for session_id, session in sessions.items():
        # Extract features for HMM training
        features = extract_session_features(session)
        
        # Create training record
        training_record = {
            'session_id': session_id,
            'features': features,
            'label': 1 if session['is_attack_session'] else 0,
            'attack_score': session['attack_score'],
            'event_sequence': [extract_event_type(log) for log in session['logs']],
            'timestamp_sequence': [get_timestamp_from_log(log) for log in session['logs']]
        }
        
        training_data['sessions'].append(training_record)
        training_data['feature_vectors'].append(features)
        training_data['labels'].append(training_record['label'])
    
    # Save training data
    with open('hmm_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    return training_data

def extract_session_features(session: Dict) -> List[float]:
    """Extract numerical features from session for ML training"""
    import math
    from collections import Counter
    
    logs = session['logs']
    
    # Basic count features
    log_count = len(logs)
    unique_event_types = len(session['event_types'])
    attack_indicators = len(session['attack_indicators'])
    
    # Temporal features
    timestamps = [get_timestamp_from_log(log) for log in logs]
    session_duration = calculate_session_duration(timestamps)
    time_variance = calculate_time_variance(timestamps)
    
    # Event distribution features
    event_types = [extract_event_type(log) for log in logs if extract_event_type(log)]
    event_entropy = calculate_entropy(event_types) if event_types else 0
    
    # Behavioral features
    process_ratio = sum(1 for log in logs if 'process' in str(extract_event_type(log)).lower()) / max(1, log_count)
    network_ratio = sum(1 for log in logs if 'network' in str(extract_event_type(log)).lower()) / max(1, log_count)
    file_ratio = sum(1 for log in logs if 'file' in str(extract_event_type(log)).lower()) / max(1, log_count)
    
    # Risk indicators
    suspicious_processes = count_suspicious_processes(logs)
    unusual_network_activity = count_unusual_network_activity(logs)
    
    return [
        log_count,
        unique_event_types,
        attack_indicators,
        session_duration,
        time_variance,
        event_entropy,
        process_ratio,
        network_ratio,
        file_ratio,
        suspicious_processes,
        unusual_network_activity,
        session['attack_score']
    ]

def calculate_session_duration(timestamps: List[str]) -> float:
    """Calculate session duration in minutes"""
    if len(timestamps) < 2:
        return 0.0
    
    try:
        # Parse timestamps (handle different formats)
        parsed_times = []
        for ts in timestamps:
            if isinstance(ts, str):
                # Try different timestamp formats
                for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        parsed_times.append(datetime.datetime.strptime(ts, fmt))
                        break
                    except ValueError:
                        continue
                else:
                    # Try parsing as Unix timestamp
                    try:
                        parsed_times.append(datetime.datetime.fromtimestamp(int(ts) / 1000))
                    except (ValueError, OSError):
                        continue
        
        if len(parsed_times) >= 2:
            duration = (max(parsed_times) - min(parsed_times)).total_seconds() / 60
            return duration
    except Exception:
        pass
    
    return 0.0

def calculate_time_variance(timestamps: List[str]) -> float:
    """Calculate variance in timing between events"""
    if len(timestamps) < 3:
        return 0.0
    
    try:
        parsed_times = []
        for ts in timestamps:
            if isinstance(ts, str):
                for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ']:
                    try:
                        parsed_times.append(datetime.datetime.strptime(ts, fmt))
                        break
                    except ValueError:
                        continue
                else:
                    try:
                        parsed_times.append(datetime.datetime.fromtimestamp(int(ts) / 1000))
                    except (ValueError, OSError):
                        continue
        
        if len(parsed_times) >= 3:
            parsed_times.sort()
            intervals = [(parsed_times[i+1] - parsed_times[i]).total_seconds() 
                        for i in range(len(parsed_times)-1)]
            
            if intervals:
                mean_interval = sum(intervals) / len(intervals)
                variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                return variance
    except Exception:
        pass
    
    return 0.0

def calculate_entropy(items: List[str]) -> float:
    """Calculate Shannon entropy of item distribution"""
    if not items:
        return 0.0
    
    from collections import Counter
    import math
    
    counts = Counter(items)
    total = len(items)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy

def count_suspicious_processes(logs: List[Dict]) -> int:
    """Count suspicious process indicators"""
    suspicious_indicators = [
        'powershell.exe -executionpolicy bypass',
        'cmd.exe /c',
        'wmic process',
        'net user',
        'whoami',
        'encoded'
    ]
    
    count = 0
    for log in logs:
        log_str = json.dumps(log).lower()
        count += sum(1 for indicator in suspicious_indicators if indicator in log_str)
    
    return count

def count_unusual_network_activity(logs: List[Dict]) -> int:
    """Count unusual network activity indicators"""
    unusual_indicators = [
        'malware',
        'phishing',
        'suspicious',
        'c2',
        'command injection',
        'directory traversal'
    ]
    
    count = 0
    for log in logs:
        log_str = json.dumps(log).lower()
        count += sum(1 for indicator in unusual_indicators if indicator in log_str)
    
    return count

def analyze_log_quality(logs: List[Dict]) -> Dict:
    """Analyze the quality and realism of generated logs"""
    analysis = {
        'total_logs': len(logs),
        'unique_sessions': len(set(extract_session_id(log) for log in logs)),
        'event_type_distribution': {},
        'timestamp_coverage': {},
        'attack_indicator_ratio': 0,
        'format_consistency': True,
        'realism_score': 0.0
    }
    
    # Event type distribution
    event_types = [extract_event_type(log) for log in logs if extract_event_type(log)]
    from collections import Counter
    analysis['event_type_distribution'] = dict(Counter(event_types))
    
    # Attack indicator analysis
    attack_logs = sum(1 for log in logs if is_attack_indicator(log))
    analysis['attack_indicator_ratio'] = attack_logs / len(logs)
    
    # Timestamp analysis
    timestamps = [get_timestamp_from_log(log) for log in logs]
    analysis['timestamp_coverage'] = {
        'first_log': min(timestamps),
        'last_log': max(timestamps),
        'span_hours': calculate_session_duration(timestamps) / 60
    }
    
    # Calculate realism score based on various factors
    realism_factors = [
        min(1.0, analysis['unique_sessions'] / 100),  # Session diversity
        min(1.0, len(analysis['event_type_distribution']) / 10),  # Event diversity
        min(1.0, analysis['attack_indicator_ratio'] * 20),  # Appropriate attack ratio
        1.0 if analysis['timestamp_coverage']['span_hours'] > 1 else 0.5  # Time span
    ]
    analysis['realism_score'] = sum(realism_factors) / len(realism_factors)
    
    return analysis

def create_detection_rules(logs: List[Dict]) -> List[Dict]:
    """Create detection rules based on attack patterns in logs"""
    rules = []
    
    # Analyze attack patterns
    attack_logs = [log for log in logs if is_attack_indicator(log)]
    
    # Rule 1: Suspicious PowerShell execution
    powershell_attacks = [log for log in attack_logs if 'powershell' in json.dumps(log).lower()]
    if powershell_attacks:
        rules.append({
            'rule_id': 'SUSP_POWERSHELL_001',
            'name': 'Suspicious PowerShell Execution',
            'description': 'Detects PowerShell execution with bypass policies',
            'severity': 'High',
            'pattern': 'powershell.exe -ExecutionPolicy Bypass',
            'occurrences': len(powershell_attacks),
            'confidence': min(0.95, len(powershell_attacks) / 10)
        })
    
    # Rule 2: Credential harvesting attempts
    cred_attacks = [log for log in attack_logs if any(x in json.dumps(log).lower() 
                                                    for x in ['password', 'credential', 'keylog'])]
    if cred_attacks:
        rules.append({
            'rule_id': 'CRED_HARVEST_001',
            'name': 'Credential Harvesting Activity',
            'description': 'Detects potential credential harvesting activities',
            'severity': 'Critical',
            'pattern': 'passwords.txt|credentials.db|keylogger.exe',
            'occurrences': len(cred_attacks),
            'confidence': min(0.9, len(cred_attacks) / 5)
        })
    
    # Rule 3: Lateral movement indicators
    lateral_attacks = [log for log in attack_logs if any(x in json.dumps(log).lower() 
                                                       for x in ['net user', 'wmic', 'admin'])]
    if lateral_attacks:
        rules.append({
            'rule_id': 'LATERAL_MOVE_001',
            'name': 'Lateral Movement Activity',
            'description': 'Detects potential lateral movement techniques',
            'severity': 'High',
            'pattern': 'net user administrator|wmic process call create',
            'occurrences': len(lateral_attacks),
            'confidence': min(0.85, len(lateral_attacks) / 8)
        })
    
    # Rule 4: Web application attacks (for Cloudflare logs)
    web_attacks = [log for log in attack_logs if any(x in json.dumps(log).lower() 
                                                   for x in ['injection', 'traversal', 'shell.php'])]
    if web_attacks:
        rules.append({
            'rule_id': 'WEB_ATTACK_001',
            'name': 'Web Application Attack',
            'description': 'Detects web application attack attempts',
            'severity': 'High',
            'pattern': 'command injection|directory traversal|shell.php',
            'occurrences': len(web_attacks),
            'confidence': min(0.9, len(web_attacks) / 6)
        })
    
    return rules

def main():
    """Enhanced main function with comprehensive analysis"""
    print("🔬 Enhanced Pluggable Security Log Generator")
    print("=" * 60)
    
    # Show available formats
    available_formats = LogFormatRegistry.list_formats()
    print(f"📋 Available log formats: {', '.join(available_formats)}")
    
    # Generate sample data for each format
    print(f"\n🎯 Generating sample logs for all formats...")
    
    results = generate_multi_format_logs(
        alert_count=1000,
        session_count=50,
        attack_count=3
    )
    
    print(f"\n📊 GENERATION SUMMARY:")
    for format_name, result in results.items():
        print(f"   {format_name}: {result['count']} logs → {result['file']}")
        print(f"      Provider: {result['provider']}")
    
    # Analyze log quality for each format
    print(f"\n🔍 QUALITY ANALYSIS:")
    for format_name, result in results.items():
        analysis = analyze_log_quality(result['logs'])
        print(f"   {format_name}:")
        print(f"      Sessions: {analysis['unique_sessions']}")
        print(f"      Event Types: {len(analysis['event_type_distribution'])}")
        print(f"      Attack Ratio: {analysis['attack_indicator_ratio']:.2%}")
        print(f"      Realism Score: {analysis['realism_score']:.2f}/1.0")
    
    # Generate HMM training data using best format
    print(f"\n🤖 Generating HMM training data...")
    
    config = SyntheticConfig(
        total_alerts=5000,
        target_sessions=200,
        attack_campaigns=5,
        log_format='defender'  # Use Defender as default for training
    )
    
    generator = EnhancedSyntheticDataGenerator(config)
    logs = generator.generate_logs()
    
    # Group logs into sessions for HMM analysis
    sessions = group_logs_into_sessions(logs)
    
    # Export for HMM training
    training_data = export_for_hmm_training(sessions)
    
    print(f"   ✅ HMM training data → hmm_training_data.json")
    print(f"   📈 {len(sessions)} sessions ({training_data['metadata']['attack_sessions']} attack sessions)")
    
    # Generate detection rules
    print(f"\n🚨 Generating detection rules...")
    rules = create_detection_rules(logs)
    
    with open('detection_rules.json', 'w') as f:
        json.dump(rules, f, indent=2)
    
    print(f"   ✅ Generated {len(rules)} detection rules → detection_rules.json")
    for rule in rules:
        print(f"      {rule['rule_id']}: {rule['name']} (Confidence: {rule['confidence']:.2f})")
    
    print(f"\n🎉 Multi-format log generation complete!")
    print(f"\n📚 Usage examples:")
    print(f"  # Generate specific format")
    print(f"  config = SyntheticConfig(log_format='crowdstrike')")
    print(f"  generator = EnhancedSyntheticDataGenerator(config)")
    print(f"  logs = generator.generate_logs()")
    print(f"")
    print(f"  # Analyze existing logs")
    print(f"  sessions = group_logs_into_sessions(logs)")
    print(f"  analysis = analyze_log_quality(logs)")
    print(f"")
    print(f"  # Export for ML training")
    print(f"  training_data = export_for_hmm_training(sessions)")
    print(f"")
    print(f"  # Generate detection rules")
    print(f"  rules = create_detection_rules(logs)")

if __name__ == "__main__":
    main()