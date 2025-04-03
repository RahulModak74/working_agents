#!/usr/bin/env python3

"""
Security Detection Adapter Module
This module provides an adapter layer between the workflow JSON tool format and
the actual tool implementation in top_10_std_attacks.py.
"""

import os
import sys
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

# Ensure both directories are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import the detection module
try:
    from top_10_std_attacks import SecurityDetection
    real_tools_available = True
except ImportError:
    print("Warning: Real security detection module not available. Using mock implementations.")
    real_tools_available = False

# Mock data setup for when real module isn't available
def create_mock_data():
    """Creates mock data for testing when the real module isn't available"""
    return pd.DataFrame({
        'hostname': ['win-host-1', 'linux-srv-2'],
        'os_type': ['Windows', 'Linux'],
        'pid': [1234, 5678],
        'name': ['powershell.exe', 'bash'],
        'exe_path': ['C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe', '/bin/bash'],
        'cmdline': ['powershell.exe -EncodedCommand SQBuAHYAbwBrAGUALQBXAG0AaQBNAGUAdABoAG8AZAA=', 'cat /etc/shadow'],
        'user': ['SYSTEM', 'root'],
        'ppid': [900, 1],
        'timestamp': ['2023-11-01 08:25:43', '2023-11-01 09:15:22'],
        'credential_access': [1, 0],
        'remote_mem_operations': [2, 0],
        'rwx_segments_count': [5, 1],
        'file_writes': [120, 5],
        'file_reads': [150, 10],
        'outbound_bytes': [15000000, 1000],
        'conn_count': [6, 1],
        'obfuscated_script': [1, 0],
        'script_execution': [1, 0],
        'registry_keys_modified': [
            ['HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run'], 
            []
        ],
        'sensitive_file_access': [
            [], 
            ['/etc/shadow', '/etc/passwd']
        ],
        'remote_ips': [
            ['192.168.1.100', '45.123.45.67'], 
            ['10.0.0.5']
        ],
        'dns_queries': [
            ['suspicious-domain.com', 'cdn.legitimate.com'], 
            ['update.service.net']
        ],
        'listening_ports': [
            [445, 3389], 
            [22]
        ]
    })

# Mock SecurityDetection class for when real module isn't available
class MockSecurityDetection:
    """Mock implementation of the SecurityDetection class"""
    
    def __init__(self, data_source):
        self.data = data_source
    
    def detect_living_off_the_land(self, limit=100):
        """Mock implementation for living off the land detection"""
        print(f"[MOCK] Detecting living off the land (limit: {limit})")
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user', 'timestamp']]
    
    def detect_credential_dumping(self, limit=100):
        """Mock implementation for credential dumping detection"""
        print(f"[MOCK] Detecting credential dumping (limit: {limit})")
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user', 'timestamp', 'credential_access']]
    
    def detect_process_injection(self, limit=100):
        """Mock implementation for process injection detection"""
        print(f"[MOCK] Detecting process injection (limit: {limit})")
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user', 'timestamp', 'rwx_segments_count', 'remote_mem_operations']]
    
    def detect_unusual_child_processes(self, limit=100):
        """Mock implementation for unusual child processes detection"""
        print(f"[MOCK] Detecting unusual child processes (limit: {limit})")
        # Create an empty DataFrame with expected columns for unusual child processes
        return pd.DataFrame(columns=['hostname', 'parent_pid', 'parent_name', 'parent_path', 'parent_user',
                           'pid', 'name', 'exe_path', 'cmdline', 'timestamp'])
    
    def detect_persistence_mechanisms(self, limit=100):
        """Mock implementation for persistence mechanisms detection"""
        print(f"[MOCK] Detecting persistence mechanisms (limit: {limit})")
        # Add persistence_type column
        self.data['persistence_type'] = 'Registry Persistence'
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user',
                     'timestamp', 'persistence_type']]
    
    def detect_obfuscated_scripts(self, limit=100):
        """Mock implementation for obfuscated scripts detection"""
        print(f"[MOCK] Detecting obfuscated scripts (limit: {limit})")
        # Add script_type column
        self.data['script_type'] = 'PowerShell'
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'user', 'cmdline',
                     'script_type', 'obfuscated_script', 'timestamp']]
    
    def detect_data_exfiltration(self, limit=100):
        """Mock implementation for data exfiltration detection"""
        print(f"[MOCK] Detecting data exfiltration (limit: {limit})")
        # Add connected_ips and dns_lookups columns
        self.data['connected_ips'] = self.data.apply(
            lambda row: ', '.join(row.get('remote_ips', [])) if isinstance(row.get('remote_ips', []), list) else '',
            axis=1
        )
        self.data['dns_lookups'] = self.data.apply(
            lambda row: ', '.join(row.get('dns_queries', [])) if isinstance(row.get('dns_queries', []), list) else '',
            axis=1
        )
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user',
                     'timestamp', 'outbound_bytes', 'conn_count', 'connected_ips', 'dns_lookups']]
    
    def detect_defense_evasion(self, limit=100):
        """Mock implementation for defense evasion detection"""
        print(f"[MOCK] Detecting defense evasion (limit: {limit})")
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user', 'timestamp']]
    
    def detect_lateral_movement(self, limit=100):
        """Mock implementation for lateral movement detection"""
        print(f"[MOCK] Detecting lateral movement (limit: {limit})")
        # Add target_systems column
        self.data['target_systems'] = self.data.apply(
            lambda row: ', '.join(row.get('remote_ips', [])) if isinstance(row.get('remote_ips', []), list) else '',
            axis=1
        )
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user',
                     'timestamp', 'target_systems']]
    
    def detect_ransomware_behavior(self, limit=100):
        """Mock implementation for ransomware behavior detection"""
        print(f"[MOCK] Detecting ransomware behavior (limit: {limit})")
        return self.data[['hostname', 'os_type', 'pid', 'name', 'exe_path', 'cmdline', 'user',
                     'timestamp', 'file_writes', 'file_reads']]

# Adapter functions that bridge between workflow JSON format and actual tool implementation

def load_data_adapter(**kwargs):
    """Adapter for loading security telemetry data"""
    file_path = kwargs.get("file_path")
    
    if not file_path:
        return {"error": "File path not provided for data loading"}
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert timestamp column to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert list strings to actual lists where needed
        list_columns = ['remote_ips', 'dns_queries', 'registry_keys_modified', 
                       'sensitive_file_access', 'listening_ports']
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else [])
        
        return {
            "success": True,
            "message": f"Data loaded successfully. Shape: {df.shape}",
            "row_count": len(df),
            "column_count": len(df.columns),
            "data": df
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return {"error": f"Data loading failed: {str(e)}"}

def detect_living_off_the_land_adapter(**kwargs):
    """Adapter for living off the land detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for living off the land detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_living_off_the_land(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting living off the land: {e}")
            return {"error": f"Living off the land detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_living_off_the_land(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_credential_dumping_adapter(**kwargs):
    """Adapter for credential dumping detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for credential dumping detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_credential_dumping(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting credential dumping: {e}")
            return {"error": f"Credential dumping detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_credential_dumping(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_process_injection_adapter(**kwargs):
    """Adapter for process injection detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for process injection detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_process_injection(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting process injection: {e}")
            return {"error": f"Process injection detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_process_injection(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_unusual_child_processes_adapter(**kwargs):
    """Adapter for unusual child processes detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for unusual child processes detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_unusual_child_processes(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting unusual child processes: {e}")
            return {"error": f"Unusual child processes detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_unusual_child_processes(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_persistence_mechanisms_adapter(**kwargs):
    """Adapter for persistence mechanisms detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for persistence mechanisms detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_persistence_mechanisms(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting persistence mechanisms: {e}")
            return {"error": f"Persistence mechanisms detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_persistence_mechanisms(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_obfuscated_scripts_adapter(**kwargs):
    """Adapter for obfuscated scripts detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for obfuscated scripts detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_obfuscated_scripts(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting obfuscated scripts: {e}")
            return {"error": f"Obfuscated scripts detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_obfuscated_scripts(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_data_exfiltration_adapter(**kwargs):
    """Adapter for data exfiltration detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for data exfiltration detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_data_exfiltration(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting data exfiltration: {e}")
            return {"error": f"Data exfiltration detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_data_exfiltration(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_defense_evasion_adapter(**kwargs):
    """Adapter for defense evasion detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for defense evasion detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_defense_evasion(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting defense evasion: {e}")
            return {"error": f"Defense evasion detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_defense_evasion(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_lateral_movement_adapter(**kwargs):
    """Adapter for lateral movement detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for lateral movement detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_lateral_movement(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting lateral movement: {e}")
            return {"error": f"Lateral movement detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_lateral_movement(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_ransomware_behavior_adapter(**kwargs):
    """Adapter for ransomware behavior detection"""
    df = kwargs.get("data")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if df is None:
        return {"error": "No data provided for ransomware behavior detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            detector = SecurityDetection(df)
            result_df = detector.detect_ransomware_behavior(limit=limit)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting ransomware behavior: {e}")
            return {"error": f"Ransomware behavior detection failed: {str(e)}"}
    else:
        # Use mock implementation
        mock_data = create_mock_data() if isinstance(df, str) else df
        detector = MockSecurityDetection(mock_data)
        mock_result = detector.detect_ransomware_behavior(limit=limit)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def run_all_detections(**kwargs):
    """Run all security detection algorithms on the given data"""
    file_path = kwargs.get("file_path")
    limit = kwargs.get("limit", 100)
    
    # Convert string values to appropriate types
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 100
    
    if not file_path:
        return {"error": "File path not provided for all detections"}
    
    # Load data first
    load_result = load_data_adapter(file_path=file_path)
    
    if "error" in load_result:
        return load_result
    
    df = load_result.get("data")
    
    # Run all detections
    results = {}
    
    # Run all detection functions
    results["living_off_the_land"] = detect_living_off_the_land_adapter(data=df, limit=limit)
    results["credential_dumping"] = detect_credential_dumping_adapter(data=df, limit=limit)
    results["process_injection"] = detect_process_injection_adapter(data=df, limit=limit)
    results["unusual_child_processes"] = detect_unusual_child_processes_adapter(data=df, limit=limit)
    results["persistence_mechanisms"] = detect_persistence_mechanisms_adapter(data=df, limit=limit)
    results["obfuscated_scripts"] = detect_obfuscated_scripts_adapter(data=df, limit=limit)
    results["data_exfiltration"] = detect_data_exfiltration_adapter(data=df, limit=limit)
    results["defense_evasion"] = detect_defense_evasion_adapter(data=df, limit=limit)
    results["lateral_movement"] = detect_lateral_movement_adapter(data=df, limit=limit)
    results["ransomware_behavior"] = detect_ransomware_behavior_adapter(data=df, limit=limit)
    
    # Summarize all detections
    total_detections = sum(r.get("detection_count", 0) for r in results.values() if isinstance(r, dict))
    
    # Calculate detection counts by category
    detection_counts = {
        k: v.get("detection_count", 0) 
        for k, v in results.items() 
        if isinstance(v, dict)
    }
    
    # Find the top threats by detection count
    top_threats = sorted(
        detection_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    return {
        "success": True,
        "total_detection_count": total_detections,
        "detection_summary": detection_counts,
        "top_threats": top_threats,
        "detailed_results": results
    }

# Tool registry
TOOL_REGISTRY = {
    "security:load_data": load_data_adapter,
    "security:living_off_the_land": detect_living_off_the_land_adapter,
    "security:credential_dumping": detect_credential_dumping_adapter,
    "security:process_injection": detect_process_injection_adapter,
    "security:unusual_child_processes": detect_unusual_child_processes_adapter,
    "security:persistence_mechanisms": detect_persistence_mechanisms_adapter,
    "security:obfuscated_scripts": detect_obfuscated_scripts_adapter,
    "security:data_exfiltration": detect_data_exfiltration_adapter,
    "security:defense_evasion": detect_defense_evasion_adapter,
    "security:lateral_movement": detect_lateral_movement_adapter,
    "security:ransomware_behavior": detect_ransomware_behavior_adapter,
    "security:run_all": run_all_detections
}

def execute_tool(tool_id, **kwargs):
    """Main entry point for tool execution"""
    if tool_id in TOOL_REGISTRY:
        try:
            print(f"Executing tool: {tool_id}")
            result = TOOL_REGISTRY[tool_id](**kwargs)
            print(f"Tool execution completed: {tool_id}")
            return result
        except Exception as e:
            print(f"Error executing tool {tool_id}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    else:
        print(f"Unknown tool: {tool_id}")
        return {"error": f"Unknown tool: {tool_id}"}

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 2:
        tool_id = sys.argv[1]
        file_path = sys.argv[2]
        
        print(f"Running {tool_id} on {file_path}")
        result = execute_tool(tool_id, file_path=file_path)
        print(json.dumps(result, default=str, indent=2))
    else:
        print("Usage: python security_detection_adapter.py <tool_id> <file_path>")
