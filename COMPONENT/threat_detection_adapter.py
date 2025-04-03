#!/usr/bin/env python3

"""
Threat Detection Adapter Module
This module provides an adapter layer between the workflow JSON tool format and
the actual tool implementation in full_detect.py.
"""

import os
import sys
import json
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime, timedelta

# Ensure both directories are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import the detection module
try:
    from full_detect import (
        load_data,
        detect_long_dwell_time,
        detect_beaconing,
        detect_weekend_exfiltration,
        detect_distributed_reconnaissance,
        detect_service_account_anomaly,
        detect_cross_system_attack_chain
    )
    real_tools_available = True
except ImportError:
    print("Warning: Real threat detection module not available. Using mock implementations.")
    real_tools_available = False

# Mock implementations for when real tools aren't available
def mock_load_data(file_path):
    """Mock data loading implementation"""
    print(f"[MOCK] Loading data from {file_path}")
    return pd.DataFrame({
        'hostname': ['host1', 'host2', 'host3'],
        'timestamp': [datetime.now(), datetime.now(), datetime.now()],
        'name': ['process1', 'process2', 'process3'],
        'pid': [1001, 1002, 1003],
        'remote_ips': [['192.168.1.1'], ['10.0.0.1'], ['172.16.0.1']]
    })

def mock_detect_long_dwell_time(df, days_threshold=30, lookback_days=90):
    """Mock long dwell time detection implementation"""
    print(f"[MOCK] Detecting long dwell time (threshold: {days_threshold} days, lookback: {lookback_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['Critical'],
        'hostname': ['host1'],
        'pid': [1001],
        'process_name': ['suspicious_process'],
        'detection_type': ['Delayed Execution Pattern'],
        'alert_name': ['Long-Term Dwell Time Detection']
    })

def mock_detect_beaconing(df, lookback_days=60, active_days_threshold=10, consistency_threshold=0.8):
    """Mock beaconing detection implementation"""
    print(f"[MOCK] Detecting beaconing patterns (lookback: {lookback_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['High'],
        'hostname': ['host2'],
        'pid': [1002],
        'process_name': ['beacon_process'],
        'detection_type': ['Consistent Temporal Beaconing'],
        'alert_name': ['Temporal Networking Anomaly']
    })

def mock_detect_weekend_exfiltration(df, lookback_days=60):
    """Mock weekend exfiltration detection implementation"""
    print(f"[MOCK] Detecting weekend exfiltration (lookback: {lookback_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['Critical'],
        'hostname': ['host3'],
        'pid': [1003],
        'process_name': ['exfil_process'],
        'detection_type': ['Data Exfiltration via Steganography'],
        'alert_name': ['Weekend Exfiltration Detection']
    })

def mock_detect_distributed_reconnaissance(df, lookback_days=30):
    """Mock distributed reconnaissance detection implementation"""
    print(f"[MOCK] Detecting distributed reconnaissance (lookback: {lookback_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['High'],
        'detection_type': ['Multi-system Coordinated Reconnaissance'],
        'affected_systems': ['host1, host2'],
        'alert_name': ['Distributed Reconnaissance Campaign']
    })

def mock_detect_service_account_anomaly(df, baseline_days=90, recent_days=30):
    """Mock service account anomaly detection implementation"""
    print(f"[MOCK] Detecting service account anomalies (baseline: {baseline_days} days, recent: {recent_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['Critical'],
        'detection_type': ['Abnormal Service Account Usage'],
        'account': ['svc_admin'],
        'alert_name': ['Service Account Anomaly']
    })

def mock_detect_cross_system_attack_chain(df, lookback_days=60, min_hosts=3, min_days=7):
    """Mock cross system attack chain detection implementation"""
    print(f"[MOCK] Detecting cross-system attack chains (lookback: {lookback_days} days)")
    return pd.DataFrame({
        'detection_time': [datetime.now()],
        'severity': ['Critical'],
        'detection_type': ['Distributed Attack Chain'],
        'user': ['compromised_user'],
        'host_count': [4],
        'alert_name': ['Cross-System Attack Chain Detected']
    })

# Adapter functions that bridge between workflow JSON format and actual tool implementation

def load_data_adapter(**kwargs):
    """Adapter for data loading"""
    file_path = kwargs.get("file_path")
    
    if not file_path:
        return {"error": "File path not provided for data loading"}
    
    if real_tools_available:
        try:
            df = load_data(file_path)
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
    else:
        mock_df = mock_load_data(file_path)
        return {
            "success": True,
            "message": f"[MOCK] Data loaded successfully. Shape: {mock_df.shape}",
            "row_count": len(mock_df),
            "column_count": len(mock_df.columns),
            "data": mock_df
        }

def detect_long_dwell_time_adapter(**kwargs):
    """Adapter for long dwell time detection"""
    df = kwargs.get("data")
    days_threshold = kwargs.get("days_threshold", 30)
    lookback_days = kwargs.get("lookback_days", 90)
    
    # Convert string values to appropriate types
    if isinstance(days_threshold, str):
        try:
            days_threshold = int(days_threshold)
        except ValueError:
            days_threshold = 30
    
    if isinstance(lookback_days, str):
        try:
            lookback_days = int(lookback_days)
        except ValueError:
            lookback_days = 90
    
    if df is None:
        return {"error": "No data provided for long dwell time detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_long_dwell_time(df, days_threshold, lookback_days)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting long dwell time: {e}")
            return {"error": f"Long dwell time detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_long_dwell_time(df, days_threshold, lookback_days)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_beaconing_adapter(**kwargs):
    """Adapter for beaconing detection"""
    df = kwargs.get("data")
    lookback_days = kwargs.get("lookback_days", 60)
    active_days_threshold = kwargs.get("active_days_threshold", 10)
    consistency_threshold = kwargs.get("consistency_threshold", 0.8)
    
    # Convert string values to appropriate types
    if isinstance(lookback_days, str):
        try:
            lookback_days = int(lookback_days)
        except ValueError:
            lookback_days = 60
    
    if isinstance(active_days_threshold, str):
        try:
            active_days_threshold = int(active_days_threshold)
        except ValueError:
            active_days_threshold = 10
    
    if isinstance(consistency_threshold, str):
        try:
            consistency_threshold = float(consistency_threshold)
        except ValueError:
            consistency_threshold = 0.8
    
    if df is None:
        return {"error": "No data provided for beaconing detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_beaconing(df, lookback_days, active_days_threshold, consistency_threshold)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting beaconing: {e}")
            return {"error": f"Beaconing detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_beaconing(df, lookback_days, active_days_threshold, consistency_threshold)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_weekend_exfiltration_adapter(**kwargs):
    """Adapter for weekend exfiltration detection"""
    df = kwargs.get("data")
    lookback_days = kwargs.get("lookback_days", 60)
    
    # Convert string values to appropriate types
    if isinstance(lookback_days, str):
        try:
            lookback_days = int(lookback_days)
        except ValueError:
            lookback_days = 60
    
    if df is None:
        return {"error": "No data provided for weekend exfiltration detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_weekend_exfiltration(df, lookback_days)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting weekend exfiltration: {e}")
            return {"error": f"Weekend exfiltration detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_weekend_exfiltration(df, lookback_days)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_distributed_reconnaissance_adapter(**kwargs):
    """Adapter for distributed reconnaissance detection"""
    df = kwargs.get("data")
    lookback_days = kwargs.get("lookback_days", 30)
    
    # Convert string values to appropriate types
    if isinstance(lookback_days, str):
        try:
            lookback_days = int(lookback_days)
        except ValueError:
            lookback_days = 30
    
    if df is None:
        return {"error": "No data provided for distributed reconnaissance detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_distributed_reconnaissance(df, lookback_days)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting distributed reconnaissance: {e}")
            return {"error": f"Distributed reconnaissance detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_distributed_reconnaissance(df, lookback_days)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_service_account_anomaly_adapter(**kwargs):
    """Adapter for service account anomaly detection"""
    df = kwargs.get("data")
    baseline_days = kwargs.get("baseline_days", 90)
    recent_days = kwargs.get("recent_days", 30)
    
    # Convert string values to appropriate types
    if isinstance(baseline_days, str):
        try:
            baseline_days = int(baseline_days)
        except ValueError:
            baseline_days = 90
    
    if isinstance(recent_days, str):
        try:
            recent_days = int(recent_days)
        except ValueError:
            recent_days = 30
    
    if df is None:
        return {"error": "No data provided for service account anomaly detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_service_account_anomaly(df, baseline_days, recent_days)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting service account anomalies: {e}")
            return {"error": f"Service account anomaly detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_service_account_anomaly(df, baseline_days, recent_days)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def detect_cross_system_attack_chain_adapter(**kwargs):
    """Adapter for cross system attack chain detection"""
    df = kwargs.get("data")
    lookback_days = kwargs.get("lookback_days", 60)
    min_hosts = kwargs.get("min_hosts", 3)
    min_days = kwargs.get("min_days", 7)
    
    # Convert string values to appropriate types
    if isinstance(lookback_days, str):
        try:
            lookback_days = int(lookback_days)
        except ValueError:
            lookback_days = 60
    
    if isinstance(min_hosts, str):
        try:
            min_hosts = int(min_hosts)
        except ValueError:
            min_hosts = 3
    
    if isinstance(min_days, str):
        try:
            min_days = int(min_days)
        except ValueError:
            min_days = 7
    
    if df is None:
        return {"error": "No data provided for cross system attack chain detection"}
    
    if real_tools_available and not isinstance(df, str):
        try:
            result_df = detect_cross_system_attack_chain(df, lookback_days, min_hosts, min_days)
            return {
                "success": True,
                "detection_count": len(result_df),
                "detections": result_df.to_dict('records') if not result_df.empty else []
            }
        except Exception as e:
            print(f"Error detecting cross system attack chains: {e}")
            return {"error": f"Cross system attack chain detection failed: {str(e)}"}
    else:
        mock_result = mock_detect_cross_system_attack_chain(df, lookback_days, min_hosts, min_days)
        return {
            "success": True,
            "detection_count": len(mock_result),
            "detections": mock_result.to_dict('records') if not mock_result.empty else []
        }

def run_all_detections(**kwargs):
    """Run all threat detection algorithms on the given data"""
    file_path = kwargs.get("file_path")
    
    if not file_path:
        return {"error": "File path not provided for all detections"}
    
    # Load data first
    load_result = load_data_adapter(file_path=file_path)
    
    if "error" in load_result:
        return load_result
    
    df = load_result.get("data")
    
    # Run all detections
    results = {}
    
    # APT using long-dwell delayed execution
    results["long_dwell_time"] = detect_long_dwell_time_adapter(data=df)
    results["beaconing"] = detect_beaconing_adapter(data=df)
    results["weekend_exfiltration"] = detect_weekend_exfiltration_adapter(data=df)
    
    # Cross-system lateral movement
    results["distributed_reconnaissance"] = detect_distributed_reconnaissance_adapter(data=df)
    results["service_account_anomaly"] = detect_service_account_anomaly_adapter(data=df)
    results["cross_system_attack_chain"] = detect_cross_system_attack_chain_adapter(data=df)
    
    # Summarize all detections
    total_detections = sum(r.get("detection_count", 0) for r in results.values() if isinstance(r, dict))
    
    return {
        "success": True,
        "total_detection_count": total_detections,
        "detection_summary": {
            k: v.get("detection_count", 0) 
            for k, v in results.items() 
            if isinstance(v, dict)
        },
        "detailed_results": results
    }

# Tool registry
TOOL_REGISTRY = {
    "threat:load_data": load_data_adapter,
    "threat:long_dwell_time": detect_long_dwell_time_adapter,
    "threat:beaconing": detect_beaconing_adapter,
    "threat:weekend_exfiltration": detect_weekend_exfiltration_adapter,
    "threat:distributed_reconnaissance": detect_distributed_reconnaissance_adapter,
    "threat:service_account_anomaly": detect_service_account_anomaly_adapter,
    "threat:cross_system_attack_chain": detect_cross_system_attack_chain_adapter,
    "threat:run_all": run_all_detections
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
        print("Usage: python threat_detection_adapter.py <tool_id> <file_path>")
