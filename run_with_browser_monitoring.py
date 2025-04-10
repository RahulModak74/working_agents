#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import glob
import argparse
import json
from datetime import datetime

def clean_directories():
    """Clean the browser_logs and screenshots directories"""
    for directory in ['browser_logs', 'screenshots']:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            # Remove old files but keep the directory
            for file in glob.glob(os.path.join(directory, '*')):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")

def run_workflow(workflow_file, data_file=None):
    """Run the workflow with the runner_4.py script"""
    # Construct the command
    cmd = [sys.executable, 'runner_4.py', '--workflow', workflow_file]
    if data_file:
        cmd.append(data_file)
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Run the command and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Process output line by line in real-time
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    # Wait for the process to complete
    process.wait()
    
    # Check return code
    if process.returncode != 0:
        print(f"Workflow execution failed with return code {process.returncode}")
        return False
    
    return True

def count_browser_operations():
    """Count browser operations from log files"""
    operations = {
        "create": 0,
        "navigate": 0,
        "get_content": 0,
        "find_elements": 0,
        "click": 0,
        "close": 0
    }
    
    # Process all log files
    for log_file in glob.glob(os.path.join('browser_logs', 'browser_*.log')):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "Tool called: browser:" in line:
                        for op in operations.keys():
                            if f"browser:{op}" in line:
                                operations[op] += 1
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
    
    return operations

def get_unique_urls_visited():
    """Extract unique URLs that were visited during browsing"""
    urls = set()
    
    # Process all log files
    for log_file in glob.glob(os.path.join('browser_logs', 'browser_*.log')):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "Navigating to URL:" in line:
                        # Extract the URL from the log line
                        start = line.find("Navigating to URL:") + len("Navigating to URL:")
                        end = line.find(" in browser:")
                        if end > start:
                            url = line[start:end].strip()
                            urls.add(url)
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
    
    return sorted(urls)

def get_screenshot_summary():
    """Get a summary of screenshots taken during browsing"""
    screenshots = glob.glob(os.path.join('screenshots', '*.png'))
    
    # Group screenshots by browser ID and action
    grouped = {}
    
    for screenshot in screenshots:
        filename = os.path.basename(screenshot)
        parts = filename.split('_')
        
        if len(parts) >= 2:
            browser_id = parts[0]
            action = parts[1]
            
            if browser_id not in grouped:
                grouped[browser_id] = {}
            
            if action not in grouped[browser_id]:
                grouped[browser_id][action] = []
            
            grouped[browser_id][action].append(filename)
    
    return {
        "total": len(screenshots),
        "by_browser": grouped
    }

def generate_workflow_summary(workflow_file, successful):
    """Generate a summary of the workflow execution"""
    summary = {
        "workflow_file": workflow_file,
        "execution_time": datetime.now().isoformat(),
        "status": "success" if successful else "failed",
        "browser_activity": {
            "operations": count_browser_operations(),
            "urls_visited": get_unique_urls_visited(),
            "screenshots": get_screenshot_summary()
        }
    }
    
    # Calculate total operations
    total_ops = sum(summary["browser_activity"]["operations"].values())
    summary["browser_activity"]["total_operations"] = total_ops
    
    # Save the summary to a file
    summary_file = "browser_workflow_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWorkflow summary saved to {summary_file}")
    
    return summary

def print_summary(summary):
    """Print a human-readable summary of the workflow execution"""
    print("\n" + "="*80)
    print("                  WORKFLOW EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nWorkflow file: {summary['workflow_file']}")
    print(f"Execution time: {summary['execution_time']}")
    print(f"Status: {summary['status'].upper()}")
    
    print("\nBROWSER ACTIVITY:")
    
    # Print operations
    ops = summary["browser_activity"]["operations"]
    print("\n  Operations:")
    print(f"    Browser sessions created: {ops['create']}")
    print(f"    Page navigations:         {ops['navigate']}")
    print(f"    Content retrievals:       {ops['get_content']}")
    print(f"    Element searches:         {ops['find_elements']}")
    print(f"    Element clicks:           {ops['click']}")
    print(f"    Browser sessions closed:  {ops['close']}")
    print(f"    TOTAL OPERATIONS:         {summary['browser_activity']['total_operations']}")
    
    # Print URLs visited
    urls = summary["browser_activity"]["urls_visited"]
    print(f"\n  URLs visited ({len(urls)}):")
    for i, url in enumerate(urls, 1):
        print(f"    {i}. {url}")
    
    # Print screenshot summary
    screenshots = summary["browser_activity"]["screenshots"]
    print(f"\n  Screenshots ({screenshots['total']}):")
    for browser_id, actions in screenshots.get("by_browser", {}).items():
        print(f"    Browser {browser_id}:")
        for action, files in actions.items():
            print(f"      {action}: {len(files)} screenshots")
    
    print("\n" + "="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run workflow with browser monitoring")
    parser.add_argument('workflow_file', help='Path to the workflow JSON file')
    parser.add_argument('--data-file', help='Optional data file for the workflow')
    parser.add_argument('--clean', action='store_true', help='Clean browser logs and screenshots before running')
    
    args = parser.parse_args()
    
    # Check if workflow file exists
    if not os.path.exists(args.workflow_file):
        print(f"Error: Workflow file {args.workflow_file} not found")
        sys.exit(1)
    
    # Check if data file exists (if provided)
    if args.data_file and not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found")
        sys.exit(1)
    
    # Clean directories if requested
    if args.clean:
        print("Cleaning browser logs and screenshots...")
        clean_directories()
    
    # Run the workflow
    start_time = time.time()
    successful = run_workflow(args.workflow_file, args.data_file)
    end_time = time.time()
    
    # Generate and print summary
    print(f"\nWorkflow execution completed in {end_time - start_time:.2f} seconds")
    summary = generate_workflow_summary(args.workflow_file, successful)
    print_summary(summary)
    
    # Suggest next steps
    print("\nNext steps:")
    print("  1. View browser logs: python browser_monitor.py --follow")
    print("  2. View browser statistics: python browser_monitor.py --stats")
    print("  3. Examine screenshots in the 'screenshots' directory")
    print("  4. Check workflow results in the 'agent_outputs' directory")
    
    return 0 if successful else 1

if __name__ == "__main__":
    sys.exit(main())
