#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys

def setup_windows():
    """Set up the environment for Windows compatibility"""
    print("Setting up Enhanced Multi-Agent System for Windows...")
    
    # Create required directories
    os.makedirs("agent_outputs", exist_ok=True)
    os.makedirs("agent_memory", exist_ok=True)
    
    # Install required packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
    
    # Rename the Windows-compatible files to be the main files
    if os.path.exists("agent_windows.py"):
        shutil.copy("agent_windows.py", "agent.py")
        print("✅ Windows-compatible agent.py created")
        
    if os.path.exists("dynamic_agent_windows.py"):
        shutil.copy("dynamic_agent_windows.py", "dynamic_agent.py")
        print("✅ Windows-compatible dynamic_agent.py created")
        
    if os.path.exists("agent_system_windows.py"):
        shutil.copy("agent_system_windows.py", "agent_system.py")
        print("✅ Windows-compatible agent_system.py created")
        
    if os.path.exists("cli_windows.py"):
        shutil.copy("cli_windows.py", "cli.py")
        print("✅ Windows-compatible cli.py created")
        
    if os.path.exists("main_windows.py"):
        shutil.copy("main_windows.py", "main.py")
        print("✅ Windows-compatible main.py created")

    print("\nSetup complete! You can now run the system with: python main.py")
    print("Or try a workflow with: python main.py --workflow enhanced_workflow.json")

if __name__ == "__main__":
    setup_windows()
