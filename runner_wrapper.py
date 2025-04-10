#!/usr/bin/env python3

import sys
import os

# Ensure the main directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also ensure COMPONENT directory is in the path
component_dir = os.path.join(current_dir, "COMPONENT")
if os.path.exists(component_dir) and component_dir not in sys.path:
    sys.path.insert(0, component_dir)

# Import our modules to make them available
import tool_manager
import utils
from agent_runner import main

# This ensures all the required modules are loaded before main() is called
if __name__ == "__main__":
    main()
