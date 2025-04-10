#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Union
import re
import time
import copy
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("react_planning_adapter")

# Initialize the tool registry
TOOL_REGISTRY = {}

# React planning patterns
REACT_PATTERNS = {
    "basic_reasoning": {
        "description": "A systematic approach to problem-solving using reasoning, action, and observation",
        "stages": ["understand_problem", "generate_hypothesis", "plan_action", "execute_action", "observe_results", "learn_and_adapt"],
        "workflow_file": "react_basic_workflow.json"
    },
    "multi_step_reasoning": {
        "description": "Complex problem-solving with iterative reasoning and action cycles",
        "stages": ["initial_assessment", "hypothesis_generation", "action_planning", "multi_stage_execution", "comprehensive_observation", "strategic_learning"],
        "workflow_file": "react_multi_step_workflow.json"
    },
    "adaptive_reasoning": {
        "description": "Dynamic problem-solving with real-time adaptation and learning",
        "stages": ["initial_context", "contextual_reasoning", "flexible_action", "dynamic_observation", "meta_learning", "strategy_refinement"],
        "workflow_file": "react_adaptive_workflow.json"
    }
}

class ReactWorkflowManager:
    def __init__(self):
        self.workflows_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_workflows = {}
        self.session_data = {}
        
        # Load React workflows
        self.react_workflows = {}
        for pattern_id, pattern_info in REACT_PATTERNS.items():
            workflow_path = os.path.join(self.workflows_dir, pattern_info["workflow_file"])
            if os.path.exists(workflow_path):
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        self.react_workflows[pattern_id] = json.load(f)
                        logger.info(f"Loaded React workflow: {pattern_id}")
                except Exception as e:
                    logger.error(f"Error loading React workflow {pattern_id}: {str(e)}")
    
    def create_react_session(self, pattern_id: str, session_id: str = None, problem_description: str = None) -> Dict[str, Any]:
        """Create a new React planning session"""
        if pattern_id not in REACT_PATTERNS:
            return {"error": f"Unknown React pattern: {pattern_id}"}
        
        if pattern_id not in self.react_workflows:
            return {"error": f"Workflow for pattern {pattern_id} not loaded"}
        
        # Generate a session ID if not provided
        if session_id is None:
            session_id = f"react_{pattern_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create session data structure
        self.session_data[session_id] = {
            "pattern": pattern_id,
            "workflow": copy.deepcopy(self.react_workflows[pattern_id]),
            "problem_description": problem_description,
            "status": "created",
            "current_stage": 0,
            "results": {},
            "action_history": [],
            "learning_log": [],
            "created_at": time.time()
        }
        
        # Add problem description to workflow steps
        if problem_description:
            for i, step in enumerate(self.session_data[session_id]["workflow"]):
                if "content" in step and "{problem_description}" in step["content"]:
                    self.session_data[session_id]["workflow"][i]["content"] = \
                        step["content"].replace("{problem_description}", problem_description)
        
        return {
            "status": "success",
            "session_id": session_id,
            "pattern": pattern_id,
            "description": REACT_PATTERNS[pattern_id]["description"],
            "stages": REACT_PATTERNS[pattern_id]["stages"]
        }
    
    def take_react_action(self, session_id: str, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reasoning and action step in the React workflow"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        
        # Log the action
        action_log = {
            "action_type": action_type,
            "action_details": action_details,
            "timestamp": time.time()
        }
        session["action_history"].append(action_log)
        
        # Perform action based on type (this would be expanded based on specific action types)
        try:
            action_result = self._execute_action(action_type, action_details)
            
            # Log result and learn
            learning_entry = {
                "action": action_log,
                "result": action_result,
                "learning_insights": self._generate_learning_insights(action_result)
            }
            session["learning_log"].append(learning_entry)
            
            return {
                "status": "success",
                "action_result": action_result,
                "learning_insights": learning_entry["learning_insights"]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _execute_action(self, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific type of action. 
        This method would be expanded with more complex action handling logic.
        """
        # Placeholder for action execution logic
        return {
            "action_type": action_type,
            "action_details": action_details,
            "result": "Action executed successfully",
            "raw_output": {}
        }
    
    def _generate_learning_insights(self, action_result: Dict[str, Any]) -> List[str]:
        """
        Generate learning insights from action results.
        This method would use more sophisticated learning algorithms in a real implementation.
        """
        insights = [
            f"Executed action of type: {action_result.get('action_type')}",
            f"Result summary: {action_result.get('result', 'No specific result')}"
        ]
        return insights
    
    def get_session_learning_log(self, session_id: str) -> Dict[str, Any]:
        """Retrieve the learning log for a React session"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        
        return {
            "session_id": session_id,
            "pattern": session["pattern"],
            "action_history": session["action_history"],
            "learning_log": session["learning_log"]
        }

# Global manager instance
REACT_MANAGER = ReactWorkflowManager()

# Tool registration functions
def react_create_session(pattern_id: str, session_id: str = None, problem_description: str = None) -> Dict[str, Any]:
    """Create a new React planning session"""
    return REACT_MANAGER.create_react_session(pattern_id, session_id, problem_description)

def react_take_action(session_id: str, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
    """Take a reasoning and action step in the React workflow"""
    return REACT_MANAGER.take_react_action(session_id, action_type, action_details)

def react_get_learning_log(session_id: str) -> Dict[str, Any]:
    """Retrieve the learning log for a React session"""
    return REACT_MANAGER.get_session_learning_log(session_id)

def react_list_patterns() -> Dict[str, Any]:
    """List all available React planning patterns"""
    patterns = {}
    for pattern_id, pattern_info in REACT_PATTERNS.items():
        patterns[pattern_id] = {
            "description": pattern_info["description"],
            "stages": pattern_info["stages"],
            "available": pattern_id in REACT_MANAGER.react_workflows
        }
    
    return {
        "status": "success",
        "patterns": patterns
    }

# Register tools
TOOL_REGISTRY["react:create_session"] = react_create_session
TOOL_REGISTRY["react:take_action"] = react_take_action
TOOL_REGISTRY["react:get_learning_log"] = react_get_learning_log
TOOL_REGISTRY["react:list_patterns"] = react_list_patterns

# Print initialization message
print("✅ React planning tools registered successfully")
