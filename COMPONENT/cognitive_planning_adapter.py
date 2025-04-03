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
logger = logging.getLogger("cognitive_planning_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Cognitive planning patterns
COGNITIVE_PATTERNS = {
    "tree_of_thoughts": {
        "description": "A structured exploration of multiple solution paths using a tree-based approach",
        "stages": ["generate_root_approaches", "select_promising_branches", "explore_branches", "evaluate_branches", "develop_solution"],
        "workflow_file": "cognitive_tree_of_thoughts.json"
    },
    "metacognitive_reflection": {
        "description": "Analysis of reasoning processes, biases, and counterfactuals to improve solutions",
        "stages": ["initial_solution", "monitor_reasoning", "counterfactual_exploration", "bias_detection", "process_optimization", "solution_revision"],
        "workflow_file": "cognitive_metacognition.json"
    },
    "multi_agent_debate": {
        "description": "Structured debate between multiple perspectives to develop nuanced solutions",
        "stages": ["frame_debate", "present_positions", "rebuttals", "critique", "synthesis"],
        "workflow_file": "cognitive_debate_protocol.json"
    },
    "adaptive_cognition": {
        "description": "Dynamic selection and application of the most appropriate cognitive architecture",
        "stages": ["problem_framing", "architecture_selection", "solution_development", "integration", "reflection"],
        "workflow_file": "cognitive_combined.json"
    }
}

class CognitiveWorkflowManager:
    def __init__(self):
        self.workflows_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_workflows = {}
        self.session_data = {}
        
        # Load cognitive workflows
        self.cognitive_workflows = {}
        for pattern_id, pattern_info in COGNITIVE_PATTERNS.items():
            workflow_path = os.path.join(self.workflows_dir, pattern_info["workflow_file"])
            if os.path.exists(workflow_path):
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        self.cognitive_workflows[pattern_id] = json.load(f)
                        logger.info(f"Loaded cognitive workflow: {pattern_id}")
                except Exception as e:
                    logger.error(f"Error loading cognitive workflow {pattern_id}: {str(e)}")
    
    def create_cognitive_session(self, pattern_id: str, session_id: str = None, problem_description: str = None) -> Dict[str, Any]:
        """Create a new cognitive planning session using specified pattern"""
        if pattern_id not in COGNITIVE_PATTERNS:
            return {"error": f"Unknown cognitive pattern: {pattern_id}"}
        
        if pattern_id not in self.cognitive_workflows:
            return {"error": f"Workflow for pattern {pattern_id} not loaded"}
        
        # Generate a session ID if not provided
        if session_id is None:
            session_id = f"{pattern_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create session data structure
        self.session_data[session_id] = {
            "pattern": pattern_id,
            "workflow": copy.deepcopy(self.cognitive_workflows[pattern_id]),
            "problem_description": problem_description,
            "status": "created",
            "current_stage": 0,
            "results": {},
            "created_at": time.time()
        }
        
        # If problem description is provided, add it to all agent prompts
        if problem_description:
            for i, step in enumerate(self.session_data[session_id]["workflow"]):
                if "content" in step and "{problem_description}" in step["content"]:
                    self.session_data[session_id]["workflow"][i]["content"] = \
                        step["content"].replace("{problem_description}", problem_description)
        
        return {
            "status": "success",
            "session_id": session_id,
            "pattern": pattern_id,
            "description": COGNITIVE_PATTERNS[pattern_id]["description"],
            "stages": COGNITIVE_PATTERNS[pattern_id]["stages"]
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status and information about a cognitive session"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        pattern_id = session["pattern"]
        
        return {
            "session_id": session_id,
            "pattern": pattern_id,
            "description": COGNITIVE_PATTERNS[pattern_id]["description"],
            "status": session["status"],
            "current_stage": session["current_stage"],
            "stages": COGNITIVE_PATTERNS[pattern_id]["stages"],
            "created_at": session["created_at"],
            "results_available": list(session["results"].keys())
        }
    
    def get_next_step(self, session_id: str) -> Dict[str, Any]:
        """Get the next workflow step for the cognitive session"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        
        if session["status"] == "completed":
            return {"status": "completed", "message": "This cognitive session has completed all stages"}
        
        current_stage = session["current_stage"]
        workflow = session["workflow"]
        
        if current_stage >= len(workflow):
            session["status"] = "completed"
            return {"status": "completed", "message": "No more steps in this cognitive workflow"}
        
        next_step = workflow[current_stage]
        
        # Check for dynamic agent that needs to be resolved
        if next_step.get("type") == "dynamic" and session["status"] != "awaiting_action_selection":
            # Mark that we're waiting for an action selection
            session["status"] = "awaiting_action_selection"
            
            # Return the dynamic agent's initial prompt
            return {
                "status": "dynamic_step",
                "step_id": current_stage,
                "agent_name": next_step.get("agent", f"agent_{current_stage}"),
                "prompt": next_step.get("initial_prompt", "Select the next action to take"),
                "available_actions": list(next_step.get("actions", {}).keys())
            }
        
        # For regular steps or after action selection for dynamic agents
        return {
            "status": "ready",
            "step_id": current_stage,
            "agent_name": next_step.get("agent", f"agent_{current_stage}"),
            "step_details": next_step
        }
    
    def select_dynamic_action(self, session_id: str, action: str) -> Dict[str, Any]:
        """Select an action for a dynamic agent in the workflow"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        
        if session["status"] != "awaiting_action_selection":
            return {"error": "Current step is not awaiting action selection"}
        
        current_stage = session["current_stage"]
        workflow = session["workflow"]
        
        if current_stage >= len(workflow):
            return {"error": "No current step available"}
        
        current_step = workflow[current_stage]
        
        if current_step.get("type") != "dynamic":
            return {"error": "Current step is not a dynamic agent"}
        
        actions = current_step.get("actions", {})
        
        if action not in actions:
            return {"error": f"Invalid action: {action}. Available actions: {list(actions.keys())}"}
        
        # Store the selected action
        session["results"][f"{current_step.get('agent', f'agent_{current_stage}')}_action"] = action
        
        # Update status
        session["status"] = "in_progress"
        
        # Return the action details
        return {
            "status": "success",
            "action": action,
            "action_details": actions[action]
        }
    
    def submit_step_result(self, session_id: str, agent_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Submit results for a step in the cognitive workflow"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        
        # Store the results
        session["results"][agent_name] = result
        
        # Advance to the next stage
        session["current_stage"] += 1
        
        # Update status
        if session["current_stage"] >= len(session["workflow"]):
            session["status"] = "completed"
        else:
            session["status"] = "in_progress"
        
        # Get information about the next step
        next_step_info = self.get_next_step(session_id)
        
        return {
            "status": "success",
            "message": f"Results for {agent_name} submitted successfully",
            "session_status": session["status"],
            "next_step": next_step_info
        }
    
    def get_session_results(self, session_id: str, include_workflow: bool = False) -> Dict[str, Any]:
        """Get all results from a cognitive session"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        pattern_id = session["pattern"]
        
        response = {
            "session_id": session_id,
            "pattern": pattern_id,
            "description": COGNITIVE_PATTERNS[pattern_id]["description"],
            "status": session["status"],
            "results": session["results"],
            "created_at": session["created_at"]
        }
        
        if include_workflow:
            response["workflow"] = session["workflow"]
        
        return response
    
    def transform_for_research(self, session_id: str, research_query: str = None) -> Dict[str, Any]:
        """Transform cognitive session results into research planning structure"""
        if session_id not in self.session_data:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.session_data[session_id]
        pattern_id = session["pattern"]
        
        if session["status"] != "completed":
            return {"error": "Cognitive session is not yet completed"}
        
        # Extract key information based on pattern type
        if pattern_id == "tree_of_thoughts":
            return self._transform_tot_for_research(session, research_query)
        elif pattern_id == "metacognitive_reflection":
            return self._transform_metacog_for_research(session, research_query)
        elif pattern_id == "multi_agent_debate":
            return self._transform_debate_for_research(session, research_query)
        elif pattern_id == "adaptive_cognition":
            return self._transform_adaptive_for_research(session, research_query)
        else:
            return {"error": f"Transformation for pattern {pattern_id} not implemented"}
    
    def _transform_tot_for_research(self, session: Dict[str, Any], research_query: str) -> Dict[str, Any]:
        """Transform Tree of Thoughts results for research planning"""
        results = session["results"]
        
        # Extract root approaches
        root_agent_result = results.get("tot_root_agent", {})
        initial_approaches = root_agent_result.get("initial_approaches", [])
        
        # Extract branch evaluations
        evaluator_result = results.get("tot_evaluator", {})
        branch_evals = evaluator_result.get("branch_evaluations", [])
        recommended_branch = evaluator_result.get("recommended_branch", "")
        
        # Extract final solution
        solution_result = results.get("tot_solution_agent", {})
        
        # Transform into research plan
        research_plan = {
            "research_topic": research_query or "Topic derived from cognitive planning session",
            "research_plan": {
                "goal": "Conduct comprehensive research based on optimal cognitive approach",
                "key_questions": [],
                "subtopics": []
            },
            "search_queries": [],
            "reasoning": "Derived from Tree of Thoughts cognitive architecture"
        }
        
        # Add key questions from initial approaches
        for approach in initial_approaches:
            if "description" in approach:
                research_plan["research_plan"]["key_questions"].append(
                    f"How viable is the approach: {approach['description']}?"
                )
        
        # Add subtopics from branch evaluations
        for branch in branch_evals:
            if "branch_id" in branch and "strengths" in branch:
                for strength in branch["strengths"]:
                    research_plan["research_plan"]["subtopics"].append(strength)
        
        # Add search queries from recommended branch and solution
        if solution_result:
            # Extract potential search queries from the markdown sections
            if isinstance(solution_result, str):
                lines = solution_result.split("\n")
                for line in lines:
                    if line.startswith("## ") or line.startswith("# "):
                        research_plan["search_queries"].append(
                            f"{research_query or 'Research'}: {line.replace('#', '').strip()}"
                        )
        
        return {
            "status": "success",
            "cognitive_pattern": "tree_of_thoughts",
            "research_plan": research_plan
        }
    
    def _transform_metacog_for_research(self, session: Dict[str, Any], research_query: str) -> Dict[str, Any]:
        """Transform Metacognitive Reflection results for research planning"""
        results = session["results"]
        
        # Extract problem understanding
        problem_result = results.get("problem_solver", {})
        problem_understanding = problem_result.get("problem_understanding", "")
        
        # Extract reasoning patterns and biases
        metacog_result = results.get("metacognitive_monitor", {})
        reasoning_patterns = metacog_result.get("reasoning_patterns", [])
        biases = metacog_result.get("potential_biases", [])
        
        # Extract optimized framework
        optimizer_result = results.get("reasoning_optimizer", {})
        framework = optimizer_result.get("optimized_framework", {})
        
        # Transform into research plan
        research_plan = {
            "research_topic": research_query or "Topic derived from metacognitive reflection",
            "research_plan": {
                "goal": "Conduct bias-aware research using metacognitively optimized approach",
                "key_questions": [],
                "subtopics": []
            },
            "search_queries": [],
            "reasoning": "Derived from Metacognitive Reflection architecture"
        }
        
        # Add key questions from reasoning patterns
        for pattern in reasoning_patterns:
            research_plan["research_plan"]["key_questions"].append(
                f"How does {pattern} apply to {research_query or 'the research topic'}?"
            )
        
        # Add subtopics from framework steps
        if "steps" in framework:
            for step in framework["steps"]:
                if "step_name" in step:
                    research_plan["research_plan"]["subtopics"].append(step["step_name"])
        
        # Add search queries from biases to avoid
        for bias in biases:
            if isinstance(bias, dict) and "bias_type" in bias:
                research_plan["search_queries"].append(
                    f"{research_query or 'Research'} avoiding {bias['bias_type']} bias"
                )
        
        return {
            "status": "success",
            "cognitive_pattern": "metacognitive_reflection",
            "research_plan": research_plan
        }
    
    def _transform_debate_for_research(self, session: Dict[str, Any], research_query: str) -> Dict[str, Any]:
        """Transform Multi-Agent Debate results for research planning"""
        results = session["results"]
        
        # Extract debate framing
        moderator_result = results.get("debate_moderator", {})
        positions = moderator_result.get("positions", [])
        key_questions = moderator_result.get("key_questions", [])
        
        # Extract key arguments
        position_a = results.get("advocate_position_a", {})
        position_b = results.get("advocate_position_b", {})
        
        # Extract critic analysis
        critic_result = results.get("debate_critic", {})
        strongest_args = critic_result.get("strongest_arguments", {})
        potential_synthesis = critic_result.get("potential_synthesis", [])
        
        # Transform into research plan
        research_plan = {
            "research_topic": research_query or "Topic derived from multi-agent debate",
            "research_plan": {
                "goal": "Conduct balanced research exploring multiple perspectives",
                "key_questions": key_questions,
                "subtopics": []
            },
            "search_queries": [],
            "reasoning": "Derived from Multi-Agent Debate architecture"
        }
        
        # Add subtopics from positions
        for position in positions:
            research_plan["research_plan"]["subtopics"].append(f"Research on position: {position}")
        
        # Add search queries from strongest arguments
        for side, args in strongest_args.items():
            if isinstance(args, list):
                for arg in args:
                    research_plan["search_queries"].append(
                        f"{research_query or 'Research'}: evidence for {arg}"
                    )
        
        # Add search queries from potential synthesis
        for synthesis in potential_synthesis:
            research_plan["search_queries"].append(
                f"{research_query or 'Research'}: {synthesis}"
            )
        
        return {
            "status": "success",
            "cognitive_pattern": "multi_agent_debate",
            "research_plan": research_plan
        }
    
    def _transform_adaptive_for_research(self, session: Dict[str, Any], research_query: str) -> Dict[str, Any]:
        """Transform Adaptive Cognition results for research planning"""
        results = session["results"]
        
        # Extract problem framing
        framer_result = results.get("problem_framer", {})
        key_dimensions = framer_result.get("key_dimensions", [])
        potential_approaches = framer_result.get("potential_approaches", [])
        
        # Extract selected approach
        dynamic_result = results.get("dynamic_agent", {})
        selected_approach = dynamic_result.get("selected_approach", "")
        expected_benefits = dynamic_result.get("expected_benefits", [])
        
        # Extract effectiveness assessment
        architect_result = results.get("cognitive_architect", {})
        effectiveness = architect_result.get("architecture_effectiveness", {})
        improvements = architect_result.get("process_improvements", [])
        
        # Transform into research plan
        research_plan = {
            "research_topic": research_query or "Topic derived from adaptive cognition",
            "research_plan": {
                "goal": "Conduct research using dynamically selected optimal cognitive approach",
                "key_questions": [],
                "subtopics": key_dimensions
            },
            "search_queries": [],
            "reasoning": f"Derived from Adaptive Cognition using {selected_approach} approach"
        }
        
        # Add key questions from potential approaches
        for approach in potential_approaches:
            research_plan["research_plan"]["key_questions"].append(
                f"What insights can be gained through {approach}?"
            )
        
        # Add search queries from expected benefits and improvements
        for benefit in expected_benefits:
            research_plan["search_queries"].append(
                f"{research_query or 'Research'}: {benefit}"
            )
        
        for improvement in improvements:
            research_plan["search_queries"].append(
                f"{research_query or 'Research'} methodology: {improvement}"
            )
        
        return {
            "status": "success",
            "cognitive_pattern": "adaptive_cognition",
            "research_plan": research_plan
        }

# Global manager instance
COGNITIVE_MANAGER = CognitiveWorkflowManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def cognitive_create_session(pattern_id: str, session_id: str = None, problem_description: str = None) -> Dict[str, Any]:
    """
    Create a new cognitive planning session using a specified pattern.
    
    Args:
        pattern_id: The cognitive pattern to use ('tree_of_thoughts', 'metacognitive_reflection', etc.)
        session_id: Optional custom identifier for the session
        problem_description: Description of the problem to solve
    
    Returns:
        Dict with session information
    """
    return COGNITIVE_MANAGER.create_cognitive_session(pattern_id, session_id, problem_description)

def cognitive_get_status(session_id: str) -> Dict[str, Any]:
    """
    Get status and information about a cognitive session.
    
    Args:
        session_id: Identifier for the session
    
    Returns:
        Dict with session status and information
    """
    return COGNITIVE_MANAGER.get_session_status(session_id)

def cognitive_get_next_step(session_id: str) -> Dict[str, Any]:
    """
    Get the next workflow step for a cognitive session.
    
    Args:
        session_id: Identifier for the session
    
    Returns:
        Dict with information about the next step
    """
    return COGNITIVE_MANAGER.get_next_step(session_id)

def cognitive_select_action(session_id: str, action: str) -> Dict[str, Any]:
    """
    Select an action for a dynamic agent in the workflow.
    
    Args:
        session_id: Identifier for the session
        action: The action to select
    
    Returns:
        Dict with action information
    """
    return COGNITIVE_MANAGER.select_dynamic_action(session_id, action)

def cognitive_submit_result(session_id: str, agent_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit results for a step in the cognitive workflow.
    
    Args:
        session_id: Identifier for the session
        agent_name: Name of the agent that produced the result
        result: The results data
    
    Returns:
        Dict with submission status and next step information
    """
    return COGNITIVE_MANAGER.submit_step_result(session_id, agent_name, result)

def cognitive_get_results(session_id: str, include_workflow: bool = False) -> Dict[str, Any]:
    """
    Get all results from a cognitive session.
    
    Args:
        session_id: Identifier for the session
        include_workflow: Whether to include the full workflow in the response
    
    Returns:
        Dict with session results
    """
    return COGNITIVE_MANAGER.get_session_results(session_id, include_workflow)

def cognitive_transform_for_research(session_id: str, research_query: str = None) -> Dict[str, Any]:
    """
    Transform cognitive session results into a research planning structure.
    
    Args:
        session_id: Identifier for the session
        research_query: The specific research query to focus on
    
    Returns:
        Dict with transformed research plan
    """
    return COGNITIVE_MANAGER.transform_for_research(session_id, research_query)

def cognitive_list_patterns() -> Dict[str, Any]:
    """
    List all available cognitive planning patterns.
    
    Returns:
        Dict with available patterns and their descriptions
    """
    patterns = {}
    for pattern_id, pattern_info in COGNITIVE_PATTERNS.items():
        patterns[pattern_id] = {
            "description": pattern_info["description"],
            "stages": pattern_info["stages"],
            "available": pattern_id in COGNITIVE_MANAGER.cognitive_workflows
        }
    
    return {
        "status": "success",
        "patterns": patterns
    }

# Register tools
TOOL_REGISTRY["cognitive:create_session"] = cognitive_create_session
TOOL_REGISTRY["cognitive:get_status"] = cognitive_get_status
TOOL_REGISTRY["cognitive:get_next_step"] = cognitive_get_next_step
TOOL_REGISTRY["cognitive:select_action"] = cognitive_select_action
TOOL_REGISTRY["cognitive:submit_result"] = cognitive_submit_result
TOOL_REGISTRY["cognitive:get_results"] = cognitive_get_results
TOOL_REGISTRY["cognitive:transform_for_research"] = cognitive_transform_for_research
TOOL_REGISTRY["cognitive:list_patterns"] = cognitive_list_patterns

# Print initialization message
print("✅ Cognitive planning tools registered successfully")
