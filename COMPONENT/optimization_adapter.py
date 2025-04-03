#!/usr/bin/env python3

import os
import sys
import json
import logging
import time
import copy
import random
from typing import Dict, Any, List, Optional, Union
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optimization_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Optimization patterns
OPTIMIZATION_PATTERNS = {
    "multi_objective": {
        "description": "Optimization approach that balances multiple competing objectives through metacognitive reflection",
        "stages": ["initial_solution", "metacognitive_analysis", "counterfactual_exploration", "bias_detection", "reasoning_optimization", "solution_revision"],
        "workflow_file": "optimizer_multi_objective.json"
    },
    "performance_monitoring": {
        "description": "System that monitors, analyzes, and diagnoses performance metrics across agent networks",
        "stages": ["metrics_definition", "data_collection", "anomaly_detection", "root_cause_analysis"],
        "workflow_file": "optimizer_monitoring.json"
    },
    "feedback_loop": {
        "description": "Hierarchical feedback system spanning micro, meso, and macro levels for comprehensive optimization",
        "stages": ["framework_design", "micro_feedback", "meso_feedback", "macro_feedback", "feedback_integration"],
        "workflow_file": "optimizer_feedback.json"
    },
    "adaptive_optimization": {
        "description": "Dynamic selection and application of the most appropriate optimization strategy",
        "stages": ["problem_framing", "strategy_selection", "solution_implementation", "integration", "reflection"],
        "workflow_file": "optimizer_combined.json"
    }
}

class OptimizationManager:
    def __init__(self):
        self.workflows_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_optimizations = {}
        self.optimization_results = {}
        
        # Load optimization workflows
        self.optimization_workflows = {}
        for pattern_id, pattern_info in OPTIMIZATION_PATTERNS.items():
            workflow_path = os.path.join(self.workflows_dir, pattern_info["workflow_file"])
            if os.path.exists(workflow_path):
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        self.optimization_workflows[pattern_id] = json.load(f)
                        logger.info(f"Loaded optimization workflow: {pattern_id}")
                except Exception as e:
                    logger.error(f"Error loading optimization workflow {pattern_id}: {str(e)}")
    
    def create_optimization_session(self, pattern_id: str, session_id: str = None, 
                                   target_description: str = None, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new optimization session using a specified pattern"""
        if pattern_id not in OPTIMIZATION_PATTERNS:
            return {"error": f"Unknown optimization pattern: {pattern_id}"}
        
        if pattern_id not in self.optimization_workflows:
            return {"error": f"Workflow for pattern {pattern_id} not loaded"}
        
        # Generate a session ID if not provided
        if session_id is None:
            session_id = f"{pattern_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create session data structure
        self.active_optimizations[session_id] = {
            "pattern": pattern_id,
            "workflow": copy.deepcopy(self.optimization_workflows[pattern_id]),
            "target_description": target_description,
            "initial_state": initial_state or {},
            "status": "created",
            "current_stage": 0,
            "results": {},
            "created_at": time.time()
        }
        
        # If target description is provided, add it to all agent prompts
        if target_description:
            for i, step in enumerate(self.active_optimizations[session_id]["workflow"]):
                if "content" in step and "{problem_description}" in step["content"]:
                    self.active_optimizations[session_id]["workflow"][i]["content"] = \
                        step["content"].replace("{problem_description}", target_description)
        
        return {
            "status": "success",
            "session_id": session_id,
            "pattern": pattern_id,
            "description": OPTIMIZATION_PATTERNS[pattern_id]["description"],
            "stages": OPTIMIZATION_PATTERNS[pattern_id]["stages"]
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the status of an optimization session"""
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        pattern_id = session["pattern"]
        
        return {
            "session_id": session_id,
            "pattern": pattern_id,
            "description": OPTIMIZATION_PATTERNS[pattern_id]["description"],
            "status": session["status"],
            "current_stage": session["current_stage"],
            "stages": OPTIMIZATION_PATTERNS[pattern_id]["stages"],
            "created_at": session["created_at"],
            "results_available": list(session["results"].keys())
        }
    
    def get_next_step(self, session_id: str) -> Dict[str, Any]:
        """Get the next workflow step for the optimization session"""
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        
        if session["status"] == "completed":
            return {"status": "completed", "message": "This optimization session has completed all stages"}
        
        current_stage = session["current_stage"]
        workflow = session["workflow"]
        
        if current_stage >= len(workflow):
            session["status"] = "completed"
            return {"status": "completed", "message": "No more steps in this optimization workflow"}
        
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
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        
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
        """Submit results for a step in the optimization workflow"""
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        
        # Store the results
        session["results"][agent_name] = result
        
        # Advance to the next stage
        session["current_stage"] += 1
        
        # Update status
        if session["current_stage"] >= len(session["workflow"]):
            session["status"] = "completed"
            # Store the final results
            self.optimization_results[session_id] = self._process_final_results(session)
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
        """Get all results from an optimization session"""
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        pattern_id = session["pattern"]
        
        # If the session is completed and we have processed results, return those
        if session["status"] == "completed" and session_id in self.optimization_results:
            results = self.optimization_results[session_id]
        else:
            # Otherwise return the raw results
            results = session["results"]
        
        response = {
            "session_id": session_id,
            "pattern": pattern_id,
            "description": OPTIMIZATION_PATTERNS[pattern_id]["description"],
            "status": session["status"],
            "results": results,
            "created_at": session["created_at"]
        }
        
        if include_workflow:
            response["workflow"] = session["workflow"]
        
        return response
    
    def apply_optimization_to_workflow(self, session_id: str, workflow_to_optimize: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply the results of an optimization session to a workflow"""
        if session_id not in self.active_optimizations:
            return {"error": f"Session not found: {session_id}"}
        
        session = self.active_optimizations[session_id]
        
        if session["status"] != "completed":
            return {"error": "Optimization session is not yet completed"}
        
        pattern_id = session["pattern"]
        optimization_results = session_id in self.optimization_results and self.optimization_results[session_id] or session["results"]
        
        try:
            # Get the original workflow
            original_workflow = copy.deepcopy(workflow_to_optimize)
            
            # Apply different optimization strategies based on the pattern
            if pattern_id == "multi_objective":
                optimized_workflow = self._apply_multi_objective_optimization(original_workflow, optimization_results)
            elif pattern_id == "performance_monitoring":
                optimized_workflow = self._apply_performance_monitoring(original_workflow, optimization_results)
            elif pattern_id == "feedback_loop":
                optimized_workflow = self._apply_feedback_loop_optimization(original_workflow, optimization_results)
            elif pattern_id == "adaptive_optimization":
                optimized_workflow = self._apply_adaptive_optimization(original_workflow, optimization_results)
            else:
                return {"error": f"Application method for pattern {pattern_id} not implemented"}
            
            return {
                "status": "success",
                "original_workflow": original_workflow,
                "optimized_workflow": optimized_workflow,
                "changes": self._summarize_workflow_changes(original_workflow, optimized_workflow),
                "optimization_pattern": pattern_id
            }
            
        except Exception as e:
            logger.error(f"Error applying optimization to workflow: {str(e)}")
            return {"error": f"Failed to apply optimization: {str(e)}"}
    
    def _process_final_results(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure the final results of an optimization session"""
        pattern_id = session["pattern"]
        raw_results = session["results"]
        
        # Process based on pattern type
        if pattern_id == "multi_objective":
            # For multi-objective, extract the optimized solution and framework
            processed_results = {
                "optimized_solution": raw_results.get("solution_reviser", {}),
                "optimization_framework": raw_results.get("reasoning_optimizer", {}).get("optimized_framework", {}),
                "identified_biases": raw_results.get("bias_detector", {}).get("cognitive_biases", []),
                "counterfactuals_explored": raw_results.get("counterfactual_explorer", {}).get("counterfactual_scenarios", [])
            }
        
        elif pattern_id == "performance_monitoring":
            # For performance monitoring, extract metrics, anomalies, and root causes
            processed_results = {
                "monitoring_framework": raw_results.get("performance_monitor", {}).get("monitoring_framework", {}),
                "collected_metrics": raw_results.get("metric_collector", {}).get("metric_collection", {}),
                "detected_anomalies": raw_results.get("anomaly_detector", {}).get("anomaly_analysis", {}),
                "root_causes": raw_results.get("root_cause_analyzer", {}).get("root_cause_analysis", {})
            }
        
        elif pattern_id == "feedback_loop":
            # For feedback loop, extract the feedback framework and integrated results
            processed_results = {
                "feedback_framework": raw_results.get("feedback_architect", {}).get("feedback_framework", {}),
                "micro_feedback": raw_results.get("micro_feedback_collector", {}).get("micro_feedback", {}),
                "meso_feedback": raw_results.get("meso_feedback_collector", {}).get("meso_feedback", {}),
                "macro_feedback": raw_results.get("macro_feedback_collector", {}).get("macro_feedback", {}),
                "integrated_insights": raw_results.get("feedback_integrator", {}).get("integrated_feedback", {})
            }
        
        elif pattern_id == "adaptive_optimization":
            # For adaptive optimization, extract selected approach and evaluation
            dynamic_action = None
            for key, value in raw_results.items():
                if key.endswith("_action") and isinstance(value, str):
                    dynamic_action = value
                    break
            
            processed_results = {
                "problem_framing": raw_results.get("problem_framer", {}),
                "selected_approach": dynamic_action,
                "solution": raw_results.get("solution_integrator", {}),
                "architecture_evaluation": raw_results.get("cognitive_architect", {}).get("architecture_effectiveness", {})
            }
        
        else:
            # Default fallback processing
            processed_results = raw_results
        
        return processed_results
    
    def _apply_multi_objective_optimization(self, workflow: List[Dict[str, Any]], 
                                          optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply multi-objective optimization to a workflow"""
        # Create a deep copy to avoid modifying the original
        optimized_workflow = copy.deepcopy(workflow)
        
        # Extract optimization framework and identified biases
        optimization_framework = optimization_results.get("optimization_framework", {})
        identified_biases = optimization_results.get("identified_biases", [])
        
        # Enhance each agent's prompts with the optimization framework steps
        framework_steps = optimization_framework.get("steps", [])
        bias_mitigation = optimization_framework.get("bias_mitigation_techniques", [])
        
        for i, step in enumerate(optimized_workflow):
            # Only modify content field if it exists
            if "content" in step:
                # Add reasoning guidance based on the framework
                step_guidance = ""
                for framework_step in framework_steps:
                    step_name = framework_step.get("step_name", "")
                    step_desc = framework_step.get("description", "")
                    if step_name and step_desc:
                        step_guidance += f"\n- Apply {step_name}: {step_desc}"
                
                # Add bias mitigation guidance
                bias_guidance = ""
                for technique in bias_mitigation:
                    bias_guidance += f"\n- {technique}"
                
                # Only add guidance if we have content to add
                if step_guidance or bias_guidance:
                    enhancement = "\n\nApply this optimized reasoning framework:"
                    if step_guidance:
                        enhancement += f"\n{step_guidance}"
                    if bias_guidance:
                        enhancement += f"\n\nMitigate potential biases using these techniques:{bias_guidance}"
                    
                    optimized_workflow[i]["content"] += enhancement
        
        return optimized_workflow
    
    def _apply_performance_monitoring(self, workflow: List[Dict[str, Any]], 
                                    optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply performance monitoring optimization to a workflow"""
        # Create a deep copy to avoid modifying the original
        optimized_workflow = copy.deepcopy(workflow)
        
        # Extract monitoring framework and collected metrics
        monitoring_framework = optimization_results.get("monitoring_framework", {})
        detected_anomalies = optimization_results.get("detected_anomalies", {}).get("detected_anomalies", [])
        root_causes = optimization_results.get("root_causes", {}).get("system_weaknesses", [])
        
        # Add performance monitoring hooks to the workflow
        # Create a monitoring agent at the start of the workflow
        monitoring_agent = {
            "agent": "performance_monitor_agent",
            "content": "Monitor the execution of this workflow using the defined metrics framework. Track key performance indicators and alert on anomalies.",
            "tools": ["planning:chain_of_thought"],
            "output_format": {
                "type": "json",
                "schema": {
                    "monitoring_status": "string",
                    "metrics_tracked": ["string"],
                    "alerts": ["string"]
                }
            }
        }
        
        # Insert at the beginning
        optimized_workflow.insert(0, monitoring_agent)
        
        # Add a metrics collection hook after each significant agent
        for i, step in enumerate(optimized_workflow):
            # Skip the monitoring agent we just added
            if i == 0:
                continue
                
            # Add monitoring hooks every few steps
            if i % 3 == 0:
                # Create a metrics collection hook
                metrics_hook = {
                    "agent": f"metrics_collector_{i}",
                    "content": "Collect performance metrics for the preceding workflow steps. Analyze for any anomalies or performance degradation.",
                    "readFrom": ["*"],  # Read from all previous agents
                    "tools": ["planning:chain_of_thought"],
                    "output_format": {
                        "type": "json",
                        "schema": {
                            "metrics_collected": ["string"],
                            "performance_assessment": "string",
                            "optimization_suggestions": ["string"]
                        }
                    }
                }
                
                # Insert after the current step
                optimized_workflow.insert(i + 1, metrics_hook)
        
        # Add a final analysis agent at the end
        final_analysis = {
            "agent": "performance_analysis_agent",
            "content": "Analyze the overall workflow performance. Identify bottlenecks, inefficiencies, and opportunities for optimization.",
            "readFrom": ["*"],  # Read from all previous agents
            "tools": ["planning:chain_of_thought"],
            "output_format": {
                "type": "json",
                "schema": {
                    "performance_summary": "string",
                    "bottlenecks_identified": ["string"],
                    "optimization_recommendations": ["string"]
                }
            }
        }
        
        optimized_workflow.append(final_analysis)
        
        return optimized_workflow
    
    def _apply_feedback_loop_optimization(self, workflow: List[Dict[str, Any]], 
                                        optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply feedback loop optimization to a workflow"""
        # Create a deep copy to avoid modifying the original
        optimized_workflow = copy.deepcopy(workflow)
        
        # Extract feedback framework and results
        feedback_framework = optimization_results.get("feedback_framework", {})
        
        # Add feedback collection agents at the micro, meso, and macro levels
        
        # Micro feedback collector (after each agent)
        for i, step in enumerate(optimized_workflow):
            # Enhance each agent with feedback collection directives
            if "content" in step:
                step["content"] += "\n\nProvide detailed feedback on your process, challenges encountered, and quality of inputs received."
            
            # Add explicit feedback tags to the outputs where applicable
            if "output_format" in step and step["output_format"].get("type") == "json":
                schema = step["output_format"].get("schema", {})
                
                # Add feedback fields to the schema if they don't exist
                if "process_feedback" not in schema:
                    schema["process_feedback"] = {
                        "challenges": ["string"],
                        "input_quality": "number",
                        "self_assessment": "string"
                    }
                
                step["output_format"]["schema"] = schema
        
        # Meso feedback collector (after logical workflow segments)
        segment_breaks = max(1, len(optimized_workflow) // 3)  # Create ~3 segments
        
        for i in range(1, 3):
            position = i * segment_breaks
            if position < len(optimized_workflow):
                meso_collector = {
                    "agent": f"meso_feedback_collector_{i}",
                    "content": "Analyze the interaction patterns and workflow efficiency of the preceding segment. Identify collaboration issues and workflow bottlenecks.",
                    "readFrom": ["*"],  # Read from all previous agents
                    "tools": ["planning:chain_of_thought"],
                    "output_format": {
                        "type": "json",
                        "schema": {
                            "segment_assessment": "string",
                            "interaction_quality": "number",
                            "workflow_efficiency": "number",
                            "improvement_suggestions": ["string"]
                        }
                    }
                }
                
                optimized_workflow.insert(position, meso_collector)
        
        # Macro feedback collector (at the end)
        macro_collector = {
            "agent": "macro_feedback_integrator",
            "content": "Synthesize feedback from all levels of the workflow. Evaluate overall system performance, outcome quality, and user value. Provide strategic recommendations for workflow optimization.",
            "readFrom": ["*"],  # Read from all previous agents
            "tools": ["planning:chain_of_thought"],
            "output_format": {
                "type": "json",
                "schema": {
                    "overall_assessment": "string",
                    "system_level_insights": ["string"],
                    "strategic_recommendations": ["string"],
                    "future_improvements": ["string"]
                }
            }
        }
        
        optimized_workflow.append(macro_collector)
        
        return optimized_workflow
    
    def _apply_adaptive_optimization(self, workflow: List[Dict[str, Any]], 
                                   optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply adaptive optimization to a workflow"""
        # Create a deep copy to avoid modifying the original
        optimized_workflow = copy.deepcopy(workflow)
        
        # Extract selected approach and architecture evaluation
        selected_approach = optimization_results.get("selected_approach", "")
        architecture_evaluation = optimization_results.get("architecture_evaluation", {})
        
        # Create an adaptive workflow that can dynamically select techniques
        
        # Add an initial planning agent
        planning_agent = {
            "agent": "adaptive_optimization_planner",
            "content": f"Analyze this workflow and prepare an adaptive optimization strategy. Use the approaches identified as most effective from previous optimizations: {selected_approach}.",
            "tools": ["planning:create_plan"],
            "output_format": {
                "type": "json",
                "schema": {
                    "workflow_analysis": "string",
                    "optimization_strategy": {
                        "approach": "string",
                        "key_focus_areas": ["string"],
                        "expected_benefits": ["string"]
                    },
                    "execution_plan": ["string"]
                }
            }
        }
        
        # Insert at the beginning
        optimized_workflow.insert(0, planning_agent)
        
        # Add dynamic optimization agents at strategic points
        dynamic_agent = {
            "agent": "dynamic_optimizer",
            "type": "dynamic",
            "initial_prompt": "Based on the workflow execution so far, determine which optimization technique would be most beneficial to apply: 'performance_monitoring', 'feedback_collection', 'bias_mitigation', or 'process_restructuring'.",
            "readFrom": ["*"],  # Read from all previous agents
            "tools": ["planning:chain_of_thought"],
            "output_format": {
                "type": "json",
                "schema": {
                    "selected_technique": "string",
                    "justification": "string",
                    "expected_impact": "string"
                }
            },
            "actions": {
                "performance_monitoring": {
                    "agent": "performance_optimizer",
                    "content": "Implement performance monitoring for the subsequent workflow stages. Define metrics, establish baselines, and provide real-time feedback on execution efficiency.",
                    "readFrom": ["*"],
                    "tools": ["planning:chain_of_thought"]
                },
                "feedback_collection": {
                    "agent": "feedback_optimizer",
                    "content": "Implement feedback collection mechanisms for the subsequent workflow stages. Establish channels for agent-to-agent feedback and process improvement suggestions.",
                    "readFrom": ["*"],
                    "tools": ["planning:chain_of_thought"]
                },
                "bias_mitigation": {
                    "agent": "bias_optimizer",
                    "content": "Implement bias mitigation strategies for the subsequent workflow stages. Identify potential cognitive biases and provide techniques to counteract them.",
                    "readFrom": ["*"],
                    "tools": ["planning:chain_of_thought"]
                },
                "process_restructuring": {
                    "agent": "process_optimizer",
                    "content": "Implement process restructuring for the subsequent workflow stages. Analyze the workflow structure and suggest more efficient patterns of agent interaction.",
                    "readFrom": ["*"],
                    "tools": ["planning:chain_of_thought"]
                }
            }
        }
        
        # Insert at approximately the 1/3 point
        third_point = max(2, len(optimized_workflow) // 3)
        optimized_workflow.insert(third_point, copy.deepcopy(dynamic_agent))
        
        # Insert again at approximately the 2/3 point
        two_thirds_point = max(third_point + 2, 2 * len(optimized_workflow) // 3)
        optimized_workflow.insert(two_thirds_point, copy.deepcopy(dynamic_agent))
        
        # Add a final optimization assessment agent
        final_assessment = {
            "agent": "optimization_effectiveness_evaluator",
            "content": "Evaluate the effectiveness of the optimization techniques applied throughout this workflow. Identify which approaches were most impactful and make recommendations for future workflows.",
            "readFrom": ["*"],  # Read from all previous agents
            "tools": ["planning:chain_of_thought"],
            "output_format": {
                "type": "json",
                "schema": {
                    "techniques_evaluated": ["string"],
                    "effectiveness_ranking": ["string"],
                    "lessons_learned": ["string"],
                    "future_recommendations": ["string"]
                }
            }
        }
        
        optimized_workflow.append(final_assessment)
        
        return optimized_workflow
    
    def _summarize_workflow_changes(self, original: List[Dict[str, Any]], 
                                  optimized: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the changes made to a workflow"""
        original_agents = [step.get("agent", f"agent_{i}") for i, step in enumerate(original)]
        optimized_agents = [step.get("agent", f"agent_{i}") for i, step in enumerate(optimized)]
        
        added_agents = [agent for agent in optimized_agents if agent not in original_agents]
        removed_agents = [agent for agent in original_agents if agent not in optimized_agents]
        
        original_length = len(original)
        optimized_length = len(optimized)
        
        return {
            "original_workflow_length": original_length,
            "optimized_workflow_length": optimized_length,
            "added_agents": added_agents,
            "removed_agents": removed_agents,
            "net_change": optimized_length - original_length
        }

# Global manager instance
OPTIMIZATION_MANAGER = OptimizationManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def optimization_create_session(pattern_id: str, session_id: str = None, 
                              target_description: str = None, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new optimization session using a specified pattern.
    
    Args:
        pattern_id: The optimization pattern to use ('multi_objective', 'performance_monitoring', etc.)
        session_id: Optional custom identifier for the session
        target_description: Description of the target to optimize
        initial_state: Initial state or configuration for the optimization
    
    Returns:
        Dict with session information
    """
    return OPTIMIZATION_MANAGER.create_optimization_session(pattern_id, session_id, target_description, initial_state)

def optimization_get_status(session_id: str) -> Dict[str, Any]:
    """
    Get status and information about an optimization session.
    
    Args:
        session_id: Identifier for the session
    
    Returns:
        Dict with session status and information
    """
    return OPTIMIZATION_MANAGER.get_session_status(session_id)

def optimization_get_next_step(session_id: str) -> Dict[str, Any]:
    """
    Get the next workflow step for an optimization session.
    
    Args:
        session_id: Identifier for the session
    
    Returns:
        Dict with information about the next step
    """
    return OPTIMIZATION_MANAGER.get_next_step(session_id)

def optimization_select_action(session_id: str, action: str) -> Dict[str, Any]:
    """
    Select an action for a dynamic agent in the optimization workflow.
    
    Args:
        session_id: Identifier for the session
        action: The action to select
    
    Returns:
        Dict with action information
    """
    return OPTIMIZATION_MANAGER.select_dynamic_action(session_id, action)

def optimization_submit_result(session_id: str, agent_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit results for a step in the optimization workflow.
    
    Args:
        session_id: Identifier for the session
        agent_name: Name of the agent that produced the result
        result: The results data
    
    Returns:
        Dict with submission status and next step information
    """
    return OPTIMIZATION_MANAGER.submit_step_result(session_id, agent_name, result)

def optimization_get_results(session_id: str, include_workflow: bool = False) -> Dict[str, Any]:
    """
    Get all results from an optimization session.
    
    Args:
        session_id: Identifier for the session
        include_workflow: Whether to include the full workflow in the response
    
    Returns:
        Dict with session results
    """
    return OPTIMIZATION_MANAGER.get_session_results(session_id, include_workflow)

def optimization_apply_to_workflow(session_id: str, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply the results of an optimization session to a workflow.
    
    Args:
        session_id: Identifier for the session
        workflow: The workflow to optimize
    
    Returns:
        Dict with original and optimized workflows
    """
    return OPTIMIZATION_MANAGER.apply_optimization_to_workflow(session_id, workflow)

def optimization_list_patterns() -> Dict[str, Any]:
    """
    List all available optimization patterns.
    
    Returns:
        Dict with available patterns and their descriptions
    """
    patterns = {}
    for pattern_id, pattern_info in OPTIMIZATION_PATTERNS.items():
        patterns[pattern_id] = {
            "description": pattern_info["description"],
            "stages": pattern_info["stages"],
            "available": pattern_id in OPTIMIZATION_MANAGER.optimization_workflows
        }
    
    return {
        "status": "success",
        "patterns": patterns
    }

# Register tools
TOOL_REGISTRY["optimization:create_session"] = optimization_create_session
TOOL_REGISTRY["optimization:get_status"] = optimization_get_status
TOOL_REGISTRY["optimization:get_next_step"] = optimization_get_next_step
TOOL_REGISTRY["optimization:select_action"] = optimization_select_action
TOOL_REGISTRY["optimization:submit_result"] = optimization_submit_result
TOOL_REGISTRY["optimization:get_results"] = optimization_get_results
TOOL_REGISTRY["optimization:apply_to_workflow"] = optimization_apply_to_workflow
TOOL_REGISTRY["optimization:list_patterns"] = optimization_list_patterns

# Print initialization message
print("✅ Optimization tools registered successfully")
