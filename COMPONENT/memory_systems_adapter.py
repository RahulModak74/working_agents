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
logger = logging.getLogger("memory_systems_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Memory system patterns
MEMORY_PATTERNS = {
    "synchronization": {
        "description": "Cross-agent memory synchronization protocol with publishers, subscribers, and consistency verification",
        "stages": ["protocol_design", "publishing", "subscribing", "consistency_verification"],
        "workflow_file": "memory_synchronization.json"
    },
    "hierarchical": {
        "description": "Context-aware multi-tier memory hierarchy with promotion/demotion based on relevance",
        "stages": ["hierarchy_design", "working_memory", "context_detection", "tier_management"],
        "workflow_file": "memory_hirearchial_storage.json"
    },
    "federated_graph": {
        "description": "Distributed knowledge graph with federated nodes specialized for different domains",
        "stages": ["graph_initialization", "knowledge_extraction", "federation_management"],
        "workflow_file": "memory_federated_graph.json"
    },
    "integrated_memory": {
        "description": "Comprehensive memory system combining federation, hierarchy, and synchronization",
        "stages": ["architecture_design", "component_initialization", "integration", "control_system", "testing"],
        "workflow_file": "memory_combined.json"
    }
}

class MemorySystemManager:
    def __init__(self):
        self.workflows_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_systems = {}
        self.memory_stores = {}
        
        # Load memory system workflows
        self.memory_workflows = {}
        for pattern_id, pattern_info in MEMORY_PATTERNS.items():
            workflow_path = os.path.join(self.workflows_dir, pattern_info["workflow_file"])
            if os.path.exists(workflow_path):
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        self.memory_workflows[pattern_id] = json.load(f)
                        logger.info(f"Loaded memory system workflow: {pattern_id}")
                except Exception as e:
                    logger.error(f"Error loading memory system workflow {pattern_id}: {str(e)}")
    
    def create_memory_system(self, pattern_id: str, system_id: str = None, 
                           domain_description: str = None, initial_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new memory system using a specified pattern"""
        if pattern_id not in MEMORY_PATTERNS:
            return {"error": f"Unknown memory pattern: {pattern_id}"}
        
        if pattern_id not in self.memory_workflows:
            return {"error": f"Workflow for pattern {pattern_id} not loaded"}
        
        # Generate a system ID if not provided
        if system_id is None:
            system_id = f"{pattern_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create memory system data structure
        self.active_systems[system_id] = {
            "pattern": pattern_id,
            "workflow": copy.deepcopy(self.memory_workflows[pattern_id]),
            "domain_description": domain_description,
            "configuration": initial_config or {},
            "status": "created",
            "current_stage": 0,
            "results": {},
            "memory_store": {},
            "created_at": time.time()
        }
        
        # Create an associated memory store
        self.memory_stores[system_id] = {}
        
        # If domain description is provided, add it to relevant agent prompts
        if domain_description:
            for i, step in enumerate(self.active_systems[system_id]["workflow"]):
                if "content" in step:
                    content = step["content"]
                    if "{domain_topic}" in content:
                        content = content.replace("{domain_topic}", domain_description)
                    if "{application_domain}" in content:
                        content = content.replace("{application_domain}", domain_description)
                    self.active_systems[system_id]["workflow"][i]["content"] = content
        
        return {
            "status": "success",
            "system_id": system_id,
            "pattern": pattern_id,
            "description": MEMORY_PATTERNS[pattern_id]["description"],
            "stages": MEMORY_PATTERNS[pattern_id]["stages"]
        }
    
    def get_system_status(self, system_id: str) -> Dict[str, Any]:
        """Get the status of a memory system"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        pattern_id = system["pattern"]
        
        return {
            "system_id": system_id,
            "pattern": pattern_id,
            "description": MEMORY_PATTERNS[pattern_id]["description"],
            "status": system["status"],
            "current_stage": system["current_stage"],
            "stages": MEMORY_PATTERNS[pattern_id]["stages"],
            "created_at": system["created_at"],
            "memory_size": len(system["memory_store"]) if "memory_store" in system else 0,
            "results_available": list(system["results"].keys())
        }
    
    def get_next_step(self, system_id: str) -> Dict[str, Any]:
        """Get the next workflow step for the memory system"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        
        if system["status"] == "completed":
            return {"status": "completed", "message": "This memory system has completed all stages"}
        
        current_stage = system["current_stage"]
        workflow = system["workflow"]
        
        if current_stage >= len(workflow):
            system["status"] = "completed"
            return {"status": "completed", "message": "No more steps in this memory system workflow"}
        
        next_step = workflow[current_stage]
        
        # Check for dynamic agent that needs to be resolved
        if next_step.get("type") == "dynamic" and system["status"] != "awaiting_action_selection":
            # Mark that we're waiting for an action selection
            system["status"] = "awaiting_action_selection"
            
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
    
    def select_dynamic_action(self, system_id: str, action: str) -> Dict[str, Any]:
        """Select an action for a dynamic agent in the workflow"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        
        if system["status"] != "awaiting_action_selection":
            return {"error": "Current step is not awaiting action selection"}
        
        current_stage = system["current_stage"]
        workflow = system["workflow"]
        
        if current_stage >= len(workflow):
            return {"error": "No current step available"}
        
        current_step = workflow[current_stage]
        
        if current_step.get("type") != "dynamic":
            return {"error": "Current step is not a dynamic agent"}
        
        actions = current_step.get("actions", {})
        
        if action not in actions:
            return {"error": f"Invalid action: {action}. Available actions: {list(actions.keys())}"}
        
        # Store the selected action
        system["results"][f"{current_step.get('agent', f'agent_{current_stage}')}_action"] = action
        
        # Update status
        system["status"] = "in_progress"
        
        # Return the action details
        return {
            "status": "success",
            "action": action,
            "action_details": actions[action]
        }
    
    def submit_step_result(self, system_id: str, agent_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Submit results for a step in the memory system workflow"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        
        # Store the results
        system["results"][agent_name] = result
        
        # Update the memory store based on the agent type and result
        pattern_id = system["pattern"]
        self._update_memory_store(system_id, agent_name, result, pattern_id)
        
        # Advance to the next stage
        system["current_stage"] += 1
        
        # Update status
        if system["current_stage"] >= len(system["workflow"]):
            system["status"] = "completed"
        else:
            system["status"] = "in_progress"
        
        # Get information about the next step
        next_step_info = self.get_next_step(system_id)
        
        return {
            "status": "success",
            "message": f"Results for {agent_name} submitted successfully",
            "system_status": system["status"],
            "next_step": next_step_info
        }
    
    def get_system_results(self, system_id: str, include_workflow: bool = False) -> Dict[str, Any]:
        """Get all results from a memory system"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        pattern_id = system["pattern"]
        
        # Prepare a structured result
        structured_results = self._process_system_results(system)
        
        response = {
            "system_id": system_id,
            "pattern": pattern_id,
            "description": MEMORY_PATTERNS[pattern_id]["description"],
            "status": system["status"],
            "results": structured_results,
            "created_at": system["created_at"]
        }
        
        if include_workflow:
            response["workflow"] = system["workflow"]
        
        return response
    
    def memory_store_operation(self, system_id: str, operation: str, key: str = None, 
                             value: Any = None, query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform an operation on the memory store"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        if system_id not in self.memory_stores:
            self.memory_stores[system_id] = {}
        
        memory_store = self.memory_stores[system_id]
        
        if operation == "get" and key:
            if key in memory_store:
                return {"status": "success", "key": key, "value": memory_store[key]}
            else:
                return {"status": "error", "message": f"Key not found: {key}"}
        
        elif operation == "put" and key and value is not None:
            memory_store[key] = value
            return {"status": "success", "key": key}
        
        elif operation == "delete" and key:
            if key in memory_store:
                del memory_store[key]
                return {"status": "success", "key": key}
            else:
                return {"status": "error", "message": f"Key not found: {key}"}
        
        elif operation == "list":
            return {"status": "success", "keys": list(memory_store.keys())}
        
        elif operation == "query" and query:
            # Simple query implementation
            results = {}
            for k, v in memory_store.items():
                match = True
                for qk, qv in query.items():
                    if isinstance(v, dict) and qk in v:
                        if v[qk] != qv:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    results[k] = v
            
            return {"status": "success", "results": results}
        
        else:
            return {"status": "error", "message": f"Invalid operation: {operation}"}
    
    def integrate_with_research(self, system_id: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate memory system with research data"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        system = self.active_systems[system_id]
        pattern_id = system["pattern"]
        
        try:
            # Extract key information from research data
            entities = []
            relations = []
            concepts = []
            
            # Extract entities
            if "key_concepts" in research_data:
                for concept in research_data["key_concepts"]:
                    concepts.append({
                        "concept": concept,
                        "type": "concept",
                        "source": "research_data"
                    })
            
            # Extract potentially more complex entities
            if "search_results" in research_data:
                for result in research_data["search_results"]:
                    if isinstance(result, dict):
                        entity = {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "description": result.get("snippet", ""),
                            "type": "source",
                            "source": "search_results"
                        }
                        entities.append(entity)
            
            # Create relationships between entities and concepts
            for entity in entities:
                for concept in concepts:
                    # Simple relevance check (this could be more sophisticated)
                    if concept["concept"].lower() in entity.get("title", "").lower() or \
                       concept["concept"].lower() in entity.get("description", "").lower():
                        relation = {
                            "source": entity.get("title", ""),
                            "relation": "mentions",
                            "target": concept["concept"],
                            "confidence": 0.8
                        }
                        relations.append(relation)
            
            # Store in memory system
            if pattern_id == "federated_graph":
                # For federated graph, store as graph nodes and edges
                graph_data = {
                    "entities": entities,
                    "relations": relations,
                    "concepts": concepts
                }
                self.memory_stores[system_id]["research_graph"] = graph_data
                
            elif pattern_id == "hierarchical":
                # For hierarchical memory, store in appropriate tiers
                hierarchy_data = {
                    "working_memory": concepts,  # More immediate relevance
                    "long_term_memory": entities,  # More detailed but less immediately relevant
                    "relations": relations
                }
                self.memory_stores[system_id]["research_hierarchy"] = hierarchy_data
                
            elif pattern_id == "synchronization":
                # For sync systems, create a publication
                sync_data = {
                    "publication_id": f"research_{int(time.time())}",
                    "timestamp": time.time(),
                    "originating_agent": "research_integration",
                    "memory_elements": concepts + entities,
                    "relations": relations
                }
                self.memory_stores[system_id]["research_sync"] = sync_data
                
            else:
                # For integrated systems, use a comprehensive approach
                self.memory_stores[system_id]["research_data"] = {
                    "entities": entities,
                    "relations": relations,
                    "concepts": concepts,
                    "raw_data": research_data
                }
            
            return {
                "status": "success",
                "message": f"Research data integrated with {pattern_id} memory system",
                "entities_added": len(entities),
                "concepts_added": len(concepts),
                "relations_added": len(relations)
            }
            
        except Exception as e:
            logger.error(f"Error integrating research data: {str(e)}")
            return {"error": f"Failed to integrate research data: {str(e)}"}
    
    def extract_research_recommendations(self, system_id: str) -> Dict[str, Any]:
        """Extract research recommendations based on memory system"""
        if system_id not in self.active_systems:
            return {"error": f"System not found: {system_id}"}
        
        if system_id not in self.memory_stores:
            return {"error": f"No memory store for system: {system_id}"}
        
        system = self.active_systems[system_id]
        pattern_id = system["pattern"]
        memory_store = self.memory_stores[system_id]
        
        # Default recommendations
        recommendations = {
            "research_gaps": [],
            "promising_areas": [],
            "knowledge_clusters": [],
            "suggested_queries": []
        }
        
        try:
            # Extract recommendations based on pattern type
            if pattern_id == "federated_graph" and "research_graph" in memory_store:
                graph_data = memory_store["research_graph"]
                
                # Identify concepts with few relations (potential gaps)
                concept_relations = {}
                for relation in graph_data.get("relations", []):
                    target = relation.get("target", "")
                    if target:
                        concept_relations[target] = concept_relations.get(target, 0) + 1
                
                # Concepts with few connections might be research gaps
                for concept in graph_data.get("concepts", []):
                    concept_name = concept.get("concept", "")
                    if concept_name in concept_relations:
                        if concept_relations[concept_name] <= 1:
                            recommendations["research_gaps"].append(concept_name)
                    else:
                        recommendations["research_gaps"].append(concept_name)
                
                # Concepts with many connections might be promising areas
                for concept, count in concept_relations.items():
                    if count >= 3:
                        recommendations["promising_areas"].append(concept)
                
                # Generate suggested queries
                for gap in recommendations["research_gaps"][:3]:
                    recommendations["suggested_queries"].append(f"detailed analysis of {gap}")
                
            elif pattern_id == "hierarchical" and "research_hierarchy" in memory_store:
                hierarchy_data = memory_store["research_hierarchy"]
                
                # Working memory concepts are likely promising areas
                for concept in hierarchy_data.get("working_memory", []):
                    recommendations["promising_areas"].append(concept.get("concept", ""))
                
                # Find clusters in the relations
                relation_clusters = {}
                for relation in hierarchy_data.get("relations", []):
                    target = relation.get("target", "")
                    if target:
                        if target not in relation_clusters:
                            relation_clusters[target] = []
                        relation_clusters[target].append(relation.get("source", ""))
                
                # Identify knowledge clusters
                for target, sources in relation_clusters.items():
                    if len(sources) >= 2:
                        recommendations["knowledge_clusters"].append({
                            "central_concept": target,
                            "related_sources": sources
                        })
                
                # Generate suggested queries for sparsely connected areas
                working_memory_concepts = [c.get("concept", "") for c in hierarchy_data.get("working_memory", [])]
                for concept in working_memory_concepts:
                    if concept not in relation_clusters or len(relation_clusters[concept]) <= 1:
                        recommendations["suggested_queries"].append(f"expanding knowledge on {concept}")
                
            elif pattern_id == "synchronization" and "research_sync" in memory_store:
                sync_data = memory_store["research_sync"]
                
                # Extract concepts from memory elements
                concepts = []
                for element in sync_data.get("memory_elements", []):
                    if isinstance(element, dict) and element.get("type") == "concept":
                        concepts.append(element.get("concept", ""))
                
                # Generate recommendations
                recommendations["promising_areas"] = concepts[:5]  # Top concepts
                recommendations["suggested_queries"] = [f"latest developments in {concept}" for concept in concepts[:3]]
                
            else:
                # General approach for other patterns or integrated systems
                # Extract concepts from any research data
                if "research_data" in memory_store:
                    research_data = memory_store["research_data"]
                    
                    # Extract concepts
                    for concept in research_data.get("concepts", []):
                        concept_name = concept.get("concept", "")
                        if concept_name:
                            recommendations["promising_areas"].append(concept_name)
                    
                    # Generate suggested queries
                    recommendations["suggested_queries"] = [
                        f"detailed analysis of {area}" for area in recommendations["promising_areas"][:3]
                    ]
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "memory_system": pattern_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting research recommendations: {str(e)}")
            return {
                "error": f"Failed to extract research recommendations: {str(e)}",
                "partial_recommendations": recommendations
            }
    
    def _update_memory_store(self, system_id: str, agent_name: str, result: Dict[str, Any], pattern_id: str) -> None:
        """Update the memory store based on agent results"""
        if system_id not in self.memory_stores:
            self.memory_stores[system_id] = {}
        
        # Store the raw result
        memory_key = f"agent_result_{agent_name}"
        self.memory_stores[system_id][memory_key] = result
        
        # Now handle specific memory operations based on agent and pattern
        
        # Federated Graph Pattern
        if pattern_id == "federated_graph":
            if agent_name == "knowledge_graph_initializer" and "knowledge_graph_schema" in result:
                self.memory_stores[system_id]["graph_schema"] = result["knowledge_graph_schema"]
            
            elif agent_name == "domain_knowledge_extractor":
                if "extracted_entities" in result:
                    if "entities" not in self.memory_stores[system_id]:
                        self.memory_stores[system_id]["entities"] = []
                    self.memory_stores[system_id]["entities"].extend(result["extracted_entities"])
                
                if "extracted_relations" in result:
                    if "relations" not in self.memory_stores[system_id]:
                        self.memory_stores[system_id]["relations"] = []
                    self.memory_stores[system_id]["relations"].extend(result["extracted_relations"])
        
        # Hierarchical Pattern
        elif pattern_id == "hierarchical":
            if agent_name == "memory_hierarchy_architect" and "memory_hierarchy" in result:
                self.memory_stores[system_id]["hierarchy_definition"] = result["memory_hierarchy"]
            
            elif agent_name == "working_memory_manager" and "working_memory_state" in result:
                self.memory_stores[system_id]["working_memory"] = result["working_memory_state"]
            
            elif agent_name == "context_detector" and "context_analysis" in result:
                self.memory_stores[system_id]["current_context"] = result["context_analysis"]
            
            elif agent_name == "memory_promoter_demoter" and "memory_operations" in result:
                if "memory_operations_history" not in self.memory_stores[system_id]:
                    self.memory_stores[system_id]["memory_operations_history"] = []
                self.memory_stores[system_id]["memory_operations_history"].append(result["memory_operations"])
        
        # Synchronization Pattern
        elif pattern_id == "synchronization":
            if agent_name == "memory_sync_coordinator" and "sync_protocol" in result:
                self.memory_stores[system_id]["sync_protocol"] = result["sync_protocol"]
            
            elif agent_name == "memory_publisher" and "memory_publication" in result:
                if "publications" not in self.memory_stores[system_id]:
                    self.memory_stores[system_id]["publications"] = []
                self.memory_stores[system_id]["publications"].append(result["memory_publication"])
            
            elif agent_name == "memory_subscriber" and "subscription_processing" in result:
                if "subscriptions" not in self.memory_stores[system_id]:
                    self.memory_stores[system_id]["subscriptions"] = []
                self.memory_stores[system_id]["subscriptions"].append(result["subscription_processing"])
            
            elif agent_name == "consistency_verifier" and "consistency_verification" in result:
                if "verifications" not in self.memory_stores[system_id]:
                    self.memory_stores[system_id]["verifications"] = []
                self.memory_stores[system_id]["verifications"].append(result["consistency_verification"])
        
        # Integrated Memory Pattern
        elif pattern_id == "integrated_memory":
            if agent_name == "distributed_memory_architect" and "architecture_design" in result:
                self.memory_stores[system_id]["architecture"] = result["architecture_design"]
            
            elif agent_name == "memory_integrator" and "integration_results" in result:
                self.memory_stores[system_id]["integration"] = result["integration_results"]
            
            elif agent_name == "memory_system_controller" and "control_system" in result:
                self.memory_stores[system_id]["control_system"] = result["control_system"]
    
    def _process_system_results(self, system: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure the results of a memory system"""
        pattern_id = system["pattern"]
        raw_results = system["results"]
        
        # Process based on pattern type
        if pattern_id == "federated_graph":
            return {
                "knowledge_graph_schema": raw_results.get("knowledge_graph_initializer", {}).get("knowledge_graph_schema", {}),
                "extracted_knowledge": {
                    "entities": raw_results.get("domain_knowledge_extractor", {}).get("extracted_entities", []),
                    "relations": raw_results.get("domain_knowledge_extractor", {}).get("extracted_relations", [])
                },
                "federation_status": raw_results.get("federation_manager", {}).get("federation_status", {})
            }
        
        elif pattern_id == "hierarchical":
            return {
                "memory_hierarchy": raw_results.get("memory_hierarchy_architect", {}).get("memory_hierarchy", {}),
                "working_memory": raw_results.get("working_memory_manager", {}).get("working_memory_state", {}),
                "current_context": raw_results.get("context_detector", {}).get("context_analysis", {}),
                "memory_operations": raw_results.get("memory_promoter_demoter", {}).get("memory_operations", {})
            }
        
        elif pattern_id == "synchronization":
            return {
                "sync_protocol": raw_results.get("memory_sync_coordinator", {}).get("sync_protocol", {}),
                "publications": raw_results.get("memory_publisher", {}).get("memory_publication", {}),
                "subscriptions": raw_results.get("memory_subscriber", {}).get("subscription_processing", {}),
                "consistency": raw_results.get("consistency_verifier", {}).get("consistency_verification", {})
            }
        
        elif pattern_id == "integrated_memory":
            # Extract dynamic action if present
            dynamic_action = None
            for key, value in raw_results.items():
                if key.endswith("_action") and isinstance(value, str):
                    dynamic_action = value
                    break
            
            return {
                "architecture": raw_results.get("distributed_memory_architect", {}).get("architecture_design", {}),
                "initialization": dynamic_action,
                "integration": raw_results.get("memory_integrator", {}).get("integration_results", {}),
                "control_system": raw_results.get("memory_system_controller", {}).get("control_system", {}),
                "test_results": raw_results.get("memory_system_tester", {})
            }
        
        else:
            # Default fallback processing
            return raw_results

# Global manager instance
MEMORY_SYSTEM_MANAGER = MemorySystemManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def memory_create_system(pattern_id: str, system_id: str = None, 
                       domain_description: str = None, initial_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new memory system using a specified pattern.
    
    Args:
        pattern_id: The memory pattern to use ('synchronization', 'hierarchical', etc.)
        system_id: Optional custom identifier for the system
        domain_description: Description of the domain for this memory system
        initial_config: Initial configuration settings
    
    Returns:
        Dict with system information
    """
    return MEMORY_SYSTEM_MANAGER.create_memory_system(pattern_id, system_id, domain_description, initial_config)

def memory_get_status(system_id: str) -> Dict[str, Any]:
    """
    Get status and information about a memory system.
    
    Args:
        system_id: Identifier for the system
    
    Returns:
        Dict with system status and information
    """
    return MEMORY_SYSTEM_MANAGER.get_system_status(system_id)

def memory_get_next_step(system_id: str) -> Dict[str, Any]:
    """
    Get the next workflow step for a memory system.
    
    Args:
        system_id: Identifier for the system
    
    Returns:
        Dict with information about the next step
    """
    return MEMORY_SYSTEM_MANAGER.get_next_step(system_id)

def memory_select_action(system_id: str, action: str) -> Dict[str, Any]:
    """
    Select an action for a dynamic agent in the workflow.
    
    Args:
        system_id: Identifier for the system
        action: The action to select
    
    Returns:
        Dict with action information
    """
    return MEMORY_SYSTEM_MANAGER.select_dynamic_action(system_id, action)

def memory_submit_result(system_id: str, agent_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit results for a step in the memory system workflow.
    
    Args:
        system_id: Identifier for the system
        agent_name: Name of the agent that produced the result
        result: The results data
    
    Returns:
        Dict with submission status and next step information
    """
    return MEMORY_SYSTEM_MANAGER.submit_step_result(system_id, agent_name, result)

def memory_get_results(system_id: str, include_workflow: bool = False) -> Dict[str, Any]:
    """
    Get all results from a memory system.
    
    Args:
        system_id: Identifier for the system
        include_workflow: Whether to include the full workflow in the response
    
    Returns:
        Dict with system results
    """
    return MEMORY_SYSTEM_MANAGER.get_system_results(system_id, include_workflow)

def memory_store_operation(system_id: str, operation: str, key: str = None, 
                         value: Any = None, query: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform an operation on the memory store.
    
    Args:
        system_id: Identifier for the system
        operation: Operation to perform ('get', 'put', 'delete', 'list', 'query')
        key: Key for the operation (for get, put, delete)
        value: Value to store (for put)
        query: Query parameters (for query)
    
    Returns:
        Dict with operation result
    """
    return MEMORY_SYSTEM_MANAGER.memory_store_operation(system_id, operation, key, value, query)

def memory_integrate_research(system_id: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate research data with a memory system.
    
    Args:
        system_id: Identifier for the system
        research_data: Research data to integrate
    
    Returns:
        Dict with integration status
    """
    return MEMORY_SYSTEM_MANAGER.integrate_with_research(system_id, research_data)

def memory_extract_recommendations(system_id: str) -> Dict[str, Any]:
    """
    Extract research recommendations based on memory system.
    
    Args:
        system_id: Identifier for the system
    
    Returns:
        Dict with research recommendations
    """
    return MEMORY_SYSTEM_MANAGER.extract_research_recommendations(system_id)

def memory_list_patterns() -> Dict[str, Any]:
    """
    List all available memory system patterns.
    
    Returns:
        Dict with available patterns and their descriptions
    """
    patterns = {}
    for pattern_id, pattern_info in MEMORY_PATTERNS.items():
        patterns[pattern_id] = {
            "description": pattern_info["description"],
            "stages": pattern_info["stages"],
            "available": pattern_id in MEMORY_SYSTEM_MANAGER.memory_workflows
        }
    
    return {
        "status": "success",
        "patterns": patterns
    }

# Register tools
TOOL_REGISTRY["memory:create_system"] = memory_create_system
TOOL_REGISTRY["memory:get_status"] = memory_get_status
TOOL_REGISTRY["memory:get_next_step"] = memory_get_next_step
TOOL_REGISTRY["memory:select_action"] = memory_select_action
TOOL_REGISTRY["memory:submit_result"] = memory_submit_result
TOOL_REGISTRY["memory:get_results"] = memory_get_results
TOOL_REGISTRY["memory:store_operation"] = memory_store_operation
TOOL_REGISTRY["memory:integrate_research"] = memory_integrate_research
TOOL_REGISTRY["memory:extract_recommendations"] = memory_extract_recommendations
TOOL_REGISTRY["memory:list_patterns"] = memory_list_patterns

# Print initialization message
print("✅ Memory systems tools registered successfully")
