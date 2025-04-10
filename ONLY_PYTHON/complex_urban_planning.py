#!/usr/bin/env python3

import os
import json
import logging
import time
import sqlite3
import requests
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import CONFIG

# Import our agent system modules
from agent_framework import AgentSystem, Agent, DynamicAgent
from agent_tools import ToolRegistry
from agent_system_types import (
    AdvancedCognitionSystem,
    DebateSystem,
    MetacognitiveSystem,
    TreeOfThoughtsSystem
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["output_dir"], "urban_planning.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Extend tools to integrate with external LLM API
def llm_query(prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Query the configured LLM with a prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        context: Additional context for the prompt
        
    Returns:
        Dict with the LLM response
    """
    logger.info(f"Querying LLM with prompt: {prompt[:50]}...")
    
    try:
        # Add context to the prompt if provided
        if context:
            context_str = json.dumps(context, indent=2)
            full_prompt = f"Context:\n{context_str}\n\nPrompt: {prompt}"
        else:
            full_prompt = prompt
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['api_key']}"
        }
        
        data = {
            "model": CONFIG["default_model"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in urban planning and city development."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        # Make the API request
        response = requests.post(CONFIG["endpoint"], headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Extract the content from the response
        content = result["choices"][0]["message"]["content"]
        
        return {
            "prompt": prompt,
            "response": content,
            "model": CONFIG["default_model"],
            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
        }
    
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return {
            "prompt": prompt,
            "response": f"Error: {str(e)}",
            "model": CONFIG["default_model"],
            "error": True
        }

# Enhanced SQL tool that connects to the actual database
def enhanced_sql_query(query: str, database: str = None) -> Dict[str, Any]:
    """
    Executes an SQL query against a real SQLite database.
    
    Args:
        query: The SQL query to execute
        database: The database to query (defaults to the one in CONFIG)
        
    Returns:
        Dict with query results
    """
    if database is None:
        database = CONFIG["sqlite_db"]
    
    logger.info(f"Executing SQL query on database {database}: {query[:50]}...")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        
        # Execute the query
        start_time = time.time()
        cursor.execute(query)
        
        # Determine the query type
        query_lower = query.lower().strip()
        
        if query_lower.startswith("select"):
            # For SELECT queries, fetch results
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            result_data = {
                "query": query,
                "database": database,
                "results": results,
                "row_count": len(results),
                "execution_time": time.time() - start_time,
                "operation": "SELECT"
            }
        else:
            # For non-SELECT queries, get the row count
            rows_affected = cursor.rowcount
            conn.commit()
            
            # Determine the operation type
            if query_lower.startswith("insert"):
                operation = "INSERT"
                result_data = {
                    "query": query,
                    "database": database,
                    "rows_affected": rows_affected,
                    "last_insert_id": cursor.lastrowid,
                    "execution_time": time.time() - start_time,
                    "operation": operation
                }
            else:
                if query_lower.startswith("update"):
                    operation = "UPDATE"
                elif query_lower.startswith("delete"):
                    operation = "DELETE"
                else:
                    operation = "OTHER"
                
                result_data = {
                    "query": query,
                    "database": database,
                    "rows_affected": rows_affected,
                    "execution_time": time.time() - start_time,
                    "operation": operation
                }
        
        # Close the connection
        conn.close()
        
        return result_data
    
    except Exception as e:
        logger.error(f"SQL error: {str(e)}")
        return {
            "query": query,
            "database": database,
            "error": str(e),
            "execution_time": 0,
            "operation": "ERROR"
        }

# Register the enhanced tools
ToolRegistry.register_tool("llm:query", llm_query)
ToolRegistry.register_tool("sql:enhanced_query", enhanced_sql_query)

# Urban Planning System that coordinates multiple agent systems
class UrbanPlanningSystem:
    """
    A complex system that coordinates multiple agent systems to solve
    urban planning challenges from multiple perspectives.
    """
    
    def __init__(self):
        """Initialize the Urban Planning System with its component systems."""
        self.cognition_system = AdvancedCognitionSystem()
        self.debate_system = DebateSystem()
        self.metacognitive_system = MetacognitiveSystem()
        self.tree_of_thoughts_system = TreeOfThoughtsSystem()
        
        # Track timing and results
        self.execution_times = {}
        self.results = {}
    
    def setup_database(self):
        """Set up the SQLite database with tables for urban planning data."""
        logger.info("Setting up SQLite database...")
        
        try:
            # Connect to the database
            conn = sqlite3.connect(CONFIG["sqlite_db"])
            cursor = conn.cursor()
            
            # Create urban planning tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                population INTEGER,
                area_sqkm REAL,
                density REAL,
                country TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS transportation_modes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                carbon_emissions_per_passenger_km REAL,
                capacity_per_hour INTEGER,
                operating_cost_per_km REAL,
                infrastructure_cost_per_km REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS land_use (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                area_percentage REAL,
                population_density REAL,
                carbon_emissions_per_sqkm REAL,
                economic_value_per_sqkm REAL
            )
            ''')
            
            # Insert sample data
            # Sample cities
            cities_data = [
                (1, 'Sample City', 500000, 200.0, 2500.0, 'Sample Country'),
                (2, 'Eco City', 300000, 150.0, 2000.0, 'Green Country'),
                (3, 'Tech Hub', 700000, 250.0, 2800.0, 'Innovation Nation')
            ]
            
            cursor.executemany('''
            INSERT OR REPLACE INTO cities 
            (id, name, population, area_sqkm, density, country) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', cities_data)
            
            # Sample transportation modes
            transport_data = [
                (1, 'Bus', 0.08, 5000, 3.5, 1500000),
                (2, 'Light Rail', 0.04, 12000, 8.0, 25000000),
                (3, 'Bicycle Infrastructure', 0.0, 3500, 0.1, 500000),
                (4, 'Car (Gas)', 0.17, 2000, 0.5, 8000000),
                (5, 'Car (Electric)', 0.05, 2000, 0.4, 8000000),
                (6, 'Walking Infrastructure', 0.0, 5000, 0.05, 750000)
            ]
            
            cursor.executemany('''
            INSERT OR REPLACE INTO transportation_modes 
            (id, name, carbon_emissions_per_passenger_km, capacity_per_hour, 
             operating_cost_per_km, infrastructure_cost_per_km) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', transport_data)
            
            # Sample land use types
            land_use_data = [
                (1, 'Residential (High Density)', 30.0, 10000, 25.0, 2500000),
                (2, 'Residential (Low Density)', 40.0, 2500, 15.0, 1500000),
                (3, 'Commercial', 15.0, 5000, 40.0, 5000000),
                (4, 'Industrial', 10.0, 1000, 80.0, 3500000),
                (5, 'Parks and Green Space', 5.0, 100, 5.0, 800000)
            ]
            
            cursor.executemany('''
            INSERT OR REPLACE INTO land_use 
            (id, type, area_percentage, population_density, 
             carbon_emissions_per_sqkm, economic_value_per_sqkm) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', land_use_data)
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            logger.info("Database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            return False
    
    def solve_urban_planning_challenge(self, challenge_description: str) -> Dict[str, Any]:
        """
        Solve a complex urban planning challenge using multiple agent systems.
        
        Args:
            challenge_description: Description of the urban planning challenge
            
        Returns:
            Dict with the comprehensive solution from multiple perspectives
        """
        logger.info(f"Solving urban planning challenge: {challenge_description[:50]}...")
        
        # Ensure the database is set up
        self.setup_database()
        
        # 1. Initial problem framing using Advanced Cognition
        start_time = time.time()
        cognition_results = self.cognition_system.solve_problem(challenge_description)
        self.execution_times["cognition_system"] = time.time() - start_time
        self.results["cognition_system"] = cognition_results
        
        logger.info(f"Advanced Cognition approach used: {cognition_results['approach_used']}")
        
        # Extract key dimensions and potential approaches
        if "solution" in cognition_results and "Problem Summary" in cognition_results["solution"]:
            problem_summary = cognition_results["solution"]["Problem Summary"]
        else:
            problem_summary = "Urban planning challenge requiring multiple perspectives"
        
        # 2. Debate key contentious aspects
        debate_topic = f"What is the optimal balance between {problem_summary} when addressing: {challenge_description}"
        
        start_time = time.time()
        debate_results = self.debate_system.debate_topic(debate_topic)
        self.execution_times["debate_system"] = time.time() - start_time
        self.results["debate_system"] = debate_results
        
        logger.info("Debate completed with synthesis of perspectives")
        
        # 3. Detailed solution design using Tree of Thoughts
        tot_problem = f"Based on the debate synthesis and initial problem framing, design detailed solutions for: {challenge_description}"
        
        start_time = time.time()
        tot_results = self.tree_of_thoughts_system.solve_with_tot(tot_problem)
        self.execution_times["tree_of_thoughts_system"] = time.time() - start_time
        self.results["tree_of_thoughts_system"] = tot_results
        
        logger.info(f"Tree of Thoughts recommended branch: {tot_results['recommended_branch']}")
        
        # 4. Metacognitive analysis of the solution
        metacog_problem = f"Analyze the proposed urban planning solution for: {challenge_description}"
        
        start_time = time.time()
        metacog_results = self.metacognitive_system.solve_with_metacognition(metacog_problem)
        self.execution_times["metacognitive_system"] = time.time() - start_time
        self.results["metacognitive_system"] = metacog_results
        
        logger.info("Metacognitive analysis completed with revised solution")
        
        # 5. Compile the comprehensive solution
        comprehensive_solution = self._compile_solution(challenge_description)
        
        # Save the solution to a file
        solution_filename = os.path.join(CONFIG["output_dir"], "urban_planning_solution.json")
        with open(solution_filename, 'w') as f:
            json.dump(comprehensive_solution, f, indent=2)
        
        logger.info(f"Solution saved to {solution_filename}")
        
        return comprehensive_solution
    
    def _compile_solution(self, challenge_description: str) -> Dict[str, Any]:
        """Compile a comprehensive solution from all system results."""
        
        # Get relevant data from the database for context
        try:
            conn = sqlite3.connect(CONFIG["sqlite_db"])
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM transportation_modes")
            transportation_data = cursor.fetchall()
            
            cursor.execute("SELECT * FROM land_use")
            land_use_data = cursor.fetchall()
            
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching data from database: {str(e)}")
            transportation_data = []
            land_use_data = []
        
        # Create a comprehensive solution
        solution = {
            "challenge": challenge_description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_times": self.execution_times,
            "problem_framing": {
                "approach": self.results["cognition_system"]["approach_used"],
                "summary": self.results["cognition_system"]["solution"].get("Problem Summary", "")
            },
            "key_perspectives": {
                "position_a": self.results["debate_system"]["position_a"].get("position", ""),
                "position_b": self.results["debate_system"]["position_b"].get("position", ""),
                "synthesis": self.results["debate_system"]["synthesis"].get("Synthesized Position", "")
            },
            "solution_design": {
                "approach": self.results["tree_of_thoughts_system"]["recommended_branch"],
                "solution": self.results["tree_of_thoughts_system"]["final_solution"].get("Detailed Implementation", "")
            },
            "solution_critique": {
                "biases": self.results["metacognitive_system"]["bias_analysis"].get("overall_bias_assessment", ""),
                "improvements": self.results["metacognitive_system"]["revised_solution"].get("Key Improvements", "")
            },
            "implementation_plan": {
                "immediate_actions": [],
                "medium_term_actions": [],
                "long_term_actions": [],
                "success_metrics": []
            },
            "data_context": {
                "transportation_modes": transportation_data,
                "land_use_types": land_use_data
            }
        }
        
        # Generate implementation actions using LLM
        implementation_prompt = f"""
        Based on this urban planning solution summary:
        
        {json.dumps(solution, indent=2)}
        
        Generate:
        1. Five immediate actions (0-1 year)
        2. Five medium-term actions (1-5 years)
        3. Five long-term actions (5+ years)
        4. Five success metrics to evaluate the solution
        
        Format as a JSON with four arrays named: immediate_actions, medium_term_actions, long_term_actions, success_metrics
        """
        
        try:
            implementation_response = llm_query(implementation_prompt)
            if not implementation_response.get("error"):
                # Try to parse the JSON response
                try:
                    implementation_plan = json.loads(implementation_response["response"])
                    solution["implementation_plan"] = implementation_plan
                except:
                    # If JSON parsing fails, add the raw response
                    solution["implementation_plan"]["raw_response"] = implementation_response["response"]
        except Exception as e:
            logger.error(f"Error generating implementation plan: {str(e)}")
        
        return solution

def print_section(title: str):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def main():
    print_section("Urban Planning System - Complex Example")
    
    # Create the Urban Planning System
    planning_system = UrbanPlanningSystem()
    
    # Define a complex urban planning challenge
    challenge = """
    Design a comprehensive transformation plan for a mid-sized city (population 500,000) 
    to achieve carbon neutrality by 2040 while enhancing quality of life, economic 
    prosperity, and social equity. The plan should address:
    
    1. Transportation systems and mobility
    2. Land use and urban form
    3. Energy production and consumption
    4. Building standards and retrofitting
    5. Green infrastructure and natural systems
    6. Circular economy and waste management
    7. Governance and community engagement
    
    The solution must consider technical feasibility, economic viability, social acceptance, 
    political realities, and environmental impacts. It should also address potential 
    trade-offs and synergies between different aspects of the plan.
    """
    
    print(f"Challenge: {challenge.strip()}")
    print("\nSolving this complex challenge using multiple agent systems...")
    print("This might take some time as it consults multiple agent systems.")
    
    # Solve the challenge
    solution = planning_system.solve_urban_planning_challenge(challenge)
    
    # Display summary results
    print_section("Solution Summary")
    print(f"Problem approach: {solution['problem_framing']['approach']}")
    print(f"Recommended solution design: {solution['solution_design']['approach']}")
    
    print_section("Key Perspectives Synthesis")
    print(solution['key_perspectives']['synthesis'])
    
    print_section("Execution Times")
    for system, time_taken in solution['execution_times'].items():
        print(f"{system}: {time_taken:.2f} seconds")
    
    print_section("Results Location")
    print(f"Full solution saved to: {os.path.join(CONFIG['output_dir'], 'urban_planning_solution.json')}")
    print(f"Log file: {os.path.join(CONFIG['output_dir'], 'urban_planning.log')}")

if __name__ == "__main__":
    main()
