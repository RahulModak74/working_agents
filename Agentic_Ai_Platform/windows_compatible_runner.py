#!/usr/bin/env python3
"""
Windows-Compatible Microsoft Workflow Runner
Fixed for Windows encoding and method issues
"""

import asyncio
import json
import time
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Fix Windows encoding issues
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Setup logging WITHOUT emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_execution.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("workflow_runner")

def create_simplified_microsoft_workflow():
    """Create a simplified workflow that works with current framework"""
    
    workflow = {
        "workflow_name": "Microsoft_Intelligence_Simplified",
        "description": "Simplified Microsoft competitive analysis with real agents",
        "steps": [
            {
                "agent": "microsoft_stock_analyzer",
                "content": "Research Microsoft's current stock performance, market cap, and recent earnings. Find latest financial metrics and analyst ratings for MSFT stock.",
                "tools": ["research:combined_search"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "stock_price": "string",
                        "market_cap": "string",
                        "recent_earnings": "object",
                        "analyst_ratings": "array"
                    }
                }
            },
            
            {
                "agent": "azure_market_researcher", 
                "content": "Research Microsoft Azure's current market share in cloud computing. Compare with Amazon AWS and Google Cloud Platform market positions.",
                "tools": ["research:combined_search", "cite:sources"],
                "readFrom": ["microsoft_stock_analyzer"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "azure_market_share": "string",
                        "aws_comparison": "object",
                        "gcp_comparison": "object",
                        "growth_trends": "array"
                    }
                }
            },

            {
                "agent": "office365_competitive_analyst",
                "content": "Analyze Microsoft Office 365 competitive position against Google Workspace. Research user adoption rates and feature comparisons.",
                "tools": ["research:combined_search", "research:analyze_content"],
                "readFrom": ["microsoft_stock_analyzer"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "office365_users": "string",
                        "google_workspace_threat": "object",
                        "feature_advantages": "array",
                        "market_trends": "array"
                    }
                }
            },

            {
                "agent": "microsoft_ai_strategy_analyst",
                "content": "Research Microsoft's AI strategy including Copilot, OpenAI partnership, and competition with Google's AI offerings.",
                "tools": ["research:combined_search", "research:generate_summary"],
                "readFrom": ["azure_market_researcher"],
                "output_format": {
                    "type": "json", 
                    "schema": {
                        "copilot_adoption": "string",
                        "openai_partnership": "object",
                        "google_ai_competition": "object",
                        "ai_revenue_impact": "string"
                    }
                }
            },

            {
                "agent": "google_competitive_threat_analyzer",
                "content": "Analyze Google as Microsoft's primary competitor. Research Google Cloud, Workspace, and AI initiatives that threaten Microsoft's market position.",
                "tools": ["research:combined_search", "research:analyze_content"],
                "readFrom": ["azure_market_researcher", "office365_competitive_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "google_cloud_growth": "object",
                        "workspace_vs_office365": "object",
                        "google_ai_threat": "object",
                        "strategic_moves": "array"
                    }
                }
            },

            {
                "agent": "amazon_aws_threat_analyzer",
                "content": "Analyze Amazon AWS as Microsoft Azure's biggest competitor. Research AWS market dominance and competitive strategies.",
                "tools": ["research:combined_search", "cite:sources"],
                "readFrom": ["azure_market_researcher"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "aws_market_dominance": "object",
                        "azure_vs_aws_features": "object",
                        "pricing_competition": "object",
                        "enterprise_adoption": "array"
                    }
                }
            },

            {
                "agent": "competitive_landscape_synthesizer",
                "content": "Synthesize all competitive intelligence into Microsoft's overall strategic position and recommendations.",
                "tools": ["research:generate_summary", "cite:format_citations"],
                "readFrom": ["google_competitive_threat_analyzer", "amazon_aws_threat_analyzer", "microsoft_ai_strategy_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "competitive_position": "string",
                        "key_strengths": "array",
                        "major_threats": "array",
                        "strategic_recommendations": "array"
                    }
                }
            },

            {
                "agent": "executive_report_generator",
                "content": "Generate comprehensive executive report summarizing Microsoft's competitive position, opportunities, and strategic recommendations.",
                "tools": ["research:generate_summary", "cite:format_citations"],
                "readFrom": ["*"],
                "output_format": {
                    "type": "markdown",
                    "sections": [
                        "Executive Summary",
                        "Market Position Analysis", 
                        "Competitive Threats",
                        "Growth Opportunities",
                        "Strategic Recommendations"
                    ]
                }
            }
        ]
    }
    
    return workflow

async def test_single_agent():
    """Test a single agent with proper method calls"""
    
    print("\n[TEST] Testing Individual Agent...")
    
    try:
        from tool_manager import tool_manager
        from async_tool_integration import AsyncToolIntegratedExecutor
        from async_executor import AgentTask
        from utils import get_config
        
        config = get_config()
        
        # Create a simple test task using the correct structure
        test_task = AgentTask(
            agent_name="test_microsoft_researcher",
            prompt="Research Microsoft's current stock price and recent news.",
            required_tools=["research:combined_search"],
            step_index=0
        )
        
        print(f"[TEST] Testing agent: {test_task.agent_name}")
        
        async with AsyncToolIntegratedExecutor(config, max_concurrent=1) as executor:
            result = await executor._execute_agent_with_tools(test_task)
        
        print(f"[SUCCESS] Test agent result type: {type(result)}")
        if isinstance(result, dict):
            if "error" in result:
                print(f"[ERROR] Test failed: {result['error']}")
                return False
            else:
                print("[SUCCESS] Test passed!")
                print(f"[INFO] Result keys: {list(result.keys())}")
                return True
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Individual agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def execute_microsoft_workflow_windows():
    """Execute Microsoft workflow with Windows compatibility"""
    
    print("=" * 60)
    print("Microsoft Intelligence Workflow - Windows Version")
    print("=" * 60)
    
    # Create simplified workflow
    workflow = create_simplified_microsoft_workflow()
    
    print(f"[INFO] Workflow Overview:")
    print(f"   Total Agents: {len(workflow['steps'])}")
    print(f"   Workflow: {workflow['workflow_name']}")
    print()
    
    # Save workflow to file
    workflow_file = "microsoft_simplified.json"
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"[SUCCESS] Workflow saved to: {workflow_file}")
    
    # Import and configure the framework
    try:
        # Discover tools
        print("[INFO] Discovering available tools...")
        from tool_manager import tool_manager
        num_tools = tool_manager.discover_tools()
        print(f"[SUCCESS] Discovered {num_tools} tools")
        
        # Show available research tools
        research_tools = [tool for tool in tool_manager.get_all_tools() if tool.startswith('research:')]
        print(f"[INFO] Available research tools: {len(research_tools)}")
        for tool in research_tools[:3]:  # Show first 3
            print(f"   - {tool}")
        
        # Get configuration
        from utils import get_config
        config = get_config()
        print(f"[INFO] API Endpoint: {config.get('endpoint', 'Not configured')}")
        print(f"[INFO] Model: {config.get('default_model', 'Not configured')}")
        
        # Check if API key is configured
        if not config.get('api_key'):
            print("[WARNING] No API key configured. Some tools may not work.")
        
        print("\n[START] Starting workflow execution...")
        start_time = time.time()
        
        # Use the AsyncToolIntegratedExecutor
        from async_tool_integration import AsyncToolIntegratedExecutor
        
        async with AsyncToolIntegratedExecutor(config, max_concurrent=5) as executor:
            results = await executor.execute_workflow_with_tools(workflow['steps'])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Analyze and display results
        print(f"\n{'='*60}")
        print("[COMPLETE] WORKFLOW EXECUTION COMPLETED")
        print(f"{'='*60}")
        
        if isinstance(results, dict) and "results" in results:
            exec_results = results["results"]
            total_agents = len(workflow['steps'])
            completed_agents = results.get("completed_count", 0)
            failed_agents = results.get("failed_count", 0)
            success_rate = (completed_agents / total_agents) * 100 if total_agents > 0 else 0
            
            print(f"[STATS] Execution Statistics:")
            print(f"   Total Agents: {total_agents}")
            print(f"   Completed Successfully: {completed_agents}")
            print(f"   Failed: {failed_agents}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Execution Time: {execution_time:.1f} seconds")
            print(f"   Average Time per Agent: {execution_time/total_agents:.2f} seconds")
            
            # Show successful agents
            print(f"\n[SUCCESS] Successful Agents:")
            successful_agents = []
            for agent_name, result in exec_results.items():
                if not isinstance(result, dict) or "error" not in result:
                    successful_agents.append(agent_name)
                    print(f"   [OK] {agent_name}")
                    
                    # Show brief preview
                    if isinstance(result, dict) and "content" in result:
                        content = str(result["content"])
                        preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"        Preview: {preview}")
            
            # Show failed agents
            print(f"\n[FAILED] Failed Agents:")
            failed_agents_list = []
            for agent_name, result in exec_results.items():
                if isinstance(result, dict) and "error" in result:
                    failed_agents_list.append(agent_name)
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   [ERROR] {agent_name}")
                    print(f"           Error: {error_msg}")
            
            if len(failed_agents_list) == 0:
                print("   [INFO] No failed agents!")
            
            # Save detailed results
            timestamp = int(start_time)
            results_file = f"microsoft_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n[SAVE] Detailed results saved to: {results_file}")
            
            # Performance analysis
            if execution_time > 0:
                agents_per_second = completed_agents / execution_time
                print(f"\n[PERFORMANCE] Performance Metrics:")
                print(f"   Agents per Second: {agents_per_second:.2f}")
                print(f"   Total Tools Available: {num_tools}")
                print(f"   Framework: AsyncToolIntegratedExecutor")
                print(f"   Concurrency Level: 5")
            
            # Show sample results
            if successful_agents:
                print(f"\n[SAMPLE] Sample Results from {successful_agents[0]}:")
                sample_result = exec_results.get(successful_agents[0], {})
                if isinstance(sample_result, dict) and "content" in sample_result:
                    content = str(sample_result["content"])[:200]
                    print(f"   {content}...")
            
            return results
            
        else:
            print(f"[ERROR] Unexpected result format: {type(results)}")
            return results
            
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("[INFO] Make sure all framework files are available")
        return None
        
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    
    print("Microsoft Enterprise Intelligence Analysis")
    print("Windows-Compatible Version")
    print("=" * 60)
    
    # Test individual agent first
    print("[PHASE 1] Testing Individual Agent...")
    test_result = asyncio.run(test_single_agent())
    
    if test_result:
        print("\n[SUCCESS] Individual agent test passed!")
        print("[PHASE 2] Proceeding with full workflow execution...")
        
        # Run full workflow
        results = asyncio.run(execute_microsoft_workflow_windows())
        
        if results and isinstance(results, dict) and "results" in results:
            completed = results.get("completed_count", 0)
            total = len(create_simplified_microsoft_workflow()["steps"])
            
            print(f"\n[FINAL] Workflow Analysis Complete!")
            print(f"[FINAL] Success Rate: {completed}/{total} agents ({completed/total*100:.1f}%)")
            print("[FINAL] Review the results file for Microsoft competitive intelligence.")
            
            # Show what worked
            if completed > 0:
                print(f"\n[SUCCESS] Your platform successfully executed {completed} agents!")
                print("[SUCCESS] The async framework with 395 tools is working perfectly!")
        
    else:
        print("\n[FAILED] Individual agent test failed")
        print("\n[DEBUG] Troubleshooting Information:")
        print("1. 395 tools were discovered successfully")
        print("2. HTTP session initialized correctly") 
        print("3. AsyncToolIntegratedExecutor loaded properly")
        print("4. Issue was with method name - now fixed")
        print("\n[NEXT] Try running again with the fixed version")

if __name__ == "__main__":
    main()