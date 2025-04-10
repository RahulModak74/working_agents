#!/usr/bin/env python3

import os
import sys
import subprocess
import json
import re
import time
from datetime import datetime

def check_browser_adapter():
    """Check if the browser adapter is properly installed and available"""
    adapter_path = os.path.join(os.getcwd(), "browser_adapter.py")
    component_path = os.path.join(os.getcwd(), "COMPONENT", "browser_adapter.py")
    
    adapter_exists = os.path.exists(adapter_path)
    component_exists = os.path.exists(component_path)
    
    print("\n=== BROWSER ADAPTER CHECK ===")
    if adapter_exists:
        print(f"✅ Browser adapter found: {adapter_path}")
        adapter_to_use = adapter_path
    elif component_exists:
        print(f"✅ Browser adapter found in COMPONENT directory: {component_path}")
        adapter_to_use = component_path
    else:
        print("❌ Browser adapter not found!")
        return False
    
    # Check if the adapter contains the necessary tools
    try:
        with open(adapter_to_use, 'r') as f:
            content = f.read()
            
        if "TOOL_REGISTRY" in content and "browser:create" in content:
            print("✅ Browser adapter contains necessary tool definitions")
        else:
            print("❌ Browser adapter does not contain proper tool definitions")
            return False
        
        # Check for Playwright import
        if "from playwright.sync_api import" in content:
            print("✅ Browser adapter imports Playwright")
        else:
            print("❌ Browser adapter is missing Playwright import")
            
        # Check if browser tools are registered
        tools = re.findall(r'TOOL_REGISTRY\["browser:([^"]+)"\]', content)
        print(f"Found {len(tools)} browser tools registered: {', '.join(tools)}")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing browser adapter: {e}")
        return False

def check_playwright_installation():
    """Check if Playwright is properly installed"""
    print("\n=== PLAYWRIGHT INSTALLATION CHECK ===")
    
    # Check if playwright is installed
    try:
        import_result = subprocess.run(
            [sys.executable, "-c", "from playwright.sync_api import sync_playwright; print('Playwright imported successfully')"],
            capture_output=True,
            text=True
        )
        
        if "Playwright imported successfully" in import_result.stdout:
            print("✅ Playwright is properly installed")
            return True
        else:
            print("❌ Playwright import failed")
            print(f"Error: {import_result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error checking Playwright installation: {e}")
        return False

def run_browser_test():
    """Run a minimal browser test outside the agent system"""
    print("\n=== DIRECT BROWSER TEST ===")
    
    test_script = """
import os
from playwright.sync_api import sync_playwright

def test_browser():
    print("Starting browser test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        print("Navigating to Google...")
        page.goto('https://www.google.com')
        print(f"Page title: {page.title()}")
        print("Taking screenshot...")
        screenshot_path = os.path.join(os.getcwd(), "browser_test_screenshot.png")
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        print("Finding links...")
        links = page.query_selector_all('a')
        print(f"Found {len(links)} links")
        print("Closing browser...")
        browser.close()
        print("Browser test completed successfully")
        return True
    return False

if __name__ == "__main__":
    success = test_browser()
    print(f"Test {'succeeded' if success else 'failed'}")
    exit(0 if success else 1)
"""
    
    test_file = "browser_direct_test.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    try:
        print(f"Running direct browser test...")
        subprocess.run([sys.executable, test_file], check=True)
        print("✅ Direct browser test succeeded")
        return True
    except subprocess.CalledProcessError:
        print("❌ Direct browser test failed")
        return False
    except Exception as e:
        print(f"❌ Error running direct browser test: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def check_tool_usage_in_model():
    """Check if the model can generate proper tool usage syntax"""
    print("\n=== MODEL TOOL USAGE CHECK ===")
    
    # Create a simple test.py to check model response
    test_script = """
import json
import subprocess
import os

def call_api():
    api_key = "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"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    model = "openrouter/quasar-alpha"
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. When asked to use a tool, respond with exact format: 'I need to use the tool: [tool_name]\\nParameters:\\n{\\n  \"param1\": \"value1\"\\n}'"},
        {"role": "user", "content": "Please use the browser:create tool with browser_id=test and headless=false"}
    ]
    
    payload = {
        "model": model,
        "messages": conversation
    }
    
    payload_str = json.dumps(payload).replace("'", "'\\\\''")
    
    curl_command = f'''curl {endpoint} \\
      -H "Authorization: Bearer {api_key}" \\
      -H "Content-Type: application/json" \\
      -d '{payload_str}' '''
    
    output_file = "model_test_output.json"
    subprocess.run(curl_command + f" -o {output_file}", shell=True, check=True)
    
    with open(output_file, 'r') as f:
        response = json.load(f)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0].get("message", {}).get("content", "")
        print("Model response:")
        print(content)
        
        # Check if the response contains proper tool usage format
        tool_pattern = r"I need to use the tool: ([a-zA-Z0-9_:]+)\\s*\\nParameters:\\s*\\{([^}]+)\\}"
        match = re.search(tool_pattern, content)
        if match:
            print("✅ Model produced proper tool usage format")
            return True
        else:
            print("❌ Model did not produce proper tool usage format")
            return False
    else:
        print("❌ Invalid API response")
        return False

call_api()
"""
    
    test_file = "model_tool_test.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    try:
        print(f"Testing model tool usage...")
        subprocess.run([sys.executable, test_file], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Model tool usage test failed")
        return False
    except Exception as e:
        print(f"❌ Error testing model tool usage: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def examine_runner():
    """Examine the runner script for proper tool handling"""
    print("\n=== RUNNER EXAMINATION ===")
    
    runner_path = os.path.join(os.getcwd(), "runner_4.py")
    if not os.path.exists(runner_path):
        print(f"❌ Runner not found at {runner_path}")
        return False
    
    try:
        with open(runner_path, 'r') as f:
            content = f.read()
        
        # Check for load_all_tool_adapters
        if "load_all_tool_adapters()" in content:
            print("✅ Runner loads tool adapters")
        else:
            print("❌ Runner is missing tool adapter loading")
        
        # Check for tool pattern extraction
        tool_pattern = r"tool_usage_pattern\s*=\s*r\"([^\"]+)\""
        match = re.search(tool_pattern, content)
        if match:
            pattern = match.group(1)
            print(f"✅ Runner uses tool pattern: {pattern}")
        else:
            print("❌ Runner is missing tool usage pattern")
        
        # Check for tool execution
        if "execute_tool(tool_name, **params)" in content:
            print("✅ Runner contains tool execution code")
        else:
            print("❌ Runner is missing tool execution")
            
        return True
    except Exception as e:
        print(f"❌ Error examining runner: {e}")
        return False

def run_explicit_test_workflow():
    """Run an explicit test workflow with hardcoded tool usage"""
    print("\n=== RUNNING EXPLICIT TEST WORKFLOW ===")
    
    # Create a minimal workflow file
    workflow = [
        {
            "agent": "direct_tool_test",
            "content": """You MUST copy and paste the following tool calls exactly as they appear, one after another. Do not modify them or add any text between them:

I need to use the tool: browser:create
Parameters:
{
  "browser_id": "test_browser",
  "headless": false
}

I need to use the tool: browser:navigate
Parameters:
{
  "url": "https://www.google.com",
  "browser_id": "test_browser"
}

I need to use the tool: browser:get_content
Parameters:
{
  "browser_id": "test_browser",
  "content_type": "text"
}

I need to use the tool: browser:close
Parameters:
{
  "browser_id": "test_browser"
}""",
            "tools": [
                "browser:create",
                "browser:navigate",
                "browser:get_content",
                "browser:close"
            ],
            "output_format": {
                "type": "json",
                "schema": {
                    "result": "string"
                }
            }
        }
    ]
    
    workflow_file = "explicit_tool_test.json"
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    try:
        print(f"Running explicit tool test workflow...")
        result = subprocess.run(
            [sys.executable, "runner_4.py", "--workflow", workflow_file],
            capture_output=True,
            text=True
        )
        
        print("\nWorkflow output:")
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Workflow execution failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
        
        # Check for tool execution in the output
        if "Tool call" in result.stdout:
            print("✅ Workflow execution shows tool calls")
            return True
        else:
            print("❌ No tool calls detected in workflow execution")
            return False
    except Exception as e:
        print(f"❌ Error running explicit test workflow: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(workflow_file):
            os.remove(workflow_file)

def main():
    """Run all diagnostic tests"""
    print("=" * 80)
    print("BROWSER WORKFLOW DIAGNOSTIC")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print("=" * 80)
    
    # Run all tests
    adapter_check = check_browser_adapter()
    playwright_check = check_playwright_installation()
    browser_test = run_browser_test() if playwright_check else False
    runner_check = examine_runner()
    model_check = check_tool_usage_in_model()
    workflow_test = run_explicit_test_workflow()
    
    # Print summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Browser adapter: {'✅ PASS' if adapter_check else '❌ FAIL'}")
    print(f"Playwright installation: {'✅ PASS' if playwright_check else '❌ FAIL'}")
    print(f"Direct browser test: {'✅ PASS' if browser_test else '❌ FAIL'}")
    print(f"Runner examination: {'✅ PASS' if runner_check else '❌ FAIL'}")
    print(f"Model tool usage: {'✅ PASS' if model_check else '❌ FAIL'}")
    print(f"Explicit workflow test: {'✅ PASS' if workflow_test else '❌ FAIL'}")
    
    # Overall assessment
    if adapter_check and playwright_check and browser_test and runner_check and model_check and workflow_test:
        print("\n✅ All tests passed. Browser functionality should be working properly.")
    else:
        print("\n❌ Some tests failed. Review the diagnostic output above for details.")
    
    print("\nTo run a minimal browser test:")
    print("python runner_4.py --workflow explicit_browser_test.json")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
