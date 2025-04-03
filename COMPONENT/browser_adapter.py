#!/usr/bin/env python3

import os
import sys
import json
import time
import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("browser_adapter")

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Please install with 'pip install playwright' and run 'playwright install'")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Browser session store
class BrowserSessionManager:
    def __init__(self):
        self.playwright = None
        self.browsers = {}
        self.contexts = {}
        self.pages = {}
        self.default_browser_id = "default"
        self.default_timeout = 30000  # milliseconds
        self.screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Start playwright if available
        if PLAYWRIGHT_AVAILABLE:
            try:
                self.playwright = sync_playwright().start()
                logger.info("Playwright started successfully")
            except Exception as e:
                logger.error(f"Failed to start Playwright: {str(e)}")
    
    def __del__(self):
        """Clean up resources when the manager is destroyed"""
        if self.playwright:
            try:
                # Close all browsers
                for browser_id in list(self.browsers.keys()):
                    self.close_browser(browser_id)
                
                # Stop Playwright
                self.playwright.stop()
                logger.info("Playwright stopped")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    def create_browser(self, browser_id: str = None, browser_type: str = "chromium", 
                      headless: bool = False) -> Dict[str, Any]:
        """Create a new browser session"""
        if not PLAYWRIGHT_AVAILABLE:
            return {"error": "Playwright is not installed. Cannot create browser."}
        
        if not self.playwright:
            return {"error": "Playwright failed to initialize. Cannot create browser."}
        
        if browser_id is None:
            browser_id = self.default_browser_id
        
        # Close existing browser with this ID if it exists
        if browser_id in self.browsers:
            self.close_browser(browser_id)
        
        try:
            # Select browser type
            if browser_type.lower() == "firefox":
                browser_launcher = self.playwright.firefox
            elif browser_type.lower() == "webkit":
                browser_launcher = self.playwright.webkit
            else:
                browser_launcher = self.playwright.chromium
            
            # Launch browser
            browser = browser_launcher.launch(headless=headless)
            
            # Create a context (like a profile)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
            )
            
            # Create a page
            page = context.new_page()
            page.set_default_timeout(self.default_timeout)
            
            # Store the instances
            self.browsers[browser_id] = browser
            self.contexts[browser_id] = context
            self.pages[browser_id] = page
            
            return {
                "status": "success", 
                "browser_id": browser_id, 
                "type": browser_type,
                "message": f"Browser created with ID: {browser_id}"
            }
            
        except Exception as e:
            logger.error(f"Error creating browser: {str(e)}")
            return {"error": f"Failed to create browser: {str(e)}"}
    
    def close_browser(self, browser_id: str = None) -> Dict[str, Any]:
        """Close a browser session"""
        if browser_id is None:
            browser_id = self.default_browser_id
        
        try:
            # Clean up resources
            if browser_id in self.pages:
                try:
                    self.pages[browser_id].close()
                except:
                    pass
                del self.pages[browser_id]
            
            if browser_id in self.contexts:
                try:
                    self.contexts[browser_id].close()
                except:
                    pass
                del self.contexts[browser_id]
            
            if browser_id in self.browsers:
                try:
                    self.browsers[browser_id].close()
                except:
                    pass
                del self.browsers[browser_id]
            
            return {"status": "success", "message": f"Browser {browser_id} closed"}
            
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            return {"error": f"Failed to close browser: {str(e)}"}
    
    def get_page(self, browser_id: str = None) -> Optional[Page]:
        """Get the page object for a browser"""
        if browser_id is None:
            browser_id = self.default_browser_id
        
        page = self.pages.get(browser_id)
        
        if page:
            try:
                # Check if page is still responsive
                _ = page.url
                return page
            except Exception:
                logger.warning(f"Page for browser {browser_id} is not responsive")
                return None
        
        return None
    
    def navigate(self, url: str, browser_id: str = None, wait_until: str = "load") -> Dict[str, Any]:
        """Navigate to a URL in the specified browser"""
        if not browser_id:
            browser_id = self.default_browser_id
            
        # Create browser if it doesn't exist
        if browser_id not in self.browsers:
            result = self.create_browser(browser_id)
            if "error" in result:
                return result
        
        page = self.get_page(browser_id)
        if not page:
            return {"error": f"Page for browser {browser_id} not found or not responsive"}
        
        try:
            # Navigate to the URL
            page.goto(url, wait_until=wait_until)
            
            return {
                "status": "success",
                "url": page.url,
                "title": page.title()
            }
        except Exception as e:
            logger.error(f"Error navigating to {url}: {str(e)}")
            return {"error": f"Failed to navigate to {url}: {str(e)}"}
    
    def get_page_content(self, browser_id: str = None, content_type: str = "text") -> Dict[str, Any]:
        """Get the content of the current page"""
        page = self.get_page(browser_id)
        if not page:
            return {"error": f"Page for browser {browser_id} not found or not responsive"}
        
        try:
            if content_type.lower() == "html":
                content = page.content()
            else:  # text
                content = page.evaluate("document.body.innerText")
            
            # Get page metadata
            metadata = {
                "url": page.url,
                "title": page.title(),
                "content_length": len(content)
            }
            
            return {
                "status": "success",
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            return {"error": f"Failed to get page content: {str(e)}"}
    
    def find_elements(self, selector: str, browser_id: str = None) -> Dict[str, Any]:
        """Find elements using CSS selectors"""
        page = self.get_page(browser_id)
        if not page:
            return {"error": f"Page for browser {browser_id} not found or not responsive"}
        
        try:
            # Find all elements matching the selector
            elements = page.query_selector_all(selector)
            
            result = []
            for idx, element in enumerate(elements):
                # Get basic info about the element
                tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                
                # Get text content of the element
                try:
                    text = element.inner_text()
                    if len(text) > 100:
                        text = text[:100] + "..."
                except:
                    text = ""
                
                # Get attributes of the element
                attributes = {}
                for attr in ["id", "class", "name", "href", "src", "alt", "title", "value"]:
                    try:
                        value = element.get_attribute(attr)
                        if value:
                            attributes[attr] = value
                    except:
                        pass
                
                # Get position and size
                try:
                    bounding_box = element.bounding_box()
                    position = {
                        "x": bounding_box["x"],
                        "y": bounding_box["y"],
                        "width": bounding_box["width"],
                        "height": bounding_box["height"]
                    }
                except:
                    position = {}
                
                result.append({
                    "index": idx,
                    "tag": tag_name,
                    "text": text,
                    "attributes": attributes,
                    "position": position
                })
            
            return {
                "status": "success",
                "count": len(result),
                "elements": result
            }
        except Exception as e:
            logger.error(f"Error finding elements with selector '{selector}': {str(e)}")
            return {"error": f"Failed to find elements: {str(e)}"}
    
    def click_element(self, selector: str, index: int = 0, browser_id: str = None) -> Dict[str, Any]:
        """Click on an element using CSS selectors"""
        page = self.get_page(browser_id)
        if not page:
            return {"error": f"Page for browser {browser_id} not found or not responsive"}
        
        try:
            # Find all elements matching the selector
            elements = page.query_selector_all(selector)
            
            if not elements:
                return {"error": f"No elements found matching selector: {selector}"}
            
            if index >= len(elements):
                return {"error": f"Index {index} is out of range. Only {len(elements)} elements found."}
            
            # Click the specified element
            elements[index].scroll_into_view_if_needed()
            elements[index].click()
            
            # Allow some time for navigation or DOM updates
            page.wait_for_load_state("networkidle")
            
            return {
                "status": "success",
                "message": f"Clicked element: {selector} at index {index}",
                "current_url": page.url,
                "title": page.title()
            }
        except Exception as e:
            logger.error(f"Error clicking element: {str(e)}")
            return {"error": f"Failed to click element: {str(e)}"}

# Global browser manager instance
BROWSER_MANAGER = BrowserSessionManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def browser_create(browser_id: str = None, browser_type: str = "chromium", headless: bool = False) -> Dict[str, Any]:
    """
    Create a new browser instance.
    
    Args:
        browser_id: Optional unique identifier for the browser
        browser_type: Type of browser ('chromium', 'firefox', or 'webkit')
        headless: Whether to run the browser in headless mode
    
    Returns:
        Dict with status and browser information
    """
    return BROWSER_MANAGER.create_browser(browser_id, browser_type, headless)

def browser_navigate(url: str, browser_id: str = None, wait_until: str = "load") -> Dict[str, Any]:
    """
    Navigate to a URL.
    
    Args:
        url: URL to navigate to
        browser_id: Browser identifier (optional)
        wait_until: When to consider navigation complete ('load', 'domcontentloaded', 'networkidle')
    
    Returns:
        Dict with navigation status and page information
    """
    return BROWSER_MANAGER.navigate(url, browser_id, wait_until)

def browser_get_content(browser_id: str = None, content_type: str = "text") -> Dict[str, Any]:
    """
    Get the content of the current page.
    
    Args:
        browser_id: Browser identifier (optional)
        content_type: Type of content to retrieve ('text' or 'html')
    
    Returns:
        Dict with page content and metadata
    """
    return BROWSER_MANAGER.get_page_content(browser_id, content_type)

def browser_find_elements(selector: str, browser_id: str = None) -> Dict[str, Any]:
    """
    Find elements on the page using CSS selectors.
    
    Args:
        selector: CSS selector string
        browser_id: Browser identifier (optional)
    
    Returns:
        Dict with elements found and their details
    """
    return BROWSER_MANAGER.find_elements(selector, browser_id)

def browser_click(selector: str, index: int = 0, browser_id: str = None) -> Dict[str, Any]:
    """
    Click on an element on the page.
    
    Args:
        selector: CSS selector for the element
        index: Index of the element if multiple elements match the selector
        browser_id: Browser identifier (optional)
    
    Returns:
        Dict with click status and updated page information
    """
    return BROWSER_MANAGER.click_element(selector, index, browser_id)

def browser_close(browser_id: str = None) -> Dict[str, Any]:
    """
    Close a browser session.
    
    Args:
        browser_id: Browser identifier (optional)
    
    Returns:
        Dict with closure status
    """
    return BROWSER_MANAGER.close_browser(browser_id)

# Register tools
TOOL_REGISTRY["browser:create"] = browser_create
TOOL_REGISTRY["browser:navigate"] = browser_navigate
TOOL_REGISTRY["browser:get_content"] = browser_get_content
TOOL_REGISTRY["browser:find_elements"] = browser_find_elements
TOOL_REGISTRY["browser:click"] = browser_click
TOOL_REGISTRY["browser:close"] = browser_close

# Function for executing browser tools
def execute_tool(tool_id: str, **kwargs):
    """Execute a browser tool based on the tool ID"""
    if tool_id in TOOL_REGISTRY:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    else:
        return {"error": f"Unknown tool: {tool_id}"}

# Print initialization message
print("✅ Browser automation tools registered successfully")
