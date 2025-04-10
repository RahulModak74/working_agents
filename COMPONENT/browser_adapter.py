#!/usr/bin/env python3

import os
import sys
import json
import time
import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Set up enhanced logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure file handler for logging
log_file = os.path.join(LOG_DIR, f"browser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Set up the logger
logger = logging.getLogger("browser_adapter")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=" * 50)
logger.info("BROWSER ADAPTER INITIALIZED")
logger.info(f"Logging to: {log_file}")
logger.info("=" * 50)

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
    logger.info("✅ Playwright successfully imported")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("❌ Playwright not available. Please install with 'pip install playwright' and run 'playwright install'")

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
        logger.info(f"Screenshot directory: {self.screenshot_dir}")
        
        # Start playwright if available
        if PLAYWRIGHT_AVAILABLE:
            try:
                self.playwright = sync_playwright().start()
                logger.info("🚀 Playwright started successfully")
            except Exception as e:
                logger.error(f"❌ Failed to start Playwright: {str(e)}")
    
    def __del__(self):
        """Clean up resources when the manager is destroyed"""
        if self.playwright:
            try:
                # Close all browsers
                for browser_id in list(self.browsers.keys()):
                    self.close_browser(browser_id)
                
                # Stop Playwright
                self.playwright.stop()
                logger.info("🛑 Playwright stopped")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    def _save_screenshot(self, page, browser_id, action_name):
        """Save a screenshot of the current page state"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{browser_id}_{action_name}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            page.screenshot(path=filepath)
            logger.info(f"📸 Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save screenshot: {str(e)}")
            return None
    
    def create_browser(self, browser_id: str = None, browser_type: str = "chromium", 
                      headless: bool = True) -> Dict[str, Any]:
        """Create a new browser session"""
        logger.info(f"🔧 Creating browser: type={browser_type}, headless={headless}, id={browser_id or self.default_browser_id}")
        
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Cannot create browser - Playwright is not installed")
            return {"error": "Playwright is not installed. Cannot create browser."}
        
        if not self.playwright:
            logger.error("Cannot create browser - Playwright failed to initialize")
            return {"error": "Playwright failed to initialize. Cannot create browser."}
        
        if browser_id is None:
            browser_id = self.default_browser_id
        
        # Close existing browser with this ID if it exists
        if browser_id in self.browsers:
            logger.info(f"Closing existing browser with ID: {browser_id}")
            self.close_browser(browser_id)
        
        try:
            # Select browser type
            if browser_type.lower() == "firefox":
                browser_launcher = self.playwright.firefox
                logger.info("Selected Firefox browser")
            elif browser_type.lower() == "webkit":
                browser_launcher = self.playwright.webkit
                logger.info("Selected WebKit browser")
            else:
                browser_launcher = self.playwright.chromium
                logger.info("Selected Chromium browser")
            
            # Launch browser
            browser = browser_launcher.launch(headless=headless)
            logger.info(f"Browser launched successfully: {browser_type}")
            
            # Create a context (like a profile)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
            )
            logger.info("Browser context created")
            
            # Create a page
            page = context.new_page()
            page.set_default_timeout(self.default_timeout)
            logger.info("Page created in browser")
            
            # Store the instances
            self.browsers[browser_id] = browser
            self.contexts[browser_id] = context
            self.pages[browser_id] = page
            
            logger.info(f"✅ Browser created successfully with ID: {browser_id}")
            
            return {
                "status": "success", 
                "browser_id": browser_id, 
                "type": browser_type,
                "message": f"Browser created with ID: {browser_id}"
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating browser: {str(e)}")
            return {"error": f"Failed to create browser: {str(e)}"}
    
    def close_browser(self, browser_id: str = None) -> Dict[str, Any]:
        """Close a browser session"""
        if browser_id is None:
            browser_id = self.default_browser_id
        
        logger.info(f"🛑 Closing browser: {browser_id}")
        
        try:
            # Clean up resources
            if browser_id in self.pages:
                try:
                    self.pages[browser_id].close()
                    logger.info(f"Closed page for browser {browser_id}")
                except Exception as e:
                    logger.warning(f"Error closing page: {str(e)}")
                del self.pages[browser_id]
            
            if browser_id in self.contexts:
                try:
                    self.contexts[browser_id].close()
                    logger.info(f"Closed context for browser {browser_id}")
                except Exception as e:
                    logger.warning(f"Error closing context: {str(e)}")
                del self.contexts[browser_id]
            
            if browser_id in self.browsers:
                try:
                    self.browsers[browser_id].close()
                    logger.info(f"Closed browser instance {browser_id}")
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
                del self.browsers[browser_id]
            
            logger.info(f"✅ Browser {browser_id} successfully closed")
            return {"status": "success", "message": f"Browser {browser_id} closed"}
            
        except Exception as e:
            logger.error(f"❌ Error closing browser: {str(e)}")
            return {"error": f"Failed to close browser: {str(e)}"}
    
    def get_page(self, browser_id: str = None) -> Optional[Page]:
        """Get the page object for a browser"""
        if browser_id is None:
            browser_id = self.default_browser_id
        
        logger.info(f"Getting page for browser: {browser_id}")
        
        page = self.pages.get(browser_id)
        
        if page:
            try:
                # Check if page is still responsive
                _ = page.url
                logger.info(f"Page for browser {browser_id} is responsive")
                return page
            except Exception as e:
                logger.warning(f"Page for browser {browser_id} is not responsive: {str(e)}")
                return None
        else:
            logger.warning(f"No page found for browser {browser_id}")
        
        return None
    
    def navigate(self, url: str, browser_id: str = None, wait_until: str = "load") -> Dict[str, Any]:
        """Navigate to a URL in the specified browser"""
        logger.info(f"🌐 Navigating to URL: {url} in browser: {browser_id or self.default_browser_id}")
        
        if not browser_id:
            browser_id = self.default_browser_id
            
        # Create browser if it doesn't exist
        if browser_id not in self.browsers:
            logger.info(f"Browser {browser_id} doesn't exist, creating it")
            result = self.create_browser(browser_id)
            if "error" in result:
                logger.error(f"Failed to create browser: {result['error']}")
                return result
        
        page = self.get_page(browser_id)
        if not page:
            error_msg = f"Page for browser {browser_id} not found or not responsive"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Navigate to the URL
            logger.info(f"Starting navigation to {url} with wait_until={wait_until}")
            start_time = time.time()
            page.goto(url, wait_until=wait_until)
            end_time = time.time()
            
            # Get page information
            page_title = page.title()
            page_url = page.url
            
            logger.info(f"✅ Navigation successful to: {page_url}")
            logger.info(f"Page title: {page_title}")
            logger.info(f"Navigation took {end_time - start_time:.2f} seconds")
            
            # Take a screenshot
            screenshot_path = self._save_screenshot(page, browser_id, "navigate")
            
            return {
                "status": "success",
                "url": page_url,
                "title": page_title,
                "navigation_time_seconds": round(end_time - start_time, 2),
                "screenshot": os.path.basename(screenshot_path) if screenshot_path else None
            }
        except Exception as e:
            logger.error(f"❌ Error navigating to {url}: {str(e)}")
            return {"error": f"Failed to navigate to {url}: {str(e)}"}
    
    def get_page_content(self, browser_id: str = None, content_type: str = "text") -> Dict[str, Any]:
        """Get the content of the current page"""
        logger.info(f"📄 Getting {content_type} content from browser: {browser_id or self.default_browser_id}")
        
        page = self.get_page(browser_id)
        if not page:
            error_msg = f"Page for browser {browser_id} not found or not responsive"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            current_url = page.url
            page_title = page.title()
            logger.info(f"Getting content from page: {page_title} ({current_url})")
            
            if content_type.lower() == "html":
                logger.info("Retrieving HTML content")
                content = page.content()
            else:  # text
                logger.info("Retrieving text content")
                content = page.evaluate("document.body.innerText")
            
            # Get page metadata
            metadata = {
                "url": current_url,
                "title": page_title,
                "content_length": len(content),
                "content_type": content_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Take a screenshot
            screenshot_path = self._save_screenshot(page, browser_id, "get_content")
            
            # Log a preview of the content
            content_preview = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"Content preview: {content_preview}")
            logger.info(f"Content length: {len(content)} characters")
            
            return {
                "status": "success",
                "content": content,
                "metadata": metadata,
                "screenshot": os.path.basename(screenshot_path) if screenshot_path else None
            }
        except Exception as e:
            logger.error(f"❌ Error getting page content: {str(e)}")
            return {"error": f"Failed to get page content: {str(e)}"}
    
    def find_elements(self, selector: str, browser_id: str = None) -> Dict[str, Any]:
        """Find elements using CSS selectors"""
        logger.info(f"🔍 Finding elements with selector: '{selector}' in browser: {browser_id or self.default_browser_id}")
        
        page = self.get_page(browser_id)
        if not page:
            error_msg = f"Page for browser {browser_id} not found or not responsive"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Find all elements matching the selector
            logger.info(f"Executing query selector: {selector}")
            elements = page.query_selector_all(selector)
            logger.info(f"Found {len(elements)} elements matching selector: {selector}")
            
            result = []
            for idx, element in enumerate(elements):
                # Get basic info about the element
                tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                
                # Get text content of the element
                try:
                    text = element.inner_text()
                    if len(text) > 100:
                        text = text[:100] + "..."
                except Exception as e:
                    logger.warning(f"Could not get text for element {idx}: {str(e)}")
                    text = ""
                
                # Get attributes of the element
                attributes = {}
                for attr in ["id", "class", "name", "href", "src", "alt", "title", "value"]:
                    try:
                        value = element.get_attribute(attr)
                        if value:
                            attributes[attr] = value
                    except Exception as e:
                        logger.warning(f"Could not get attribute {attr} for element {idx}: {str(e)}")
                
                # Get position and size
                try:
                    bounding_box = element.bounding_box()
                    position = {
                        "x": bounding_box["x"],
                        "y": bounding_box["y"],
                        "width": bounding_box["width"],
                        "height": bounding_box["height"]
                    }
                except Exception as e:
                    logger.warning(f"Could not get bounding box for element {idx}: {str(e)}")
                    position = {}
                
                result.append({
                    "index": idx,
                    "tag": tag_name,
                    "text": text,
                    "attributes": attributes,
                    "position": position
                })
                
                logger.info(f"Element {idx}: {tag_name} - '{text[:30]}...' - {attributes.get('href', '')}")
            
            # Take a screenshot
            screenshot_path = self._save_screenshot(page, browser_id, "find_elements")
            
            return {
                "status": "success",
                "count": len(result),
                "elements": result,
                "screenshot": os.path.basename(screenshot_path) if screenshot_path else None
            }
        except Exception as e:
            logger.error(f"❌ Error finding elements with selector '{selector}': {str(e)}")
            return {"error": f"Failed to find elements: {str(e)}"}
    
    def click_element(self, selector: str, index: int = 0, browser_id: str = None) -> Dict[str, Any]:
        """Click on an element using CSS selectors"""
        logger.info(f"🖱️ Clicking element with selector: '{selector}' at index: {index} in browser: {browser_id or self.default_browser_id}")
        
        page = self.get_page(browser_id)
        if not page:
            error_msg = f"Page for browser {browser_id} not found or not responsive"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Save screenshot before clicking
            before_screenshot = self._save_screenshot(page, browser_id, f"before_click_{index}")
            
            current_url = page.url
            current_title = page.title()
            logger.info(f"Current page before click: {current_title} ({current_url})")
            
            # Find all elements matching the selector
            logger.info(f"Finding elements matching selector: {selector}")
            elements = page.query_selector_all(selector)
            
            if not elements:
                error_msg = f"No elements found matching selector: {selector}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            if index >= len(elements):
                error_msg = f"Index {index} is out of range. Only {len(elements)} elements found."
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Get element info before clicking
            element = elements[index]
            tag_name = element.evaluate("el => el.tagName.toLowerCase()")
            try:
                text = element.inner_text()
                if len(text) > 100:
                    text = text[:100] + "..."
            except:
                text = ""
                
            # Get href if it's a link
            href = element.get_attribute("href") if tag_name == "a" else None
            
            logger.info(f"Found element to click: {tag_name} - '{text}' - href: {href}")
            
            # Scroll the element into view
            logger.info("Scrolling element into view")
            elements[index].scroll_into_view_if_needed()
            
            # Click the element
            logger.info(f"Clicking element at index {index}")
            start_time = time.time()
            elements[index].click()
            
            # Wait for navigation or network idle
            logger.info("Waiting for network idle after click")
            page.wait_for_load_state("networkidle")
            end_time = time.time()
            
            # Get updated page info
            new_url = page.url
            new_title = page.title()
            
            logger.info(f"Page after click: {new_title} ({new_url})")
            logger.info(f"Navigation after click took {end_time - start_time:.2f} seconds")
            
            # Check if URL changed
            url_changed = current_url != new_url
            if url_changed:
                logger.info(f"URL changed after click: {current_url} -> {new_url}")
            
            # Save screenshot after clicking
            after_screenshot = self._save_screenshot(page, browser_id, f"after_click_{index}")
            
            return {
                "status": "success",
                "message": f"Clicked element: {selector} at index {index}",
                "element_info": {
                    "tag": tag_name,
                    "text": text,
                    "href": href
                },
                "navigation": {
                    "url_changed": url_changed,
                    "previous_url": current_url,
                    "current_url": new_url,
                    "previous_title": current_title,
                    "current_title": new_title,
                    "navigation_time_seconds": round(end_time - start_time, 2)
                },
                "screenshots": {
                    "before": os.path.basename(before_screenshot) if before_screenshot else None,
                    "after": os.path.basename(after_screenshot) if after_screenshot else None
                }
            }
        except Exception as e:
            logger.error(f"❌ Error clicking element: {str(e)}")
            return {"error": f"Failed to click element: {str(e)}"}

# Global browser manager instance
BROWSER_MANAGER = BrowserSessionManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def browser_create(browser_id: str = None, browser_type: str = "chromium", headless: bool = True) -> Dict[str, Any]:
    """
    Create a new browser instance.
    
    Args:
        browser_id: Optional unique identifier for the browser
        browser_type: Type of browser ('chromium', 'firefox', or 'webkit')
        headless: Whether to run the browser in headless mode
    
    Returns:
        Dict with status and browser information
    """
    logger.info(f"Tool called: browser:create(browser_id={browser_id}, browser_type={browser_type}, headless={headless})")
    result = BROWSER_MANAGER.create_browser(browser_id, browser_type, headless)
    logger.info(f"browser:create result: {result.get('status', 'error')}")
    return result

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
    logger.info(f"Tool called: browser:navigate(url={url}, browser_id={browser_id}, wait_until={wait_until})")
    result = BROWSER_MANAGER.navigate(url, browser_id, wait_until)
    logger.info(f"browser:navigate result: {result.get('status', 'error')}")
    return result

def browser_get_content(browser_id: str = None, content_type: str = "text") -> Dict[str, Any]:
    """
    Get the content of the current page.
    
    Args:
        browser_id: Browser identifier (optional)
        content_type: Type of content to retrieve ('text' or 'html')
    
    Returns:
        Dict with page content and metadata
    """
    logger.info(f"Tool called: browser:get_content(browser_id={browser_id}, content_type={content_type})")
    result = BROWSER_MANAGER.get_page_content(browser_id, content_type)
    content_length = len(result.get('content', '')) if 'content' in result else 0
    logger.info(f"browser:get_content result: {result.get('status', 'error')}, content length: {content_length}")
    return result

def browser_find_elements(selector: str, browser_id: str = None) -> Dict[str, Any]:
    """
    Find elements on the page using CSS selectors.
    
    Args:
        selector: CSS selector string
        browser_id: Browser identifier (optional)
    
    Returns:
        Dict with elements found and their details
    """
    logger.info(f"Tool called: browser:find_elements(selector={selector}, browser_id={browser_id})")
    result = BROWSER_MANAGER.find_elements(selector, browser_id)
    element_count = result.get('count', 0) if 'count' in result else 0
    logger.info(f"browser:find_elements result: {result.get('status', 'error')}, found {element_count} elements")
    return result

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
    logger.info(f"Tool called: browser:click(selector={selector}, index={index}, browser_id={browser_id})")
    result = BROWSER_MANAGER.click_element(selector, index, browser_id)
    logger.info(f"browser:click result: {result.get('status', 'error')}")
    return result

def browser_close(browser_id: str = None) -> Dict[str, Any]:
    """
    Close a browser session.
    
    Args:
        browser_id: Browser identifier (optional)
    
    Returns:
        Dict with closure status
    """
    logger.info(f"Tool called: browser:close(browser_id={browser_id})")
    result = BROWSER_MANAGER.close_browser(browser_id)
    logger.info(f"browser:close result: {result.get('status', 'error')}")
    return result

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
    logger.info(f"🔧 Executing tool: {tool_id} with params: {kwargs}")
    if tool_id in TOOL_REGISTRY:
        handler = TOOL_REGISTRY[tool_id]
        try:
            result = handler(**kwargs)
            success = "error" not in result
            logger.info(f"Tool execution {'succeeded' if success else 'failed'}: {tool_id}")
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_id}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    else:
        error_msg = f"Unknown tool: {tool_id}"
        logger.error(error_msg)
        return {"error": error_msg}

# Store browser tool usage statistics
TOOL_USAGE_STATS = {
    "browser:create": 0,
    "browser:navigate": 0,
    "browser:get_content": 0,
    "browser:find_elements": 0,
    "browser:click": 0,
    "browser:close": 0,
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0
}

# Monkey patch original tool functions to track statistics
for tool_id in TOOL_REGISTRY:
    original_handler = TOOL_REGISTRY[tool_id]
    
    def create_tracking_handler(original, tool_name):
        def tracking_handler(**kwargs):
            TOOL_USAGE_STATS[tool_name] += 1
            TOOL_USAGE_STATS["total_calls"] += 1
            try:
                result = original(**kwargs)
                if "error" not in result:
                    TOOL_USAGE_STATS["successful_calls"] += 1
                else:
                    TOOL_USAGE_STATS["failed_calls"] += 1
                return result
            except Exception as e:
                TOOL_USAGE_STATS["failed_calls"] += 1
                raise e
        return tracking_handler
    
    TOOL_REGISTRY[tool_id] = create_tracking_handler(original_handler, tool_id)

def get_browser_stats():
    """Get browser tool usage statistics"""
    stats = TOOL_USAGE_STATS.copy()
    stats["active_browsers"] = len(BROWSER_MANAGER.browsers)
    stats["success_rate"] = (stats["successful_calls"] / stats["total_calls"] * 100) if stats["total_calls"] > 0 else 0
    return stats

# Add stats tool
TOOL_REGISTRY["browser:stats"] = get_browser_stats

# Print initialization message and tool usage instructions
logger.info("=" * 80)
logger.info("✅ Browser automation tools registered successfully")
logger.info(f"Available browser tools: {', '.join(sorted(TOOL_REGISTRY.keys()))}")
logger.info("Logs will be written to: " + log_file)
logger.info("Screenshots will be saved to: " + BROWSER_MANAGER.screenshot_dir)
logger.info("=" * 80)

print(f"✅ Browser automation tools registered successfully. Log file: {log_file}")
