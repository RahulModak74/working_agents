#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import requests
import urllib.parse
import re
import hashlib
import base64
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("research_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. Install with 'pip install beautifulsoup4' for better HTML parsing")

# Try to import nltk for text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with 'pip install nltk' for better text analysis")

# Configuration
class ResearchConfig:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_cache")
        self.cache_ttl = 86400  # 24 hours in seconds
        self.request_timeout = 15  # seconds
        self.max_results = 10
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        self.bing_api_key = os.environ.get("BING_API_KEY", "")
        self.search_engine = "serper"  # or "bing" or "duckduckgo"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

# Global configuration
RESEARCH_CONFIG = ResearchConfig()

class ResearchManager:
    def __init__(self, config=None):
        self.config = config or RESEARCH_CONFIG
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
    
    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search using the configured search engine"""
        if self.config.search_engine == "serper" and self.config.serper_api_key:
            return self._search_serper(query, num_results)
        elif self.config.search_engine == "bing" and self.config.bing_api_key:
            return self._search_bing(query, num_results)
        else:
            return self._search_duckduckgo(query, num_results)
    
    def _search_serper(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Serper.dev API"""
        # Check cache first
        cache_key = f"serper_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': self.config.serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'num': min(num_results, self.config.max_results)
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            # Process and format results
            results = []
            if 'organic' in data:
                for item in data['organic'][:num_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Google (via Serper.dev)',
                        'position': item.get('position', 0)
                    })
            
            result = {
                'status': 'success',
                'query': query,
                'num_results': len(results),
                'results': results
            }
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error with Serper search: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'results': []
            }
    
    def _search_bing(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Bing API"""
        # Check cache first
        cache_key = f"bing_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {
                'Ocp-Apim-Subscription-Key': self.config.bing_api_key
            }
            params = {
                'q': query,
                'count': min(num_results, self.config.max_results),
                'responseFilter': 'Webpages'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            # Process and format results
            results = []
            if 'webPages' in data and 'value' in data['webPages']:
                for item in data['webPages']['value'][:num_results]:
                    results.append({
                        'title': item.get('name', ''),
                        'link': item.get('url', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Bing Search API',
                        'position': len(results) + 1
                    })
            
            result = {
                'status': 'success',
                'query': query,
                'num_results': len(results),
                'results': results
            }
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error with Bing search: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'results': []
            }
    
    def _search_duckduckgo(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo (no API key required)"""
        # Check cache first
        cache_key = f"ddg_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            url = "https://html.duckduckgo.com/html/"
            headers = {
                'User-Agent': self.config.user_agent
            }
            data = {
                'q': query,
                'kl': 'us-en'
            }
            
            response = requests.post(url, headers=headers, data=data, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            results = []
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                result_elements = soup.select('.result')[:num_results]
                
                for i, element in enumerate(result_elements):
                    title_element = element.select_one('.result__title')
                    link_element = element.select_one('.result__url')
                    snippet_element = element.select_one('.result__snippet')
                    
                    title = title_element.text.strip() if title_element else ''
                    
                    # Extract actual link from the redirect URL
                    link = ''
                    if link_element:
                        link = link_element.text.strip()
                    elif title_element and title_element.find('a'):
                        href = title_element.find('a').get('href', '')
                        if href.startswith('/'):
                            link_match = re.search(r'uddg=([^&]+)', href)
                            if link_match:
                                link = urllib.parse.unquote(link_match.group(1))
                    
                    snippet = snippet_element.text.strip() if snippet_element else ''
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet,
                        'source': 'DuckDuckGo',
                        'position': i + 1
                    })
            
            result = {
                'status': 'success',
                'query': query,
                'num_results': len(results),
                'results': results
            }
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error with DuckDuckGo search: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'results': []
            }
    
    def fetch_content(self, url: str, format: str = 'text') -> Dict[str, Any]:
        """Fetch and extract content from a URL"""
        # Check cache first
        cache_key = f"content_{hashlib.md5(url.encode()).hexdigest()}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            # If content is too large to include in the response, truncate it
            if 'content' in cached_result and len(cached_result['content']) > 100000:
                cached_result['content'] = cached_result['content'][:100000] + '... [content truncated]'
            return cached_result
        
        try:
            # Fetch the URL
            headers = {'User-Agent': self.config.user_agent}
            response = requests.get(url, headers=headers, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
                # For non-HTML content, just return the raw text
                return {
                    'status': 'success',
                    'url': url,
                    'title': os.path.basename(url),
                    'content_type': content_type,
                    'content': response.text[:100000] + ('...' if len(response.text) > 100000 else ''),
                    'content_format': 'text'
                }
            
            # Parse and extract content
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "iframe", "noscript"]):
                    script.decompose()
                
                # Get the page title
                title = soup.title.string.strip() if soup.title else os.path.basename(url)
                
                # Extract metadata
                metadata = {
                    'url': url,
                    'title': title,
                    'content_type': content_type,
                }
                
                # Extract meta tags
                meta_tags = {}
                for tag in soup.find_all('meta'):
                    if tag.get('name'):
                        meta_tags[tag.get('name')] = tag.get('content', '')
                    elif tag.get('property'):
                        meta_tags[tag.get('property')] = tag.get('content', '')
                
                if meta_tags:
                    metadata['meta_tags'] = meta_tags
                
                if format == 'html':
                    # For HTML format, return cleaned HTML
                    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
                        comment.extract()
                    
                    # Keep only the main content tags
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'}) or soup.find('div', {'class': 'content'}) or soup.body
                    
                    content = str(main_content) if main_content else str(soup.body)
                    content_format = 'html'
                else:
                    # For text format, extract clean text
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'}) or soup.find('div', {'class': 'content'}) or soup.body
                    
                    if main_content:
                        text = main_content.get_text(separator=' ', strip=True)
                    else:
                        text = soup.get_text(separator=' ', strip=True)
                    
                    # Clean up the text: remove extra whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    content = text
                    content_format = 'text'
                
                # Truncate very long content
                if len(content) > 100000:
                    content = content[:100000] + '... [content truncated]'
                
                result = {
                    'status': 'success',
                    'url': url,
                    'title': title,
                    'content': content,
                    'content_format': content_format,
                    'metadata': metadata
                }
                
                # Cache the result
                self._save_to_cache(cache_key, result)
                
                return result
            else:
                # If BeautifulSoup is not available, return the raw HTML
                return {
                    'status': 'success',
                    'url': url,
                    'title': url.split('/')[-1],
                    'content': response.text[:100000] + ('...' if len(response.text) > 100000 else ''),
                    'content_format': 'text',
                    'warning': 'BeautifulSoup not available for better parsing'
                }
                
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'url': url
            }
    
    def analyze_content(self, content: str, max_length: int = 5000) -> Dict[str, Any]:
        """Analyze the content and extract key information"""
        if not content:
            return {
                'status': 'error',
                'error': 'No content provided for analysis'
            }
        
        try:
            # Truncate content if needed
            if len(content) > max_length:
                content = content[:max_length] + '... [content truncated]'
            
            # Calculate basic metrics
            word_count = len(content.split())
            
            # Extract key sentences
            key_sentences = []
            
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(content)
                stop_words = set(stopwords.words('english'))
                
                # Calculate word frequency (excluding stop words)
                word_freq = {}
                for word in content.lower().split():
                    word = re.sub(r'[^\w\s]', '', word)
                    if word and word not in stop_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Score sentences based on word frequency
                sentence_scores = {}
                for i, sentence in enumerate(sentences):
                    for word in sentence.lower().split():
                        word = re.sub(r'[^\w\s]', '', word)
                        if word in word_freq:
                            sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]
                
                # Get top sentences
                top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                key_sentences = [sentences[i] for i, _ in top_sentences]
            else:
                # Simple fallback if NLTK is not available
                sentences = re.split(r'(?<=[.!?])\s+', content)
                # Get sentences that are likely to be informative (not too short, not too long)
                key_sentences = [s for s in sentences if 10 < len(s) < 200][:5]
            
            # Identify key topics based on word frequency
            key_topics = []
            if NLTK_AVAILABLE:
                # Use the most frequent non-stop words as topics
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                key_topics = [word for word, _ in top_words]
            
            return {
                'status': 'success',
                'word_count': word_count,
                'key_sentences': key_sentences,
                'key_topics': key_topics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_research_summary(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Generate a summary of the research findings"""
        try:
            # Combine all content
            all_content = []
            url_titles = {}
            
            for result in results:
                if 'content' in result and result['content']:
                    all_content.append(result['content'])
                if 'url' in result and 'title' in result:
                    url_titles[result['url']] = result['title']
            
            combined_content = " ".join(all_content)
            
            # Analyze the combined content
            analysis = self.analyze_content(combined_content)
            
            # Create source list
            sources = [{"url": url, "title": title} for url, title in url_titles.items()]
            
            return {
                'status': 'success',
                'query': query,
                'num_sources': len(sources),
                'sources': sources,
                'key_findings': analysis.get('key_sentences', []),
                'key_topics': analysis.get('key_topics', []),
                'total_word_count': analysis.get('word_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error generating research summary: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if available and not expired"""
        cache_file = os.path.join(self.config.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if cache has expired
                timestamp = data.get('_timestamp', 0)
                if time.time() - timestamp <= self.config.cache_ttl:
                    # Remove the timestamp field before returning
                    data.pop('_timestamp', None)
                    return data
            except Exception as e:
                logger.warning(f"Error reading from cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]) -> bool:
        """Save data to cache with timestamp"""
        cache_file = os.path.join(self.config.cache_dir, f"{key}.json")
        
        try:
            # Add timestamp
            data_with_timestamp = data.copy()
            data_with_timestamp['_timestamp'] = time.time()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data_with_timestamp, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            return False

# Global research manager
RESEARCH_MANAGER = ResearchManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def research_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a web search using the configured search engine.
    
    Args:
        query: The search query
        num_results: Maximum number of results to return
    
    Returns:
        Dict with search results
    """
    return RESEARCH_MANAGER.search(query, num_results)

def research_fetch_content(url: str, format: str = 'text') -> Dict[str, Any]:
    """
    Fetch and extract content from a URL.
    
    Args:
        url: The URL to fetch
        format: Content format ('text' or 'html')
    
    Returns:
        Dict with the content and metadata
    """
    return RESEARCH_MANAGER.fetch_content(url, format)

def research_analyze_content(content: str, max_length: int = 5000) -> Dict[str, Any]:
    """
    Analyze content to extract key information.
    
    Args:
        content: The text content to analyze
        max_length: Maximum content length to analyze
    
    Returns:
        Dict with analysis results
    """
    return RESEARCH_MANAGER.analyze_content(content, max_length)

def research_generate_summary(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Generate a summary of research findings from multiple sources.
    
    Args:
        results: List of research results
        query: The original research query
    
    Returns:
        Dict with summary information
    """
    return RESEARCH_MANAGER.generate_research_summary(results, query)

def research_combined_search(query: str, depth: int = 1, num_results: int = 3) -> Dict[str, Any]:
    """
    Perform a comprehensive search, fetching content from top results.
    
    Args:
        query: The search query
        depth: Depth of research (1-3)
        num_results: Number of search results to process
    
    Returns:
        Dict with research findings
    """
    # Limit parameters to reasonable values
    depth = min(max(depth, 1), 3)
    num_results = min(max(num_results, 1), 10)
    
    try:
        # Step 1: Search for the query
        search_results = RESEARCH_MANAGER.search(query, num_results)
        
        if search_results.get('status') == 'error':
            return search_results
        
        # Step 2: Fetch content from each search result
        content_results = []
        raw_contents = []
        
        for result in search_results.get('results', []):
            if 'link' in result and result['link']:
                content = RESEARCH_MANAGER.fetch_content(result['link'])
                if content.get('status') == 'success' and 'content' in content:
                    content_results.append(content)
                    raw_contents.append(content['content'])
        
        # Step 3: For depth > 1, search for follow-up queries based on initial findings
        if depth >= 2 and NLTK_AVAILABLE and raw_contents:
            # Combine all content
            combined_content = " ".join(raw_contents)
            
            # Extract key topics for follow-up searches
            analysis = RESEARCH_MANAGER.analyze_content(combined_content)
            key_topics = analysis.get('key_topics', [])
            
            # Generate follow-up queries
            follow_up_queries = []
            for topic in key_topics[:2]:  # Limit to top 2 topics
                follow_up_query = f"{query} {topic}"
                follow_up_results = RESEARCH_MANAGER.search(follow_up_query, 2)
                
                if follow_up_results.get('status') == 'success':
                    for result in follow_up_results.get('results', []):
                        if 'link' in result and result['link']:
                            # Check if we already have this URL
                            if not any(res.get('url') == result['link'] for res in content_results):
                                content = RESEARCH_MANAGER.fetch_content(result['link'])
                                if content.get('status') == 'success' and 'content' in content:
                                    content_results.append(content)
                                    raw_contents.append(content['content'])
        
        # Step 4: Generate a summary of all findings
        summary = RESEARCH_MANAGER.generate_research_summary(content_results, query)
        
        return {
            'status': 'success',
            'query': query,
            'search_results': search_results.get('results', []),
            'content_results': [
                {
                    'url': res.get('url', ''),
                    'title': res.get('title', ''),
                    'content_length': len(res.get('content', ''))
                }
                for res in content_results
            ],
            'summary': summary,
            'research_depth': depth
        }
        
    except Exception as e:
        logger.error(f"Error in combined research: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

# Register tools
TOOL_REGISTRY["research:search"] = research_search
TOOL_REGISTRY["research:fetch_content"] = research_fetch_content
TOOL_REGISTRY["research:analyze_content"] = research_analyze_content
TOOL_REGISTRY["research:generate_summary"] = research_generate_summary
TOOL_REGISTRY["research:combined_search"] = research_combined_search

# Function for executing research tools
def execute_tool(tool_id: str, **kwargs):
    """Execute a research tool based on the tool ID"""
    if tool_id in TOOL_REGISTRY:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    else:
        return {"error": f"Unknown tool: {tool_id}"}

# Print initialization message
print("✅ Deep research tools registered successfully")
