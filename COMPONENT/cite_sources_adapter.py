#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import re
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cite_sources_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Configuration
class CitationConfig:
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citation_cache")
        self.cache_ttl = 86400 * 7  # 7 days in seconds
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.request_timeout = 15  # seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

# Global configuration
CITATION_CONFIG = CitationConfig()

# Try to import BeautifulSoup for metadata extraction
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. Install with 'pip install beautifulsoup4' for better metadata extraction")

class CitationManager:
    def __init__(self, config=None):
        self.config = config or CITATION_CONFIG
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
    
    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a URL for citation purposes"""
        # Check cache first
        cache_key = f"metadata_{hashlib.md5(url.encode()).hexdigest()}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Fetch the URL
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            # Initialize metadata with basic information
            metadata = {
                'url': url,
                'accessed_date': datetime.now().strftime('%Y-%m-%d'),
                'domain': self._extract_domain(url)
            }
            
            # Extract more detailed metadata if BeautifulSoup is available
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = None
                if soup.title:
                    title = soup.title.string.strip()
                if not title:
                    og_title = soup.find('meta', property='og:title')
                    if og_title:
                        title = og_title.get('content', '').strip()
                metadata['title'] = title or "Untitled Document"
                
                # Extract authors
                authors = []
                # Method 1: Check for standard author meta tags
                author_meta = soup.find('meta', attrs={'name': 'author'})
                if author_meta and author_meta.get('content'):
                    authors.append(author_meta.get('content').strip())
                
                # Method 2: Check for schema.org author markup
                author_elements = soup.select('[itemprop="author"]')
                for element in author_elements:
                    name_element = element.select_one('[itemprop="name"]')
                    if name_element:
                        authors.append(name_element.text.strip())
                    elif element.text.strip():
                        authors.append(element.text.strip())
                
                # Method 3: Check for common author class names
                for author_class in ['author', 'byline', 'contributor']:
                    elements = soup.select(f'.{author_class}')
                    for element in elements:
                        if element.text.strip() and len(element.text.strip()) < 100:  # Avoid getting large blocks
                            authors.append(element.text.strip())
                
                metadata['authors'] = list(set(authors))  # Remove duplicates
                
                # Extract publication date
                pub_date = None
                # Method 1: Check for standard date meta tags
                for date_meta in ['published_time', 'article:published_time', 'date', 'pubdate']:
                    date_element = soup.find('meta', property=date_meta) or soup.find('meta', attrs={'name': date_meta})
                    if date_element and date_element.get('content'):
                        pub_date = date_element.get('content').strip()
                        break
                
                # Method 2: Check for time elements
                if not pub_date:
                    time_elements = soup.select('time')
                    for time_element in time_elements:
                        if time_element.get('datetime'):
                            pub_date = time_element.get('datetime').strip()
                            break
                
                # Method 3: Check for common date class names
                if not pub_date:
                    for date_class in ['date', 'published', 'pubdate', 'timestamp']:
                        elements = soup.select(f'.{date_class}')
                        for element in elements:
                            if element.text.strip() and len(element.text.strip()) < 50:
                                pub_date = element.text.strip()
                                break
                        if pub_date:
                            break
                
                metadata['publication_date'] = pub_date or "n.d."  # "no date" if not found
                
                # Extract publisher/site name
                publisher = None
                # Method 1: Check for standard publisher meta tags
                for publisher_meta in ['og:site_name', 'application-name', 'publisher']:
                    publisher_element = soup.find('meta', property=publisher_meta) or soup.find('meta', attrs={'name': publisher_meta})
                    if publisher_element and publisher_element.get('content'):
                        publisher = publisher_element.get('content').strip()
                        break
                
                # Method 2: Use domain as fallback
                if not publisher:
                    publisher = metadata['domain']
                
                metadata['publisher'] = publisher
            else:
                # If BeautifulSoup is not available, use basic information
                metadata['title'] = "Content from " + metadata['domain']
                metadata['authors'] = []
                metadata['publication_date'] = "n.d."
                metadata['publisher'] = metadata['domain']
            
            # Cache the result
            self._save_to_cache(cache_key, metadata)
            
            return {
                'status': 'success',
                'metadata': metadata
            }
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'url': url
            }
    
    def format_citation(self, metadata: Dict[str, Any], style: str = 'apa') -> Dict[str, Any]:
        """Format a citation based on the provided metadata and citation style"""
        try:
            if not metadata or not isinstance(metadata, dict):
                return {
                    'status': 'error',
                    'error': 'Invalid or missing metadata'
                }
            
            # Get core metadata fields with fallbacks
            url = metadata.get('url', '')
            title = metadata.get('title', 'Untitled Document')
            authors = metadata.get('authors', [])
            pub_date = metadata.get('publication_date', 'n.d.')
            publisher = metadata.get('publisher', '')
            accessed_date = metadata.get('accessed_date', datetime.now().strftime('%Y-%m-%d'))
            
            # Format authors based on the style
            formatted_authors = ''
            if style.lower() == 'apa':
                if not authors:
                    formatted_authors = ''
                elif len(authors) == 1:
                    # Format: Last, F.
                    author = authors[0]
                    if ',' in author:
                        formatted_authors = author  # Already in Last, First format
                    else:
                        parts = author.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            # Get initials
                            initials = ''.join([name[0] + '.' for name in firsts.split() if name])
                            formatted_authors = f"{last}, {initials}"
                        else:
                            formatted_authors = author
                else:
                    # Format multiple authors
                    author_list = []
                    for author in authors[:6]:  # APA uses up to 6 authors
                        if ',' in author:
                            author_list.append(author)
                        else:
                            parts = author.split()
                            if len(parts) > 1:
                                last = parts[-1]
                                firsts = ' '.join(parts[:-1])
                                initials = ''.join([name[0] + '.' for name in firsts.split() if name])
                                author_list.append(f"{last}, {initials}")
                            else:
                                author_list.append(author)
                    
                    if len(authors) <= 7:
                        formatted_authors = ', '.join(author_list[:-1])
                        if len(author_list) > 1:
                            formatted_authors += f", & {author_list[-1]}"
                        else:
                            formatted_authors = author_list[0]
                    else:
                        # For 8+ authors, list first 6 followed by "..." and the last author
                        formatted_authors = ', '.join(author_list[:6]) + f", ... {author_list[-1]}"
            
            elif style.lower() == 'mla':
                if not authors:
                    formatted_authors = ''
                elif len(authors) == 1:
                    # Format: Last, First
                    author = authors[0]
                    if ',' in author:
                        formatted_authors = author  # Already in Last, First format
                    else:
                        parts = author.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            formatted_authors = f"{last}, {firsts}"
                        else:
                            formatted_authors = author
                elif len(authors) == 2:
                    # Format: Last1, First1, and First2 Last2
                    author1 = authors[0]
                    author2 = authors[1]
                    
                    if ',' in author1:
                        formatted_authors = author1
                    else:
                        parts = author1.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            formatted_authors = f"{last}, {firsts}"
                        else:
                            formatted_authors = author1
                    
                    if ',' in author2:
                        # Reverse the order for second author
                        name_parts = author2.split(', ', 1)
                        if len(name_parts) > 1:
                            formatted_authors += f", and {name_parts[1]} {name_parts[0]}"
                        else:
                            formatted_authors += f", and {author2}"
                    else:
                        formatted_authors += f", and {author2}"
                else:
                    # Format: Last, First, et al.
                    author = authors[0]
                    if ',' in author:
                        formatted_authors = author + ", et al."
                    else:
                        parts = author.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            formatted_authors = f"{last}, {firsts}, et al."
                        else:
                            formatted_authors = author + ", et al."
            
            elif style.lower() == 'chicago':
                if not authors:
                    formatted_authors = ''
                elif len(authors) == 1:
                    # Format: Last, First
                    author = authors[0]
                    if ',' in author:
                        formatted_authors = author  # Already in Last, First format
                    else:
                        parts = author.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            formatted_authors = f"{last}, {firsts}"
                        else:
                            formatted_authors = author
                else:
                    # Format first author as Last, First and others as First Last
                    author1 = authors[0]
                    if ',' in author1:
                        formatted_authors = author1
                    else:
                        parts = author1.split()
                        if len(parts) > 1:
                            last = parts[-1]
                            firsts = ' '.join(parts[:-1])
                            formatted_authors = f"{last}, {firsts}"
                        else:
                            formatted_authors = author1
                    
                    # Add remaining authors
                    for author in authors[1:]:
                        formatted_authors += f", {author}"
            
            else:  # Default/generic style
                if authors:
                    formatted_authors = ', '.join(authors)
            
            # Format the date
            formatted_date = pub_date
            try:
                # Try to parse and reformat the date if it looks like a standard format
                if re.match(r'\d{4}-\d{2}-\d{2}', pub_date):
                    date_obj = datetime.strptime(pub_date, '%Y-%m-%d')
                    if style.lower() == 'apa':
                        formatted_date = date_obj.strftime('%Y, %B %d')
                    elif style.lower() == 'mla':
                        formatted_date = date_obj.strftime('%d %b. %Y')
                    elif style.lower() == 'chicago':
                        formatted_date = date_obj.strftime('%B %d, %Y')
                elif re.match(r'\d{4}-\d{2}', pub_date):
                    date_obj = datetime.strptime(pub_date, '%Y-%m')
                    if style.lower() == 'apa':
                        formatted_date = date_obj.strftime('%Y, %B')
                    elif style.lower() == 'mla':
                        formatted_date = date_obj.strftime('%b. %Y')
                    elif style.lower() == 'chicago':
                        formatted_date = date_obj.strftime('%B %Y')
                elif re.match(r'\d{4}', pub_date):
                    if style.lower() in ['apa', 'chicago', 'mla']:
                        formatted_date = pub_date  # Just use the year
            except:
                # If date parsing fails, use the original string
                formatted_date = pub_date
            
            # Now assemble the full citation based on the style
            citation = ""
            if style.lower() == 'apa':
                if formatted_authors:
                    citation += f"{formatted_authors}. "
                citation += f"({formatted_date}). "
                citation += f"{title}. "
                if publisher:
                    citation += f"{publisher}. "
                citation += f"Retrieved {accessed_date}, from {url}"
            
            elif style.lower() == 'mla':
                if formatted_authors:
                    citation += f"{formatted_authors}. "
                citation += f"\"{title}.\" "
                if publisher:
                    citation += f"{publisher}, "
                citation += f"{formatted_date}, {url}. Accessed {accessed_date}."
            
            elif style.lower() == 'chicago':
                if formatted_authors:
                    citation += f"{formatted_authors}. "
                citation += f"\"{title}.\" "
                if publisher:
                    citation += f"{publisher}, "
                citation += f"{formatted_date}. {url}."
            
            else:  # Default/generic style
                if formatted_authors:
                    citation += f"{formatted_authors}. "
                citation += f"{title}. "
                if formatted_date != 'n.d.':
                    citation += f"{formatted_date}. "
                if publisher:
                    citation += f"{publisher}. "
                citation += f"{url} (Accessed: {accessed_date})"
            
            return {
                'status': 'success',
                'citation': citation,
                'style': style,
                'metadata': metadata
            }
                
        except Exception as e:
            logger.error(f"Error formatting citation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def validate_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a list of sources"""
        if not isinstance(sources, list):
            return {
                'status': 'error',
                'error': 'Sources must be provided as a list'
            }
        
        validated_sources = []
        invalid_sources = []
        
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                invalid_sources.append({
                    'index': i,
                    'source': source,
                    'reason': 'Source must be a dictionary'
                })
                continue
            
            # Check for required fields
            missing_fields = []
            for field in ['url']:
                if field not in source or not source[field]:
                    missing_fields.append(field)
            
            if missing_fields:
                invalid_sources.append({
                    'index': i,
                    'source': source,
                    'reason': f"Missing required fields: {', '.join(missing_fields)}"
                })
                continue
            
            # Validate URL
            url = source.get('url', '')
            if not self._is_valid_url(url):
                invalid_sources.append({
                    'index': i,
                    'source': source,
                    'reason': 'Invalid URL format'
                })
                continue
            
            # Source is valid, add any missing metadata
            validated_source = source.copy()
            if 'title' not in validated_source or not validated_source['title']:
                # Extract domain as a fallback title
                domain = self._extract_domain(url)
                validated_source['title'] = f"Content from {domain}"
            
            if 'accessed_date' not in validated_source or not validated_source['accessed_date']:
                validated_source['accessed_date'] = datetime.now().strftime('%Y-%m-%d')
            
            validated_sources.append(validated_source)
        
        return {
            'status': 'success',
            'validated_count': len(validated_sources),
            'invalid_count': len(invalid_sources),
            'validated_sources': validated_sources,
            'invalid_sources': invalid_sources
        }
    
    def generate_bibliography(self, sources: List[Dict[str, Any]], style: str = 'apa') -> Dict[str, Any]:
        """Generate a bibliography from a list of sources"""
        if not isinstance(sources, list):
            return {
                'status': 'error',
                'error': 'Sources must be provided as a list'
            }
        
        # Validate sources first
        validation_result = self.validate_sources(sources)
        if validation_result.get('status') != 'success':
            return validation_result
        
        valid_sources = validation_result['validated_sources']
        citations = []
        
        for source in valid_sources:
            # Check if we need to extract metadata
            if 'metadata' not in source or not source['metadata']:
                url = source.get('url', '')
                metadata_result = self.extract_metadata(url)
                if metadata_result.get('status') == 'success':
                    metadata = metadata_result['metadata']
                else:
                    # Use basic metadata from validation
                    metadata = {
                        'url': url,
                        'title': source.get('title', ''),
                        'authors': source.get('authors', []),
                        'publication_date': source.get('publication_date', 'n.d.'),
                        'publisher': source.get('publisher', ''),
                        'accessed_date': source.get('accessed_date', datetime.now().strftime('%Y-%m-%d'))
                    }
            else:
                metadata = source['metadata']
            
            # Format the citation
            citation_result = self.format_citation(metadata, style)
            if citation_result.get('status') == 'success':
                citations.append({
                    'source': source,
                    'citation': citation_result['citation'],
                    'metadata': metadata
                })
            else:
                # Create a basic citation if formatting failed
                url = source.get('url', '')
                title = source.get('title', 'Untitled Document')
                basic_citation = f"{title}. {url}"
                citations.append({
                    'source': source,
                    'citation': basic_citation,
                    'metadata': metadata,
                    'warning': 'Failed to format citation properly'
                })
        
        # Sort citations alphabetically
        sorted_citations = sorted(citations, key=lambda x: x['citation'].lower())
        
        return {
            'status': 'success',
            'style': style,
            'citation_count': len(sorted_citations),
            'citations': sorted_citations,
            'bibliography': '\n\n'.join([item['citation'] for item in sorted_citations])
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
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

# Global citation manager
CITATION_MANAGER = CitationManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def cite_source(url: str, style: str = 'apa') -> Dict[str, Any]:
    """
    Generate a citation for a single URL.
    
    Args:
        url: The URL to cite
        style: Citation style ('apa', 'mla', 'chicago')
    
    Returns:
        Dict with citation information
    """
    # First extract metadata
    metadata_result = CITATION_MANAGER.extract_metadata(url)
    
    if metadata_result.get('status') != 'success':
        return metadata_result
    
    # Then format citation
    return CITATION_MANAGER.format_citation(metadata_result['metadata'], style)

def cite_sources(sources: List[Dict[str, Any]], style: str = 'apa') -> Dict[str, Any]:
    """
    Generate citations for multiple sources.
    
    Args:
        sources: List of source dictionaries with at least a 'url' field
        style: Citation style ('apa', 'mla', 'chicago')
    
    Returns:
        Dict with citation information for all sources
    """
    return CITATION_MANAGER.generate_bibliography(sources, style)

def validate_sources(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a list of sources.
    
    Args:
        sources: List of source dictionaries to validate
    
    Returns:
        Dict with validation results
    """
    return CITATION_MANAGER.validate_sources(sources)

def format_citations(citations: List[Dict[str, Any]], style: str = 'apa', format: str = 'markdown') -> Dict[str, Any]:
    """
    Format a list of citations into a bibliography.
    
    Args:
        citations: List of citation dictionaries
        style: Citation style ('apa', 'mla', 'chicago')
        format: Output format ('markdown', 'text', 'html')
    
    Returns:
        Dict with formatted bibliography
    """
    try:
        # Convert citations to sources format if needed
        sources = []
        for citation in citations:
            if isinstance(citation, dict):
                if 'url' in citation:
                    sources.append(citation)
                elif 'link' in citation:
                    # Convert from search result format
                    sources.append({
                        'url': citation.get('link', ''),
                        'title': citation.get('title', ''),
                        'publisher': citation.get('source', '')
                    })
                elif 'source' in citation and isinstance(citation['source'], dict):
                    # Already in the right format
                    sources.append(citation['source'])
            elif isinstance(citation, str) and CITATION_MANAGER._is_valid_url(citation):
                # Plain URL
                sources.append({'url': citation})
        
        if not sources:
            return {
                'status': 'error',
                'error': 'No valid citation sources found'
            }
        
        # Generate bibliography
        bib_result = CITATION_MANAGER.generate_bibliography(sources, style)
        
        if bib_result.get('status') != 'success':
            return bib_result
        
        # Format the bibliography according to the requested format
        citations = bib_result['citations']
        
        if format.lower() == 'html':
            html_output = "<h2>Bibliography</h2>\n<ul>"
            for item in citations:
                html_output += f"\n  <li>{item['citation']}</li>"
            html_output += "\n</ul>"
            bib_result['formatted_bibliography'] = html_output
        
        elif format.lower() == 'markdown':
            md_output = "## Bibliography\n\n"
            for item in citations:
                md_output += f"- {item['citation']}\n\n"
            bib_result['formatted_bibliography'] = md_output
        
        else:  # text
            text_output = "BIBLIOGRAPHY\n\n"
            for item in citations:
                text_output += f"* {item['citation']}\n\n"
            bib_result['formatted_bibliography'] = text_output
        
        return bib_result
        
    except Exception as e:
        logger.error(f"Error formatting citations: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

# Register tools
TOOL_REGISTRY["research:cite_source"] = cite_source
TOOL_REGISTRY["research:cite_sources"] = cite_sources
TOOL_REGISTRY["research:validate_sources"] = validate_sources
TOOL_REGISTRY["research:format_citations"] = format_citations

# Function for executing citation tools
def execute_tool(tool_id: str, **kwargs):
    """Execute a citation tool based on the tool ID"""
    if tool_id in TOOL_REGISTRY:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    else:
        return {"error": f"Unknown tool: {tool_id}"}

# Print initialization message
print("✅ Citation and source tools registered successfully")
