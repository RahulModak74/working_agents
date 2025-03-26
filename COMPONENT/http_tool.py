#!/usr/bin/env python3

import requests
import json
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from urllib.parse import urlparse, urljoin


class HTTPTool:
    """Tool for making HTTP requests to REST APIs"""
    
    def __init__(self, base_url: str = None, headers: Dict[str, str] = None, 
                 auth: Tuple[str, str] = None, timeout: int = 30,
                 verify_ssl: bool = True):
        """Initialize with optional base URL and default headers"""
        self.base_url = base_url
        self.headers = headers or {}
        self.default_timeout = timeout
        self.auth = auth
        self.verify_ssl = verify_ssl
        self.last_response = None
        self.rate_limit_delay = 0  # Seconds between requests
        self.last_request_time = 0
    
    def _prepare_url(self, endpoint: str) -> str:
        """Prepare the full URL from the endpoint"""
        if not endpoint.startswith(('http://', 'https://')):
            if not self.base_url:
                raise ValueError("Base URL not set and endpoint is not a full URL")
            return urljoin(self.base_url, endpoint)
        return endpoint
    
    def _respect_rate_limit(self):
        """Respect rate limiting by waiting if needed"""
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
    
    def set_base_url(self, base_url: str):
        """Set or update the base URL"""
        self.base_url = base_url
    
    def set_headers(self, headers: Dict[str, str]):
        """Set or update the default headers"""
        self.headers = headers
    
    def add_header(self, key: str, value: str):
        """Add or update a single header"""
        self.headers[key] = value
    
    def set_auth(self, username: str, password: str):
        """Set basic authentication credentials"""
        self.auth = (username, password)
    
    def set_bearer_token(self, token: str):
        """Set bearer token authentication"""
        self.headers['Authorization'] = f"Bearer {token}"
    
    def set_rate_limit(self, requests_per_minute: int):
        """Set rate limiting for requests"""
        if requests_per_minute <= 0:
            self.rate_limit_delay = 0
        else:
            self.rate_limit_delay = 60.0 / requests_per_minute
    
    def get(self, endpoint: str, params: Dict[str, Any] = None, 
            headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a GET request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.get(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def post(self, endpoint: str, data: Dict[str, Any] = None, 
             json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
             headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a POST request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.post(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def put(self, endpoint: str, data: Dict[str, Any] = None, 
            json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
            headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a PUT request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.put(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def delete(self, endpoint: str, params: Dict[str, Any] = None, 
               headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a DELETE request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.delete(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def _process_response(self, response) -> Tuple[bool, Any]:
        """Process the response and return data in appropriate format"""
        success = 200 <= response.status_code < 300
        
        # Try to parse as JSON first
        try:
            data = response.json()
        except json.JSONDecodeError:
            # If not JSON, return text
            data = response.text
        
        if not success:
            return False, {
                "status_code": response.status_code,
                "error": data if isinstance(data, dict) else {"message": data},
                "headers": dict(response.headers)
            }
        
        return True, data
    
    def get_response_headers(self) -> Dict[str, str]:
        """Get headers from the last response"""
        if self.last_response:
            return dict(self.last_response.headers)
        return {}
    
    def get_status_code(self) -> Optional[int]:
        """Get status code from the last response"""
        if self.last_response:
            return self.last_response.status_code
        return None
    
    def download_file(self, endpoint: str, output_path: str, params: Dict[str, Any] = None,
                      headers: Dict[str, str] = None) -> Tuple[bool, str]:
        """Download a file from the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            response = requests.get(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                stream=True,
                verify=self.verify_ssl
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True, f"File downloaded successfully to {output_path}"
            else:
                return False, f"Failed to download file: HTTP {response.status_code}"
        except Exception as e:
            return False, f"Error downloading file: {str(e)}"
    
    def upload_file(self, endpoint: str, file_path: str, form_field: str = 'file',
                    extra_data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Tuple[bool, Any]:
        """Upload a file to the specified endpoint"""
        try:
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            files = {form_field: open(file_path, 'rb')}
            data = extra_data or {}
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.post(
                url,
                files=files,
                data=data,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=self.default_timeout,
                verify=self.verify_ssl
            )
            
            # Close the file
            for file_obj in files.values():
                file_obj.close()
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}


def get_http_tool(base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> HTTPTool:
    """Factory function to get an HTTP tool instance"""
    return HTTPTool(base_url, headers)
