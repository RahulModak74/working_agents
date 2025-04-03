#!/usr/bin/env python3

import os
import time
import json
import logging
import paramiko
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("privesc-tools")

# Tool registry to store available tools
TOOL_REGISTRY = {}

class SSHTool:
    """SSH tool for connecting to and interacting with remote systems"""
    
    def __init__(self):
        self.connections = {}  # Cache of SSH connections
        self.shells = {}       # Cache of interactive shells
    
    def connect(self, hostname: str, port: int, username: str, password: str, 
                key_file: Optional[str] = None) -> Dict[str, Any]:
        """Connect to a remote system via SSH"""
        connection_id = f"{username}@{hostname}:{port}"
        
        # Return existing connection if available
        if connection_id in self.connections:
            return {
                "status": "success",
                "message": "Already connected",
                "connection_id": connection_id
            }
        
        try:
            # Create SSH client
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect with password or key file
            if key_file:
                client.connect(
                    hostname=hostname,
                    port=port,
                    username=username,
                    key_filename=key_file,
                    timeout=10
                )
            else:
                client.connect(
                    hostname=hostname,
                    port=port,
                    username=username,
                    password=password,
                    timeout=10
                )
            
            # Store the connection
            self.connections[connection_id] = client
            
            # Create an interactive shell
            shell = client.invoke_shell()
            shell.settimeout(10)
            
            # Receive the initial prompt
            output = b""
            while not output.endswith(b"$ ") and not output.endswith(b"# "):
                if shell.recv_ready():
                    chunk = shell.recv(4096)
                    output += chunk
                time.sleep(0.1)
            
            # Store the shell
            self.shells[connection_id] = shell
            
            return {
                "status": "success",
                "message": "Connection established",
                "connection_id": connection_id,
                "banner": output.decode('utf-8', errors='ignore')
            }
            
        except Exception as e:
            logger.error(f"SSH connection error: {str(e)}")
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}"
            }
    
    def execute(self, connection_id: str, command: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute a command on the remote system"""
        if connection_id not in self.connections:
            return {
                "status": "error",
                "message": "No active connection"
            }
        
        try:
            # Get the client
            client = self.connections[connection_id]
            
            # Execute the command
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Get the output
            stdout_data = stdout.read().decode('utf-8', errors='ignore')
            stderr_data = stderr.read().decode('utf-8', errors='ignore')
            exit_code = stdout.channel.recv_exit_status()
            
            return {
                "status": "success",
                "command": command,
                "stdout": stdout_data,
                "stderr": stderr_data,
                "exit_code": exit_code
            }
            
        except Exception as e:
            logger.error(f"SSH execution error: {str(e)}")
            return {
                "status": "error",
                "message": f"Command execution failed: {str(e)}"
            }
    
    def execute_shell(self, connection_id: str, command: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute a command on the interactive shell"""
        if connection_id not in self.shells:
            return {
                "status": "error",
                "message": "No active shell"
            }
        
        try:
            # Get the shell
            shell = self.shells[connection_id]
            
            # Send the command
            shell.send(command + "\n")
            
            # Wait for the output
            time.sleep(0.5)
            output = b""
            start_time = time.time()
            
            while True:
                if shell.recv_ready():
                    chunk = shell.recv(4096)
                    output += chunk
                
                # Check for command completion (prompt or timeout)
                if output.endswith(b"$ ") or output.endswith(b"# "):
                    break
                
                if time.time() - start_time > timeout:
                    break
                
                time.sleep(0.1)
            
            # Process the output (remove the command and prompt from output)
            output_str = output.decode('utf-8', errors='ignore')
            lines = output_str.splitlines()
            
            # Remove the first line (command) and the last line (prompt)
            if len(lines) > 2:
                result = "\n".join(lines[1:-1])
            else:
                result = output_str
            
            return {
                "status": "success",
                "command": command,
                "raw_output": output_str,
                "output": result.strip()
            }
            
        except Exception as e:
            logger.error(f"Shell execution error: {str(e)}")
            return {
                "status": "error",
                "message": f"Shell command execution failed: {str(e)}"
            }
    
    def upload(self, connection_id: str, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload a file to the remote system"""
        if connection_id not in self.connections:
            return {
                "status": "error",
                "message": "No active connection"
            }
        
        try:
            # Get the client
            client = self.connections[connection_id]
            
            # Create an SFTP client
            sftp = client.open_sftp()
            
            # Check if local_path is a directory
            if os.path.isdir(local_path):
                # Create the remote directory if it doesn't exist
                self.execute_shell(connection_id, f"mkdir -p {remote_path}")
                
                # Upload each file in the directory
                for root, dirs, files in os.walk(local_path):
                    for dir_name in dirs:
                        local_dir = os.path.join(root, dir_name)
                        rel_path = os.path.relpath(local_dir, local_path)
                        remote_dir = os.path.join(remote_path, rel_path)
                        self.execute_shell(connection_id, f"mkdir -p {remote_dir}")
                    
                    for file_name in files:
                        local_file = os.path.join(root, file_name)
                        rel_path = os.path.relpath(local_file, local_path)
                        remote_file = os.path.join(remote_path, rel_path)
                        sftp.put(local_file, remote_file)
                
                return {
                    "status": "success",
                    "message": f"Directory {local_path} uploaded to {remote_path}"
                }
            else:
                # Upload a single file
                sftp.put(local_path, remote_path)
                return {
                    "status": "success",
                    "message": f"File {local_path} uploaded to {remote_path}"
                }
                
        except Exception as e:
            logger.error(f"SSH upload error: {str(e)}")
            return {
                "status": "error",
                "message": f"File upload failed: {str(e)}"
            }
    
    def download(self, connection_id: str, remote_path: str, local_path: str) -> Dict[str, Any]:
        """Download a file from the remote system"""
        if connection_id not in self.connections:
            return {
                "status": "error",
                "message": "No active connection"
            }
        
        try:
            # Get the client
            client = self.connections[connection_id]
            
            # Create an SFTP client
            sftp = client.open_sftp()
            
            # Download the file
            sftp.get(remote_path, local_path)
            
            return {
                "status": "success",
                "message": f"File {remote_path} downloaded to {local_path}"
            }
            
        except Exception as e:
            logger.error(f"SSH download error: {str(e)}")
            return {
                "status": "error",
                "message": f"File download failed: {str(e)}"
            }
    
    def disconnect(self, connection_id: str) -> Dict[str, Any]:
        """Disconnect from the remote system"""
        if connection_id not in self.connections:
            return {
                "status": "error",
                "message": "No active connection"
            }
        
        try:
            # Close the shell
            if connection_id in self.shells:
                self.shells[connection_id].close()
                del self.shells[connection_id]
            
            # Close the connection
            self.connections[connection_id].close()
            del self.connections[connection_id]
            
            return {
                "status": "success",
                "message": "Disconnected"
            }
            
        except Exception as e:
            logger.error(f"SSH disconnect error: {str(e)}")
            return {
                "status": "error",
                "message": f"Disconnect failed: {str(e)}"
            }
    
    def disconnect_all(self) -> Dict[str, Any]:
        """Disconnect all active connections"""
        try:
            # Close all shells
            for shell in self.shells.values():
                try:
                    shell.close()
                except:
                    pass
            
            # Close all connections
            for client in self.connections.values():
                try:
                    client.close()
                except:
                    pass
            
            # Clear the caches
            self.shells = {}
            self.connections = {}
            
            return {
                "status": "success",
                "message": "All connections closed"
            }
            
        except Exception as e:
            logger.error(f"SSH disconnect_all error: {str(e)}")
            return {
                "status": "error",
                "message": f"Disconnect all failed: {str(e)}"
            }

# Register tool handlers
def ssh_connect(hostname, port, username, password, key_file=None):
    """Handler for ssh:connect tool"""
    ssh_tool = SSHTool()
    return ssh_tool.connect(hostname, port, username, password, key_file)

def ssh_execute(connection_id, command, timeout=10):
    """Handler for ssh:execute tool"""
    ssh_tool = SSHTool()
    return ssh_tool.execute_shell(connection_id, command, timeout)

def ssh_upload(connection_id, local_path, remote_path):
    """Handler for ssh:upload tool"""
    ssh_tool = SSHTool()
    return ssh_tool.upload(connection_id, local_path, remote_path)

def ssh_download(connection_id, remote_path, local_path):
    """Handler for ssh:download tool"""
    ssh_tool = SSHTool()
    return ssh_tool.download(connection_id, remote_path, local_path)

# Register the tools in the registry
TOOL_REGISTRY["ssh:connect"] = ssh_connect
TOOL_REGISTRY["ssh:execute"] = ssh_execute
TOOL_REGISTRY["ssh:upload"] = ssh_upload
TOOL_REGISTRY["ssh:download"] = ssh_download

def execute_tool(tool_id: str, **kwargs) -> Any:
    """Execute a tool by its ID with the provided parameters"""
    if tool_id not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_id}"}
    
    try:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    except Exception as e:
        logger.error(f"Error executing tool {tool_id}: {str(e)}")
        return {"error": str(e)}
