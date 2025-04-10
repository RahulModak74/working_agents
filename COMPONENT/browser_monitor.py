#!/usr/bin/env python3

import os
import sys
import time
import json
import glob
import argparse
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()

def get_latest_log_file(log_dir="browser_logs"):
    """Get the most recent browser log file"""
    log_files = glob.glob(os.path.join(log_dir, "browser_*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=os.path.getmtime)

def follow_log(log_file, poll_interval=0.5):
    """Follow a log file similar to 'tail -f'"""
    print(f"{Fore.CYAN}Monitoring log file: {log_file}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Press Ctrl+C to stop monitoring{Style.RESET_ALL}")
    print("\n" + "="*80 + "\n")
    
    with open(log_file, 'r') as f:
        # Move to the end of the file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                continue
            
            # Process and print the line with color coding
            print_formatted_log_line(line.strip())

def print_formatted_log_line(line):
    """Print a log line with color formatting based on content"""
    # Check for timestamp and split it from the rest
    parts = line.split(' - ', 3)
    if len(parts) >= 3:
        timestamp = parts[0]
        log_level = parts[2]
        message = parts[3] if len(parts) > 3 else ""
        
        # Color by log level
        if "ERROR" in log_level:
            level_color = Fore.RED
        elif "WARNING" in log_level:
            level_color = Fore.YELLOW
        elif "INFO" in log_level:
            level_color = Fore.GREEN
        else:
            level_color = Fore.WHITE
        
        # Color by event type
        message_color = Fore.WHITE
        
        if "🚀" in message or "✅" in message:
            message_color = Fore.CYAN + Style.BRIGHT
        elif "🌐" in message:
            message_color = Fore.BLUE
        elif "🔍" in message:
            message_color = Fore.MAGENTA
        elif "🖱️" in message:
            message_color = Fore.YELLOW
        elif "📄" in message:
            message_color = Fore.GREEN
        elif "📸" in message:
            message_color = Fore.CYAN
        elif "Tool called" in message:
            message_color = Fore.MAGENTA + Style.BRIGHT
        
        # Print formatted line
        print(f"{Fore.WHITE}{timestamp}{Style.RESET_ALL} - {level_color}{log_level}{Style.RESET_ALL} - {message_color}{message}{Style.RESET_ALL}")
    else:
        # Fallback for lines that don't match the expected format
        print(line)

def get_browser_stats(log_dir="browser_logs", screenshot_dir="screenshots"):
    """Analyze browser logs and screenshots to generate usage statistics"""
    # Get log files and sort by date
    log_files = glob.glob(os.path.join(log_dir, "browser_*.log"))
    log_files.sort(key=os.path.getmtime)
    
    # Get screenshots and sort by date
    screenshots = glob.glob(os.path.join(screenshot_dir, "*.png"))
    screenshots.sort(key=os.path.getmtime)
    
    # Basic stats
    stats = {
        "total_log_files": len(log_files),
        "total_screenshots": len(screenshots),
        "latest_log_file": os.path.basename(log_files[-1]) if log_files else None,
        "latest_screenshot": os.path.basename(screenshots[-1]) if screenshots else None,
        "first_activity": datetime.fromtimestamp(os.path.getmtime(log_files[0])).isoformat() if log_files else None,
        "last_activity": datetime.fromtimestamp(os.path.getmtime(log_files[-1])).isoformat() if log_files else None,
    }
    
    # Count operations by parsing the latest log file
    if log_files:
        latest_log = log_files[-1]
        operations = {
            "navigations": 0,
            "clicks": 0,
            "get_content": 0,
            "find_elements": 0,
            "browsers_created": 0,
            "browsers_closed": 0
        }
        
        with open(latest_log, 'r') as f:
            for line in f:
                if "Tool called: browser:navigate" in line:
                    operations["navigations"] += 1
                elif "Tool called: browser:click" in line:
                    operations["clicks"] += 1
                elif "Tool called: browser:get_content" in line:
                    operations["get_content"] += 1
                elif "Tool called: browser:find_elements" in line:
                    operations["find_elements"] += 1
                elif "Tool called: browser:create" in line:
                    operations["browsers_created"] += 1
                elif "Tool called: browser:close" in line:
                    operations["browsers_closed"] += 1
        
        stats["operations"] = operations
    
    return stats

def print_stats(stats):
    """Print browser usage statistics in a nice format"""
    print(f"\n{Fore.CYAN}======== BROWSER ACTIVITY STATISTICS ========{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}Activity Period:{Style.RESET_ALL}")
    print(f"  First activity: {stats.get('first_activity', 'N/A')}")
    print(f"  Last activity:  {stats.get('last_activity', 'N/A')}")
    print()
    
    print(f"{Fore.YELLOW}Files:{Style.RESET_ALL}")
    print(f"  Log files:    {stats.get('total_log_files', 0)}")
    print(f"  Screenshots:  {stats.get('total_screenshots', 0)}")
    print(f"  Latest log:   {stats.get('latest_log_file', 'N/A')}")
    print(f"  Latest image: {stats.get('latest_screenshot', 'N/A')}")
    print()
    
    if 'operations' in stats:
        ops = stats['operations']
        print(f"{Fore.YELLOW}Browser Operations (latest log):{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Browsers created:  {ops['browsers_created']}{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}Page navigations:  {ops['navigations']}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Content retrieved: {ops['get_content']}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}Elements found:    {ops['find_elements']}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Clicks performed:  {ops['clicks']}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Browsers closed:   {ops['browsers_closed']}{Style.RESET_ALL}")
        
        # Calculate total operations
        total_ops = sum(ops.values())
        print(f"\n  {Fore.WHITE}Total operations:  {total_ops}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}============================================{Style.RESET_ALL}\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor browser activities in real-time")
    parser.add_argument('--log-dir', default='browser_logs', help='Directory containing browser log files')
    parser.add_argument('--screenshot-dir', default='screenshots', help='Directory containing browser screenshots')
    parser.add_argument('--stats', action='store_true', help='Show statistics instead of following logs')
    parser.add_argument('--follow', action='store_true', help='Follow the latest log file in real-time')
    
    args = parser.parse_args()
    
    if args.stats:
        # Show browser usage statistics
        stats = get_browser_stats(args.log_dir, args.screenshot_dir)
        print_stats(stats)
    else:
        # Follow the latest log file
        latest_log = get_latest_log_file(args.log_dir)
        if not latest_log:
            print(f"{Fore.RED}Error: No log files found in {args.log_dir}{Style.RESET_ALL}")
            sys.exit(1)
        
        try:
            follow_log(latest_log)
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Stopped monitoring{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
