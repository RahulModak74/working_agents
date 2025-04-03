#!/usr/bin/env python3

import json
import os
import sys
import re

def extract_report_to_txt(json_file_path, output_file_path=None):
    """
    Extract the final report from the workflow results JSON and save it to a text file.
    
    Args:
        json_file_path (str): Path to the workflow results JSON file
        output_file_path (str, optional): Path to save the output text file. 
                                         If not provided, will use the same name with .txt extension
    
    Returns:
        str: Path to the created text file
    """
    # Determine output file path if not provided
    if not output_file_path:
        base_name = os.path.splitext(json_file_path)[0]
        output_file_path = f"{base_name}_report.txt"
    
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the report content from the comprehensive_report_generator
        if 'comprehensive_report_generator' in data:
            report_content = data['comprehensive_report_generator'].get('markdown_content', '')
            
            # Clean up the markdown content
            # Remove markdown code block markers
            report_content = re.sub(r'```markdown\n', '', report_content)
            report_content = re.sub(r'```', '', report_content)
            
            # Write to the output file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            print(f"Report successfully extracted to: {output_file_path}")
            return output_file_path
        
        # If comprehensive_report_generator is not found, check for report_generator
        elif 'report_generator' in data:
            report_content = data['report_generator'].get('markdown_content', '')
            
            # Clean up the markdown content
            report_content = re.sub(r'```markdown\n', '', report_content)
            report_content = re.sub(r'```', '', report_content)
            
            # Write to the output file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            print(f"Report successfully extracted to: {output_file_path}")
            return output_file_path
        
        else:
            # If no report generators found, check if there's any content that looks like a report
            for key, value in data.items():
                if isinstance(value, dict) and 'markdown_content' in value:
                    report_content = value['markdown_content']
                    
                    # Clean up the markdown content
                    report_content = re.sub(r'```markdown\n', '', report_content)
                    report_content = re.sub(r'```', '', report_content)
                    
                    # Write to the output file
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                        
                    print(f"Report successfully extracted from {key} to: {output_file_path}")
                    return output_file_path
                
                # Check for text_content which might contain a report
                elif isinstance(value, dict) and 'text_content' in value:
                    report_content = value['text_content']
                    
                    # Write to the output file
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                        
                    print(f"Content successfully extracted from {key} to: {output_file_path}")
                    return output_file_path
            
            print("No report content found in the JSON file.")
            return None
            
    except Exception as e:
        print(f"Error extracting report: {str(e)}")
        return None

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_report.py <workflow_results.json> [output_file.txt]")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_report_to_txt(json_file_path, output_file_path)

if __name__ == "__main__":
    main()
