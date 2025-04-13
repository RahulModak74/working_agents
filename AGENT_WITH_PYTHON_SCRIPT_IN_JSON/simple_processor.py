#!/usr/bin/env python3

import sys
import os
import json
import datetime

def process_data(input_file=None):
    """
    Simple data processor that demonstrates script execution
    
    If input_file is provided, reads the file and counts lines/words
    If not provided, generates sample data
    """
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "processor": "simple_processor.py",
        "metrics": {}
    }
    
    # Check if we have an input file
    if input_file and os.path.exists(input_file):
        try:
            # Read the file
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Process the content
            lines = content.splitlines()
            words = content.split()
            chars = len(content)
            
            # Store metrics
            result["metrics"] = {
                "file": input_file,
                "file_size_bytes": os.path.getsize(input_file),
                "line_count": len(lines),
                "word_count": len(words),
                "char_count": chars
            }
            
            # Generate some sample insights
            avg_line_length = chars / len(lines) if lines else 0
            avg_word_length = chars / len(words) if words else 0
            
            result["insights"] = [
                f"The file contains {len(lines)} lines and {len(words)} words.",
                f"Average line length is {avg_line_length:.2f} characters.",
                f"Average word length is {avg_word_length:.2f} characters."
            ]
            
            # Print summary for stdout capture
            summary = f"""
File Analysis Summary:
---------------------
File: {input_file}
Total Lines: {len(lines)}
Total Words: {len(words)}
Total Characters: {chars}
Average Line Length: {avg_line_length:.2f} characters
Average Word Length: {avg_word_length:.2f} characters
            """
            print(summary)
            
        except Exception as e:
            result["error"] = str(e)
            print(f"Error processing file: {str(e)}")
    else:
        # Generate sample data if no input file
        result["metrics"] = {
            "sample_data": True,
            "random_metric_1": 42,
            "random_metric_2": 123,
            "status": "success"
        }
        
        result["insights"] = [
            "This is sample data generated because no input file was provided.",
            "In a real scenario, this script would process actual data files.",
            "Consider passing a text file to see actual processing results."
        ]
        
        # Print summary for stdout capture
        print("""
Sample Data Generated:
--------------------
No input file was provided, so sample data was generated.
Random Metric 1: 42
Random Metric 2: 123
Status: success
        """)
    
    # Add processing recommendations
    result["recommendations"] = [
        "Consider further analysis based on content patterns",
        "Review insights to identify areas for improvement",
        "This script can be extended to perform more detailed analysis"
    ]
    
    return result

if __name__ == "__main__":
    # Get input file from command line if provided
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Process the data
    result = process_data(input_file)
    
    # Output result as JSON
    print(json.dumps(result, indent=2))
