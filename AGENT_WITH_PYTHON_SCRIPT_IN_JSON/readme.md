This folder allows python script execution within json workflows
# Script Execution Demo

This is a simple demonstration of the script execution feature in the workflow system.

## Files Included

1. `simple_processor.py` - A basic Python script that processes text files and generates metrics
2. `sample_data.txt` - A sample text file for processing
3. `simple_workflow.json` - A workflow definition that uses the script

## How to Run

1. Make sure all files are in the same directory
2. Run the workflow using your runner_wrapper.py:

```bash
python3 runner_wrapper.py --workflow simple_workflow.json
```

## Workflow Steps

1. The first agent ("data_analyzer") will:
   - Execute the `simple_processor.py` script with `sample_data.txt` as input
   - Use the script's output as content for the AI call
   - Generate a JSON analysis based on the script's output

2. The second agent ("report_generator") will:
   - Read the results from both the `data_analyzer` (AI output) and `data_analyzer_script` (script output)
   - Generate a markdown report based on the combined analysis

## Expected Results

- The script will process the sample text file and count lines, words, and characters
- The data_analyzer agent will receive the script's output and add its own analysis
- The report_generator will combine all information into a formatted report

## Script Functionality

The `simple_processor.py` script:
- Counts lines, words, and characters in a text file
- Calculates average line and word length
- Generates insights based on the metrics
- Returns a structured JSON result

If no input file is provided, it generates sample data instead.

## Extending the Demo

You can modify the sample_data.txt file to test with different content, or create your own scripts to process different types of data.
