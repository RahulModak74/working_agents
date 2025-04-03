#!/bin/bash

# Output file name
OUTPUT_FILE="cyber_analysis_combined.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Add header to output file
echo "===================================================" >> "$OUTPUT_FILE"
echo "COMBINED CYBERSECURITY ANALYSIS OUTPUTS" >> "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "===================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Files to process - you can edit this array to include only specific files
FILES=(
  "threat_landscape_modeler_output.txt"
  "defense_capability_analyzer_output.txt"
  "risk_quantification_modeler_output.txt"
  "optimization_model_constructor_output.txt"
  "optimization_algorithm_selector_output.txt"
  "cybersecurity_optimizer_output.txt"
  "resilience_solution_evaluator_output.txt"
  "implementation_planner_output.txt"
  "dynamic_agent_output.txt"
  "incident_response_architect_output.txt"
  "executive_cyber_advisor_output.txt"
  "cyber_resilience_metrics_designer_output.txt"
  "cyber_resilience_simulator_output.txt"
  "future_readiness_assessor_output.txt"
  "workflow_results.json"
)

# Process each file
for filename in "${FILES[@]}"; do
    # Check if file exists
    if [ -f "$filename" ]; then
        echo "Processing: $filename"
        
        # Add decorative file separator
        echo "" >> "$OUTPUT_FILE"
        echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓" >> "$OUTPUT_FILE"
        echo "┃ FILE: $filename" >> "$OUTPUT_FILE"
        echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # Add file content with light formatting
        echo "CONTENT:" >> "$OUTPUT_FILE"
        echo "----------------------------------------" >> "$OUTPUT_FILE"
        cat "$filename" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "----------------------------------------" >> "$OUTPUT_FILE"
        
        # Add ending separator with file size information
        FILE_SIZE=$(du -h "$filename" | cut -f1)
        echo "END OF FILE: $filename (Size: $FILE_SIZE)" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        echo "Warning: File '$filename' not found, skipping."
    fi
done

# Add table of contents at the beginning
TMP_FILE=$(mktemp)
echo "TABLE OF CONTENTS:" > "$TMP_FILE"
echo "----------------------------------------" >> "$TMP_FILE"
for filename in "${FILES[@]}"; do
    if [ -f "$filename" ]; then
        FILE_SIZE=$(du -h "$filename" | cut -f1)
        echo "- $filename (Size: $FILE_SIZE)" >> "$TMP_FILE"
    fi
done
echo "" >> "$TMP_FILE"
echo "----------------------------------------" >> "$TMP_FILE"
echo "" >> "$TMP_FILE"
cat "$OUTPUT_FILE" >> "$TMP_FILE"
mv "$TMP_FILE" "$OUTPUT_FILE"

echo "All cybersecurity analysis files have been combined into $OUTPUT_FILE"
echo "Combined file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
