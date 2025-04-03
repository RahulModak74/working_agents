#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import requests
import json
import re
import time
from urllib.parse import quote_plus
import random

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'[^\w\s-]', '', text).strip()
    return text

def search_duckduckgo(query, num_results=10):
    """Search DuckDuckGo for a query and return results."""
    # Add a user agent to prevent getting blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Encode the query
    encoded_query = quote_plus(query)
    
    # Use the DuckDuckGo API-like endpoint (not an official API)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract the results
        results = []
        
        # Add related topics from RelatedTopics
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics']:
                if 'Text' in topic:
                    results.append(topic['Text'])
        
        # Add abstract text if available
        if 'AbstractText' in data and data['AbstractText']:
            results.append(data['AbstractText'])
            
        return results[:num_results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def extract_keyphrases(text, max_phrases=5):
    """Extract key phrases from text using simple heuristics."""
    # Split into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    phrases = []
    for sentence in sentences:
        # Look for phrases with certain patterns
        noun_phrase_patterns = [
            r'\b(?:the|a|an)?\s*([A-Z][a-z]+(?:\s+[a-z]+){1,3})',  # Capitalized noun phrases
            r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|in|for|with)\s+[a-z]+',  # X of Y patterns
            r'\b([a-z]+(?:\s+[a-z]+){1,2})\s+(?:and|or)\s+([a-z]+(?:\s+[a-z]+){1,2})',  # X and Y patterns
        ]
        
        for pattern in noun_phrase_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if isinstance(match, tuple):
                    phrases.extend(match)
                else:
                    phrases.append(match)
    
    # Clean and deduplicate
    cleaned_phrases = []
    for phrase in phrases:
        phrase = phrase.strip().lower()
        if (len(phrase.split()) >= 2 and 
            phrase not in cleaned_phrases and 
            len(phrase) > 5 and 
            not any(phrase in p for p in cleaned_phrases if phrase != p)):
            cleaned_phrases.append(phrase)
    
    return cleaned_phrases[:max_phrases]

def generate_dimensions_for_topic(topic, num_dimensions=10):
    """Generate dimensions for a topic using search results."""
    print(f"Generating {num_dimensions} dimensions for topic: {topic}")
    
    # Search variations to get diverse dimensions
    search_queries = [
        f"{topic} key aspects",
        f"{topic} main components",
        f"{topic} important factors",
        f"{topic} latest developments",
        f"{topic} future trends",
        f"{topic} challenges",
        f"{topic} technologies",
        f"{topic} applications",
        f"{topic} research areas",
        f"{topic} impact on industry"
    ]
    
    all_dimensions = []
    all_search_results = []
    
    # For each search query
    for query in search_queries:
        print(f"Searching for: {query}")
        search_results = search_duckduckgo(query, num_results=3)
        all_search_results.extend(search_results)
        
        # Add a delay between searches to avoid rate limiting
        time.sleep(1)
    
    # Extract key phrases from all search results
    for result in all_search_results:
        dimensions = extract_keyphrases(result)
        all_dimensions.extend(dimensions)
    
    # If we don't have enough dimensions from the search, add some generic ones
    generic_dimensions = [
        f"Technical aspects of {topic}",
        f"Historical development of {topic}",
        f"Future implications of {topic}",
        f"Economic impact of {topic}",
        f"Ethical considerations for {topic}",
        f"Regulatory framework for {topic}",
        f"Global adoption of {topic}",
        f"Social implications of {topic}",
        f"Limitations and challenges of {topic}",
        f"Comparative analysis of {topic}"
    ]
    
    # Deduplicate and clean dimensions
    unique_dimensions = []
    for dim in all_dimensions:
        clean_dim = clean_text(dim)
        # Check if this is a meaningful dimension (not too short, not just the topic name)
        if (len(clean_dim) > 5 and 
            clean_dim not in unique_dimensions and 
            not clean_dim.lower() == topic.lower() and
            not any(clean_dim in d for d in unique_dimensions if clean_dim != d)):
            unique_dimensions.append(clean_dim)
    
    # If we didn't get enough dimensions, add some generic ones
    while len(unique_dimensions) < num_dimensions:
        # Find a generic dimension not already in the list
        for gen_dim in generic_dimensions:
            if gen_dim not in unique_dimensions:
                unique_dimensions.append(gen_dim)
                break
    
    # Format dimensions to be more consistent
    final_dimensions = []
    for dim in unique_dimensions[:num_dimensions]:
        # Capitalize first letter of each word
        formatted_dim = ' '.join(word.capitalize() for word in dim.split())
        final_dimensions.append(formatted_dim)
    
    # Always add "in" or "of" to make dimensions more descriptive
    for i, dim in enumerate(final_dimensions):
        if not any(x in dim.lower() for x in ['in', 'of', 'for', 'with']):
            if random.choice([True, False]):
                final_dimensions[i] = f"{dim} in {topic}"
            else:
                final_dimensions[i] = f"{dim} of {topic}"
    
    return final_dimensions

def create_csv_file(topic, dimensions, output_file="research_dimensions.csv"):
    """Create a CSV file with the topic and dimensions."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['topic'] + [f'dimension{i+1}' for i in range(len(dimensions))]
        writer.writerow(header)
        
        # Write data
        row = [topic] + dimensions
        writer.writerow(row)
    
    print(f"Created CSV file: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate research dimensions for a topic.")
    parser.add_argument('topic', help='The research topic to generate dimensions for')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file name')
    parser.add_argument('--dimensions', '-d', type=int, default=10, help='Number of dimensions to generate (default: 10)')
    args = parser.parse_args()
    
    # Generate dimensions
    dimensions = generate_dimensions_for_topic(args.topic, args.dimensions)
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Clean topic for filename
        topic_filename = re.sub(r'[^\w\s-]', '', args.topic).strip().lower()
        topic_filename = re.sub(r'[-\s]+', '_', topic_filename)
        if len(topic_filename) > 40:
            topic_filename = topic_filename[:40]
        output_file = f"research_dimensions_{topic_filename}.csv"
    
    # Create CSV file
    csv_path = create_csv_file(args.topic, dimensions, output_file)
    
    print("\nGenerated Dimensions:")
    for i, dim in enumerate(dimensions):
        print(f"{i+1}. {dim}")
    
    print(f"\nYou can now use this CSV with your research workflow generator:")
    print(f"python generate_research_workflow.py {csv_path}")

if __name__ == "__main__":
    main()
