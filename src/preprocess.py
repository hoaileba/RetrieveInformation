import os
import re
import json
import argparse
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

def extract_url_from_file(file_path: str) -> str:
    """
    Extract the URL from the first line of the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line.startswith('http'):
            return first_line
    return ""

def clean_html(text: str) -> str:
    """
    Remove HTML tags and clean the text
    """
    # Use BeautifulSoup to parse and clean HTML
    soup = BeautifulSoup(text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()    
    
    # Get text
    text = soup.get_text()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    text_line = text.split('\n')
    text_line = [line.strip() for line in text_line if line.strip()]
    text = ' '.join(text_line)
    
    return text.strip()

def process_file(id,file_path: str) -> Dict[str, str]:
    """
    Process a single file and return a dictionary with url, original_text, and processed_text
    """
    # Extract URL
    url = extract_url_from_file(file_path)
    name = file_path.split('/')[-1].split('.')[0]
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the URL from the content if it exists
    if url:
        content = content.replace(url, '', 1).strip()
    
    # Clean the HTML
    processed_text = clean_html(content)
    
    return {
        "id": id,
        "name": name,
        "url": url,
        "original_text": content,
        "processed_text": processed_text
    }

def process_directory(directory_path: str, output_path: str) -> None:
    """
    Process all files in a directory and save the results to a JSON file
    """
    results = []
    
    # Process each file in the directory
    id = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                result = process_file(id,file_path)
                results.append(result)
                print(f"Processed: {filename}")
                id += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Save results to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(results)} files. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess medical documents by removing HTML tags and formatting as JSON")
    parser.add_argument("--input_dir", type=str, default="datasets/corpus",
                        help="Directory containing the input files")
    parser.add_argument("--output_file", type=str, default="datasets/processed/processed_data.json",
                        help="Path to save the processed data")
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_file)

if __name__ == "__main__":
    main() 