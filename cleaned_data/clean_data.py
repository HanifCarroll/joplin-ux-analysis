#!/usr/bin/env python3
"""
Data cleaning script for Joplin user feedback analysis.
Processes JSON files from GitHub, Reddit, and Discourse sources.
"""

import json
import re
import glob
import os
from typing import List, Dict, Any
from bs4 import BeautifulSoup


def preprocess_text(raw_text: str, source: str) -> str:
    """
    Clean and preprocess raw text content for LLM analysis.

    Args:
        raw_text: The original comment body text
        source: The data source ("GitHub", "Reddit", or "Discourse")

    Returns:
        Cleaned text ready for analysis
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text

    # Step 1: HTML to Text Conversion (for Discourse)
    if source == "Discourse":
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

    # Step 2: Boilerplate and Template Removal
    boilerplate_patterns = [
        r"^### Operating system.*$",
        r"^### Joplin version.*$",
        r"^### Desktop version info.*$",
        r"^### Current behaviour.*$",
        r"^### Expected behaviour.*$",
        r"^### Logs.*$",
        r"^_No response_.*$",
        r"^Operating system.*$",
        r"^Joplin version.*$",
        r"^Desktop version info.*$",
        r"^Current behaviour.*$",
        r"^Expected behaviour.*$",
        r"^Logs.*$",
    ]

    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        line_stripped = line.strip()
        should_remove = False

        for pattern in boilerplate_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                should_remove = True
                break

        if not should_remove:
            filtered_lines.append(line)

    text = "\n".join(filtered_lines)

    # Step 3: Markdown Syntax Stripping

    # Remove Markdown image links
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove code fences and their content
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`\n]*`", "", text)  # inline code

    # Convert Markdown links to plain text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove emphasis characters
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # italic
    text = re.sub(r"__(.*?)__", r"\1", text)  # underline
    text = re.sub(r"_(.*?)_", r"\1", text)  # italic underscore

    # Step 4: General Content Cleanup

    # Remove quoted reply blocks
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not line.strip().startswith(">")]
    text = "\n".join(filtered_lines)

    # Remove automated email footers
    text = re.sub(r"^On .+, .+ wrote:.*$", "", text, flags=re.MULTILINE)

    # Remove common GitHub/forum signatures
    text = re.sub(r"--+\s*$", "", text, flags=re.MULTILINE)

    # Normalize whitespace
    # Replace multiple consecutive newlines with maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def process_json_file(input_file: str) -> None:
    """
    Process a single JSON file and create cleaned version.

    Args:
        input_file: Path to the input JSON file
    """
    print(f"Processing {input_file}...")

    try:
        # Load JSON data
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {input_file} does not contain a list at root level")
            return

        # Process each thread/issue
        processed_count = 0
        for item in data:
            if not isinstance(item, dict) or "comments" not in item:
                continue

            source = item.get("source", "Unknown")
            comments = item.get("comments", [])

            # Process each comment
            for comment in comments:
                if isinstance(comment, dict) and "body" in comment:
                    raw_body = comment["body"]
                    cleaned_body = preprocess_text(raw_body, source)
                    comment["cleaned_body"] = cleaned_body
                    processed_count += 1

        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned.json"

        # Save cleaned data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Processed {processed_count} comments → {output_file}")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def main():
    """
    Main function to process all JSON files in the current directory.
    """
    print("Joplin Feedback Data Cleaner")
    print("=" * 30)

    # Find all JSON files that don't end with _cleaned.json
    json_files = []
    for file_path in glob.glob("*.json"):
        if not file_path.endswith("_cleaned.json"):
            json_files.append(file_path)

    if not json_files:
        print("No JSON files found to process.")
        return

    print(f"Found {len(json_files)} file(s) to process:")
    for file_path in json_files:
        print(f"  - {file_path}")
    print()

    # Process each file
    for json_file in json_files:
        process_json_file(json_file)
        print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
