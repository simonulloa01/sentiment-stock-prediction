# data-preprocessing/sentiment_analysis.py
import os
import json
import sys
from transformers import pipeline
import torch # Using torch as default, user might need to install it

# --- Configuration ---
MODEL_NAME = "ProsusAI/finbert"
TWEET_PREPROCESSED_DIR = "tweet/preprocessed"
# --- End Configuration ---

def get_sentiment(text, sentiment_pipeline):
    """Applies sentiment analysis pipeline to text."""
    if not text:
        return {'label': 'neutral', 'score': 0.0, 'error': 'Empty text'}
    try:
        # Limit text length for the model if necessary (FinBERT typical limit is 512 tokens)
        # This basic truncation might split words/tokens improperly, but is a simple safeguard.
        max_length = 512
        # Ensure text is a string before slicing
        text_str = str(text)
        truncated_text = text_str[:max_length] if len(text_str) > max_length else text_str
        result = sentiment_pipeline(truncated_text)
        return result[0] # Pipeline returns a list with one dict
    except Exception as e:
        print(f"    Error during sentiment analysis for text: '{str(text)[:50]}...' - {e}", file=sys.stderr)
        return {'label': 'error', 'score': 0.0, 'error': str(e)}

def process_file(file_path, sentiment_pipeline):
    """Reads a file, adds sentiment to JSON objects, and overwrites the file."""
    print(f"  Processing file: {os.path.basename(file_path)}")
    updated_lines = []
    needs_update = False
    original_lines_count = 0
    processed_lines_count = 0
    skipped_lines_count = 0
    error_lines_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        original_lines_count = len(lines)

        for i, line in enumerate(lines):
            line_content = line.strip()
            if not line_content: # Skip empty lines
                continue

            try:
                tweet_data = json.loads(line_content)
                # Check if sentiment already exists and is valid (optional check: might want to re-run if format is wrong)
                if 'sentiment' not in tweet_data or not isinstance(tweet_data.get('sentiment'), dict):
                    needs_update = True
                    # Join the text list into a single string
                    tweet_text = " ".join(tweet_data.get("text", []))
                    sentiment_result = get_sentiment(tweet_text, sentiment_pipeline)
                    tweet_data['sentiment'] = sentiment_result
                    processed_lines_count += 1
                else:
                    skipped_lines_count += 1 # Already has sentiment

                updated_lines.append(json.dumps(tweet_data))

            except json.JSONDecodeError:
                print(f"    Warning: Skipping invalid JSON line {i+1} in {os.path.basename(file_path)}", file=sys.stderr)
                updated_lines.append(line_content) # Keep invalid line as is
                error_lines_count += 1
            except Exception as e:
                print(f"    Error processing line {i+1} in {os.path.basename(file_path)}: {line_content[:50]}... - {e}", file=sys.stderr)
                # Try to add error to JSON if possible, otherwise keep original
                try:
                    # Attempt to load JSON again to add error info
                    tweet_data = json.loads(line_content)
                    tweet_data['sentiment'] = {'label': 'processing_error', 'score': 0.0, 'error': str(e)}
                    updated_lines.append(json.dumps(tweet_data))
                except:
                    updated_lines.append(line_content) # Keep original if loading fails
                error_lines_count += 1

        # Write the updated data back to the file only if changes were made
        if needs_update:
            # Create backup before overwriting (optional but recommended)
            # backup_path = file_path + ".bak"
            # os.rename(file_path, backup_path)
            # print(f"    Created backup: {os.path.basename(backup_path)}")
            with open(file_path, 'w', encoding='utf-8') as f:
                for updated_line in updated_lines:
                    f.write(updated_line + '\n')
            print(f"    Updated {processed_lines_count}/{original_lines_count} lines (skipped: {skipped_lines_count}, errors: {error_lines_count}). Saved: {os.path.basename(file_path)}")
        else:
            # Check if there were errors even if no updates were needed
            if error_lines_count > 0:
                 print(f"    No updates needed, but found {error_lines_count} errors in file: {os.path.basename(file_path)} (skipped: {skipped_lines_count})")
            else:
                 print(f"    No updates needed for file: {os.path.basename(file_path)} (skipped: {skipped_lines_count})")


    except Exception as e:
        print(f"  Failed to process file {file_path}: {e}", file=sys.stderr)


def main():
    """Main function to iterate through directories and process files."""
    # Determine workspace root dynamically
    try:
        # Assumes script is in data-preprocessing subdir relative to workspace root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(script_dir, '..'))
    except NameError:
        # Fallback if __file__ is not defined (e.g., running interactively or exec)
        workspace_root = os.getcwd()
        print("Warning: Could not determine script location dynamically. Assuming current directory is workspace root.", file=sys.stderr)


    preprocessed_dir = os.path.join(workspace_root, TWEET_PREPROCESSED_DIR)

    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Directory not found: {preprocessed_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialize sentiment pipeline
    print(f"Loading model {MODEL_NAME}...")
    try:
        # Check for GPU availability
        device_num = 0 if torch.cuda.is_available() else -1
        # Explicitly use cpu if needed: device_num = -1
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=device_num)
        print(f"Model loaded successfully. Using device: {'GPU 0' if device_num == 0 else 'CPU'}")
    except Exception as e:
        print(f"Error loading sentiment analysis pipeline: {e}", file=sys.stderr)
        print("Please ensure 'transformers' and a backend ('torch' or 'tensorflow') are installed.", file=sys.stderr)
        sys.exit(1)

    # Iterate through company directories
    for company in sorted(os.listdir(preprocessed_dir)): # Sort for consistent order
        company_path = os.path.join(preprocessed_dir, company)
        if os.path.isdir(company_path):
            print(f"\nProcessing company: {company}")
            # Iterate through date files within each company directory
            for filename in sorted(os.listdir(company_path)): # Sort for consistent order
                file_path = os.path.join(company_path, filename)
                # Basic check for valid files (e.g., ignore hidden files like .DS_Store)
                if os.path.isfile(file_path) and not filename.startswith('.'):
                    process_file(file_path, sentiment_pipeline)

    print("\nSentiment analysis complete.")

if __name__ == "__main__":
    main()

