import argparse
import json
import os
import shutil
import tempfile

def main():
    parser = argparse.ArgumentParser(description="De-duplicate an experiment result file (jsonl).")
    parser.add_argument("filepath", help="Path to the jsonl result file.")
    parser.add_argument('--group-limit', type=int, default=None, help="Maximum number of groups to keep per sample.")
    parser.add_argument('-f', '--force', action='store_true', help="Force overwrite without asking for confirmation.")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' does not exist.")
        return

    original_lines_count = 0
    deduped_lines_count = 0
    seen = set()

    # Create a temporary file first so we can stream directly to it, preventing OOM
    fd, temp_path = tempfile.mkstemp(suffix=".jsonl", dir=os.path.dirname(args.filepath))
    
    try:
        print(f"Reading {args.filepath} and streaming unique data to temporary file...")
        with open(args.filepath, 'r') as f_in, os.fdopen(fd, 'w') as f_out:
            for line in f_in:
                original_lines_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = (data.get('sample_idx'), data.get('group_idx'))
                    
                    if args.group_limit is not None and key[1] >= args.group_limit:
                        continue
                    
                    if key not in seen:
                        seen.add(key)
                        f_out.write(line + '\n')
                        deduped_lines_count += 1
                except Exception as e:
                    print(f"Warning: Skipping ill-formed line: {line[:50]}... Error: {e}")
        
        print(f"Original lines: {original_lines_count}")
        print(f"De-duplicated lines: {deduped_lines_count}")

        if deduped_lines_count == original_lines_count:
            print("No duplicates found. Nothing to do.")
            os.remove(temp_path)
            return

        if args.force:
            confirm = 'y'
        else:
            confirm = input(f"Overwrite the original file '{args.filepath}'? (y/n): ").strip().lower()

        if confirm == 'y':
            # Use os.replace for atomic replacement (like compact.py does)
            os.replace(temp_path, args.filepath)
            print(f"Successfully overwrote '{args.filepath}'.")
        else:
            print(f"Operation cancelled. Temporary file remains at: {temp_path}")
            
    except Exception as e:
        print(f"Error during file operations: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
