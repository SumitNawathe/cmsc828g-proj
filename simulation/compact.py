import argparse
import glob
import os
import re

from env_vars import RESULTS_DIR

def extract_chunk_idx(filepath: str):
    match = re.search(r'_chunk(\d+)\.jsonl$', filepath)
    return int(match.group(1)) if match else -1

def compact_experiment(output_dir: str, model: str, dataset: str, G: int, B: int, dry_run: bool = False) -> bool:
    """
    Compact all chunk files for a single experiment into chunk0.
    Returns True if compaction was performed, False otherwise.
    """
    model_safe = model.replace('/', '-')
    pattern = os.path.join(output_dir, f"inference_{model_safe}_{dataset}_G{G}_B{B}_chunk*.jsonl")
    chunk_files = glob.glob(pattern)
    if not chunk_files:
        print(f"No chunk files found matching pattern: {pattern}")
        return False

    chunk_files_with_idx = [(f, extract_chunk_idx(f)) for f in chunk_files]
    chunk_files_with_idx.sort(key=lambda x: x[1])

    if len(chunk_files) == 1:
        return False

    print(f"\nCompacting {model}/{dataset}/G{G}/B{B}:")
    print(f"  Found {len(chunk_files)} chunk files")

    chunk0_file = os.path.join(output_dir, f"inference_{model_safe}_{dataset}_G{G}_B{B}_chunk0.jsonl")
    tmp_file = os.path.join(output_dir, f"inference_{model_safe}_{dataset}_G{G}_B{B}_compact.tmp")
    
    if dry_run:
        # Just count lines for stdout
        total_lines = 0
        for filepath, idx in chunk_files_with_idx:
            with open(filepath, 'r') as f:
                total_lines += sum(1 for _ in f)
        print(f"  [DRY RUN] Would write {total_lines} lines to {os.path.basename(chunk0_file)}")
        for filepath, idx in chunk_files_with_idx:
            if idx != 0 and filepath != chunk0_file:
                print(f"  [DRY RUN] Would delete {os.path.basename(filepath)}")
        return True

    # Safely stream chunks into a temporary file first to avoid OOM and data loss on crash
    total_lines = 0
    with open(tmp_file, 'w') as f_out:
        for filepath, idx in chunk_files_with_idx:
            with open(filepath, 'r') as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_lines += 1
        f_out.flush()
        os.fsync(f_out.fileno())
        
    # Atomically replace chunk0
    os.replace(tmp_file, chunk0_file)
    print(f"  Total lines collected: {total_lines}")
    print(f"  Wrote {total_lines} lines to {os.path.basename(chunk0_file)}")

    deleted_count = 0
    for filepath, idx in chunk_files_with_idx:
        if idx != 0 and filepath != chunk0_file:
            os.remove(filepath)
            deleted_count += 1

    print(f"  Compaction complete: {deleted_count} chunk files deleted")
    return True

def discover_experiments(output_dir: str) -> list[tuple[str, str, int, int]]:
    """
    Discover all unique experiment combinations from existing chunk files.
    Returns a list of (model, dataset, G, B) tuples.
    """
    pattern = os.path.join(output_dir, "inference_*_chunk*.jsonl")
    all_files = glob.glob(pattern)
    
    experiment_regex = re.compile(
        r'inference_(.+?)_(.+?)_G(\d+)_B(\d+)_chunk\d+\.jsonl$'
    )
    
    experiments = set()
    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = experiment_regex.match(filename)
        if match:
            model, dataset, G, B = match.groups()
            experiments.add((model, dataset, int(G), int(B)))
    
    return sorted(experiments)

def compact_all(output_dir: str, dry_run: bool = False) -> None:
    experiments = discover_experiments(output_dir)
    
    if not experiments:
        print(f"No experiments found in {output_dir}")
        return
    
    print(f"Discovered {len(experiments)} experiment(s):")
    for model, dataset, G, B in experiments:
        print(f"  - {model}/{dataset}/G{G}/B{B}")
    
    compacted = 0
    for model, dataset, G, B in experiments:
        if compact_experiment(output_dir, model, dataset, G, B, dry_run):
            compacted += 1
    
    print(f"\n{'Would compact' if dry_run else 'Compacted'} {compacted} experiment(s)")

def main():
    parser = argparse.ArgumentParser(
        description="Compact inference results from multiple chunk files into a single file"
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Name of the model (as provided in basic_inference.py)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of the dataset')
    parser.add_argument('--group-size', '-G', type=int, default=None,
                        help='Group size for vLLM generation (G value)')
    parser.add_argument('--max-new-tokens', '-B', type=int, default=None,
                        help='Maximum number of new tokens (B value)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory containing results (default: RESULTS_DIR)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--all', action='store_true',
                        help='Compact all available experiment combinations')
    
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_DIR

    if args.all:
        compact_all(output_dir, args.dry_run)
    else:
        if not all([args.model, args.dataset, args.group_size is not None, args.max_new_tokens is not None]):
            parser.error("--model, --dataset, -G, and -B are required unless --all is specified")
        
        compact_experiment(output_dir, args.model, args.dataset, args.group_size, args.max_new_tokens, args.dry_run)

if __name__ == "__main__":
    main()
