"""
Embed prompts from inference JSONL files using a sentence-embedding model.

For each unique sample_idx, extracts the prompt from method_output.prompt,
encodes it, and saves embeddings to a Parquet file in the output directory.

Usage:
    python embed.py --input /path/to/inference_*.jsonl
    python embed.py --input /path/to/inference_*.jsonl --model Qwen/Qwen3-Embedding-0.6B
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool using the last non-padding token (required for decoder-only models)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[batch_size * torch.arange(batch_size, device=last_hidden_states.device) // batch_size, sequence_lengths]


def collect_unique_prompts(jsonl_path: str) -> dict[int, str]:
    """Read JSONL and return {sample_idx: prompt} keeping first occurrence per sample."""
    prompts = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d["sample_idx"]
            if sid not in prompts:
                prompts[sid] = d["method_output"]["prompt"]
    return prompts


@torch.no_grad()
def encode_prompts(
    prompts: dict[int, str],
    model_name: str,
    batch_size: int = 4,
    max_length: int = 8192,
) -> dict[int, np.ndarray]:
    """Encode prompts with the embedding model, return {sample_idx: embedding}."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    sample_ids = sorted(prompts.keys())
    texts = [prompts[sid] for sid in sample_ids]

    embeddings_map: dict[int, np.ndarray] = {}

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[start : start + batch_size]
        batch_ids = sample_ids[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encoded)
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        embeddings_cpu = embeddings.cpu().float().numpy()

        for sid, emb in zip(batch_ids, embeddings_cpu):
            embeddings_map[sid] = emb

    return embeddings_map


def build_output_name(jsonl_path: str, model_name: str) -> str:
    """Construct output Parquet filename from input JSONL name and model."""
    base = os.path.splitext(os.path.basename(jsonl_path))[0]
    # Use the last path component of model name (e.g. "Qwen3-Embedding-4B")
    model_tag = model_name.rstrip("/").split("/")[-1]
    return f"{base}__{model_tag}.parquet"


def main():
    parser = argparse.ArgumentParser(description="Embed prompts from inference JSONL files.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=4, help="Encoding batch size (default: 4)")
    parser.add_argument("--max-length", type=int, default=8192, help="Max token length for truncation (default: 8192)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading prompts from: {args.input}")
    prompts = collect_unique_prompts(args.input)
    print(f"Found {len(prompts)} unique samples")

    embeddings_map = encode_prompts(
        prompts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Build DataFrame: sample_idx + embedding dimensions as columns
    sample_ids = sorted(embeddings_map.keys())
    emb_matrix = np.stack([embeddings_map[sid] for sid in sample_ids])
    dim = emb_matrix.shape[1]

    df = pd.DataFrame(emb_matrix, columns=[f"dim_{i}" for i in range(dim)])
    df.insert(0, "sample_idx", sample_ids)

    out_name = build_output_name(args.input, args.model)
    out_path = os.path.join(args.output_dir, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} embeddings (dim={dim}) to: {out_path}")


if __name__ == "__main__":
    main()
