# preprocess

Extracts and embeds prompts from inference JSONL rollout files.

## embed.py

For each unique `sample_idx` in a rollout JSONL, pulls the prompt from `method_output.prompt`, encodes it with a Qwen3 embedding model, and saves a Parquet file of normalized embeddings.

**Output:** `output/<jsonl_basename>__<model_tag>.parquet`  
**Schema:** `sample_idx, dim_0, dim_1, ..., dim_N` (one row per sample, L2-normalized)

### Usage

```bash
# Default model (Qwen/Qwen3-Embedding-4B)
python embed.py --input /path/to/inference_qwen3-4b_math500_G4_B32768_chunk0.jsonl

# Smaller model
python embed.py --input /path/to/file.jsonl --model Qwen/Qwen3-Embedding-0.6B

# Tune batch size if OOM
python embed.py --input /path/to/file.jsonl --batch-size 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to rollout JSONL file |
| `--model` | `Qwen/Qwen3-Embedding-4B` | HuggingFace model name |
| `--batch-size` | `4` | Encoding batch size |
| `--max-length` | `8192` | Token truncation limit |
| `--output-dir` | `output/` | Output directory |

Uses `cuda:0` if available, otherwise CPU.
