import os
import argparse
import json
import time
from vllm import LLM, SamplingParams, RequestOutput
from typing import Any

from env_vars import RESULTS_DIR
from model_loader import get_model_and_tokenizer, MODEL_REGISTRY
from dataset_loader import get_dataset, DATASET_REGISTRY


def basic_inference(model: LLM, sampling_params: SamplingParams, sample: dict[str, Any]) -> tuple[RequestOutput, float]:
    messages = [
        {"role": "user", "content": f"Please think step by step, and provide your answer in \\boxed{{}}. Question: {sample['question']}"}
    ]
    
    model_name = getattr(model.llm_engine.model_config, 'model', '')
    start_time = time.perf_counter()
    if "qwen3" in model_name.lower():
        outputs = model.chat([messages], sampling_params, chat_template_kwargs={'enable_thinking': True})
    else:
        outputs = model.chat([messages], sampling_params)
    elapsed_time = time.perf_counter() - start_time
    return outputs[0], elapsed_time

def serialize_request_output(output: RequestOutput) -> dict:
    return {
        'request_id': output.request_id,
        'prompt': output.prompt,
        'prompt_token_ids': output.prompt_token_ids,
        'prompt_logprobs': output.prompt_logprobs,
        'outputs': [
            {
                'index': o.index,
                'text': o.text,
                'token_ids': list(o.token_ids),
                'cumulative_logprob': o.cumulative_logprob,
                'logprobs': o.logprobs,
                'finish_reason': o.finish_reason,
                'stop_reason': o.stop_reason
            } for o in output.outputs
        ],
        'finished': output.finished,
    }


def collect_processed_keys(output_dir, model_name, dataset_name, B, G, num_chunks):
    processed = set()
    model_safe = model_name.replace('/', '-')
    for chunk_idx in range(num_chunks):
        file_path = os.path.join(output_dir, f"inference_{model_safe}_{dataset_name}_G{G}_B{B}_chunk{chunk_idx}.jsonl")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        processed.add((data["sample_idx"], data["group_idx"]))
                    except json.JSONDecodeError:
                        continue
    return processed


def main():
    parser = argparse.ArgumentParser(description="Run basic inference on reasoning benchmarks")
    parser.add_argument('--model', type=str, required=True, 
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Name of the model to use')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=list(DATASET_REGISTRY.keys()),
                        help='Name of the dataset to use')
    parser.add_argument('--num-chunks', type=int, required=True,
                        help='Total number of chunks to split the dataset into')
    parser.add_argument('--chunk-idx', type=int, required=True,
                        help='Index of the current chunk (0-indexed)')
    parser.add_argument('--max-new-tokens', '-B', type=int, required=True,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--num-rollouts', '-N', type=int, required=True,
                        help='Number of groups per sample (total rollouts = N * G)')
    parser.add_argument('--group-size', '-G', type=int, default=1,
                        help='Group size for vLLM generation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: RESULTS_DIR/basic)')

    args = parser.parse_args()

    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx
    B = args.max_new_tokens
    N = args.num_rollouts
    G = args.group_size
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError(f"Chunk index {chunk_idx} must be in range [0, {num_chunks - 1}]")
    
    # Set up output directory
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = get_dataset(args.dataset)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Load model
    print(f"Loading model: {args.model}...")
    model, sampling_params = get_model_and_tokenizer(args.model, device="auto", max_model_len=B)
    sampling_params.n = G
    print(f"Model loaded!")

    # Calculate start and end indices for this chunk
    total_samples = len(dataset)
    samples_per_chunk = total_samples // num_chunks
    remainder = total_samples % num_chunks
    start_idx = chunk_idx * samples_per_chunk + min(chunk_idx, remainder)
    end_idx = start_idx + samples_per_chunk + (1 if chunk_idx < remainder else 0)
    print(f"Chunk {chunk_idx}/{num_chunks}: Processing samples {start_idx} to {end_idx - 1} ({end_idx - start_idx} samples)")

    output_file = os.path.join(
        output_dir, 
        f"inference_{args.model.replace('/', '-')}_{args.dataset}_G{G}_B{B}_chunk{chunk_idx}.jsonl"
    )
    processed_keys = collect_processed_keys(output_dir, args.model, args.dataset, B, G, num_chunks)
    print(f"Found {len(processed_keys)} already processed (sample, group) pairs across all chunks")

    with open(output_file, 'a') as f:
        for sample_idx in range(start_idx, end_idx):
            sample = dataset[sample_idx]
            
            # Process N groups of size G
            for group_idx in range(N):
                # We skip if the entire group is processed
                if (sample_idx, group_idx) in processed_keys:
                    continue
                
                request_output, elapsed_time = basic_inference(model, sampling_params, sample)
                
                result = {
                    "sample_idx": sample_idx,
                    "group_idx": group_idx,
                    "model": args.model,
                    "dataset": args.dataset,
                    "B": B,
                    "G": G,
                    "method_output": serialize_request_output(request_output),
                    "generation_time": elapsed_time,
                    "sample": sample,
                }
                f.write(json.dumps(result) + '\n')
                f.flush()
            
            # Progress logging
            samples_done = sample_idx - start_idx + 1
            if samples_done % 10 == 0:
                print(f"Processed {samples_done}/{end_idx - start_idx} samples")

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()


