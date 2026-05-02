"""
run_inference.py — Multi-GPU scheduled inference with work-stealing dispatch.

Main scheduler that:
1. Loads a dataset and builds a batching strategy (baseline or logt).
2. Spawns N worker processes, one per GPU, each loading the full model.
3. For each epoch: dispatches batches via multiprocessing queues,
   collects results, logs per-batch and per-epoch data to JSONL.
4. Handles preemption: on restart, completed epochs are skipped;
   incomplete epochs are re-run from scratch for timing accuracy
   (unless --fast-resume is set).
"""

import os
import sys
import argparse
import json
import time
import datetime
import signal
import traceback
import multiprocessing as mp
from functools import partial
from typing import Any

import torch

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader_hf import get_model_and_tokenizer, MODEL_REGISTRY
from dataset_loader import get_dataset, DATASET_REGISTRY
from strategies import BaselineStrategy, LogTMaxStrategy, LogTSumStrategy
from gpu_generate import GPU_GENERATE_REGISTRY
from env_vars import RESULTS_DIR


# ─── Sentinel for stopping workers ───────────────────────────────────────────
_STOP_SENTINEL = "STOP"


# ─── Worker process ──────────────────────────────────────────────────────────
def worker_fn(
    gpu_id: int,
    num_gpus: int,
    model_name: str,
    gen_strategy_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_event: mp.Event,
    G: int,
    B: int,
    chunk_size: int = 32,
    mem_threshold: float = 0.95,
):
    """
    Worker subprocess: loads model on assigned GPU, then loops consuming
    tasks from task_queue and posting results to result_queue.

    The inner loop uses task_queue.get() with NO timeout. This is a
    blocking OS-level wait (semaphore/condition variable) — the process
    sleeps and uses 0% CPU while waiting between epochs or during queue
    drain. There is no busy-spinning.
    """
    try:
        # ── Pin this process to a dedicated subset of CPU cores ──────────
        # With --cpus-per-gpu=4, Slurm gives 4 CPUs per GPU. We assign
        # each worker its own slice so cores are never shared.
        total_cpus = os.cpu_count() or (num_gpus * 4)
        cpus_per_worker = max(1, total_cpus // num_gpus)
        start_cpu = gpu_id * cpus_per_worker
        end_cpu = min(start_cpu + cpus_per_worker, total_cpus)
        try:
            os.sched_setaffinity(0, set(range(start_cpu, end_cpu)))
            print(f"[Worker {gpu_id}] Pinned to CPU cores {start_cpu}-{end_cpu - 1}.",
                  flush=True)
        except (AttributeError, OSError) as e:
            print(f"[Worker {gpu_id}] CPU affinity not set ({e}); sharing all cores.",
                  flush=True)

        # ── Pin this process to a single GPU ─────────────────────────────
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"  # After setting CUDA_VISIBLE_DEVICES, device 0 is our GPU

        print(f"[Worker {gpu_id}] Loading model '{model_name}' on GPU {gpu_id}...",
              flush=True)
        model, tokenizer = get_model_and_tokenizer(model_name, device=device)
        print(f"[Worker {gpu_id}] Model loaded.", flush=True)

        # Signal that we are ready
        ready_event.set()

        # Resolve the generation strategy function from the registry.
        # Import here (inside spawn'd process) after sys.path is set.
        from gpu_generate import GPU_GENERATE_REGISTRY
        if gen_strategy_name not in GPU_GENERATE_REGISTRY:
            raise ValueError(f"Unknown gen strategy: '{gen_strategy_name}'. "
                             f"Available: {list(GPU_GENERATE_REGISTRY.keys())}")
        gen_fn = GPU_GENERATE_REGISTRY[gen_strategy_name]

        # Bind extra hyperparameters for the memory-safe strategy
        if gen_strategy_name == "pruned_kernel_safe":
            gen_fn = partial(gen_fn, K=chunk_size, mem_threshold=mem_threshold)

        while True:
            # Blocking wait — 0% CPU usage while queue is empty.
            # The main process guarantees the queue will eventually contain
            # a task or a _STOP_SENTINEL (see drain logic in main).
            task = task_queue.get()
            if task == _STOP_SENTINEL:
                break

            epoch = task["epoch"]
            batch_idx = task["batch_idx"]
            samples = task["samples"]
            sample_indices = task["sample_indices"]

            ts = time.strftime("%H:%M:%S")
            print(
                f"[GPU {gpu_id}] {ts} | epoch={epoch} batch={batch_idx} "
                f"n_samples={len(samples)} sample_ids={sample_indices}",
                flush=True,
            )

            try:
                result = gen_fn(model, tokenizer, samples, G=G, B=B)

                ts_done = time.strftime("%H:%M:%S")
                oom_tag = " [OOM-split]" if result["oom_occurred"] else ""
                print(
                    f"[GPU {gpu_id}] {ts_done} | epoch={epoch} batch={batch_idx} "
                    f"done in {result['generation_time']:.1f}s{oom_tag}",
                    flush=True,
                )

                result_queue.put({
                    "status": "ok",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "sample_indices": sample_indices,
                    "gpu_id": gpu_id,
                    "decoded_outputs": result["decoded_outputs"],
                    "output_token_lengths": result["output_token_lengths"],
                    "generation_time": result["generation_time"],
                    "oom_occurred": result["oom_occurred"],
                })
            except Exception as e:
                # Report error but keep worker alive for future tasks
                result_queue.put({
                    "status": "error",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "sample_indices": sample_indices,
                    "gpu_id": gpu_id,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })

    except Exception as e:
        print(f"[Worker {gpu_id}] Fatal error: {e}", flush=True)
        traceback.print_exc()
        ready_event.set()  # Unblock main process even on failure


# ─── Log file I/O ────────────────────────────────────────────────────────────

def load_existing_log(log_path: str) -> list[dict]:
    """Load all records from an existing JSONL log file."""
    records = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_completed_epochs(records: list[dict]) -> set[int]:
    """Return set of epoch indices that have a valid epoch_summary."""
    return {
        r["epoch"] for r in records
        if r.get("type") == "epoch_summary" and r.get("timing_valid", True)
    }


def get_batch_records_for_epoch(records: list[dict], epoch: int) -> list[dict]:
    """Return all batch records for a given epoch."""
    return [
        r for r in records
        if r.get("type") == "batch" and r.get("epoch") == epoch
    ]


def write_record(log_path: str, record: dict):
    """Append a single JSON record to the log file."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ─── Strategy construction ───────────────────────────────────────────────────

def build_strategy(args, dataset_keys: list[int]):
    """Build the strategy object from CLI args."""
    if args.strategy == "baseline":
        return BaselineStrategy(keys=dataset_keys, G=args.group_size, bg=args.batch_grouping)
    elif args.strategy == "logt_max":
        return LogTMaxStrategy(
            keys=dataset_keys,
            G=args.group_size,
            mem_limit=args.mem_limit,
            alpha=args.alpha,
        )
    elif args.strategy == "logt_sum":
        return LogTSumStrategy(
            keys=dataset_keys,
            G=args.group_size,
            mem_limit=args.mem_limit,
            alpha=args.alpha,
            mc_nsamples=args.mc_nsamples,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


# ─── Warm up strategy from prior epoch data ──────────────────────────────────

def warm_up_strategy(strategy, records: list[dict], completed_epochs: set[int]):
    """
    Feed completed epoch data into the strategy so that adaptive strategies
    (LogTStrategy) have their distributions fitted before the next epoch.
    """
    for epoch_idx in sorted(completed_epochs):
        batch_records = get_batch_records_for_epoch(records, epoch_idx)
        for br in batch_records:
            key_length_data = []
            for s_idx, lengths in zip(br["sample_indices"], br["output_token_lengths"]):
                for length in lengths:
                    key_length_data.append((s_idx, length))
            strategy.ingest_run_data(key_length_data)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU scheduled inference with batching strategies"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model name from MODEL_REGISTRY")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset name from DATASET_REGISTRY")
    parser.add_argument("--num-gpus", type=int, required=True,
                        help="Number of GPU workers")
    parser.add_argument("--num-epochs", type=int, required=True,
                        help="Number of epochs to run")
    parser.add_argument("--group-size", "-G", type=int, required=True,
                        help="Number of rollouts per prompt per epoch")
    parser.add_argument("--max-new-tokens", "-B", type=int, required=True,
                        help="Max new tokens for generation")
    parser.add_argument("--strategy", type=str, default="baseline",
                        choices=["baseline", "logt_max", "logt_sum"],
                        help="Batching strategy to use")
    parser.add_argument("--gen-strategy", type=str, default="recursive_retry",
                        help="GPU generation strategy name (see GPU_GENERATE_REGISTRY in gpu_generate.py)")
    parser.add_argument("--batch-grouping", "--bg", type=int, default=1,
                        help="Batch grouping parameter for baseline strategy")
    parser.add_argument("--mem-limit", type=int, default=250000,
                        help="KV-cache total token budget (for logt strategy)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="OOM probability threshold (for logt strategy)")
    parser.add_argument("--mc-nsamples", type=int, default=10000,
                        help="Monte Carlo samples (for logt strategy)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: RESULTS_DIR/scheduled_inference)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSONL filename (auto-named if absent)")
    parser.add_argument("--fast-resume", action="store_true",
                        help="Use cached batch results on resume (timing marked invalid)")
    parser.add_argument("--chunk-size", "-K", type=int, default=32,
                        help="Tokens per memory-check chunk (for pruned_kernel_safe strategy)")
    parser.add_argument("--mem-threshold", type=float, default=0.95,
                        help="GPU memory fraction (0-1) that triggers batch split "
                             "(for pruned_kernel_safe strategy)")

    args = parser.parse_args()

    # ─── Validate GPUs ────────────────────────────────────────────────────
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("ERROR: No CUDA devices available.", flush=True)
        sys.exit(1)
    if args.num_gpus > available_gpus:
        print(f"ERROR: Requested {args.num_gpus} GPUs but only {available_gpus} available.",
              flush=True)
        sys.exit(1)
    if args.num_gpus <= 0:
        print("ERROR: --num-gpus must be > 0.", flush=True)
        sys.exit(1)

    # ─── Output path ─────────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "scheduled_inference")
    os.makedirs(output_dir, exist_ok=True)

    if args.output_file:
        log_path = os.path.join(output_dir, args.output_file)
    else:
        name_parts = [
            f"inference_{args.model}_{args.dataset}_{args.strategy}",
            f"gen{args.gen_strategy}",
            f"bg{args.batch_grouping}",
            f"G{args.group_size}_B{args.max_new_tokens}",
            f"gpus{args.num_gpus}"
        ]
        if args.gen_strategy == "pruned_kernel_safe":
            name_parts.append(f"K{args.chunk_size}_mem{args.mem_threshold}")
            
        log_path = os.path.join(output_dir, "_".join(name_parts) + ".jsonl")

    print(f"Output log: {log_path}", flush=True)

    # ─── Load dataset ────────────────────────────────────────────────────
    print(f"Loading dataset '{args.dataset}'...", flush=True)
    dataset = get_dataset(args.dataset)
    dataset_keys = list(range(len(dataset)))
    print(f"Dataset loaded: {len(dataset)} samples.", flush=True)

    # ─── Resume logic ────────────────────────────────────────────────────
    existing_records = load_existing_log(log_path)
    completed_epochs = get_completed_epochs(existing_records)
    if completed_epochs:
        print(f"Resuming: {len(completed_epochs)} epoch(s) already complete: "
              f"{sorted(completed_epochs)}", flush=True)

    # ─── Build strategy and warm up with prior data ──────────────────────
    strategy = build_strategy(args, dataset_keys)
    if completed_epochs:
        warm_up_strategy(strategy, existing_records, completed_epochs)

    # ─── Use 'spawn' start method for CUDA safety ────────────────────────
    ctx = mp.get_context("spawn")

    # ─── Validate gen strategy ────────────────────────────────────────────
    if args.gen_strategy not in GPU_GENERATE_REGISTRY:
        print(f"ERROR: Unknown --gen-strategy '{args.gen_strategy}'. "
              f"Available: {list(GPU_GENERATE_REGISTRY.keys())}", flush=True)
        sys.exit(1)

    # ─── Spawn workers ───────────────────────────────────────────────────
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    ready_events = []
    workers = []

    print(f"Spawning {args.num_gpus} worker(s) "
          f"[gen_strategy='{args.gen_strategy}']...", flush=True)
    model_load_start = time.perf_counter()

    for gpu_id in range(args.num_gpus):
        ready_event = ctx.Event()
        ready_events.append(ready_event)
        p = ctx.Process(
            target=worker_fn,
            args=(gpu_id, args.num_gpus, args.model, args.gen_strategy,
                  task_queue, result_queue, ready_event,
                  args.group_size, args.max_new_tokens,
                  args.chunk_size, args.mem_threshold),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Wait for all workers to load their models
    print("Waiting for all workers to load models...", flush=True)
    for gpu_id, event in enumerate(ready_events):
        event.wait()
        if not workers[gpu_id].is_alive():
            print(f"ERROR: Worker {gpu_id} died during model loading.", flush=True)
            # Terminate all workers and exit
            for w in workers:
                w.terminate()
            sys.exit(1)

    model_load_time = time.perf_counter() - model_load_start
    print(f"All workers ready. Model load time: {model_load_time:.2f}s", flush=True)

    # ─── Epoch loop ──────────────────────────────────────────────────────
    try:
        for epoch_idx in range(args.num_epochs):
            if epoch_idx in completed_epochs:
                print(f"Epoch {epoch_idx}: already complete, skipping.", flush=True)
                # Strategy was already warmed up above
                continue

            print(f"\n{'='*60}", flush=True)
            print(f"Epoch {epoch_idx}/{args.num_epochs - 1}", flush=True)
            print(f"{'='*60}", flush=True)

            # Get batch ordering from strategy
            batches = strategy.get_ordering()
            num_batches = len(batches)
            print(f"Strategy '{args.strategy}' produced {num_batches} batches.", flush=True)

            # Validate that every key appears exactly once
            all_keys_in_batches = []
            for b in batches:
                all_keys_in_batches.extend(b)
            if sorted(all_keys_in_batches) != sorted(dataset_keys):
                print("ERROR: Strategy ordering does not cover all keys exactly once!",
                      flush=True)
                print(f"  Expected {len(dataset_keys)} keys, got {len(all_keys_in_batches)}.",
                      flush=True)
                sys.exit(1)

            # Check for fast-resume: load cached batch results for this epoch
            cached_batches = {}
            if args.fast_resume:
                for br in get_batch_records_for_epoch(existing_records, epoch_idx):
                    cached_batches[br["batch_idx"]] = br

            # Dispatch all batches to task queue
            epoch_wall_start = time.perf_counter()
            batches_to_generate = 0

            for batch_idx, batch_keys in enumerate(batches):
                if batch_idx in cached_batches and args.fast_resume:
                    # Fast-resume: process cached result immediately
                    br = cached_batches[batch_idx]
                    key_length_data = []
                    for s_idx, lengths in zip(br["sample_indices"], br["output_token_lengths"]):
                        for length in lengths:
                            key_length_data.append((s_idx, length))
                    strategy.ingest_run_data(key_length_data)
                    continue

                # Build sample list for this batch
                samples = [dict(dataset[k]) for k in batch_keys]
                task = {
                    "epoch": epoch_idx,
                    "batch_idx": batch_idx,
                    "samples": samples,
                    "sample_indices": batch_keys,
                }
                task_queue.put(task)
                batches_to_generate += 1

            # Collect results
            completed = 0
            error_count = 0
            while completed < batches_to_generate:
                result = result_queue.get()

                if result["status"] == "error":
                    error_count += 1
                    print(f"  [ERROR] Batch {result['batch_idx']} on GPU {result['gpu_id']}: "
                          f"{result['error']}", flush=True)
                    print(result.get("traceback", ""), flush=True)

                    if error_count > batches_to_generate // 2:
                        print("ERROR: Too many batch failures, aborting epoch.", flush=True)
                        break

                    # Write error record
                    write_record(log_path, {
                        "type": "batch_error",
                        "epoch": result["epoch"],
                        "batch_idx": result["batch_idx"],
                        "sample_indices": result["sample_indices"],
                        "gpu_id": result["gpu_id"],
                        "error": result["error"],
                        "timestamp": datetime.datetime.now().isoformat(),
                    })
                    completed += 1
                    continue

                # Success: write batch record
                batch_record = {
                    "type": "batch",
                    "epoch": result["epoch"],
                    "batch_idx": result["batch_idx"],
                    "sample_indices": result["sample_indices"],
                    "gpu_id": result["gpu_id"],
                    "generation_time": result["generation_time"],
                    "decoded_outputs": result["decoded_outputs"],
                    "output_token_lengths": result["output_token_lengths"],
                    "oom_occurred": result["oom_occurred"],
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                write_record(log_path, batch_record)

                # Feed data into strategy
                key_length_data = []
                for s_idx, lengths in zip(result["sample_indices"],
                                          result["output_token_lengths"]):
                    for length in lengths:
                        key_length_data.append((s_idx, length))
                strategy.ingest_run_data(key_length_data)

                completed += 1
                if completed % max(1, batches_to_generate // 10) == 0 or completed == batches_to_generate:
                    print(f"  Completed {completed}/{batches_to_generate} batches.", flush=True)

            epoch_wall_time = time.perf_counter() - epoch_wall_start

            # Only write epoch summary if all batches succeeded
            timing_valid = (error_count == 0 and not args.fast_resume)
            if error_count > batches_to_generate // 2:
                print(f"Epoch {epoch_idx}: ABORTED due to errors.", flush=True)
                continue

            epoch_summary = {
                "type": "epoch_summary",
                "epoch": epoch_idx,
                "epoch_wall_time": epoch_wall_time,
                "num_batches": num_batches,
                "num_batches_generated": batches_to_generate,
                "num_errors": error_count,
                "strategy": args.strategy,
                "batch_grouping": args.batch_grouping,
                "gen_strategy": args.gen_strategy,
                "timing_valid": timing_valid,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if args.gen_strategy == "pruned_kernel_safe":
                epoch_summary["chunk_size"] = args.chunk_size
                epoch_summary["mem_threshold"] = args.mem_threshold
            # Include model load time in epoch 0
            if epoch_idx == 0:
                epoch_summary["model_load_time"] = model_load_time

            write_record(log_path, epoch_summary)
            print(f"Epoch {epoch_idx} complete. Wall time: {epoch_wall_time:.2f}s "
                  f"(timing_valid={timing_valid})", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down workers...", flush=True)
    except Exception as e:
        print(f"\nFatal error in main: {e}", flush=True)
        traceback.print_exc()
    finally:
        # ── Drain the task queue before placing stop sentinels ───────────
        # If we exit mid-epoch (exception/KeyboardInterrupt), there may be
        # pending tasks already in the queue. Workers process in FIFO order,
        # so sentinels placed at the end would only be reached AFTER all
        # remaining tasks are processed — potentially a very long wait.
        # Solution: drain all pending tasks first, then place sentinels so
        # each worker's very next dequeue is the sentinel.
        drained = 0
        try:
            import queue as _queue_module
            while True:
                task_queue.get_nowait()
                drained += 1
        except Exception:
            pass  # Queue is empty
        if drained:
            print(f"Drained {drained} pending task(s) from queue.", flush=True)

        # Send exactly one stop sentinel per worker
        for _ in range(args.num_gpus):
            try:
                task_queue.put(_STOP_SENTINEL)
            except Exception:
                pass

        # Wait for workers to finish (with timeout)
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                print(f"Worker {w.pid} did not stop gracefully; terminating.", flush=True)
                w.terminate()

        print("Done.", flush=True)


if __name__ == "__main__":
    main()
