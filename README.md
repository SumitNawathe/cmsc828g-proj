# Efficient Rollout Generation for Reasoning LLMs

This is a code submission for our final project for CMSC828G.

Sumit Nawathe, Parsa Hoseini, Yang Fan Chiang


## Project Structure

```
/efficient_reasoning_inference
├── src/							# code for pruned inference
├── benchmark.py					# benchmarking code for pruned inference (Appendix G)
/len_pred
├── preprocess/						# extracting and processing embeddings for all prompts
├── pred/							# raw length and distribution prediction code and analysis
/simulation							# strategy development, simulated hyperparamter tuning
dataset_loader.py					# loading MATH500 dataset
model_loader_hf.py					# loading models with hyperparameters
gpu_generate.py						# single-GPU inference: huggingface, pruning
strategies.py						# batching strategies: static, length distribution estimation
run_inference.py					# main experiment script
run_inference.sh					# slurm script for experiments
```


## Code Highlights

In `strategies.py`:
* `BatchingStrategy`: an abstract class. Enforces an interface with two methods, one to supply rollout length information at the end of each epoch, and one to obtain a list of batches at the start of an epoch.
* `BaselineStrategy`: static batching in dataset order
* `LogTMaxStrategy`: length distribution estimation batching for padded generation (Algorithm 1 in the paper)
* `LogTSumStrategy`: length distribution estimation batching for pruned generation (Algorithm 2 in the paper). Includes Monte Carlo estimation of the OOM probability

In `gpu_generate.py`:
* `recursive_retry`: basic padded LLM inference using HuggingFace transformers
* `pruned_kernel`: pruned generation, using a token-by-token loop to remove completed sequences from the current batch

