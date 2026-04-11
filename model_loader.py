import torch
from huggingface_hub import try_to_load_from_cache
from vllm import LLM, SamplingParams
from env_vars import CACHE_DIR


MODEL_REGISTRY = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B", 
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "deepseek-r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def is_model_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """Check if a model is available in the local cache."""
    result = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cache_dir,
    )
    return isinstance(result, str)


def get_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    max_model_len: int = 16384,
) -> tuple[LLM, SamplingParams]:
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model_id = MODEL_REGISTRY[model_name]
    
    llm = LLM(
        model=model_id,
        download_dir=CACHE_DIR,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
    )
    
    if "qwen3" in model_name:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_tokens=max_model_len,
        )
    elif model_name == "deepseek-r1-llama-8b" or model_name == "deepseek-r1-qwen-7b":
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=max_model_len,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_model_len,
        )
    
    return llm, sampling_params

