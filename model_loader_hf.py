import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import try_to_load_from_cache
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
    # Check for config.json as a proxy for the model being cached
    result = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cache_dir,
    )
    # Returns the path if cached, None if not cached, or _CACHED_NO_EXIST sentinel
    return isinstance(result, str)



def get_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    if 'deepseek' in model_name and transformers.__version__.split('.')[0] != '4':
        raise ValueError("DeepSeek R1 requires transformers version <5.0")
    model_id = MODEL_REGISTRY[model_name]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    local_files_only = is_model_cached(model_id, CACHE_DIR)
    
    import transformers.tokenization_utils_base as tu_base
    import huggingface_hub.constants as hf_constants
    old_tu_offline = tu_base.is_offline_mode
    old_hf_offline = hf_constants.HF_HUB_OFFLINE
    
    if local_files_only:
        tu_base.is_offline_mode = lambda: True
        hf_constants.HF_HUB_OFFLINE = True

    tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		trust_remote_code=trust_remote_code,
		cache_dir=CACHE_DIR,
		local_files_only=local_files_only,
	)
    model = AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype=torch_dtype,
		device_map=device if device == "auto" else None,
		trust_remote_code=trust_remote_code,
		cache_dir=CACHE_DIR,
		local_files_only=local_files_only
	)
    model.eval()
    
    if local_files_only:
        tu_base.is_offline_mode = old_tu_offline
        hf_constants.HF_HUB_OFFLINE = old_hf_offline
    
    if device != "auto": model = model.to(device)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    if "Qwen3" in model_id:
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.95
        model.generation_config.top_k = 20
        model.generation_config.min_p = 0
    elif model_name == "deepseek-r1-llama-8b":
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.95
    elif model_name == "deepseek-r1-qwen-7b":
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.95
    
    return model, tokenizer