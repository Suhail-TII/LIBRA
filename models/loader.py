"""
models/loader.py

Resolve a models.yaml entry to a loaded wrapper. The stage runners
(screen_framings, screen_templates, run_stage_b) call load_model_from_config;
it is the only model-loading indirection they need.
"""

from typing import Dict


def load_model_from_config(model_config: Dict, force_eager_attention: bool = False):
    """Load a model from a models.yaml entry, choosing the loading strategy."""
    from models.model_wrapper import (
        load_model, get_gpu_memory_gb, determine_loading_strategy,
    )

    provider = model_config["provider"]
    api_name = model_config["api_name"]
    print(f"  Loading {model_config['name']}...")

    if provider == "HuggingFace":
        available_gb = get_gpu_memory_gb()
        strategy = determine_loading_strategy(api_name, available_gb)
        print(f"    Strategy: {strategy['method']} ({strategy['reason']})")

        if strategy["method"] == "airllm":
            model = load_model(
                "huggingface", api_name,
                use_airllm=True,
                airllm_compression=strategy["quantization"] or "4bit",
            )
        else:
            load_kwargs = {"force_eager_attention": force_eager_attention}
            if strategy["quantization"] == "8bit":
                load_kwargs["load_in_8bit"] = True
            elif strategy["quantization"] == "4bit":
                load_kwargs["load_in_4bit"] = True
            model = load_model("huggingface", api_name, **load_kwargs)
    else:
        provider_map = {
            "openai": "gpt", "anthropic": "claude",
            "google": "gemini", "xai": "xai",
        }
        model = load_model(provider_map.get(provider.lower(), "gpt"), api_name)

    print("  ✓ Model loaded")
    return model
