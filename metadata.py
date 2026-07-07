"""
Static descriptions for models and quantization methods.

Keyed by the identifiers that flow through the pipeline: HF repo id for models,
precision label (fp16, int4, ...) for quant methods. Used to fill the models
and quant_methods lookup tables. Add an entry when you benchmark something new;
unknown keys still store, just with blank metadata.
"""

MODEL_METADATA = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "display_name": "Qwen2.5 0.5B Instruct", "param_count": 500_000_000, "family": "Qwen",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "display_name": "Qwen2.5 1.5B Instruct", "param_count": 1_500_000_000, "family": "Qwen",
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "display_name": "TinyLlama 1.1B Chat", "param_count": 1_100_000_000, "family": "TinyLlama",
    },
}

QUANT_METADATA = {
    "fp32": {"method_family": "none", "bits": 32, "description": "Full precision baseline (fp32)"},
    "fp16": {"method_family": "none", "bits": 16, "description": "Full precision baseline (fp16)"},
    "int8": {"method_family": "bitsandbytes", "bits": 8, "description": "bitsandbytes 8-bit"},
    "int4": {"method_family": "bitsandbytes", "bits": 4, "description": "bitsandbytes 4-bit (nf4)"},
    "gguf_q4_k_m": {"method_family": "gguf", "bits": 4, "description": "GGUF Q4_K_M via llama.cpp"},
}


def model_meta(model_id: str) -> dict:
    """Metadata for a model, falling back to a sensible default for unknown repos."""
    return MODEL_METADATA.get(model_id, {
        "display_name": model_id, "param_count": None, "family": None,
    })


def quant_meta(precision: str) -> dict:
    """Metadata for a precision label, falling back to blanks for unknown labels."""
    return QUANT_METADATA.get(precision, {
        "method_family": "unknown", "bits": None, "description": None,
    })
