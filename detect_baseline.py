from huggingface_hub import HfApi

api = HfApi()


def detect_baseline(model_id: str) -> str | None:
    """
    Given a model ID, try to find the full-precision baseline it was quantized from.
    Returns the baseline model ID, or None if it can't be determined.
    """
    info = api.model_info(model_id)
    tags = info.tags or []

    # Most reliable: HF explicitly tags the relationship type, e.g.
    # "base_model:quantized:zai-org/GLM-5.2"
    for tag in tags:
        if tag.startswith("base_model:quantized:"):
            return tag.removeprefix("base_model:quantized:")

    # Less reliable fallback: card_data.base_model exists, but we can't confirm
    # it's a quantization relationship (could be a finetune, adapter, merge, etc.)
    card_data = info.card_data
    if card_data and getattr(card_data, "base_model", None):
        base = card_data.base_model
        base_id = base[0] if isinstance(base, list) else base
        print(f"Warning: found base_model '{base_id}' but relationship type is unconfirmed "
              f"(not explicitly tagged as 'quantized'). Verify before treating as a baseline.")
        return base_id

    return None


if __name__ == "__main__":
    for model_id in ["nvidia/GLM-5.2-NVFP4", "Qwen/Qwen2.5-0.5B-Instruct"]:
        baseline = detect_baseline(model_id)
        print(f"{model_id} -> baseline: {baseline}")