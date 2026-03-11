"""Shared utilities for the Typed Context project."""

import torch
import yaml
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path="configs/llama8b.yaml"):
    """Load model configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", precision="fp16",
               device_map="auto", attn_implementation=None):
    """
    Load a causal LM with the specified precision.

    Args:
        model_name: HuggingFace model identifier
        precision: one of 'fp16', 'fp8', 'int8', 'int4', 'auto'
        device_map: device placement strategy
        attn_implementation: 'eager', 'sdpa', or 'flash_attention_2'. Use 'eager'
                             when you need output_attentions=True.

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    dtype = torch.float16

    if precision == "auto":
        dtype = "auto"
    elif precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp8":
        # FP8 models store weights in FP8 format; load with bfloat16 compute
        dtype = torch.bfloat16
    elif precision == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Check GPU availability
    if not torch.cuda.is_available():
        device_map = "cpu"
        dtype = torch.float32
        quant_config = None  # quantization requires GPU

    load_kwargs = dict(
        device_map=device_map,
    )
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_model_from_config(config_path="configs/llama8b.yaml", precision=None,
                           device_map="auto", attn_implementation=None):
    """
    Load model using parameters from a YAML config file.

    The config must contain 'model_name'. If precision is not specified,
    it is auto-detected from the model name (e.g., FP8 suffix -> 'fp8').

    Returns:
        model, tokenizer, config_dict
    """
    config = load_config(config_path)
    model_name = config["model_name"]

    if precision is None:
        # Auto-detect precision from model name
        name_lower = model_name.lower()
        if "fp8" in name_lower:
            precision = "fp8"
        elif "awq" in name_lower or "gptq" in name_lower:
            precision = "auto"
        else:
            precision = "bf16"

    model, tokenizer = load_model(model_name, precision=precision,
                                  device_map=device_map,
                                  attn_implementation=attn_implementation)
    return model, tokenizer, config


def apply_chat_template(tokenizer, system_content, user_content, add_generation_prompt=True):
    """
    Tokenize a system+user conversation using the model's chat template.

    Returns:
        input_ids: list[int] — full token sequence
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt,
            enable_thinking=False
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt
        )
    return input_ids


def assign_source_labels(tokenizer, system_content, user_content, add_generation_prompt=True):
    """
    Assign token-level source labels using the system-only tokenization boundary method.

    Method: tokenize the system-only message to find the boundary, then label
    everything up to that boundary as system (0) and the rest as user (1).

    Args:
        tokenizer: HuggingFace tokenizer with chat template support
        system_content: system prompt text
        user_content: user message text

    Returns:
        input_ids: list[int] — full token sequence
        source_labels: np.ndarray of shape (seq_len,) — 0=system, 1=user
    """
    # Tokenize system-only to find boundary
    sys_messages = [{"role": "system", "content": system_content}]
    try:
        sys_ids = tokenizer.apply_chat_template(
            sys_messages, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        sys_ids = tokenizer.apply_chat_template(
            sys_messages, add_generation_prompt=False
        )
    sys_len = len(sys_ids)

    # Tokenize full conversation
    full_ids = apply_chat_template(tokenizer, system_content, user_content, add_generation_prompt)
    full_len = len(full_ids)

    # Labels: 0 for system tokens, 1 for user tokens
    source_labels = np.array([0] * sys_len + [1] * (full_len - sys_len))
    return full_ids, source_labels


def assign_source_labels_3type(tokenizer, system_content, user_content, external_content,
                                add_generation_prompt=True):
    """
    Assign token-level source labels for 3-type setup.

    system=0, user=1, external=2

    The external content is appended after the user message within the user turn.

    Returns:
        input_ids: list[int]
        source_labels: np.ndarray of shape (seq_len,) — 0=system, 1=user, 2=external
    """
    # System-only boundary
    sys_messages = [{"role": "system", "content": system_content}]
    try:
        sys_ids = tokenizer.apply_chat_template(
            sys_messages, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        sys_ids = tokenizer.apply_chat_template(
            sys_messages, add_generation_prompt=False
        )
    sys_len = len(sys_ids)

    # User-only (system + user, no external) boundary
    user_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        user_ids = tokenizer.apply_chat_template(
            user_messages, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        user_ids = tokenizer.apply_chat_template(
            user_messages, add_generation_prompt=False
        )
    user_len = len(user_ids)

    # Full conversation with external content appended to user turn
    combined_user = user_content + "\n\n" + external_content
    full_ids = apply_chat_template(tokenizer, system_content, combined_user, add_generation_prompt)
    full_len = len(full_ids)

    # Labels
    source_labels = np.zeros(full_len, dtype=np.int64)
    source_labels[:sys_len] = 0       # system
    source_labels[sys_len:user_len] = 1  # user
    source_labels[user_len:] = 2      # external

    return full_ids, source_labels
