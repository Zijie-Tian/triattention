"""Test script for build_rotary — prints RoPE cos/sin matrices for inspection."""
from __future__ import annotations

import torch
from transformers import AutoConfig

from triattention.common.rope_utils import build_rotary, compute_frequency_scaling


def test_qwen_style_rotary() -> None:
    """Build a Qwen-style RoPE and print cos/sin tables."""
    print("=" * 60)
    print("Test: Qwen-style RoPE (model_type='qwen3')")
    print("=" * 60)

    # Manually construct a config so we don't need to download a real model
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
    )
    # Override to small dimensions for readable output
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 32
    config.max_position_embeddings = 128
    config.rope_scaling = None

    device = torch.device("cpu")
    dtype = torch.float32

    rotary = build_rotary(
        cache_device=device,
        model_path=None,  # type: ignore[arg-type]
        dtype=dtype,
        config=config,
    )
    print(f"Rotary type: {type(rotary).__name__}")
    print(f"RoPE style: {getattr(rotary, '_rope_style', 'N/A')}")
    print(f"inv_freq shape: {rotary.inv_freq.shape}")
    print(f"inv_freq (first 5): {rotary.inv_freq[:5]}")

    # Generate cos/sin for positions 0..7
    seq_len = 8
    head_dim = config.head_dim
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(base, position_ids)
    
    cos = cos[0]   # [seq_len, head_dim]
    sin = sin[0]   # [seq_len, head_dim]    

    print(f"\ncos matrix shape: {cos.shape}")
    print("cos (position=0..7, first 8 dims):")
    print(cos[:, :8])

    print(f"\nsin matrix shape: {sin.shape}")
    print("sin (position=0..7, first 8 dims):")
    print(sin[:, :8])

    # Compute frequency scaling factors
    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    print(f"\nfreq_scale shape: {freq_scale.shape}")
    print(f"freq_scale values: {freq_scale}")

    # Verify RoPE property: cos² + sin² = 1 for each frequency pair
    freq_count = head_dim // 2
    for pos in [0, 1, 4, 7]:
        c = cos[pos, 0::2]
        s = sin[pos, 0::2]
        mag = torch.sqrt(c * c + s * s)
        print(f"\nPosition {pos}: sqrt(cos² + sin²) for each freq pair = {mag[:5]}")


def test_llama_style_rotary() -> None:
    """Build a Llama-style RoPE and print cos/sin tables."""
    print("\n" + "=" * 60)
    print("Test: Llama-style RoPE (model_type='llama')")
    print("=" * 60)

    try:
        config = AutoConfig.from_pretrained(
            "meta-llama/Llama-3.2-1B",
        )
    except Exception:
        # Fallback: manually create a llama-like config
        from transformers import LlamaConfig
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            rope_theta=10000.0,
            vocab_size=32000,
        )

    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 32
    config.max_position_embeddings = 128
    config.rope_scaling = None

    device = torch.device("cpu")
    dtype = torch.float32

    try:
        rotary = build_rotary(
            cache_device=device,
            model_path=None,  # type: ignore[arg-type]
            dtype=dtype,
            config=config,
        )
    except ImportError as e:
        print(f"Skipped: {e}")
        return

    print(f"Rotary type: {type(rotary).__name__}")
    print(f"RoPE style: {getattr(rotary, '_rope_style', 'N/A')}")
    print(f"inv_freq shape: {rotary.inv_freq.shape}")
    print(f"inv_freq (first 5): {rotary.inv_freq[:5]}")

    seq_len = 8
    head_dim = config.head_dim
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(base, position_ids)
    cos = cos[0]
    sin = sin[0]

    print(f"\ncos matrix shape: {cos.shape}")
    print("cos (position=0..7, first 8 dims):")
    print(cos[:, :8])

    print(f"\nsin matrix shape: {sin.shape}")
    print("sin (position=0..7, first 8 dims):")
    print(sin[:, :8])

    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    print(f"\nfreq_scale shape: {freq_scale.shape}")
    print(f"freq_scale values: {freq_scale}")

    # Verify RoPE property
    for pos in [0, 1, 4, 7]:
        c = cos[pos, 0::2]
        s = sin[pos, 0::2]
        mag = torch.sqrt(c * c + s * s)
        print(f"\nPosition {pos}: sqrt(cos² + sin²) for each freq pair = {mag[:5]}")


if __name__ == "__main__":
    test_qwen_style_rotary()
    test_llama_style_rotary()
