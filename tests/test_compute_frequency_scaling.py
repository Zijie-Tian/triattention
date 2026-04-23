"""Test compute_frequency_scaling to understand what scale it actually captures."""
from __future__ import annotations

import torch
from transformers import AutoConfig

from triattention.common.rope_utils import build_rotary, compute_frequency_scaling


def test_standard_rope():
    """Standard RoPE without any scaling — should give all 1s."""
    print("=" * 60)
    print("Test 1: Standard RoPE (no scaling)")
    print("=" * 60)

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 32
    config.max_position_embeddings = 128
    config.rope_scaling = None  # No scaling

    device = torch.device("cpu")
    dtype = torch.float32

    rotary = build_rotary(device, None, dtype, config=config)  # type: ignore[arg-type]
    scale = compute_frequency_scaling(rotary, config.head_dim, dtype, device)

    print(f"scale values: {scale}")
    print(f"All ones? {torch.allclose(scale, torch.ones_like(scale))}")


def test_with_attention_scaling():
    """RoPE with attention_scaling — scale may differ from 1."""
    print("\n" + "=" * 60)
    print("Test 2: RoPE with attention_factor scaling")
    print("=" * 60)

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 32
    config.max_position_embeddings = 128
    # Simulate attention scaling (Qwen uses this for long-context)
    config.rope_scaling = {
        "rope_type": "default",
        "attention_factor": 1.5,
    }

    device = torch.device("cpu")
    dtype = torch.float32

    rotary = build_rotary(device, None, dtype, config=config)  # type: ignore[arg-type]
    scale = compute_frequency_scaling(rotary, config.head_dim, dtype, device)

    print(f"scale values: {scale}")
    print(f"All ones? {torch.allclose(scale, torch.ones_like(scale))}")
    print(f"attention_scaling attr on rotary: {getattr(rotary, 'attention_scaling', 'N/A')}")


def test_why_position_zero():
    """
    Explain why position 0 is used.

    At position 0:
        cos(0 * theta) = 1, sin(0 * theta) = 0
    So if rotary() returns PURE rotation matrices (no extra scaling),
    scale should be [1, 1, 1, ...].

    BUT if the rotary embedding applies an EXTRA scaling factor
    (e.g., attention_factor, YaRN mscale), that factor multiplies
    BOTH cos and sin. Then:
        scale = sqrt((s*1)^2 + (s*0)^2) = s

    This is exactly what compute_frequency_scaling measures:
    the per-frequency EXTRA scaling applied by the RoPE implementation.
    """
    print("\n" + "=" * 60)
    print("Test 3: Why position 0? — Manual verification")
    print("=" * 60)

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 32
    config.max_position_embeddings = 128
    config.rope_scaling = None

    device = torch.device("cpu")
    dtype = torch.float32

    rotary = build_rotary(device, None, dtype, config=config)  # type: ignore[arg-type]

    # At position 0, the raw rotation should be cos=1, sin=0
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, config.head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]

    print(f"At position 0, cos values (first 8): {cos0[:8]}")
    print(f"At position 0, sin values (first 8): {sin0[:8]}")
    print(f"cos0[0::2] (freq pairs): {cos0[0::2][:5]}")
    print(f"sin0[0::2] (freq pairs): {sin0[0::2][:5]}")

    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    print(f"Computed scale (first 5): {scale[:5]}")

    # Now compare with position 1
    position_ids = torch.ones(1, 1, device=device, dtype=torch.long)
    cos1, sin1 = rotary(probe, position_ids)
    cos1_0 = cos1[0, 0]
    sin1_0 = sin1[0, 0]
    scale1 = torch.sqrt(cos1_0[0::2].pow(2) + sin1_0[0::2].pow(2))
    print(f"\nAt position 1, scale (first 5): {scale1[:5]}")
    print("Note: At position 1, cos=cos(theta), sin=sin(theta), but the")
    print("      EXTRA scaling factor (if any) is the SAME as position 0.")
    print("      Using position 0 simplifies the math to just measure the scaling.")


def test_llama_with_yarn():
    """Test Llama with YaRN scaling."""
    print("\n" + "=" * 60)
    print("Test 4: Llama RoPE with YaRN scaling")
    print("=" * 60)

    try:
        from transformers import LlamaConfig
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            rope_theta=10000.0,
            vocab_size=32000,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            },
        )
    except Exception as e:
        print(f"Skipped: {e}")
        return

    device = torch.device("cpu")
    dtype = torch.float32

    try:
        rotary = build_rotary(device, None, dtype, config=config)  # type: ignore[arg-type]
    except ImportError as e:
        print(f"Skipped: {e}")
        return

    scale = compute_frequency_scaling(rotary, 32, dtype, device)
    print(f"scale values: {scale}")
    print(f"All ones? {torch.allclose(scale, torch.ones_like(scale))}")

    # Print attention_scaling if present
    print(f"attention_scaling attr: {getattr(rotary, 'attention_scaling', 'N/A')}")

    # For YaRN, there is usually an mscale factor. Let's check if it's reflected.
    if not torch.allclose(scale, torch.ones_like(scale)):
        print("\n>>> YaRN scaling IS reflected in freq_scale!")
    else:
        print("\n>>> YaRN scaling is NOT in freq_scale (may be handled elsewhere).")


if __name__ == "__main__":
    test_standard_rope()
    test_with_attention_scaling()
    test_why_position_zero()
    test_llama_with_yarn()
