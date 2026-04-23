"""Test TriAttention with real Llama-3.1-8B-Instruct on GPU1.

This script:
1. Calibrates frequency stats (or reuses existing)
2. Loads Llama-3.1-8B-Instruct on GPU1
3. Patches TriAttention with a small budget
4. Runs generation and logs compression behavior
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Force GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Calibration (self-contained, adapted from scripts/calibrate.py)
# ---------------------------------------------------------------------------

def _determine_rope_style(config):
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        return "half"
    return "half"


def _rotate_half(x, *, style="half"):
    if style == "interleaved":
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_rope(rotated, cos, sin, scale, *, style="half"):
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    if style == "interleaved":
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - _rotate_half(base, style=style) * sin_unit


def _to_complex_pairs(tensor, *, style="half"):
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[-1] // 2
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


def calibrate_stats(model_path: str, device: str, max_length: int = 512) -> Path:
    """Run lightweight calibration and return stats path."""
    from transformers import AutoConfig

    stats_path = Path("tests/outputs/llama31_8b_stats.pt")
    if stats_path.exists():
        print(f"[calibrate] Reusing existing stats: {stats_path}")
        return stats_path

    print(f"[calibrate] Loading model from {model_path} ...")
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    rope_style = _determine_rope_style(config)

    # Locate rotary
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "rotary_emb"):
        rotary = backbone.rotary_emb
    else:
        rotary = backbone.layers[0].self_attn.rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Short calibration text
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models are trained on large datasets. "
        "Attention mechanisms allow transformers to focus on relevant parts of the input. "
        "Rotary position embeddings encode positional information through rotation matrices. "
        "KV cache compression reduces memory usage during autoregressive generation."
    )
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    print(f"[calibrate] Tokenized length: {seq_len}")

    # Precompute cos/sin
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)

    # Hook to capture Q
    captured_q: dict[int, torch.Tensor] = {}
    attn_layers = [layer.self_attn for layer in backbone.layers]

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            bsz, q_len, _ = hidden_states.shape
            q = module.q_proj(hidden_states)
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            pos_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            p = torch.zeros(1, q_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            cos, sin = rotary(p, pos_ids)
            q_rot = (q * cos.unsqueeze(1)) + (_rotate_half(q, style=rope_style) * sin.unsqueeze(1))
            q_rot = q_rot * attn_scale
            captured_q[layer_idx] = q_rot.detach()
        return hook_fn

    handles = []
    for layer_idx, attn in enumerate(attn_layers):
        h = attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
        handles.append(h)

    print("[calibrate] Running forward pass...")
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    # Compute stats
    print("[calibrate] Computing frequency statistics...")
    sampled_heads = []
    stats_dict: dict[str, dict[str, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        q_rot = captured_q.get(layer_idx)
        if q_rot is None:
            print(f"  [warn] No Q for layer {layer_idx}, skipping.")
            continue

        cos = cos_table[:, :seq_len, :].unsqueeze(1)
        sin = sin_table[:, :seq_len, :].unsqueeze(1)
        q_base = _invert_rope(q_rot, cos, sin, attn_scale, style=rope_style)

        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]
            q_complex = _to_complex_pairs(q_head, style=rope_style)
            q_mean_complex = q_complex.mean(dim=0)
            q_abs_mean = q_complex.abs().mean(dim=0)

            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            stats_dict[key] = {
                "q_mean_real": q_mean_complex.real.cpu(),
                "q_mean_imag": q_mean_complex.imag.cpu(),
                "q_abs_mean": q_abs_mean.cpu(),
            }
            sampled_heads.append((layer_idx, head_idx))

        del captured_q[layer_idx]

    rope_scaling = getattr(config, "rope_scaling", {}) or {}
    rope_type = (
        rope_scaling.get("rope_type")
        or rope_scaling.get("type")
        or getattr(config, "rope_type", "default")
        or "default"
    )

    metadata = {
        "num_traces": 1,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": "eager",
        "rope_style": rope_style,
        "rope_type": rope_type,
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "stats": stats_dict}, stats_path)
    print(f"[calibrate] Saved stats ({len(sampled_heads)} heads) -> {stats_path}")

    del model
    torch.cuda.empty_cache()
    return stats_path


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def test_triattention_with_llama31() -> None:
    """Load Llama-3.1-8B, patch TriAttention, run generation, log compression."""
    model_path = str(Path.home() / "models" / "Llama-3.1-8B-Instruct")
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 maps cuda:0 to physical GPU1

    # 1. Calibrate (or reuse)
    stats_path = calibrate_stats(model_path, device, max_length=512)

    # 2. Load model
    print(f"\n[test] Loading model for generation: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # 3. Patch TriAttention
    from triattention.methods.triattention import apply_triattention_patch

    # Small budget so compression triggers quickly
    kv_budget = 32
    divide_length = 8

    apply_triattention_patch(
        model,
        stats_path=stats_path,
        model_path=Path(model_path),
        kv_budget=kv_budget,
        divide_length=divide_length,
        score_aggregation="mean",
        pruning_seed=42,
        normalize_scores=False,
        count_prompt_tokens=False,
        allow_prefill_compression=False,
        disable_mlr=False,
        disable_trig=False,
    )

    # 4. Wrap compute_keep_indices to log compression events
    compressor = model._triattention_compressor
    orig_compute_keep_indices = compressor.compute_keep_indices

    call_count = 0

    def logged_compute_keep_indices(pkv_tuple, prefix_length=0):
        nonlocal call_count
        call_count += 1
        seq_len_before = pkv_tuple[0][0].shape[-2] if pkv_tuple else 0
        result = orig_compute_keep_indices(pkv_tuple, prefix_length=prefix_length)
        if result.dim() == 1:
            kept = result.numel()
            print(f"  [compress #{call_count}] seq_len={seq_len_before} -> keep={kept} tokens, "
                  f"budget={kv_budget}, prefix={prefix_length}")
            print(f"    keep_indices (first 10): {result[:10].tolist()}")
            print(f"    keep_indices (last 10):  {result[-10:].tolist()}")
        elif result.dim() == 2:
            print(f"  [compress #{call_count}] seq_len={seq_len_before} -> per-head keep shape={tuple(result.shape)}")
        elif result.dim() == 3:
            print(f"  [compress #{call_count}] seq_len={seq_len_before} -> per-layer-per-head keep shape={tuple(result.shape)}")
        return result

    compressor.compute_keep_indices = logged_compute_keep_indices  # type: ignore[method-assign]

    # 5. Run generation
    prompt = "Explain the concept of attention mechanism in transformers:"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(device)
    prompt_len = inputs.shape[1]
    print(f"\n[test] Prompt length: {prompt_len} tokens")
    print(f"[test] KV budget: {kv_budget}, divide_length: {divide_length}")
    print(f"[test] Generating 48 new tokens...\n")

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    print(f"\n[test] Generated text:\n{generated}")
    print(f"\n[test] Total compression calls: {call_count}")

    # Final cache size check
    # Note: after generation, past_key_values may not be easily accessible
    # because generate() returns only output_ids, not the final cache.
    # We inspect the compressor state instead.
    print(f"[test] Final cache_positions length: {len(compressor.cache_positions)}")
    print(f"[test] Final absolute_position: {compressor.absolute_position}")


if __name__ == "__main__":
    test_triattention_with_llama31()
