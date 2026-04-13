# TriAttention core code research (round 1)

Scope: static inspection of the HuggingFace/research code path in `/home/zijie/Code/triattention`, focused on calibration, stats loading, RoPE/frequency handling, scoring, pruning cadence, head aggregation, integration, and baseline hooks. This note intentionally avoids the vLLM runtime except where a code contrast is useful.

## Architecture map

### 1) Offline calibration path

- `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate`
  - Loads `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM`.
  - Finds attention layers with `/home/zijie/Code/triattention/scripts/calibrate.py::_find_attention_layers`.
  - Reuses the model's live rotary embedding (`backbone.rotary_emb` or `attn_layers[0].rotary_emb`).
  - Registers per-layer forward pre-hooks via `/home/zijie/Code/triattention/scripts/calibrate.py::_make_pre_hook`.
  - Recomputes query projections, reapplies RoPE locally, then inverts RoPE with `/home/zijie/Code/triattention/scripts/calibrate.py::_invert_rope`.
  - Converts pre-RoPE query vectors into complex frequency pairs with `/home/zijie/Code/triattention/scripts/calibrate.py::_to_complex_pairs`.
  - Writes per-head payload entries:
    - `q_mean_real`
    - `q_mean_imag`
    - `q_abs_mean`
  - Saves metadata including `head_dim`, `rope_style`, `rope_type`, `attn_implementation`, and `sampled_heads`.

### 2) Main HuggingFace TriAttention inference path

- `/home/zijie/Code/triattention/scripts/worker.py::main`
  - For `method == "triattention"`, it does **not** use `triattention/integration/monkeypatch.py`.
  - Instead it resolves the stats file and calls `/home/zijie/Code/triattention/triattention/methods/triattention.py::apply_triattention_patch`.

- `/home/zijie/Code/triattention/triattention/methods/triattention.py::TriAttentionConfig`
  - Carries runtime parameters for the research/HF path:
    - `stats_path`, `model_path`, `budget`
    - `offset_max_length`
    - `score_aggregation`
    - `normalize_scores`
    - `count_prompt_tokens`
    - `allow_prefill_compression`
    - `divide_length`, `use_slack_trigger`
    - `per_head_pruning`, `per_layer_perhead_pruning`
    - `layer_perhead_aggregation`
    - `disable_mlr`, `disable_trig`

- `/home/zijie/Code/triattention/triattention/methods/triattention.py::TriAttention`
  - Loads model config with `AutoConfig`.
  - Loads stats via `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::load_head_frequency_stats`.
  - Validates metadata via `/home/zijie/Code/triattention/triattention/common/stats_utils.py::validate_stats_metadata`.
  - Builds rotary utilities with `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_rotary`.
  - Precomputes:
    - `self.omega` from `rotary.inv_freq`
    - `self.offsets` from `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_geometric_offsets`
    - `self.freq_scale_sq` from `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::compute_frequency_scaling`
  - Tracks cache positions in three modes:
    - shared `cache_positions`
    - per-KV-head `cache_positions_per_head`
    - per-(layer, KV head) `cache_positions_per_layer_perhead`

- `/home/zijie/Code/triattention/triattention/methods/triattention.py::apply_triattention_patch`
  - Builds a `TriAttention` compressor.
  - Runs `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::verify_rotary_alignment` against the model's rotary embedding when available.
  - Stores the compressor as `model._triattention_compressor`.
  - Monkeypatches `model.forward` with the nested `triattention_forward` function.

### 3) Shared scoring and RoPE helpers actually used by TriAttention

- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::invert_rope`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::to_complex_pairs`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::compute_frequency_statistics_from_means`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::score_keys_for_round`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_geometric_offsets`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_rotary`
- `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::determine_rope_style`

### 4) Baseline integration path, separate from TriAttention

- `/home/zijie/Code/triattention/triattention/integration/monkeypatch.py::{replace_llama, replace_qwen2, replace_qwen3}`
  - Replaces HF attention classes and `CausalLM.forward` globally.

- `/home/zijie/Code/triattention/triattention/integration/modeling.py`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::LlamaAttention_init`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::Qwen2Attention_init`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::Qwen3Attention_init`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::LlamaAttention_forward`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::Qwen2Attention_forward`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::Qwen3Attention_forward`
  - `/home/zijie/Code/triattention/triattention/integration/modeling.py::CausalLM_forward`
  - These instantiate baseline compressors from `KV_COMPRESSION_MAP`:
    - `/home/zijie/Code/triattention/triattention/methods/baselines/r1_kv.py::R1KV`
    - `/home/zijie/Code/triattention/triattention/methods/baselines/snapkv.py::SnapKV`

### 5) "Common" utilities requested in the task

- `/home/zijie/Code/triattention/triattention/common/stats_utils.py::validate_stats_metadata`
  - Used by `TriAttention.__init__`.
- `/home/zijie/Code/triattention/triattention/common/rope_utils.py`
  - Contains duplicate rotary helpers (`determine_rope_style`, `build_rotary`, `compute_frequency_scaling`) but is **not** what `triattention/methods/triattention.py` imports; the active path imports the same-named helpers from `triattention/methods/pruning_utils.py`.

## Execution flow

### A) Calibration flow

1. `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate` loads model + tokenizer.
2. It discovers attention modules with `_find_attention_layers`.
3. It reads the model's rotary embedding and attention scaling factor.
4. It tokenizes a plain-text calibration file.
5. It precomputes `cos_table` and `sin_table` for the whole calibration sequence.
6. For each attention layer, `_make_pre_hook`:
   - grabs `hidden_states`
   - applies `attn.q_proj`
   - reshapes to `[batch, num_heads, seq_len, head_dim]`
   - reapplies RoPE locally
   - stores rotated queries in `captured_q[layer_idx]`
7. After a single forward pass, calibration removes the hooks.
8. For each `(layer_idx, head_idx)`:
   - invert RoPE with `_invert_rope`
   - convert to complex frequency pairs with `_to_complex_pairs`
   - compute:
     - `q_mean_complex = q_complex.mean(dim=0)`
     - `q_abs_mean = q_complex.abs().mean(dim=0)`
9. The saved `.pt` file contains metadata plus per-head query statistics.

### B) Inference/generation flow for TriAttention

1. `/home/zijie/Code/triattention/scripts/worker.py::main` loads the HF model.
2. If `--method triattention`, it calls `apply_triattention_patch(...)`.
3. `apply_triattention_patch` builds `TriAttentionConfig`, then `TriAttention`.
4. `TriAttention.__init__`:
   - loads the model config
   - loads stats from disk
   - filters `metadata["sampled_heads"]` to valid layers
   - builds local rotary tables and frequency metadata
   - computes `self.omega`, `self.offsets`, and `self.freq_scale_sq`
   - infers GQA structure (`num_attention_heads`, `num_key_value_heads`, `num_key_value_groups`)
5. The patched `triattention_forward` runs on every `model.forward` call during `model.generate(...)`.
6. On an empty cache, it resets compression state before computing positions.
7. On decode steps, it overrides:
   - `position_ids` with **absolute** positions derived from `comp.absolute_position`
   - `cache_position` with relative positions based on current cache length
8. It calls the original `model.forward`.
9. It converts `past_key_values` into a tuple form for manipulation.
10. It updates cache-position bookkeeping:
    - prefill: initializes positions as `range(seq_len)`
    - decode: appends new absolute positions
11. It computes `effective_size`:
    - total cache length if `count_prompt_tokens=True`
    - decode-only length if `count_prompt_tokens=False`
12. Compression trigger:
    - if `use_slack_trigger=True`: trigger once `effective_size >= budget + divide_length`
    - else: trigger when `effective_size >= budget` **and** `absolute_position % divide_length == 0`
13. If triggered, `TriAttention.compute_keep_indices(...)` scores all currently cached keys.
14. `compute_keep_indices(...)`:
    - preserves prefill by default unless `allow_prefill_compression=True`
    - builds decode token positions from `cache_positions`
    - in per-head/per-layer modes, builds per-head position arrays for correct future RoPE inversion after independent compression
15. For each layer, `_compute_layer_head_scores(...)`:
    - finds sampled attention heads belonging to the layer
    - maps each attention head to a KV head under GQA using `head // num_key_value_groups`
    - gathers cached keys for that KV head
    - reconstructs RoPE `cos/sin` tables, either shared or per KV head
    - inverts RoPE with `pruning_utils.invert_rope`
    - computes online statistics with `compute_frequency_statistics_from_means`
    - scores keys with `score_keys_for_round`
16. Token selection then depends on pruning mode:
    - global mode: `_select_union_based`
    - per-head mode: `_select_per_head_independent`
    - per-layer-per-head mode: `_select_per_layer_perhead_independent`
17. The cache tensors are rewritten:
    - global mode: `index_select`
    - per-head / per-layer-per-head: `gather`
18. Position bookkeeping is rewritten to match the pruned cache.
19. A new `CausalLMOutputWithPast` is returned with the compressed cache.

### C) Baseline execution flow (for comparison)

1. `/home/zijie/Code/triattention/scripts/worker.py::main` calls `replace_llama`, `replace_qwen2`, or `replace_qwen3` only for `r1kv` / `snapkv`.
2. Patched attention constructors instantiate `self.kv_cluster` from `KV_COMPRESSION_MAP`.
3. Patched attention forwards maintain `past_key_value.query_cache` for recent queries.
4. `/home/zijie/Code/triattention/triattention/integration/modeling.py::CausalLM_forward` toggles `layer.self_attn.config.compression` based on `divide_method` and `divide_length`.
5. Each baseline compressor runs `update_kv(...)` inside attention forward.

## Paper-to-code mapping

| Paper concept / method idea | Code realization | Notes from inspection |
|---|---|---|
| Offline calibration of pre-RoPE statistics | `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate` | The saved statistics are per-layer, per-head query means and absolute means in the frequency domain. |
| Load calibrated statistics at inference | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::load_head_frequency_stats`; `/home/zijie/Code/triattention/triattention/methods/triattention.py::TriAttention.__init__` | `TriAttention` loads metadata and a `HeadFrequencyStats` map keyed by `(layer, head)`. |
| RoPE inversion before frequency analysis | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::invert_rope`; `/home/zijie/Code/triattention/triattention/methods/triattention.py::_compute_layer_head_scores` | Keys are unrotated online before scoring. Calibration likewise unrotates queries offline. |
| Complex frequency-domain representation | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::to_complex_pairs` | Head dimensions are split into complex pairs. |
| Query-center / norm terms | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::compute_frequency_statistics_from_means` | Computes `amp`, `phi`, and `extra` from saved query means plus current unrotated keys. |
| Trigonometric score over future distances | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::score_keys_for_round` | Uses `cos(delta * omega + phi)` with future offsets and an additive non-trig term. |
| Future-offset averaging / maxing | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_geometric_offsets`; `score_keys_for_round(... aggregation=...)` | Offsets are geometric powers of two up to `offset_max_length`; aggregation is `mean` or `max`. |
| Prune every N decoding steps | `triattention_forward` in `/home/zijie/Code/triattention/triattention/methods/triattention.py` | Two modes exist: strict modulo cadence or slack trigger at `budget + divide_length`. |
| GQA-aware head mapping | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_compute_layer_head_scores` | Attention heads are mapped to KV heads with `head // num_key_value_groups`. |
| Head aggregation for token selection | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_select_union_based`; `_select_per_head_independent`; `_select_per_layer_perhead_independent` | The code supports three aggregation/selection regimes, with per-head pruning effectively the default experiment path. |
| Research-path integration into HF generation | `/home/zijie/Code/triattention/scripts/worker.py::main`; `/home/zijie/Code/triattention/triattention/methods/triattention.py::apply_triattention_patch` | TriAttention uses a patched `model.forward`, not the baseline monkeypatch path. |
| Baseline hooks for R-KV / SnapKV comparisons | `/home/zijie/Code/triattention/triattention/integration/monkeypatch.py`; `/home/zijie/Code/triattention/triattention/integration/modeling.py`; baseline classes under `/home/zijie/Code/triattention/triattention/methods/baselines/` | Baselines compress inside attention forward with query caches and explicit step triggers. |

## Evidence table

| Aspect | Evidence | What the code shows |
|---|---|---|
| Calibration payload contents | `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate` | The saved stats contain only `q_mean_real`, `q_mean_imag`, and `q_abs_mean` per head. |
| Stats file loading | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::load_head_frequency_stats` | The inference path reconstructs `HeadFrequencyStats(q_mean_complex, q_abs_mean)` from the `.pt` file. |
| Stats metadata check | `/home/zijie/Code/triattention/triattention/common/stats_utils.py::validate_stats_metadata` | Only `rope_style` and `head_dim` are actually enforced. |
| Rotary alignment safeguard | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::verify_rotary_alignment`; `/home/zijie/Code/triattention/triattention/methods/triattention.py::apply_triattention_patch` | The runtime tries to ensure its locally built rotary embedding matches the model's live one. |
| RoPE inversion on cached keys | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_compute_layer_head_scores`; `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::invert_rope` | Keys are gathered from KV cache, unrotated, then scored. |
| Frequency score structure | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::compute_frequency_statistics_from_means`; `score_keys_for_round` | The score is `base_scores + additive`, where `base_scores` is trigonometric and `additive` is norm-based. |
| Future offsets | `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::build_geometric_offsets` | Offsets are `1, 2, 4, ...` up to `offset_max_length`. |
| Compression cadence | nested `triattention_forward` in `/home/zijie/Code/triattention/triattention/methods/triattention.py` | Compression is gated by decode-step detection plus either modulo cadence or slack thresholding. |
| Global token selection | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_select_union_based` | Each head proposes top-k, the union is formed, and final picks come from that union using combined scores. |
| Per-head aggregation | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_select_per_head_independent` | For each KV head, sampled attention heads are grouped by `(layer, kv_head)`; the code takes per-layer max, then mean across layers. |
| Per-layer-per-head mode | `/home/zijie/Code/triattention/triattention/methods/triattention.py::_select_per_layer_perhead_independent` | Each `(layer, KV head)` can prune independently, with `max` or `mean` aggregation within that layer's sampled heads. |
| Experiment defaults favor per-head mode | `/home/zijie/Code/triattention/scripts/worker.py::parse_arguments`; `/home/zijie/Code/triattention/scripts/cli.py::build_config` | The official HF/experiment path defaults `per_head_pruning=True`, `triattention_normalize_scores=True`, `count_prompt_tokens=True`, and `slack_budget_trigger=True`. |
| Baseline path is separate | `/home/zijie/Code/triattention/scripts/worker.py::main`; `/home/zijie/Code/triattention/triattention/integration/monkeypatch.py` | `replace_*` is only used for `r1kv` / `snapkv`; TriAttention uses `apply_triattention_patch` instead. |
| Baseline prefill hook | `/home/zijie/Code/triattention/triattention/integration/modeling.py::{LlamaAttention_forward,Qwen2Attention_forward,Qwen3Attention_forward}`; `/home/zijie/Code/triattention/triattention/methods/baselines/r1_kv.py::attach_prefill_length` | The baseline path explicitly stores prefill length for the `protect_prefill` ablation in `R1KV`. |

## Ambiguities / mismatches

1. **"Attention-layer compression" is not literally how the TriAttention HF path is implemented.**
   - `/home/zijie/Code/triattention/triattention/methods/triattention.py` says it "triggers compression inside the attention forward pass" and uses "attention-layer compression".
   - But `/home/zijie/Code/triattention/triattention/methods/triattention.py::apply_triattention_patch` actually patches `model.forward`, runs the original forward, then prunes `outputs.past_key_values` afterward.
   - By contrast, the baseline path in `/home/zijie/Code/triattention/triattention/integration/modeling.py` really does patch attention modules directly.

2. **The active TriAttention research path does not use `window_size` as a hard recency window.**
   - `scripts/worker.py` parses `--window_size` and `--round_window` for TriAttention.
   - `scripts/cli.py::build_config` also sets `window_size=128` and `round_window=32` by default for TriAttention configs.
   - But `/home/zijie/Code/triattention/triattention/methods/triattention.py::TriAttentionConfig` has no `window_size` or `round_window` field, and `compute_keep_indices` does not pin recent tokens. Only prefill can be pinned.
   - So the HF TriAttention code path is different from R-KV/SnapKV, which explicitly preserve a recent window.

3. **Some experiment-facing TriAttention flags are parsed or emitted but not consumed by the core compressor.**
   - `scripts/worker.py` parses `--attention_layer_compression`, but the TriAttention branch always calls `apply_triattention_patch(...)` and never branches on that flag.
   - `--round_window` is included in the TriAttention method config object in `scripts/worker.py`, but that method config is not what drives `apply_triattention_patch(...)`.

4. **Calibration stores query statistics only, even though docs/README language suggests broader Q/K stats.**
   - `/home/zijie/Code/triattention/docs/calibration.md` says TriAttention uses "Q/K centers and norms".
   - The actual `.pt` payload written by `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate` stores only query-side frequency summaries: `q_mean_*` and `q_abs_mean`.
   - At inference, current keys are processed online in `_compute_layer_head_scores` and combined with saved query means.

5. **Qwen3 calibration looks potentially inconsistent with the live model forward path.**
   - `/home/zijie/Code/triattention/triattention/integration/modeling.py::Qwen3Attention_forward` applies `self.q_norm(...)` and `self.k_norm(...)` before RoPE.
   - `/home/zijie/Code/triattention/scripts/calibrate.py::_make_pre_hook` manually computes `q = attn.q_proj(hidden_states)` and reapplies RoPE, but it does **not** apply `q_norm`.
   - If the live HF Qwen3 module also uses `q_norm`, calibration stats for Qwen3 may not match the actual runtime queries.

6. **Metadata validation is weaker than the saved metadata suggests.**
   - `scripts/calibrate.py` saves `rope_type`, `dtype`, `attn_implementation`, `use_chat_template`, and `system_prompt`.
   - `TriAttention.__init__` builds expectations including `rope_type`.
   - But `/home/zijie/Code/triattention/triattention/common/stats_utils.py::validate_stats_metadata` only checks `rope_style` and `head_dim`.
   - So `rope_type` is currently carried around but not actually enforced.

7. **There is duplicated rotary/scoring utility logic in multiple places.**
   - The live TriAttention path imports from `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py`.
   - `/home/zijie/Code/triattention/triattention/common/rope_utils.py` duplicates `determine_rope_style`, `build_rotary`, and `compute_frequency_scaling`.
   - `/home/zijie/Code/triattention/scripts/calibrate.py` also embeds local copies of RoPE helper logic.
   - This increases the chance of drift between calibration and inference.

8. **Library defaults differ from experiment defaults.**
   - `TriAttentionConfig` defaults: `count_prompt_tokens=False`, `use_slack_trigger=False`, `per_head_pruning=False`.
   - But `/home/zijie/Code/triattention/scripts/worker.py::parse_arguments` and `/home/zijie/Code/triattention/scripts/cli.py::build_config` default the experiment path to `count_prompt_tokens=True`, `slack_budget_trigger=True`, and `per_head_pruning=True`.
   - Reproducing paper numbers likely depends on the worker/CLI defaults, not just the raw class defaults.

9. **The code contains signs of an updated head-aggregation rule.**
   - The docstring of `/home/zijie/Code/triattention/triattention/methods/triattention.py::_select_per_head_independent` explicitly says it changed from "max(all 196 heads)" to "mean(max(7 heads per layer))" as a bug fix.
   - That implies the current official code already encodes a post-hoc correction or refinement relative to an earlier implementation.

10. **Head sampling infrastructure exists, but the official calibration script currently captures all heads.**
    - `/home/zijie/Code/triattention/triattention/methods/pruning_utils.py::load_or_create_sample` suggests the codebase once supported or planned sampled-head calibration.
    - `/home/zijie/Code/triattention/scripts/calibrate.py::calibrate` writes every `(layer, head)` into `sampled_heads`.
    - So the current HF path is effectively full-head calibration, even though the internal abstractions allow a sampled subset.

## Bottom line

The HuggingFace/research implementation is centered on a patched `model.forward` that loads per-head **query** frequency statistics, unrotates cached keys online, computes a trig-plus-norm importance score over geometric future offsets, and prunes either globally, per KV head, or per (layer, KV head). The most important reproducibility caveats are that the experiment defaults differ from the raw `TriAttentionConfig` defaults, the HF TriAttention path does not appear to use a hard recent-token window, and calibration/inference helper logic is duplicated enough that drift is plausible.
