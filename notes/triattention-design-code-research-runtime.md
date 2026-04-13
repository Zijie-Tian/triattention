# TriAttention runtime/deployment surface notes

Reference URLs used:
- Paper: https://arxiv.org/abs/2604.04921
- Project page: https://weianmao.github.io/tri-attention-project-page/
- Repo README: https://github.com/WeianMao/triattention/blob/main/README.md
- OpenClaw doc: https://github.com/WeianMao/triattention/blob/main/docs/openclaw.md
- MLX doc: https://github.com/WeianMao/triattention/blob/main/docs/mlx.md

## runtime architecture

1. **Packaging and activation path**
   - `setup.py` registers a `vllm.general_plugins` entry point: `triattention = triattention.vllm.plugin:register_triattention_backend`.
   - The plugin is intended to auto-activate when vLLM loads general plugins after `pip install -e .`.
   - `triattention/vllm/plugin.py` gates activation with `ENABLE_TRIATTENTION`, supports `TRIATTENTION_INTERFACE`, bridges older `TRIATTENTION_*` env vars into the newer `TRIATTN_RUNTIME_*` namespace, and defaults several strict runtime flags on.
   - The plugin explicitly says legacy V1 custom-backend registration is retired; the supported path is runtime monkeypatching.

2. **Actual hook points are vLLM internal monkeypatches**
   - `triattention/vllm/runtime/integration_monkeypatch.py` patches vLLM internals rather than using a narrow public extension surface.
   - Patched targets include:
     - `vllm.v1.core.sched.scheduler.Scheduler.__init__`, `schedule`, `update_from_output`
     - `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots`
     - `vllm.v1.engine.core.EngineCore.step_with_batch_queue`
     - `vllm.v1.worker.gpu_worker.Worker.init_device`, `execute_model`
     - KV cache memory-check helpers in `vllm.v1.core.kv_cache_utils`
   - Important nuance: plugin comments call this the runtime “V2” path, but the code imports and patches `vllm.v1.*` modules directly. Compatibility therefore depends on vLLM’s current private/internal layout, not just stable public APIs.

3. **Scheduler-side logic**
   - `triattention/vllm/runtime/config.py` defines `TriAttentionRuntimeConfig`, loaded from environment variables.
   - `triattention/vllm/runtime/scheduler.py` adds request-level state tracking:
     - prefill length tracking
     - effective cache length tracking after compression
     - compression planning via `CompressionPlanner`
   - Scheduler emits `triattention_signals` per request, attached onto `scheduler_output` as side-channel metadata.
   - Triggering can come from:
     - **length threshold**: effectively `kv_budget + divide_length` (plus prefill offset in some modes)
     - **KV-usage pressure**: optional hysteresis trigger using `kv_usage_trigger` / `kv_usage_release`

4. **Worker/model-runner path**
   - `triattention/vllm/runtime/worker.py` keeps the native vLLM worker initially untouched and lazily injects a `TriAttentionModelRunner` proxy only when scheduler signals indicate TriAttention behavior is needed.
   - `triattention/vllm/runtime/runner.py` then:
     - consumes scheduler signals
     - can self-trigger compression using worker-side block-table truth if the scheduler lags
     - executes compression actions through a runner hook
     - synchronizes reclaimed blocks back into worker block-table state
     - installs runtime input patches when “effective length” overrides are active
     - attaches compression events back to model outputs for the scheduler to consume

5. **Compaction/selection implementation**
   - `triattention/vllm/runtime/hook_impl.py` installs `triattention_apply_compression` on the base runner.
   - `triattention/vllm/runtime/selector_hf.py` builds a selector around `triattention.vllm.core` components (`TriAttentionConfig`, `TriAttentionCompressor`, Triton scoring code), with a hard dependency on a stats file (`TRIATTN_RUNTIME_SPARSE_STATS_PATH`) and optional model path.
   - The selector supports shared/per-head selection modes, streaming scoring over paged KV blocks, and heavy reliance on Triton scoring.
   - Compression appears designed for paged KV cache layouts and GPU worker execution, not generic CPU inference.

6. **Block reclaim and allocation sync**
   - `triattention/vllm/runtime/worker_reclaim_sync.py` and scheduler-side reclaim code update block tables and per-request bookkeeping after compression.
   - `triattention/vllm/runtime/kv_allocation_sync.py` maintains an “effective vs logical” computed-token offset so vLLM allocation math stays consistent after physical compaction.
   - This is a strong sign that the runtime path is doing deep integration with vLLM’s scheduler/allocator invariants, not just pruning tensors in isolation.

7. **Separate MLX runtime**
   - `triattention/mlx/triattention_mlx.py` is a distinct Apple Silicon path using `mlx` / `mlx_lm`, not the vLLM plugin.
   - It provides a native MLX compressor and helper generation step, described in `docs/mlx.md` as experimental.

## configuration surface

### Publicly documented runtime/deployment controls

From `README.md` and deployment docs:
- `TRIATTN_RUNTIME_KV_BUDGET`
- `TRIATTN_RUNTIME_DIVIDE_LENGTH`
- `TRIATTN_RUNTIME_WINDOW_SIZE`
- `TRIATTN_RUNTIME_PRUNING_MODE`
- `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
- `TRIATTN_RUNTIME_PROTECT_PREFILL`
- `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION`
- `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM`
- `ENABLE_TRIATTENTION`

Operational launch assumptions documented in README/OpenClaw guidance:
- `vllm serve <model_path>` with `--dtype bfloat16`
- `--max-model-len 32768`
- `--enforce-eager`
- `--trust-remote-code`
- `--enable-prefix-caching false`
- for chat/OpenClaw: `--max-num-batched-tokens 1024`
- OpenClaw points to the vLLM OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1`

### Additional code-level runtime controls not fully surfaced in README

`triattention/vllm/runtime/config.py` exposes a broader config surface, including:
- `TRIATTN_RUNTIME_DISABLE_COMPRESSION`
- `TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER`
- `TRIATTN_RUNTIME_KV_USAGE_TRIGGER`
- `TRIATTN_RUNTIME_KV_USAGE_RELEASE`
- `TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING`
- `TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM`
- `TRIATTN_RUNTIME_LOG_DECISIONS`
- `TRIATTN_RUNTIME_FAIL_ON_EFFECTIVE_LEN_REGRESSION`
- `TRIATTN_RUNTIME_EFFECTIVE_LEN_REGRESSION_RATIO`
- `TRIATTN_RUNTIME_EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES`
- `TRIATTN_RUNTIME_SCORE_CHUNK_MAX_TOKENS`
- `TRIATTN_RUNTIME_MODEL_PATH`
- `TRIATTN_RUNTIME_SPARSE_SCORE_AGGREGATION`
- `TRIATTN_RUNTIME_SPARSE_NORMALIZE_SCORES`
- `TRIATTN_RUNTIME_INCLUDE_PREFILL_IN_BUDGET`
- `TRIATTN_RUNTIME_PER_HEAD_SELECTION_SEMANTICS`
- `TRIATTN_RUNTIME_LAYER_PERHEAD_AGGREGATION`
- `TRIATTN_RUNTIME_PER_LAYER_AGGREGATION`
- `TRIATTN_RUNTIME_ALLOW_PER_LAYER_MODE`
- `TRIATTN_RUNTIME_DISABLE_MLR`
- `TRIATTN_RUNTIME_DISABLE_TRIG`
- `TRIATTN_RUNTIME_DISABLE_TOP_N_HIGH_FREQ`
- debug log options: `TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG*`

### Plugin-only activation/compatibility controls

From `triattention/vllm/plugin.py`:
- `TRIATTN_RUNTIME_PATCH_SCHEDULER`
- `TRIATTN_RUNTIME_PATCH_WORKER`
- `TRIATTENTION_INTERFACE`
- `TRIATTENTION_QUIET`

Legacy env bridging in the plugin:
- `TRIATTENTION_KV_BUDGET` -> `TRIATTN_RUNTIME_KV_BUDGET`
- `TRIATTENTION_DIVIDE_LENGTH` -> `TRIATTN_RUNTIME_DIVIDE_LENGTH`
- `TRIATTENTION_WINDOW_SIZE` -> `TRIATTN_RUNTIME_WINDOW_SIZE`
- `TRIATTENTION_LOG_DECISIONS` -> `TRIATTN_RUNTIME_LOG_DECISIONS`
- `TRIATTENTION_STATS_PATH` -> `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
- `TRIATTENTION_PRUNING_MODE` -> `TRIATTN_RUNTIME_PRUNING_MODE` (with normalization of `per_layer_head` -> `per_layer_per_head`)

### Stats/config artifacts used by runtime

- vLLM serving expects precomputed frequency stats under `triattention/vllm/stats/`.
- HF-style evaluation/calibration stats also exist under `triattention/calibration/...`.
- Binary inspection of `triattention/vllm/stats/qwen3_32b_int4_stats.pt` shows embedded metadata such as `kv_budget`, `attn_implementation`, `rope_style`, `rope_type`, and sampled heads; the runtime is clearly model/stats dependent.

## deployment claims and support

1. **OpenAI-compatible vLLM server / OpenClaw compatibility**
   - README claims TriAttention’s vLLM server exposes an OpenAI-compatible API and can be used as a custom OpenClaw provider.
   - Support in repo is real, not just aspirational:
     - plugin registration exists in `setup.py`
     - runtime patching code exists under `triattention/vllm/runtime/`
     - `docs/openclaw.md` gives concrete provider JSON and optional SSH-tunnel instructions
   - What is *not* shown in the reviewed files: container images, packaged deployment manifests, or higher-level orchestration.

2. **Transparent activation claim**
   - README says vLLM discovers/activates the plugin automatically after installation.
   - This is supported by the entry point registration.
   - However, actual functionality still depends on internal monkeypatch compatibility with the installed vLLM version and on required stats/env configuration.

3. **Compression-based memory support**
   - README claims deployment can fit long-context reasoning better; the runtime also relaxes vLLM’s KV memory check in code, explicitly warning that compression will keep actual usage within limits.
   - The code therefore assumes compression is sufficiently reliable to justify bypassing default vLLM capacity guards.

4. **Chat deployment assumptions are explicit**
   - Prefix caching is documented as incompatible and must be disabled.
   - Large prefill chunks are treated as dangerous; README recommends `--max-num-batched-tokens 1024` to avoid overshooting the KV budget before compression can trigger.
   - These are not generic vLLM defaults; they are part of the deployment contract for this runtime.

5. **MLX support claim**
   - README news item and `docs/mlx.md` describe experimental Apple Silicon support.
   - Code exists under `triattention/mlx/`, so this is more than a placeholder.
   - But it is clearly a separate runtime path, not the same deployment surface as the vLLM plugin.

6. **Support boundaries visible in repo/docs**
   - README roadmap still marks SGLang integration and Ollama integration as not yet done.
   - In the reviewed runtime/deployment files, vLLM is the production/server path and MLX is an experimental alternate path.

7. **Model-support/deployment-doc mismatch worth noting**
   - README’s benchmark support table emphasizes Qwen3-8B / DeepSeek distill models.
   - The OpenClaw chat deployment example uses `triattention/vllm/stats/qwen3_32b_int4_stats.pt` instead.
   - That suggests the deployment target matrix is broader or at least different from the benchmark table, but the exact supported-serving matrix is not clearly documented in the reviewed files.

## current-env executability notes

1. **What is executable from dependency standpoint right now**
   - Current environment successfully imports: `torch`, `transformers`, `datasets`, `accelerate`, `triton`, `flash_attn`.
   - CUDA is available in the current environment.
   - This means the non-vLLM PyTorch/HF portions of the repo are at least closer to runnable than the server/runtime surfaces.

2. **What is not executable right now**
   - `vllm` is **not installed** in the current environment.
   - Because `triattention/vllm/runtime/integration_monkeypatch.py` imports `vllm.*` at module import time, the vLLM plugin/runtime path is not currently executable here without an additional install.
   - `mlx` and `mlx_lm` are also **not installed**, so the Apple Silicon path is not executable in this environment either.

3. **Base install does not provision deployment extras**
   - `setup.py` base dependencies do **not** include `vllm`, `mlx`, or `mlx-lm`.
   - `docs/environment.md` lists vLLM and MLX as optional components.
   - So `pip install -e .` alone is not enough to make the deployment/runtime surfaces runnable.

4. **HF experiment path still has nontrivial external requirements**
   - `scripts/cli.py run-one` expects local models under `experiments/models/<model-name>` and dataset JSONL files/symlinks in specific locations.
   - TriAttention evaluation mode also requires stats paths to exist.
   - I did not attempt heavyweight installs, model downloads, or server launches, so this note is strictly based on structure/import requirements, not on full end-to-end execution.

5. **MLX path is structurally separate and platform-specific**
   - `triattention/mlx/triattention_mlx.py` imports `mlx.core`, `mlx.nn`, and `numpy` directly.
   - On a Linux/CUDA environment without MLX packages (and not Apple Silicon), that path should be treated as non-executable here.

## evidence table

| Surface | Evidence | Takeaway | Public URL |
|---|---|---|---|
| Plugin registration | `setup.py` entry point `vllm.general_plugins` -> `triattention.vllm.plugin:register_triattention_backend` | Auto-discovery is intended to happen through vLLM plugin loading | https://github.com/WeianMao/triattention/blob/main/README.md |
| Plugin behavior | `triattention/vllm/plugin.py` | Runtime path = env bridging + monkeypatch install; legacy custom backend retired | https://github.com/WeianMao/triattention/blob/main/README.md |
| Monkeypatch scope | `triattention/vllm/runtime/integration_monkeypatch.py` | Deep patching of `vllm.v1.*` scheduler/worker/engine internals; compatibility risk tied to vLLM internals | n/a |
| Scheduler trigger path | `triattention/vllm/runtime/scheduler.py`, `planner.py`, `signals.py` | Compression is planned per request via length or KV-usage signals | n/a |
| Worker/runner path | `triattention/vllm/runtime/worker.py`, `runner.py`, `executor.py` | Proxy runner is lazily injected and performs request-level compression + state sync | n/a |
| Runtime config surface | `triattention/vllm/runtime/config.py` | Code-level env surface is larger than README documents | n/a |
| Selector/compaction | `triattention/vllm/runtime/hook_impl.py`, `selector_hf.py`, `triattention/vllm/core/*` | Runtime depends on stats-driven Triton scoring and paged KV compaction | n/a |
| OpenClaw deployment doc | `docs/openclaw.md` | Concrete manual provider config and remote SSH-tunnel guidance exist | https://github.com/WeianMao/triattention/blob/main/docs/openclaw.md |
| MLX deployment doc/code | `docs/mlx.md`, `triattention/mlx/triattention_mlx.py` | Experimental MLX support is implemented as a separate path | https://github.com/WeianMao/triattention/blob/main/docs/mlx.md |
| Public vLLM deployment guidance | `README.md` | Requires disabling prefix caching and often limiting prefill chunk size; deployment assumptions are explicit | https://github.com/WeianMao/triattention/blob/main/README.md |
| Stats artifacts | `triattention/vllm/stats/qwen3_32b_int4_stats.pt`, `triattention/calibration/for_aime24_experiment/qwen3_8b.pt`, `.../for_aime25_experiment/qwen3_8b.pt` | Runtime is coupled to precomputed model-specific stats files | n/a |
| Optional dependency posture | `requirements.txt`, `docs/environment.md` | Base install covers PyTorch stack; vLLM and MLX are optional extras, not default deps | https://github.com/WeianMao/triattention/blob/main/README.md |
| Current environment check | Live import check in this session | `vllm`, `mlx`, `mlx_lm` missing; CUDA PyTorch stack present | n/a |

## unresolved gaps

1. **Exact vLLM version support is unclear**
   - The runtime imports and patches `vllm.v1.*` internals, but no precise vLLM version pin or compatibility matrix was identified in the reviewed files.

2. **“V2 runtime” naming vs `vllm.v1` internals**
   - The plugin comments call the supported path “V2”, but the concrete implementation targets `vllm.v1` module paths. The intended versioning terminology is ambiguous.

3. **Serving-model support matrix is underdocumented**
   - README benchmark support table and deployment examples do not line up cleanly (8B benchmark models vs `qwen3_32b_int4_stats.pt` serving example).
   - It is not clear from reviewed files which serving models are officially supported for the plugin path.

4. **No end-to-end runtime validation in this round**
   - I did not install `vllm`, download heavyweight models, or launch a server, so plugin auto-discovery, scheduler/worker patches, and OpenClaw interop were not empirically verified here.

5. **MLX runtime not verifiable in current host environment**
   - The repo contains MLX code/docs, but current environment lacks `mlx`/`mlx_lm` and is not an Apple Silicon setup.

6. **Deployment packaging beyond manual CLI launch is unclear**
   - In the files reviewed, deployment guidance is manual (`vllm serve`, OpenClaw JSON config, SSH tunnel). I did not verify whether Docker/K8s/system-level deployment artifacts exist elsewhere in the repository.
