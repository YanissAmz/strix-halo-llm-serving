# Serving 100B+ MoE models on AMD Strix Halo (128 GB unified memory)

Launcher configs, drafter settings and paired benchmarks for running large
mixture-of-experts models on a 128 GB Strix Halo box with llama.cpp — ROCm/HIP
and Vulkan, no discrete GPU.

The hardware is cheap for the memory it carries and there is very little
published data on what actually makes it fast. This is mine, measured, with the
method attached: [METHOD.md](METHOD.md).

## Headline results

| model | change | decode | window |
|---|---|---|---|
| [DeepSeek-V4-Flash-Vision-Exp](ds4v/) | added an external DSpark drafter | 15.57 → **24.04 t/s** (+54.4%) | 131072 |
| [DeepSeek-V4-Flash-Vision-Exp](ds4v/) | ported the neighbouring lane's prefill flags | prefill depth-loss -27.3% → **-8.8%** | 131072 |
| [GLM-5.3-Flash](glm-5.3-flash/) | DFlash2 drafter, `p_min` 0.60 | 13.89 → **17.14 t/s** (+23.4%) | measured at 98304, served at 65536 |
| [GLM-5.3-Flash](glm-5.3-flash/) | then the free MTP head, for the 3.1 GiB | 15.77 → **17.37 t/s** (+10.2%) | 65536 |
| [Qwen3.8-Flash-Next](qwen3.8-flash-next/) | shared MTP head as drafter | 27.0 → **36.0 t/s** (+33%) | 200704 |

All figures are paired duels, both legs reported, one variable at a time, taken
2026-09-03. Each directory carries the full sweep, not just the winner.

Two lanes need a patched llama.cpp build. **Read [BUILD.md](BUILD.md) before
running any launcher here** — one of the kernel levers is an environment
variable that a stock build ignores in silence.

## Contents

- [`ds4v/`](ds4v/) — DeepSeek-V4-Flash-Vision-Exp (deepseek4, UD-IQ3_XXS, ~97 GB)
- [`glm-5.3-flash/`](glm-5.3-flash/) — GLM-5.3-Flash
- [`qwen3.8-flash-next/`](qwen3.8-flash-next/) — Qwen3.8-Flash-Next
- [`qwen3.8-27b-dense/`](qwen3.8-27b-dense/) — Qwen3.8-27B dense
- [`BUILD.md`](BUILD.md) — which patches and PRs each lane needs, and how a
  missing one fails silently
- [`patches/`](patches/) — the flash-attention row-gather commits, as `git am`
  patches
- [`METHOD.md`](METHOD.md) — how the numbers were taken, and the traps that
  produce convincing wrong ones
- [`KNOWN_CRASH.md`](KNOWN_CRASH.md) — reproducible ROCm illegal-memory-access,
  isolated to one flag
- [`bench/`](bench/) — the paired-duel harness
- [`tools/sanitize.sh`](tools/sanitize.sh) — rewrites a private launcher into the
  form published here

## If you take one thing from this repo

**Diff the lane next door before you go looking for a new flag.** The single
biggest prefill win here — depth-loss cut by a factor of three — came from
copying settings that were already in production on another lane of the same
box. No new kernel work, no new flag, no discovery. Just reading what I had
already written down and had stopped looking at.

## Related

Reference GGUF builds on Hugging Face: [YanissAmz](https://huggingface.co/YanissAmz).
Upstream llama.cpp: model support for NVIDIA Nemotron-3-Puzzle
([ggml-org/llama.cpp#25444](https://github.com/ggml-org/llama.cpp/pull/25444)).

## License

MIT.
