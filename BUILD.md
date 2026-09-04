# Build

**Read this before running any launcher in this repo.** Two of the four lanes need
a llama.cpp build that is not plain `master`. Run them on a stock build and you
will not reproduce the numbers — and in one case a flag will be silently ignored
rather than rejected.

Reference: upstream `llama.cpp` at `4aa6ffba2` (2026-09-03). Hardware: AMD Strix
Halo (gfx1151), 128 GB unified memory, ROCm / HIP.

## What is already upstream

Nothing to do for these — they are in master and the launchers use them as-is:

| Feature | Flag | Status |
|---|---|---|
| DFlash2 drafter | `--spec-type draft-dflash` | merged (`#27342`, via `#27816`) |
| DSpark drafter | `--spec-type draft-dspark` | merged |
| MTP head drafter | `--spec-type draft-mtp` | merged |
| Draft-side confidence cutoff | `--spec-draft-p-min` | merged |
| Draft KV quantization | `-ctkd` / `-ctvd` | merged |

## What you have to apply yourself

### 1. Flash-attention row gather (DS4v prefill only)

`patches/000{1,2,3}-*.patch` — three commits against `ggml/src/ggml-cuda/`,
~500 lines. They add a prepass that walks only the KV rows some query of a tile
attends to, instead of every tile the sparse top-k mask left partly alive.

```sh
git am /path/to/strix-halo-llm-serving/patches/*.patch
```

Enable at runtime with `DSV4_FA_ROW_GATHER=1`.

> **The trap:** on a stock build `DSV4_FA_ROW_GATHER=1` is an unset environment
> variable that nothing reads. No error, no warning, no log line. The prefill
> numbers in `ds4v/RESULTS.md` (depth loss −27.3% → −8.8%) come entirely from
> this patch, so without it you measure the baseline and think you measured the
> result. Check with `DSV4_FA_PICK_DEBUG=1`, which reports which flash-attention
> instantiation actually launched.
>
> Note also that this variable is read **by presence** in some builds — `=0` can
> disable just as `=1` enables. Unset it; do not set it to zero.

Applies to DeepSeek-V4-Flash only. It is gated on a sparse attention mask, so on
a dense model the prepass never fires.

### 2. DeepSeek-V4-Flash **vision** (DS4v lane only)

Text-only DS4 needs nothing extra. Vision needs two PRs that were still open at
the time of writing:

- `ggml-org/llama.cpp#28133` — `mtmd: support DeepSeek-V4-Flash-Vision-Exp`
- `ggml-org/llama.cpp#28154` — `model: correctly support input vision for deepseek4`

Fetch and cherry-pick both, in that order.

## Build flags

```sh
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j"$(nproc)"
```

Then point each launcher at that binary explicitly:

```sh
export DS4V_SERVER=/path/to/build-hip/bin/llama-server
```

Every launcher in this repo **requires** its `*_SERVER` variable rather than
defaulting to whatever `llama-server` is first on `$PATH`. That is deliberate: a
`$PATH` binary is exactly how you end up benchmarking a build that does not
contain the patch you are trying to measure.
