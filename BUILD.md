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

### 3. GLM-5.3-Flash architecture (`glm5next`)

Upstream llama.cpp has no `glm5next` architecture at all — the model will not
load. The port is 26 commits (hparams and tensor loading, KDA linear attention,
mHC wide residual, MoE feed-forward with clamped SwiGLU, dense DSA attention) and
is too large to ship as patch files here; it lives on a branch of my llama.cpp
fork. It is not upstream and I make no claim that it will be.

The GLM lane additionally carries a fix for the MTP drafter skipping image
batches (see `glm-5.3-flash/RESULTS.md` §4): mark the sequence when a batch
carries an image, clear the drafter's KV for it on the next text batch. Without
it, `--spec-type draft-mtp` plus `--mmproj` returns HTTP 500 on the first image.

### 4. `MMVQ_MAX_BATCH_SIZE` = 4 (Qwen3.8-27B dense)

Upstream ships this macro at 8. The dense lane runs it at 4, and `MMVF_MAX_BATCH_SIZE`
with it — a `static_assert` ties them together:

```diff
--- a/ggml/src/ggml-cuda/mmvq.cuh
+++ b/ggml/src/ggml-cuda/mmvq.cuh
-#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
+#define MMVQ_MAX_BATCH_SIZE 4 // Max. batch size for which to use MMVQ kernels.
--- a/ggml/src/ggml-cuda/mmvf.cuh
+++ b/ggml/src/ggml-cuda/mmvf.cuh
-#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.
+#define MMVF_MAX_BATCH_SIZE 4 // Max. batch size for which to use MMVF kernels.
```

This is the single biggest win in that lane: the drafter's per-position cost drops
from 24.3 ms to 2.25 ms, and best decode moves from 16.8 to 19.2 t/s. Every number
in `qwen3.8-27b-dense/RESULTS.md` was taken at 4, not at 8 — build it at 8 and you
will reproduce something slower.

Note it is a **compilation** parameter, not a dispatch switch: the macro also feeds
`__launch_bounds__`, so you cannot answer this by setting it to 0. To sweep the
threshold, patch the dispatch function and read it from an environment variable, so
the whole curve runs on one binary.

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
