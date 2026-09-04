# DeepSeek-V4-Flash-Vision-Exp — measurements

Hardware: AMD Strix Halo, 128 GB unified memory, ROCm/HIP.
Model: `deepseek4` arch, UD-IQ3_XXS, ~97 GB weights + F16 mmproj.
Served window: **131072**. Every number below states its own window — a number
taken at a different window is a different measurement, not a comparison.

## 1. Speculative decoding: +54.4%

The lane ran with **no drafter at all** from the start, while the neighbouring
text lane had one since 2026-07-17. Paired duel, 2 alternating legs, 11 prompts,
ctx 131072, one variable:

| arm | leg 1 | leg 2 |
|---|---|---|
| no drafter | 15.57 t/s | 15.57 t/s |
| `draft-dspark` | **24.05 t/s** | **24.04 t/s** |

**+54.4%.** Acceptance 0.735 (175/238), mean accepted length 3.19.
Eviction delta 0 on all four legs; GTT 110.0 GB; worst MemAvailable 9.6 GB.

Drafter: [`YanissAmz/DeepSeek-V4-Flash-DSpark-draft-GGUF`](https://huggingface.co/YanissAmz).
Same vocab (129280) as the vision model, 3 blocks, 81 tensors — compatibility was
checked on the metadata *before* the duel, not hoped for.

`draft-dflash` GGUFs refuse to load on this binary:
`key not found in model: dflash.attention.sliding_window_pattern`.

### Both drafter dials, swept

Paired duels, 3 arms x 2 legs, 11 prompts, eviction delta 0 throughout.

`n_max`: 3 → **23.64** | 4 → 22.00 | 5 → 20.82 t/s.
The drafter's `block_size=5` caps `n_max` at 5, and everything above 3 degrades:
the draft doubles without lengthening what gets accepted. The default was
already the optimum.

`p_min`: 0.00 → 23.64 | **0.20 → 24.50** | 0.30 → 24.36 | 0.40 → 22.91 |
0.60 → 23.18 t/s. Interior optimum at 0.20 (+3.6%), which is what ships.

> The `draft acceptance` line in a leg's log is the **last prompt's**, not the
> leg's aggregate. Judge on mean throughput, not on that line.

## 2. Prefill depth-loss cut by a factor of three

This lane was new; the neighbouring **text** lane carried months of measured
tuning it had never been given. Porting that block over, unchanged:

| | 4096 | 32768 | depth loss |
|---|---|---|---|
| before | 209.8 t/s | 152.5 t/s | -27.3% |
| **after** | **262.7 t/s** | **239.5 t/s** | **-8.8%** |

No new kernel work for this result. The patch exists (`patches/`, see
[BUILD.md](../BUILD.md)) and had been running in production on the text-only
lane for weeks; the vision lane had simply never been given the flags. The whole
win was reading a launcher I had already written and copying two lines.

If you are reproducing this on a stock llama.cpp build, read
[BUILD.md](../BUILD.md) first: `DSV4_FA_ROW_GATHER=1` on an unpatched binary is
an environment variable nothing reads, and you will measure the `before` row
twice.

**The transferable lesson: diff the neighbouring lane before you go looking for
a new flag.**

## 3. Decode is flat with depth

28.66 t/s @4k → 29.31 t/s @32k. Unlike GLM, which loses 36% over the same kind
of span, this lane does not degrade with context.

## 4. What is NOT measured

No quality score. This lane has never been put in front of a judge. Throughput
numbers here say nothing about output quality.
