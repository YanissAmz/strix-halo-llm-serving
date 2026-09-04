# GLM-5.3-Flash — what was measured

Model: GLM-5.3-Flash, arch `glm5next`, 320B total parameters, UD-Q2_K_XL (~101 GB).
Box: AMD Strix Halo, gfx1151, 128 GB unified memory, ROCm/HIP.
Method: [../METHOD.md](../METHOD.md). Launcher: [launch.sh](launch.sh).
Build: [../BUILD.md](../BUILD.md) — **this model does not run on upstream llama.cpp**.

Two boxes appear below. They are not the same machine and their numbers do not
compare across rows; only ratios inside one row mean anything.

## 1. The drafter flag that was doing nothing

DFlash2 was first judged a dead heat against no drafter at all. It wasn't. The
missing flag was `--spec-draft-p-min`, whose default is `0.00` — the draft is
never filtered, the drafter emits every token it can, and the target model pays
to verify tokens that were always going to be rejected. Acceptance sat at 36%.

Same model, same window, same box, one variable:

| arm | decode |
|---|---|
| no drafter | 13.68 t/s |
| MTP head (`draft-mtp`, n_max=1) | 16.45 t/s (+20%) |
| DFlash2, `p_min` left at its default | 13.45 t/s (**−1.7%**) |
| **DFlash2, `p_min` 0.60** | **18.64 t/s (+36%)** |

The threshold sweep, same protocol:

| `--spec-draft-p-min` | 0.50 | 0.55 | **0.60** | 0.65 | 0.75 | 0.90 |
|---|---|---|---|---|---|---|
| decode (t/s) | 13.53 | 14.29 | **18.64** | 18.07 | 17.62 | 16.61 |

A cliff on the left and a declining plateau on the right — not a slope. Anywhere
below 0.60 and the drafter is worse than useless. This is why "we tried DFlash2,
it did nothing" is a claim worth re-testing rather than believing.

## 2. Paired duel, 2026-09-03 — DFlash2 vs no drafter

Window 98304 (**not** the 65536 this lane serves; stated because the number
carries its instrument), both legs reported:

| arm | decode |
|---|---|
| no drafter | 13.89 t/s |
| **DFlash2, p_min 0.60** | **17.14 t/s** |

**+23.4%**, acceptance 74.7% (109/146 drafted tokens).

## 3. Then the free drafter won anyway

DFlash2 costs 3.10 GiB, measured, not estimated — 1.17 GiB of weights plus ~1.9
GiB of its own KV cache. Everything else equal, at ctx 102400 with vision:

| arm | GTT | available at idle | outcome |
|---|---|---|---|
| with DFlash2 | 116.5 GiB | 3.12 GiB | dies at ~24k of context |
| without | 113.4 GiB | 6.33 GiB | passes 40k, stays alive |

That 3.10 GiB was exactly the margin the box did not have. So the lane moved to
the MTP head embedded in the GGUF, which costs nothing at all. Two rounds, zero
evictions:

| arm | decode |
|---|---|
| DFlash2 | 15.77 t/s |
| **MTP head, n_max=1** | **17.37 t/s (+10.2%)** |

A drafter that is 8% faster in a benchmark and kills the lane at 24k of real
context is not the faster drafter. The window has to be full when you judge.

## 4. The MTP head broke vision, silently

Swapping to MTP produced HTTP 500 on the first image, on both boxes. The drafter
was skipping the image batch and desynchronising its KV against the target
(target at position 7979, drafter at 68).

Fixed at the source rather than by dropping the drafter: mark the sequence when
a batch carries an image, and clear the drafter's KV for it on the next text
batch. Verified on both boxes with an image and a correct description of it.

Same protocol, after the fix:

| arm | decode |
|---|---|
| DFlash2 | 15.20 t/s |
| **MTP head** | **16.96 t/s** (box A) · **16.91 t/s** (box B) |

**A text-only duel does not validate a vision lane.** Both drafters passed every
text benchmark while returning 500 on the first image.

## 5. KV cache in q8_0 — faster *and* more accurate

| window | f16 | q8_0 | |
|---|---|---|---|
| 131072 | 2.65 t/s | **3.02 t/s** | +14% |
| 32768 | 7.64 t/s | 7.62 t/s | no change |
| perplexity @131k | 4.0970 | **4.0774** | −0.48% (better) |

Perplexity improving under quantization is the tell that llama.cpp's Hadamard
rotation for quantized KV is actually running on this model — a bare quantized
cache does not beat f16. It also halves the KV, which is what pays for the
window.

## 6. Decode really does decay with depth here — and it is unexplained

| context | 14.7k | 49.7k | 84.9k |
|---|---|---|---|
| decode | 14.53 t/s | 10.40 t/s | **9.23 t/s** |

−36% across that range. For comparison, the DeepSeek lane on the same box is
flat with depth and the Qwen lane steps down once and then plateaus. GLM is the
only one of the three that keeps falling. I do not know why yet; it is the best
open question in this repo.

Prefill decays too: 139.3 t/s at 14.7k → 88.3 at 49.7k → 63.8 at 84.9k (−54%).

⚠ False friend: "context is nearly free on GLM" is a statement about **memory**
(+56% of window for +0.67 GiB). It says nothing about throughput, and the table
above is the throughput.

## 7. Window: 65536, not 131072

| window | free at idle under agentic load | evictions in 3 min |
|---|---|---|
| 131072 | 0.52 GB | 131 |
| **65536** | **5.58 GB** | **0** |

Measured with the window actually full. A window judged empty is not a window.

## What is NOT measured

- No quality score for the drafter arms. Speculative decoding verifies every
  drafted token against the target model, so quality is bounded by construction
  — but bounded is not the same as measured.
- The −36% depth decay has no explanation attached, only a shape.
- Box A and box B numbers are never compared to each other, only within a row.
