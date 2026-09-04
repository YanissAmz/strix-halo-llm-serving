# DeepSeek-V4-Flash-Vision-Exp — measurements

Hardware: AMD Strix Halo, 128 GB unified memory, ROCm/HIP.
Model: `deepseek4` arch, UD-IQ3_XXS, ~97 GB weights + F16 mmproj.
Served window: **131072**. Every number below states its own window **and its
own depth** — those are different things, and a number taken at either a
different window or a different depth is a different measurement, not a
comparison.

## 1. Speculative decoding: +54.4%

The lane ran with **no drafter at all** from the start, while the neighbouring
text lane had one since 2026-07-17. Paired duel, 2 alternating legs, 11 prompts,
window 131072, one variable.

**Depth: ~275 tokens.** The window is the lane's served window; the prompts in
this duel are short. This is a short-context result and does not transfer to
depth — see §1b, where the same drafter loses at 176k on the sibling text
model.

| arm | leg 1 | leg 2 |
|---|---|---|
| no drafter | 15.57 t/s | 15.57 t/s |
| `draft-dspark` | **24.05 t/s** | **24.04 t/s** |

**+54.4%.** Acceptance **0.609** aggregated over all 24 prompt runs of both
legs (3626 accepted / 5952 generated), mean accepted length **2.81**.
Eviction delta 0 on all four legs; GTT 110.0 GB; worst MemAvailable 9.6 GB.

> An earlier version of this file quoted 0.735 and 3.19 here. Those are the
> **last prompt's** figures, taken from the summary line — the exact trap this
> file warns about four paragraphs below. Caught by re-deriving the aggregate
> from the logs. The throughput rows are unaffected: they were always per-leg
> means over 11 prompts.

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
> leg's aggregate. Judge on mean throughput, not on that line — and if you want
> an acceptance figure, sum the accepted/generated pairs across every prompt
> yourself. I quoted that line as an aggregate once; see the correction above.

## 1b. The same drafter loses at 176k on the sibling model

The model card for this drafter reports that DSpark **does not pay** at 176k of
real KV on this same box, against the text `DeepSeek-V4-Flash`: 9.4 t/s drafting
versus 13.4 t/s raw, 0.70x. That is not a contradiction of the +54.4% above, and
both pages are right. The variable is **step cost**, and it is computable from
the numbers each page already publishes.

A speculative step produces `mean_len` tokens, so `decode = mean_len / step`:

| | depth | mean accepted len | decode | implied step |
|---|---|---|---|---|
| this lane, vision | ~275 | 2.81 | 24.04 t/s | **117 ms** |
| card, text model | 176k | ~3.7 (acc 0.54, n_max 5) | 9.4 t/s | **391 ms** |

The two targets decode at a comparable rate without a drafter (64 ms/token here,
75 ms there). The drafting **overhead** is what differs: 53 ms here against
316 ms there, a factor of six. Two things plausibly cause it and this duel
separates neither — the card's campaign ran `n_max=5`, which the sweep below
shows is the worst of the three settings on this lane, and its verify pass runs
over 176k of KV rather than 275 tokens. The `n_max` difference is worth ~12% on
this lane; it does not flip a sign on its own. I did not isolate the rest.

The card's rule stands and this lane is the other side of it: *the faster the
target and the deeper the context, the more acceptance you need before drafting
pays.* At 275 tokens the bar is low and DSpark clears it easily. At 176k it is
not cleared by anything the drafter has ever posted, including 0.92 on verbatim
recall.

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

## 3. Decode is flat from 4k to 32k

28.66 t/s @4k → 29.31 t/s @32k, with the drafter and the §2 prefill flags in.
Over **that span** the lane does not degrade, where GLM loses 36%.

Scope it to 32k: I have no paired decode measurement between 32k and the served
131072 on this configuration, and §1b is a warning against extrapolating one.
The §1 figure of 24.04 t/s is not the missing 131k point either — it is a
different configuration at ~275 tokens, before the §2 flags.

## 4. What is NOT measured

No quality score. This lane has never been put in front of a judge. Throughput
numbers here say nothing about output quality.
