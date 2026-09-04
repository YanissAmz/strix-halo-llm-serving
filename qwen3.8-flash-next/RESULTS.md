# Qwen3.8-Flash-Next — what was measured

Model: Qwen3.8-Flash-Next, arch `qwen4exp`, 176B total (125B core + 51B n-gram
table), 6B active, UD-Q4_K_XL with q6dense (~103 GB).
Box: AMD Strix Halo, gfx1151, 128 GB unified memory, Vulkan.
Method: [../METHOD.md](../METHOD.md). Launcher: [launch.sh](launch.sh).

This is the lane that serves as the daily default on both boxes, at 200704
tokens of context, with vision.

## 1. The shared MTP head — +33%

Warm against warm, 11 prompts, one variable (the binary):

| arm | decode |
|---|---|
| no drafter | 27.0 t/s |
| **shared MTP head, n_max=3** | **36.0 t/s** |

**+33%**, acceptance 74.2% over 3396 drafted tokens, same 200704 window.

"Shared" means the head borrows `token_embd` and `output` from the target model
rather than shipping its own copies: ~1.3 GB cheaper than a standalone drafter,
at the same window. On a box where the margin is the whole problem, that is the
difference between a drafter you can serve and one you cannot.

A one-shot 39.1 t/s was seen the same day and is **not** reported as the result.
It is the weakest measurement of the set — one prompt, one draw. The paired 36.0
is the number that commits.

### The promotion gate

The faster binary was not promoted on speed alone. Before switching, at ctx 8192
on both sides:

| check | result |
|---|---|
| `/props`, 7 fields compared | identical |
| `/apply-template` jinja render | identical to the character (1831 chars) |
| tool-call round trip | `finish=tool_calls`, 1 call, valid JSON, both sides |

The template comes from a FILE, not from the GGUF's jinja — which is why 10770
commits of divergence do not change the render.

## 2. The drafter that measured 3x and was actually −40%

`ngram-mod` was enabled by default for one day. First measurement, 8 legs, using
the `/metrics` counters rather than a stopwatch:

| workload | control | ngram-mod |
|---|---|---|
| rewriting | 24.51 t/s | **74.55 t/s** (368 drafts, 368 accepted, 100%) |
| prose | 25.19 t/s | 25.08 t/s (no draft emitted at all) |

A lever that triples in one case and costs nothing in the other looks like it has
no trade-off. It gets switched on.

Re-measured properly — one variable, three draws per depth, a unique salt per
draw so nothing is served from cache:

| depth | ngram-mod | no drafter |
|---|---|---|
| 3085 | 16.5 t/s | **27.3 t/s** (range 0.3) |
| 24607 | 14.8 t/s | **25.9 t/s** (range 0.0) |

**−40%.** And the lane also HANGS: 214 tokens then a dead stop, GPU pinned at
93%, `n_decoded` frozen. Six draws without speculation: no hang.

The 74.55 t/s was not reproducible. The benchmark that produced it replayed a
prompt already in cache (4 tokens of prefill instead of 3051) and drew once.

**The acceptance rate warned of nothing**: 40% accepted, mean accepted length
26.6, and still slower than emitting no draft at all. Verifying a 59-token batch
costs more than the accepted tokens return. The per-position histogram says it
plainly — positions 44 to 63 accept zero.

Two lessons, and the second is the expensive one:

- A benchmark that replays a cached prompt is measuring the cache.
- **Acceptance rate is not throughput.** The only criterion that decides is
  `tokens_predicted_total / n_decode_total`.

## 3. Vulkan beats ROCm on this APU

Two alternating passes, 8 measurements:

| | ROCm | Vulkan | |
|---|---|---|---|
| decode @16k | 20.4 t/s | **23.9 t/s** | +17% |
| prefill @64k | 183.7 t/s | **245.6 t/s** | +28% |
| slope 16k → 64k | −46% | **−32%** | gain grows with depth |

## 4. q6dense — 7 to 9% for 0.06% of perplexity

`attn_qkv` / `attn_gate` / `ssm_out` / `attn_q` from Q8_0 down to Q6_K; the
experts do not move.

| | before | after | |
|---|---|---|---|
| size | 106.26 GiB | **102.98 GiB** | |
| decode, flat | 24.84 t/s | **27.00 t/s** | +7.4% |
| decode @65k | 22.26 t/s | **24.32 t/s** | +9.3% |
| perplexity (wikitext, 40 blocks, n_ctx 4096) | 3.8725 ±0.0288 | 3.8749 ±0.0288 | +0.062% |

The gain grows with depth because the dense weight per token is constant while
the cache gets heavier. The perplexity difference is twenty times under the error
bar: there is nothing here to arbitrate.

## 5. A ±126 standard deviation that was not the flag

The `-ub` sweep first said 512 was both faster at depth and far more stable:
2048 gave 218.40 ±126.62, 512 gave 278.76 ±5.53. The lane was moved to 512.

Re-run with two repetitions it says the opposite: 2048 gives 374 flat and 337.76
±6.03 at d65536; 512 gives 333 and 309 ±9.

What varied was not the batch size but the **rank**: the ±130 landed on the FIRST
leg at `-d 65536` every time, whatever configuration was running there. It is the
cost of touching pages for the first time, paid by whoever goes first. A warm-up
leg at depth was added to the harness so it cannot happen again.

## 6. Window: 200704, not 262144

262144 is the native ceiling and the device-memory projection says it fits (85.7
GB against 123.7 free). With the window actually full, host MemAvailable falls to
3.7 GiB and keeps drifting — under earlyoom's threshold. It survives on swap
alone, and swapping during decode ruins decode. `llama-server` is in earlyoom's
`--avoid` list, so it would not even be killed cleanly: it would hit the
allocator wall mid-request.

**A device-memory projection is not a memory measurement.**

## 7. One slot is not enough for two conversations

`--parallel 1` gives every caller the same slot, so two conversations evict each
other's prefix and each turn re-prefills the whole preamble: 6.7 s per switch,
and in one observed session 26842 tokens re-prefilled for 97 s per turn while the
lane itself had not moved at all. `--parallel 4 --kv-unified
--no-cache-idle-slots --slot-prompt-similarity 0.9` is the fix.

Prefix reuse does work on this recurrent-memory architecture (the log says
`context seq_rm type = RS`): on a real multi-turn conversation with an image,
1611/1636 then 1632/1660 tokens reused, 1.6 s instead of 5.8. Raising the
checkpoint budget from the defaults (8 / 8192) to 512 / 16 changed **nothing**
and each checkpoint weighs ~113 MiB, so the defaults stand.

## 8. The publisher's `medium` reasoning branch is empty

The GGUF's jinja sets `reasoning_instructions = ''`, fills it for `xhigh` and
`low`, and has no `elif` for `medium`. `medium` therefore passes validation
without raising, the variable stays empty, and the injection sites emit nothing.
Since this model became the fleet default it had been serving with no reasoning
instruction at all — neither the publisher's nor ours.

Not a setting. A box the publisher declared and forgot to fill.

## What is NOT measured

- No quality score attached to the drafter arms. Speculative decoding verifies
  every drafted token against the target, so quality is bounded by construction;
  bounded is not measured.
- The Vulkan/ROCm comparison is decode and prefill only, not quality, not vision.
- The 39.1 t/s one-shot is reported here only to say it is not the result.
