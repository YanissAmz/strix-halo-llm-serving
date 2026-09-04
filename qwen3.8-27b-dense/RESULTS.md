# Qwen3.8-27B dense — kernel work on gfx1151

Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.92 GB), dense, not MoE.
Box: AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151 / RDNA3.5), 40 CUs, 128 GB
unified memory, ~256 GB/s theoretical and ~210 GB/s effective, 32 MB Infinity
Cache, wave32.
Method: [../METHOD.md](../METHOD.md). Build: [../BUILD.md](../BUILD.md).

⚠ **Two configurations appear in this repo for this model, and they are not the
same one.** [`launch.sh`](launch.sh) is the lane actually served: a DFlash2
drafter at `n_max 5`, `-ub 256`, 200000 tokens of context. Everything below is a
kernel investigation run with a **DSpark** drafter and `-ub 192`. Numbers from
one do not transfer to the other; each says which it is.

Every number below is measured on one box, one variable at a time, with the
baseline re-measured inside the same session. Where a claim is a prediction
rather than a measurement, it says so.

Drafter for the investigation: `Qwen3.8-27B-DSpark-Q8_0.gguf` (1.44 GB, arch
`dflash`, `block_size = 7`, borrowing the target's 1043 MB Q6_K output head).

---

## The numbers that frame everything

Model weights 17.92 GB / 210 GB/s = **85.3 ms** is the floor for one mono-token pass.
The drafter adds 2.89 GB = **13.8 ms**. So a speculative pass cannot go below **99.1 ms**.

Measured at 30k context depth:

| config | ms/pass | tokens/pass | t/s |
|---|---:|---:|---:|
| no drafter | 94.3 | 1.00 | 10.6 |
| DSpark alone, `n-max 7` | 174.6 | 4.13 | 23.3 |
| **DSpark → n-gram cascade** (shipped) | **167.9** | **4.83** | **28.8** |

Decode is **2.7×** the no-drafter rate. Two things in that table are worth more than
the headline number.

**The cascade is not a union — it's a fallback, and the order is load-bearing.**
`--spec-type a,b` takes the first non-empty draft per sequence; the second engine only
runs when the first declines. Adding n-gram behind DSpark buys +23.7% (23.3 → 28.8 t/s)
for zero memory and zero compilation, because n-gram covers exactly the positions where
a learned drafter abstains — literal repetition of the prompt.

**But the order that wins depends on depth, and it inverts.** At 30k, DSpark-first wins
(28.8 vs 28.1). On short tasks, n-gram-first wins, and not narrowly (28.1 and 30.3 vs
23.9 and 28.1). Plausible reading, offered as a hypothesis: at depth the n-gram finds
many matches, but short and often wrong, and by winning the race it substitutes a poor
draft for a good one. **Going first is only an advantage if you have something better to
propose.** If you ship a cascade, pick its order at the depth you actually serve.

Now the decomposition: the drafter is only **14.5%** of the pass. A *free* drafter would
give 25 t/s at the old acceptance. Everything left is in the target model's verification
pass, which is why the rest of this post is about the target's kernels and not the
drafter.

## What worked

**`-ub 192`.** One flag, no compilation, and it survives the test that the previous
sweep failed: it is an **interior** maximum, not a boundary.

> ⚠ `-ub 192` **crashes** with a DFlash2 drafter — ROCm illegal memory access on
> the first draft call. 192 is safe with DSpark, and with no drafter. See
> [../KNOWN_CRASH.md](../KNOWN_CRASH.md). The served lane runs 256 for that
> reason.

| `-ub` | `pp2048 @ d0` | `@ d32768` |
|---:|---:|---:|
| 256 | 341.80 ± 0.83 | 203.34 ± 0.89 |
| **192** | 341.50 ± 0.91 | **209.96 ± 0.82** |
| 128 | 314.00 ± 0.63 | 185.57 ± 0.90 |

An earlier sweep stopped at 256 and reported it as the optimum — but 256 was the last
value tested, and a maximum on the edge of a sweep is a boundary, not a maximum.
Extending downward found 192, which beats both neighbours at depth (+3.3% over 256,
+13.1% over 128) and costs nothing at `d0` (−0.3%, inside σ). 128 loses on *both*
sides at once, so there isn't even a trade-off to argue about.

The mechanism — a trade between re-reading the weights once per micro-batch and the
attention tile size — has nothing to do with Qwen3.8, so this should transfer to other
dense models on the same hardware. But be honest about the size: +3.3% on 203 t/s is
210 t/s. It is an *acquired* lever, and a *minor* one.

## What was refuted, and why the refutations are worth more than the wins

### Stream-K on RDNA3.5: measured loss

Q4_K at J=16 uses I=64, so 4096/64 = **64 tiles on 40 CUs** = 1.6 waves — 24 CUs idle in
the tail. Stream-K splits the K reduction to remove that partial wave, and upstream enables
it on Ampere/Blackwell/CDNA while leaving it `false` on **all four** RDNA tables.

Enabled for the model's types, validated first (q4_K 59/59, q5_K 27/27, q6_K 27/27,
q8_0 63/63, zero FAIL), then A/B'd at 30k over three tasks:

| config | verified width | path | Δ ms/pass |
|---|---:|---|---:|
| no drafter | 1 | MMVQ | **+0.1%** |
| `n3` | 4 | MMVQ | **−0.1 / +0.5 / +0.1** |
| `n4` | 5 | MMQ | **+25%** |
| `n7` | 8 | MMQ | **+30%** |

The table reads because of the rows that *don't* move. Widths 1 and 4 sit below
`MMVQ_MAX_BATCH_SIZE` (4 on this build — see the disclosure further down; upstream ships
8), so they never enter MMQ and are outside the change by
construction — they're flat. Every row that crosses into MMQ moves, and moves the wrong
way. The fixup pass costs more than the wave imbalance it corrects. Upstream's `false` is
right, and now there's a measurement behind it.

### J = 8: compiles, runs, computes garbage

`mul_mat_q_switch_J` keeps the first J giving a single column-tile. Verifying 8 positions
with no J=8 entry means running in a **16-column tile, half of it padding**. J=8 is shipped
code — rdna2, Ampere and Pascal all instantiate it for these types. Added to the RDNA3.5
table:

```
MUL_MAT(type_a=q4_K,m=4096,n=5,k=2048): ERR = 1348.4 > 0.0005   FAIL
q4_K 46/61 · q5_K 19/29 · q6_K 19/29 · q8_0 50/65
```

Widths 1, 2, 4 pass — they stay on MMVQ. Every width that switched to the new tile fails.
**RDNA3/3.5/4 have WMMA, whose tile is 16 wide, so `J % 16 == 0` is mandatory.** The only
AMD table carrying J=8 is rdna2 — the only one without WMMA, running the dp4a path. Those
three tables aren't under-tuned; they state a hardware constraint. Nothing in the code
asserts it, so the bad config builds and runs.

### The general lesson

Twice in one day, a kernel table value that looked like an oversight was a constraint, and
in both cases **the wrong version passed the upstream test suite**. The shipped MUL_MAT
cases use `m=16`, and `fallback` is selected on `ne01 % 128` — so 16 rows only ever exercise
the bounded fallback path, never the one a 4096-row model takes. Validating at the model's
shapes is what caught both.

## Non-power-of-2 GQA ratios are a minefield

Qwen3.8 has `head_count 24, head_count_kv 4` → **gqa_ratio = 6**. `fattn-tile.cuh` tries
`% 8`, `% 4`, `% 2`; 6 matches only `% 2`, giving `ncols2 = 2` — which means **the K/V tile
is read three times** instead of once. Fixing that requires `cols_per_block = 48`, and
that trips three separate implicit power-of-2 assumptions:

1. `ncols1 = cols_per_block/ncols2` is integer division, so 32/6 = 5 and `ncols = 30`,
   absent from every table. An early `return` does **not** prevent instantiation of the
   later branches — each needs `if constexpr (cols_per_block % ncols2 == 0)`.
2. `cpw = ncols/nwarps` = 48/8 = 6 half-floats = 12 bytes into `ggml_cuda_memcpy_1`, which
   only supports {1,2,4,8,16}. Fixed with `nthreads = 384` (12 warps → cpw = 4).
3. The table entry stayed invisible: `static assertion failed: get_config(256, 256, 8*6)`.
   The `static_assert` lives in a `__global__` template body, so it is evaluated in the
   **host** pass too — and `RDNA` derives from `__gfx1151__`, a device-pass macro. At host
   pass the selector falls through to the non-RDNA table. The entry must go into **all
   four** tables. Nobody hits this upstream because every shipped `ncols` (2/4/8/16/32)
   exists in all four; the tables diverge only on nthreads/occupancy/nbatch.

The kernel body itself forbids nothing: `ncols2` is only used as `jc/ncols2`, `jc%ncols2`,
`blockIdx.z/(ne02/ncols2)` — integer arithmetic, no power-of-2 assumption.

### The correction that matters: `ncols2` is not free

The obvious reading — "K/V read 3× instead of 1×, so fixing it divides the dominant
term by 3" — is **wrong**, and the error is worth more than the fix. Attention K/V
traffic is a *product* of two tilings:

```
K/V streams = ceil(n_queries / ncols1) × (gqa_ratio / ncols2),   ncols1 = cols_per_block / ncols2
```

At constant `cols_per_block`, tripling `ncols2` divides `ncols1` by 3 — so the second
factor improves 3× while the first degrades 3× as soon as there are enough queries to
fill the tile. At decode there aren't (`ceil(1/x) = 1`), so the gain is full. At prefill
there are, and the two factors nearly cancel.

Which path is actually taken settles it. The `cols_per_block = 64` branch is guarded by
`if constexpr (DKQ <= 128)` — **Qwen3.8 has DKQ = 256, so it never fires**; on HIP the
next branch's `DKQ <= 256` guard is compiled out, so prefill lands on 32:

| | ncols2 | cpb | ncols1 | streams @256 q | @1 q |
|---|---:|---:|---:|---:|---:|
| shipped | 2 | 32 | 16 | 48 | 3 |
| `cols_per_block = 48` | 6 | 48 | 8 | 32 | **1** |
| `cols_per_block = 96` | 6 | 96 | 16 | **16** | — |

So a single tile width can't serve both: 48 gives the full 3× at decode and only 1.5×
at prefill; 96 gives the full 3× at prefill and wastes half the tile at decode. The fix
is a *ladder*, and the threshold can be lifted verbatim from the shipped rule — each
branch activates when `ne[1] > (cols_per_block/2)/ncols2`, i.e. "don't take a tile that
is less than half full". For 96 at ncols2 = 6 that is 8.

**Then we measured it, and the decode half of that paragraph is wrong.** `cols_per_block
= 48` at width 1 costs **+17.1 %** (110.4 vs 94.3 ms/pass at 30k, dispersion 0.06 ms —
about 180σ). The table above says `ncols1 = 8`, and I read that as "8 columns available".
At one real query the tile runs **1/8 full**, and the 7 ghost columns still do their
QK^T and PV over the entire context. That cost is not a fixed launch overhead: it is
0.9 ms at 1k and 16.1 ms at 30k, i.e. it *scales with depth* — the exact opposite of the
regime the change was meant to improve.

The natural fix is a `cols_per_block = 6` rung (`ncols1 = 1`, the exact analogue of the
shipped ratio-2 `cpb = 2`). Don't build it. One more row of the same run kills it:

| width | shipped (ncols2=2) | ncols2=6 | tile shape | K/V streams |
|---|---:|---:|---|---:|
| 1 | 94.35 / 94.31 / 94.37 | 110.44 / 110.35 / 110.44 | ref full, new **1/8** | 3 → 1 |
| 8 | 174.63 / 173.88 / 174.18 | 174.41 / 173.76 / 172.93 | **both `ncols1 = 8`, both full** | 3 → 1 |

At width 8 the two binaries run the *same tile shape* and differ only in K/V streams,
3 → 1. They tie. The mechanism, tested with a perfectly full tile, buys nothing at
decode.

The reason is arithmetic you can run on any GPU before writing a line of kernel code:

```
depth penalty  = 94.34 − 86.15 = 8.19 ms      (measured, 1k → 30k)
one K/V read   = 1.9 GiB / 210 GB/s = 9.6 ms  (16 full-attn layers, 64 KiB/token)
effective streams = 8.19 / 9.6 = 0.85
```

**0.85, for a logical ratio of 6.** Blocks sharing a K/V head run concurrently and the
32 MB Infinity Cache absorbs the redundancy before it reaches DRAM. You cannot divide by
3 a cost that is never paid. The GQA ratio is a *logical* ratio; what you pay is an
*effective* one, and it is measurable with two runs at two depths — no profiler needed.

At this point the idea still had one life left: take the `ncols2 = 6` path only where the
tile can be full — i.e. at prefill — and fall back to the shipped ladder everywhere else.
That guard is worth describing even though the next section kills it too, because getting
it *placed* right is the non-obvious part. It has to sit in
`switch_ncols2`, not `switch_ncols1` — once `ncols2 = 6` is instantiated, the shipped
rungs (16/8/4/2) are all removed at compile time by `cpb % ncols2 == 0`, so there is
nothing left to fall back to. *A guard belongs where the alternative still exists;
after an `if constexpr` specialisation, the alternative has already been deleted.*

**And at prefill — where the whole idea was supposed to pay off — it loses too.** This is
the part worth carrying away, because the shipped code looks incomplete here and actively
invites this patch:

| variant | pp2048 @ d0 | pp2048 @ d32768 | vs shipped |
|---|---:|---:|---:|
| shipped | 340.85 ± 0.55 | **203.62 ± 0.44** | — |
| `ncols2=6`, cpb 96 (needs `nbatch_fa`/2) | 332.49 ± 0.80 | 155.84 ± 0.48 | **−23.5 %** |
| `ncols2=6`, cpb 48 (`nbatch_fa` intact) | 334.62 ± 0.74 | 165.48 ± 0.52 | **−18.7 %** |

The middle row confounds two variables — cpb 96 does not fit in LDS, so `nbatch_fa` had to
be halved, which doubles the number of K/V iterations. The third row removes that
confound entirely, and still loses by 18.7 %. Dropping the confound recovers 9.6 of the
23.5 points: it mattered, but it was not the cause. **The mechanism is the cause.**

The traffic arithmetic was never wrong. At `ub 256` the shipped path issues
`ceil(256/16) × 3 = 48` streams and the cpb-48 path issues `ceil(256/8) × 1 = 32` — a
third less traffic, for 18.7 % less throughput. The arithmetic was answering a question
the hardware doesn't ask: **the redundant K/V traffic is never paid.** It is absorbed by
the last-level cache before it reaches DRAM — measured at 0.85 effective streams for a
logical ratio of 6. So `ncols2` trades a saving that doesn't exist against costs that do:
twice as many position tiles (each reloading Q), and a wider tile that raises LDS per
block, lowering occupancy and hiding less latency.

If you take one thing from this post: on any architecture with a large last-level cache,
**do not pay for GQA K/V traffic reduction until you have measured the effective stream
count.** Two runs at two depths give it to you: `(ms@deep − ms@shallow) / (ΔKV bytes /
bandwidth)`. Ours came out 0.85 where the logical ratio said 6. Everything downstream of
that number was decided before we wrote a line of kernel code — we just didn't measure it
first.

Validation was green throughout (2936/2936, 0 FAIL, ratio-6 cases confirmed exercised),
and the shipped baseline reproduced three times inside the session (341.45 / 340.85 /
341.39 at d0). This is a correct kernel that is simply slower — which is the most useful
kind of negative result, because it rules out "you had a bug" and leaves the performance
question standing.

One hardware wall worth stating, since it bounds the whole idea at DKQ = 256:
`cols_per_block = 96` does not fit in LDS. `Q_tmp = ncols × DKQ × 2 B` = 49152 B on its
own — 62 % of the 64 KiB budget, and irreducible. The compiler is blunt about it:
`local memory (78848) exceeds limit (65536)`. Only `KQ` and `KV_tmp` are negotiable, via
`nbatch_fa`; halving it compiles, at the cost of chopping the K/V into twice as many
iterations. The 96 rung is therefore not "48 with a wider tile" — it is a different
trade, and it can lose.

Validation passes at the model's shapes (2926/2926, 0 FAIL) with ratio-6 cases actually
exercised — and the coverage counter is not optional: an earlier run reported 0 ratio-6
cases because the pattern searched for `nr=6` while the harness prints `nr23=[6,1]`.
A counter that finds nothing and a branch that never runs look identical from outside.

## A tuning table that exists for CDNA and not for RDNA

`ggml_cuda_should_use_mmvq` carries per-quant-type thresholds for when to leave the
vector kernel for the tiled one — entirely inside `GGML_CUDA_CC_IS_CDNA(cc)`:

| type | CDNA1 | CDNA2 | RDNA (all) |
|---|---:|---:|---:|
| Q4_0 / Q4_1 | ≤ 7 | — | 4 |
| Q8_0 | ≤ 6 | — | 4 |
| Q3_K | ≤ 3 | ≤ 3 | 4 |
| **Q4_K** | **≤ 2** | **≤ 3** | **4** |
| Q6_K | ≤ 4 | ≤ 5 | 4 |

Someone measured, on CDNA, that Q4_K leaves MMVQ at width 3 while Q4_0 stays until 7 —
a wide spread, in a direction with a physical reading: K-quants carry two levels of
scales, so dequantisation per byte read is heavier, and the tiled kernel (which
dequantises once into LDS and reuses) wins sooner. On RDNA the fallback is flat for every
type — `ne11 <= MMVQ_MAX_BATCH_SIZE`, the macro default rather than a measurement.

One disclosure that matters for reading every number in this post: **upstream ships that
macro at 8, and this work runs it at 4.** Lowering it (and `MMVF_MAX_BATCH_SIZE` with it,
forced by a `static_assert`) cut the drafter's per-position cost from 24.3 ms to 2.25 ms —
a factor of 10.8, validated at 0 FAIL first — and moved best decode from 16.8 to 19.2 t/s.
That is *the* biggest single win in this whole effort, and it is a two-constant change, so
the attribution is to the pair, not to `MMVQ` alone. Everything below therefore sweeps
around 4, not around 8; upstream's 8 was already refuted on this hardware.

The interesting part is that this is *not* general neglect of RDNA. The **upper**
boundary, MMQ vs hipBLAS, does carry an RDNA table, and a finely-grained one:
`Q2_K ≤ 128`, `Q6_K ≤ 128` on RDNA3.0 but `≤ 256` on **RDNA3.5**, and an `IQ2_XS`
case special-cased for RDNA3.5 specifically (`mmq.cu:347-368`). Somebody measured this
hardware carefully. They measured the boundary that matters for prefill, and left the
one that matters for decode at its default. That is a missing column, not a missing
table — which makes it a much easier thing to contribute.

And it lands exactly on speculative decoding with a Q4_K model, where the verification
widths are 3, 4, 5 and 8 — straddling the threshold.

One thing to check before running this experiment, because disabling a dispatch does not
tell you where you land: after MMVQ is refused, the next switch is
`ggml_cuda_should_use_mmq`, and if *that* refused width 1 you would fall through to
hipBLAS — measured elsewhere in this work at 4–8× worse — and read a third kernel's cost
believing you had priced MMQ. Reading it settles it: for Q4_K on RDNA3.5 the switch ends
at `default: return true`, with no lower bound on `ne11`. The trap is not armed here, but
it is one grep away from being armed for a different quant type.

One implementation note, because it cost a build: you cannot answer this by setting
`MMVQ_MAX_BATCH_SIZE` to 0. The macro also feeds
`__launch_bounds__(... * warp_size)`, and a 0-thread block does not exist
(`amdgpu_flat_work_group_size ... min must not be greater than max`). It is a kernel
*compilation* parameter, not a dispatch switch. Patch the dispatch function instead —
and make the threshold an environment variable, so the whole sweep runs on **one**
binary and no compilation difference can hide inside the curve.

Here is the curve. `ms_per_pass` at 30 375 tokens of context, one binary, threshold set at
runtime, three repetitions per point. **Verified width** is `n_draft + 1` — the drafted
tokens plus the target's own — and it is what the dispatch actually compares against the
threshold.

| config | verified width | cut=4 | cut=3 | cut=2 | cut=0 |
|---|---:|---:|---:|---:|---:|
| no drafter | 1 | **94.29** | 94.30 | 94.33 | **109.98 (+16.6%)** |
| DSpark `n-max 3` | 4 | 171.17 | **148.91 (−13.0%)** | 151.36 | 149.70 |
| DSpark `n-max 4` | 5 | 165.42 | 165.79 | 166.32 | 165.82 |
| DSpark `p-min 0.30, n-max 7` | 8 | 161.60 | 158.42 | 160.32 | 164.40 |

Two numbers carry it. **At width 1, removing MMVQ costs 16.6%** — it is emphatically the
right kernel for single-token decode. **At width 4, using MMVQ costs 13.0%** — it is
emphatically the wrong one. The crossover is somewhere in between, and upstream's single
compiled-in threshold of 8 puts widths 5 through 8 on the losing side of it. That is the
band where real speculative decoding lives.

The controls are what make this readable. `no drafter` keeps the same kernel path across
thresholds 4, 3 and 2, and moves by **0.03%** across all three — the machine did not drift
during the sweep. `n-max 4` (width 5, MMQ on every arm) moves 0.2%. Against controls that
flat, 13.0% and 16.6% need no defending.

Two cautions we had to write into our own notes:

- The width-8 row moves by up to 2.0% and *cannot* serve as a control, because `p-min 0.30`
  cuts the draft adaptively — its verified width varies per request. Its spread was already
  1.5% *within* a single arm. A variable-width config proves neither drift nor its absence.
- `MMVQ_MAX_BATCH_SIZE` is a **global** constant. It governs the drafter's matmuls too, at
  batch widths that are not the verified width in the table. So each arm changes two things
  at once on any config that has a drafter, and fine comparisons between arms with the same
  *target* path (cut=3 vs cut=2) are not interpretable. The 13.0% survives this because the
  config with no drafter at all doesn't budge.

One more check, because the sweep above tests four configs and *none of them is the one we
ship*. The shipped config is a cascade, and a cascade does not have a width — it has one
per arm. DSpark submits 8; the n-gram fallback submits whatever it matched. Measured, the
cascade's mean draft at 30k is **4.43 tokens, not 7**, so a real share of its passes land
on the MMVQ side of the threshold. The premise "it verifies 8, so the threshold can't touch
it" was simply wrong.

Lowering the threshold for it buys 1.6% of pass cost and gives back 1.8% of acceptance:
28.83 → 28.74 t/s, i.e. nothing. Changing the kernel moves the logits, which moves
acceptance, which cancels the hardware win. Worth stating because it is the general shape
of this whole area: **a pass-cost win under speculative decoding is not a throughput win
until you have checked what it did to acceptance.**

The honest limit: none of our configs verifies 2 or 3 tokens, so we can say the optimum is
below 4 and at least 1, and not which. Pinning it needs two more drafter widths. What is
already established is that **one compiled-in threshold cannot serve both regimes** — the
same constant wants to be high for single-token decode and low for speculative
verification. That is an argument for a per-width dispatch rule, not for a different
constant.

## Method notes that cost us time, so they may save yours

- **A kernel is judged on `ms_per_pass`, not `t/s`.** Under speculative decoding, t/s mixes
  hardware cost with acceptance, and any change to reduction order shifts the logits, hence
  acceptance. A frozen draft pass moved throughput from −8.8% to +16.8% with no hardware
  change.
- **`draft_n_accepted = 0` exactly, three times, is a wrong kernel, not a bad policy.**
  A drifting acceptance gives 0.42 or 0.52; exact zero is arithmetic. A bench capturing
  only timings would have filed that bug as a mild negative result.
- **A build directory whose sources were restored is no longer a reference.** These
  experiments patch sources, build, then restore. Rebuilding *anything* in that
  directory afterwards recompiles from the restored sources and silently produces a
  binary that carries the experimental arm's name without its patch — no error, no
  warning. Either rebuild everything under patch, or rebuild nothing.
- **A missing build target only shows up at bench time.** `--target llama-server
  test-backend-ops` does not build `llama-bench`. Build green, validation green, and the
  benchmark discovers the gap after it has already taken the GPU.
- **Same prompt at temperature 0 does not mean same output across two builds.** Any kernel
  change alters prefill arithmetic, which alters the KV cache, which alters generation.
  Output hashes detect *whether arithmetic moved*, and say nothing about quality.
