# Qwen3.8-27B dense on Strix Halo

The one dense model in this repo, and the one where the kernels were actually
opened up. 17.92 GB of Q4_K weights, 200000 tokens of context, with vision.

- [`launch.sh`](launch.sh) — the served lane (DFlash2 drafter, `n_max 5`)
- [`RESULTS.md`](RESULTS.md) — the kernel investigation (DSpark drafter,
  `-ub 192`): what moved the needle, and rather more about what didn't

Read the launcher and the results as two different configurations of the same
model, because that is what they are.

Three results worth the click:

1. **`MMVQ_MAX_BATCH_SIZE` at 4 instead of upstream's 8.** A two-constant change
   that cut the drafter's per-position cost from 24.3 ms to 2.25 ms. At width 1
   removing MMVQ costs 16.6%; at width 4 using it costs 13.0%. One compiled-in
   threshold cannot serve both regimes, and speculative decoding lives exactly in
   the band where the shipped value is on the losing side.
2. **A GQA K/V traffic optimisation that was correct, validated, and slower.**
   The logical GQA ratio is 6; the *effective* stream count, measured with two
   runs at two depths, is **0.85** — a 32 MB last-level cache absorbs the
   redundancy before it reaches DRAM. You cannot divide by three a cost that is
   never paid. Two runs would have saved the whole patch.
3. **Longer drafts are nearly free on a dense model, and filtering them hurts.**
   `n_max 5` beats `n_max 3` by 14% while acceptance *falls* from 54% to 39%; a
   `p_min` sweep raises acceptance from 50% to 93% and lowers throughput
   monotonically. The MoE lanes in this repo conclude the exact opposite. Same
   flag, same box, inverted answer.
