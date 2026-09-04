# DeepSeek-V4-Flash-Vision-Exp on Strix Halo

UD-IQ3_XXS (~97 GB) with vision, at 131072 tokens of context.

> **Prerequisite:** this lane needs three flash-attention patches and two
> unmerged vision PRs. Read [../BUILD.md](../BUILD.md) first — one of the levers
> is an environment variable that a stock build ignores in silence, so an
> unpatched run reproduces the baseline and looks like the result.

- [`launch.sh`](launch.sh) — the served configuration, annotated
- [`RESULTS.md`](RESULTS.md) — the duels, both legs, plus the dial sweeps

Two results:

1. **+54.4% decode** from adding a drafter to a lane that had none (15.57 → 24.04
   t/s). The MTP head is not embedded in this GGUF — checked across all four
   shards — so an external drafter was the only option.
2. **Prefill depth-loss cut from −27.3% to −8.8%**, by copying a block of flags
   that the text-only lane on the same box had been serving for weeks.
