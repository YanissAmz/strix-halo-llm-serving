# GLM-5.3-Flash on Strix Halo

320B-parameter MoE (arch `glm5next`), UD-Q2_K_XL, ~101 GB of weights in 128 GB of
unified memory, with vision, at 65536 tokens of context.

> **Prerequisite:** upstream llama.cpp does not support this architecture. See
> [../BUILD.md](../BUILD.md) before running the launcher.

- [`launch.sh`](launch.sh) — the served configuration, every flag annotated with
  the measurement that justifies it
- [`RESULTS.md`](RESULTS.md) — the duels, both legs, including the ones that
  changed my mind

Three things here are worth more than the speed number:

1. **`--spec-draft-p-min` defaults to 0.00**, which makes a DFlash2 drafter
   slightly *worse* than no drafter. Set it to 0.60 and the same drafter is +36%.
   "We tried it, it did nothing" was wrong, and the flag was the whole story.
2. **The free drafter won.** DFlash2 benchmarked faster, cost 3.10 GiB, and
   killed the lane at 24k of real context. The MTP head embedded in the GGUF
   costs nothing and serves.
3. **A text-only duel does not validate a vision lane.** The MTP head passed
   every text benchmark and returned HTTP 500 on the first image, on both boxes.
