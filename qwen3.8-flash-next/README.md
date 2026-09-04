# Qwen3.8-Flash-Next on Strix Halo

176B MoE (arch `qwen4exp`), 6B active, ~103 GB of weights, 200704 tokens of
context, with vision. This is the lane that serves as the daily default.

- [`launch.sh`](launch.sh) — the served configuration, annotated
- [`RESULTS.md`](RESULTS.md) — the duels, both legs

The speed number is +33% from the shared MTP head. The result worth reading is
§2: a drafter that measured **3x faster** on a benchmark replaying a cached
prompt, and was **40% slower** — and hung the lane — once the benchmark was
fixed. Its acceptance rate was 40% the whole time and never warned of anything.
