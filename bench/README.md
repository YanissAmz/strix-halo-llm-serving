# Paired-duel harness

Every headline number in this repo came out of `paired-duel.sh`.

```
DUEL_BENCH_CMD='python3 my-bench.py "$PORT" "$RUNID"' \
  ./paired-duel.sh ds4spec ../ds4v/launch.sh \
    "nodrafter:DS4V_SPEC=none" \
    "dspark:DS4V_SPEC=draft-dspark,DS4V_DRAFT=/path/to/drafter.gguf"
```

Two rounds, arms alternating, one variable. Reports both legs.

## Read the eviction delta before the throughput

An arm that evicted paid for something other than the flag under test. If the
eviction deltas differ between arms, the comparison is **void** — report it as
void, do not arbitrate it.

Do not judge memory pressure by `MemAvailable` alone. On unified memory the
lane's cost sits in GTT, and a process can show a 3 GB RSS while holding 116 GB
of GTT.

## The `draft acceptance` line lies

The line in a leg's log is the **last prompt's** acceptance, not the leg's
aggregate. Judge on mean throughput.

## Waiting for memory, not sleeping

Between legs the harness waits for GTT to actually come back — no llama-server
left, then `MemAvailable` flat for three samples. A fixed sleep loses legs to
`failed to allocate ROCm0 buffer` once the model is large enough.
