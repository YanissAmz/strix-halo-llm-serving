# Reproducible crash: `-ub 192` + dflash drafter on ROCm

Not filed upstream — I am blocked from commenting on the repository. Documenting
it here so it is public and citable; anyone able to file it, please do.

## Symptom

Prefill completes to 100%, then the **very first** call to
`common_speculative_impl_draft_dflash::draft()` dies with:

```
ROCm error: an illegal memory access was encountered
```

## Isolation

One variable at a time, same prompt (12,149 tokens), same binary:

| `-ub` | result |
|---|---|
| 192 | **crash** |
| 224, 256, 320, 384, 448, 512 | pass |

Not caused by `--cache-reuse`, `ngram-mod`, or `--spec-draft-poll`: each removed
separately, still crashes.

## Why 192 was there at all

The sweep that picked 192 ran **without a drafter**. It optimised a code path
the served lane does not take. The cost of moving back to 256 is nil:
pp @12k = 298.8 t/s at 256, 295.8 at 224, 287.3 at 512 — flat within noise.

256 is two steps above the boundary, not one.
