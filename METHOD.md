# How these numbers were taken

The hardware is unusual and the numbers are good, so the method has to be
checkable. This is the part worth copying even if you never run these models.

## Paired duels, one variable

Every headline figure is a **paired duel**: two arms, alternating legs, the same
prompt set, the same binary, the same window, one flag different. Both legs are
reported. If the two legs of one arm disagree, the measurement is noise and does
not get a headline.

`n=1 decides nothing.` A single fast run is a rumour. One number in this repo —
39.1 t/s on Qwen — was seen once and is deliberately *not* the headline; the
paired figure of 36.0 t/s is.

## A number carries its instrument

Two numbers from different harnesses do not compare, even for the same model on
the same box. Every table here states the date, the window, and what was
measured. Where a duel ran at a window other than the served one, it says so.

## The control has to reproduce

Before believing an A/B, re-run the baseline. A baseline that does not reproduce
means the harness moved, not the flag.

## Env flags read by presence

Several `ggml`/`llama.cpp` env switches are read with `getenv() != NULL`. Setting
one to `0` **disables nothing** — it enables. A control arm built with `FLAG=0`
against an arm with `FLAG=1` compares enabled to enabled and returns a
convincing null result. Test by unsetting, and verify the served process:
`tr '\0' '\n' < /proc/<pid>/environ`.

## Read the command line the process actually got

A comment inside a `\` line continuation truncates the command silently.
`bash -n` passes. The lane comes up serving no drafter, no metrics, a different
window — under the right alias. Check `/proc/<pid>/cmdline`, not the script.

## Config edited is not config served

A daemon can keep the previous config in memory and only fail at the next
restart. A served window can differ from the declared one. Verify what is
serving, not what is written.

## First touch is slower than the flag you are testing

The first leg of a session pays page faults, allocator growth and GTT
first-touch. On this box that cost is large enough to swamp the effect being
measured: a flag once looked worth ±126 σ and was the arm order, not the flag.
Two defences, and only one of them is in every duel here. **Alternating arms**
over two rounds is in the harness itself, so residual drift hits both arms
equally — that is what `bench/paired-duel.sh` does. A **discarded warm-up leg at
the target depth** was added later, after the `-ub` sweep on the Qwen lane where
the rank effect was caught; the duels taken before that date do not have it, and
the harness does not enforce it. If you reuse this harness, add the warm-up leg
yourself — the harness will not do it for you.

## Both directions of a lesson are real

`--spec-draft-p-min` filtering is worth +36% on the MoE GLM lane and is
actively harmful on the dense 27B lane, where throughput falls monotonically
while acceptance climbs to 93%. A setting that helps one lane is not a default.
Re-measure per model, per quant.

## An acceptance line is one prompt, not the run

llama.cpp's `draft acceptance = ...` in a slot's timing block is that **request's**
figure. Read at the end of a multi-prompt leg it looks like the leg's aggregate
and is not. Two numbers in an earlier version of this repo were quoted that way;
one was re-derived (0.735 → **0.609**) and the other withdrawn. Sum the
`accepted / generated` pairs across every request yourself.

## A window is not a depth

`-c 131072` says what the lane can hold. It says nothing about how full the KV
was when the number was taken. The largest win in this repo, +54.4%, was
measured at **~275 tokens** in a 131072 window — and the same drafter is a net
loss at 176k. State both, always, and never let the window stand in for the
depth.
