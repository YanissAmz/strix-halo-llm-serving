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
