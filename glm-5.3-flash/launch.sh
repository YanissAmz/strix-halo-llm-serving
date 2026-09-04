#!/usr/bin/env bash
# GLM-5.3-Flash (arch `glm5next`, 320B total, UD-Q2_K_XL ~101 GB) on llama.cpp /
# HIP, AMD Strix Halo 128 GB unified memory.
#
# ⚠ This model needs an architecture port that is NOT in upstream llama.cpp.
#   Read ../BUILD.md before running anything here.
#
# Every flag below is here because a paired duel said so. The measurement behind
# each one is in RESULTS.md, with its date and its window.
set -euo pipefail

MODEL=${GLM_MODEL:?path to GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf}
SERVER=${GLM_SERVER:?path to llama-server built from the glm5next tree — see ../BUILD.md}
PORT=${GLM_PORT:-8289}

# --- window ------------------------------------------------------------------
# Native window is 1048576. That is not a target: on this box the weights and the
# KV cache come out of the same 128 GB. 65536 is what this lane serves.
#
# 131072 was tried and given up: under agentic load it produced 131 evictions in
# 3 minutes with 0.52 GB left. At 65536 the same load leaves 5.58 GB and evicts
# zero times. Both numbers were taken with the window actually FULL — a window
# measured empty tells you nothing about the one you serve.
CTX=${GLM_CTX:-65536}

# --- KV cache ----------------------------------------------------------------
# q8_0, and this one is not a trade:
#   131k : decode 2.65 -> 3.02 t/s (+14%)   perplexity 4.0970 -> 4.0774 (BETTER)
#    32k : 7.64 -> 7.62 t/s (no change either way)
# A setting that buys 14% while improving perplexity does not get left off. It
# also halves the KV, which is what pays for the window in the first place.
#
# Perplexity improving under quantization is not magic: llama.cpp applies a
# Hadamard rotation to quantized KV (llama-kv-cache.cpp, gated on
# ggml_is_quantized(type_k) && n_embd_head_k % 64 == 0 — generic, not a model
# list). A bare quantized cache does not beat f16; a rotated one can.
#
# The indexer's own key cache stays f16 regardless of -ctk. The model code forces
# it, and the server logs "indexer key cache stays f16" at load. Do not go
# looking for the missing memory saving.
KV=${GLM_KV:-q8_0}

# --- speculative decoding ----------------------------------------------------
# Two drafters work on this model. Both are measured in RESULTS.md.
#
#   mtp     — the MTP head embedded in the GGUF. Free: no second model, no extra
#             weights, no extra KV. n_max=1; deeper does not pay on this head.
#   dflash  — an external DFlash2 drafter (~1.2 GB of weights, ~1.9 GB of its own
#             KV = 3.1 GB measured, not guessed). Faster than MTP at 98304 but
#             the 3.1 GB is exactly the margin this box does not have.
#
# ⚠ --spec-draft-p-min is NOT optional with DFlash2. Its default is 0.00, which
#   means the draft is never filtered: the drafter emits everything and the
#   server pays to verify tokens that were always going to be rejected. Without
#   it DFlash2 measured a dead heat against no drafter at all. With p_min=0.60 it
#   is +36%. Sweep in RESULTS.md — there is a cliff on the left, not a slope.
SPEC=()
case "${GLM_SPEC:-mtp}" in
  mtp)
    SPEC+=(--spec-type draft-mtp --spec-draft-n-max "${GLM_NMAX:-1}")
    ;;
  dflash)
    SPEC+=(--spec-type draft-dflash
           -md "${GLM_DRAFT:?path to GLM-5.3-Flash-DFlash2-Q8_0.gguf}"
           --spec-draft-p-min "${GLM_PMIN:-0.60}"
           --spec-draft-n-max "${GLM_NMAX_DF:-3}"
           # Force the placement. -ngld auto fails on this model
           # ("dflash requires ctx_other to be set"), and quantize the drafter's
           # KV too — f16 there is half of the 3.1 GB it costs.
           -ngld "${GLM_NGLD:-99}" -devd "${GLM_DEVD:-${GLM_DEVICE:-ROCm0}}"
           --spec-draft-type-k "${GLM_DKV:-q8_0}" --spec-draft-type-v "${GLM_DKV:-q8_0}")
    ;;
  none|0) ;;
  *) echo "GLM_SPEC must be mtp, dflash or none" >&2; exit 2 ;;
esac

# --- vision ------------------------------------------------------------------
# Images ride in the SAME sequence on the SAME slot, so the KV is not evicted and
# the conversation survives an image. F16 is mandatory: llama-quantize refuses
# the "clip" architecture, there is no q8 mmproj.
# Costs 1.05 GiB of GTT and 0% of decode speed.
MMPROJ=()
[ "${GLM_MMPROJ:-0}" != 0 ] && MMPROJ=(--mmproj "$GLM_MMPROJ")

# --- sampling ----------------------------------------------------------------
# The GGUF declares temp=1.0 and top_p=0.95, and declares NO top_k. llama.cpp
# nevertheless stacks top_k=40 and min_p=0.05 by default. Nobody asked for those.
# On a 154880-token vocabulary they are two invisible truncations, so turn them
# off explicitly.
#
# Honest nuance: the publisher did not say "top_k disabled", they said nothing.
# Setting 0 is a choice — between an arbitrary llama.cpp value and no truncation
# at all, no truncation is the one that invents nothing.

# --- chat template -----------------------------------------------------------
# The GGUF's own jinja accepts ONLY 'low' and 'high':
#   reasoning_effort if reasoning_effort in ['low','high'] else 'max'
# Everything else — including 'medium' — falls through to 'max' in silence. With
# no kwarg at all you have been serving 'max' since day one. 'high' is a real
# rung of the template; it is the default here.
EFFORT=${GLM_EFFORT:-high}

# --- mmap --------------------------------------------------------------------
# --no-mmap by default, to match the rest of the fleet. But keep the measurement:
# at ~101 GB of weights in 128 GB of RAM, the resident set survives ONE pass and
# then collapses. Measured on a comparable lane: pp512 @65k 165.15 then 51.19 on
# the second pass, decode 19.55 then 15.06, with `free` showing 0 GB available.
# mmap reproduced to 0.3%. If this lane slows down or gets killed at depth,
# GLM_MMAP=1 is where you come back.
MMAP=(--no-mmap)
[ "${GLM_MMAP:-0}" = 1 ] && MMAP=()

# --- context cache -----------------------------------------------------------
# --cache-reuse is measured and it does LESS than its name suggests. Four calls
# on a 22585-token prompt, reading the delta of llamacpp:prompt_tokens_total
# (which excludes cached tokens, so it counts real work):
#   1. fresh context          232.4 s | actually prefilled  22576
#   2. identical prefix         0.6 s | actually prefilled      4
#   3. MIDDLE edited          233.1 s | actually prefilled  22576
#   4. back to the original   233.8 s | actually prefilled  22576
# Exact-prefix caching is perfect. Re-gluing is never attempted: even returning
# to a middle already prefilled in run 1 pays the whole thing again. The server
# keeps one prefix chain, it does not stitch shifted fragments. So a compaction
# or an edited message costs ~230 full seconds on this lane, and the flag only
# ever saves the conversation that grows at the end.
#
# ⚠ This flag can also be INERT with no error: the server disables cache_reuse
#   when the memory cannot shift (get_can_shift() == false). The only evidence is
#   one log line at load — "cache_reuse is not supported by this context, it will
#   be disabled". Check for it before believing a measurement.
#
# Those three timings reproduce to 0.3% (232.4 / 233.1 / 233.8). That is the
# control this lane requires before any A/B run on it.

exec "$SERVER" \
  --host 127.0.0.1 --port "$PORT" \
  -m "$MODEL" \
  --device "${GLM_DEVICE:-ROCm0}" -ngl 99 -fa on --jinja \
  --chat-template-kwargs "{\"reasoning_effort\":\"$EFFORT\"}" \
  "${MMAP[@]}" "${MMPROJ[@]}" \
  -c "$CTX" -b "${GLM_UB:-2048}" -ub "${GLM_UB:-2048}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --parallel "${GLM_NP:-1}" \
  --ctx-checkpoints "${GLM_CKPT:-64}" --checkpoint-min-step "${GLM_CMS:-1024}" \
  --cache-reuse "${GLM_REUSE:-0}" --no-warmup \
  --metrics \
  --temp "${GLM_TEMP:-1.0}" --top-p "${GLM_TOP_P:-0.95}" \
  --top-k "${GLM_TOP_K:-0}" --min-p "${GLM_MIN_P:-0.0}" \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --alias "${GLM_ALIAS:-GLM-5.3-Flash}" \
  "${SPEC[@]}"
