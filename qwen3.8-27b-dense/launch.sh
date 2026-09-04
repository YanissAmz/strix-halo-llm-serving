#!/usr/bin/env bash
# Qwen3.8-27B dense, UD-Q4_K_XL (17.92 GB), on llama.cpp / HIP, AMD Strix Halo.
# 200000 tokens of context, with vision.
#
# ⚠ This is the SERVED lane. The kernel investigation in RESULTS.md ran a
#   different drafter (DSpark) and a different -ub (192). Do not mix the two.
#
# Read ../BUILD.md: this lane runs MMVQ_MAX_BATCH_SIZE / MMVF_MAX_BATCH_SIZE at
# 4 where upstream ships 8, which is the single biggest win in the whole effort.
set -euo pipefail

MODEL=${Q38_MODEL:?path to Qwen3.8-27B-UD-Q4_K_XL.gguf}
SERVER=${Q38_SERVER:?path to llama-server — see ../BUILD.md, do NOT use \$PATH}
PORT=${Q38_PORT:-8278}

# --- speculative decoding ----------------------------------------------------
# Not optional: 11.9 t/s without a drafter, ~32 t/s with one.
#
# n_max 5, measured (5 passes per point, depth 7.5k, temp 1.0):
#   n3 = 17.22 t/s · n5 = 19.68 (+14%) · n7 = 19.6
#
# ⚠ This REFUTES an earlier note that said "n3 ≈ n5, n7/n8 refuted". That note
#   had been taken with an n-gram engine in the chain and at temp 0.6. Two
#   variables, and it stood for eleven days.
#
# The mechanism, and it is the interesting part: on a DENSE model, verifying 5
# tokens costs the SAME pass as verifying 3. A longer draft is therefore nearly
# free, and it pays even as the acceptance rate FALLS (39% at n5 against 54% at
# n3). That is the exact inverse of the MoE lanes in this repo, where the same
# knob is a memory-bandwidth trade.
#
# ⚠ --spec-draft-p-min stays at its default 0.00 — no filtering — and here that
#   is the RIGHT setting, which is the opposite of the GLM lane's conclusion.
#   Swept at 0.0 / 0.5 / 0.7 / 0.9, throughput is MONOTONICALLY DECREASING
#   (16.5 / 15.7 / 16.0 / 14.5) while acceptance climbs from 50% to 93%.
#   Filtering throws away drafts that would have been accepted, and on a dense
#   model a rejection costs nothing.
#   Acceptance going up while throughput goes down is the whole lesson.
SPEC=()
if [ -n "${Q38_DRAFT:-}" ]; then
  SPEC+=(--spec-type draft-dflash -md "$Q38_DRAFT"
         --spec-draft-n-max "${Q38_NMAX:-5}" -ngld "${Q38_NGLD:-99}"
         --poll 100 --spec-draft-poll 1)
fi
# ⚠ ngram-mod is NOT in the chain: −18% measured on one MoE lane, −40% on
#   another, two independent tests.
#
# ⚠ A drafter path that does not exist takes the whole lane down SILENTLY.
#   `-md` once pointed at a deleted file: llama-server exited on "model loading
#   error", the proxy reported "upstream command exited prematurely", and the
#   router fell through to a cloud provider without telling anyone. A lane that
#   fails to start does not return an error to the user — it returns an answer
#   from somewhere else. Check that the file exists before you check anything.

# --- batch -------------------------------------------------------------------
# -ub 256. RESULTS.md finds 192 faster at depth, but 192 + a DFlash2 drafter is a
# reproducible ROCm illegal memory access (../KNOWN_CRASH.md). 256 is the
# fastest value that does not crash this configuration.
UB=${Q38_UB:-256}

# --- context cache -----------------------------------------------------------
# ⚠ --cache-reuse is deliberately absent, because it is INERT on this
#   architecture and says nothing about it. qwen35/qwen35moe return
#   LLAMA_ROPE_TYPE_IMROPE unconditionally => n_pos_per_embd() == 4 =>
#   get_can_shift() == false => llama.cpp disables cache_reuse at load. The flag
#   only ever lied in the config file. Do not put it back.

# --- reasoning template ------------------------------------------------------
# Same empty-`medium`-branch problem as Qwen3.8-Flash-Next: the stock template
# accepts `medium`, has no branch for it, and injects nothing. Pass a template
# file that fills the branch, or the kwargs below does nothing at all.
TMPL=()
[ -n "${Q38_TMPL:-}" ] && TMPL=(--chat-template-file "$Q38_TMPL")

# --- sampling ----------------------------------------------------------------
# The publisher's preset, read from the GGUF actually served (`general.sampling`):
# temp 1.0, top_p 0.95, top_k 20. An earlier --temp 0.6 here was a house setting,
# not theirs. min_p 0.0 is set EXPLICITLY to overwrite llama.cpp's default 0.05,
# which the authors do not recommend.

MMPROJ=()
[ "${Q38_MMPROJ:-0}" != 0 ] && MMPROJ=(--mmproj "$Q38_MMPROJ")

exec env HSA_USE_SVM=0 ROCBLAS_USE_HIPBLASLT=1 "$SERVER" \
  --host 127.0.0.1 --port "$PORT" \
  -m "$MODEL" \
  -ngl 99 -fa on --jinja \
  "${TMPL[@]}" \
  --chat-template-kwargs "{\"reasoning_effort\":\"${Q38_EFFORT:-medium}\"}" \
  "${MMPROJ[@]}" \
  --parallel "${Q38_NP:-4}" --kv-unified --no-cache-idle-slots \
  --slot-prompt-similarity "${Q38_SPS:-0.9}" \
  -c "${Q38_CTX:-200000}" -b "${Q38_B:-2048}" -ub "$UB" \
  --metrics \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --alias "${Q38_ALIAS:-Qwen3.8-27B-Dense}" \
  "${SPEC[@]}"
