#!/usr/bin/env bash
# DeepSeek-V4-Flash-Vision-Exp (arch deepseek4, UD-IQ3_XXS, ~97 GB) on
# llama.cpp / HIP, AMD Strix Halo 128 GB unified memory.
#
# Every flag below is here because a paired duel said so. The measurement that
# justifies each one is in RESULTS.md, with its date and its window.
#
# Why llama.cpp and not a faster prefill server: vision goes through mmproj/mtmd,
# which only llama.cpp implements. Slower prefill is the price of vision.
set -euo pipefail

MODEL=${DS4V_MODEL:?path to DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf}
MMPROJ=${DS4V_MMPROJ:?path to mmproj-F16.gguf}
SERVER=${DS4V_SERVER:?path to llama-server from a build with the patches in ../BUILD.md — do NOT default to $PATH, see BUILD.md}

# --- chat template -----------------------------------------------------------
# Pass a template explicitly. Benching the GGUF's internal jinja while serving a
# different one measures a config nobody runs.
# An array, not unquoted ${VAR:+...}: unquoted expansion would render the JSON
# quotes literally.
TMPL=()
[ -n "${DS4V_TMPL:-}" ]   && TMPL+=(--chat-template-file "$DS4V_TMPL")
[ -n "${DS4V_EFFORT:-}" ] && TMPL+=(--chat-template-kwargs "{\"reasoning_effort\":\"$DS4V_EFFORT\"}")

# --- speculative decoding ----------------------------------------------------
# This lane ran with NO drafter at all until 2026-09-03. Adding one is worth
# +54.4% decode (15.57 -> 24.04 t/s, paired duel, 11 prompts, ctx 131072).
#
# The MTP head is NOT embedded in this GGUF: tensors read across all 4 shards,
# blocks 0->42, no `nextn` / `mtp` / `eh_proj`. So a free `draft-mtp` is
# impossible here, unlike GLM. An external drafter or ngram is the only option.
#
# Draft KV in q8_0, not f16: f16 costs ~9.6 GiB at 200k and would leave the box
# with ~0.2 GiB free. That is a memory fact, not a quality preference.
SPEC=()
if [ -n "${DS4V_SPEC:-}" ] && [ "${DS4V_SPEC}" != "none" ]; then
  SPEC+=(--spec-type "${DS4V_SPEC}")
  [ -n "${DS4V_DRAFT:-}" ] && SPEC+=(--spec-draft-model "${DS4V_DRAFT}")
  SPEC+=(--spec-draft-n-max "${DS4V_NMAX:-3}")
  SPEC+=(-ctkd "${DS4V_DKV:-q8_0}" -ctvd "${DS4V_DKV:-q8_0}")
  [ -n "${DS4V_PMIN:-}" ] && SPEC+=(--spec-draft-p-min "${DS4V_PMIN}")
fi

# --- kernel env --------------------------------------------------------------
# DSV4_FA_ROW_GATHER is a PREFILL lever: the kernel visits only the union of rows
# the mask lets through, per Q tile, and it is not gated on nt==1
# (ggml/src/ggml-cuda/fattn-common.cuh), so it acts during prefill where
# DSV4_FA_COMPACT (gated on nt == 1) cannot.
#   Test it BY VALUE. `=0` really disables it — several of these env flags are
#   read by presence, so `=0` and "unset" are not the same experiment.
ENVX=()
[ "${DS4V_GATHER:-1}"  = 1 ] && ENVX+=(DSV4_FA_ROW_GATHER=1)
[ "${DS4V_COMPACT:-0}" = 1 ] && ENVX+=(DSV4_FA_COMPACT=1)

exec env "${ENVX[@]}" "$SERVER" \
  --host 127.0.0.1 --port "${DS4V_PORT:-8289}" \
  -m "$MODEL" --mmproj "$MMPROJ" \
  --device ROCm0 -ngl 99 -fa on --jinja \
  "${TMPL[@]}" "${SPEC[@]}" \
  --parallel "${DS4V_NP:-4}" --kv-unified --no-cache-idle-slots \
  --slot-prompt-similarity "${DS4V_SPS:-0.9}" \
  -c "${DS4V_CTX:-131072}" -b "${DS4V_UB:-2048}" -ub "${DS4V_UB:-2048}" \
  -ctk "${DS4V_KV:-q8_0}" -ctv "${DS4V_KV:-q8_0}" \
  --ctx-checkpoints "${DS4V_CKPT:-8}" \
  --cache-reuse 256 \
  --no-mmap --no-warmup --metrics \
  --temp "${DS4V_TEMP:-1.0}" --top-p "${DS4V_TOP_P:-1.0}" \
  --top-k "${DS4V_TOP_K:-0}" --min-p "${DS4V_MIN_P:-0.0}" \
  --alias DeepSeek-V4-Flash-Vision-Exp
