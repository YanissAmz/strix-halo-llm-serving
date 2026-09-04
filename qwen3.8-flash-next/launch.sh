#!/usr/bin/env bash
# Qwen3.8-Flash-Next (arch `qwen4exp`, 176B total = 125B core + 51B n-gram table,
# 6B active), UD-Q4_K_XL-q6dense ~103 GB, on llama.cpp / Vulkan, AMD Strix Halo
# 128 GB unified memory. Serves 200704 tokens of context, with vision.
#
# Every flag below is here because a paired duel said so. The measurement behind
# each one is in RESULTS.md. Read ../BUILD.md first.
set -euo pipefail

MODEL=${Q4E_MODEL:?path to Qwen3.8-Flash-Next-q6dense-00001-of-00004.gguf}
SERVER=${Q4E_SERVER:?path to llama-server — see ../BUILD.md, do NOT use \$PATH}
PORT=${Q4E_PORT:-8278}

# --- device: resolve by NAME, never by index ---------------------------------
# A GPU that appears or disappears renumbers the Vulkan enumeration. After one
# reboot an RTX 3090 came back and took `Vulkan0` from the APU:
#   Vulkan0: NVIDIA GeForce RTX 3090             ( 24822 MiB)
#   Vulkan1: Radeon 8060S Graphics (STRIX_HALO)  (129536 MiB)
# The lane then loaded 103 GB of weights into 24 GB and failed in 1.4 s on a
# 964 MiB block, with 120 GB free right next to it. The error names the buffer
# ("Vulkan0 buffer"), never the device — which is why an empty GTT and a
# compacted memory map explained nothing.
if [ -z "${Q4E_DEVICE:-}" ]; then
  Q4E_DEVICE=$("$SERVER" --list-devices 2>/dev/null \
    | awk -F: '/STRIX_HALO|Radeon 8060S/ {gsub(/ /,"",$1); print $1; exit}')
  : "${Q4E_DEVICE:=Vulkan0}"
fi
echo "[launch] Vulkan device selected by name: $Q4E_DEVICE" >&2

# --- window ------------------------------------------------------------------
# 200704 served, 262144 native (1M with YaRN). 262144 was tried and given back
# the same day: with the window actually FULL, MemAvailable fell to 3.7 GiB and
# kept drifting — below earlyoom's threshold. It only survived on swap, and
# swapping during decode ruins throughput. Worse, llama-server is in earlyoom's
# --avoid list, so it would not be killed cleanly; it would hit the allocator
# wall mid-request.
#
# ⚠ `common_params_fit_impl` projects DEVICE memory (85.7 GB against 123.7 free
#   at 262k) and says nothing about host MemAvailable once the window is full.
#   Any increase gets re-measured by really filling the window, never from that
#   projection.
CTX=${Q4E_CTX:-200704}

# --- speculative decoding: the shared MTP head -------------------------------
# +33% decode (27.0 -> 36.0 t/s, 11 prompts, warm against warm), acceptance
# 74.2% over 3396 drafted tokens.
#
# "Shared" head: it borrows token_embd and output from the target model instead
# of carrying its own copies, so it costs ~1.3 GB less than a standalone drafter
# at the same window.
#
# ⚠ -devd is MANDATORY here and the failure is not obvious. A shared head borrows
#   tensors from the target; if the draft is scheduled on another device, ggml
#   gives up at graph reservation:
#     ggml-backend.cpp:941: pre-allocated tensor (output.weight) in a buffer
#     (Vulkan1) that cannot run the operation (NONE)
#   Without -devd the draft went to Vulkan0 (the 3090) while the target was on
#   Vulkan1 (the APU). Reuse the resolved device, do not re-derive an index.
SPEC=()
if [ -n "${Q4E_DRAFT:-}" ]; then
  SPEC+=(--spec-type "${Q4E_SPEC:-draft-mtp}"
         --spec-draft-model "$Q4E_DRAFT"
         --spec-draft-n-max "${Q4E_DRAFT_NMAX:-3}"
         -devd "$Q4E_DEVICE" -ngld "${Q4E_DRAFT_NGL:-99}")
fi
# ⚠ ngram-mod is NOT here on purpose. It measured 3x on rewriting, then measured
#   −40% and hung the lane when the benchmark was fixed. RESULTS.md §2.

# --- slots -------------------------------------------------------------------
# --parallel 1 means ONE slot for every caller. Two conversations then evict each
# other's prefix and each turn re-prefills the whole preamble: 6.7 s per switch,
# and in one observed case 26842 tokens re-prefilled for 97 s per turn while the
# lane itself had not moved.
#
# --kv-unified lets the slots share one KV pool instead of each reserving
# CTX/N. --no-cache-idle-slots frees an idle slot's cache instead of pinning it.
# --slot-prompt-similarity routes a request to the slot that already holds the
# closest prefix, which is what makes several conversations coexist.
NP=${Q4E_NP:-4}
SPS=${Q4E_SPS:-0.9}

# --- quantization ------------------------------------------------------------
# q6dense: attn_qkv / attn_gate / ssm_out / attn_q drop from Q8_0 to Q6_K, the
# experts do not move. Model 106.26 -> 102.98 GiB, decode 24.84 -> 27.00 t/s flat
# (+7.4%) and 22.26 -> 24.32 at 65k (+9.3%) — the gain GROWS with depth, because
# the dense weight per token is constant while the cache gets heavier.
# Perplexity, wikitext, 40 blocks, n_ctx 4096: 3.8725 ±0.0288 control against
# 3.8749 ±0.0288. +0.062%, twenty times under the error bar. Nothing to arbitrate.

# --- KV cache ----------------------------------------------------------------
# q8_0. Honest version: at this window f16 fits with ~38 GB to spare and q8_0
# buys no speed (27/08, three depths, everything within 2.6%). It is served
# because both boxes must serve the same recipe, and the 2.9 GiB saved is a
# bonus, not the reason.
#
# It costs no quality either: llama.cpp's Hadamard rotation for quantized KV runs
# on this architecture. Proof by perplexity, one variable: 2.6074 without the
# rotation, 2.5779 with. Two different numbers mean it is really running —
# llama-bench never prints an "attn_rot_k" line, so its absence proved nothing,
# and an earlier comment here asserting the rotation was unimplemented was
# assumed rather than measured, and wrong. It had been costing 1.14% of quality
# for free.
# The INDEXER's key cache stays f16 regardless: its keys decide which cells get
# selected at all.
KV=${Q4E_KV:-q8_0}

# --- batch -------------------------------------------------------------------
# -ub 2048. The first benchmark said 512 was better at depth and far more stable
# (218 ±126 against 278 ±5). Re-run with two repetitions, it says the opposite
# (2048: 374 flat, 337.76 ±6.03 at d65536; 512: 333 and 309 ±9). What varied was
# not the batch but the ORDER: the ±130 always landed on the FIRST leg at
# -d 65536 whatever its configuration. That is the cost of touching pages for the
# first time, paid by whoever runs first. A warm-up leg at depth now precedes the
# measurement. See ../METHOD.md.
UB=${Q4E_UB:-2048}

# --- mmap --------------------------------------------------------------------
# --no-mmap, and keep the measurement that argues the other way. Alternated A/B,
# four legs, at d16384 and d65536:
#   mmap     (1)  pp512@65k 163.98   tg32@65k 19.06
#   resident (1)  pp512@65k 165.15   tg32@65k 19.55   <- +2.6%, i.e. noise
#   mmap     (2)  pp512@65k 164.36   tg32@65k 19.12
#   resident (2)  pp512@65k  51.19   tg32@65k 15.06   <- COLLAPSE
# `free` during the fourth leg: 106 GB used, 0 GB available. The resident set
# survives one pass and then gets eaten; mmap reproduced to 0.3%. Q4E_MMAP=1 is
# where to come back if this lane slows down at depth or gets killed.
MMAP=(--no-mmap)
[ "${Q4E_MMAP:-0}" = 1 ] && MMAP=()

# --- vision ------------------------------------------------------------------
# The clip tower is not quantizable; F16 mmproj is the only option.
MMPROJ=()
[ "${Q4E_MMPROJ:-0}" != 0 ] && MMPROJ=(--mmproj "$Q4E_MMPROJ")

# --- sampling ----------------------------------------------------------------
# The publisher's generation_config.json asks for temp 1.0, top_k 20, top_p 0.95.
# top_k=20 is WANTED here and is kept. But llama.cpp also stacks min_p=0.05 that
# nobody asked for; on a ~248k vocabulary that is one more invisible truncation,
# so min_p=0 goes in explicitly.

# --- reasoning template ------------------------------------------------------
# ⚠ The GGUF's own jinja has an EMPTY `medium` branch. Line 57 sets
#   reasoning_instructions = '', lines 66 and 68 fill it for xhigh and low, and
#   there is no elif for medium. So 'medium' passes validation without raising,
#   the variable stays empty, and the injection sites emit nothing. Since this
#   model became the default it had been running with no reasoning instruction at
#   all — not the publisher's, not ours. That is not a setting, it is a box the
#   publisher declared and forgot to fill.
#   Pass a template FILE that fills the branch. Q4E_TMPL=0 to use the GGUF's.
#
# Also: `enable_thinking` via --chat-template-kwargs is deprecated (the server
# says so at startup). --reasoning on/off carries that flag now; reasoning_effort
# stays a template argument and still goes through the kwargs.
TMPL=()
[ -n "${Q4E_TMPL:-}" ] && [ "${Q4E_TMPL}" != 0 ] && TMPL=(--chat-template-file "$Q4E_TMPL")

exec "$SERVER" \
  --host 127.0.0.1 --port "$PORT" \
  -m "$MODEL" \
  --device "$Q4E_DEVICE" -ngl 99 -fa on --jinja \
  "${TMPL[@]}" \
  --reasoning-budget -1 --reasoning on \
  --chat-template-kwargs "{\"reasoning_effort\":\"${Q4E_EFFORT:-medium}\"}" \
  "${MMAP[@]}" "${MMPROJ[@]}" \
  --parallel "$NP" --kv-unified --no-cache-idle-slots \
  --slot-prompt-similarity "$SPS" \
  -c "$CTX" -b "${Q4E_B:-2048}" -ub "$UB" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --ctx-checkpoints "${Q4E_CKPT:-8}" --checkpoint-min-step "${Q4E_CKPT_STEP:-8192}" \
  --no-warmup \
  --metrics \
  --temp "${Q4E_TEMP:-1.0}" --top-p "${Q4E_TOP_P:-0.95}" \
  --top-k "${Q4E_TOP_K:-20}" --min-p "${Q4E_MIN_P:-0.0}" \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --predict "${Q4E_PREDICT:--1}" \
  --alias "${Q4E_ALIAS:-Qwen3.8-Flash-Next}" \
  "${SPEC[@]}"
