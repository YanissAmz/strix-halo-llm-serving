#!/usr/bin/env bash
# Paired duel over N arms of ONE binary, the arms differing only by environment
# variables. Arms ALTERNATE over two rounds, so a slow drift (thermals, page
# cache) hits every arm equally instead of favouring whichever ran first.
#
#   ./paired-duel.sh <tag> <launcher> <arm...>
#   arm = "name:VAR=val[,VAR=val...]"   ("name:" alone = an arm with no variable)
#
# What this captures BEYOND tokens/s, and why each one earns its place:
#
#   - the EVICTION COUNTER delta per leg. An arm that evicts pays a cost that has
#     nothing to do with the lever under test. If the counters differ between
#     arms, the tg comparison is void and must be reported as void, not
#     arbitrated.
#   - draft acceptance and mean draft length, to be read BEFORE the tg number.
#   - GTT and MemAvailable at the start and end of every leg.
#
# ⚠ This kills llama-server processes it started. It does NOT kill anything
#   else — an earlier version killed every llama-server on the box and took out
#   two unrelated validation runs. If you want the old blunt behaviour for a
#   dedicated bench box, set DUEL_KILL_ALL=1 and know what you are asking for.
set -u

TAG="${1:?tag}"; LAUNCH="${2:?launcher}"; shift 2
ARMS=("$@")
OUTDIR="${DUEL_OUTDIR:-.}"
OUT="$OUTDIR/duel-$TAG.tsv"
JRN="$OUTDIR/duel-$TAG.journal"
PORT="${DUEL_PORT:-8289}"
# The measurement itself is pluggable: give it a command that talks to $PORT and
# prints one TSV row per sample on stdout.
BENCH="${DUEL_BENCH_CMD:?set DUEL_BENCH_CMD to your bench command, it receives PORT and RUNID}"

: > "$OUT"; : > "$JRN"
ev(){  journalctl -b 0 -k 2>/dev/null | grep -cE 'restore_userptr|svm_range_restore'; }
gtt(){ awk '{printf "%.1f",$1/1073741824}' /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null || echo NA; }
mem(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
say(){ echo "$*" | tee -a "$JRN"; }

PIDS=()
reap(){
  if [ "${DUEL_KILL_ALL:-0}" = 1 ]; then
    ps -eo pid,comm --no-headers | awk '$2 ~ /^llama-server/ {print $1}' | xargs -r kill 2>/dev/null
  else
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  fi
  sleep 5
  if [ "${DUEL_KILL_ALL:-0}" = 1 ]; then
    ps -eo pid,comm --no-headers | awk '$2 ~ /^llama-server/ {print $1}' | xargs -r kill -9 2>/dev/null
  else
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  fi
  PIDS=()
}
trap reap EXIT

say "### duel $TAG started $(date '+%F %H:%M:%S')  evictions at start: $(ev)"

for round in 1 2; do
for spec in "${ARMS[@]}"; do
  arm="${spec%%:*}"; vars="${spec#*:}"
  ENVV=(); [ -n "$vars" ] && IFS=',' read -r -a ENVV <<< "$vars"

  reap
  # A fixed `sleep 15` is NOT enough. One duel lost all four legs to
  # "failed to allocate ROCm0 buffer of size 102560175104": the kernel had not
  # finished returning the previous lane's ~97 GB of GTT. So wait for the memory
  # to be actually returned instead of betting on a delay:
  #   1. no llama-server process left (at most 120 s),
  #   2. MemAvailable stops rising for 3 consecutive samples (at most 180 s).
  # Self-calibrating: no hard-coded threshold, so it holds for a 101 GB lane as
  # well as a 97 GB one.
  for _ in $(seq 1 60); do
    pgrep -x llama-server >/dev/null || break
    sleep 2
  done
  _prev=0; _stable=0
  for _ in $(seq 1 90); do
    _now=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$_now" -le "$_prev" ]; then _stable=$((_stable+1)); else _stable=0; fi
    [ "$_stable" -ge 3 ] && break
    _prev=$_now; sleep 2
  done
  say "### memory returned before $arm r$round : $(mem) GB"

  LOG="$OUTDIR/duel-$TAG-$arm-r$round.log"
  E0=$(ev)
  nohup env "${ENVV[@]:-IGNORE=1}" "$LAUNCH" > "$LOG" 2>&1 &
  PIDS+=("$!")

  ok=0
  for _ in $(seq 1 60); do
    [ "$(curl -s --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = '{"status":"ok"}' ] && { ok=1; break; }
    sleep 10
  done
  if [ "$ok" != 1 ]; then say "### $arm r$round DID NOT LOAD — leg lost, continuing"; continue; fi

  say "### $arm r$round start $(date +%H:%M:%S) env=[${vars:-none}] gtt=$(gtt)GB mem=$(mem)GB ev=$E0"
  PORT="$PORT" RUNID="$TAG-$arm-r$round" bash -c "$BENCH" \
    | sed "s/^/$arm\t$round\t/" >> "$OUT"
  E1=$(ev)

  ACC=$(grep -oE 'draft acceptance = [0-9.]+.*mean len = *[0-9.]+' "$LOG" | tail -1)
  say "### $arm r$round end   $(date +%H:%M:%S) gtt=$(gtt)GB mem=$(mem)GB ev=$E1 (delta $((E1-E0)))  $ACC"
done
done

say "### END $TAG $(date '+%F %H:%M:%S')  evictions at finish: $(ev)"
