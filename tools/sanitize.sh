#!/usr/bin/env bash
# Rewrite a private launcher/config into the publishable form kept in this repo.
#
# The point is that regenerating after a local change cannot silently re-leak
# host detail. Run it, diff the result, commit. Never hand-copy a launcher.
#
#   PRIVATE_NAMES='alice|bob' tools/sanitize.sh \
#       < ~/.config/llama-swap/config.yaml > /tmp/config.pub.yaml
#
# PRIVATE_NAMES: extra regex alternation of hostnames or people to redact. Keep
# it out of this file — a redaction list committed to the repo publishes exactly
# what it was meant to hide.
set -euo pipefail

sed -E \
  -e 's#/home/[a-z0-9_-]+#$HOME#g' \
  -e 's#[a-z0-9-]+\.tail[a-z0-9]+\.ts\.net#<tailnet-host>#g' \
  -e 's#100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}#<tailnet-ip>#g' \
  ${PRIVATE_NAMES:+-e "s#\\b($PRIVATE_NAMES)\\b#<node>#g"}
