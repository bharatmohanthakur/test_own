#!/usr/bin/env bash
# PreToolUse Bash hook — blocks vastai/runpod destroy-class commands.
# User's feedback memory: "Never destroy compute pods without explicit permission."
# Bypass: set env KAGGLE_ALLOW_DESTROY=1 for this session when truly intended.
set -u
payload=$(cat)
cmd=$(printf '%s' "$payload" | /usr/bin/python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("tool_input",{}).get("command",""))' 2>/dev/null)

if [ -z "$cmd" ]; then exit 0; fi

if [ "${KAGGLE_ALLOW_DESTROY:-0}" = "1" ]; then exit 0; fi

# Patterns: vastai destroy, runpodctl remove pod, runpod ... terminate
if echo "$cmd" | grep -Eq '(^|[^a-z_])vastai[[:space:]]+destroy([[:space:]]|$)|runpodctl[[:space:]]+(remove|stop)[[:space:]]+pod|runpod[[:space:]]+.*--terminate|runpod[[:space:]]+.*terminate[[:space:]]+pod'; then
  echo "[block_destroy] Blocked destroy-class command:"  >&2
  echo "  $cmd" >&2
  echo "Feedback memory (feedback_pod_destroy.md): never destroy compute pods without explicit user approval — re-setup is 15-30 min + re-upload cost." >&2
  echo "If you truly intend to destroy: ask user for confirmation, then rerun with KAGGLE_ALLOW_DESTROY=1 prefix, OR use 'vastai stop' / 'runpodctl stop pod' instead." >&2
  exit 2
fi
exit 0
