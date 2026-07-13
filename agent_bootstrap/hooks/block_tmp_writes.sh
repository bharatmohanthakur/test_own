#!/usr/bin/env bash
# PreToolUse Write hook — blocks writing scripts/data to /tmp.
# User's feedback memory: "NEVER store scripts in /tmp - use project working directory."
# Past incident: lost all previous training scripts because they were in /tmp.
set -u
payload=$(cat)
path=$(printf '%s' "$payload" | /usr/bin/python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)

if [ -z "$path" ]; then exit 0; fi

# Only block inside /tmp for persistent file types
case "$path" in
  /tmp/*.py|/tmp/*.sh|/tmp/*.jsonl|/tmp/*.md|/tmp/*.yaml|/tmp/*.yml|/tmp/*.json|/tmp/*.txt)
    echo "[block_tmp_writes] Blocked write to $path" >&2
    echo "/tmp is cleaned up by macOS and previously wiped out training scripts (feedback_no_tmp.md)." >&2
    echo "Write to /Users/bharat/Downloads/kaggle/scripts/ or a project subdir instead." >&2
    exit 2
    ;;
esac
exit 0
