#!/usr/bin/env bash
# PreToolUse Bash hook — reminder before submitting to Kaggle.
# User's pattern: 4 disasters (v27, v28, v29, GRPO_B200) would have been caught by local bench.
# This hook WARNS (does not block). Claude must still respect the rule.
set -u
payload=$(cat)
cmd=$(printf '%s' "$payload" | /usr/bin/python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("tool_input",{}).get("command",""))' 2>/dev/null)

if [ -z "$cmd" ]; then exit 0; fi

# Detect submission commands
if echo "$cmd" | grep -Eq 'kaggle[[:space:]]+competitions[[:space:]]+submit|kaggle[[:space:]]+kernels[[:space:]]+push[[:space:]]+.*submit_'; then
  # Check for a recent bench result file (modified within last 4 hours)
  bench_recent=$(/usr/bin/find /Users/bharat/Downloads/kaggle/tracking -name '*.json' -mmin -240 2>/dev/null | head -1)
  if [ -z "$bench_recent" ]; then
    cat >&2 <<'EOF'
[warn_before_submit] WARNING: no local bench result in tracking/ newer than 4 hours.
Past disasters caught by local bench (v27=0.82, v28=0.67, v29=0.22, GRPO_B200=0.47).
Rule (kaggle-submit skill): NEVER submit without a local bench result newer than the adapter.
Proceeding anyway is allowed, but confirm you ran tracking/run_bench_fast.py first.
EOF
  fi
fi
exit 0
