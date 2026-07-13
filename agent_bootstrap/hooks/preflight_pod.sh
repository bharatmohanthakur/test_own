#!/usr/bin/env bash
# PreToolUse Bash hook — preflight checks before `vastai create instance`.
# Guards: balance > $5, vastai-pod skill loaded this session (best-effort),
#         data files exist locally, training script ready.
# Bypass: append `# preflight-acknowledged` to the bash command.
set -u
payload=$(cat)
cmd=$(printf '%s' "$payload" | /usr/bin/python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("tool_input",{}).get("command",""))' 2>/dev/null)

if [ -z "$cmd" ]; then exit 0; fi

# Only run for `vastai create instance` commands
if ! echo "$cmd" | grep -Eq '(^|[^a-z_])vastai[[:space:]]+create[[:space:]]+instance'; then
  exit 0
fi

# Bypass if explicitly acknowledged
if echo "$cmd" | grep -q 'preflight-acknowledged'; then exit 0; fi

ERR=()

# 1. Balance check (parse vastai show invoices)
balance=$(vastai show invoices 2>/dev/null | tail -1 | grep -oE "'credit': [0-9.]+" | grep -oE '[0-9.]+' || echo 0)
if [ -z "$balance" ]; then balance=0; fi
balance_int=${balance%.*}
if [ "${balance_int:-0}" -lt 5 ] 2>/dev/null; then
  ERR+=("BALANCE: \$$balance — need ≥\$5 to safely run training (~\$3-4/run + buffer). Top up at https://cloud.vast.ai/billing/")
fi

# 2. Training data file exists locally
if [ ! -f /Users/bharat/Downloads/kaggle/data/crypt_sft_v1_mixed.jsonl ]; then
  ERR+=("DATA: /Users/bharat/Downloads/kaggle/data/crypt_sft_v1_mixed.jsonl missing. Run scripts/build_crypt_sft_v1_data.py first.")
fi

# 3. Training script exists
if [ ! -f /Users/bharat/Downloads/kaggle/scripts/train_crypt_sft_v1.py ]; then
  ERR+=("SCRIPT: /Users/bharat/Downloads/kaggle/scripts/train_crypt_sft_v1.py missing.")
fi

# 4. Image flag check — must use runpod/pytorch:1.0.3-cu1281...
if ! echo "$cmd" | grep -q 'runpod/pytorch:1.0.3-cu1281'; then
  ERR+=("IMAGE: command should use --image runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404 (per vastai-pod skill). Other images break Blackwell or sshd.")
fi

# 5. Disk size check — need ≥150 for model + adapter + checkpoints
disk=$(echo "$cmd" | grep -oE -- '--disk[[:space:]]+[0-9]+' | grep -oE '[0-9]+' | head -1)
if [ -z "$disk" ] || [ "$disk" -lt 150 ] 2>/dev/null; then
  ERR+=("DISK: --disk should be ≥150 (60GB model + 4GB adapter + 80GB checkpoints + buffer). Got: ${disk:-none}")
fi

# 6. Driver check — vllm 0.19+ needs driver 580+ for PTX
offer_id=$(echo "$cmd" | grep -oE 'create instance [0-9]+' | grep -oE '[0-9]+')
if [ -n "$offer_id" ]; then
  drv=$(vastai search offers "id=$offer_id" --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0].get('driver_version','?') if d else '?')" 2>/dev/null)
  if [ -n "$drv" ] && [ "$drv" != "?" ]; then
    drv_major=$(echo "$drv" | cut -d. -f1)
    if [ -n "$drv_major" ] && [ "$drv_major" -lt 580 ] 2>/dev/null; then
      ERR+=("DRIVER: offer $offer_id has driver $drv. vLLM 0.19+ needs ≥580 (PTX toolchain). Pick a newer host.")
    fi
  fi
fi

if [ ${#ERR[@]} -gt 0 ]; then
  echo "[preflight_pod] Blocked vastai create — preflight failed:" >&2
  for e in "${ERR[@]}"; do echo "  - $e" >&2; done
  echo "" >&2
  echo "If all issues addressed: append '# preflight-acknowledged' to the command." >&2
  exit 2
fi

exit 0
