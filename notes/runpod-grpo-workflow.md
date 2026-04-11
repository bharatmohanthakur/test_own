# RunPod Nemotron GRPO/GSPO Workflow (validated April 8, 2026)

This is the full step-by-step workflow we used to train v31 (GRPO) and v32 (GSPO) on RunPod when Kaggle's weekly GPU quota was exhausted. **Total cost: ~$7 for both runs.** Both adapters submitted to the competition without burning ANY Kaggle GPU quota.

---

## 0. Prerequisites

- RunPod account topped up (~$10 buys 2-3 full runs)
- Local SSH key at `~/.ssh/id_rsa.pub`
- Kaggle credentials at `~/.kaggle/kaggle.json`
- v22 SFT adapter downloaded locally (or push directly to a Kaggle dataset and skip the SCP)

---

## 1. Choose the right pod

```python
# Query GPU types and prices
import requests
headers = {'Authorization': 'Bearer YOUR_KEY', 'Content-Type': 'application/json'}
query = '''query GpuTypes($input: GpuLowestPriceInput) {
  gpuTypes {
    id memoryInGb
    lowestPrice(input: $input) { uninterruptablePrice stockStatus }
  }
}'''
r = requests.post('https://api.runpod.io/graphql',
    json={'query': query, 'variables': {'input': {'gpuCount': 1}}}, headers=headers)
```

**Pick: `NVIDIA RTX PRO 6000 Blackwell Server Edition`** ($1.69/hr, High stock, 96GB VRAM).

Why this and not the cheaper options:
- It MATCHES Kaggle's RTX Pro 6000 Blackwell exactly → all our v28 patches and wheels work as-is
- 96GB > the ~95GB peak we hit during GRPO
- High stock means you'll get one fast
- H200 NVL was $0.50/hr but Low stock + different sm_90 architecture = risk of recompiling everything

---

## 2. Register your SSH key BEFORE creating the pod

The `PUBLIC_KEY` env var on the pod sets up `authorized_keys`, but RunPod also stores a default key in your account that some templates use. Make sure they match:

```python
with open('/Users/bharat/.ssh/id_rsa.pub') as f:
    pubkey = f.read().strip()
mut = '''mutation UpdateUserSettings($input: UpdateUserSettingsInput!) {
  updateUserSettings(input: $input) { pubKey }
}'''
requests.post('https://api.runpod.io/graphql',
    json={'query': mut, 'variables': {'input': {'pubKey': pubkey}}}, headers=headers)
```

---

## 3. Create the pod

```python
mutation = '''mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id desiredStatus }
}'''
variables = {"input": {
    "cloudType": "ALL",
    "gpuCount": 1,
    "volumeInGb": 100,
    "containerDiskInGb": 100,
    "minVcpuCount": 8,
    "minMemoryInGb": 64,
    "gpuTypeId": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "name": "nemotron-grpo",
    "imageName": "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404",  # py3.12, has sshd
    "ports": "22/tcp,8888/http",
    "volumeMountPath": "/workspace",
    "env": [{"key": "PUBLIC_KEY", "value": pubkey}],
}}
```

**Image gotchas:**
- ❌ `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` — official PyTorch image has NO sshd, pod hangs
- ❌ `runpod/pytorch:2.10.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu24.04` — that tag doesn't exist on Docker Hub
- ❌ `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` — Python 3.11, mayukh18 wheels are cp312, won't load
- ✅ `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404` — Ubuntu 24.04 = py3.12, has sshd, CUDA 12.8.1 matches

**DO NOT pass custom `dockerArgs`** — that overrides the runpod startup script that sets up SSH from PUBLIC_KEY.

---

## 4. Wait for SSH (poll for runtime)

```python
query = '''query Pod($input: PodFilter!) {
  pod(input: $input) { runtime { ports { ip publicPort privatePort isIpPublic } } }
}'''
# Loop until runtime.ports has the privatePort=22 entry with isIpPublic=true
# Then wait another 30-60 sec for sshd to actually accept connections
```

Test SSH:
```
ssh -o StrictHostKeyChecking=no -p PORT root@IP 'echo OK; nvidia-smi'
```

---

## 5. Bootstrap the environment ON THE POD (NOT local-then-SCP)

The mayukh18 dataset is 5.5 GB. Downloading directly on the pod takes 50 sec on RunPod's network. Doing it locally and SCPing would take 5+ min and waste bandwidth.

```bash
# 1. Upgrade torch to 2.10 (image has 2.9)
pip install -q --upgrade --break-system-packages "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128

# 2. Set up Kaggle creds and download mayukh18 wheels
mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json << EOF
{"username":"YOUR_USER","key":"YOUR_KEY"}
EOF
chmod 600 /root/.kaggle/kaggle.json
pip install -q --break-system-packages kaggle
kaggle datasets download mayukh18/nemotron-packages -p /workspace/nemotron-pkgs --unzip

# 3. Install ALL deps from the offline wheel mirror (NOT pypi, NOT pip install --upgrade)
PKG=/workspace/nemotron-pkgs
pip install -q --break-system-packages --no-index --find-links $PKG/packages \
    unsloth trl peft transformers datasets accelerate bitsandbytes
pip install -q --break-system-packages $PKG/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install -q --break-system-packages $PKG/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 4. Verify
python3 -c "import unsloth, trl, mamba_ssm, causal_conv1d; print('all OK')"
# Expected output: 'all OK'

# 5. Download Nemotron-3-Nano-30B (takes 2-3 min with hf_transfer)
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/model
python3 -c "
import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', local_dir='/workspace/model', max_workers=8)
"
```

---

## 6. Transfer your training inputs (small files) via SCP

```bash
# Stage everything locally first
mkdir -p staging/{adapter_v22,data}
cp /Users/bharat/Downloads/kaggle/adapters/v22_real/adapter_*.* staging/adapter_v22/
cp /Users/bharat/Downloads/kaggle/data/training_v31/grpo_goldilocks.jsonl staging/data/
cp train_runpod.py staging/

# Pre-create dirs on pod
ssh -o StrictHostKeyChecking=no -p PORT root@IP 'mkdir -p /workspace/{adapter_v22,data,output}'

# SCP small files first (script + data + config)
scp -o StrictHostKeyChecking=no -P PORT \
  staging/train_runpod.py staging/adapter_v22/adapter_config.json staging/data/grpo_goldilocks.jsonl \
  root@IP:/workspace/

# Then the BIG one (3.3 GB v22 adapter) in background
scp -o StrictHostKeyChecking=no -P PORT \
  staging/adapter_v22/adapter_model.safetensors \
  root@IP:/workspace/adapter_v22/  # ~3 min
```

---

## 7. Train

```bash
ssh -p PORT root@IP 'cd /workspace && nohup python3 train_runpod.py > train.log 2>&1 & echo "PID: $!"'
```

Monitor:
```bash
ssh -p PORT root@IP 'tr "\r" "\n" < /workspace/train.log | grep -E "[0-9]+/10" | tail -3; ls /workspace/output/grpo_output/ | grep checkpoint; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

**Tight config** (validated to fit ~95GB on 96GB RTX Pro 6000 Blackwell):
- max_steps=10, save_steps=2 (NOT 50/10 — v28 OOMed at step 9)
- num_generations=2, max_completion_length=512
- per_device_train_batch_size=1, gradient_accumulation_steps=4
- VRAM grows step 1→3 then plateaus at ~94.5GB
- ~12 min/step

---

## 8. Disk quota gotcha

`/workspace` is on RunPod's MooseFS volume with a **per-user quota**. Symptoms:
```
cp: failed to close 'file': Disk quota exceeded
OSError: [Errno 122] Disk quota exceeded
```

**CRITICAL danger**: Python's `with open(path, "w") as f: json.dump(...)` opens the file in 'w' mode which **truncates the file to 0 bytes BEFORE writing**. If `json.dump` then crashes (with quota exceeded or anything else), you've **destroyed the original file irrecoverably**. This is exactly how I corrupted `checkpoint-10/adapter_config.json` mid-script.

**Mitigations:**
1. Stage final adapters on the **container disk** `/root/work` (100 GB, no MooseFS quota)
2. Delete old checkpoints aggressively (`rm -rf /workspace/output/grpo_output/checkpoint-{2,4,6}` once you have higher checkpoints)
3. NEVER do destructive file writes when quota is tight — use `tmpfile + os.rename(tmpfile, target)` instead of `open(target, "w")`
4. Delete `/workspace/nemotron-pkgs` after install (5.5 GB freed)

---

## 9. Recover when post-train save fails

If `model.save_pretrained(ADAPTER_OUT)` at script end leaves `adapter_config.json` as 0 bytes:
- The trainer's auto-save still wrote `output/grpo_output/checkpoint-N/` with the FULL valid adapter
- Copy from the highest checkpoint dir
- If that dir's config is also corrupted (e.g., my mistake), fall back to the previous checkpoint — they all have IDENTICAL config (same LoRA architecture)

```bash
# Recover
cp /workspace/output/grpo_output/checkpoint-8/adapter_config.json \
   /workspace/output/grpo_output/checkpoint-10/adapter_config.json
```

---

## 10. Push adapter directly from pod to Kaggle (NO local hop)

```bash
mkdir -p /root/v31_dataset
cp /workspace/output/grpo_output/checkpoint-10/adapter_config.json /root/v31_dataset/
cp /workspace/output/grpo_output/checkpoint-10/adapter_model.safetensors /root/v31_dataset/
cat > /root/v31_dataset/dataset-metadata.json << EOF
{"title":"Nemotron V31 GRPO Adapter","id":"bharatmohan/nemotron-v31-grpo-adapter","licenses":[{"name":"CC0-1.0"}]}
EOF
cd /root/v31_dataset && kaggle datasets create -p .
```

3.3 GB uploads in **27 sec** from RunPod's network. Saves the local download+upload roundtrip entirely.

---

## 11. CPU-only Kaggle submission notebook (zero GPU quota)

The submission flow needs only to wrap the adapter files into `submission.zip` and click "Submit to Competition". No GPU needed at all.

`notebooks/submit_v31/kernel-metadata.json`:
```json
{
  "id": "bharatmohan/nemotron-submit-v31",
  "title": "Nemotron Submit v31",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": ["bharatmohan/nemotron-v31-grpo-adapter"],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]
}
```

`notebooks/submit_v31/train.py` walks `/kaggle/input/`, finds `adapter_config.json` + `adapter_model.safetensors`, copies to `/kaggle/working/nemotron-lora-adapter/`, patches config (`inference_mode=True`), zips into `submission.zip`.

```bash
kaggle kernels push -p notebooks/submit_v31
# Wait for status COMPLETE (~30 sec to 2 min)
kaggle kernels status bharatmohan/nemotron-submit-v31
# When COMPLETE, submit via Playwright UI (Output tab → Submit to Competition button)
```

**Caveat: dataset propagation lag.** After `kaggle datasets create`, the dataset takes ~1-2 min to be queryable as a kernel input. First `kernel push` may fail with "not valid dataset sources". Wait 60-90 sec and re-push.

---

## 12. Submit via Playwright (kaggle CLI doesn't support this)

```javascript
// Navigate to the kernel output page
// Click "Submit to Competition" button
// Click "Submit" in the modal dialog
// Page redirects to /submissions
```

Kaggle CLI has no submit-from-kernel command, only `kaggle competitions submit -f file.zip` (which requires the local file). Since our submission.zip is on Kaggle's side and the local download is broken (0-byte safetensors bug), we MUST use the UI.

---

## 13. Terminate the pod

```python
mut = '''mutation TerminatePod($input: PodTerminateInput!) { podTerminate(input: $input) }'''
requests.post('https://api.runpod.io/graphql',
    json={'query': mut, 'variables': {'input': {'podId': POD_ID}}}, headers=headers)
```

Verify all adapters are on Kaggle as datasets BEFORE terminating — once the pod is gone you can't re-extract from it.

---

## Cost summary (Apr 8, 2026 actual)

| Phase | Time | Cost |
|---|---|---|
| Wrong image attempts (terminated early) | ~30 min | ~$0.70 |
| Pod setup + model download | ~5 min | ~$0.15 |
| v31 GRPO 10 steps × 12 min | ~120 min | ~$3.40 |
| v32 GSPO 2 steps × 12 min (same pod) | ~24 min | ~$0.70 |
| Idle time (debugging save bugs) | ~30 min | ~$0.85 |
| Storage / network egress | — | ~$0.50 |
| **Total** | **~3.5 hr** | **~$6.30** |

Started with $10, ended with $2.93. Total spent: **$7.07**.

Two more iterations possible with the remaining $2.93 (if you skip setup overhead by reusing a paused pod template).

---

## Validated outcomes

- v31 GRPO adapter: 3.3 GB → `bharatmohan/nemotron-v31-grpo-adapter` → submitted
- v32 GSPO adapter: 3.3 GB → `bharatmohan/nemotron-v32-gspo-adapter` → submitted
- Zero Kaggle GPU quota burned for submission (CPU-only kernels)
- Adapters cleanly recovered from checkpoint-10 / checkpoint-2 even when post-train save broke
