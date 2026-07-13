---
name: runpod_workflow
description: Full RunPod workflow for Nemotron GRPO/GSPO training when Kaggle quota is exhausted — image, deps, mayukh18 wheels, SCP, direct Kaggle dataset push, CPU-only Kaggle submission. Reference: notes/runpod-grpo-workflow.md
type: reference
---

# RunPod Nemotron Training (validated Apr 8, 2026)

## Pod choice

- **GPU**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` ($1.69/hr) — matches Kaggle's RTX Pro 6000 architecture exactly so all v28 monkey-patches work as-is. High stock. NOT H200 NVL ($0.50/hr but Low stock + different arch).
- **Image**: `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404` — Ubuntu 24.04 has Python 3.12 by default, which matches the cp312 wheels in mayukh18/nemotron-packages. Has SSH server ready.
- **DO NOT use** `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` — official PyTorch image has no sshd, pod hangs indefinitely.
- **DO NOT use** `runpod/pytorch:2.10.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu24.04` — that tag doesn't exist on Docker Hub, pod hangs.
- **Disk**: 100 GB containerDisk, /workspace mounted on RunPod's network FS (per-user quota that bites if you fill it).

## Pod creation (GraphQL)

```python
import requests
headers = {'Authorization': 'Bearer YOUR_RUNPOD_KEY', 'Content-Type': 'application/json'}
with open('/Users/bharat/.ssh/id_rsa.pub') as f:
    pubkey = f.read().strip()
mutation = '''mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id desiredStatus }
}'''
variables = {"input": {
    "cloudType": "ALL", "gpuCount": 1, "volumeInGb": 100, "containerDiskInGb": 100,
    "minVcpuCount": 8, "minMemoryInGb": 64,
    "gpuTypeId": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "name": "nemotron-grpo",
    "imageName": "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404",
    "ports": "22/tcp,8888/http", "volumeMountPath": "/workspace",
    "env": [{"key": "PUBLIC_KEY", "value": pubkey}]
}}
```

`PUBLIC_KEY` env var is required for sshd to set up authorized_keys. DO NOT pass custom `dockerArgs` — that overrides the default startup script that does the SSH setup.

## CRITICAL: Don't SCP large files from local — download directly on the pod

**Validated Apr 8, 2026 (the second time)**: SCP from local Mac → RunPod is bandwidth-limited (~80 KB/s effective on slow upload links). 3.3 GB adapter takes 30-50 min. The pod has fast internet — ALWAYS download directly on the pod.

For Kaggle dataset downloads: `kaggle datasets download` works fine on the pod.

For Kaggle KERNEL output downloads (adapters that live as a kernel output, not a dataset): the kaggle CLI has a server-side 0-byte bug that hits both macOS and Linux. Use the **URL trick**:

```python
import json, requests
with open("/root/.kaggle/kaggle.json") as f:
    c = json.load(f)
auth = (c["username"], c["key"])
r = requests.get("https://www.kaggle.com/api/v1/kernels/output",
                 params={"userName": "bharatmohan", "kernelSlug": "nemotron-sft-v22"},
                 auth=auth)
target = next((f for f in r.json()["files"]
               if f["fileName"] == "nemotron-lora-adapter/adapter_model.safetensors"), None)
resp = requests.get(target["url"], stream=True)
with open("/workspace/adapter_v22/adapter_model.safetensors", "wb") as fh:
    for chunk in resp.iter_content(chunk_size=8*1024*1024):
        if chunk:
            fh.write(chunk)
```

3.3 GB downloads in ~3 min on pod — vs 30+ min via SCP from local.

## Setup steps on the pod (after SSH works)

1. **Upgrade torch** to 2.10:
   ```
   pip install -q --upgrade --break-system-packages "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
   ```
2. **Download mayukh18/nemotron-packages directly on the pod** (NOT local-then-SCP — saves 5.5GB transfer):
   ```
   mkdir -p /root/.kaggle && echo '{"username":"USER","key":"KEY"}' > /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json
   pip install -q --break-system-packages kaggle
   kaggle datasets download mayukh18/nemotron-packages -p /workspace/nemotron-pkgs --unzip
   ```
3. **Install all the wheels** (NOT pip install from pypi — those will conflict):
   ```
   PKG=/workspace/nemotron-pkgs
   pip install -q --break-system-packages --no-index --find-links $PKG/packages \
       unsloth trl peft transformers datasets accelerate bitsandbytes
   pip install -q --break-system-packages $PKG/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
   pip install -q --break-system-packages $PKG/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
   ```
4. **Download Nemotron-3-Nano-30B from HF** (~60 GB, takes ~2 min with hf_transfer):
   ```python
   import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", local_dir="/workspace/model", max_workers=8)
   ```

## Disk quota gotcha

`/workspace` is on RunPod's MooseFS network volume with a **per-user quota** that you WILL hit during training (writing checkpoints, tmp files). Symptoms: `cp` and `python json.dump` fail with `Errno 122 Disk quota exceeded`. Mitigation:
- Stage final adapter copies on the **container disk** (`/root/work`) instead of `/workspace`
- Delete old checkpoints aggressively (`rm -rf /workspace/output/grpo_output/checkpoint-{2,4,6}` once you have checkpoint-10)
- The container disk has 100 GB and is unaffected by the network FS quota
- **NEVER write to a /workspace file from python while disk is full** — `json.dump` opens the file in 'w' mode and truncates it to 0 bytes BEFORE failing, irreversibly destroying the original

## Recovering when post-train save breaks

trainer.save_model() at script end can fail without raising. If `output/nemotron-lora-adapter/adapter_config.json` is 0 bytes after training:
- The trainer's auto-save still wrote `output/grpo_output/checkpoint-N/` with the FULL valid adapter
- Recover by copying `adapter_config.json` (1.2 KB) and `adapter_model.safetensors` (3.3 GB) from the highest checkpoint dir
- If the LATEST checkpoint config was also corrupted by my python re-write attempt, copy from the previous checkpoint (they all have identical config — same LoRA architecture)

## Push adapter directly from pod to Kaggle (no local hop)

```
mkdir -p /root/v31_dataset
cp output/grpo_output/checkpoint-10/adapter_config.json /root/v31_dataset/
cp output/grpo_output/checkpoint-10/adapter_model.safetensors /root/v31_dataset/
cat > /root/v31_dataset/dataset-metadata.json << EOF
{"title":"Nemotron V31 GRPO Adapter","id":"bharatmohan/nemotron-v31-grpo-adapter","licenses":[{"name":"CC0-1.0"}]}
EOF
cd /root/v31_dataset && kaggle datasets create -p .
```
3.3 GB uploads in ~28 sec from RunPod's network. Saves the local download+upload roundtrip entirely.

## CPU-only Kaggle submission notebook

Skip GPU quota completely. The submission flow only needs to zip the adapter files. `enable_gpu: false` in kernel-metadata.json. Notebook just walks `/kaggle/input/`, finds adapter_config.json + adapter_model.safetensors, copies them out, builds submission.zip. Local templates: `notebooks/submit_v31/`, `notebooks/submit_v32/`. Push, wait for COMPLETE, then submit via Playwright UI (Submit to Competition button).

## Cost reality check

- v31 GRPO 10 steps × ~12 min/step on RTX Pro 6000 Blackwell = ~120 min training = ~$3.40
- Plus model download (~3 min), wheel install (~2 min), idle (~10 min for setup mistakes) = ~$1.50 overhead
- Plus a failed first pod with wrong image = ~$1
- **Total per full v31 run: ~$5-7**
- Per cheaper v32 GSPO run on the same pod (no re-download): ~$0.85 for 2 steps × 12 min

## Validated Apr 8, 2026
- v31: 10 GRPO steps × 12 min = 121 min total. checkpoint-10 has full adapter.
- v32: 2 GSPO steps × 12 min = 24 min total. Same pod, loads checkpoint-10.
- Both adapters pushed to Kaggle datasets `bharatmohan/nemotron-v31-grpo-adapter` and `bharatmohan/nemotron-v32-gspo-adapter`.
- Both submitted via CPU-only kernels — zero Kaggle GPU quota burned for submission.
