---
name: bench-on-kaggle-not-vast
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: ssh\s+.*python.*bench_vllm
  - field: command
    operator: not_contains
    pattern: kaggle-bench-checked
action: block
---

🛑 **Running bench on a vast.ai pod.**

Skill `bench_vllm_workflow.md` says: bench on **Kaggle** (validated 20-min run, no driver issues, no cost). Today's $5 burn was a vast.ai bench that hit driver 570 PTX failure.

**Use the Kaggle bench kernel instead:**
```bash
kaggle kernels push -p notebooks/bench_v28_v26_vllm --accelerator NvidiaRtxPro6000
```

Add the new adapter as a `dataset_sources` entry. Run on Kaggle's GPU. Free + reliable.

If you've confirmed the vast.ai pod has driver ≥580 AND the Kaggle bench kernel is unavailable for some reason, append `# kaggle-bench-checked` to bypass.
