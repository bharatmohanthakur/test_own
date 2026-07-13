---
name: no-mamba-source-build
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: pip\s+install.*--no-build-isolation.*(mamba_ssm|causal_conv1d)
  - field: command
    operator: not_contains
    pattern: nemotron-pkgs
action: block
---

🛑 **Building mamba_ssm / causal_conv1d from source.**

This takes ~10 min and burns $0.20+. The `mayukh18/nemotron-packages` Kaggle dataset has prebuilt wheels:
- `causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- `mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`

**Install instead:**
```bash
kaggle datasets download mayukh18/nemotron-packages -p /workspace/nemotron-pkgs --unzip
pip install /workspace/nemotron-pkgs/mamba_ssm-*.whl /workspace/nemotron-pkgs/causal_conv1d-*.whl
```

**Note**: those wheels are torch 2.10 ABI. If pod has torch 2.9, the wheel will fail. In that case ONLY, source build is needed. Verify torch version first — if it's already 2.10, use wheels. If 2.9, append `# torch29-source-needed` to bypass.
