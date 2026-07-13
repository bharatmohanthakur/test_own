---
name: Komil GRPO fix BLOCKED on Kaggle
description: Komil's transformers 5.3+ native NemotronH works on RunPod but NOT on Kaggle. Dependency chain (huggingface-hub, trl, peft) conflicts with mayukh18 wheels. Use proven Unsloth + TRL 0.24 path on Kaggle instead.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
## Komil fix on Kaggle: BLOCKED after 10 attempts

Errors encountered (v43 v1-v10):
1. P100 sm_60 (expected, need UI save)
2. trust_remote_code required (Kaggle model has custom code)
3. Stripped ALL .py files → deleted tokenizer parser
4. Stripped only modeling_.py → config.json still has auto_map
5. Stripped auto_map + set model_type → transformers 4.57 doesn't know nemotron_h
6. Installed transformers 5.5.3 wheel → mayukh18 overrode it back to 4.57
7. Fixed install order (mayukh18 first, then 5.5.3) → huggingface-hub 1.4.1 too old (needs 1.5.0+)

**Root cause:** Kaggle has NO internet on RTX Pro 6000 batch mode for this competition. Can't pip install missing deps. Need a complete wheel set: transformers 5.5.3 + trl 1.1.0 + huggingface-hub 1.5.0 + all transitive deps. Too many to manage.

**Working alternative:** Unsloth + TRL 0.24 + mayukh18 wheels + trust_remote_code=True + v28 monkey-patch. Slow (12 min/step) but all deps are pre-packaged and compatible.

**Why:** Komil's fix saves 20x on RunPod (27s vs 12min per step) but the dep chain makes it impractical on Kaggle without building a full offline wheel dataset.
**How to apply:** Use Komil fix ONLY on RunPod. On Kaggle, use proven Unsloth path. Script at `runpod_v31/train_runpod.py`.
