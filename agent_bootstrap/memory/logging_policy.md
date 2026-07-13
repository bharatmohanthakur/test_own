---
name: Logging policy
description: ALWAYS use wandb for loss tracking. report_to="none" + nohup = blind training. Never again.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**MANDATORY: Use wandb for ALL training runs.**

Problem: With `report_to="none"` + `nohup`, loss values are invisible. tqdm swallows them. trainer_state.json only appears after save_steps. We ran blind on B200 and RTX PRO 6000 with zero loss visibility.

Fix for cloud GPU (Vast.ai/RunPod):
```python
import wandb
wandb.login(key="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")
wandb.init(project="nemotron-reasoning", name="RUN_NAME")

# In TrainingArguments:
report_to="wandb",
logging_steps=5,
```

Fix for Kaggle:
```python
# Try online first, fallback to offline
try:
    wandb.login(key="...", relogin=True)
    wandb.init(project="nemotron-reasoning", settings=wandb.Settings(init_timeout=120))
    report_to = "wandb"
except:
    os.environ["WANDB_MODE"] = "offline"
    report_to = "none"
```

**Why:** Wasted hours training blind on Apr 13-14. Couldn't tell if loss was converging, diverging, or flat. wandb is free and works on both Kaggle and cloud.
**How to apply:** Add wandb to EVERY training script before launching. No exceptions.
