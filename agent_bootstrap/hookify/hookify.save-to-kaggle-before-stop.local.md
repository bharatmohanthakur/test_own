---
name: save-to-kaggle-before-stop
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: vastai\s+stop\s+instance
  - field: command
    operator: not_contains
    pattern: copy-verified
  - field: command
    operator: not_contains
    pattern: kaggle-uploaded
action: block
---

🛑 **About to stop a pod that may have unsaved training output.**

Skill `vastai-pod` "Sequential workflow":
> train on pod A → **save adapter to Kaggle dataset** → `vastai stop` pod A → spin up pod B → download from Kaggle

**Before stopping a training pod, ensure ONE of:**
1. Adapters uploaded to Kaggle dataset (proven, recoverable from anywhere)
2. `vastai copy` to destination pod **completed and verified** (not just "initiated")
3. Pod's data is genuinely disposable

**Verify uploads:**
```bash
ssh <pod> "kaggle datasets create -p /workspace/adapters/<v>" # then verify on Kaggle
# OR
ssh <dst> "du -sh /workspace/adapters" # must match expected size
```

Yesterday: stopped Tennessee with `vastai copy initiated` but not completed → adapters stranded → $1.50 idle on Spain.

If verified, append `# copy-verified` OR `# kaggle-uploaded` to bypass.
