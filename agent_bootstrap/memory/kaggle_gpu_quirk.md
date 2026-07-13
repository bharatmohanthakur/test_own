---
name: Kaggle CLI push resets accelerator to P100
description: Every `kaggle kernels push` resets GPU to P100. Must re-select RTX Pro 6000 via UI AFTER every push, BEFORE Save Version.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**UPDATE 2026-06-10:** P100 is now FULLY UNUSABLE for torch: Kaggle's current PyTorch build supports sm_70+ only, P100 is sm_60 → every CUDA op crashes ("Tesla P100 ... not compatible with the current PyTorch installation"). So CLI-pushed GPU kernels (which default to P100) cannot run any torch model at all. Workarounds: (a) select T4 x2 / other accelerator in UI after every push (flow below), or (b) code a CPU fallback: check `torch.cuda.get_device_capability(0)` against `torch.cuda.get_arch_list()` and use device "cpu" if unsupported (see arc_agi_3 `llm_backends.py`).

**Rule:** After EVERY `kaggle kernels push`, the Accelerator setting reverts to P100 regardless of what was selected before. You MUST:
1. Open the edit URL in browser
2. Settings → Accelerator → GPU RTX Pro 6000
3. Confirm "Turn on GPU RTX Pro 6000"
4. Only THEN click Save Version → Save

Otherwise the batch run starts on P100 (16 GB) which can't load 30B model.

**Why:** Kaggle CLI metadata doesn't persist the accelerator choice — only `enable_gpu: true` which defaults to P100. The accelerator field is a session-level UI setting.

**Verification via Playwright:** After opening edit page, click Settings → Accelerator. Check which item has `hasCheck=true`. If P100, select RTX Pro 6000 and confirm dialog.

**Evidence of the trap (Apr 18, 2026):**
- v29 training kernel pushed via CLI at T+0
- I saved Version with UI setting RTX 6000 → batch ran on P100 anyway (CLI metadata overrode UI)
- v29 re-pushed via CLI → UI accelerator reset back to None/P100 default
- Had to re-select RTX Pro 6000 in UI AGAIN before second Save Version
- Only the second Save (without CLI push in between) ran on RTX 6000

**Workflow that actually works:**
```
1. CLI push kernel
2. Kaggle starts P100 batch (will error for 30B models)  
3. Open /edit URL
4. Settings → Accelerator → GPU RTX Pro 6000 → confirm
5. Verify hasCheck=true in Accelerator dropdown
6. Escape menus
7. Save Version → Save (NO CLI push between steps 4-7)
```

## UPDATE Apr 18: Settings menu ≠ Session options panel

The "Settings → Accelerator" menu I was using is a QUICK PICKER that sets session-local only. The actual PERSISTENT setting that Save Version uses lives in **Expand Session options panel** (right sidebar on /edit page).

Full CLAUDE.md playbook:
1. CLI push
2. Navigate /edit
3. Click "Expand Session options" (right side)
4. Accelerator dropdown → "GPU RTX Pro 6000" → confirm dialog
5. Toggle Internet OFF then ON (this commits the settings change)
6. Environment combobox → "Always use latest environment"
7. Verify COMPETITIONS/MODELS/UTILITY SCRIPTS visible
8. Save Version → Save

Steps 3 (expand session), 5 (internet toggle), 6 (env latest) were missing from my Playwright flow. That's why Save kept picking up P100 from the CLI-pushed default.
