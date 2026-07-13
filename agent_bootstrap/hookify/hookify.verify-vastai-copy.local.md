---
name: verify-vastai-copy
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: vastai\s+stop\s+instance\s+[0-9]+
  - field: command
    operator: not_contains
    pattern: copy-verified
action: block
---

🛑 **About to `vastai stop instance` — was a `vastai copy` issued recently?**

**Lesson from 2026-05-03**: stopped a pod after `vastai copy initiated`, copy never completed because source pod was exited. Lost ~30 min waiting for restart queue + ~$1 idle cost on destination pod.

**Before stopping a source pod:**
1. SSH into BOTH pods
2. Verify destination has the expected files: `ssh <dst> "du -sh /workspace/<path>"`
3. Compare byte counts: source vs destination must match
4. Confirm the file list: `ssh <dst> "find /workspace/<path> -type f | sort"`

**Vastai copy is async** — the "initiated" message is NOT confirmation of completion. Files transfer via vast's network and can fail silently if source goes exited.

**Better workflow** (per `vastai-pod` skill "Sequential workflow"):
- Train on pod A → upload adapter as Kaggle dataset → stop pod A
- Pod B downloads from Kaggle → run bench

This is the proven path. Vastai copy is a shortcut that breaks when timing is wrong.

If verified, append `# copy-verified` to bypass.
