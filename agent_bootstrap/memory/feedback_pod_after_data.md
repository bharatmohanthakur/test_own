---
name: feedback-pod-after-data
description: "Don't start billed GPU pods ahead of uncertain-duration dependencies (e.g. agent-generated training data). Start pod only when the data artifact exists. User flagged 2026-06-09."
metadata:
  type: feedback
---

**What happened:** Started a $4.49/hr B200 + 13-min bootstrap in parallel with a subagent
building the training data, betting the agent would finish in ~15 min. Agent ran long
(uncertain by nature) → pod sat idle-billing ~25 min waiting for data. User: "don't you
think you should have started GPU after solver?"

**Why:** Parallelizing feels efficient, but only one side of the overlap (bootstrap) has a
known duration. Agent/codegen/data-validation tasks have high variance. Idle B200 = pure waste,
and this account has repeatedly run out of Vast credit — every idle minute matters.

**How to apply:** Sequence = produce data artifact FIRST (verify file exists + passes checks),
THEN create pod. Parallelize only fixed-duration, known-length steps. Exception: if data is
verifiably ≥90% done (file growing, final validation running), pod creation may start.
