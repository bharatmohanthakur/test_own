---
name: feedback-model-split
description: "Fable for planning/strategy (main loop), Opus for code-writing subagents — pass model:'opus' on implementation agent() calls"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

User directive (2026-07-06, updated 2026-07-10): use **Fable for planning AND for complex implementation**; **Opus for routine code-writing**.

**Why:** Fable's reasoning suits strategy/arbitration and subtle algorithmic work (e.g. the global-lock decider: calibration + mode enumeration + posterior integration). Opus remains the default for straightforward implementation (and avoids Fable usage-credit exhaustion on long mechanical agents, which killed several on 2026-07-06).

**How to apply:** Main loop stays on whatever the user set (/model). In Workflow scripts: pass `model: 'fable'` in agent() opts for complex/algorithmically-subtle implementation stages and planning/synthesis/arbitration; pass `model: 'opus'` for routine code-writing stages.

**MANDATORY (2026-07-10): Fable ALWAYS reviews Opus work.** Every Opus implementation stage gets a Fable review pass (methodology + code correctness + does-the-number-mean-what-it-claims) before its results are trusted, checkpointed as fact, or shipped. In workflows: append a `model:'fable'` review agent after Opus stages, or the Fable main loop reviews the artifacts itself before acting on them. Related: [[feedback_workflow_fleet]], [[rogii_competition]].
