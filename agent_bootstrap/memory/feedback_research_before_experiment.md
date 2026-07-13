---
name: feedback-research-before-experiment
description: "User directive 2026-07-12: before ANY new experiment, do literature research (papers in the same area) + explicit mathematical/scientific reasoning — no experiment launches on intuition alone"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

User (2026-07-12, Rogii 4-band campaign): "whenever you plan for a new thing / new experiment, do research of papers, think mathematically and scientifically."

**Why:** The r204 literature sweep proved the value immediately — published geosteering research (Jahani/Alyaev arXiv:2103.05384, Muhammad arXiv:2402.06377) reframed our measured drift bottleneck as a sequential low-dim estimation problem and flagged two of our "dead" verdicts as implementation artifacts (one-shot vs sequential fits; wavelet-power vs GR-shape warps). Intuition-only experiments burned rounds that papers would have redirected.

**How to apply — the experiment gate (before every new round):**
1. **Literature check**: WebSearch for papers in the same problem area (geosteering, well-log correlation/DTW, sequence tracking, whatever matches); cite what exists; check whether the idea is published, improved-upon, or refuted.
2. **Math first**: write the quantitative case — what measured quantity does it attack, what is the theoretical headroom (in G*/ft), what would the result look like under the null vs the hypothesis.
3. **Dead-list check**: verify against the ledger that no measured kill covers it (and if a paper contradicts a kill, name the implementation difference before re-testing).
4. Only then launch, with pre-registered thresholds.

Applies to Rogii and future campaigns. Related: [[feedback_no_public_overfit]], [[feedback_diagnose_first]], [[rogii_competition]].

**Addendum (2026-07-12, user): "Before rejecting anything, think and see if anything else is needed."**
A kill is only final after asking: which INGREDIENT failed — the structure or a component? If a
component (emission, observable, regularizer, gate), name the best available replacement and either
test it or ledger why not. Every rejection entry must state: structure-kill vs component-kill.

**ENFORCEMENT (2026-07-12, after a second miss — user: "where to update that you will NEVER miss"):**
1. The UserPromptSubmit hook (.claude/hooks/inject_reminders.sh) now injects the gate every turn.
2. HARD PROTOCOL: every Workflow launch for an experiment MUST contain a literal `GATE:` block in
   the agent prompt stating (papers found, math case, dead-list check, matched control). If I draft
   a Workflow call without it — stop, run the literature pass first (a research workflow with
   WebSearch), THEN build. Candidates from my own reasoning are HYPOTHESES for the research pass,
   never direct builds — no matter how math-flavored they sound (the Stein-shrinkage slip: sounded
   like math, was actually an unresearched hunch about applicability).
