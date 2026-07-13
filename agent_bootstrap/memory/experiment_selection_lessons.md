---
name: experiment-selection-lessons
description: "Meta-analysis of the Rogii campaign's failed experiments (r184-r214): the 5 recurring selection/design mistakes and the checks that prevent each — read before designing ANY experiment"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

Meta-analysis requested by user 2026-07-12: what mistakes did we make in DOING/SELECTING failed experiments. From ~15 failures across r184-r214:

**MISTAKE 1 — Precondition-blindness (the most frequent: 5 failures).**
Ported methods or built models without verifying the data satisfies their identifiability/information preconditions.
- r205 sequential PF: papers re-anchor on EM/resistivity logs with sharp signatures; our GR is self-similar → likelihood uninformative. A 1-hour cost-landscape check would have predicted this BEFORE the 2-day build.
- r206 monotone warp: free-end warp on self-similar signal wandered 62 ft — the papers pin tie-points; we tested the algorithm without its anchoring context.
- r193 v0 / r195: built full-well/per-well models while the FEATURE CACHE didn't contain the hypothesis-critical signal (head tokens inert/absent — the entire point wasn't in the tensors).
- r194/GPU synth-10x: scaled corpus QUANTITY before the 30-min QUALITY audit (KS domain gap) that killed it.
**Check:** before any build — "does the input provably contain the signal this method needs? run the cheapest direct probe of that first."

**MISTAKE 2 — Own-history amnesia.**
exp140 had already measured capacity-scaling dead in our own repo; we nearly re-ran it (r187 audit found it mid-flight) and the GPU box DID re-run it.
**Check:** grep own scripts/notes/ledger for the axis before launching; the answer may be on disk.

**MISTAKE 3 — Instrument overreach (cost a submission slot).**
driver36: shipped spatial-label legs using G*+confirm-law calibrated ONLY on tracker legs → +0.118 public miss. Instruments validate within an evidence family; a NEW information-source family needs a small-weight probe, not a full-confidence ship.
**Check:** classify every candidate by information source; new family ⇒ probe-grade prereg with wide band.

**MISTAKE 4 — Intuition launches (no gate).**
r202 distance-weights had no paper, no mechanism, no headroom math — and was the cleanest null. The research gate (papers+math+dead-list) now exists; this entry is why.

**MISTAKE 5 — Unfair baselines nearly credited wrong causes.**
r187 almost attributed refit-freedom gains (~0.025) to the new architecture; only the "fair refit control" caught it. Every eval needs the matched control that isolates EXACTLY the tested change.

**Operational recurrents:** agent background processes get reaped (use nohup+disown + file-marker watchers); zsh `===`/heredoc quirks break compound commands; StructuredOutput can fail after real work — check disk before re-running; stage_push must poll dataset processing before verify (r192 bug).

**What SELECTION did right (keep):** oracle-headroom tests before builds (r184/185 saved weeks); twin wells as noise-floor instrument; pre-registration; per-bin/per-band decompositions turning "it failed" into "it failed HERE"; control wells in forensics designs.

Related: [[feedback-research-flow]], [[feedback-research-before-experiment]], [[rogii_competition]].
