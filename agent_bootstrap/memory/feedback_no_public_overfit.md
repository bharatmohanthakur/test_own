---
name: feedback-no-public-overfit
description: "User directive 2026-07-11: don't overfit the public LB / G* instrument — final pair hedged, weights local-only, ≤1 public-arbitrated direction per day"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

User (2026-07-11, Rogii): "we should be careful to not overfit current test / unseen data" — said right after G* went 3/3 and the champion ladder dropped 6.816→6.733 in one day.

**Why:** Private split decides final ranking. G* was discovered by fitting rules to ~10 public scores; its predictive streak is on one fixed public sample. Directions the public split rewards (esp. ttail, whose gains concentrate in deep-tail extrapolation) may not transfer to private-weighted rows.

**How to apply:**
1. Final pair = 1 aggressive (best public) + 1 conservative (driver29/33 class, broad local optimum, minimal public-guided choices). Never two siblings of one publicly-tuned direction; keep a ttail-free leg.
2. G* selects directions only; blend weights always fit on local LRO pools, never on leaderboard feedback. No micro-tuning weights via repeated submissions (e.g., t=0.10 kept at the publicly-validated value instead of the sweep optimum 0.15).
3. ≤1 public-arbitrated direction question per day. Count decisions spent against the public LB, not submission slots.
4. Every kernel ships the fallback ladder (leg fail → trio → d17 → const-hold).

Full policy block in `rogii/notes/agent_coord_2026-07-10.md`. Related: [[rogii_competition]], [[feedback_target_high]].
