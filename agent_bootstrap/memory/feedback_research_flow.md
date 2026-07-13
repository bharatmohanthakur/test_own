---
name: feedback-research-flow
description: "THE OPERATING FLOW (user-shaped, 2026-07-12): diagnose→gate→rank by pool÷effort→validate→ship→keep engines loaded. The master loop for competition campaigns."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

User (2026-07-12, after shaping it through ~10 instructions): "attack the hardest part / most promising jump with research so we have less effort and more gain" + "save this flow in memory."

**THE FLOW (one loop, ordered by effort-to-gain):**

1. **Diagnose before building.** When something fails, find WHY at the mechanism level (e.g., r211 cost-landscape microscope: wrong-basin vs flat-landscape vs broken-fusion — each names its discipline: physics / stats / engineering). No build until the failure is named.

2. **Gate every new idea** (see [[feedback-research-before-experiment]]): literature search → the math (headroom quantified, null vs hypothesis stated) → dead-list check → ingredient check (structure-kill vs component-kill; before rejecting, ask what was missing).

3. **Pick targets by decodable-pool ÷ effort.** Maintain the ranking table explicitly (pool %, effort estimate, status). Attack the largest pool with the cheapest mechanism first. Never attack measured-dead pools (e.g., Rogii sub-800ft texture = interpreter-private).

4. **Validate ruthlessly, ship cheaply.** Pre-registered predictions; instrument used only within its validated family (Rogii: G* for tracker legs only); confirm law; sibling-sanity screen; fallback ladders in every kernel; ≤1 public direction question/day; hedged final pair regardless of confidence.

5. **Keep every engine loaded in parallel.** Local CPU = diagnostics + algorithmic decoders; Kaggle free quota = seed fleets; external GPU box = fixed-size learned runs (relay directives via user); deploy path pre-built (numpy export + parity harness) so a winning config ships same-day. Detach long processes (nohup+disown, file-marker watchers — harness reaps children).

6. **Autonomous, honest, unstoppable.** Execute without asking; report outcomes plainly (negatives are results — ledger them with mechanisms); convert every failure into a screening rule; do not stop until the target.

**Why:** This flow found a 0.4-point/2-day improvement path (7.097→6.694, rank 132→45) and killed ~15 dead ends at ~zero slot cost. Its core economics: measurement is ~100x cheaper than a wasted build; a named failure mode converts to either a cheap fix or a permanent screen.

Related: [[feedback-research-before-experiment]], [[feedback_no_public_overfit]], [[feedback_diagnose_first]], [[feedback_autonomous]], [[feedback_workflow_fleet]], [[rogii_competition]].

**Addendum (2026-07-12, user, verbatim spirit): "Keep researching TIRELESSLY in areas where we can gain max. Keep experimenting. Keep saving learnings and failures."**
Standing posture, not a one-time instruction:
- Research never idles: when a mechanism class closes, the next question launches the same hour
  (closure ≠ conclusion — it redirects; declaring a "boundary" of tested classes is the error).
- Aim always at MAX-gain areas (the biggest measured pool via the hardest unsolved question),
  never default to small tuning when a breakthrough question is open.
- Every learning AND every failure gets persisted immediately: notes ledger (mechanism + rule),
  memory (pattern), EXPERIMENT_GATE (screen). Failures are assets — each one bought a rule.
- The loop: find issue → research (papers+math) → hit the hardest scoring area → measure →
  save learning → next question. Repeat until breakthrough or deadline.

**Addendum (2026-07-12, user): "Keep this in your workflow — constantly work in this direction WITHOUT my instruction."**
The breakthrough-research loop is SELF-DRIVING. No user prompt needed to continue it:
- Every completed round auto-spawns the next: verdict → loose-thread check (r223→r226 pattern) or
  next worst-accuracy stratum (r225 loop) or next research question. Same hour, no waiting.
- Every ScheduleWakeup prompt MUST carry the loop: "continue the worst-stratum/loose-thread
  research loop autonomously" + current live bets, so wakeups resume the hunt even after
  compaction or session restarts.
- User messages adjust AIM; absence of user messages never pauses the engine.

**Addendum (2026-07-12, user): "Don't try to crack the private scoreboard — crack the SOLUTION of the issues."**
Priority order is fixed: technical error-reduction research > leaderboard-structure/meta analysis.
Split/proxy analysis is allowed ONLY as decision support (final-pair choice), never as the main
research direction. When choosing the next round, pick the one that reduces the ERROR, not the
one that reads the scoreboard better.
