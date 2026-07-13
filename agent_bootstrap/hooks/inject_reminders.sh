#!/usr/bin/env bash
# UserPromptSubmit hook — injects a short reminder of the top guardrails.
# Keeps the 4 most frequently-violated rules visible on every turn.
cat <<'EOF'
<kaggle-guardrails>
- Baseline: thk_v26 = 0.85 Kaggle. Any new adapter must beat v26 on local bench BEFORE submit.
- Pods: stop, never destroy (saves 15-30 min re-setup). Use `vastai copy` between pods.
- Data: every training example needs \boxed{ANSWER} AFTER </think>. Verify before train. ≥95% required.
- Run with wandb online. report_to="none" + nohup = blind = never again.
- ROGII GOAL RULE (outranks everything): SOLVE FULL DC. Failures change the ATTACK, never the
  TARGET. Sub-pool order: late re-locks 42% (WHERE + HOW MUCH) > ramps 30%. Every round states its
  DC sub-pool + headroom. Closures kill channels, never the goal — when channels run out, INVENT one.
  Slope/wander/polish only in the gaps, never displacing a DC move. DISCOVERY NEVER STOPS
  (manager, 2026-07-13): "stop discovery" is NOT a valid conclusion for any engine or convergence
  — reprioritize yes, halt never; halting is manager-reserved. Runway = Aug 5. (memory: rogii_competition)
- AUDIT-AUTOPILOT: PAUSED by user 2026-07-13 ("pause validator for time being") — do NOT auto-launch
  the experiment-auditor on round completions; accept round verdicts directly (their internal
  controls/preregistration still apply). Auditor remains available ON-DEMAND. Re-enable when user says so.
- RIVAL LOOP (standing): the rival-builder runs in a permanent cycle — BUILD mission -> autopilot AUDIT
  -> error-anatomy names next attack -> next mission launches SAME TURN -> champion checkpoint each cycle
  (vs 7.085 honest harness). Rival never idles; clean-room wall holds until its Phase 4. WIN >=0.10 ->
  integration gate; converge -> basin-confirm ledgered. (agent: .claude/agents/rival-builder.md)
- CODEX COLLEAGUE (standing, 2026-07-13): Codex (gpt-5.6-sol via plugin companion script) is BOTH
  (a) adversarial reviewer of every round's committed diff (stop-review-gate ENABLED; verify findings
  in-house before ledgering) AND (b) planning colleague — consult when designing experiments,
  analyzing failures, choosing next steps (companion `task` subcommand; proposals feed the research
  gate, never bypass it). (memory: feedback_codex_colleague)
- RESEARCH GATE (never skip): before ANY new experiment/build — WEB+PAPERS first, MATH written
  (headroom, null vs hypothesis), dead-list + own-history checked, matched control named.
  Every Workflow launch prompt must contain a GATE: block stating all four. No exceptions,
  including "obvious" ideas. (memory: feedback_research_before_experiment, experiment_selection_lessons)
(see skills: kaggle-train, kaggle-submit, vastai-pod, kaggle-data-quality, experiment-gate)
</kaggle-guardrails>
EOF
