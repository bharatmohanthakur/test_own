---
name: feedback-codex-colleague
description: "Standing rule (2026-07-13) — use the Codex plugin/CLI as BOTH step-reviewer and planning colleague for experiments, failure analysis, and next-step decisions"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fe35a038-db2a-414e-9cdc-f5035bda8cd8
---

User directives (2026-07-13, Rogii session): "keep asking codex review on your steps" and "while planning experiments, failures, next steps — you can use codex as colleague to discuss and plan."

**Why:** an external frontier model from a different lab is a decorrelated check — its first review (AUDIT-CODEX-1) caught a real process slip (a smoke artifact ledgered as authoritative) that the house missed while the validator was paused.

**How to apply:**
1. **Review lane:** after each round's scripts/diff are committed, run an adversarial review via the plugin companion: `node ~/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs adversarial-review --background --base <ref> "<focus>"`. Stop-review-gate is ENABLED (`setup --enable-review-gate`). Verify each finding in-house before ledgering (dispositions: CONFIRMED / ACCEPTED-wording / NOT-REALIZED).
2. **Colleague lane:** at planning points (experiment design, failure postmortems, next-attack selection), consult Codex via `codex-companion.mjs task --background [prompt]` — give it the strategic map + dead-list and ask for attacks/designs; treat its proposals as input to the research gate (papers+math+dead-list+control), not as verdicts.
2a. **MANAGER ALIGNMENT DIRECTIVE (2026-07-13, binding, STRICT since 13:05 "without alignment don't take any decision"):** NO decision — launch, kill, spec change, resource move, restart — executes before Codex has seen it and the position is converged (or the disagreement is explicitly ledgered for the manager). Every verdict/interpretation goes into the thread AS IT HAPPENS. If the thread is busy (single-writer, mid-build), the decision WAITS in a queue; running work continues but new decisions freeze until aligned. Violation precedent to avoid repeating: the 2026-07-13 rival-loop restart went out unaligned and needed retroactive ratification.
2b. **DIALOGUE, not one-shots (user-directed 2026-07-13):** threads are resumable — `codex-companion.mjs task --resume-last [follow-up]` (or `--resume <id>` / `codex resume <session-id>`). Consults and reviews are CONVERSATIONS: return with objections/results to the SAME thread and iterate to convergence (challenge its proposals with the dead-list; after fixing review findings, resume the review thread for re-check). Only open a fresh thread for a genuinely new topic.
3. Model: default `gpt-5.6-sol` + effort high from `~/.codex/config.toml` (user confirmed this choice). `--model`/`--effort` per-call only on the `task` subcommand; `spark` = fast tier for mechanical jobs.
4. Auth: ChatGPT login (thakurb009@gmail.com), already active. Codex reviews/consults are read-only on the repo (never let it modify the champion, kernels, or ledger).

Related: [[feedback-research-before-experiment]], [[experiment-selection-lessons]].
