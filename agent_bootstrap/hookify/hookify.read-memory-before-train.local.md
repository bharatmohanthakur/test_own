---
name: read-memory-before-train
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: ssh\s+.*python.*(train_|sft|dpo|grpo|gspo|tinker)
  - field: command
    operator: not_contains
    pattern: memory-checked
action: block
---

🛑 **About to launch a training run on remote pod.**

**Required reading before training:** memories that document failures and constraints for THIS specific config.

Quick checklist (top 5 only):
- [ ] `grpo_blockers.md` — 4-bit forbidden with Mamba
- [ ] `errors_fixes.md` — 28 documented errors
- [ ] `blackwell_training_fix.md` — 5-step Blackwell patch
- [ ] `logging_policy.md` — wandb online MANDATORY (`report_to="wandb"`)
- [ ] `tracking_system.md` — local bench MANDATORY before submit

Also verify your training script:
- `load_in_4bit=False`
- `attn_implementation="eager"` (Nemotron H doesn't support sdpa)
- `gradient_checkpointing=True, use_reentrant=False`
- `report_to="wandb"` with `WANDB_PROJECT` env set
- `\boxed{}` after `</think>` in 95%+ of training data
- `lr ≤ 5e-5` if refining from v26

After confirming all of the above, append `# memory-checked` to bypass.
