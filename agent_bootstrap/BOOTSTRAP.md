# Agent Bootstrap — full memory + config transfer (2026-07-13)

You are a remote Claude agent starting fresh on a big box. This bundle is the complete
persistent state of the campaign agent (minus credentials). Install it BEFORE doing anything.

## 1. Install memory
Copy `memory/` into your project memory directory so recall works natively:
```bash
mkdir -p ~/.claude/projects/<your-project-slug>/memory
cp -r agent_bootstrap/memory/* ~/.claude/projects/<your-project-slug>/memory/
```
`memory/MEMORY.md` is the index — it loads every session. Read it first.

## 2. Install agent configs + hooks + skills
```bash
mkdir -p .claude/agents .claude/hooks .claude/skills
cp agent_bootstrap/agents/*  .claude/agents/     # experiment-auditor, rival-builder personas
cp -r agent_bootstrap/hooks/*   .claude/hooks/   # inject_reminders.sh (the every-turn rules), block_destroy, etc.
cp -r agent_bootstrap/skills/*  .claude/skills/
cp agent_bootstrap/hookify/* .claude/            # hookify local rules
```
Wire `hooks/inject_reminders.sh` as a UserPromptSubmit hook in `.claude/settings.json` —
it carries the binding rules (GOAL RULE, DISCOVERY NEVER STOPS, CODEX COLLEAGUE, RESEARCH GATE).

## 3. Credentials (NOT in this bundle — set via environment)
- Kaggle (bharatmohan): put kaggle.json in ~/.kaggle/ yourself
- Nency's Kaggle token, Vast.ai, RunPod, wandb, DeepSeek keys: ask the manager (Bharat) or
  read them from the LOCAL machine's memory dir (they were deliberately excluded from git).

## 4. Read-in order (30 minutes to full context)
1. `memory/MEMORY.md` + `memory/rogii_competition.md` (mission state + 2026-07-13 pivot)
2. `CLAUDE.md` at repo root (project rules)
3. `rogii/notes/window_summary_2026-07-13.md` (the campaign arc, Parts 1-4)
4. `rogii/notes/agent_coord_2026-07-10.md` LAST ~150 lines (r322-r331: the mechanism pivot,
   the joint statement road map, all binding gates)
5. `rogii/.intel_20260713/REPORT.md` (public-method census — the formulation gap)
6. `memory/experiment_selection_lessons.md` + `memory/feedback_research_before_experiment.md`
   (the gates you must pass before ANY experiment)

## 5. Current state you inherit (as of 2026-07-13 night)
- Champion driver35 public 6.694 FROZEN. NO submission without the manager. Purpose rule:
  discover honestly, never exploit (duplicate channel deliberately retired).
- PRIMARY LINE = r331 mechanism round: dTVT + dZ = dANCC (differencing kills the per-well
  datum exactly); ~15 local annotation states/well; 3 arms (C0 cdeotte control / armA state
  distillation / armB cumulative reconstruction) with preregistered gates in the ledger.
- SECONDARY: r330-B1 dense pseudo-cut GBDT (paused-resumable on the laptop; you can rerun
  fully: scripts/exp330_dense_gbdt.py). r329 robustified transport = parked insurance.
- The Codex-colleague thread is LOCAL to the laptop (codex CLI session). On this box, either
  install the codex plugin + `codex login`, start a NEW thread, and paste it the read-in
  order above — or coordinate through the ledger only.
- The laptop agent may still be running its own compute. COORDINATE VIA GIT: pull before
  building, commit ledger entries promptly, prefix your ledger entries with [REMOTE-BOX].

## 6. What the big box should run first (memory-heavy jobs the laptop can't)
1. `rogii/scripts/exp330_dense_gbdt.py` full 772-well pass (needs >16GB for the training matrices)
2. r331 expansions if arms pass their gates (bigger K, full permutation batteries)
3. The r328-D mixture-resampling stress portfolio (20k cohorts — RAM-hungry)
Data deps: `rogii/train/` is in the repo; `.exp_cache/` is NOT in git (16GB) — regenerate via
each script's build stage, or pull the staged Kaggle datasets (see rogii/kaggle_ds_* metadata).
