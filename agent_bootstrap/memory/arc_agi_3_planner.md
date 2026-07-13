---
name: arc_agi_3_planner
description: ARC-AGI-3 interactive-agent competition — BFS planner approach and first verified milestone
metadata: 
  node_type: memory
  type: project
  originSessionId: 96f594c9-ad06-4ed1-82e3-7ce7a9a8cb54
---

ARC-AGI-3 (`arc-prize-2026-arc-agi-3`) is a **separate** Kaggle competition from the
Nemotron text-reasoning challenge that fills CLAUDE.md. It is an **interactive agent
benchmark**: agents play 64×64-frame games via actions RESET + ACTION1–7 (ACTION6 =
click with x,y). Score = levels completed across games, not a static grid prediction.
Workspace: `arc_agi_3/`. Local runtime works via `arc_agi_3/.venv` (Python 3.12);
public games extracted under `data/extracted/environment_files/` (25 games).

**Key technique (non-obvious):** the local public games are deterministic Python game
objects (`env._game`) that are **`copy.deepcopy`-able in ~9 ms**. That makes them a
perfect free forward model → classical tree search works. Built a general BFS planner
(`scripts/plan_keyboard_game.py`) that uses **only public observations** (frame,
available_actions, levels_completed, state — no hidden-state reads), clones to snapshot
states, BFS over distinct frame bytes branching on simple actions, rewards
`levels_completed` increases, chains levels. Verify any plan honestly through the real
scorecard with `scripts/verify_plan.py` (env.reset + env.step).

**First verified milestone (2026-06-06):** ls20 level 1 solved, optimal 13 moves
`[3,3,3,1,1,1,1,4,4,4,1,1,1]`, scorecard score 3.57 — vs the prior blind sweep
(`valid_action_search_agent.py`) which got 0/7. ls20 action map: ACTION1=up, 2=down,
3=left, 4=right; win needs path-planning + matching carried shape/color/rotation.

**Limit:** BFS doesn't scale — `deepcopy`/node (~13–56 ms) caps depth; ls20 level 2 not
reached in 180 s. Next: heuristic best-first (A*/greedy toward target sprites), generality
sweep across keyboard games (`scripts/sweep_planner.py`), and an online (no-clone) port
for submission. See `arc_agi_3/notes/planner_results.md`.

**CRITICAL scoring finding (`arc_agi/scorecard.py::add_level`):** ARC-AGI-3 scores
**EFFICIENCY, not completion**: per level `score = min((baseline_actions/actions_taken)^2
*100, 115)`, where `actions_taken` = ALL actions spent in that level incl. exploration &
resets; env score is level-index-weighted sum. Implication: completing a level via
brute-force search scores ~0 (online `SearchReplayAgent` completed ls20 lvl1 in ~20k
actions vs baseline 22 → score 0.0). So **completion ≠ points**; you must act near
`baseline_actions`. Built the competition-legal online agent
`agents/templates/search_replay_agent.py` (RESET+replay, no deepcopy/hidden-state, works
offline & competition) — it is legal+general but penalized by its own exploration cost.

**CORRECTED 2026-06-07 (read official Technical Report — `notes/research_arc_agi3.md`):**
The 25 public games (183 levels) are a **demo set, NEVER officially scored**. Eval is on
**PRIVATE held-out games** (55 semi-private + 55 fully-private), **first-contact**, with a
**5× human-baseline action budget per level**. ARC ships an open-source replay harness that
hits 100% on public and explicitly calls solve-and-replay **disallowed task-specific
overfitting**. ⇒ **My earlier "solve+replay the 25 public games" plan is the disallowed
harness — abandon it as a submission path** (keep the offline solver only as a dev/analysis
oracle). Real target = a **general agent** efficient on first contact with unseen games.
Frontier LLMs at release: Opus 4.6 0.50%, Gemini 3.1 0.40%, GPT-5.4 0.20%, Grok 0.10% —
WIDE OPEN. **Scoring (RHAE):** `S_l = min(1.15, h/a)²`, weights `w_l=l`, env cap = weighted
fraction of levels done, total = mean over envs; local `baseline_actions` = the human
baselines `h`. **Path forward:** width-based planning (RolloutIW) and/or LLM harness over frames. game tags:
5 keyboard, 6 click, 13 keyboard_click, 1 none.

**Solver verdict (2026-06-07, tested):** Built `scripts/perception.py` (object extractor) +
`scripts/solve_rolloutiw.py` (Rollout-IW(1)/(2), faithful depth-novelty; fixed a real bug —
CHECK_NOVELTY must treat depth==D[f] as novel on re-traversal/is_new=False or the tree
collapses). On ls20 L1: BFS solves in 451 nodes (13 moves); **IW(1) and IW(2) with object
atoms do NOT solve L1** (IW(2) explored 10,997 nodes, still missed it). Hand-crafted object
features don't isolate ls20's controllable/goal state (panel/counter flicker; high-width
mechanic) — matches the literature ("IW needs symbolic features; learning them is open").
⇒ **Stop hand-crafting search features.** Real levers: **π-IW learned features** (NN hidden
layer as Φ) or **LLM-reasoning harness**. BFS (`plan_keyboard_game.py`) is the only complete
solver, good only as a shallow-level dev oracle. Frontier LLMs <0.5% → genuinely open.

**Competition-aligned LLM harness built (2026-06-07):** Kaggle ARC-AGI-3 is a CODE/agent
competition — submit an `Agent` subclass that plays in competition mode, scored by RHAE on
held-out games. Built `agents/templates/llm_reasoning_agent.py` + `scripts/run_llm_agent.py`:
object-perception → compact frame text (object list + hex ASCII) → LLM reasons → parse 1
action → observe diff → accumulate (action→change) memory. Model-agnostic; DeepSeek backend
via `requests` (frames are int grids → text, no vision). **Scaffolding validated** but NOT yet
against a real LLM: **DeepSeek key is OUT OF BALANCE (HTTP 402)** so the cd82 runs silently
fell back to ACTION1 (not real reasoning). Fixed the silent failure (agent.stats + runner
WARNs on 0 LLM actions). **BLOCKER: need a funded LLM key** (DeepSeek top-up / frontier API /
MiniMax endpoint — `sk-cp-` key isn't native format). Improve via: stronger model
(deepseek-reasoner/frontier), better prompt (no-op avoidance, hypothesis tracking), Undo for
cheap checks, controllable-object detection. **RESOLVED: scored Kaggle eval has NO INTERNET**
(arcprize.org: "No internet access during evaluation"; Kaggle notebook, <12h, code CC0/MIT-0,
≤$10k runtime). ⇒ **LLMs allowed ONLY as offline/local models on the Kaggle GPU — external
API calls (OpenAI/Anthropic/DeepSeek) NOT permitted at scoring time.** The API harness is
dev/community-leaderboard only; submission backend must be a **bundled local open-weights
model** (ties to repo's Nemotron/vLLM/unsloth infra). DONE: backend is now pluggable
(`agents/templates/llm_backends.py`: HeuristicBackend / DeepSeekBackend[dev] / LocalHFBackend
[offline submission]; agent `BACKEND` attr, runner `--backend`). **Loop validated end-to-end**
with heuristic backend on ls20+cd82 (stats llm:19,fallback:0,error:0; memory populates). Fixed
2 bugs (silent 402 fallback now WARNs; `_parse_action` wasn't setting `_prev_action`).
**Submission kernel staged + flow rehearsed (2026-06-10):**
`notebooks/arc_agi_3_submission/` (submission.py: offline wheels →
listen_and_serve(competition_mode=True) thread → `main.py --agent=llmreasoningagent`
subprocess over local HTTP). Rehearsed exact flow locally end-to-end (clean scorecard
open/close). Rehearsal-found fixes: export `Swarm` in agents/__init__.py (main.py needs it);
pass backend via env `LLM_AGENT_BACKEND` (subprocess — class attrs don't propagate). Agents
registered: validactionsearch, searchreplayagent, llmreasoningagent. **PUSHED TO KAGGLE (2026-06-10):**
kernel `bharatmohan/arc-agi-3-llm-agent` v1 — self-contained submission.py built by
`scripts/build_submission_kernel.py` (embeds llm_reasoning_agent/llm_backends/perception as
string literals — script kernels upload ONLY the code file; a Kaggle dataset upload for agent
code was permission-blocked, inlining is cleaner anyway). Model attached:
`qwen-lm/qwen-3/transformers/4b/1` (Qwen3-4B; fp16 fits P100/T4; LocalHFBackend uses
float16 + enable_thinking=False for speed). First run = smoke on cd82 only (GAMES env
defaults "cd82"; set "" for all games). Push gotchas hit: title must equal slug; model_sources
need explicit /version suffix. DeepSeek key [[deepseek_key]] OUT OF BALANCE (402) — dev-only
anyway since eval no-internet.

**✅ STACK VALIDATED ON KAGGLE (v7, 2026-06-10):** Qwen3-4B genuinely playing cd82 on Kaggle
— varied LLM-driven actions (1,2,3,6-click,4,RESET,5 = the prompt's explore-untried behavior),
0 backend errors, ~20-90s/action on CPU. Iteration ledger v1→v7: (1) wheels path → glob
discovery; (2) upstream __init__ imports langgraph → replace with minimal registry; (3) bare
Arcade() phones home → OPERATION_MODE=offline env; (4) P100 sm_60 unsupported by torch →
CPU fallback (see [[kaggle_gpu_quirk]]); (5/6) agent swallowed errors → print first 2
tracebacks; (7) **apply_chat_template(return_tensors="pt") returns BatchEncoding in new
transformers → must use return_dict=True + generate(**enc)** — was the final blocker.
**Next:** (a) ONE MANUAL UI STEP for speed: flip accelerator to T4 x2 after push (P100 can't
run torch at all); (b) raise AGENT_MAX_ACTIONS/LLM_MAX_NEW_TOKENS + GAMES="" for full runs;
(c) iterate reasoning quality — plumbing is done, intelligence is the open problem.

**Game-diversity finding:** the 25 public games are NOT uniform mazes — they are diverse
perception puzzles (ls20=carried-shape matching w/ rotation; cd82=color/pattern select;
cn04=glyph completion; many use ACTION6 click). Frames ARE fully observable (walls,
panels, counters all rendered) but each game has its own mechanic + win condition. No
single A* agent scores across them. The real lever = per-game-mechanic perception/reasoning
that acts near-optimally with minimal probing. This is why public LB tops ~1.30. Oracle of
max-achievable level-1 scores: `scripts/build_oracle.py` → `notes/oracle_lvl1.json`.

**Solver R&D (2026-06-06):** 7/25 games BFS-solve level 1 (cd82,cn04,ls20,sk48,sp80 hit the
115 cap; m0r0,tu93 partial). Depth blocker: BFS explodes past ls20 L2. Tried (a) **IW(1)**
`scripts/solve_iw.py` — FAILED (exhausts 75 nodes, raw-pixel novelty saturates; ls20 is
width>1; needs object-level atoms / IW(2)); (b) **level-clone snapshot** for 6× sim
speedup — works for replay-from-start but INCOMPLETE for mid-level restore (≠ deepcopy,
verified), so `scripts/solve_fast.py` is BROKEN/unsafe — do NOT bank from it; use deepcopy
`plan_keyboard_game.py` (correctness-verified) until a complete snapshot is found. Cracking
deep levels is the open problem (matches LB ~1.30). Full log: `notes/STRATEGY.md`.
