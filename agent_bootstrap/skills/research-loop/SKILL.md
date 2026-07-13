---
name: research-loop
description: Run parallel research while training/scoring is in progress. Use whenever a training run or submission scoring is in flight (10–90 min idle). Rotates through 5 categories — competition intel, technique scouting, top solution analysis, dataset discovery, error pattern analysis — and saves findings to notes/.
---

# research-loop — Never idle during training

## Trigger
- A training run is in progress (10–90 min wait)
- A submission is scoring (15–30 min wait)
- Score didn't improve from last submission
- Gap to #1 is > 0.05

Launch research as **background Agent tasks** (parallel to training monitoring).

## 5 categories (rotate)

### 1. Competition intel
- Playwright: leaderboard → top score, our rank, gap
- Playwright: discussion tab → sort by recent → read new threads
- WebSearch: `"nemotron reasoning challenge kaggle" 2026`
- Look for: new public notebooks, shared techniques, organizer clarifications

### 2. Technique scouting
- WebSearch: `"LoRA fine-tuning reasoning model [technique] 2026"`
- Techniques to rotate through: RAFT, DAPO, GSPO, curriculum learning, rejection sampling, STaR self-training, tool-integrated reasoning
- HuggingFace: new reasoning datasets, new Nemotron variants
- GitHub: `NVIDIA-NeMo/Nemotron` repo for new recipes

### 3. Top solution analysis
- WebSearch: `"kaggle reasoning competition winning solution 1st place"`
- Study: AIMO-2 1st (NemoSkills), ARC prize, any Nemotron-family wins
- Pull top-voted public notebooks for THIS competition
- Key question: what data + method did winners use?

### 4. Dataset discovery
- WebSearch: `"reasoning dataset huggingface 2026"`
- Check: `nvidia/Llama-Nemotron-Post-Training-Dataset`, OpenMathInstruct, GSM8K-CoT, MATH, code-reasoning
- Match to the 6 puzzle types (cipher, bit-ops, cryptarithm, gravity, roman, unit)

### 5. Error pattern analysis
- On score received: which puzzle types got worse?
- Generate failing test predictions locally
- Focus next data gen on weakest categories

## Output format

Save to `notes/research-YYYY-MM-DD.md`:
```markdown
# Research YYYY-MM-DD

## What was searched
- <query 1>
- <query 2>

## Key findings (actionable only, no fluff)
- Finding 1 → change X
- Finding 2 → try Y

## Specific changes to try
- [ ] Swap dataset for <new>
- [ ] Add <technique> to v<N+1>

## Resources
- <url 1>
- <url 2>
```

## Applying research

After each research session, update:
1. `CLAUDE.md` → "Currently trying" section if strategy shifts
2. Training script for next run → incorporate best technique found
3. Data generation pipeline → add new sources or reformatting

## What NOT to research (dead ends — skip)
Per memory:
- vLLM colocate with PeftModel + NemotronH (`vllm_grpo_learnings.md`)
- NeMo-RL "single GPU" recipe (actually multi-node)
- GRPO without high reward signal (always regresses: `grpo_b200_result.md`)
- Token-priority SFT on base Nemotron (from-base needs format prior first — `v29_disaster.md`)

## Time budget
- 10 min wait → 1 research thread
- 30 min wait → 2 research threads in parallel
- 60+ min wait → 3 research threads + draft next training script

## Memory pointers
- `research_apr9.md`, `research_apr11.md` — prior findings
- `thk_winning_solution.md` — current ceiling (0.85, SFT-only)
- `path_to_090.md` — levers to beat THK
- `training_methods_sota.md` — current SOTA ordering
