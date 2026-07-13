---
name: prompt-format-bug
description: "CRITICAL 2026-06-11: ALL local benches v1-v10 + rollout collection used handmade '<|im_start|>user...' prompts MISSING the empty system block that (a) TRL training via chat template AND (b) Kaggle's OFFICIAL metric (apply_chat_template, add_generation_prompt=True, enable_thinking=True) both include. Every specialist holdout 0/24 was measured out-of-distribution. Likely explains part of the +0.12 local→Kaggle offset."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**The bug.** Nemotron-3-Nano's chat template prepends an EMPTY system turn even when no
system message is given: `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n...`.
- Training (TRL SFTTrainer on `messages`) renders via the template → system block PRESENT.
- Official Kaggle metric (`notebooks/metric/nvidia-nemotron-metric.ipynb`) builds prompts with
  `apply_chat_template([user], add_generation_prompt=True, enable_thinking=True)` → PRESENT.
- Our `scripts/bench_indist.py`, `scripts/bench_vllm.py`, `scripts/collect_rollouts.py`, and
  `tracking/run_bench_fast.py` all hand-built `<|im_start|>user\n{...}<|im_end|>\n<|im_start|>assistant\n<think>\n`
  → system block ABSENT.

**Evidence it mattered:** v9 memorized 515 traces to loss 0.136, all sharing one constant
60-char opening — yet only 12% of 1,218 rollouts reproduced that opening (first tokens depend
only on the prompt; no exposure bias possible). Token-level check confirmed the template
generation-prompt is an exact prefix of the training render, differing from our bench string
ONLY by the system block.

**Consequences:**
1. ALL v1–v10 crypt holdout numbers (and the in-dist 0/20, pass@k, DPO/SimPO rollouts) were
   measured OOD. The "verbosity drift / preference-doesn't-transfer" conclusions in
   [[crypt-final-closure]] need re-validation with corrected prompts.
2. The consistent +0.12 local→Kaggle offset in [[tracking-system]] is probably partly this.
3. SECOND mismatch: official eval appends the FULL suffix
   `\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}``
   unconditionally to the raw prompt. Our DD training rows had NO suffix; SS rows a shorter one.
   Future training data must use raw prompt + the exact official suffix.

**How to apply:**
- Bench with `--system-block` (added to scripts/bench_indist.py) or, better, always build
  prompts via `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`.
- Fix tracking/run_bench_fast.py the same way before the next submit gate.
- Match the official suffix verbatim in all future training data.
