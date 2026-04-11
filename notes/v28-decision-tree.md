# v28 Outcome Decision Tree (Apr 7, 2026)

## v28 v2 outcomes and next moves

### Scenario 1: v28 completes & adapter scores > 0.66
**Meaning**: GRPO with simple correctness/format rewards on top of v22 SFT helps a tiny bit.
**Action**:
1. Submit v28
2. Push v29 SFT immediately (Donald-templated data)
3. Wait for v29 SFT to finish (~2 hrs)
4. Push v30 GRPO (Donald rewards, reduced config)
5. Submit v30

### Scenario 2: v28 completes & adapter scores ≤ 0.65 (i.e. matches v22 base)
**Meaning**: GRPO didn't help — either too few steps, weak rewards, or saturated SFT.
**Action**:
1. Submit v28 (record the score)
2. Skip pure GRPO iteration; focus on SFT v29 with Donald data
3. Push v29 SFT immediately
4. After v29 SFT, try v30 GRPO with reduced steps + Donald rewards
5. If v29 SFT itself > 0.69, that's a clear win — don't bother with GRPO

### Scenario 3: v28 errors / crashes mid-run
**Meaning**: Some other Unsloth/Mamba bug we haven't caught yet.
**Action**:
1. Download logs immediately
2. Identify error
3. Add error to errors_fixes.md memory
4. Decide: fix in v28 (retry) or skip GRPO entirely until SFT v29 lands
5. Push v29 SFT regardless — it doesn't depend on Unsloth GRPO

### Scenario 4: v28 hits 12-hr limit before saving
**Meaning**: GRPO took too long, no adapter saved.
**Action**:
1. Add periodic checkpoint saving to v30 (save_steps=5)
2. Use save_steps=5 strategy in v30 — every 5 steps saves the LoRA
3. Push v29 SFT (independent path)

## Universal next steps regardless of v28 outcome

1. **Push v29 SFT** as the immediate next experiment. Donald's templated traces are likely a much bigger lever than GRPO refinement.
2. **Submit v22 base re-roll** to get baseline under new metric (we know v7=0.69 was under old metric; new metric should drop ~0.35-0.40 → expected 0.65)
3. **Wait for Monday rescore** — Ryan Holbrook will rescore old submissions under the new strict metric. Our 0.69 should become ~0.65, putting us at the same level as recent submissions and confirming the "metric drop" narrative.

## What NOT to do
- **Do not run multiple parallel notebooks.** RTX Pro 6000 has limited concurrency on Kaggle. One run at a time.
- **Do not skip the v28 submission** even if score is low — it's a data point for comparing GRPO contribution.
- **Do not push v29 SFT until v28 finishes.** CLI push starts a new P100 run, which would need UI switch and burn quota.
