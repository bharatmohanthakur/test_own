---
name: testing_policy
description: Always test fixes with sample data before full runs — saves GPU quota
type: feedback
---

## Rule: Test with Sample Data First

**When fixing an error or trying a new approach, ALWAYS run with 20-50 samples first.**

### Why
- v24 SFT+GRPO wasted 73 min of GPU before erroring on torch.compile bug
- v25 "fixed" version wasted another 93 min on same bug (fix didn't work)
- v26 with 20 samples tests the same fix in ~20 min
- A 15-20 min test run saves 1-2 hours of wasted GPU quota when the fix is wrong

### Workflow
1. Identify bug/write fix
2. Create test version with 20-50 samples (sft_20 + grpo_20)
3. Deploy with `--accelerator NvidiaRtxPro6000`
4. Wait ~15-25 min for validation
5. If it runs end-to-end → deploy full data version
6. If it errors → fix and re-test (still only 20 min wasted)

### What counts as a "fix validation"
- Script reaches end without errors
- Adapter saves successfully
- submission.zip created
- No need to get a good score — just prove the pipeline works

**Why:** GPU quota is finite (30 hrs/week RTX Pro 6000). Each failed full run costs 1-2 hours. Sample tests cost 20 min.
**How to apply:** For EVERY new fix or approach, create a "test" version with minimal data first. Only run full data after the test passes.
