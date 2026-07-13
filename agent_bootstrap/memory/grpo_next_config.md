---
name: GRPO next run config (user specified)
description: User wants max_completion=4096, num_generations=1, smaller batch. No other changes.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
Next GRPO run config (user specified Apr 12):
- max_completion_length = 4096 (was 1024, up to 8192 allowed by eval)
- num_generations = 1 (was 4, saves VRAM for longer completions)
- batch_size = 1 (or smaller effective batch)
- Everything else stays same

**Why:** 70% of GRPO steps get 0 reward because completions clip at 1024. Model needs room to finish reasoning on hard puzzles. 4096 tokens should be enough for most puzzle types.
