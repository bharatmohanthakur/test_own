---
name: Vast.ai usage policy
description: Only use Vast.ai if Kaggle OOMs. NEVER change GRPO config (max_token, speedup, etc) — use exact same config as Kaggle but with more memory.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Vast.ai = backup ONLY for OOM situations.**

Rules:
1. Only use Vast.ai if Kaggle gets OOM error
2. Rent a higher-memory GPU (A100 80GB or H100) to handle what Kaggle couldn't
3. **NEVER change any GRPO config** — same max_completion, same num_generations, same batch, same everything
4. The ONLY difference is more VRAM, nothing else
5. Balance: $10 on account <REDACTED-set-via-env>

**Why:** User explicitly said "under no condition any configuration of GRPO will change — max token, speedup and all". Vast.ai is purely for memory headroom.
