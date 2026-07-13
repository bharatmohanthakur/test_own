---
name: THK winning solution
description: THK's full Progress Prize solution (0.85). SFT only, maximize min logprob, deterministic CoT, Tinker training. GitHub code available.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# THK Progress Prize Solution — 0.85 (Apr 13, 2026)

## Core Insight
**SFT only. No GRPO needed.** "I already have the optimal policy for solvable problems."
Maximize minimum logprob, not average loss. If any token has logprob < 0.69, it will fail at temp=0.0 inference.

## Target Solve Rates
- numeral: 100% (1576/1576)
- unit_conversion: 100% (1594/1594)  
- gravity: 100% (1597/1597)
- cipher: 100% (1576/1576)
- bit_manipulation: 85.1% (1364/1602)
- equation_numeric_deduce: 90.6% (540/596)
- equation_numeric_guess: 15.4% (21/136)
- cryptarithm_deduce: 8.2% (54/659)
- cryptarithm_guess: 6.7% (11/164)
- **TOTAL: 87.7% (8333/9500)**

## Training Details
- 27.8M tokens for winning submission, 599M tokens total across iterations
- Cost: $212 Tinker + $60 Modal + $10 Kaggle
- 20 min per Tinker iteration (vs 4 hours on Kaggle)
- Tinker adapter needs conversion: unfuse experts, SVD for in_proj (lossy — 75% singular mass)

## CoT Design Principles
1. **Deterministic** — temp=0.0 means just get most likely token right
2. **Simple** — each token derived from few sources, break division into subtraction+addition
3. **Coverage** — iterate through ALL options (e.g., all 77 wonderland words) not just first match
4. **Within 7680 token limit**
5. **Tokenization aware** — avoid model weaknesses in char splitting
6. **Generalizable** — no memorization

## Non-goals (important!)
- Diversity: not needed at temp=0.0
- Conciseness: longer is fine if within limit
- Verification: assume model generates every token correctly

## Key Code
- GitHub: linked from discussion (search THK's profile)
- reasoning.py: generates completion traces
- train_sft.py: sends traces to Tinker
- upload_adapter.py: uploads to Kaggle

## What's Still Unsolved
- Cryptarithm: only 7-8% solve rate, needs guessing approach
- Complex bit_manipulation: multi-gate operations run out of tokens
- Equation operators not in examples

## Predictions for Final Leaderboard
- 0.877 baseline to medal
- Top solutions need to maximize bit_manipulation + progress on cryptarithm
- May need base-64 bit representation to fit longer operations

**Why this matters for us:** Our GRPO approach was fundamentally wrong. THK proves SFT with perfect programmatic traces is the winning strategy. We should generate traces using THK's code and train on those.
