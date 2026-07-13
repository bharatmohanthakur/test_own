---
name: feedback_data_verification
description: ALWAYS verify training data answers match ground truth. Wrong labels = worse model.
type: feedback
---

ALWAYS verify every training example's answer matches the ground truth from train.csv.
**Why:** We found that the perfect_cot_1000.jsonl had WRONG answers (6.41 instead of 57.0) because prompt matching was broken. This caused v8 to score 0.60 (DOWN from 0.69). Wrong labels actively harm the model.
**How to apply:**
1. After generating ANY training data, verify EVERY \boxed{} answer against train.csv ground truth
2. Never match prompts by prefix — use full prompt text or row ID
3. Print verification stats: how many match, how many don't
4. If even 1% of labels are wrong, fix before training
