---
name: feedback_diagnose_first
description: Understand model failures FIRST, then create targeted data to fix them. Don't throw generic data.
type: feedback
---

Don't throw more generic data at the model. DIAGNOSE issues first, then create targeted fixes.
**Why:** The base model was already trained on Cascade-2 data — retraining on it won't help. User wants to understand WHAT the model gets wrong and WHY, then create specific data to fix those issues.
**How to apply:**
1. Run model on training set, analyze errors by category
2. Understand failure patterns (wrong operation identified? wrong calculation? format issue?)
3. Create targeted training data that specifically addresses each failure mode
4. Train on diagnostic-driven data, not generic bulk data
