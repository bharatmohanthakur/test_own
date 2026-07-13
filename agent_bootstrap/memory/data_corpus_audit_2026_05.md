---
name: data_corpus_audit_2026_05
description: 2026-05-09 audit of all training jsonl files. thk_training_v30 is the only v26-grade clean corpus (100% boxed-after-think, 93.6% truth-match). thk_training_v3 is broken (3.2% compliance).
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Audit run via** `scripts/audit_training_data.py` on 2026-05-09.

| File | rows | boxed-after-think | truth-match | dups | verdict |
|---|---|---|---|---|---|
| `thk_training_v30.jsonl` | 10,143 | **100%** | **93.6%** | 271 | ✅ USE THIS |
| `thk_crypt_clean.jsonl` | 918 | 100% | — | 0 | ✅ clean (no IDs) |
| `crypt_sft_v2_clean.jsonl` | 572 | 100% | — | 0 | ✅ post-fix |
| `crypt_sft_v1_mixed.jsonl` | 572 | 51% | — | 0 | ❌ broken (pre-fix) |
| `thk_training_v29_mixed.jsonl` | 1,538 | 35% | 0% | 269 | ❌ |
| `thk_training_v3.jsonl` | 16,720 | **3.2%** | 0% | 9,417 | ❌ DO NOT USE |

**Why:** v26 was trained on a clean corpus equivalent to v30. v3 (the one auto-loaded by some scripts) has the boxed answer INSIDE `<think>` tags — completion-only loss masks the boxed token away. v29 disaster (0.22 score) was caused by training on this broken format.

**How to apply:**
- Default to `thk_training_v30.jsonl` for any from-base training.
- For refinement (from v26): use `thk_training_v30.jsonl` filtered to weak categories + V3.1 distilled CoTs for unit/gravity.
- For crypt-specific work: use `crypt_sft_v2_clean.jsonl` (572 rows, 292 solver-verified + 280 preservation, 100% format).
- Run `scripts/audit_training_data.py <file>` BEFORE every training. Verdict must be PASS (≥95% boxed-after-think).

**Bug pattern to watch for:** scripts that build training data by mixing categories may emit `\boxed{}` inside `<think>` if the source CoTs put it there. The model trains, but loss is computed on `<think>` content where the boxed lives — so the model never learns to emit `\boxed{}` AFTER `</think>` at inference time. Always run audit post-build.
