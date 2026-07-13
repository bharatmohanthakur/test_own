---
name: no-openrouter
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: (gen_crypt_verified_traces\.py|openrouter\.ai|OPENROUTER_API_KEY)
  - field: command
    operator: not_contains
    pattern: KAGGLE_ALLOW_OPENROUTER=1
action: block
---

🛑 **OpenRouter is forbidden in this project.**

Do **not** run `scripts/gen_crypt_verified_traces.py`, call `openrouter.ai`, or use `OPENROUTER_API_KEY` in agent workflows.

**Use instead:** `scripts/format_crypt_verified_trace.py` + `data/crypt_verified_v4_batches/` fan-out.

If you have an explicit user override, prefix the command with `KAGGLE_ALLOW_OPENROUTER=1` (not recommended).
