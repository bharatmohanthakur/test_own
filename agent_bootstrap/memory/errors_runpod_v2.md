---
name: RunPod and submission errors (session Apr 9)
description: New errors discovered during v34/v36 RunPod training and Kaggle submission. Prevents repeating in future sessions.
type: feedback
---

# Errors (Apr 9, 2026 session)

## Submit kernel loose files → dialog confusion
Copying adapter_config.json + adapter_model.safetensors as LOOSE files to /kaggle/working alongside submission.zip causes the "Submit to Competition" button to appear on the wrong file (adapter_config.json instead of submission.zip). The dialog then says "Could not find provided output file".
**Fix:** Use the v31 submit pattern: copy to a SUBDIR of /kaggle/working, only submission.zip at the top level. Use `code_file: train.py` and `enable_internet: false`.
**Why:** Kaggle's output page shows Submit buttons per file; loose adapter files get their own buttons.
**How to apply:** Always copy the working v31 submit notebook pattern verbatim, just change the dataset source name.

## Kaggle CLI submit → 400 Bad Request
`kaggle competitions submit -f submission.zip` uploads the full 3GB file but gets rejected with 400 error. This competition requires notebook-based submission.
**Fix:** Must submit via notebook Output tab → "Submit to Competition" button.
**Why:** Competition rules require the adapter to come from a kernel output, not direct upload.
**How to apply:** Never try CLI submit for this competition. Always use the notebook flow.

## GRPO enable_thinking required for v34c+ adapters
v34c adapter generates only ~8 tokens per completion without `<think>` tag. v22 adapter worked without it because v22's training was less dependent on thinking mode.
**Fix:** Patch tokenizer.apply_chat_template to add `enable_thinking=True` by default.
**Why:** v34c was trained with Unsloth's apply_chat_template(enable_thinking=True). Without it, the model emits <|im_end|> almost immediately.
**How to apply:** Add the patch after tokenizer setup in ANY GRPO script that uses a v34c+ base adapter.

## use_vllm=True crashes on Nemotron
TRL GRPOTrainer with `use_vllm=True` fails: "NemotronHForCausalLM has no attribute vllm_engine". Nemotron's Mamba-Transformer hybrid doesn't support TRL's vLLM integration.
**Fix:** Must use `use_vllm=False`. Generation is slow (~12 min/step) but works.
**Why:** TRL expects model.vllm_engine attribute which PeftModel/Nemotron doesn't have.
**How to apply:** Never set use_vllm=True for Nemotron GRPO. Accept slow generation or skip GRPO entirely.

## Attribution float precision
`abs(45.45 - 45.44) = 0.01000000005 > 0.01` causes false mismatches in attribution.py.
**Fix:** Use `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` and prefer precomputed `_ok` flags.
**Why:** IEEE 754 float representation.
**How to apply:** Already fixed in tracking/attribution.py.
