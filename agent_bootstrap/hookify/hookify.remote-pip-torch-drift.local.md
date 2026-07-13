---
name: remote-pip-torch-drift
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: ssh\s+.*pip\s+install.*(transformers|vllm|torch|peft|trl)
  - field: command
    operator: not_contains
    pattern: --no-deps
  - field: command
    operator: not_contains
    pattern: --find-links
  - field: command
    operator: not_contains
    pattern: --no-index
action: block
---

🛑 **Remote `pip install transformers/vllm/torch/peft/trl` without isolation flags.**

Last session this pulled **torch+cu13** as a transitive dep, breaking torchvision (compiled for cu128) → cascading import failures across transformers/peft/trl → wasted ~$2 debugging.

**Use one of:**
- `pip install --no-deps <pkg>` — skip transitive deps (pin them yourself)
- `pip install --no-index --find-links /workspace/nemotron-pkgs/packages <pkg>` — only mayukh18 wheels
- Pin the version explicitly: `pip install "transformers==5.5.4"` (already-known-good)

**Reference:** `vastai_h200_workflow.md` Step 3+4. Install mayukh18 wheels first, then Komil overrides with `--no-deps`.

If you really need to pull deps from PyPI, append `# allow-deps` to bypass.
