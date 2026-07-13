---
name: vastai-pod-skill-required
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: vastai\s+create\s+instance
  - field: command
    operator: not_contains
    pattern: skill-loaded
action: block
---

🛑 **STOP — Load `Skill(vastai-pod)` first.**

Last session burned **$7+** by skipping the validated workflow. The skill specifies:
- Image: `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404`
- mayukh18 wheels FIRST, then Komil overrides
- Blackwell patches BEFORE imports

Also read `vastai_h200_workflow.md` for exact pip-install order. Don't `pip install transformers/vllm` directly — they pull torch+cu13 and break torchvision.

**Before retrying `vastai create instance`:**
1. Invoke Skill(vastai-pod)
2. Verify balance with: `vastai show invoices | tail -1`
3. Confirm budget covers expected runtime

If you've already loaded the skill in this session AND verified budget, you can append `# acknowledged` to the command to bypass.
