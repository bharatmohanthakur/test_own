---
name: feedback_vastai_copy_verify
description: Vastai copy is async; "initiated" is NOT "completed". Always verify destination file size before stopping source pod. Burned 30 min + $1 on 2026-05-03 by not checking.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Rule**: After ANY `vastai copy A:src B:dst`, before stopping pod A:
1. SSH into B: `ssh -p $PORT root@<host>`
2. Verify destination size: `du -sh /workspace/<dst_path>`
3. Compare to source: should match within 5%
4. List files: `find /workspace/<dst_path> -type f | wc -l`
5. ONLY THEN consider stopping pod A

**Why**: 2026-05-03, after training SFT on Tennessee:
- Issued `vastai copy 36051231:/workspace/adapters/ 36057330:/workspace/adapters/`
- CLI returned "initiated" within 1 second
- I stopped Tennessee 5 seconds later
- Spain `/workspace/adapters` was empty (0 bytes)
- Tennessee restart queued at host capacity, 30+ min wait
- Spain idle cost ~$1

**The skill memory** (`vastai-pod` skill) claims "vastai copy works between instances (incl. stopped ones)". This is WRONG or context-dependent — in practice, copy from an exited pod did not transfer files.

**Trust path** (from `vastai-pod` skill "Sequential workflow"):
- Save adapter to Kaggle dataset → stop source → download on destination
- This is the proven path

**Hookify guard added**: `.claude/hookify.verify-vastai-copy.local.md` blocks `vastai stop` unless command appended with `# copy-verified`.
