---
name: feedback_no_tmp
description: NEVER store scripts in /tmp - use project working directory. /tmp gets cleaned up and work is lost.
type: feedback
---

NEVER store training scripts or important files in /tmp. Always use the project directory.
**Why:** /tmp gets cleaned up automatically. User lost all previous training scripts because they were in /tmp. This wasted significant time.
**How to apply:** Store all scripts in `/Users/bharat/Downloads/kaggle/scripts/` or similar project subdirectory. Only use /tmp for truly temporary files that are immediately consumed.
