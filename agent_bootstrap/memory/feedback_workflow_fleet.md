---
name: feedback-workflow-fleet
description: "Keep at least 3 workflows running at all times during competition campaigns; when one ends, plan and launch a replacement immediately"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a44fa216-6af5-41c1-82e4-1394a6f4d38d
---

During active competition work (Rogii, and any future campaign), the user wants **at least 3 workflows running concurrently at all times**. Whenever a workflow completes, immediately plan and launch a replacement — never let the fleet drop below 3 without an explicit reason.

**Why:** Wall-clock throughput is the binding constraint in competition sprints; serial investigation wastes the parallel capacity of the Workflow tool, and the user has repeatedly pushed for aggressive parallelization ("run 3-4 together", "parallelize fast", 2026-07-04/05).

**How to apply:** After reporting any workflow's verdict, check the live count; if < 3, pick the next-best items from the standing lever/idea queue (maintain one in memory or the session) and launch. Ideas need not be equally big — pair one big bet with small audits/intel refreshes. Related: [[feedback_autonomous]], [[rogii_competition]].
