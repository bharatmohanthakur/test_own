# Memory Index — Nemotron Reasoning Challenge

**60 active memories** (14 stale files + 6 consolidated-then-superseded in `archive/`).

## ARC-AGI-3 (separate competition — interactive agents)
- [arc_agi_3_planner.md](arc_agi_3_planner.md) — BFS planner over deepcopy-able deterministic games; ls20 1/7 verified (vs blind sweep 0/7)

## ACTIVE competition (July 2026)
- [rogii_competition.md](rogii_competition.md) — **ACTIVE July 2026**: Rogii wellbore TVT prediction ($50k, Aug 5), code comp, const baseline 15.6ft, harness in rogii/scripts/

## Baseline & strategy (top priority)
- [thk_v26_baseline.md](thk_v26_baseline.md) — **0.85 VERIFIED** (Apr 16), our current floor
- [thk_adapter_analysis.md](thk_adapter_analysis.md) — THK exact config + weaknesses (crypt/bit_manip/eq_guess)
- [thk_winning_solution.md](thk_winning_solution.md) — SFT-only, deterministic CoT, Tinker. GRPO is wrong path.
- [path_to_090.md](path_to_090.md) — full plan to beat THK (crypt CSP + 3-input gates)
- [competition.md](competition.md) — competition facts, deadlines, rules
- [training_methods_sota.md](training_methods_sota.md) — SOTA ordering: RAFT first, then token-priority SFT, GSPO last

## Mandatory workflow (load-bearing)
- [deploy_workflow.md](deploy_workflow.md) — 12-step deploy checklist
- [blackwell_training_fix.md](blackwell_training_fix.md) — 5-step Blackwell patch (RTX Pro 6000, B200)
- [errors_fixes.md](errors_fixes.md) — 28 documented errors + fixes
- [errors_runpod_v2.md](errors_runpod_v2.md) — RunPod-specific errors
- [logging_policy.md](logging_policy.md) — wandb mandatory; `report_to="none"` + nohup forbidden
- [testing_policy.md](testing_policy.md) — test 20-50 samples before full run
- [tracking_system.md](tracking_system.md) — **MANDATORY pre-submit** 60-prompt local bench
- [bench_vllm_workflow.md](bench_vllm_workflow.md) — Kaggle RTX 6000 vLLM bench recipe
- [kaggle_submit_workflow.md](kaggle_submit_workflow.md) — CPU-only submit kernel
- [kaggle_setup.md](kaggle_setup.md) — Kaggle notebook quotas, deps, metadata
- [kaggle_gpu_quirk.md](kaggle_gpu_quirk.md) — CLI push defaults to P100 trap

## Pod / cloud workflow
- [runpod_workflow.md](runpod_workflow.md) — VALIDATED full RunPod pipeline
- [vastai_h200_workflow.md](vastai_h200_workflow.md) — VALIDATED H200 setup
- [vastai_ssh_runpod_image_fix.md](vastai_ssh_runpod_image_fix.md) — **runpod/* image SSH publickey fix**: /root group-writable → sshd StrictModes refuses key; `chmod go-w /root` in `--onstart`. Don't pack ssh flags in a var.
- [cloud_gpu_strategy.md](cloud_gpu_strategy.md) — B200 = Blackwell = Kaggle fixes apply directly
- [vastai_usage_policy.md](vastai_usage_policy.md) — usage rules

## Data quality + generation
- [data_quality_findings.md](data_quality_findings.md) — quality > quantity; Grok 4.20 best teacher
- [data_quality_metrics.md](data_quality_metrics.md) — 6-tier quality checklist
- [donald_playbook.md](donald_playbook.md) — per-type template (100% solvable structured pipelines)
- [data_corpus_audit_2026_05.md](data_corpus_audit_2026_05.md) — **2026-05-09 audit**: thk_training_v30=USE, thk_training_v3=BROKEN, run `scripts/audit_training_data.py` before every train

## Apr 20-22 DPO attempts (ALL failed — crypt 2/10 immovable)
- [dpo_apr20_results.md](dpo_apr20_results.md) — step100 = 0.84 Kaggle (−0.01 vs v26); Crypt-DPO v1 ckpt25 rejected
- [crypt_dpo_v2_solver.md](crypt_dpo_v2_solver.md) — DPO with 242 CSP-solver-verified positive pairs → crypt 2/10 ALL 4 checkpoints. DPO can't move deterministic decoding path. Next: tool-use at inference, not more training.
- [track_a1_bo8_diagnostic.md](track_a1_bo8_diagnostic.md) — **2026-05-09**: bo8@temp=0.7 voting on v26 → +0.017 marginal lift, but crypt/unit oracle = greedy = 0/8 correct paths. Sampling won't fix crypt; only training will. Submission rule = temp=0 single-shot → bo8 can't ship anyway.

- [prompt_format_bug.md](prompt_format_bug.md) — **CRITICAL**: all local benches v1-v10 omitted the empty system block that training AND official Kaggle metric include; official eval also appends full boxed suffix
- [crypt_gradient_learning_diagnosis.md](crypt_gradient_learning_diagnosis.md) — **MEASURED** loss-down-no-learning: 90% scaffold learned, decision tokens (seen 1-2x) never enter weights; first argmax miss = first puzzle-specific token in every row; fix = repetitions-per-fact

## Regressions & disasters (learn from these)
- [v27_regression.md](v27_regression.md) — 0.82 regression; LR too high, lm_head reset
- [v28_regression.md](v28_regression.md) — 0.67 regression; packing w/o flash_attn_2 cross-contaminated binary
- [bench_v26_vs_v28.md](bench_v26_vs_v28.md) — per-type bench proves training dynamics are root cause
- [v29_disaster.md](v29_disaster.md) — 0.22 catastrophe; 3.2% correct `\boxed{}` format
- [v30_vs_v31_inference_findings.md](v30_vs_v31_inference_findings.md) — v30 = 100% cipher regression
- [b200_sft_disaster.md](b200_sft_disaster.md) — 0.26 from Kh0a config mismatch on THK data
- [grpo_b200_result.md](grpo_b200_result.md) — 0.47 GRPO regression from 0.72 SFT
- [grpo_blockers.md](grpo_blockers.md) — what stops GRPO from working
- [vllm_grpo_learnings.md](vllm_grpo_learnings.md) — **DEAD END** vLLM colocate + PeftModel + NemotronH
- [kaggle_grpo_komil_blocked.md](kaggle_grpo_komil_blocked.md) — Komil fix limits
- [boxed_weight_hurts.md](boxed_weight_hurts.md) — `boxed_weight` loss term regressed
- [trl_api_changes.md](trl_api_changes.md) — TRL 0.24 → 1.1.0 breaking changes

## GRPO recipes (when it IS used)
- [grpo_strategy.md](grpo_strategy.md) — Goldilocks data selection, post-SFT pipeline
- [grpo_working_recipe.md](grpo_working_recipe.md) — 27 sec/step with Komil fix (only known working config)
- [grpo_next_config.md](grpo_next_config.md) — next config to try if resuming GRPO
- [nvidia_grpo_recipe.md](nvidia_grpo_recipe.md) — NVIDIA official DAPO hyperparams

## Version results + recent sessions
- [v28_workflow.md](v28_workflow.md) — v28 refine procedure
- [v31_v32_results.md](v31_v32_results.md) — v31 GRPO + v32 GSPO run pointers
- [v38a_result.md](v38a_result.md) — 0.72 best SFT before THK v26 (Apr 12)
- [v40_grpo_learning.md](v40_grpo_learning.md) — v40 GRPO attempt notes
- [v40_pathA_config.md](v40_pathA_config.md) — **2026-06-01** Path A v40 SFT recipe (fresh LoRA + embed_tokens, r=64, max_seq=1536, batch=1×16)
- [v40_pathA_result.md](v40_pathA_result.md) — **2026-06-01 FAILED** v40 bench 0.43 vs v26 0.85; embed_tokens-LoRA + from-base = -0.42 regression
- [research_apr11.md](research_apr11.md) — THK public release; Kh0a 0.73 config details

## Claude behavior feedback (judgment memories)
- [feedback_pod_lifecycle.md](feedback_pod_lifecycle.md) — **CONSOLIDATED** stop-never-destroy + reuse + copy + split-train-eval-GPU rules
- [feedback_autonomous.md](feedback_autonomous.md) — pick and execute, don't ask "which option"
- [feedback_workflow_fleet.md](feedback_workflow_fleet.md) — keep ≥3 workflows running; replace immediately on completion
- [feedback_target_high.md](feedback_target_high.md) — target 0.90, not incremental
- [feedback_data_quality.md](feedback_data_quality.md) — quality > quantity (stated twice)
- [feedback_data_verification.md](feedback_data_verification.md) — verify every answer against ground truth
- [feedback_diagnose_first.md](feedback_diagnose_first.md) — diagnose failure modes before adding data
- [feedback_no_public_overfit.md](feedback_no_public_overfit.md) — Rogii: don't overfit public LB/G*; hedged final pair, local-only weights, ≤1 public-arbitrated direction/day
- [feedback_research_before_experiment.md](feedback_research_before_experiment.md) — GATE: papers + math + dead-list check before EVERY new experiment (2026-07-12)
- [feedback_research_flow.md](feedback_research_flow.md) — **THE OPERATING FLOW**: diagnose→gate→rank pool÷effort→validate→ship→engines loaded (master loop, 2026-07-12)
- [experiment_selection_lessons.md](experiment_selection_lessons.md) — 5 recurring experiment-design mistakes from Rogii failures + the check for each (READ BEFORE DESIGNING)
- [feedback_no_tmp.md](feedback_no_tmp.md) — never store scripts in /tmp
- [feedback_skills_before_pod.md](feedback_skills_before_pod.md) — **MANDATORY** load Skill(vastai-pod) before ANY pod work. Hookify rules added.
- [feedback_vastai_copy_verify.md](feedback_vastai_copy_verify.md) — Vastai copy is async; "initiated" ≠ "completed". Always verify destination size BEFORE stopping source. Hookify blocks `vastai stop` w/o `# copy-verified`.
- [feedback_codex_colleague.md](feedback_codex_colleague.md) — Codex (gpt-5.6-sol) = standing step-reviewer (stop-gate ON) + planning colleague for experiments/failures/next-steps; verify findings in-house before ledgering

## User
- [user_bharat.md](user_bharat.md) — user profile

## API keys / credentials
- [deepseek_key.md](deepseek_key.md)
- [minmax_api_key.md](minmax_api_key.md)
- [runpod_key.md](runpod_key.md)
- [tinker_key.md](tinker_key.md)
- [vastai_key.md](vastai_key.md)
- [wandb_key.md](wandb_key.md)
- [team_member_key.md](team_member_key.md) — Nency's Kaggle account

## Archive
20 stale/superseded files preserved at `archive/` (contradicted, blocked-never-happened, or consolidated into above). Never delete — useful for "why did we rule this out."
- [Model split: Fable plans, Opus codes](feedback_model_split.md) — pass model:'opus' on implementation agents
