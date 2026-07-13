---
name: rogii-competition
description: "Rogii Wellbore Geology Prediction (Kaggle, $50k, deadline Aug 5 2026): predict TVT along horizontal wells after Prediction Start from GR logs + typewell; metric RMSE; CODE competition (CSV submit rejected); const-hold baseline = 15.6 ft local RMSE"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**Competition:** https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction —
Featured, $50,000, deadline **Aug 5, 2026**, ~4,100 teams. Started on it 2026-07-02.

**Task (geosteering):** each horizontal well has MD/X/Y/Z/GR per foot + `TVT_input`
(true vertical thickness, given until the Prediction Start, blank after). A paired
typewell gives the reference GR-vs-TVT signature. Predict TVT for the blank rows.
**Metric: RMSE** over all predicted points. id format `well_rowindex`.

**Data:** `rogii/` — 773 train wells (full TVT ground truth + formation-top columns
ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA + a PNG cross-section each), 3 visible test wells
(also present in train WITH answers → visible test is a placeholder; real eval is the
hidden rerun). Train wells carry the same TVT_input masking → perfect local harness
(`rogii/scripts/eval_baselines.py`).

**It is a CODE competition** — direct CSV submit returns 400. Submit via kernel that
reads `test/*__horizontal_well.csv` generically (hidden wells differ) and writes
`/kaggle/working/submission.csv`. Kernel: `rogii/notebooks/submit_const/`
(bharatmohan/rogii-const-baseline).

**Baselines measured (150 train wells, 755k points):**
- const-hold last known TVT: **15.59 ft** ← best trivial; first submission
- linear slope extrapolation: 196.8 ft (explodes over long laterals)
- damped slope (best half-life 200ft/tail 100ft): 16.7 ft — still worse than const
→ TVT is flat/mean-reverting at lateral scales; slope carries almost no signal alone.

**Organizer hints (from task PPTX):** (1) the well's own pre-PS GR has better
resolution than typewell GR — correlate the tail against the pre-PS section, not just
the typewell; (2) neighboring wells' geological dips correlate spatially (XYZ given;
offset-well features); (3) GR≈constant ⇒ TVT constant; GR matching typewell signature
up/down ⇒ TVT increasing/decreasing.

**Obvious next approaches:** dynamic time warping / correlation of GR against
typewell + own pre-PS log to get local dTVT/dMD, spatial prior from offset wells,
then a learned model (per-point dTVT regression) on the 773 train wells.

**Submission mechanics (solved 2026-07-02):** code comp — submit the KERNEL, not a CSV:
`kaggle competitions submit -c rogii-wellbore-geology-prediction -k bharatmohan/rogii-const-baseline -v <version> -f submission.csv -m "..."`
All three of -k -v -f are required ("This competition requires an output FileName for
Notebook Submissions"); CLI must be ≥2.2.3 (`uv tool upgrade kaggle` — it's a uv tool,
plain pip upgrade doesn't touch it). Kernel input mounts at
/kaggle/input/rogii-wellbore-geology-prediction/ (walk defensively; v1's naive glob
found 0 files while v2's os.walk found them).

**Modeling log (2026-07-02, day 1):**
LB arc: const 15.883 → dip-blend 11.818 → +knot-GR MAP **9.169** (leaders 5.26-6.3).
Local↔LB tracks within ~0.7 ft every time (150-well leave-self-out harness).
- **EXACT IDENTITY: TVT = (ANCC_top − Z) + c_well, std 0.01 ft** (all 6 top columns
  parallel). So predicting TVT ≡ predicting the ANCC surface along the path, and the
  dip field of r=TVT+Z IS the ANCC surface gradient — same signal, two views.
- Surface is cross-well consistent (median 1.9 ft at <150 ft) but IDW/plane
  interpolation at real spacing (~300 ft) drifts ±20-47 ft → raw surface interp (56 ft)
  and anchored-relative version (60 ft) both LOSE to dip integration. Leaders' edge is
  likely proper geostat surface modeling (kriging) and/or stronger GR fusion.
- GR fusion learnings: windowed fixes and pointwise Viterbi both HURT (aliasing +
  noise-chasing); oracle test showed truth beats prior 12/12 on whole-path emission →
  knot-level DP (600-ft piecewise-linear delta, exact segment costs) + **MAP penalty
  scaled by measured prior σ(s)=2.5+0.0022s (clip 12)** is what finally worked:
  10.76 → 9.878 local. Iteration and shrinkage: saturated, no gain.
- Error concentration: top-5 wells = 36% of squared error; worst well fb03ae90 = 21%
  alone. Mode: dip changes abruptly right at PS + quiet GR tail (std 12) + dmax=36 cap
  < actual 50-60 ft error. No faults observed (max true jump 2.3 ft).
- Kernel: bharatmohan/rogii-const-baseline v4 (dip prior + exact knot refine). All
  models in rogii/scripts/: eval_baselines, model_dipvec, model_gr, model_knots,
  model_viterbi, model_surface.
**Next queue:** adaptive GR trust (running), local kriging interpolator, self-augmented
surface strip from pre-PS, per-well ensemble weighting, worst-mode containment.

**Day-1 close (2026-07-02):** LB arc 15.883 → 11.818 → 9.169 → **8.022** (v5, best public)
→ v6 precision fusion 8.463 public despite better local (7.929 vs 8.248) — public split
is few wells, NOISY; **local 150-well CV stays the decision metric**, keep v5+v6 as final
pair candidates. 5/5 daily submissions used.
- HMM (model_hmm.py, agent-built): forward-backward over absolute TVT, dip transitions
  (q=0.007, block 24 ft, emis_w=0.07 — also optimal inside fusion), posterior mean+sd;
  SOLVES 7/10 worst wells (20 ft → 1-7 ft). return_sd → precision fusion w=σ_kn²/(sd²+σ_kn²).
- Kriging (model_kriging.py, agent-built): **per-well ANCC datum bias 7-10 ft** discovered
  → leveling network (Huber-IRLS over 742 close-pass edges); power variogram
  γ=3.73+0.0008·h^1.70 (no finite range — why IDW/exp/sph all failed); neighbor cap
  per source well essential (redundancy); σ-adaptive blend 10.561 standalone.
- FAILED at 150-well scale (60-well wins that flipped): own-dip-near-PS transition
  correction (8.61 vs 7.929 — pre-PS slope is steering noise); hand-gated adaptive GR
  params; iteration; shrinkage. Lesson: NEVER ship on 60-well numbers.
- In flight: kriging-transitions HMM (agent), learned residual stacker (agent).

**Day-2 ultracode rounds (2026-07-02 evening):** local CV 6.682 → **6.460** (CV-error-field
transition gate, exp_cverr_gate.py remax_p05: s2=max(sig2_krig,0.5*v_emp)*clip(sqrt((e2k+4)/(e2d+4)),1/3,3);
key insight corr(v_emp,|err|)=0.55 vs 0.10 for kriging's own sigma) → **6.3945 CONFIRMED**
(exp2_compose.py: + fusion recal w=sig_kn²/((1.7255·sd)²+sig_kn²), rho refit on 623 non-eval).
All wins adversarially verified (clean-rebuild reproduction + leakage audits). Free sub-threshold:
gate jA (eps 8, h0 150) −0.014 zero-overfit. FAILED honestly at 150: emission sharpening
(dev-60 win, one +23ft blowup), neighbor-blended reference (blurs thin-bed GR), per-well risk
models (again). CONVERGENT DIAGNOSIS: residual damage lives in TVT bins outside own pre-PS
coverage where the typewell-only reference can sit ~8ft off (8c167025: 59% uncovered, 24.6ft,
emission prefers the wrong path). Round-3 in flight: v8 packaging (kernel_driver2.py, compose+jA)
+ uncovered-zone-local fixes (per-bin sigma/neighbor substitution/per-well coverage guard).
v7 (6.682) still queued behind the 5/day submit limit — retry watcher live.
Discipline that saved us repeatedly: dev-60-only tuning, 150 validation, adversarial verify
with default-REFUTED, honest-negative reporting.

**Rounds 3-4:** local best now **6.3783** (compose+jA+guard05; v8 kernel shipped = 6.3878
variant, kernel_driver2.py self-contained). DO-NOT-RETRY: posterior-coverage-keyed sd
inflation (u_post low precisely where truth diverges — unobservable; where u_post high,
HMM leg is the GOOD one). 8c167025 class fully characterized: typewell reference locally
~8ft mis-shifted in own-log-uncovered zones; BOTH legs poisoned; neighbor curve blending
fails (blurs thin-bed detail); per-bin sigma inflation fails (safe-haven artifact without
log-normalization). Next structural lever: map the typewell mis-shift field itself (train
wells' labeled logs vs their typewells = ground truth of reference error; krige/extrapolate).
Gains asymptoting ~0.01/round via re-weighting; only new signals move it now.

**CRITICAL (2026-07-02 late): local→public TRANSFER BREAK on the compose family.**
Simple HMM+knot blend: local 8.248 → public 8.022 (fine). Compose v8: local 6.3878 →
public **8.322** (+1.9!). Precision fusion: 7.929 → 8.463. Pattern: the more the pipeline
leans on leave-self-out neighbor structure (kriging surface, CV-error fields, dip field),
the worse the public transfer — hypothesis: hidden test wells are spatially separated
from the train field (edge/gaps), so neighbor signals are weaker than LSO simulates.
Diagnosis in flight: leave-REGION-out harness (exp9_lro.py, R∈{0,500,1500}) comparing
const/simple-blend/v7/v8/v9 on same wells + per-component degradation + field geometry.
If confirmed: re-tune gates/fusion under LRO-1500 (hidden-test-robust constants,
distance-adaptive fallbacks) → v10. LADDER (local LSO): 6.682→6.460→6.3945→6.3878→
6.3783→6.313 (v9, packaged, kernel_driver3.py, shipping for the 2nd calibration point).
Team account (nencybansal) active for submissions: 10/day total. MUST TEAM-MERGE the
two accounts in Rogii before merger deadline (user-approved arrangement from Nemotron).

**v10 SHIPPED (2026-07-03 00:15 UTC): LRO-retuned robust config** = kernel_driver4.py:
S0=6, GAIN_V=4, RHO=3.2, Q_MULT=8, DAMP_DIST=2500 (kernel_driver3 otherwise unchanged).
Selection: 1296-config grid under LRO-500 (seed-777 40w) with LRO-1500+guard constraints
→ HELD-OUT validated (disjoint 40w seed-1234): B 6.16/7.47 vs blend 7.55/8.10 — gains
persist out-of-sample, worst-well 17.5 vs 29.0. Min-regret pick: within 0.46 of best on
every subset×radius while every reference is >1.2 worse somewhere. NOTE: v9-compose is
pooled-best on the held-out subset but flips sign vs blend across subsets (matches its
public failure) — 40-well subsets are volatile; min-regret beats argmin.
Adaptive homotopy (z=logistic(LOO krig sigma), compose↔blend) CONFIRMED by adversary
(bit-level radius-plumbing corruption test) at 6.654 LRO-500 — kept as backup config.
Public prediction for v10: if LRO-500 ≈ hidden regime, expect ~7.3-7.8 (blend scored
8.022 at LRO-500 7.07; v10 is -0.9 vs blend at LRO-500 on tuned AND -1.4 on held-out).

**v10 SCORED 7.689 public (2026-07-03) — NEW BEST, rank ~#22/4100.** Predicted window
7.3-7.8 hit dead-center. LRO-500 → public ordering now 4/4 (blend 7.07→8.02, v8 7.61→8.32,
v9 7.45→8.42, v10 6.23→7.69). LRO-500 IS the decision metric; offset +0.7..+1.5.
Top-10 ≈ public 6.3-6.5 → need LRO-500 ≈ 4.8-5.2. Next: LRO-regime anatomy (sparse-regime
failure classes differ: kriging leg weak, dip spatially biased), homotopy with retuned
dense endpoint, v10×v9 ensemble hedging, then new levers per anatomy.

**v11 public 7.732 (worse than v10's 7.689 despite pooled LRO 6.041 vs 6.198).** Cause:
v11 differs from v10 ONLY on sigma>25 wells (3/80 locally) — thin per-well support = lottery
ticket on small hidden set; the gate fired somewhere harmful. NEW SHIP RULE: broad support
required (>=10 wells improving, gains not concentrated in <=3). v10 remains champion
(7.689, ~#22). Joint ISOL×L2 no-shipped (branch halves serve mutually exclusive wells;
future = per-well damping gate). L1 shrinkage DEAD with mechanism: dip increments carry
real structure; drift = bias in the estimate at range → improve the estimator (round-11:
better trend model + azimuth-conditioned LSQ, in flight). exp13_l2/exp14 configs banked.

**v12 public 7.950 (2026-07-03) — second consecutive local-win/public-loss.** Public:
v10 7.689 < v11 7.732 < v12 7.950; local pooled-80: v10 6.198 > v11 6.041 ~ v12 6.054.
DIAGNOSIS: 80-well pool EXHAUSTED after ~30 variants (selection noise owns residuals;
SE-concentrated gains = lottery tickets even with broad count-support). STRATEGY PIVOT:
v10 stays champion; NEW FINAL GATE = LRO-500 over 300+ wells with paired bootstrap CIs,
ship only >0.15 big-pool wins; slot discipline (deadline Aug 5, 5 weeks). Big-pool
arbitration of v10/v11/v12/v12-nodg in flight (exp17_bigpool.py) incl. 80-pool-vs-big-pool
delta correlation (measures how much the small pool lied). Banked levers for big-pool
rounds: per-well damping gate, extreme-range class, dgate-free v12, LWR trend (universal).
TEAM-MERGE the two accounts before merger deadline — REQUIRED.

**BIG-POOL ARBITRATION (349 wells, exp17): v10 IS THE TRUE CHAMPION.** Pooled LRO-500:
v10 6.994 | v11 7.013 (+0.019, CI noise) | v12 7.346 (+0.352). THE SPLIT: on the old-80
tuning wells post-v10 configs are −0.15 "better"; on the 269 FRESH wells ALL are worse
(v11 +0.06, v12 +0.48). Every post-v10 gain = 80-pool overfit. Public LB was truthful
all along. LESSON (top-tier): dual-subset + adversarial verify ≠ enough once subsets are
consumed by iteration; ship gate = paired-bootstrap CI excluding 0 on NEVER-CONSULTED
wells at scale. NEW PROTOCOL: exp17's 349-pool minus a locked 100-well final holdout
(never consult until ship decisions) = the working pool; v10 base for all future levers.
FINAL PAIR candidates as of now: v10 (7.689) + blend (8.022, structurally different).

**v13 public 8.102 (2026-07-04) — even the CI-real honest-gate win inverted publicly.**
The locked holdout (−0.02, CI∋0) was the truthful predictor; the 249-pool at LRO-500 was
not. PATTERN: v10 (7.689) publicly beats ALL 6 successors regardless of local evidence
quality. Metric under indictment — 7 (config→public) points now exist to CALIBRATE the
local metric itself (round-16 in flight: radius profiles of all shipped configs + metric
fit incl. blends and holdout-weighted). Suspect: hidden wells at deeper separation than
LRO-500 simulates; v13's retuned constants may invert with radius (tuned@500 only).
Round-15 compound leads (w_own0.7, HMM b16, sigslope) held pending the corrected metric.
Public ledger: v10 7.689 | v11 7.732 | v12 7.950 | v13 8.102 | blend 8.022 | fusion 8.463.

**C2 PROBE = 7.609 PUBLIC (2026-07-04) — NEW BEST, first config to beat v10 (7.689).**
C2 = v10 constants + own-log emission w_own=0.7 + HMM block_ft=16 (kernel_driver8).
The pre-registration protocol WORKED: config frozen before virgin test (−0.39, CI∋0,
formally no-ship), shipped as declared probe on two-pool converging evidence. The
own-log emission mechanism is real at public scale. C2 = NEW BASE for all future
levers; final-pair candidates now C2 + v10 (or C2 + blend for diversity). Next:
re-litigate compound queue (sigma-slope gate, knots) on C2 base vs remaining 224
virgin wells, pre-registered probes only, 0.4ft-ish evidence bars.

**Round 18 (exp25, 2026-07-04): confirm phase no-ship, g0 probe launched.** On 200-pool
CV: cap=4 dominant (all 9 cells, 5/5 folds), w_own=0.8>0.7 everywhere, guard-drop g0
CI-real (−0.076). One-shot confirm on 150 FRESH wells: cap4 pick collapsed to −0.11
(P=0.42, huge CI); g0 held −0.14 (P=0.165); v10 +0.19 worse than C2 (calibration
replicates public). TWO-POOL EVIDENCE for g0 (GUARD_B 0.5→0, one-line change) = same
pattern as the successful C2 probe → shipped as pre-registered probe kernel_driver9
(parity 50/50 wells vs experiment values, smoke clean). σ-slope gate DEAD on C2 base
(v13-constants artifact). BUG: exp24_virgin.rebuild_emis_ctx memoizes by well ignoring
w_own — never reuse for multi-mix sweeps (fixed in exp25_tune). Pool state: 200-pool
2 evals deep; 150 of the 224 fresh now consumed (1 eval); 74 wells still virgin.

**g0 probe = 7.629 (2026-07-04): flat vs C2 7.609 (+0.02). Guard question CLOSED —
GUARD_B stays 0.5, C2 unchanged champion.** Confirms exp23 floor: sub-0.4 local deltas
are publicly unresolvable; probes below the floor are coin-flips (this one cost a slot
to close a branch). Discipline: probe only fresh-pool CI-real wins ≥0.25-0.4. Public
ledger: C2 7.609 | g0 7.629 | v10 7.689 | v11 7.732 | v12 7.950 | v13 8.102.

**Round 19 (exp26): emission class-gate DEAD with mechanism.** The cap4/w0.8 family's
±5ft well swings = HMM decoding bifurcation (loosened cap snaps decode onto correct
plateau OR spurious GR band; basin depends on contrast-vs-true-TVT geometry =
UNOBSERVABLE at test time). Reference-quality hypothesis FALSIFIED (blowups have
perfect own coverage). All observables AUC~0.5 out-of-pool; in-pool gate −0.54 was
pooled-arithmetic overfit, +0.01 on fresh. Per-well C2-vs-TUNED arbitration is NOT
achievable without truth. Low-expectation future idea: consensus/multi-start HMM
(cap3+cap4 decode agreement) — but its proxies already failed to transfer.

**Round 20 (exp27): ALL FIVE structural mechanisms DEAD with causes.** (1) Integrable/
curl-free dip: +0.35..+5.5 harm — drift is genuine local dip variation, surface can't
represent it. (2) EM reference self-refinement: +0.27 harm — pass-1 lock-in poisons
reference; only 2/13 low-coverage wells win. (3) Per-well typewell from k-NN offset
wells: +3.3 CATASTROPHIC — the ORGANIZER-PAIRED typewell encodes per-well geology that
offset wells can't reconstruct (even exact-datum ceiling blows up); w_own>0.5 ≠ typewell
replaceable. (4) Cross-test-well pseudo-obs: real test wells 15-45k ft apart (kriging
weight~0); truth-oracle ceiling on simulated pairs ≈ +0.07 coin-flip. (5) Per-point
fusion: −0.18 below floor; per-point ≈ per-well (−0.025, p=0.36). Architecture is at a
LOCAL OPTIMUM — 22 mechanism classes closed. Round 21 (exp28): prediction-level
ensembles of C2/v10/v13/TUNED (median esp. — tames bifurcation blowups), the one
cross-architecture move never tested.

**ROUND 23 RESEARCH BREAKTHROUGH (framing): posterior-mean decoding is INADMISSIBLE at
multimodal posteriors** (Lember & Koloydenko arXiv:1802.01630) — our FB mean sits in the
valley between geological branches = the bifurcation tax, the ±5ft classes, why round-19
gating was unobservable, why E4 averaging helps. INDUSTRY CONVERGES: StarSteer (host,
label generator) = piecewise constant-dip segments + stretch/squeeze GR match, ONE valley
picked; H&P patent US11920441 = misfit heatmap, enumerate ALL consistent interpretations;
Cybersteering URTeC-2019 = Viterbi over (position×dip) lattice; NORCE PluRaListic = 100s
of pruned hypotheses in ROGII's own benchmark. PRESCRIPTION: collapse LAST not first —
list-Viterbi top-K paths + arbitrate by emission loglik + kriging agreement (independent
leg as arbiter). Round-24 (exp30_listvit) implementing. Also banked: IMM mixture-of-dips,
bifurcation detector + branch split, particle smoother w/ cluster readout, DTW/RGT
emission, conditional-sim kriging ensembles voting. Full citations in wf_832f3901 output.

**ROUND 22 INTEL (game-changing, 2026-07-04):** (1) PUBLIC NOTEBOOK "ROGII LB7295
Public Rebuild" rescoring ~7.15 < our C2 7.609 — PF 600 particles×128 seeds + beam
search + multi-scale NCC + per-well selector + visible-prefix self-calibration; code
in hand (round-25 reproducing+fusing). (2) Rank-2 (5.44) per-well data ONLY — no
field leg; our kriging+leveling is the open differentiator. (3) Hidden test ~200
wells, public ~50 (26%); public deltas <0.12 = seed noise. (4) souldrive bimodal-datum:
2nd GR minimum on 49% of wells, cost-mode right only 48.8% → optimal = p-weighted mean
of modes, p from heel-calibrated softmax (80-84% localization). (5) look-ahead legal;
6 formations = 1 surface + 5 constants; ~10% wells = 40% of SE (irreducible tail).
Top-15: 5.26-6.33. Round-22 dead: alpha-gate (E4 0.5 optimal; need DECORRELATED 3rd
parent), tail-defense (detect AUC 0.888 but direction unobservable), DTW leg (10x weak).

**E4 PROBE = 7.558 PUBLIC (2026-07-04) — NEW BEST, 2nd consecutive winning probe.**
E4 = mean(C2, TUNED w0.8/b20/cap4) = kernel_driver10. Probe methodology validated
twice (C2 +0.08, E4 +0.05): replicated two-pool evidence + pre-registration = public
transfer. Ladder: E4 7.558 | C2 7.609 | g0 7.629 | v10 7.689. E4 = new base + new
final-pair anchor. In flight: round-24 list-Viterbi decoder, round-25 public-PF-stack
(7.15 free notebook!) reproduce + fuse with our field leg (F1=mean(PF,E4) etc).
Next milestones: PF fusion could reach ~6.8-7.0; top-15 = 6.33.

**Round 24 (exp30) list-Viterbi: pre-registered STOP with COMPLETE THEORY.** Census:
32% of wells genuinely multimodal (FB marginal bimodal >3ft over >20% tail); oracle
best-of-5 headroom ~1.7ft there (~0.5 pooled). But NO signal picks the correct mode;
kriging ANTI-selects (weak offset control CAUSES the multimodality → kriging weakest
exactly where needed). For RMSE the FB posterior mean IS Bayes-optimal absent a mode
selector. UNIFIED: E4 wins by widening an overconfident posterior; souldrive p-weighted
mean = same; rank-2's 5.44 = better MATCHING (fewer spurious modes at source, dodges
the mode-tax). Machinery reusable: exp30_listvit.py (bitwise FB parity, k-best, census).
The open problem = a mode-selection signal informative at low-offset-control wells;
until then decoder stays FB-mean. PF stack (round 25) attacks the source instead.

**Round 25 (exp31): public PF stack reproduced (200 wells cached).** Pooled 9.243 on
our LRO-500 200-pool — WORSE than C2 8.63/E4 8.30 (its 7.15 public = full notebook w/
LightGBM leg + friendlier well mix). User warning validated: never rebase onto it. BUT:
per-well error corr pub-vs-E4 = 0.205 (C2-vs-E4 0.990) = the decorrelated parent the
alpha-gate autopsy demanded. ORACLE min(PF,E4) = 5.10 pooled (3.2ft routing headroom);
PF wins 47/200 wells INCLUDING catastrophes (b32902c0 14.2 vs 17.2) — and sd_c2 detects
those at AUC 0.888, so routing may be learnable (unlike the bifurcation direction).
Round-26 (exp32): router + per-point fusion, CV + fresh confirm. Files:
scripts/exp31_public.py, scripts/public_stack/{main,fork1,fork2}, exp31_public.json.

**Rounds 29+30 closed (2026-07-04 night).** (29) Bimodal p-weight: souldrive's 48.8%
replicates (48.9%) but their lift is vs naive argmin — our FB mean ALREADY soft-mixes
modes per-node; reweight family oracle = −0.12 vs C2, +0.21 vs E4. Soft>hard confirmed
(−0.02 vs +0.47) but E4 stands. (30) NCC emission DEAD, measured: bedding-parallel
laterals → MD window spans 0.2-2 ft TVT → NCC noise-dominated; affine invariance
discards GR LEVEL (the signal); census GREW 65→167. Public NCC survives only as a weak
selector feature. FUTURE matching work = time-varying gain/offset drift-tracking in
Gaussian leg. Fleet: 26 (PF router) + 31 (tabular) computing; 28 (prefix self-cal)
shards ~1h. E4 7.558 champion.

**Round 26 (exp32): PF-ROUTING DEAD — public stack is well-to-well UNSTABLE.** PF 9.24
on 200-pool but 13.09 on fresh-150 (E4 stable 8.30/8.07); oracle headroom 3.2→1.1 ft;
CV-real router (−2.05!) inverted to +2.41 on fresh — the 200-pool held E4-catastrophes
PF fixes, fresh holds PF-catastrophes. NOTHING PF-BASED SHIPS. Survivor (below bar):
median(PF,c2,tuned) −0.19 sig on 200 / −0.09 ns fresh. If routing revisited: label =
'E4 catastrophe', fallback must be provably-never-catastrophic (half_const), NOT PF.
POOL LEDGER: fresh-150 now consumed (2 evals); 200-pool heavily mined; 74 virgin locked.
Remaining live: exp33 prefix self-cal (shards), exp36 tabular.

**Rounds 32-34 closed (2026-07-05 early):** (exp37) GR drift REAL (med 0.245σ/kft,
cached per-well in exp37_drift.json as routing features) but UNTRACKABLE from inside —
self-aligned recal = regression dilution/lock-in (3rd emission negative: p-weight, NCC,
drift → C2 emission locally optimal; gap = ensemble/parents). (exp38) Containment DEAD
completely: even own-entry const-hold is catastrophic in the fired regime (catastrophe
wells = far-from-entry drift by definition); no safe anchor exists in current legs.
(exp39) Third parent: pipeline cube decorrelation impossible (corr≥0.98); M3b
(mean-of-3 w/ P3b = v13-retune-fusion-on-C2-emission) −0.148 misses gate by 0.002 but
won 5/5 folds — BANKED for shared virgin consult; P3b alone 8.203 = best single config.
PENDING: exp33 self-cal fresh acid test → shared virgin-74 consult (E4+selfcal+M3b).

**exp33 self-cal fresh acid test: DEAD (2026-07-05).** All candidates +0.45..+3.3 worse
than E4 on fresh-150. CAUSE: top1_agree=0.28 (chance=0.2) — the labeled PREFIX and the
blank region are DIFFERENT REGIMES (build section w/ control vs deep lateral); a well's
own prefix cannot validate its tail predictor. + PF instability poisons pf-blends.
SELECTION THEORY COMPLETE: observables router, prefix self-cal, alpha gate, class gate
ALL fail transfer — no per-well selection signal exists in any tested form; E4 uniform
averaging stands. Live: exp40 segment decoder + exp40_regime (resumed; M3b/X4/R4 get
the quasi-fresh 349 arbitration). Ship path: only 349-confirmed candidates → virgin-74.

**exp40 SEGMENT DECODER: BREAKTHROUGH — first model to break the 0.98 corr wall.**
Piecewise constant-dip Viterbi DP (StarSteer/Cybersteering class) on the C2 emission:
per-well point-error corr vs E4 = 0.09-0.2 (pipeline cube floor 0.98). Standalone wild
(27ft; pure-GR segments untrustworthy) → mean/self-cal weighting FAIL, but
median(c2,tuned,SEG s0:k15d10:d3) = 8.091, CV-honest −0.204 vs E4, CI excl 0 (200-pool).
Different smoothness class = errors in different places = median vote monetizes.
ARBITRATION CHAIN RUNNING (wf_df6366c8): frozen configs → 349-pool transfer check
(SEGMED + M3b/X4/R4) → virgin-74 single consult → package kernel_driver11 if confirmed.

**Transfer arbitration complete (2026-07-05): SEGMED killed by 349 transfer (+0.33,
sign flip — 4th pool-luck death); RETUNE family survived 349 (R4 −0.217 P=0.004) but
failed virgin-74 ship bar (R4 −0.135, P=0.255; curse decay −0.37→−0.22→−0.13).
ALL 773 LOCAL WELLS NOW CONSUMED — public LB is the only unbiased arbiter left.
E4 virgin-74 baseline = 10.4726 (bookkeeping). R4 = ONLY candidate with 3-pool
consistent direction → shipping as pre-registered probe (kernel_driver11 packaging;
R4 = mean(P3b,P4) both under RETUNE fusion constants). Slots refreshed 10/day.

**LOCAL RESEARCH PROGRAM COMPLETE (2026-07-05).** exp40_structcat closed the last
frontier: catastrophe wells = FIELD-ANOMALY wells (structural-vs-GR error corr
0.86-0.99 there; whole neighborhood wrong same direction); tail 54-78% of SE from
~10% wells = unaddressable with dataset information. Full floor map: emission locally
optimal (3 neg), weights equal-optimal (2 neg), routing no-signal (4 neg), containment
no-safe-anchor (2 neg), parents cube exhausted + SEG transfer-killed, precision fusion
no-signal, catastrophes closed. SHIPPING: X4 (driver12, gated bitwise) + R4 (driver11,
gated bitwise) as final pre-registered probes — both 3-pool consistent. After their
scores: lock final pair (E4 7.558 + best-of-probes or diversity pick). TEAM MERGE
still pending — remind user until done.

**X4 = 7.505 NEW PUBLIC BEST + R4 = 7.521 (2026-07-05) — probes 3+4 both WIN.**
Probe methodology now 4/4 with multi-pool-consistent evidence (C2/E4/X4/R4) vs 0/4
without (v11/v12/v13/g0) — perfect separation validates the discipline. Ladder:
X4 7.505 | R4 7.521 | E4 7.558 | C2 7.609 | v10 7.689. X4 = kernel_driver12 (mean of
c2/tuned/p3b/p4 across CFG_V10+RETUNE fusion regimes). FINAL PAIR leading candidates:
X4 + E4 (sibling risk) vs X4 + v10 (diversity) — decide later w/ more public data.
Family gains shrinking (−0.05/probe); top-15 = 6.33 needs a different tier of idea.

**exp44 marriage: v10 NOT diverse (corr 0.96 = family band; p349 "win" = era
contamination). ARCHITECTURE CEILING CONFIRMED ~7.5 public; X4 sits on it.** Diversity
inside HMM+knot backbone exhausted; SEG-like decorrelated structures (0.1) too weak.
REMAINING MONTH STRATEGY: (a) X4 7.505 defended as ship anchor; final pair X4 + R4/E4
(sibling risk unavoidable — family IS the portfolio); (b) the one big bet left =
rank-2-style per-well matching ENGINE built properly (knots-leg 2.0: hi-res DP,
multi-channel GR features, explicit stretch/squeeze, dip priors from train stats) —
weeks-scale build, only route to 6.x; validation = union-773 indicative + public
probes only. TEAM MERGE STILL PENDING (user action).

**exp47 band audit: pad=120 = FREE VERIFIED CORRECTION (fold into next probe; −0.015
pooled, zero harm, parity clean). Band ≠ catastrophe mechanism (68/70 cats fail INSIDE
corridor; binding constraint = emission/prior). Flagged: knot dmax=36 binds on drift
wells (untested widening — needs its own no-harm pass). Truth drift routinely exceeds
40ft vs tvt0 but corridor follows the prior path so mostly covers. Engine (exp45/46)
running: truth function-class mining → calibrated-prior DP engine.**

**USER STRATEGY RULING (2026-07-05): do NOT rebase on public forks — "they release it
to get people stuck on it."** Evidence agrees: ~600 teams piled at 7.2-7.3 on the fork
lineage; pilkwang 42 versions → 7.501 stuck; fork PF measured unstable on our pools;
5.x teams don't use it. HYBRID re-weighted: disclosed MECHANISMS from discussions
(Mamarin datum, cdeotte anchor) = legit intel to implement in OUR stack; the fork
artifact itself = at most a late diversity submission, never the base. Fold-ins banked
for kernel_driver13: pad=120 + dmax=54. New mechanism lever: knot sigma_p cap 12ft vs
observed 68-104ft drifts (exp50 running). RANK TRUTH: X4 7.505 = 1013/4297; <7.15 =
~rank 113; <7.0 = ~64.

**exp50: CATASTROPHE ROOT CAUSE MEASURED = EMISSION ALIASING.** On 389ae58f GR cost at
TRUTH (1.23) beats zero-offset (2.64) — evidence points at truth! — but an alias at
22ft scores lower still. Penalty stack falsified (deleting MAP entirely lifts class
<1ft; sigma_p cap 12 is CORRECT for typical wells, MAD plateau 11-13ft, only tail
grows). Cause = gain/offset drift distorting emission surface (exp37 measured, exp37
post-hoc recal locks in). ATTACK: exp51 joint gain-drift×path DP (calibration as DP
state; aliases need implausible g-jumps, truth needs only measured smooth drift).
model_knots.py got sigma_p kwarg (backward compatible, parity clean). Also: dmax54's
gain is 1 well (059c8f24 −4.3); exp48 fold-ins stand. upr does NOT depend on p_kn
(exp50 correction). Fleet: engine + datum + jointdp, ALL converging on the aliasing/
drift mechanism from 3 angles.

**exp45/46 ENGINE MILESTONE (2026-07-05):** Truth function class MEASURED: piecewise-
linear to 0.27ft RMS (segment-native correct, smoothing wasteful); catastrophe tail is
STRUCTURAL (NCC same as easy wells! but 64ft excursions, 1.5x fault jumps NO current
leg can express, 14x band violations); full priors: seg 330-400ft, 2.3 breaks/kft,
dip-change Laplace(2.8), jumps 0.07/kft ±2-10ft, corridor ±120 = 100% coverage.
ENGINE BUILT: decorrelation 0.529 PASS (first strong+decorrelated leg!), median-well
5.6-6.8ft ≈X4, but MAP wrong-basin tail (7% wells walk 20-70ft on GR self-similarity
16.5ft shift-MAD) → pooled 12.4, not viable ungated. Blowups SELF-DIAGNOSABLE (nseg
20-46 vs 12, kriging residual) → exp49-gate running (MEDg/EXg; v74 EX was −0.52
ungated = best-ever candidate on hardest pool). Also dead tonight: exp49 datum
(Mamarin localization 54% not 80%, thin-zone laterals lack datum info), exp51 joint
gain-drift DP (aliases are drift-plausible; within-well GR flexibility DEEPENS
degeneracy). exp52 alias-scale veto + driver13 packaging running.

**exp55 PORTFOLIO FRAMEWORK (2026-07-05):** Final pair = X4 + MEDg if MEDg public
≤7.66, else X4+R4. Anchor rotates ONLY on ≥0.35 public delta (2σ of sibling-delta sd
0.175) or full probe evidence; public-chasing measured −0.05 E[private]. sd(50-well
public)=1.66; P(sign agree X4-vs-R4)=0.46. MEDg-the-ensemble corr 0.99 w/ X4 (gate+
median crush ENG votes; value = locally strongest 7.786, not diversity). Freeze pair
Aug 3. ⚠️ TEAM MERGE by JUL 29 HARD — and Kaggle BLOCKS merges when combined subs >
days×5; both accounts at 5/day → MERGE THIS WEEK. exp55_portfolio.py = rerunnable
update rule. MEDg public score pending (engine debut).

**MEDg = 7.516 public (2026-07-05): engine debut = statistical tie w/ X4 7.505 (+0.011,
inside ±0.06 rerun noise). FINAL PAIR CONFIRMED per exp55 rule: X4 + MEDg.** WRITEUP-
FLUSH INTEL (huge): (1) ORACLE FRAME: heel-pinned LINE oracle=7.59 (X4 IS there),
QUADRATIC oracle=4.28, leaders 5.26-5.6 in between → FRONTIER = PER-WELL CURVATURE
(exp56 running). (2) Hidden test wells CLUSTER, median ~1kft from nearest train
(offset-probe measured) — spatial legs underweighted. (3) odyssey midpoint hedge −0.16
public verified: gentle α=0.2 pull to midpoint of 2 close minima, LOOSE trigger
(exp57 running; different from our failed exp49 full shifts). (4) Wireline→LWD operator:
gain 0.79, off 17.8 GAPI; toe alias band INSTRUMENTALLY ERASED (coherence 0.03-0.11) —
physics confirms our aliasing dead-ends. (5) Public deltas <±0.06 = rerun noise.
Ladder: X4/x4b 7.505 | MEDg 7.516 | R4 7.521 | E4 7.558.

**exp56 CURVATURE MAP (landmark):** Line oracle 8.505 / quad oracle 6.075 on our 773;
X4 7.851 already sub-line (matches public frame). 71.3% of X4 SE = smooth MIS-LOCATED
shape in 177 wells (77% SE); shape/total=1.01 (not noise); decoders do NOT under-curve
(b_p/b_t=0.90) — mis-LOCATION not stiffness. Projection fix already baked in (0/4).
(a_t,b_t) quadratic labels CACHED all 773 wells. exp57 hedge DEAD (our minima pair
13-16ft off datum; odyssey's gain = their better misfit object). exp59 RUNNING:
per-well (a,b) coefficient regressor from long-wavelength/spatial covariates (kriged
neighbor coefficients = theory favorite) — 2-number decision space, 1.78ft gap, even
20% capture = -0.35 = biggest number left. Writeup drafted+corrected at
notes/rogii_working_note.md (posting = user decision, deadline tonight).

**exp59/exp61: the −3.15 curvature-regressor claim = LEAKAGE (tail_bias, truth-measured,
carried >100% of it; caught in one adversarial pass — R2≈0-but-deploy-huge was the tell).
Honest residue: SHRINK_K coefficient shrink k_a≈0.97/k_b≈0.93, −0.06 union, 3/4 pools,
free to stack. Curvature frontier honest frame: 1.78ft gap real but NO legal covariate
signal; spatial curvature smoothness REJECTED (IDW R2<0 @478ft). Path to it = matching
quality (Tucker's engine), not coefficient regression. RULE ADDED: any exp56-derived
feature must strip everything computed from ctx['truth'].**

**Jul-6 flush resweep:** WRITEUP NOT POSTED — ineligible (medal zone ~r430@7.216, we're
r1021) + would give away sole non-commoditized assets (exp56 mislocation numbers, (a,b)
labels, self-diag gate, operator figures; audit: everything else already public).
URIBE writeup = the blueprint: GR-cost minima contain truth ±5ft on 97% wells (~14
cands), selection oracle 6.42, ONLY banked gain = LISTWISE learned mode-scorer w/
POSTERIOR-MEAN output (−0.215 CV); his deploy trap = quadrature (+0.80 moving 3.5ft on
fine baseline) → gate correctors by move-vs-residual. Tucker re-confirms per-well-only
~5ft. Villa: engines correlate ρ0.89-0.96, re-blending flat. k256: two-group CV rule
(LB tracks only one group) — worth replicating. exp62 operator: physics real (loc
54→65%, kernel~2ft) but NO harvest (decoder needs sharp typewell; 12ft datum MAE
unactionable). exp64 mode-scorer program RUNNING (candidates→listwise scorer). driver15
submitted (54359830, pending). Medal cutoff r430@7.216 = the realistic 30-day target.

**driver15 = 7.540 public (54359830): noise-band w/ MEDg/X4, qualifies per exp55 rule.
FINAL PAIR UPDATED: X4 (7.505 anchor) + driver15 (locally strongest, union 7.73).**
Ladder: X4/x4b 7.505 | MEDg 7.516 | R4 7.521 | d15 7.540 | E4 7.558. Six family configs
within 0.053 public = textbook sibling noise-band, exactly as exp55 modeled. All grind
now on exp64 mode-scorer (the −0.215-class program). Medal target r430@7.216.

**FINAL PAIR LOCKED (exp68 refresh): X4 + d15.** d15 = best full-773 (7.713; shrink
−0.073 on 4/4 pools); pair P(contains private best)=0.815, E[regret] 5x lower than
X4+MEDg. Anchor bar empirically re-derived from 6 publics = 0.343 (≡ old 0.35).
exp66: NO two-group rule (58 rules vs null; k256's rule undetectable from our ledger).
exp67: DENSE OBJECT = containment@5 95.7%, ceiling 5.709 (re-decode dead: GR can't
self-arbitrate; seg-local −1.98 banked for 2-D selector). LIVE 5.x BETS: exp68-scorer
on dense object (realize the 5.71 ceiling via orthogonal features) + exp65 FastSLAM
(self-consistency information source). d15 caches carry full moment terms (any future
K refreeze scores closed-form).

**FASTSLAM BREAKTHROUGH (exp65, 2026-07-05):** Mechanism REAL + instrumented: wrong-basin
particles lose weight AT re-crossings (6/6 wells, I=−0.039 ll/pt); exp50's alias trap
visible AND neutralized (7e721392: alias fits fixed map better fresh, dies at re-cross →
SLAM 0.91 vs X4 27.3!). Catastrophe class CRACKED (389ae58f 56→25; standalone cat −1.5,
EX-on-cats −3.65 P=0.001) using per-well data only — the 'unaddressable' verdict was a
fixed-map artifact. VIABLE: MED(E4,R4,SLAM)=7.749 union, −0.101 vs X4, 3/4 pools P<0.2
(two at P=0.001) — first new-mechanism viable since the family plateau. Standalone noisy
on easy wells → fusion is the form. driver16 packaging (3-seed MED + shrink) + exp69
SLAM-v2 (seed depth, cat-targeted routing via validated detectors, easy-well ESS gating)
BOTH RUNNING. Scorer-on-dense (exp68) also live. 5.x path: SLAM upgrades + dense-object
scorer are now complementary attacks.

**USER DIRECTIVE: target 5.2, not 7.2. THE 5.2 PROGRAM = SLAM AS PRIMARY DECODER
(exp70/v3, running):** v1 is a first-draft PF (genealogy smoothing collapses at heel =
weakest tail posterior; blind proposals; hand-set params; 1s/well vs 30x kernel budget).
Upgrades: FFBSi backward smoothing, look-ahead APF proposals, PREFIX-REENTRY DATUM
ANCHORING (prefix map bins are truth-anchored → absolute datum info wherever tail
revisits prefix TVT — the 5.36 debias-floor attack; answers v1's gauge-blindness),
per-well marginal-likelihood param adaptation. Milestone: standalone beats C2 8.27 →
primary-decoder trajectory toward 5.x. exp68 scorer program CLOSED at measured floor
(3 approaches → same precision/recall wall; 30% top-1 on hard class < keep-arm bar).
Also running: driver16 packaging, SLAM v2 (cat routing), SLAM portfolio.

**SLAM V2 = BIGGEST LOCAL WIN (exp69): COMB2 union 7.368 vs X4 7.851 (−0.48, 5x v1).**
Form: MED7(E4,R4,SLAM-7seed) + detector routing (sd_c2>1.927 OR div_rms>11.656 →
0.75*MED7+0.25*SLAM7). Cat-class −1.84 w/ REST-PARITY HOLDING; viable 3/4 vs v1.
Offset-line projection: public ~7.05 + curse ~ 7.15-7.35 → MEDAL LINE IN WINDOW.
Findings: seed-sd falls 1/sqrt(k), flattens k=5; easy-well containment DEAD (median IS
the containment); routing is load-bearing (MED7 alone 2/4). sd_c2-only variant admitted
(fit 6.959, no ENG dep) as kernel fallback. Honesty: dual admission criterion adopted
after 1st pass failed (both recorded); f150 +0.16 (SLAM's weak pool). driver17
packaging RUNNING; driver16 ship chain RUNNING; SLAM v3 (5.2 program) RUNNING.

**NEAR-MISS (2026-07-05): driver16 kernel ran WRONG MODEL on Kaggle — KeyError 'e4'
(local driver14 was edited to expose e4, dataset copy stale) → per-well fallback,
caught by log inspection BEFORE submitting. RULE ADDED TO SHIP CHECKLIST: before every
kernel push, cmp -s ALL staged dataset modules vs repo (the drift check now in the
chain); after dataset version, download-verify the changed file. Full module sync
pushed. Also: 'LOG_DIRTY' grep saved the slot — never submit on rows-count alone.**

**D17 = 7.132 PUBLIC (2026-07-05) — MEDAL ZONE BREACHED. d16 = 7.442.** From rank 1021
to ~top-120 in one ship. COMB2 (MED7+detector-routed SLAM+shrink) delivered exactly the
offset-line projection (7.1-7.35). Probe methodology now 6/6. Ladder: d17 7.132 | d16
7.442 | X4 7.505. Final pair: d17 anchors; rerun portfolio w/ d17+d16 caches
(kd17_comb2_ref.json has per-well+moments). SLAM v3 union (driver18, toward 6.x) still
computing. Submitted from MAIN account (team hit 5/day cap) — MERGE NOW CRITICAL:
best submission lives on bharatmohan, rest of ledger on nencybansal!

**Jul-6 state:** FINAL PAIR = d17 + d15 (exp72: E_regret8 0.008, P(contains best)=0.86;
d17 partially DECORRELATED from family 0.86-0.93 — routing did structural work; all
d17-pairs ~0.4 E[min] better than pre-d17 pairs). d17 full-773 = 7.335, public 7.132
(~rank 107). SLAM v3 union: NOT viable raw (easy-well noise 10.97 vs 5.10) but owns
cats (−2.80 vs X4, EXv3-cat 15.45 best ever); fixes running: exp71 postpass (mean-map
deterministic redecode = the 6.x path) + exp76 COMB2v3 swap. Intel flush running.
Slots fresh 5+5. MERGE STILL PENDING — d17 on main, ledger on team acct!

**exp73 D17 ANATOMY = STRATEGIC REDIRECT (2026-07-06): DATUM, not curvature, is our
frontier.** d17 error decomp: DC 55% / SLOPE 23% / CURV 8% / WIGGLE 14%. Oracle ladder
(union): d17 7.335 → debias(DC removed) 4.907 → +slope 3.405 → quadfix 2.985. Remove-one:
DC→4.907, SLOPE→6.43, CURV→7.04 (curvature worthless!), WIG→6.82. Cat class 66% DC,
debias 9.92 from 17.02. Composite floors: allDC_100=4.907, catDC_100=5.80, SO2_curv_100
=7.04 (SO2 is MINOR). K-refit gives nothing (7.336). **THE PLAY: datum (DC) correction.
Levers: (a) heel-fit affine calibration (Mamarin 54%→80% localization — we're at 54%),
(b) v3/redecode self-consistency (cat DC, working: 28→15). exp71 postpass DEV WORKS:
rd_slam redecode cat 15.2 vs C2 28.2, easy 7.6 (fixes v3 noise), det.; union shards
running local → driver18 via routing.** SO2 reframed = gate-signal + minor curv. Uribe
gate table cached (per-class residual scales). Fleet redirect: heel-fit datum program
(flagship) + postpass-arb/driver18 + SO2-gate.

**exp75 postpass arb: deterministic redecode = SUB-THRESHOLD micro-gain, no-ship.**
COMB2rd (redecode routed, shrunk) ~7.28 union (−0.06 vs d17), all pools mildly negative
but P(worse) 0.22-0.46 → 0-1/4 clears bar. Redecode DID work (killed v3 variance: v74
+0.017→−0.018; kept cat wins 16.9<20.5) but rest-class 10.09 vs median 5.06 = too weak
on easy wells. CONFIRMS ANATOMY: redecode moves cat-DC only; bulk 55% DC needs the DATUM
lever. d17 stands. ENDGAME NOW = datum (DC) mechanisms: heel-fit calibration (exp74
flagship, running) + prefix-cut datum CV (exp76, launching). SO2 minor. Postpass micro-
gain banked for later composition. On Opus max-effort now.

**THEORY OF VICTORY (2026-07-06, Opus max-effort): THE ENTIRE GAP TO #1 IS THE DATUM.**
debias oracle 4.907 < Tucker #2 (5.44); quadfix 2.985 < Rishikesh #1 (5.26). We do NOT
need better dip/curvature — just c_well (per-well scalar, std 0.0064, 55% of d17 error).
TWO INDEPENDENT DATUM CHANNELS (nobody public fuses both): (1) GEOMETRIC — TVT=ANCC-Z+
c_well identity, c_well pinned from prefix, tail error = kriging ANCC extrapolation
(exp77, LRO-honest); (2) GR-MATCHING — heel-fit calibration datum scan, per-well only,
Tucker's channel (exp74, 54%→80% target). Two independent unbiased estimates of ONE
scalar → fusion beats either (exp78). Also: prefix-cut CV (exp76) tests prefix→tail DC
persistence. FLEET=4 datum assault + SO2. If ANY channel/fusion cracks datum → 4.9-6.5
→ top. This is the whole game now. odyssey James-Stein=0 / Uribe neg-OOF-R2 were the
GR-only channel on THEIR reference; geometric channel + fusion is untested by anyone.

**exp74 heel-fit datum: NON-VIABLE, cap = 65% localization (not 80%).** KEY MECHANISM:
residual DC after datum-scan correction = the scan's OWN localization error; localization
is a property of the MATCHER not the calibrator. Operator affine part PROVABLY absorbed
by heel refit (zero contribution); only spectral sigma2 smoothing helps (+9-12pts,
already in match_s2). Tabish denoise HURTS on our object (-4.4). Mixed>pure ref (+3.7).
So GR-scan/calibration CANNOT crack datum — capped by matcher. SURVIVING datum routes
(both bypass the cap): (1) GEOMETRIC channel exp77 (non-matching, spatial c_well), (2)
GEOMETRIC-ARBITRATED datum scan exp79 (geometric c_well breaks the GR-scan's 35%
wrong-candidate ties; extends exp52's 90% coarse arbitration w/ prefix-pinned c_well).
Fleet: geometric+fusion, prefix-cut, SO2, geom-arbitration. Datum = whole game.

**★ DECISIVE FINDING (exp77, 2026-07-06): THE DATUM IS MATCHER-LIMITED, NOT
TRICK-CRACKABLE.** d17 own datum RMS = 5.45ft (near-optimal for our matcher). Geometric
channel datum err = 24.98ft RMS (kriged ANCC 13-16ft error even at prefix; c_well IS
constant std 0.0065 but ANCC too noisy to pin it). GR-scan ~12-16ft. ALL channels WORSE
than d17's own datum → re-centering catastrophic (25.45). exp79 geom-arbitration 83%
(real signal) but candidates displaced 12ft → only 13% oracle, no-ship. CONFIRMS
odyssey(JS=0)/Uribe(neg OOF): datum unobservable BEYOND matcher extraction. debias oracle
4.907 = truth-based, NOT realizable. **KEY ARITHMETIC: our non-DC floor 4.91; Tucker #2
(5.44) ⇒ his datum RMS ~2.35ft = 2.3x sharper = a BETTER MATCHING ENGINE. THE WHOLE GAME
= matcher quality.** Datum-trick era CLOSED (heel-fit, geometric, arbitration all
matcher-limited). PIVOT: matching engine. Vehicle = SO2 (amerhu 2nd-order smoother,
HMM 5.18/blend 4.57 on his samples) + SLAM. Target: standalone matcher datum RMS < 5.45.

**exp76 prefix-cut: corr(prefix_DC,tail_DC)≈0.0-0.06 — DATUM NOT PREFIX-RECOVERABLE
(3rd datum confirmation, = odyssey JS=0).** exp77b SO2 (amerhu 2nd-order smoother):
standalone ~21 union (cat-brilliant 56→5.7/27→1.5 but bulk-disastrous, WORSE than v3);
config step=1.0 vs amerhu's 0.35 → likely hit his documented GRID-FREEZE bug (kernel
freezes rate when step>per-step-noise). SO2 decorrelates (corr 0.38). PIVOT REFINED:
we have MULTIPLE independent cat-matchers (v3 stochastic, SO2 det); their AGREEMENT =
better router than sd_c2/div_rms. Play = consensus routing (route to matcher-mean only
where v3+SO2 AGREE + cat-detector) → converts 57%-SSE cat tail WITHOUT easy-well tax.
Plus fix-SO2 (step 0.35 diffusion-matched = amerhu's uniform 5.18 matcher shot). Datum
era fully closed; matcher-consensus era.

**exp76 prefix-cut CLOSED (definitive): prefix carries ZERO tail-datum info (OOF R²
−0.02; 3 estimators E1/E3/heel-fit + v3 all agree; every prefix-correction HURTS).
Heel/prefix datum line fully dead.** BUT KEY POSITIVE: datum is a REAL shared property
(corr d17↔C2=0.913, d17↔v3=0.439 — v3 has 0.56 INDEPENDENT datum info → consensus
averaging can reduce it). Recommendation: attack datum from TAIL-SIDE spatial (krige
lateral ANCC from NEIGHBORING wells' laterals, not the heel) — Yanza: test wells ~1kft
from nearest train, closer than assumed. LAST untested datum channel = exp81 tail-lateral
kriging. Fleet: fix-SO2 (uniform matcher), consensus-routing (cat capture via v3/SO2
agreement), tail-lateral-kriging (last datum shot).

**exp77 SO2 CLOSED: amerhu 2nd-order smoother = cat-only, not viable (standalone 21;
MED +0.46; 0/4).** ROOT CAUSE = FIXED typewell map (not config — diffusion-matched
kernel validated 2e-4 vs amerhu; oracle-plumb 0.17ft). d17 ALREADY routes SLAM7 (smoother
same object) → SO2 adds nothing, swapping HURTS. SO2-SLAM7 corr 0.228. Posterior-std gate
WEAK on our data (corr 0.13 not amerhu's 0.66-0.92; his samples = GR-rich easy wells).
STRATEGIC: all our better-matchers (v3/SLAM/SO2/redecode) are SLAM-family, correlated,
cat-only; d17 near-optimal on easy wells (uses kriging+knots+heel = more info than any
single matcher). Gains ONLY possible on cat tail (57% SSE). CLOSED fix-SO2 (redundant).
Live: consensus-routing (cat capture via agreement), tail-lateral-kriging (last datum),
LEARNED-EMISSION matcher exp82 (the matcher-quality axis = plausibly Tucker's edge; if
it fails, d17 7.13 = confirmed machinery ceiling, medal zone, consolidate).

**exp77/78 DATUM FULLY EXHAUSTED (5 channels): geom −746%, geom+heel fuse 13.26ft
(theory CONFIRMED: 26% better than either, indep corr 0.018) but all >> d17 own 5.45ft;
BLUE weight d17=0.87-1.0, best realizable fuse 7.322 (−0.013, 1/4 pools). Only v3 has
signal (0.44 corr, +3-6% ceiling, fails bar). DATUM LEG CLOSED — d17 near-optimal.**
Fusion recommends: attack SLOPE (23% of error, oracle 7.335→6.428, worth 0.91). d17's
slope realizability NEVER tested (unlike datum). exp83 slope-realizability launching.
Live fleet: consensus-routing (cat), tail-lateral-kriging (last datum, low odds),
learned-emission (matcher quality = 5.x shot), slope-realizability. Datum era done.

**exp81 tail-lateral kriging = 4th & FINAL datum confirmation: 16.08ft (3x worse than
d17 5.45). Oracle-c_well 16.28 ≈ pinned 16.08 → bottleneck is ANCC SURFACE kriging bias
(~12-16ft), not c_well. NO spatial channel beats d17's matcher-derived datum. DATUM
PROGRAM 100% CLOSED (heel/prefix/geom/tail-lateral all confirm 5.45 irreducible).** Note:
lateral-focus (16.08) beats production-bank geom (24.98) → production kriging under-
weights tail/lateral (banked, but doesn't help datum). THE FRONTIER IS UNAMBIGUOUSLY
MATCHER QUALITY. 3 live shots: slope-realizability (exp83, physics-favored crux: does
dip persist prefix→tail unlike datum?), learned-emission (exp82, the 5.x matcher shot),
consensus-routing (exp85, cat capture). If all fail → d17 7.132 = confirmed ceiling,
rank ~107 medal zone, consolidate. Weeks to Aug 5 deadline. MERGE STILL PENDING.

**exp83 SLOPE = MATCHER-LIMITED (like datum). corr(prefix_dip,tail_dip)=0.05 (physics-
favored persistence FALSIFIED, same as datum ~0.05); krig dip corr −0.017 (zero signal);
d17 slope corr_true 0.824 = best channel; swapping any in HURTS (+1.46). ★ COMPLETE PROOF:
78% of d17 error (datum 55% + slope 23%) MATCHER-LIMITED, d17 near-optimal. Datum AND
slope both unrecoverable from prefix/spatial. THE ONLY LEVER = matcher quality (Tucker
2.35ft datum = sharper GR matcher).** Tried matchers: HMM/knots/DTW/SLAM/SO2/PF/
list-Viterbi = all cat-only. LAST SHOTS: learned-emission (exp82, matcher-quality, 5.x),
consensus-routing (exp85, cat capture, top-40). If both fail → d17 7.132 rank~107 =
proven ceiling of GR-matcher+geometric-fusion; consolidate + defend. Endgame consol
(exp84) verifying curv/wiggle also matcher-limited + final-pair private insurance.

**★ exp80 CONSENSUS ROUTING = FIRST HELD-CONFIRMED BEAT OF d17! union 7.335→7.265
(−0.070, P=0.013), 3/4 pools P<0.2, BOTH frozen-holdout (f150 −0.044 P0.11, v74 −0.011
P0.0) PASS.** Mechanism: route to matcher-mean only where ALL 3 (v3,SO2,redecode) AGREE
tight (tau3.5) + cat-detector + shrink. Routed cats 2/2 won, easy dmg −0.09. Matcher
agreement IS the router that doesn't mis-fire. BIG PRIZE: ABSTAINED on 59 cat wells it
would WIN (vs 16 lose) — too conservative (all3+require_so2). RELAX to 2-of-3 → capture
those → driver19 toward −0.3-0.5. Frozen cfg tau3.5/dis7/all3/shrink = driver18 (ships).
The cat-capture-without-easy-tax that eluded 40 rounds finally landed via AGREEMENT gate.

**★ exp84 CEILING PROOF 100% COMPLETE: DC55+SLOPE23+CURV8+WIGGLE14 = ALL matcher/noise-
limited.** corr(prefix,tail)≈0.05 for datum/slope/curv (prefix carries zero tail signal);
d17 corr_true highest on all 4 (0.86/0.82/0.86/0.86); wiggle 86% captured, rest below
emission resolution. 7.335 = RIGOROUS machinery ceiling. ONLY lever past it = sharper
EMISSION (exp82 learned-emission; early signal datum MAE −2.245ft, decode pending).
**FINAL-PAIR PIVOT: can't per-well-mix 2 submissions (each = whole pooled RMSE) → slot-2
must be COMPLETE+pooled-competitive+cat-hedging. Raw v3 USELESS (P(v3<d17)=0.000, easy-well
disaster). WINNER slot-2 = CONSENSUS ROUTER (driver18): pooled-competitive −0.07 AND
cat-hedge AND decorrelated → supersedes d15/m25v3. FINAL PAIR = d17 + driver18.** m25v3
= fallback if consensus doesn't package. Ship driver18/19 → new champion AND slot-2.

**exp82 learned-emission: TWO-PART. (1) Emission IS sharpenable — localization 0.613→0.698,
MAE 10.20→7.95ft, P=1.00 (breaks the exp74 CALIBRATOR ceiling; emission FUNCTION not
capped). (2) BUT insufficient: sharper single-emission datum 7.95 STILL > d17 fused 5.45;
hard-decode AMPLIFIES alias lock-on (confident-wrong on 30% bimodal → +9-11ft blowups →
standalone 13.35, MED 7.44, 0/4). CEILING HOLDS: d17 fusion beats any single sharpened
matcher.** Tucker's 5.44 needs sharper-emission + alias-robust SOFT decode we haven't
built. REALISTIC ENDGAME: d17 7.13 (rank~107) = machinery ceiling; CONSENSUS routing
(driver18 −0.07 held-confirmed, driver19 relaxed pending) = the ONE working lever →
~6.9-7.0 (deep medal/top-60). FINAL PAIR = d17 + driver18/19 (pooled-competitive AND
cat-hedge). Top-5 (5.x) beyond reach of tried machinery. Ship consensus, defend medal.

**★ CRITICAL (exp-driver18 packaging, caught pre-submit): consensus matcher legs use
get_ctx-BY-ID (needs train-pool membership + TRUTH) → on HIDDEN wells they RAISE →
router abstains → ships plain d17. kd18 as-packaged = NO-OP on hidden set (would score
flat 7.132, waste slot). DO NOT SUBMIT kd18. FIX (tractable, pieces exist): v3 IS
self-containable (kd16 ran it on hidden, d16=7.442); kd17 builds full C2 ctx on hidden
(d17=7.132) → redecode reuses that internal ctx not get_ctx-by-ID. Need: self-contained
matcher-leg infra (v3 via kd16 port + redecode-ctx07 adapter over predict_well17) so the
consensus router FIRES on hidden wells. Until then NO consensus gain materializes public.
Building infra now (exp-selfleg). Then package winning config (relaxed/independent
consensus) with self-contained legs → ship → gain delivers. This is THE critical path.

**★★ TWO HELD-CONFIRMED BEATS of d17 (2026-07-06):**
1. **e80_vrsv = SHIP WINNER (driver19)**: exp80 all-3 base + v3/redecode gated by SLAM
   SEED-VARIANCE (seedvar_thr 2.0, tau 6.5, target vrs7). union 7.335→7.247 (−0.088,
   P=0.002), 3/4 pools, BOTH held pass (f150 −0.040 P0.195, v74 −0.100 P0.087), easy_dmg
   −0.066 (IMPROVES easy!). ship=True. Legs: v3+redecode+SO2 (NO C2_LE). Projects public
   ~7.04. **Relaxed pure-2of3 FAILED (co-drift +0.33) — seed-variance gate is the fix.**
2. exp85b C2_LE independent: union 7.235 (−0.100 vs d17, 3/4+both held) — LOWER but e24
   regression (+0.054) + easy_dmg +0.208; beats exp80 only 2/4 (strict bar). Captures 9
   cat-wins vs exp80's 2 (+7 from non-SLAM vote, 0 loss). Riskier; needs C2_LE leg
   self-contained. DEFER as v2. Combined exp80+vrsv+C2_LE = ultimate (tuning).
CRITICAL: BOTH need self-contained legs (selfleg round92) to fire on HIDDEN wells (else
no-op=d17). Package driver19=e80_vrsv on selfleg legs → validate hidden-fires → ship ~7.04.

**★ kernel_driver19.py BUILT + 4 GATES PASS (bitwise reproduces e80_vrsv 7.2470, −0.088
vs d17; smoke clean; runtime 34-47min/200; parity). LAYER-1 exp80 all-3 (v3+SO2+rd, 17
wells) + LAYER-2 vrs7 (v3+rd, 21 wells). BUT same NO-OP-ON-HIDDEN as kd18 (vendors
get_ctx-by-ID legs, docs lines 61-69; raises on hidden → abstain → flat 7.132). DO NOT
SHIP until selfleg wires legs to kernel-own ctx. All 3 legs needed (dropping SO2 →
non-ship 7.3211). selfleg (round92) building v3(kd16 pattern)+SO2(emission rebuild)+
redecode(ctx07 via predict_well17). = THE ship blocker.** Live: selfleg, combined
(round93, ultimate cfg), shipprep (round94, final pair d17+driver19). On selfleg done →
wire kd19 → validate hidden-fires → ship → public ~7.04 (~rank 90). Merge still pending.

**★ SHIP PLAN (exp86 shipprep, authoritative): FINAL PAIR = d17 + driver19 (E_min
7.231/9.641/11.303 = LOWEST every regime, dominates d15/m25v3/c2v3; improves floor AND
hedges cats, corr 0.937 on cat class). exp85b C2_LE = OVERFIT MIRAGE (fails ship bar,
defer v2). Main acct 0/5 today, all slots open, d17 live as slot-1 floor.** SHIP SEQUENCE
(do NOT submit till step 4): 1) selfleg.py (v3 kd16-pattern + SO2 emission-rebuild +
redecode ctx07-via-predict_well17); 2) legs reproduce exp80 scalars 1e-4 + run on HIDDEN
w/o get_ctx; 3) wire kd19.matcher_record→selfleg, re-run 4 gates; 4) GO/NO-GO hidden-fires:
kd19 on hidden-shaped wells → routed N>0 in log (N=0=flat 7.132=STOP); 5) stage dataset
+cmp-drift+download-verify; 6) submit kernel rogii-driver19-m (CPU/no-net/no-GPU); 7) log
check routed>0/14151rows/0fallback; 8) submit → pair {d17, driver19} public ~7.04. Post:
if lands 7.132→selfleg silently no-fired→revert d17+m25v3. MERGE d17(main)+ledger(team)!

**★★★ exp86 COMBINED ROUTER = BIG WIN (driver20 target): union 7.1074 (−0.2279 vs d17
P=0.000; −0.140 vs driver19, transfers OOS both held f150 −0.111/v74 −0.141). 3/4+both
held pass. catThr 13 routed 13W/0L (17.02→16.16), easy_dmg −0.066 (IMPROVES).** Synthesis:
exp80 all-3 base + VRSV leg (seed-var vr, 21 routes) + C2_LE leg (independent anchor,
catThr-restricted tau5.5, 10 co-drift-rejected cat wins). PROJECTS PUBLIC ~6.90 (~rank
40-50!). Needs 5 LEGS: v3+rd+so2+slam7 (driver19/selfleg) + **C2_LE** (NEW load-bearing:
exp82 GBM emission + exp84 C2-FB decode; without it = just driver19 7.247). DEPLOY: fit
GBM on all 773, ship model, decode hidden (honest, hidden≠train). SHIP SEQUENCE: (1)
driver19 7.247→~7.04 FIRST (proves legs fire on hidden + calibrates offset); (2) driver20
7.107→~6.90 (add C2_LE leg). exp85b's 7.235 was HANDICAPPED (dropped exp80 base on 188
wells); fair = 7.178, combined beats by −0.071. Live: selfleg, validator, C2_LE-leg build.

**★★ selfleg.py DONE — SHIP BLOCKER SOLVED. All 3 matcher legs (v3/SO2/rd) reproduce
exp80 within 1.6e-05 from SELF-CONTAINED ctx (no get_ctx, no truth); HIDDEN PROOF passes
(self_idx=None path, 3/3 finite). build_ctx07 bitwise=build_ctx on train, kd13 self_idx
=None convention on hidden. Feeds EXACT exp70/71/77/80 cores.** APIs: setup_state,
hidden_views, matcher_legs(rows,tw,well,Sv,fields,iso,P)→{v3,so2,rd,mapc,ctx07,d},
build_slam_d, build_ctx07, leg_v3_rd, leg_so2. Runtime ~5-7s/well/3legs, 200-well ~20min.
NOW: wire kd19.matcher_record→selfleg.matcher_legs (driver19 fires hidden, ships ~7.04);
round96 builds driver20 (+C2_LE leg via same ctx07 → ~6.90). round95 validator = GO/NO-GO
(routed N>0 hidden). SHIP driver19 first (calibrate offset) then driver20.

**SHIP STAGING MANIFEST (transitive closure, 2026-07-06): kaggle_ds_scripts_m needs +15
modules: exp25_confirm exp25_guard exp28_ensemble exp42_virgin74 exp44_marriage
exp69_slamv2 exp70_slamv3 exp70_union exp71_postpass exp77_so2 exp80_consensus
exp85_relaxcons kd17_shrink_check kernel_driver19 selfleg.** (38 of 53 deps already
present.) driver20 adds C2_LE → also needs exp82_learnemis + exp84 + the GBM artifact +
sklearn (verify Kaggle has it). Ship: add 15 → cmp-drift all vs repo → kaggle datasets
version → download-verify the 15 → submit kernel rogii-driver19-m (CPU/no-net/no-GPU)
imports kernel_driver19 → LOG check routed>0/14151rows/0fallback → submit.

**★★★ 5.x PIVOT (2026-07-06, user "target 5 not 7"): the ceiling proof is for HAND-BUILT
matchers ONLY. Leaders (5.26-5.44) ≈ debias oracle 4.907 → they've near-SOLVED the datum.
KEY PHYSICS: datum is NOT an independent scalar — it's ACCUMULATED DIP error integrated
heel→toe (0.001 ft/ft × 3124ft = 3ft). Yanza: 5.35 ≈ dip accuracy 1 ft/kft. quad oracle
(heel-pinned) 4.28 = per-well curvature IS the learnable frontier. Leaders use LEARNED
models (Tucker single model per-well-only; k256 tabular "predict aux future target feed
back") — the ONE axis we avoided (50+ rounds all physics). exp36(tabular residual)/exp82
(emission-in-decoder) were NOT end-to-end. BREAKTHROUGH BET = LEARNED HEEL-ANCHORED DIP
INTEGRATOR: biGRU/1D-CNN over GR+typewell-align feats → per-point dip-rate → integrate
from known heel → TVT; loss on integrated TVT; 773 wells GroupKFold; decouples
unpredictable datum from learnable shape, captures curvature, whole-lateral self-consist.
Deploy: train all-773 (GPU/local, small model), ship weights, CPU inference on hidden.
driver19/20 (7.04/6.90) = SAFE BANK; learned model = 5.x SWING.

**exp90 5.x realizability = CLEAN NULL for post-correction (5th confirm) BUT precise
redirect. dip-rate RESIDUAL OOF R2 0.083 (<0.1); ABLATION kill-shot: geology channel
(GR+emission, no d17) R2=0.0013 ≈ ZERO independent dip signal — laterals bedding-parallel
(TVT 15-50ft over 3-6kft MD), GR barely moves → pointwise matching = noise on easy 90%.
Learned corrector on d17 WORSENS (7.67). BUT: heel-anchored frame EXACT (true-resid oracle
0.130); 100% of 7.335→4.9 gap IS dip. WHERE 5.x LIVES = CATASTROPHE CLASS (77 wells, 57%
SSE, d17 17.02): nail to cat-quad-oracle 7.29 → pooled 5.34 (easy untouched)! Leaders nail
CATS (where GR moves), not easy. ONE UNTRIED LEVER (analysis-named): sharper alias-robust
MATCHER = exp82 learned emission (sharpenable 10.2→7.95, but hard-decode aliases) INSIDE
SLAM soft multi-hypothesis decode (alias-robust), run ONLY on cats. exp95 launching.
driver19/20 ship as bank. Post-correction era CLOSED; matcher-engine-on-cats is the shot.

**★★★ driver20 SHIP-READY: kernel_driver20.py ALL 5 GATES PASS incl HIDDEN-FIRES
--expect fire N>0 (routes 2.8-9.0ft off d17 on hidden). Reproduces exp86 union 7.1074
BITWISE (self-contained legs 2.04e-5). C2_LE leg = selfleg_c2le.py + deployment GBM
(fit all-773, 1080KB pickle .exp_cache/selfleg_c2le_gbm.pkl + kaggle_ds_scripts_main/).
Runtime 49min/200. driver20 ⊇ driver19 (additive) → SHIP driver20 as CHAMPION (~6.90),
driver19 = no-pickle fallback. NEW RISK: C2_LE pickle+sklearn on Kaggle kernel (driver19
has none) → kernel smoke must confirm pickle loads + c2 routes fire (log c2-routed count).
Ship: driver20 submit kernel + stage 15 mods + selfleg_c2le.py + GBM pkl → push → LOG
check (pickle ok? routed>0? c2 fires?) → submit. FINAL PAIR = d17 (live floor) + driver20.

**★★ CRITICAL (2026-07-06): driver19 hidden-fires = NO-GO. Reveals CONTEXT-TRANSFER RISK
in the whole consensus approach. Gain (7.247/7.107) measured on LRO-500 (harsh exclusion);
on Kaggle-faithful (twin-excluded, full 773 bank) the SLAM seedvar + leg values SHIFT →
routing decisions FLIP (ee288b routed in dev, abstains seedvar_hi on hidden; routed well
0.37ft off e80_vrsv cache, not <0.01). Routing IS context-sensitive → +0.088 gain may NOT
transfer to Kaggle. HOLDING the driver20 push. DECISIVE GATE = re-measure router gain under
Kaggle-faithful context WITH truth on held-out train wells (LOO/twin-excl, not LRO-500). If
beats d17 there → ship; if not → LRO-500 artifact. d17 (7.132) transfers (validated). The
hidden-fires validator CAUGHT this before a wasted slot = the gate working as designed.

**★★ CRITICAL LEAKAGE (2026-07-06, caught by round101 deploy analysis): driver20's C2_LE
leg (LEG C, the 7.247→7.107 gain) is gated on catThr = TRUTH-DERIVED (c2-rmse-vs-truth
>12.086). On hidden = no truth → catThr absent → LEG C INERT → driver20 DEGRADES to
driver19 (LEG A/B) on Kaggle. 7.107 was a TRUTH-LEAKED number. RETRACT "driver20=6.90".
driver20 safe (graceful degrade) but NOT a real gain over driver19. REAL ship = driver19
(7.247, LEG A/B truth-FREE detectors sd_c2/div_rms/seedvar), pending round103 transfer.
The C2_LE mechanism (+7 clean cat-wins, co-drift rejection) is REAL but needs a TRUTH-FREE
cat detector to replace catThr (exp98). Same detector needed by 5.x leslam (round100) to
route on hidden. driver20 pickle loads clean (sklearn 1.6.1; Kaggle version = residual
risk but moot since LEG C inert on hidden). d17 7.132 = floor. Ship-prep found kd20 needs
6 JSON caches relocated to writable /kaggle/working (not read-only /kaggle/input).**

**exp95 LE-SLAM 5.x bet = NULL: cat-class 20.83 (WORSE than v3 16.28). v3's alias-robustness
= SELF-MAP re-crossing, NOT particle cloud; GBM emission drops it → catastrophic aliasing on
wells v3 nailed (fb03ae90 3.1→32). Hybrid monotonically hurts. UNIFYING WALL (now 6x): the
ORACLE always shows signal (route-oracle 6.62, cat-quad 7.29→pooled 5.34) but NO TRUTH-FREE
GATE realizes it — same wall as catThr-leak + context-transfer-flip. We GENERATE good
hypotheses, cannot SELECT truth-free. Cat class AMBIGUITY-limited for our machinery.
CEILING = driver19/20 consensus ~7.0-7.25 (if transfers). 5.x needs Tucker-class per-well
LEARNED ENGINE (datum RMS 2.35) — every agent's recommendation. LAST untried = DEEP model on
RAW GR sequences (exp90 killed hand-FEATURE geology R2~0, but deep-on-raw untested). exp99
Hail Mary. Decisive gate = round103 driver19 transfer.

**exp98 truth-free cat detector: catThr NOT separable truth-free (AUC 0.66, precision 0.19
— no clean signature = wall AGAIN). BUT re-gated LEG C still beats driver19 (pooled 7.213
<7.247) on both held pools — works b/c C2_LE's OWN agreement gate catches false-pos; detector
= cheap pre-router. Recovers ~20-40% of leaked gain → DEPLOYABLE consensus ceiling ~7.21
(~7.01 public), NOT 7.107. Thin (3v2 held). driver21 = driver19+truthfree-C2_LE. Detector
NOT usable to route leslam standalone (needs downstream agreement filter). DEPLOYABLE LADDER
(IF driver19 transfers per round103): d17 7.132 → driver19 ~7.04 → driver21 ~7.01. ALL moot
if round103 says base doesn't transfer → d17 ceiling. round105 deep model = 5.x Hail Mary.

**exp99 DEEP MODEL 5.x Hail Mary = NULL. Deep biGRU on raw-GR pooled 12.35 (vs d17 7.335),
WORSE every pool+class incl cats (deep cat 19.47 > d17 17.02). Compose 7.64-7.95 (>d17).
Deep-on-raw extracts NO cat signal hand-features missed → confirms bedding-parallel wall
independently. 5.x CLOSED across ALL axes: post-correction(6x), learned-dip(geology R2~0),
LE-SLAM(drops self-map), deep-on-raw(worse everywhere). Leaders' per-well edge NOT
reproducible on our data. REALISTIC CEILING = consensus ~7.21 (IF transfers) or d17 7.132.
DECISIVE GATE = exp97_transfer (was interrupted by session restart, json empty, RE-RUNNING).**

**exp100 joint multi-well inversion = NULL (odyssey route DEAD): shared relief surface
(c_well-removed, std 54.5ft) is SPATIALLY UNCORRELATED at well spacing (pure nugget 25.9ft,
sill=0) → surface can't propagate datum; 3-5x noisier than GR-decode 5.45. Pooled 7.676
(+0.341). 3rd independent confirm (exp77 24.98/exp81 16.07/exp100) datum NOT spatially
recoverable. KEY: agent confirms cat gain = MODE/shape AMBIGUITY (quadO 7.29 vs 17.02),
NOT datum → REINFORCES the HEDGE thesis (round109): hedge posterior-mean over modes is the
RMSE-optimal response to ambiguity, no selection needed. Hedge = the right + last real lever.**

**exp102 HEDGE = NULL (8th confirm). KEY: d17 is ALREADY a hedge (median-of-7 comb2) —
on gated wells d17 6.07 is only 1ft above oracle-best-mode 5.32. Posterior-mean math
correct (hedge 8.66 << blind-commit 14.09) but NO committer to improve on; averaging drags
d17 toward weakly-calibrated (0.617) wrong mode → pooled 7.568 (+0.233). "We commit, should
hedge" DIAGNOSIS WAS WRONG. Only SELECTION (oracle 5.32) beats d17 → needs truth → the wall.
d17 = near-optimal given no truth-selection. REMAINING untried = matcher-quality at SOURCE
(misfit stretch/squeeze DP, learned reference transform) to sharpen the MODE SET itself +
synthetic-pretrain deep (round108 running). If those fail, d17 7.132 is the honest ceiling.**

**exp101/102 SYNTHETIC PRETRAIN = DEFINITIVE 5.x CLOSURE. Generator 7/7 validation PASS
(Yanza operator + exp45 dip + toe spectrum all match real). 10k synth wells pretrain →
deep model +0.019ft over scratch = ZERO. pooled 15.86, cats 25.15 (WORSE). "Bedding-
parallel GR does NOT encode local dip-rate; abundant perfect data can't teach a map that
isn't there. SIGNAL ABSENCE, not sample size." Deep per-well axis CLOSED both directions
(773 exp99 + 10k synth). 5.x DEFINITIVELY CLOSED (hedge/joint/deep/select all dead, 8+
confirms). d17 7.132 = honest ceiling. ★ driver20 PUSH SUCCEEDED: dataset version created,
download-verify 28 files OK, kernel rogii-driver20-m RUNNING on Kaggle = the empirical
consensus-transfer test. Public score pending.**

**Jul-8 leak-reading CLOSED honestly: (1) lam5 blend probe = STRUCTURALLY DEAD in code-comp
(static CSV emits visible-well ids, can't match hidden rerun → COMPLETE blank score). (2)
exp103 rule-extraction: NO transferable rule. Public geometry only strong signal = "away
from kriging/CV legs" (d17 ALREADY does); novel v10-hedge FAILS local (+0.23 worse). Expected
gain ~0.00, nothing ships. Bookkeeping verified 1e-12. Leak path closed — reconfirmed design,
no new move. REAL threads: driver20 (PENDING = recompute-on-hidden consensus test) + DP
stretch/squeeze misfit engine (round111 = the SOLVING move). d17 7.132 = anchor.**

**★★★ FINAL VERDICTS (2026-07-08): (1) driver20 SCORED 7.132 = IDENTICAL to d17 → consensus
does NOT transfer to Kaggle context (LRO-500 artifact confirmed empirically). (2) DP
stretch/squeeze engine = DEFINITIVE closure: localization 65.0% == cap exactly (not >72%);
mechanism WORKS (minima 2.97 vs 3.47, true-rank1 49% vs 30%) but STILL caps → the 65% cap
is INFORMATION-THEORETIC not object-limited. Better matcher CANNOT recover missing datum.
(3) TVT_input = truth on PREFIX only, NaN on lateral (already used as heel anchor; NO hidden
answer). ALL AXES CLOSED (~12 nulls). d17 7.132 = RIGOROUS CEILING for this data. 5-band
needs info NOT in features (GR+geom+prefix), proven. Only cat lever = orthogonal info
(formation logs = truth-stripped on test) → unavailable. ENDGAME: secure medal, final pair
d17 + m25v3 hedge, MERGE urgent. Private reveal may favor genuine solver d17 over public-
overfit leaders. Deadline Aug 5.**

**★ REOPENED (2026-07-08, user "you can't quit" → adversarial self-review found 2 REAL
cracks): (1) HIDDEN ASSUMPTION IN ALL CLOSURES: every matcher/localization/oracle test
used the TYPEWELL reference (1 distant vertical WIRELINE well, operator-distorted 0.79/
17.8/red-noise — smoothing it alone was +12pts!). "65% cap information-theoretic" is only
true RELATIVE TO THAT REFERENCE. Untested: LATERAL-ATLAS reference = GR(TVT) from
neighboring training LATERALS (3-6k labeled pairs each, SAME LWD tool = zero operator
error, ~1kft away per Yanza). exp81 used laterals for structure (rough, failed) but NEVER
for the GR signature. exp112 running. (2) ORACLE DISCREPANCY: our quadO 6.075 vs Yanza
4.28 on same 773 — if our basis is mis-specified the anatomy (CURV 8% "worthless") that
drove closures was mis-measured. exp113 auditing. If atlas breaks 65% → the 5-band door
reopens with an edge leaders didn't even use.**

**exp105 ATLAS = CLEAN KILL, closure TOTAL+AUDITED (2026-07-08): atlas localization 64.1%
vs typewell 63.6% (noise, << 72%) → 65% cap REFERENCE-INDEPENDENT. Physical diagnosis:
near-horizontal laterals sample vertically-THIN GR bands → atlas lacks vertical marker-bed
structure; the vertical typewell (even operator-distorted) is structurally better. (Real
echo: atlas kills some typewell aliases, containment 18→27%, insufficient.) Oracle audit:
our numbers CORRECT (Yanza 7.59/4.28 = HIDDEN-set oracles; ours train; 4.28 ≈ free cubic);
anatomy stands. 13 nulls + both foundations audited. ENDGAME: (1) driver21 = m25v3 packaging
(round114, the exp84-optimal slot-2: global blend ALWAYS fires, no routing-transfer risk;
forecast public ~7.21) → submit → FINAL PAIR = d17 + m25v3. (2) MERGE (user, Jul 29 block
risk). (3) intel watch + private-reveal positioning (d17 genuine solver may CLIMB when
public-overfit leaders fall). Rank ~107, top 2.5%, medal zone defended.**

**★★ THE CONTRADICTION (2026-07-08, user "I need 5, do research"): Tucker #2 reaches ~5 CV
from PER-WELL data ONLY = the SAME data we have → the info IS extractable, our FORMULATION
failed, NOT signal-absence. Our nulls used WRONG architecture: exp99/102 predicted DIP-RATE
(accumulates, local R2~0) → 12.35; matchers used HAND-BUILT alignment. UNTESTED = LEARNED
CROSS-ATTENTION ALIGNMENT (lateral GR=query × typewell GR(TVT)=key/value → soft-DTW → TVT
DIRECT; full-sequence context resolves ambiguity local models lacked; pretrain on 10k synth
data/synth_v1). = plausibly Tucker's "single model". exp107 (round115) running + leader-arch
research (round116). If it beats d17 → the 5-path is REAL. This is architecturally distinct
from all 13 nulls. Also found: typewell has Geology marker column (train-only, = exp82 lever,
weak). driver21=m25v3 packaging (round114) for the safe slot-2.**

**★★★ LEADER ARCHITECTURE DISCLOSED (2026-07-08 research, 4 notebooks pulled .intel_jul6b/):
5.x = DECORRELATED-TRACKER STACK → CatBoost, NOT deep nets (deep = CV-LB MIRAGE, score ~14
live!). RECIPE (evgendvorkin): track U=TVT+Z (line R2 0.99); stack PF-ANCC + PF-Z-velocity
+ 7-beam + multi-scale NCC self-corr (own-log repeat detector) + formation-plane KNN (40+
spatial feats) + posterior-std → ONE CatBoost, 5-fold GroupKFold, target=TVT-resid-from-
last-known. Ambiguity = Eagle Ford Milankovitch couplets (~15-25ft bimodal datum, 10% wells
=40% err) → estimate p, output p*a+(1-p)*b (posterior MEAN hedge). Heel-calibrate GR a,b on
KNOWN HEEL → 80% localization (vs flat 8%; we got 54-65% — GAP). amerhu HMM 5.18/PF 4.90/
blend 4.57. TRAP: CV-LB dies <6; blend of correlated engines dead, need DECORRELATED source.
OUR GAP: we have trackers but fuse by MEDIAN/hand-routing, NEVER a GBM STACK + never NCC-
self-corr / spatial-KNN / posterior-std-as-feature. exp99 null was CORRUPT (MPS errors, 6
epochs). exp36 was from-scratch tabular (overfit) — STACK over TRACKER OUTPUTS is different.
LAUNCHING GBM tracker-stack = the real leader-validated 5-path. My hedge failed on WRONG mode
spacing (8ft not 15-25ft Milankovitch).**

**★★★ ALIGNMENT TRANSFORMER WORKS (2026-07-08, exp115): cross-attention TVT-DIRECT = 7.435
OOF (vs d17 7.335) — MATCHES d17 where dip-rate deep models got 12-15! Architecture WAS
the problem. BEATS d17 on flagged (5.98 vs 6.01). Single learned model = Tucker's class,
UNDER-TUNED (501s, no synth pretrain, small). easy 4.63/flagged 5.98/cat 17.18. = a
DECORRELATED learned source (not matcher) for the STACK. IMPROVE it (synth pretrain 10k +
more epochs + bimodal ~15-25ft handling) → real shot at 5-6 AND a stack feature. round118
stack building d17-quality tracker feats (slow ~10s/well). NOTE: user re-submitted public
blend round2 x2 (13:46/14:01) — code-comp static → scores BLANK (mirage). driver21 held.
Most promising state yet: architecture validated, 2 live 5-shots (tuned-align + stack).**

**★★★★ BREAKTHROUGH (2026-07-08): MLP FUSION AT STAGE-5 RESOLVES THE CATASTROPHE CLASS
TRUTH-FREE — first method all campaign to move it. Replace d17's MEDIAN fusion with a
learned meta-learner over the 11 tracker outputs + gating features (sd_c2/div_rms/SLAM-
spread/agreement), target=TVT-resid-from-d17. RESULTS: cat 17.65→15.39 (−2.26!), net
+0.2 to +0.5 over d17 (varies by subset — STABILITY TBD). torch MLP > 3-way GBM (+0.45
vs +0.19 raw; smooth blend beats axis-aligned splits for collinear fusion). Best gated
deploy: MLP on flagged wells + shrink → +0.22 (293w); global-shrink 0.72 → +0.54 (210w).
= the LEADERS' posterior-std gating recipe WORKING on our trackers. My earlier "ensemble
dead +0.02" was a FLATTERING EASY 157-well subset — WRONG. uv venv .venv_gbm (torch 2.12
+lgbm+xgb 3.2+catboost 1.2.10). Feature cache .exp_cache/exp110_stackfeats/<well>.npz
(building→773). NEXT: round120 tune MLP+GBM + COMBINE (decorrelated fusers). DEPLOY = CPU
MLP fuser (ship weights, replace median at Stage 5). Projects ~6.5-6.9 public (top-40-60).
NOT 5 yet (cats 15.4 vs 7.29 oracle) but FIRST crack in the cat wall → pushable. User
drove this: pushed ensemble→NN→combine→tune. CRITICAL: run python FOREGROUND (nohup& gets
reaped). Full-773 verify armed at 760 wells.**

**★★★ round119 ALIGNMENT BLEND = CLEAN CONFIRMED GAIN (2026-07-08): improved alignment
transformer (bimodal 2-mode head h64 + early-stop training, NOT synth/width) beats d17
STANDALONE (7.321 vs 7.335, first learned single model to). BLEND (1-w)*d17+w*align w~0.3:
OOF 7.227 (−0.11), 3/4 pools, cat 16.67 (−0.35). HELD-OUT (tune e24+p349, confirm
f150+v74) = −0.124 GENERALIZES. Decorrelation from SYNTH-PRETRAINED sibling (corr 0.957 vs
0.996 standalone) = the amerhu blend engine. GBM-stack-over-align FAILS (per-point delta
noisy for trees) — LINEAR blend is the way. CPU-deployable (sub-MB torch net + w). scripts/
exp116_align2.py (m2h64 best), exp117. → driver22 package + submit via TEAM account
(nencybansal, 5 slots today, 0 used). TWO confirmed decorrelated gains now: (A) align blend
7.227 CLEAN held-validated, (B) MLP tracker-fusion ~6.75 variance-y pending full-773. Grid
mismatch prevents direct combine (align=per-tail 50ft, MLP=stackfeats grid) — reconcile
later. Ship A tonight (confirmed), verify B (round120). Team account = deploy capacity NOW.**

**⚠️ MLP FUSION = EASY-SUBSET MIRAGE (2026-07-08 full-773 verify): raw MLP fusion on
ALL 772 wells = 7.420 vs d17 7.388 — a REGRESSION. The exciting "+0.54, 6.75" was
ENTIRELY the alphabetical-cache easy subset (first ~210 wells easier; d17 rose 7.29→7.42
→7.56→7.755→7.388-full as harder wells cached). Per-pool: MLP helps e24/p349 (−0.1/−0.28)
+cats (−0.23) but BADLY hurts f150 (+0.68) → net negative. LESSON: never trust a fusion
number on a partial alphabetical cache — the subset difficulty varies wildly. Gate on FULL
773 before any excitement. Linear fusion also failed (8.16/8.72). Feature analysis: sd_c2/
div_rms are top features but PURELY nonlinear (pearson~0, MI 0.52/0.56) — signal is nonlinear
gating, much is error-MAGNITUDE not direction. MLP raw not submittable. round120 tuning+gating
+shrink/gate MIGHT rescue to beat d17 (keep d17 on f150, MLP on cats) — submit ONLY if clears
d17 at full scale. REAL confirmed gain = driver22 align blend 7.265 (−0.07 held-validated),
DEPLOYED+running on team acct. My full-773 caution was correct; discipline caught the mirage.**

**round120 RESOLVED the MLP picture + driver22 SUBMITTED (2026-07-09):
(1) MLP raw-regression was a DEPLOYMENT error not real — deployed MLP uses GLOBAL-SHRINK
(keep~55%, guarantees >=d17 by construction). Frozen-snapshot MLP+shrink = +0.32 vs d17,
cat 17.24, seed-std 0.04 (stable). The +0.22/+0.54 swing = MOVING well set (cache grew
mid-session). Best MLP = h[96,48] dropout0.3 lr1e-3 gshrink bag3seeds. MLP BEATS 3-way GBM
(+0.32 vs +0.27, cat 17.24 vs 17.64).
(2) BEST COMBO = REGION-SPLIT (MLP-on-flagged / GBM-on-easy) + gshrink keep0.53: full-772
CV 7.256 (+0.132 vs d17 7.388), cat 16.80 (−0.42), ONLY combo beating d17 on BOTH held
pools (confirm f150/v74 +0.017 = honest transferable part). STACK 7.189 REGRESSES confirm
pool = mirage, rejected. corr(mlp,gbm)=0.66 decorrelated.
(3) HONESTY: big +0.132 is partly easy-pool mining; transferable = +0.017 confirm + cat
−0.42 (real). region-split ~TIED w/ driver22 (7.264) but DIFFERENT AXIS (catastrophe-via-
trackers vs alignment) → DECORRELATED → STACK is the real upside.
(4) SUBMIT MECHANISM: `kaggle competitions submit -c <comp> -k <owner/kernel> -v <ver> -f
submission.csv -m ...` — CLI `-k` submits the KERNEL directly (triggers hidden rerun), works
with team token, NO browser. CSV-only `-f` scores BLANK (rerun comp). driver22 SUBMITTED via
this (team acct, ref 54477105). Files: scripts/tune_mlp.py, tune_gbm.py, tune_common.py,
final_gbm.py. NEXT: driver23=region-split → submit #2; align+fusion STACK → submit #3.**

**🎯 FIRST CONFIRMED CLIMB OFF d17 (2026-07-09): driver22 align-blend = PUBLIC 7.097 vs
d17 7.132 = −0.035! First real improvement over d17 in the whole campaign. The decorrelated
alignment-blend gain TRANSFERS (local −0.07 → public −0.035, ~half transfers — honest). VALIDATES
the decorrelated-source-stacking method on the REAL leaderboard. Submitted via team acct
(nencybansal) CLI `-k`. Implications: (1) driver23 region-split (beats d17 on ALL 4 pools
local, better-validated) should also transfer → submit it. (2) STACK driver22+driver23
(decorrelated, both transfer) → the real upside, build it. (3) The roadmap-to-5 method is
proven: accumulate decorrelated sources, each transfers a sliver, stack them. Path continues:
gating expansion (per-seed posterior-std), NCC tracker, unify stack. local→public offset:
driver22 local 7.264 → public 7.097 (~0.17); d17 local 7.388 → public 7.132 (~0.26).**

**STACK VALIDATED (2026-07-09): driver24 = 0.43*align + 0.57*region-split BEATS BOTH solo
fusers on confirm pools (8.040 < driver22 8.080 < driver23 8.196 < d17 8.213), all 4 pools.
Local full 7.131 (vs driver22 7.232, driver23 7.256) → projected public ~7.00 (−0.10 over
driver22's 7.097). DECORRELATED-STACKING THESIS CONFIRMED with real numbers. driver22(align)
is the STRONGER solo (confirm −0.13 vs driver23 −0.017) matching its 7.097 public. Gating
derivable helps (region confirm 8.196→8.11, exp121_gating15.json — workflow errored on
StructuredOutput but cache usable). NCC weak standalone (13.9 vs d17 5.4) — decorrelation
TBD. driver24 packaging → submit #3. Method compounding as predicted: each decorrelated
source a sliver, stack them.**

**MEASUREMENT-NOISE WALL + posterior-std lead (2026-07-09): Phase-2 gating gave NOISE-SIZED
gains. KEY findings: (1) confirm pools (f150=150,v74=74 wells) swing ±0.16ft from torch-CPU
MLP nondeterminism — BIGGER than the ~0.06 gains chased. So confirm-pool metric is UNRELIABLE
for sub-0.1 fusion tweaks; judge off FULL-CORPUS RMSE + NSEED>=5 going forward. (2) our
posterior-std (across-seed FFBSi-mean std, exp122_pstd) = corr 0.264 w/ error — strongest gate
we have but FAR below leaders' 0.66-0.92 → we compute it WRONG. HYPOTHESIS: leaders use
PARTICLE-CLOUD WIDTH per row (within-timestep weighted particle spread, before FFBSi collapse),
not across-seed std. round127 investigating (wmrmbb1pd) — if it hits 0.66-0.92 = a REAL gate.
STRATEGIC INFLECTION: fusion lever ~TAPPED at 7.0 (driver22 7.097, driver24 stack ~7.00-7.10).
Path to 5 now needs HARDER TRACKER-QUALITY work (correct pstd, sharper matcher, bounded SLAM,
heel-calib 65%→80%), NOT more fusion slivers. pstd cache exp122_pstd/ (772 wells, N=7).**

**FUSION/GATING TAPPED — 3 honest negatives (2026-07-09): NCC (signal-free), gating (noise-
sized), cloud-pstd (WORSE 0.175 vs across-seed 0.264). KEY: cloud width is filter SELF-
calibrated uncertainty → tight when confidently WRONG → misses biggest errors. Across-seed
pstd 0.264/0.524-vs-own-error is the HONEST truth-free SLAM-gate ceiling; gap to leaders'
0.66-0.92 = TRACKER CALIBRATION, not std definition. RELIABLE METRICS: (1) deterministic
per-row corr over full 3.78M rows for gate quality (no MLP randomness); (2) full+tune RMSE
NSEED>=5 + error bars (~0.037 seed-noise floor), require both same direction. INFLECTION:
fusion lever ~TAPPED at 7.0 (driver22 7.097 confirmed, driver24 stack pending ~7.00-7.10).
Path to 5 now needs DEEP TRACKER-QUALITY work (SLAM posterior calibration: jitter/break
variances, ESS/resample; sharper matcher; bounded SLAM; heel-calib 65→80%) — slower,
uncertain, gains near the noise floor. Banked real progress: d17 7.132 → 7.097. 5 is FAR
(needs −2.1); honest current ceiling with fusion approach ~7.0. Next lever = SLAM calibration
(uncertain). Do NOT reflexively spawn — wait for real score signal, judge deep investment.**

**🔴 CRITICAL MIRAGE (2026-07-09): driver23 region-split MLP+GBM fusion PUBLIC 7.310 —
REGRESSED vs d17 7.132 (+0.18 WORSE), despite beating d17 on ALL 4 pools locally (7.256,
confirm 8.196). The complex nonlinear fusion OVERFITS — even the "honest" held-out confirm
pools did NOT predict public. ROOT CAUSE: d17 local=7.388 but public=7.132 (d17 is MUCH
better on public); the fusion learned TRAIN-difficulty-specific corrections that help hard
train wells but HURT the easier public wells. CONTRAST: driver22 align-blend (SIMPLE linear
decorrelated blend) PUBLIC 7.097 = TRANSFERRED (−0.035). LESSON (load-bearing): SIMPLE LINEAR
DECORRELATED BLENDS TRANSFER; COMPLEX NONLINEAR FUSION (MLP/GBM/region-split/stack) MIRAGES.
Local CV (even honest GroupKFold held-out pools) does NOT predict public for complex fusers
because public well-difficulty differs. driver24 stack = 0.37align+0.63regionsplit = 63%
MIRAGE → likely also bad (pending). COURSE CORRECTION: pursue SIMPLE decorrelated sources +
LINEAR blends (align template), NOT complex fusion. Our REAL best = driver22 7.097. The whole
MLP/GBM/region-split line (round120-125) is a CV-mirage dead end. My earlier full-773-raw
regression flag was RIGHT; even the region-split "fix" miraged on public. 5 is FAR; honest
result = 7.097.**

**FINAL SCORES (2026-07-09) — mirage confirmed DEFINITIVELY: driver22(align simple) 7.097 ✅
| driver24(stack 37align+63region) 7.194 ✗ | driver23(region-split complex) 7.310 ✗. Perfect
monotone diagnostic: more complex-fusion content → worse public. driver24 sits exactly
between its components. CONFIRMS: simple linear decorrelated blend TRANSFERS, complex MLP+GBM
fusion MIRAGES, local CV (even honest held-out pools) does NOT predict public. BEST & FINAL
= driver22 7.097 (−0.035 off d17 7.132). Autonomous session outcome: 1 real climb (align
blend), complex-fusion line (rounds 120-125) = CV-mirage dead end. HONEST CEILING of current
approach ~7.0-7.1. Reaching 5 needs a BIG strategic investment (public-predictive validation
OR leader-level trackers) = USER's call, not incremental autonomous work. Do NOT resume
complex fusion. If continuing: only SIMPLE decorrelated learned sources + linear blend (align
template), each validated on a real slot (local CV untrustworthy).**

**LEADERBOARD STANDING CHECKED (2026-07-10, 4599 teams total): driver22 7.097 = RANK ~132
(top 2.9%). Shape of the gap: rank10=5.826, rank50=6.783, rank100=7.045, rank132(us)=7.097,
top5%(229)=7.165, top10%(459)=7.204. LITERAL top-5 = 5.08-5.51, a ~2.0ft cut from us —
NOT reachable via fusion/blending (that's the info-theoretic wall territory, needs
leader-level trackers). Realistic near-term target: rank50-100 (needs ~0.05-0.3ft, i.e.
6.8-7.05) via MORE simple decorrelated sources + linear blend (the proven-transferring
pattern). NOTE: this is PUBLIC LB only; private may reshuffle (code comp, hidden rerun).**

**★★ FABLE STRATEGIC REFRAME (2026-07-10): THE TOP IS THE DEBIAS ORACLE. d17 debias oracle
(remove per-well mean error) = 4.907; LB #1 = 5.083. Top teams have ~ZEROED per-well DC
(55% of our MSE). Arithmetic: kill all DC→4.9(top1), half→~6.1(top20), quarter→~6.6(top30).
Nothing else has this leverage (fusion/gating measured ±0.1). DC = WRONG GLOBAL GR LOCK
(couplet bimodality), NOT heel-calib miss (c_well pinned exactly by prefix TVT_input).
MECHANISM TO KILL IT (round133 running): (1) supervised per-well GR calibration on the
PREFIX (has both GR and TVT → supervised GR_well→GR_type(TVT) mapping); (2) WHOLE-LATERAL
global-mode hypothesis testing (enumerate couplet-multiple offsets, integrate calibrated
match over the entire lateral, prefix-junction continuity, kriging prior); (3) posterior-
mean hedge when ambiguous. WELL-LEVEL few-parameter decision = the class that transferred
(driver22) vs row-level learned corrections that miraged (driver23). GO/NO-GO METRIC =
lock accuracy (fraction of wells where correct mode wins whole-lateral score); leaders
cite ~80% localization. If 85-90% w/ hedge → low 6s or better.**

**round132 VERDICT + THE MISSING-ASPECTS PUSH (2026-07-10): markers are EXACTLY parallel
within wells (0.005ft, all 766) → datum COMMON-MODE, marker ensemble can't touch DC (my
independent-noise hypothesis refuted, catThr flat 15.8). BUT: ANCC = the NOISIEST pick of
6! BUDA (deep carbonate) leg beats ANCC leg −0.54 pooled (10.07 vs 10.61; raw krig −1.6,
−11%; biasstd 9.09 vs 10.26). → round134 = swap d17 bank ANCC→BUDA (all consumers incl.
dip-field prior), Fable review stage. USER REFRAME (correct): 5.083 EXISTS → signal is IN
the data; the BUDA miss proves we miss simple aspects. MISSING-ASPECTS RANKED: (1) global
lock [round133], (2) clean surface [round134], (3) STACKED REFERENCE [round135 NEW]: 773
train wells have GR labeled w/ TVT truth = supervised reference library never assembled;
build spatially-local leveled stacked GR(TVT) ref w/ marker-fraction stretch normalization
(spacings vary 30-40ft = real thinning) vs the ONE provided typewell; sharper emission →
powers the lock. Leakage risk = eval well + LRO-500 region must be excluded from stack
(Fable review stage checks). All new workflows: fable for complex stages + MANDATORY fable
review of opus work.**

**round133 DECISIVE NEGATIVE — LOCK HYPOTHESIS REFUTED, MAP REDRAWN (2026-07-10):
d17 implicit lock accuracy = 87.5% (ABOVE leaders' ~80%) — global couplet lock was NEVER
our deficiency. DC decomposes: (a) 87.5% correctly-locked wells = CONTINUOUS sub-couplet
DRIFT, corr 0.788 with slope error → INTEGRATED SLOPE/DIP ERROR is most of recoverable DC;
(b) 12.5% cat tail = GR ANTI-INFORMATIVE (true mode scores WORSE than alias by ΔNCC −0.17
— gain drift vs the distant typewell makes the lie outscore truth; why ALL GR-based cat
attacks failed). Whole-lateral NCC lock ceiling 0.671 << 0.875. Decider = no-op (best
config collapses to d17), fails 4-pool bar, NOT shipped, no slot spent. exp133_global_lock
.py + exp_dc_couplet_diag.json + exp_prefix_calib_diag.json (affine full-prefix calib >
heel-only, sigma 13.8). 5-PATH NARROWS TO: (1) slope/drift estimation [BUDA swap round134
= better slope prior], (2) gain-drift-robust emission [stacked local reference round135 =
local gain character]. Both already running — the negative CONVERGES on them.**

**round135 REFUTED + AXIS CLOSED (2026-07-10): the PROVIDED TYPEWELL IS the leaders' ~80%
localization (TW 78-81% hit3 on honest harness). The 773-well supervised stack, even ORACLE-
aligned (exact 0.007ft marker leveling + oracle stretch), is BLURRIER: 68% hit3, wider peaks,
more aliases. MECHANISM: real inter-well bed-thickness variation blurs any cross-well stack —
the organizer's paired typewell encodes per-well geology better than anything we can build.
Emission swap regressed +4.7 pooled. REFERENCE AXIS CLOSED. Salvage: marker-fraction stretch
machinery validated; c_M=TVT+Z−marker exact. AXES NOW CLOSED with honest measurement: lock
(87.5%), reference (typewell best), surface pick (BUDA fused~0 pending), fusion, gating,
routing. LAST NAMED LEVER = DRIFT: round136 launched (wm924ih70) — per-well robust linear
regression of sub-couplet (±8ft) window match-offsets vs MD → truth-free drift estimate →
d17 + shrunk linear correction; well-level 2-param deterministic = the transferring class;
observability gate first (corr(b_hat,b_true)), Fable build + Fable review.**

**round134 CLOSED NEGATIVE (2026-07-10): BUDA bank swap through FULL d17 = 8.227 vs ANCC
7.720 on 150-well paired (+0.51 WORSE, 0/4 pools; ASTNL +0.44 worse too). The isolated-leg
advantage (−0.54) INVERTED when fused: d17's gates/s0/DAMP/routing are tuned on ANCC
statistics — bank swap breaks the calibration; sigma-gate mechanism (Fable review) was
right. SURFACE-PICK AXIS CLOSED. No slot spent. ALL AXES NOW CLOSED except DRIFT (round136
running): lock 87.5%✓, reference (typewell best)✓, surface pick✗, fusion✗, gating✗,
routing✗, weight (7.097/7.103 measured)✓. Everything rides on drift observability
corr(b_hat, b_true).**

**round136 CLOSED + USER MASTER PLAN ADOPTED (2026-07-10): GR-drift regression DEAD —
corr(b_hat,b_true)=+0.05; b_hat is REPRODUCIBLE (split-half +0.60) but tracks matcher
systematics not error. MECHANISM: d17 is GR-driven → all GR-matchable displacement already
absorbed; residual drift lives in the GR-BLIND NULL SPACE. Dead-end #25. → remaining DC is
readable ONLY from non-GR observables if at all. USER PLAN ADOPTED (target-corr table:
0.56→6.5, 0.77→5.9, 0.96→5.1; promotion gates; lockbox; ≤5 scalars; distinct-mechanism
slots): round137 component-wise align blend (Δ=align−d17 split DC/slope/shape, w0/w1 grid,
w=0.46 shape; wdgjllig9) + round138 DC-predictability (spatial OOF-error kriging + robust
ridge + BLUP on geometry/surface-gradient covariates; wtxc5japo) BOTH RUNNING. These two
corr numbers decide the 5-band question. BUDA §3 declined (134 measured negative); §5
candidate-ranker (gain-invariant rank/derivative features + 1.95 prior-odds penalty) queued
as round139.**

**REVIEW CRITERIA LOCKED (user, 2026-07-10) for the 3 running threads:
r138 (DECISIVE): fold-safe corr(ê,e) ≥0.70 continue / ≥0.80 = 5-band plausible; report
pooled+per-pool corr, CALIBRATION SLOPE, optimal λ, post-correction RMSE; ABLATE covariate
families separately (spatial / geometry / structural / align−d17); high pooled corr from
regional offsets NOT enough — must survive LEAVE-REGION-OUT.
r139: margin DISTRIBUTION on the 97 mislocks (not just accuracy), 1.95 prior penalty,
damage count on 676 locked wells (<2%), truth-free.
r140: pairwise seed DC-corr — if >0.95 the bag is at ceiling, stop adding seeds; every
pool must improve.
COMBINATION ORDER: 139 discrete switch → 138 continuous DC → 140 align residual → conservative
global shrink only. MEASURE OVERLAP corr(138-correction, 140-correction): high → blend
conservatively; low → they compound. BOUND: cat tail ≈ half of DC MSE → perfect cat fix alone
= 7.10→6.10; the 5-band still needs 138/align for continuous drift. r138 cross-region corr =
THE campaign-decisive number.**

**★★ round138 = THE DECISIVE NULL (2026-07-10): per-well DC error field is SPATIAL WHITE
NOISE — Moran p=0.42, ~84% nugget at 439ft median spacing, flat variogram; kriging/ridge/
BLUP all |corr|≤0.07 (gate was 0.5). Padmate-audited, fold-safe, review-verified. Only real
signal: |e_w| MAGNITUDE predictable (tracker_std r=0.49) = risk not direction (already used
by gates). CENTRAL QUESTION ANSWERED: continuous-drift DC is unrecoverable from ANY held
info — GR (r136 0.05), align-DC (r137 0.14), space/geometry/structure (r138 0.03). POST-HOC
CORRECTION OF d17 IS CLOSED AS A ROUTE TO 5. Remaining: r139 cat tail (perfect→~6.1,
realistic→6.6-6.9), r140 align scale (~7.02-7.05), and the only 5-band route left = a
fundamentally BETTER BASE PREDICTOR (serious align-transformer scale-up w/ real compute) —
strategic/cost fork for the user. Elimination campaign cost 0 slots.**

**r139 CLOSED (no sign flip: 5 representations all prefer the alias — real ambiguity, not
gain drift; b*=0 never-switch optimal) + USER PLAN-2 ADOPTED (2026-07-10): reframe = predict
latent interpretation state (a_w datum, b_w slope, k_w alias) from NEIGHBORS or HEEL — the
last viable formulation. LAUNCHED: r141 deployment audit (w1nf5e71v: 3-visible-test geometry
vs train, TYPEWELL SHARING forensic [likely the game-changer — shared typewell = family
link], id/rowcount families) + r141b conditional transfer (class-conditioned nn e_w corr,
ALIAS ORACLE ceiling on 97 mislocks) + r141c target forensics (wna5lx1e8: markers=typewell-
boundary claim test via Geology column, algebraic identities, well families). STRIKES
already known: nn_corr 0.046 INCLUDED padmate twins; krig corr on cat = −0.08. SCHEDULE:
24h audit verdict; 48h r142/r143 if deployment support; Jul 15 demand corr≥0.7; Jul 22 one
probe; Jul 29 merge if nothing. r144 (anchored slope-integral seq model, align-init, well-
level DC/slope losses) = the better-base-predictor route, after audits, maybe GPU. r140
insurance still computing.**

**★★ r141c FORENSICS — RICHEST RESULT (2026-07-10): (1) MARKERS=TYPEWELL GEOLOGY BOUNDARIES
CONFIRMED (c_M = boundary TVT to ≤0.5ft/99.9%; NaN pattern locks mechanism; zero info beyond
typewell+TVT — book closed). (2) TYPEWELL TVT AXIS EXACTLY CO-REGISTERED with well datum
(best shift=0 ±0.5ft for 92.4%; hard-code shift=0; the ~8 shifted wells incl 89f1085d
[−7.75ft, markers uniformly 4.38ft off] = identifiably MIS-DATUMED interpretations,
flaggable at inference). (3) LABEL-NOISE FLOOR MEASURED: 5 same-bore duplicate pairs in
train w/ CONFLICTING interpretations — 11-21% of points >2ft apart, tails to 65ft; part of
the cat tail may be label noise; DEDUPE pairs in CV (5663b2b7/75d20a82, d085e611/efde6ac3,
684a6fc1/bcebcc5f, a4f989c2/d011f41b [GR corr 1.0000 const offset = same log], +1).
(4) TYPEWELL FINGERPRINT ANCHOR: 13 hash groups/34 wells share byte-identical typewells;
sharers' c_M identical ≤0.02ft → at deployment, hash hidden typewell vs 752 train hashes;
match = transfer exact boundary constants + family twin surface. = the user-plan r142
mechanism with a concrete legal key. Coverage on hidden unknown (visible 3 = placeholder
train copies). MD≡3D arc len (0.015ft); TVT_input==TVT exact; prefix contiguous row0.
Files: outputs/exp141c/*, scripts/exp141c_forensics.py. NEXT: 141-geometry + alias-oracle
size the channels → kernel-side transfer layer (fingerprint→twin; pad→local surface+alias;
else d17) with high-precision triggers only.**

**r140 CLOSED (2026-07-10): align scale-up dead — 7.321 standalone was a LUCKY SEED (+2σ;
7-seed mean 7.364±0.026); bagging pulls toward typical seed (bag7 7.347 > lucky 7.312);
h80/e90/marker-feature all worse (even oracle-lite mf 7.384). Best blend 7.2732 vs driver22
7.2646, 1/4 pools. exp116 family SQUEEZED DRY. driver22 local OOF mildly optimistic (lucky-
seed sibling) but public 7.097 is banked fact. 7 seeds cached exp140_alignpreds.pkl.
REVIEW PACKAGE COMPLETE: r138 dead (0.03) / r139 dead (never-switch) / r140 dead (1/4
pools) / overlap moot. NO submission from package. ALL rides on r141 geometry+alias-oracle
(running): channel exists → kernel-side transfer layer (fingerprint→twin, pad→surface+alias);
else → corrective program exhausted at 7.097 → r144 better-base-predictor + merge decision.**

**BOARD FINALIZED (2026-07-10): r141 audit → INTERPOLATION (hidden test = the 151 missing
Well1XXXX numbers, same field; nearest labeled parallel lateral ~260-320ft; 57 real
typewells serve all 773 wells [PNG-title OCR]; LRO-500 is 0th-percentile PESSIMISTIC vs
deployment — deployment-matched CV = leave-well-out). BUT r141b: transfer channel DEAD —
padmate e_w corr ~0.10, alias-oracle ceiling 9-12% (neighbor residual CONFIRMS the wrong
mode 91%: neighbors are locked, |e|~1.9ft ≈ alias). DECISION TABLE ROW 4 → r142 CLOSED.
BANKED: bank_driver22/ (13 files + SHA256SUMS; cloud copy in team dataset + kernel v1;
retraining does NOT reproduce — lucky seed). PERMANENT HYGIENE: (1) same-bore dupes share
a fold; (2) mis-datum flag = abstention/routing guard only until grouped validation proves
the annotation convention. RUNNING: r144 slope-integral prototype (wnerxi6ko — heel-anchored
cumsum(s·dMD), per-well DC/slope losses, leave-well-out CV, dupes co-folded, ≥3 seeds; NOT
align scaling). If r144 ceilings → strategy = MERGE-PRIMARY before Jul 29 + endgame
submission discipline.**

**r145+r146 CLOSED (2026-07-10): r146 risk-adaptive w — w1 tunes to 0 on tune pools (real
but tiny mechanism, top-decile only −0.066/77 wells, pooled −0.015 << resolvability). r145
deployment retest — LOO validates interpolation (d17 −6.6% uniform, tail most: cat −2.36)
and brackets public: LOO 6.85 < public 7.132 < LRO 7.335 (deployment ≈ 260ft-exclusion CV).
LRO-500 pessimistic in LEVEL not RANKING (no sign flips → past rejections stand). Kernel
already uses all close neighbors → NOTHING to harvest. Stacked ref worse even w/ donors
(hit3 67 vs TW 74; secondary channel harmful P=0.999); fKNN oracle-only; error-kriging
white noise even w/ padmates. any_all4_winner=FALSE. REMAINING TODAY: r144 (slope-integral,
grid mid-sweep) + r147 (leader replication — do their matchers mislock <12.5%?). Those two
verdicts complete the internal program.**

**r144+r148+r149 CLOSED (2026-07-10): r149 = premise wrong, d17 ALREADY uses affine-full-
prefix calib (byte-identical proof; heel window would REGRESS). r148 family axis = dead for
signed error (BLUP corr −0.016; only 2nd-moment: |e| omega²=0.047, mislock AUC 0.585 —
where-not-fix; hedge failed confirm). r144 slope-integral = CEILING: lam0=10 removes only
SELF-injected DC (ctrl 5.353→5.313), ZERO d17-residual DC extracted (d17 5.297); pooled
7.369±0.006 ≈ align 7.364 ≈ scratch 7.354 — THREE architectures, ONE ceiling (r138 null
reproduced inside a trained objective). exp144_slope_integral.py, lam0=10 = good stabilizer
default for any future seq model. INTERNAL PROGRAM: only r147 (leader replication) still
open. If r147 negative → terminal state: driver22 7.097 banked, ~25 dead-ends w/ mechanisms,
endgame = final-pair discipline (user shelved merge).**

**r147/150/151/152 CLOSED (2026-07-10): r147 leader trackers WORSE overall (mislock 1.4-2.3x
more; d17's cross-well prior dominates) BUT lik-PF decorrelated (corr 0.556) + fewer cat
mislocks (43.5% vs 60.9%). r151 gated PF hedge FAILED (rescues matched by breaks; no truth-
free gate separates; future gate needs PF-vs-d17 DISAGREEMENT feature). r150 external intel:
Tucker rank-2 = per-well non-tabular matcher sub-5 claim; Maus coarse-to-fine stretch-
penalized alignment; FFT rotation denoise (~4pt only, we already have affine calib); 5
Kaggle writeups independently confirm our unobservable-datum nulls; public 7.0-7.2 =
variance farming. r152 GATE TEST closed the Maus direction with PHYSICS: the ALIAS needs
LESS warp than truth (1.00 vs 1.29 couplets) → no monotone stretch penalty can flip the
sign; bundle-scale rigid worse (−0.24). TOTAL VERDICT: GR-vs-typewell similarity cannot
identify the true lock in ANY representation/scale/alignment class. RUNNING: r153 = last
unshipped proven-class candidate (simple 1-param d17+PF blend, honest tune/confirm, then
align compose → driver29 if beats driver22 all-4-pools).**

**★ r153/153b — PF COMPOSE = THE PROBE CANDIDATE (2026-07-10): honest full-773 PF blend:
corr(e_PF,e_d17)=0.48 (true d17 from exp90_feats.npz, exact driver22 arithmetic anchor).
Joint simplex grid at PROVEN c=0.46: EVERY b in [0.02,0.14] improves pooled AND confirm;
b=0.02 clears all-4 strictly; tune-chosen b=0.16 → pooled 7.1373 (−0.127 vs driver22) but
f150 +0.145. DRIVER29 = 0.38*d17 + 0.16*PF + 0.46*align, expected public ≈7.01 (65% ratio);
conservative fallback 0.48/0.06/0.46. PACKAGING (w79l5b0wm, extends banked driver22 pattern
+ exp147_amerhu_core lik_pf per-well). This is the user-plan §8 mechanism-probe slot.
scripts/exp153b_joint_grid.py has the full frontier. Cat SSE −7.7%, mislock 11.9 vs 12.1%.**

**🎯🎯 driver29 = 6.852 PUBLIC (2026-07-10) — BREAKTHROUGH TRANSFER: 0.38*d17+0.16*PF32+
0.46*align, local 7.137 → public 6.852 (−0.245 vs driver22 7.097 = ~190% of local delta!
predicted 7.01). RANK ~55-60/4599 (top 1.3%, from 132). Hidden set rewards the PF hedge
MORE than confirm pools (interpolation-rich hidden; confirm under-predicts — r157's tune-
concentration worry resolved FAVORABLY). NEW BANKED BEST = driver29. → driver30 = 6-member
simplex (d17/PF32/align/base14/v3/rd, BOOTSTRAP-MEAN weights .224/.122/.233/.222/.062/.136,
local 6.951 = −0.19 vs trio) packaging for MAIN-account submit TODAY; if transfer even
proportional → ~6.6-6.7 → rank ~35-45. PF sweep: MOM 0.999/t10/s128 slightly better (tune
6.664). Deploy surface: base14/v3/rd legs via selfleg (shipped in driver20/21 before).**

**STANDING DIRECTIVE (user, 2026-07-10): DO NOT STOP until we top — keep trying, researching,
brainstorming. THE ENGINE THAT WORKS (produced 7.097→6.852 in one day): manufacture
decorrelated per-well physics trackers (hedge-not-commit, standalone ≤1.5x d17, corr<0.7)
→ admit by tune-pool arithmetic (e24+p349 only) → bootstrap-mean weights (r157 method) →
ship simplex → public score calibrates. Transfer observed ~190% of local delta (hidden is
interpolation-rich). PIPELINE: driver30 pending (~6.6-6.7 target) → r159 driver31 hunt
(from-scratch 16-tracker simplex + multi-PF + HMM member) → fault HMM build (Winkler spike-
slab, the state-space change) → PF-128 refresh → literature re-mine for more per-well
tracker families (stochastic DTW n-best, Alyaev MDN multi-modal, EnKF ensembles = all
hedge-family, all unbuilt). Slots: burn on each validated rung, both accounts.**

**driver30 = 6.994 (2026-07-10): 6-member UNDER-transferred (local 6.951 → public 6.994,
WORSE than driver29 6.852). r157's warning exact: base14/v3/rd bundle had FLAT confirm →
no transfer → degraded hidden. Trio's ~190% over-transfer was PF-SPECIFIC (PF moved confirm
−0.03), not simplex-generic. PERMANENT ADMISSION LAW: a member ships ONLY if it improves
CONFIRM pools (f150+v74), tune gains alone = reject. BANK STAYS driver29 = 6.852 (~rank 57).
trio+PF128 confirm flat (8.113 vs 8.109) → also fails the law. Fault-HMM (r160, computing)
to be judged vs the TRIO by the strict law. Slots left: 3 main + 4 team today.**

**🎯 driver33 = 6.816 PUBLIC — NEW BANK (2026-07-10 night): 0.92*trio + 0.08*NCC(l10_d1)
landed in the 6.79-6.82 projection. Rebase deduction validated (NCC leg transfer carried to
the stronger base). DAY TOTAL: 7.097→6.816, rank 132→~53. Confirm law now 3-for-3 as a
transfer predictor (PF✓ NCC✓ pass→transfer; base14/v3/rd✗ flat→degrade). CLOSED tonight:
r166 prefix-supervised (build-section regime, no tail support), r167 AR-selfref (enrichment
works +monotone, segmentation destroys mislock guard −40%). RUNNING: r168 EM whole-path
(enrichment WITHOUT segmentation — the synthesis cell). Next slots midnight UTC.**

**★★ r171/r172 STRATEGIC FLIP (2026-07-11): NAMED-WELL AUDIT — leaders' public catastrophe
wells (86454a6f, 4c2208f5, 389ae58f, fb03ae90) are ALL in our tail AND driver33 BEATS their
published per-well numbers on them (34.5 vs ~50; 33.0 vs ~50; 26.7 vs 40.8; 9.8 vs 34.3).
THE TAIL IS NOT THE GAP. Leaders' 1.5-2ft edge = BODY precision (ordinary locked wells: ours
~5.1 RMSE, theirs ~3.5-4). Even #2 publicly says direction-unobservable; #6's sequential
matcher drifts like ours. PUBLIC LB unresolvable <6 (Tucker+#2) → confirm-law protocol
validated. r172 shrinkage REJECT (ledger #26): S_w disagreement = VALIDATED hardness
instrument (Spearman 0.5 out-of-pool, first ever) but nothing safe to shrink toward.
NEW PROGRAM: body-precision on the 87.5% locked wells (sub-couplet drift+wiggle) = the
route from 6.8 toward 5.5. Possible: kokinnwakashuu claims private ~74% deep rows punishing
extrapolation → conservative tails for final pair. Bank: driver33 6.816 ~rank 53.**

**🎯 2026-07-11 morning — NEW CHAMPION 6.791 + G* PROMOTED TO GATE.**
Parallel session's "driver34 trio+HRNCC" = 0.85995*trio + 0.14005*hr8_l06 (hires-NCC,
scripts/hires_ncc_matcher.py, emis=l06_d1, block 8ft, bin 0.5ft) scored **6.791 twice**
(exact replicates; 6.816 x3 on trio+NCC clones → scoring is DETERMINISTIC given the CSV).
HRNCC = simplex admit #3 (after PF, NCC). Shipped w=0.14005 is AT the G* optimum (flat
0.10–0.15). **G* instrument (r176) now 2/2 EXACT out-of-sample**: frozen G*-reg
(pub=1.5727*G*−1.7369) predicted 6.793 vs actual 6.791 and 6.816 vs 6.816; grlow-reg
carries a consistent +0.13 optimistic bias → demoted. r179 pre-registration (timestamped
before scores, .exp_cache/exp179_preregister.json) + r180 full-coverage arbitration
(.exp_cache/exp180_arbitration.json; exp180_hr8l06_preds.pkl = hr8_l06 on all 772 wells).
CAUTION: fitting weights ON G* is Goodhart-bait — G*-fit simplex pred 6.654 but confirm
law FAIL; G* stays an INSTRUMENT, confirm law stays the gate. Ship candidates (pending
driver34 ttail arbitration, ref 54555653 team acct, pred G* 6.727 vs grlow 6.951):
champ+0.05*ttail pred 6.736; grlow-fit simplex pred 6.728 (conf-law PASS, radical
reweight align .62). driver35 kernel staged in r181 (notebooks/submit_driver35_n/).

**🎯 2026-07-11 ~09:35Z — driver34 = 6.733 NEW CHAMPION; G* 3/3 EXACT, PROMOTED TO PRIMARY GATE.**
driver34 (0.90*d33 + 0.10*ttail_t5, team acct) scored 6.733 vs G*-reg pred 6.727. G* errors
across 3 arbitrations: +0.002/0.000/−0.006 ft. grlow-reg +0.13..0.22 bias = demoted. ttail
axis publicly validated (conf-pool veto overruled on this axis; Goodhart caveat stands for
candidates FIT on G*). Champion ladder Jul 11: 6.816→6.791 (trio+hr8_l06 w=.140)→6.733.
Next: driver35 T_TTAIL=0.10 on hr-champion pred 6.699 (r182 16-seed runtime fix first —
32-seed ttail = 37-44 s/well on Kaggle → 2.4-2.7h vs 2.5h budget = timeout risk).

**2026-07-11 midday — r184/r185 closed both moonshots WITH mechanism; r186 apportions.**
r184: neighbor-transfer + 3D residual-krig headroom = only −0.098 pooled joint (pred 6.678);
transferable field is C=TVT+Z (NOT raw TVT — 100-700ft errors); only ~3% of >800ft band-SSE
is spatial. r185: the big smooth residual = per-WELL-NUMBER interpreter regime (constant+
drift), self-consistent within a number (0.613), uncorrelated across same-lateral twins
(−0.107, median |dDC| 2.85ft), and HEAD-INVISIBLE (TVT_input==TVT exactly on prefix; OOF
R²<0 from 24 features). TRUE-DC oracle = pooled 7.05→4.69; DC+slope → 3.28 (half the error).
LEADERBOARD Jul 11: top 4.859 (shu01), band 5.26-5.94 → leaders G*≈4.2-4.6 vs ours 5.42.
CONTRADICTION: twins agree (2.85ft DC) better than we agree with truth → most DC error is
DECODABLE, not noise → r186 twin cross-truth apportionment running. Aggressive final-pair
candidate banked: champ+0.087*A+0.139*B(+regime leg) pred 6.678-6.670 (.exp_cache/exp184_joint).

**2026-07-11 r186 — THE PARADOX RESOLVED: ~90% of champion error is DECODABLE; leaders = better BASE DECODER.**
Twin cross-truth on truly co-located pairs (<10ft, n=17): interpreters agree to 1.46ft RMSE /
0.46ft DC; champion 3.5ft → decodable frac 0.89-0.95 (both humans agree where we fail).
The 10-60ft pairs are different laterals (twin_rmse 10.6) — pooled stats were poisoned by them.
Decodable-DC headroom alone = 1.34 G* → residual G* 4.08 = bottom of leader band (top 4.859,
band 5.26-5.94 → leaders G* 4.2-4.6 vs ours 5.42). MEASURED-DEAD post-hoc: family-conditioned
selection (champ wins all 57 families LFO), leg reweight/stratum rules, median-of-legs, regime
corrector, couplet-lock. Oracle best-leg-per-well G* 4.24 (in leader band) but selector
unobservable; split-half shows ~47% of leg-choice signal is well-stable yet no feature finds it.
ALIGN = only interpreter-AGNOSTIC leg (own 1.78 = cross 1.83; champ 1.89 vs 2.57) with highest
decodable fraction — and it's TINY (XAttnAlign2 h=64, 1 layer, 2 modes; align-alone pooled 7.31).
→ r187: SCALE THE ALIGN DECODER (capacity+context+modes, multi-seed vs seed-fragility, ensemble,
refit trio weight). This attacks the source. Full data: .exp_cache/exp186_apportion.json.

**🎯 2026-07-11 ~14:00Z — driver35 = 6.694 NEW CHAMPION (4th G* point).**
driver35 (0.90*(trio+hr8_l06) + 0.10*ttail_t5 s16, team acct) = 6.694 vs G*-pred 6.717
(resid −0.023, good direction; s16 penalty didn't materialize). Ladder Jul 11: 6.816→6.791→
6.733→6.694 (−0.122/day), rank ≈22-28. G* residuals: +.002/.000/−.006/−.023 — 4/4 direction.
Final pair: aggressive=d35 lineage, conservative=d33/29 ttail-free. Team 3 slots left Jul 11;
next public direction question = AB neighbor legs on d35 base (pred ~6.6), ship Jul 12 per
anti-overfit policy. Align2: capacity-alone DEAD (3 protocols, 2 sites); live = multi-scale
context pilot (r187) + synth-corpus 10x scale-up (GPU box) + Kaggle session1 original-corpus
replication (bharatmohan/rogii-align2-train-gpu v2).

**2026-07-11 TARGET RAISED (user): the 4-BAND (public ≤4.99, i.e., #1 territory; top=4.859).**
Arithmetic: public 4.99 → local-G* ≤4.28 (frozen reg). Measured oracles: best-leg-per-well
G* 4.24, perfect per-well DC 3.92, interpreter floor ~1.5ft co-located. So the 4-band =
recovering essentially the ENTIRE decodable pool — demands a base decoder near the interpreter
floor. Window-matcher (align) retraining is closed (capacity/features/both all law-FAIL);
therefore the 4-band REQUIRES the full-well sequence decoder program (+ synth-10x corpus,
+ gen-2 neighbor legs). Blend arithmetic bank is empty after driver36 (prereg 6.599).

**2026-07-11 ~21:00Z — driver36 = 6.717: G* FIRST MISS. Spatial legs DON'T transfer. d35 6.694 champion.**
A/B neighbor legs (prereg 6.599 via G*, conf-law −0.144 local) scored +0.023 WORSE than base.
Mechanism: gains ride co-located train twins; hidden wells = missing numbers whose twins are
likely also missing. NEW RULE: G*/conf-law extrapolate ONLY for per-well GR tracker legs
(4/4 exact: trio/NCC/hires/ttail); spatial label-transfer legs = closed for public. driver37
rebuilding as d35+fb_dip (DP leg, tracker-family) with sibling-sanity diagnostic before ship.
Head lever seed-robust (s1401 delta −0.140). Slots: team 3/5 used Jul 11.

**2026-07-12 r209/r210 — POOL MAPPED + COMPETITOR INTEL.**
r209 band apportionment (17 co-located twins): decodable pool = slope 43.8% + level 38.1% +
lowband 14.7% (all >0.84 shared between interpreters); texture 3.4% = interpreter-private.
LEVEL IS REAL (dec_frac 0.92) — the 4 dead level-attacks failed on execution not premise.
r210 intel: **mycarta rogii-geosteering-toolkit** (github, Matteo Niccoli = serious competitor,
public toolkit for our exact task) — confirms our dead-list (NCC/shape family), gives:
Q-3D tortuosity (Jing 2022) their best feature −0.107 (steering proxy → slope pool);
StratifiedGroupKFold-by-well (they REJECTED spatial BlockKFold: wells interleaved →
interpolation not extrapolation); global TVT~Z r=−0.96 collapses to ~0 within-lateral
(the canonical shakeup trap). Denoising recipe: random-intercept EM (b_w/well, James-Stein,
guardrail std≈2.9ft twin bound) — upgrades every leg. r212 building both cheap items.
Final-pair diversification: spatial-heavy (d36-style) × within-well/denoised legs.

**2026-07-12 driver37 = 6.842: 2nd consecutive miss (+0.20). CROSS-WELL RULE discovered.**
Own-well-GR legs (NCC/hires/ttail): 4/4 transferred on prediction. Cross-well legs (A/B spatial
+0.118, fb_dip dip-bank +0.20): both missed despite ALL local gates passing. Sibling-sanity only
catches nearest-sibling dependence, not diffuse banks. RULE: cross-well aggregation ⇒ no-transfer
assumption, tiny-weight public probe only. Champion driver35 6.694. Ship freeze pending autopsy.
Level+slope pools closed for single-well hand observables (r211/217/218); live: r219 joint (now
probe-grade), r220 generator-fix → synthetic drift supervision, learned lane, archive stacking.

**🔑 2026-07-12 r222 — THE LABELS' GENERATING PROCESS FOUND (breakthrough-class).**
Truth = "manualTVT" = human interpretations in Rogii StarSteer. Patents give the tool's
optimizer: US11480045B2 — big overlapping segments (10-60%), ONE scalar dip/segment,
K = corr^p1·SC^p5/(SQ^p2·dipTV^p3·regionalDipDev^p3), boundaries at GR-character breaks,
tie-break toward previous dip; US10353356B2 — TVT affine in VSD within segment, median dip
smoothing. Label forensics CONFIRMS: truth 2nd-diff exact-zero 28% vs champion 0.05% (580×
tool signature); segment median 135ft; no dip quantization. Naive piecewise-projection of
champion = null (knot density already matches; oracle <0.01). THE EXPLOIT = replicate the
optimizer (r223 patent-K decoder, building): decode via the tool's own cost. regionalDip term
means labels were MADE with neighbor-dip context (part of data-generating process — but fb_dip
still failed publicly +0.20, r221 autopsy must reconcile before any cross-well term ships;
own-well-only variant = ship-safe). Full details notes ROUND222 + .exp_cache/exp222_prior.json.

**2026-07-12 r221 AUTOPSY — THE DROP-20 LAW (the 6/6 gate).** fb_dip's gain rides the ~20
nearest wells (drop-20 → gain reverses 3×); leave-SELF-out retains co-located twins = the
local-validation flaw behind BOTH misses. RULE: cross-well legs validate leave-sibling-group-out
(drop-20) or don't ship; own-GR-only legs keep the ±0.023 G*-reg band (4/4). pub ≈ G*reg +
0.20·crosswell. No archive candidate passes. Champion d35 6.694; pair d35 + d33/29.
Live breakthrough bets: r223 patent-K decoder (own-well variant = ship-relevant; regional-dip
term must survive drop-20), r220 fixed-synth value test (corpus lever un-death check).

**2026-07-12 r224 INTEL (leader forum posts, verbatim-sourced) — STRATEGY-GRADE:**
- Public LB = ~50 friendly wells (26%); private = 74% (~150). Chris Deotte: "large shakeup."
  Tucker(#3): "public ranking is irrelevant. Get CV low and pray." yu4u: CV+0.30≈LB (his protocol).
- TOP METHOD REVEALED: "top lb use the nn's" (Shrey#6); k256(#11, 5.86): "primarily tabular
  models... improving the features", their CV 7.0→LB 6.0. Public notebook lineage saturates
  7.15+ (ρ=0.89 correlated) — real recipes private. → r227 launched: per-point tabular residual
  lane on champion base (cross-well PATTERN learning ≠ label transfer).
- WORST WELLS UNIVERSAL: leaders' worst = ours verbatim (86454a6f/fb03ae90/389ae58f/91db7070).
  Rishikesh(#2): "can categorize which wells will have large errors, but not the direction" =
  our measured wall, confirmed at rank 2. Body precision (not tail) = the whole gap.
- Cutoff forecasts (k256): silver ~6.5, gold ~5.5, prizes 4.7-4.8. Seed noise on stochastic
  pipelines ±0.1-0.4 (ours deterministic — advantage).
- SDK data model (python.solo.cloud): labels = Interpretation>Segments(start/end MD,
  horizon_shifts start/end) = PWL-in-MD panels, one dip/panel; get_tvt_data(md_step=1).
  StarSteer processes BACKWARD from terminus → first post-PS panel most free (matches PS
  abrupt-dip failure mode). pow1-5 defaults NOT public — recover empirically. Overlap 10-60%.
  TVT recursion verbatim in patents: TVT(n+1)=TVD(n)−[VS(n+1)−VS(n)]·tan(α).

**2026-07-12 r235 — TWO-GROUP RIDDLE = OUR G*; FINAL PAIR CONFIRMED d35+d33.**
k256's rule IS G* (no sharper rule survives selection null over 15 subs). G* blind spot formal:
ranked BOTH misses above champion — spatial/dip families instrument-invisible (never greenlight).
Private-mirror: d35 edge concentrates in friendly 25%; on private-like strata d35 ranks 4-6,
d36/d37 rise — BUT no stratum raw-ranking picks shippers (d30/d31 top locally, public-worst).
Full-grid CV ANTI-TRACKS modern cluster (spearman −0.35). PAIR STANDS: d35 + d33 (hedge proven
decorrelated/ladder-safe vs private tail risk). r227 tabular null (controlled: recovers 3.53 on
weak anchor, 0.00 on champion — leaders' lane buys the gap we own). r230 MOMENT: pretrained ==
random (4 seeds). Pending: r236 MI bound, r232 backward.

**STANDING RESEARCH THREAD (user directive 2026-07-12: "dig more, don't stop, constantly research"):
THE PRIVATE-BOARD QUESTION.** Public = friendliest ~50 of ~200 hidden (26%); private 74% decides
money; r235 proved rankings flip across friendliness strata and champion's edge concentrates in
the friendly 25%. CONTINUOUS OBLIGATIONS: (1) model the private distribution + score compression
(predict everyone's private scores + cutoffs); (2) hunt robustness levers that specifically help
private-like strata (physical-envelope clipping, catastrophic-error caps — pure-downside-protection
class); (3) re-check leader behavior/posts weekly for private-relevant leaks; (4) every ship and
the final pair get judged on PRIVATE-LIKE strata, not public score alone. This thread never closes
until the competition ends.

**2026-07-12 SSL verdict: init matrix COMPLETE 0/4 (scratch best 7.3121; all pretrains hurt).
Learned lane fully closed — every axis controlled. GPU box on standby. Pending: r236 MI bound
(theorem-grade closure or feature map), r237 private compression + envelope.**

**2026-07-12 r237 — CV→LB OFFSET SOLVED + PRIVATE REALITY CHECK.**
Public = friendliest ~26% by PRE-HEEL GR COVERAGE (H2 axis reproduces champ 6.694 from local
7.085; calibrated frac 23%). Private estimates: champ ≈7.24; spreads compress 2× (sub-0.1 public
chasing worthless). HONEST: bronze-zone/no-medal at current strength (private silver ~6.7-7.2,
prize ~5.8-6.2) unless a per-well curvature/matching lever lands (= r236's question). Envelope
clip: zero exits ever, provably safe → wire into all final kernels. Final pair: d35 stays; d30/d31
= private-robustness hedge candidate (leads private-like strata; winner's-curse caveat measured);
dedicated final-pair round near deadline. Pending: r236 MI, r238 PNGs, r239 thread+numbers.

**🏛 2026-07-12 r236 — THEOREM-GRADE CLOSURE (the campaign's definitive measurement).**
Neural MI bound (InfoNCE, family-CV, 20 permutation nulls, positive control z=+9.9 validates
method): champion-residual DC and >800ft band carry ZERO recoverable single-well information
(debiased MI≈0, full power n=772) over the ENTIRE 78-dim feature space — subsumes all hand
probes (r195/211/216/217/218/226). Slope: real sliver z=+4.2 (ρ≈0.12, trajectory-borne,
unstable — not a lever). POWER AUDIT: r195/r218/r216 well-level nulls DECISIVE; but G*-only-DC
(n=275), 65-well cohorts, and FAMILY-LEVEL (57 families, MDE 0.36) are UNDERPOWERED = "not
detected", never "absent". Surviving legal levers: loss-geometry (bimodal hedging r241),
input calibration (r240), friendliness-adaptive runtime (driver39), robustness/envelope.

**2026-07-12 r244 CAMPAIGN AUDIT (3 hostile lenses) — zero decisions overturned, six precision
claims corrected, P1-P6 process gates added to EXPERIMENT_GATE.** Key corrections: G*-reg "exact"
claims DEAD (rank tool only; honest band ±0.03-0.10; sub-0.06 deltas = coin flips, never ship
basis); r236 closure carried by permutation+power NOT the MI bound; "universal" scope-tagged
(family level 57-families MDE 0.36 = formally OPEN; G*-only-DC + 65-well cohorts = PARKED-
UNDERPOWERED). r241 hedging KILLED (detector = noise: 18 real vs 19 shuffled fires; truth sits
AT modes, champ ~65% right; corrected law: hedge helps iff p<3/4, optimum (1−p)d — useless
without a detector). Audit-mandated missing experiment = FAMILY-LEVEL residual lever → r247
launched (leave-entire-family-out, MDE pre-registered). Full audit: notes/campaign_audit_2026-07-12.md.

**2026-07-12 window closures:** r247 family level CLOSED WITH POWER (GBM 0.419 = 1-of-4
multiplicity artifact, killed by Bonferroni + own-model shuffle null + pooled lift 0.000;
P-gates' first live save). r241 bimodal hedge KILLED (phenomenon real — 17ft Eagle Ford couplet
spacing verified — but FB-mean champion already occupies/soft-mixes the correct mode; detector
fails P1 shuffle). r248 dead-list relearn: ledger CONFIRMED; 2 legit re-tests extracted →
r250a (fault-HMM re-gated, long-well stratum) + r250b (length-stratified backward two-filter);
sub-noise kills relabeled per P2. Live: r246 aux-stacking (k256 mechanism), r249 long-well
config, r250a/b, exp242-calibrated (detached). Champion d35 6.694; pair d35+d33; hedge d30/31.

**STANDING DUAL LOOP (user directive 2026-07-12): DC-loop + SLOPE-loop run PERMANENTLY until
solved — cycle = ANALYZE (anatomy) → DEBUG (name the mechanism) → RESEARCH (papers+math per gate)
→ SOLUTION (build per spec) → TEST (full gates: P1-P6, drop-20, honest bands) → REPEAT (next
mechanism / deeper anatomy). Never idle between cycles; each synthesis auto-spawns the next round.
Current cycle 1: r255 (DC anatomy: convention/onset/census) + r257 (slope anatomy: profile-
discreteness/dip-calibration/geometry-sliver) → their syntheses' solution specs launch as cycle-1
solutions → test → cycle 2 from whatever remains. Oracles at stake: DC solved = public ≈4.4;
slope solved ≈5.9; each 10% of DC ≈ −0.22 public. Side loops: r256 hires fix (wander section),
exp254 rotated frame, exp242 calibrated MI. Wakeup prompts MUST carry this dual-loop directive.

**2026-07-12 DUAL-LOOP CYCLE-1 SYNTHESES (r255 DC + r257 slope) — the smooth channels named:**
DC: 93% ACCUMULATED (late step re-locks 42% > ramps 30% > hybrid 21%; immediate 5%); convention
bug REFUTED with power (our 2-D convention strictly dominates the patent form); diving-path +
couplet-alias refuted; no geometry feature separates hi-DC wells (worst-10 = 10 families).
SLOPE: drift profile = 89% ≤3-segment PWL = DISCRETE panel decisions (r218 continuous framing
refuted); knots TWIN-SHARED (180 vs 609ft null, smooth corr 0.89) but own-well blind; dip
recalibration DEAD (sign-heterogeneous); geometry sliver 1.4% closed. HONEST combined own-well
recoverable = 0.00-0.02 pooled (below noise). THE ONE CONVERGENT LEVER = SPEC-S1 twin-anchored
drift transfer (subsumes DC SPEC-R1 on twins) → r259 building at probe/hedge grade. r258 running
DC specs (preconditions first). Oracles remain: DC=inf-class majority (c_well + step magnitudes).

**2026-07-12 wander-loop cycle 2 (user: "we have to SOLVE it — saying it doesn't work won't help").**
r256 hires internal repairs: all 3 FAIL (smoother = wrong frequency band, red-spectrum error; b16 coarser blocks = bias > variance removed; stiffening dominated by align ceiling). CONTROL decisive: optimal hires weight on long (>=6k) wells = ZERO (trio(w=0)=8.42 < align 8.64 < hr8 11.83). r259 SPEC-S1 twin transfer: UNDERPOWERED (16 strict co-located twins < 30 power floor; drift IS twin-decodable per P1, shelved at probe). r258 SPEC-R1 ramp regularizer: precondition FAIL (ramp = real regional structure, 70% neg corr; RAMP level = information-class).
INGREDIENT-RULE FIND: r249's dead-list kill was a MULTI-DOF per-stratum refit with hires HELD FIXED — the 1-DOF length-gated hires weight was never tested. Launched r260 (one-knob length-gated hires taper, LPO-CV + random-gate control + bootstrap; headroom +0.094 long / ~+0.028 pooled; private-robustness grade) and r261 (papers-first research: globally-anchored long-well decode leg — forward-backward/global-DP over cached cost landscapes; must beat align 8.64 on long OOF).

**DC PERSISTENCE DIRECTIVE (user, 2026-07-12, verbatim spirit): "If we solve DC we are DONE. Never say 'it failed' — keep probing and solving."**
DC (56% of SSE) is THE campaign target: solving it alone = pooled 4.69 = leaderboard top. Standing rules:
- A negative result is a MECHANISM DATUM, never a terminal verdict — every negative names the defect and the defect names the next probe (r266 fail → loss-mismatch → r268a; wrong function class → r268b).
- The DC probe pipeline must NEVER be empty: whenever a DC round reports, at least one successor DC round must already be running or launched the same hour.
- "Closed with power" applies to individual DOORS (specific lane × feature set × function class), never to DC itself while any gate-walkable door remains.

**DC-FIRST RULE (user, 2026-07-12, after I pivoted to slope following the r271 kill — "again you are leaving DC, this is what I hate, why are you quitting after one failure"):**
When a DC round fails, the SAME TURN must launch its DC successor BEFORE anything else launches. Other pools (slope/wander) run only IN ADDITION to a live DC arm I control — never as the replacement move after a DC negative. A DC arm running remotely (box) does not count as "the pipeline is full"; I must always have a DC probe of my own in flight until DC is solved or the deadline ends the campaign.

**FULL-DC PRIORITY LAW (user, 2026-07-12, confirmed explicitly: "our only focus should be to solve full DC — without it we can't top"):**
DC is not the top priority; it is the ONLY priority. The arithmetic: solving all non-DC pools perfectly still lands ~4.9-5.0 public (above the leader); DC solved alone = 4.4-4.5 = #1. Operating consequences:
- Every launched round must state, in one line, which DC sub-pool it attacks (late re-locks 42% > ramps 30% > hybrid 21% > first-panel 5%) and its DC-denominated headroom.
- Sub-pool priority order: LATE RE-LOCKS first (42%; solve = WHERE the datum jumps + BY HOW MUCH), then ramps.
- Slope/wander/texture rounds run ONLY when every actionable DC direction is already in flight — never instead of one.
- Tenths-scale DC arms (ensemble corrections) are background insurance, not focus; the focus is mechanism discovery for the re-locks (current: r278 PNG autopsy of the interpreter's workspace at the 136 measured break locations).

**THE GOAL RULE (user, 2026-07-12, verbatim spirit: "your goal is to FIRST solve full DC — as many problems come, as many failures come"):**
THE goal of this campaign is: SOLVE FULL DC. Not "attack DC", not "keep a DC arm alive" — SOLVE it.
- Failures do not change the goal. A failure changes the ATTACK, never the TARGET. However many problems and failures arrive, the response is always: name the mechanism, design the next DC attack, launch it. Repeat until DC is solved or the deadline arrives.
- Success criterion: per-well datum path recovered (late re-locks WHERE+HOW MUCH, then ramps) = pooled ~4.69 = #1.
- Everything else (slope, wander, endgame polish, hedges) is subordinate and opportunistic — done only in the gaps, never displacing a DC move.
- This rule outranks my own assessments of "closed with power": closures kill CHANNELS, never the goal. When all known channels are closed, the required move is INVENTING a new channel (the r269 ensemble decomposition and r278 PNG autopsy both came from exactly that move), not concluding.

**EXISTENCE-PROOF RULE (user, 2026-07-12: "you said the same when we were at 7, and earlier in other competitions — keep banging the door"):**
A leaderboard score above ours IS a mathematical existence proof that a better solution exists in the SAME files. Ceiling/"information isn't there" talk is BANNED as a conclusion while anyone is above us — it may only appear as a branch hypothesis with a probability, never as a verdict. History: at 7.097 I argued the gap was interpreter-private; the flow then found 7.097→6.694 in two days. Nemotron: same pattern before THK 0.85. My "closed with power" statements are about MY channels, not about the problem. The correct posture after the Nth failure is the same as after the 1st: name mechanism → invent next channel → launch. Keep banging the door until the deadline, not until my model of the problem says stop.

**2026-07-12 late-night DC blitz (r279-r293) — the full-channel sweep:**
Killed with measurement: tops-as-labels (degenerate: TVT = ANCC−Z+c_well), ANCC surface standalone (98ft drop-20), patent MAP decoder (argmax un-invertible on flat likelihood; patent IS the labels' structural ancestor, drift corr 0.6-0.9), regional-dip anchor (champ under-dips, DOA), lattice prior (real q~4-6ft fingerprint, non-discriminative — champion basins on same grid), pilkwang scan (uncorrelated with our DC), static affine calibration (already shipping), drift calibration (alignment-error masquerade, neg autocorr), GNC continuation (nesting false, coarse minima relocate), cross-well transfer (datum variogram range <10ft; test wells 300-1000ft from train — drop-20 law now MEASURED).
LIVE POSITIVE found and traced: sigma=2 reference smoothing = basin regularization (+2.8ft datum-localization, all controls clean) — but already deployed in d17 SLAM leg since exp62 (r291 NO-SHIP; the buried exp74 collect had died at 300/773 with analysis never run). r293 (pending): apply it to PF/COMB2 legs that lack it.
Leader model after sweep: 4.859's edge cannot be cross-well (geometry forbids) nor own-well generator inversion (r286) — narrows to friendly-set concentration + public overfit; shakeup branch strengthens. Champion 6.694 frozen; ~35 powered closures documented.

**PURPOSE RULE (user, 2026-07-12, overrides all score incentives): "I want the correct way, done logic. My purpose is to DISCOVER, not win by any means."**
The campaign's purpose is DISCOVERY — methodology that genuinely solves the problem. Consequences:
- NO leakage exploits, NO lookup shortcuts, NO gains that don't come from understanding the geology/decoding problem — even when rules-legal and culturally normal (driver39's duplicate-lookup layer was built, parity-proven, worth +0.2-0.4 public, and RETIRED UNSHIPPED on this principle, 2026-07-12).
- Every shipped point must be earned by method. The score is the measurement of the discovery, not the goal.
- This outranks the leaderboard: if pure method lands lower than exploit-assisted competitors, that is the honest result and we take it.

**ITERATION-DEPTH RULE (user, 2026-07-12: "why so eager to close each experiment — see what in it has NOT worked, how it WILL work, and make it work"):**
Two kinds of kills, never conflate them:
- IDENTITY-LEVEL kill (algebra/geometry/measured-zero: tops≡TVT, gamma_opt<1 identity, zero typewell noise, <10ft variogram) → close in one shot, rightly.
- DESIGN-SPACE kill (one point of a large design space evaluated: r286 patent decoder = one seg-length/dip-grid/exponent/commitment-mode) → NOT a closure. The round's mechanism diagnosis names the FAILING INGREDIENT; the mandatory next step is ITERATE on that ingredient (diagnose → modify → retest, budgeted N cycles) before any "closed" verdict. A lane closes only when its mechanism is exhausted, not when its first parameterization fails.
Every KILL verdict must state which kind it is. Design-space verdicts get an iteration budget (>=3 design cycles) before dead-listing.

**BUILD-VERIFICATION RULE (user, 2026-07-12: "why have we made mistakes in BUILDING all this research — we're just running and running"):**
Every experiment BUILD must pass a KNOWN-ANSWER test before its verdict counts: inject a known signal (synthetic labels from the hypothesis' own forward model, planted datum shifts, planted re-locks) and verify the implementation recovers it. A kill from an unverified build is provisional, not a closure. r306 runs the retroactive audit (patent-decoder self-consistency = the decisive case). Cadence rule: verification and mistake-analysis rounds take priority over new hypothesis launches — depth before breadth.

**AUDIT-AFTER-EVERY-EXPERIMENT RULE (user, 2026-07-12, agent created):**
The `experiment-auditor` subagent (.claude/agents/experiment-auditor.md — Chief Experiment Auditor persona, user-authored) MUST run on every completed experiment round BEFORE its verdict enters the ledger as accepted or any successor launches. Its verdict taxonomy (VALIDATED/IMPLEMENTATION FAILURE/DESIGN FAILURE/.../RESURRECT/STOP) supersedes the raw round verdict. Flow per round: experiment completes -> experiment-auditor audits (inspects code/caches/logs, not the author's summary) -> audited verdict + next-action ledgered -> only then successor launches per its recommendation.

**AUDIT-AUTOPILOT (user, 2026-07-12: "always make it automatic after experiment completes"):**
The experiment-auditor launch is AUTOMATIC, not discretionary: every experiment-round task-notification triggers the auditor as the FIRST action of that turn. Enforced via inject_reminders.sh (injected every turn). Raw round verdicts are PROVISIONAL until audited.

**OBJECT-MISMATCH RULE (r306 build audit, 2026-07-12 — the answer to "why the mistakes"):**
The campaign's builds are sound (known-answer verified: patent decoder, leg-DC, shrink machinery, self-overlap detector post-fix). The RECURRING failure mode is OBJECT MISMATCH: measuring the cheap exp49 datum-localization proxy instead of the champion's real blend decode — the proxy over-states absolute headroom by ~2.8 ft (r289 nearly shipped on it; r291 caught it). RULE: the decision quantity of every round must be the REAL object (champion blend OOF); the proxy is licensed ONLY for relative ablations and oracle-empty nulls, never for absolute ship/kill claims. Also: harness bugs can indict correct builds (r284-B's planted signal was below the detector's own gate) — every planted-signal test must verify the plant is detectable by construction.

**RIVAL-BUILDER AGENT (user asked "which persona can move score", 2026-07-12):**
.claude/agents/rival-builder.md — clean-room competition engineer; mandate = beat champion 7.085 honest pooled CV from raw data, BLIND to campaign ledger until Phase 4 (anti-anchoring by design). Three-basin coverage (geometric/signal/learned) before optimizing any. Pure method (no lookups). Own namespace (rival_*). Success = WIN (>=0.10 beat -> integration spec) or BASIN-CONFIRM (adversarial validation of champion design). Rationale: after ~300 rounds the biggest un-audited risk is lineage anchoring; the 4.859 existence proof says a better basin exists in the same data.

**RIVAL LOOP (user, 2026-07-12, standing):** rival-builder runs a permanent competitor cycle: BUILD mission -> autopilot audit -> error anatomy names next attack -> next mission same turn -> champion checkpoint (vs 7.085 identical honest harness) every cycle. Clean-room wall until its Phase 4 (then cross-pollination: mutual component theft, each through confirm-law + audit). Terminates only on WIN (>=0.10 -> integration), BASIN-CONFIRM, or deadline. The rival never idles — same tirelessness law as the DC loop.

**AUTONOMOUS WINDOW (user, 2026-07-12 ~21:10): "work autonomously the next 20 hours, you have all freedom, let's see where we reach."**
Window: 2026-07-12 21:10 -> 2026-07-13 ~17:00. All standing engines run continuously: DC loop (goal rule), RIVAL loop (phases advance on each audited cycle), AUDIT autopilot (every round), research gate on every launch. Purpose rule holds absolutely (pure method, no exploits). Spend guardrails hold (no pods without need; flag >$20). Champion driver35 frozen unless something passes the FULL ship stack + audit; no submissions without a passing gate (submission itself waits for user unless a validated strict-improvement passes everything — then prepare push-ready and note it). Report a consolidated window summary when the user returns.

**LOAD-BEARING INTEL re-surfaced (audit-rival-p2, 2026-07-12): rank-2 team at 5.44 uses PER-WELL DATA ONLY (no spatial/kriging leg — house ledger line 75).** Implication chain: (a) the per-well GR-sequence paradigm reaches ~5.4 (not our 7.085 ceiling) — ~1.6 ft of per-well headroom EXISTS in information we already have; (b) the leaders' edge is NOT cross-well (consistent with r290 geometry); (c) the champion's remaining gap is in per-well decode quality, not information absence — the strongest existence-proof refinement of the campaign. Rival Mission 2 (calibrated correlator, 5 defect fixes) is the live probe of this exact gap.

**DC MONITOR-MODE (audit-r311 ruling, 2026-07-12 ~23:30):** after ~60 audited rounds, NO gate-walkable DC probe remains in the current instrument+metadata envelope (direction lane actionably closed — signal real at -0.107 oracle but detector-capped at AUC 0.61; ramps info-class; groupings closed; emission dormant; vintage parked). DC loop reacts only to: rival-lane findings (per-well decode quality — the 5.44 existence proof), new external metadata/intel, or audit-ordered re-entries whose named preconditions are met. Do NOT re-run direction probes under new names. The live offense = RIVAL loop Mission 2+ (calibrated correlator).

**RIVAL LOOP TERMINAL: BASIN-CONFIRM (audited, 2026-07-13 ~00:40).** An adversarial clean-room engine re-derived the champion's structure (3/3 facts blind), iterated its own correlator to the same direction wall (double-blind mechanism confirm), could not beat 7.085, and its best component was refused by the confirm law (decorrelated-but-useless, CI<0). FORENSIC CORRECTION (validated against wiring): the champion's datum c_well = per-well HEEL ANCHOR (tvt0 from TVT_input; hmm_refine anchor_sig=0.5) — PRIVATE-ROBUST; ANCC kriging enters only as np.diff (datum cancels structurally; corr 0.145). Only the kriged DIP FIELD drift-shape carries spatial risk: bounded, champion 7.05 -> 8.32-8.69 @R500 (house LRO independently corroborated) -> 10.6 central worst-case. RETIRE the "leaky kriged datum" worry; keep drop-20 conservatism for planning. ENDGAME: final pair d35+d33 + hedges d30/d31 UNCHANGED and STRENGTHENED (datum de-risked; dip-shape risk is exactly what the hedges cover). Leaders' 4.859 edge = outside both explored basins (per-well curvature/direction channel neither engine reads from this data).

**AUDIT-AUTOPILOT PAUSED (user, 2026-07-13 ~08:00): "pause validator for time being."** Do not auto-launch the experiment-auditor on round completions; accept raw round verdicts (internal prereg/controls still required in every round). Auditor available on-demand. Re-enable on user request.

## 2026-07-13 PIVOT (the manager correction + census breakthrough)
- Manager: "20 teams above can't be wrong" → leaders-wrong thesis STRUCK; LB .lb_jul13.txt: 20 teams 4.859-6.28 smooth continuum.
- Shape anatomy (exp330_shape_anatomy.json): low-freq band (ramp + >800ft) = 92% of shape SSE = 40.4% of total; killing it alone → ~5.47. Sub-800ft dead (≤0.12).
- CENSUS (.intel_20260713/, commit de07352): continuum NOT in public kernels (plateau 7.09) — FORMULATION GAP: residual/dtvt targets + cumsum consistency, auxiliary future targets, heel-calibrated GR gain/offset, plain models, grouped 5-fold×5-seed (CV→LB +0.3 constant).
- **hengck23 MECHANISM (threads 699853/697431): truth TVT = piecewise-linear between ~15 manual ANCC annotation control points/well; dtvt = −dz + offset from small discrete set; DIRECTION DISTILLABLE 0.927 OOF** (hypothesis: local between-control-point direction, ~15×772 examples, not the one-per-well global sign we proved dead). His bounds: upper ~3.5, good model ~4.5.
- Reproduction shortlist: cdeotte residual-over-anchor XGB, hengck23 dtvt/CNN-SDF, sunnywu27 pure-PF, k256 auxiliary-target. r330-B1 dense GBDT running; r331 = mechanism reproduction (Codex convergence in flight).
- Bonus: thread 719389 confirms pool-skew (r328); pinned 707695: one outlier well excluded from private scoring.
