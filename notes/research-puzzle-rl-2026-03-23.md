# Research: RL Methods for Puzzle-Type Reasoning Tasks
**Date**: 2026-03-23
**Focus**: RL approaches specifically for our 6 puzzle types (bit_manipulation, cipher, gravity, unit_conversion, numeral_system, equation_transform)

---

## 1. RL for Cryptography / Cipher Puzzles

### Key Paper: "Improving LLM Agents with RL on Cryptographic CTF Challenges" (arXiv:2506.02048)
- **Method**: GRPO (Group Relative Policy Optimization) with tool-augmented Llama-3.1-8B
- **Setup**: Model writes and executes Python inside an isolated REPL (tool-integrated reasoning)
- **Results**: +53% absolute jump in Pass@8 on unseen crypto tasks (0.35 -> 0.88), Majority@8 to 0.41
- **Generalization**: Improvements transferred to picoCTF benchmark (+13pp Pass@8) -- genuine skill acquisition, not overfitting
- **Challenge types**: 8 high-level cryptographic archetypes including substitution ciphers, frequency analysis
- **Key insight**: Gains came from more reliable tool invocation and code synthesis, NOT prompt adaptation
- **Reward**: Binary (solved/not-solved for CTF flag extraction)

**Actionable for us**: Our cipher puzzles involve substitution mapping derivation. Training the model to write Python to decode ciphers (TIR approach) could be more effective than pure CoT reasoning. The model learns to use frequency analysis and systematic mapping through code.

Source: [arXiv:2506.02048](https://arxiv.org/abs/2506.02048)

---

## 2. RL for Bit Manipulation / Binary Operations

### Finding: No direct RL papers on bit manipulation for LLMs
- Searched extensively; no papers specifically address RL for bit operation learning in LLMs
- The closest approach: **Enigmata** includes "Crypto" category tasks that involve pattern recognition similar to bit ops
- **LogicPuzzleRL** includes "Cryptarithm" puzzles requiring symbolic arithmetic -- analogous to bit manipulation

**Actionable for us**: Bit manipulation puzzles are best approached via:
1. **TIR (Tool-Integrated Reasoning)**: Train model to write Python that performs bit operations
2. **RLVR with exact match**: Since answers are deterministic, use binary reward (correct/incorrect)
3. **Pattern recognition training**: Use Enigmata-style generators to create diverse bit operation examples

---

## 3. Enigmata: Synthetic Puzzles with RL (arXiv:2505.19914)

### THIS IS THE MOST RELEVANT PAPER FOR OUR COMPETITION

**Overview**: First comprehensive suite tailored for improving LLMs with puzzle reasoning skills via RLVR.

**7 Categories, 36 Tasks**:
1. **Cryptographic Puzzles** -- pattern recognition, decoding encrypted messages (KPA, KKA)
2. **Arithmetic Puzzles** -- numerical reasoning under constraints (Game24, Countdown)
3. **Logic Puzzles** -- deductive reasoning (Knights and Knaves, Zebra Logic)
4. **Grid Puzzles** -- spatial reasoning on grids
5. **Graph Puzzles** -- graph-theoretic reasoning
6. **Search Puzzles** -- exploration and search strategies
7. **Sequential Puzzles** -- temporal/sequential pattern recognition

**Mapping to our puzzle types**:
| Our Type | Enigmata Equivalent | Category |
|----------|-------------------|----------|
| cipher | Crypto KPA / Crypto KKA | Cryptographic |
| bit_manipulation | Pattern recognition tasks | Cryptographic/Sequential |
| equation_transform | Symbolic arithmetic | Arithmetic |
| numeral_system | Number system conversion | Arithmetic |
| unit_conversion | Numerical reasoning | Arithmetic |
| gravity | Physics-based (spatial) | Grid/Sequential |

**Training Method**: Two-stage approach:
1. **Stage 1: Rejection Fine-Tuning (RFT)** -- establish foundational reasoning patterns
2. **Stage 2: Multi-task RL with VC-PPO** -- develop general reasoning skills that transfer

**Key Results**:
- Qwen2.5-32B-Enigmata surpasses o3-mini-high and o1 on puzzle benchmarks
- Also improves ARC-AGI (32.8%) and ARC-AGI 2 (0.6%)
- Generalizes to out-of-domain puzzles AND mathematical reasoning
- Crypto and Arithmetic tasks yield highest accuracy; spatial/sequential remain hardest

**Generator-Verifier Design**:
- Each task has a **generator** producing unlimited examples with controllable difficulty
- Each task has a **rule-based verifier** for automatic evaluation (binary reward)
- Supports scalable, multi-task RL training

**Critical Insight**: The two-stage approach (RFT first, then RL) works MUCH better than RL alone. SFT/RFT creates the foundation; RL refines and generalizes.

Sources:
- [arXiv:2505.19914](https://arxiv.org/abs/2505.19914)
- [Enigmata Website](https://seed-enigmata.github.io/)
- [GitHub: BytedTsinghua-SIA/Enigmata](https://github.com/BytedTsinghua-SIA/Enigmata)
- [MarkTechPost Analysis](https://www.marktechpost.com/2025/06/01/enigmatas-multi-stage-and-mix-training-reinforcement-learning-recipe-drives-breakthrough-performance-in-llm-puzzle-reasoning/)

---

## 4. Reasoning Gym: Procedural Puzzles for RLVR

**Overview**: 100+ procedurally generated reasoning environments specifically designed for RLVR training.

**Domains**: algebra, arithmetic, computation, cognition, geometry, graph theory, logic, common games

**Key Features**:
- Python library: `pip install reasoning-gym`
- Unlimited training data with adjustable complexity
- Binary-scored verifiers (correct/incorrect) for most tasks
- Framework-agnostic -- works with any RL training framework
- Integrates with `verifiers` library for easy RLVR training
- NeurIPS 2025 Spotlight paper

**Usage**:
```python
import reasoning_gym
data = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
# Use data.score_answer(answer=x['answer'], entry=x) for verification
```

**Relevance**: We can use Reasoning Gym tasks to augment our training data with diverse procedural puzzles, improving generalization beyond the 6 competition types.

Sources:
- [GitHub: open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym)
- [arXiv:2505.24760](https://arxiv.org/abs/2505.24760)

---

## 5. NeMo Gym: NVIDIA's Own RL Training Environments

**Overview**: NVIDIA's library for building RL training environments for LLMs. 60+ environments.

**Puzzle-Type Environments**:
- **Arc AGI**: Puzzles designed to test intelligence
- **Reasoning Gym integration**: Logic, graph theory, computation tasks
- **Circle Click**: Image-based puzzle environment

**Integration**: Works with NeMo RL, OpenRLHF, and Unsloth

**Key Insight**: NeMo Gym was used in post-training of Nemotron-3-Nano itself! The competition model was trained with GRPO across math, code, science, instruction following, multi-step tool use, multi-turn conversations, and structured output environments.

**Actionable**: Since our competition model (Nemotron-3-Nano) was already trained with NeMo Gym + GRPO, further GRPO on puzzle-specific data should be a natural extension of its existing training.

Sources:
- [GitHub: NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym Docs](https://docs.nvidia.com/nemo/gym/latest/index.html)

---

## 6. Curriculum Learning: Easy-to-Hard for Puzzles

### Paper 1: E2H Reasoner (arXiv:2506.06632)
- **Method**: Schedule tasks from easy to hard during RL training
- **Key finding**: Easy tasks are important initially, but FADING THEM OUT is essential to prevent overfitting
- **Best for small models (1.5B-3B)** -- exactly our model size (3B active params!)
- **Theoretical backing**: Curriculum stages require fewer total samples than direct learning
- **Result**: Small LLMs that struggle with vanilla RL alone see significant improvements

### Paper 2: Self-Evolving Curriculum / SEC (arXiv:2505.14970)
- **Method**: Treat curriculum selection as Multi-Armed Bandit problem
- **Each problem category = one "arm"** in the bandit
- **Uses advantage from policy gradient as proxy for learning gain**
- **Updated with TD(0)**
- **Results**: 13% on Countdown, 21% on Zebra puzzles, 22% on ARC-1D, 33% on AIME24 (vs random curriculum)
- **Training setup**: 10,000 problems per difficulty level, 200 held-out for eval

### Paper 3: Goldilocks RL (Apple Research, ICLR 2025)
- **Method**: Teacher model selects questions of appropriate difficulty for student (Goldilocks principle)
- **Neither too easy nor too hard** -- questions at the frontier of the student's ability
- **Teacher continuously adapts** to student's evolving abilities during training
- **Setup**: 8 GPUs total -- 2 for teacher (data selection), 6 for student (training)
- **Works with GRPO** as the student training algorithm

### Paper 4: GRPO-LEAD (EMNLP 2025, arXiv:2504.09696)
- **Three innovations**:
  1. Length-dependent accuracy reward (foster concise solutions)
  2. Explicit penalty for low precision caused by length reward
  3. Difficulty-aware advantage reweighting (amplify learning signals for hard problems)
- **Difficulty weight**: determined from empirical correctness of responses per question

**Actionable for us**:
- Our puzzles have natural difficulty levels (# examples, complexity of operations)
- **Start with easy puzzles (simple ciphers, basic bit ops), then increase difficulty**
- **SEC approach is most practical**: treat each of our 6 puzzle types as an arm in a bandit, use advantage as proxy for learning gain
- **For 3B models, curriculum is critical** -- vanilla RL alone may not work
- **GRPO-LEAD's difficulty reweighting is easy to implement**: weight harder puzzles more in the advantage calculation

Sources:
- [E2H Reasoner - arXiv:2506.06632](https://arxiv.org/abs/2506.06632)
- [SEC - arXiv:2505.14970](https://arxiv.org/abs/2505.14970)
- [Goldilocks RL - Apple Research](https://machinelearning.apple.com/research/goldilocks)
- [GRPO-LEAD - arXiv:2504.09696](https://arxiv.org/abs/2504.09696)
- [SEC GitHub: ServiceNow/sec](https://github.com/ServiceNow/sec)

---

## 7. Verifiable Rewards for Deterministic Puzzle Answers

### RLVR is the perfect fit for our competition

**Why**: All 6 puzzle types have deterministic correct answers that can be verified with exact string matching or numerical tolerance. This is exactly what RLVR was designed for.

**Reward Function Design for Our Puzzles**:
```python
def puzzle_reward(prediction, ground_truth):
    # Binary reward -- exact match (case-insensitive) or numerical tolerance
    pred = prediction.strip().lower()
    truth = ground_truth.strip().lower()
    if pred == truth:
        return 1.0
    try:
        if abs(float(pred) - float(truth)) < 1e-2:
            return 1.0
    except:
        pass
    return 0.0
```

**Key Research Findings**:
- RLVR avoids reward hacking, overfitting, and alignment drift present in RLHF
- Binary rewards work well for structured domains with clear answers (our case!)
- For puzzles with deterministic answers, binary is SUFFICIENT -- no need for soft rewards
- GRPO + binary verifiable reward is the DeepSeek-R1 recipe that scaled reasoning
- Once the model has good pre-training priors, GRPO gradient increases probability of correct CoTs

**The critical equation**: Pre-training establishes logic priors -> SFT establishes format/approach -> RLVR amplifies correct reasoning paths

Sources:
- [RLVR Explained - Promptfoo](https://www.promptfoo.dev/blog/rlvr-explained/)
- [arXiv:2506.14245](https://arxiv.org/abs/2506.14245)
- [arXiv:2503.06639 - GRPO Dynamics](https://arxiv.org/abs/2503.06639)
- [awesome-RLVR GitHub](https://github.com/opendilab/awesome-RLVR)

---

## 8. Process Reward vs Outcome Reward for Puzzles

### For our puzzles: USE OUTCOME REWARD (ORM), not PRM

**Reasoning**:
- PRMs are better for multi-step math where intermediate steps matter
- Our puzzles have short, deterministic answers -- ORM is sufficient
- PRMs suffer from reward hacking and are harder to train
- Binary outcome reward is simpler, more robust, and works with GRPO directly

**When PRM might help**:
- If the model consistently gets the right approach but wrong final answer
- If we wanted to reward partial progress (e.g., correctly identifying cipher type but wrong decryption)
- For very complex multi-step puzzles where guidance on intermediate steps helps

**Hybrid approach (PROF)**: Uses consistency-driven sample selection -- keeps correct responses with higher process values and incorrect responses with lower process values. This is more practical than pure PRM.

**Practical recommendation**: Start with ORM (binary reward). Only consider PRM if the model plateaus and we identify that it's failing on specific reasoning steps.

Sources:
- [Process Reward Models - Stephen Diehl](https://www.stephendiehl.com/posts/process_reward/)
- [PROF - arXiv:2509.03403](https://arxiv.org/abs/2509.03403)
- [Linking Process to Outcome - arXiv:2509.26578](https://arxiv.org/abs/2509.26578)
- [Rewarding Progress - OpenReview](https://openreview.net/forum?id=A6Y7AqlzLW)

---

## 9. Self-Play for Puzzle Generation

### Key Approaches

**SeRL (Self-play RL with Limited Data, arXiv:2505.20347)**:
- Two modules: self-instruction (generates new puzzles) + self-rewarding (majority voting for answers)
- Generates additional instructions based on available data at each training step
- No external annotations needed

**SQLM (Self-Questioning Language Models, arXiv:2508.03682)**:
- Asymmetric self-play: Proposer generates puzzles, Solver attempts them
- Proposer rewarded if problem is neither too easy nor too difficult
- Solver rewarded based on majority voting
- Fully self-supervised -- no curated training data needed

**GASP (Guided Asymmetric Self-Play, arXiv:2603.15957)**:
- Teacher objective guides problem generation
- Generates problems that are challenging yet solvable

**Challenges**:
- Self-play can collapse after a few iterations (proposer drifts to trivial/unsolvable tasks)
- Solutions: ground generation in external corpora, use frozen teacher to verify

**Actionable for us**:
- We have puzzle GENERATORS for all 6 types (they're procedural)
- Self-play is less relevant since we can generate unlimited training data with known answers
- However, **self-play for difficulty calibration** is useful: let the model try puzzles, identify which difficulty levels it fails on, then generate more at that difficulty
- This is essentially the Goldilocks principle applied to our generators

Sources:
- [SeRL - arXiv:2505.20347](https://arxiv.org/abs/2505.20347)
- [SQLM - arXiv:2508.03682](https://arxiv.org/abs/2508.03682)
- [GASP - arXiv:2603.15957](https://arxiv.org/abs/2603.15957)

---

## 10. Tool-Integrated Reasoning (TIR) for Puzzles

### NemoSkills: NVIDIA's TIR Framework
- The AIMO-2 winning solution used TIR (model writes Python, executes it, uses results)
- OpenMath-Nemotron-32B with TIR: 78.4% pass@1 on AIME24, 93.3% majority voting
- Training data distilled from DeepSeek-R1 and QwQ-32B
- Code available at [GitHub: NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills)

### Pattern-Aware TIR (arXiv:2509.23292)
- Two patterns: **calculator** (code for direct computation) vs **algorithmic** (encode problem as program)
- Misaligned pattern choice causes failures even when reasoning is sound
- Two-stage framework: build code competence, then align pattern selection
- Results: Code@1 on MATH500: 64.0% -> 70.5%, AIME24: 26.7% -> 50.0%

### Relevance to Our Puzzles
| Puzzle Type | TIR Approach | Pattern |
|-------------|-------------|---------|
| cipher | Write Python to apply substitution mapping | Algorithmic |
| bit_manipulation | Write Python to apply bit ops | Algorithmic |
| gravity | Write Python to compute d=0.5*g*t^2 | Calculator |
| unit_conversion | Write Python to apply conversion factor | Calculator |
| numeral_system | Write Python for base conversion | Algorithmic |
| equation_transform | Write Python to apply symbol mapping | Algorithmic |

### Key Challenge for Our Competition
- The eval uses vLLM with `max_tokens=7680` -- no code execution available at inference
- The model's output is pure text, not executed code
- TIR training helps the model learn SYSTEMATIC REASONING even if code isn't executed
- The model learns to think algorithmically, which improves CoT quality even without execution

Sources:
- [NVIDIA NemoSkills Blog](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
- [NeMo Skills GitHub](https://github.com/NVIDIA-NeMo/Skills)
- [Pattern-Aware TIR - arXiv:2509.23292](https://arxiv.org/abs/2509.23292)

---

## 11. BONUS: LogicPuzzleRL and ToTRL

### LogicPuzzleRL (arXiv:2506.04821) -- "Play to Learn"
- 7 custom logic puzzles, each targeting distinct reasoning skills:
  - Constraint propagation, spatial consistency, symbolic deduction
  - Includes **Cryptarithm** (symbolic arithmetic -- like our equation_transform)
  - Includes **Sudoku** (constraint satisfaction)
  - Includes **Zebra** (logic grid -- like our cipher mapping)
  - Includes **Knights and Knaves** (deductive reasoning)
  - Includes **Magic Square** (numerical constraints)
- **Binary reward** based on puzzle correctness
- **Key finding**: Individual training is better for Cryptarithm, Magic Square, KK (domain-specific heuristics); combined training is better for Sudoku, Zebra, Graph (abstract reasoning generalizes)
- Significantly improves out-of-distribution math performance, especially mid-difficulty problems

### ToTRL (arXiv:2505.12717) -- Tree-of-Thoughts via Puzzles
- Trains LLMs to develop parallel Tree-of-Thought (ToT) reasoning through puzzle solving
- **Two-stage**: first train ToT in non-thinking mode, then enable reasoning mode
- **Trained on only 1440 puzzle games** (6x6 Sudoku + Alphametic puzzles)
- **Generalizes** to 5x5 Crossword, 9x9 Sudoku, K&K, Poker 24, Make 24, AIME 2024-2025
- Achieves significant improvement in both performance AND reasoning efficiency
- ToT is more systematic than trial-and-error CoT -- identifies, assesses, prunes unproductive paths

Sources:
- [LogicPuzzleRL - arXiv:2506.04821](https://arxiv.org/abs/2506.04821)
- [ToTRL - arXiv:2505.12717](https://arxiv.org/abs/2505.12717)

---

## Summary: Recommended RL Strategy for Our Competition

### Priority 1: RLVR with GRPO (most practical, best evidence)
- **Algorithm**: GRPO with binary verifiable reward (exact match / numerical tolerance)
- **Why**: Nemotron-3-Nano was already GRPO-trained; extending with puzzle data is natural
- **Reward**: Binary (1.0 for correct, 0.0 for wrong) -- matches competition eval
- **Implementation**: Use Unsloth or TRL's GRPO implementation

### Priority 2: Two-Stage Training (Enigmata recipe)
- **Stage 1**: SFT/RFT on high-quality CoT puzzle solutions (we're doing this)
- **Stage 2**: GRPO with puzzle generators providing unlimited training data
- **Critical**: Stage 1 creates the foundation; Stage 2 refines and generalizes

### Priority 3: Curriculum Learning (critical for 3B models)
- **Method**: Start with easy puzzles, progressively increase difficulty
- **SEC approach**: Treat each puzzle type as a bandit arm, allocate training time based on learning gain
- **GRPO-LEAD**: Weight harder puzzles more in advantage calculation
- **E2H finding**: Fade out easy tasks to prevent overfitting

### Priority 4: Diverse Puzzle Data (prevent overfitting to 6 types)
- **Reasoning Gym**: Use for diverse procedural puzzle generation
- **Enigmata**: Use crypto, arithmetic categories for additional training data
- **Hidden test set may include broader tasks** -- generalization matters

### Priority 5: TIR-Style CoT (even without code execution at inference)
- Train model to write systematic, algorithmic reasoning in CoT
- For cipher: show step-by-step mapping derivation
- For bit ops: show binary operation identification process
- For gravity: show g calculation then formula application
- The model won't execute code at inference, but algorithmic thinking improves accuracy

### What NOT to Do
- Don't use PRM (process reward) -- ORM is sufficient for our deterministic puzzles
- Don't use PPO for small models -- GRPO is more stable and efficient
- Don't train on only our 6 puzzle types -- hidden test set may be broader
- Don't skip SFT and go straight to RL -- two-stage is critical per Enigmata results
- Don't use uniform random curriculum -- difficulty-aware sampling is strictly better

---

## Specific Changes to Try (Next Training Runs)

1. **Generate more diverse training data**: Use Reasoning Gym to create cipher-like, arithmetic, and logic puzzles beyond our 6 types
2. **Implement GRPO after SFT**: Use TRL's GRPOTrainer with our 6 puzzle generators as the environment
3. **Add curriculum**: Sort training examples by difficulty, train easy->hard with fade-out
4. **Improve CoT quality**: Make CoT traces more algorithmic (step-by-step mapping, explicit calculations) rather than narrative
5. **GRPO-LEAD style advantage reweighting**: Weight harder puzzles more during RL training
6. **Multi-task RL**: Train on all 6 puzzle types simultaneously with SEC-style bandit selection
