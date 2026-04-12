# Efficiency Analysis — What Works, What Doesn't, GRPO Reward Framework

## Score Progression & What Drove Each Jump

| Version | Score | Delta | Key Change | WHY it helped |
|---------|-------|-------|------------|---------------|
| v34c | 0.67 | — | 469 ex, 4 targets, alpha=64, LR=2e-4 | Baseline: v22 CoT + 50 binary verifier |
| v37 | 0.69 | +0.02 | Kh0a config + 2450 subsampled data | **Config change**: alpha=16 (less LoRA override → base model reasoning preserved), 11 targets (embed_tokens+lm_head → better token representation), LR=1e-4 (gentler training). **Data scale**: 2450 vs 469 → more equation/cipher coverage |
| v38a | **0.72** | +0.03 | THK binary traces (1131) replacing Kh0a binary (900) | **Data quality**: our algorithmic traces show ACTUAL per-bit derivation (Identity, AND-NOT, sequence matching) vs Kh0a's teacher-distilled traces that are less structured. Model learns the PROCESS not just the pattern |

### Key Insight: What Drove 0.67→0.72
1. **Config** (+0.02): alpha=16 + 11 targets + LR=1e-4. The old alpha=64 was too aggressive — overwrote base model's reasoning. Alpha=16 is a gentle nudge.
2. **Binary trace quality** (+0.03): Algorithmic per-bit derivation > teacher distillation. The model learns to CHECK each bit independently rather than pattern-match from examples.

---

## Per-Type Breakdown (estimated from Kh0a validation + our tracking)

| Type | v34c (0.67) | v38a (0.72 est) | Ceiling | Gap to close |
|------|-------------|-----------------|---------|-------------|
| gravity | 100% | 100% | 100% | 0 |
| roman | 100% | 100% | 100% | 0 |
| unit | 100% | 100% | 100% | 0 |
| cipher | 80% (16/20) | ~92% | ~95% | ~3% |
| binary | 25% (5/20) | ~35-40% | 85% (THK) | **45-50%** ← BIGGEST |
| equation (digit) | 0% (0/20) | ~50-60% | ~70% | ~10-20% |
| equation (cipher) | 0% | 0% | unknown | **unknown** |

### What's STILL NOT Working

**1. Binary (35-40% → need 85%)**
- Our THK solver covers 70.6% of train rows with expanded 2-input gates + sequence matching
- Missing: 3-input gates (MAJ, CHO, PAR3) and complex multi-level compositions
- ~30% of binary puzzles use 3+ input gates that we can't algorithmically solve
- GRPO target: teach model to EXPLORE more gate combos during inference

**2. Symbol_transform / Cipher-digit equation (0%)**
- Nobody public solves this (Kh0a 0/82, m4nocha 0/82)
- Our Z3 solver tried 1120+ combos, solved 0/30 real puzzles
- Donald's "47 combo" claim appears wrong for cipher-digit
- The rule family is structurally different from what we've enumerated
- GRPO can't help here — model can't get reward signal if 0% baseline

**3. Max completion length (512 tokens in GRPO)**
- Most completions hit 512 token cap (`clipped_ratio: 1.0`)
- The model needs longer traces for complex puzzles (THK uses 5000+ tokens)
- GRPO reward is 0 if trace gets cut off before `\boxed{}`
- Fix: increase max_completion_length to 2048+ (needs VRAM management)

**4. Base model GRPO (rewards mostly 0)**
- Current GRPO runs from BASE model → barely any correct answers → no reward signal
- GRPO from SFT base (v38a at 0.72) would give ~40% baseline correctness → real gradient signal
- This is THE most important fix for next GRPO run

---

## GRPO Reward Creation Framework

### Current Rewards (too simple)
```python
correctness_reward: 2.0 if exact match, 0.0 if wrong
format_reward: 0.5 if \boxed{} present, 0.0 if not
```

### Proposed Multi-Level Reward System

#### Level 1: Format Compliance (0-1 points)
```python
def format_reward(completion):
    score = 0.0
    if '<think>' in completion: score += 0.2        # Uses thinking mode
    if '</think>' in completion: score += 0.1        # Closes thinking tag
    if '\\boxed{' in completion: score += 0.5        # Has answer box
    if completion.count('\\boxed{') == 1: score += 0.2  # Exactly one box (not scattered)
    return min(score, 1.0)
```

#### Level 2: Per-Type Process Rewards (0-2 points)
```python
def process_reward(completion, puzzle_type):
    if puzzle_type == 'binary':
        score = 0.0
        if 'bit[' in completion or 'bit ' in completion: score += 0.5  # Per-bit analysis
        if any(op in completion for op in ['AND', 'OR', 'XOR', 'NOT', 'IDENTITY']): score += 0.5  # Gate identification
        if 'verify' in completion.lower() or '✓' in completion: score += 0.5  # Verification step
        if re.search(r'[01]{8}', completion): score += 0.5  # Produces 8-bit output
        return min(score, 2.0)
    
    elif puzzle_type == 'cipher':
        score = 0.0
        if '->' in completion or 'mapping' in completion.lower(): score += 0.5  # Builds mapping
        if re.search(r'[a-z] -> [a-z]', completion): score += 0.5  # Letter-level mapping
        if 'decrypt' in completion.lower(): score += 0.5  # Decryption step
        return min(score, 2.0)
    
    elif puzzle_type == 'gravity':
        score = 0.0
        if 'g =' in completion or 'gravitational' in completion.lower(): score += 0.5
        if re.search(r'\d+\.\d{2}', completion): score += 0.5  # X.XX format
        if '0.5' in completion and 't^2' in completion: score += 1.0  # Uses d=0.5gt²
        return min(score, 2.0)
    
    elif puzzle_type == 'equation':
        score = 0.0
        if any(op in completion for op in ['add', 'multiply', 'subtract', '+', '*', '-']): score += 0.5
        if 'symbol' in completion.lower() or 'digit' in completion.lower(): score += 0.5
        if 'verify' in completion.lower(): score += 0.5
        return min(score, 2.0)
    
    return 0.0  # Unknown type
```

#### Level 3: Correctness (0-5 points, tiered)
```python
def correctness_reward(predicted, expected, puzzle_type):
    if exact_match(predicted, expected):
        return 5.0  # Champagne bonus (Donald's superlinear reward)
    
    if puzzle_type == 'binary' and len(predicted) == 8 and len(expected) == 8:
        # Partial credit: per-bit match
        bits_correct = sum(p == e for p, e in zip(predicted, expected))
        return bits_correct * 0.5  # 0-4 points for partial
    
    if puzzle_type in ('gravity', 'unit'):
        try:
            # Numeric proximity reward
            diff = abs(float(predicted) - float(expected))
            if diff < 0.1: return 3.0
            if diff < 1.0: return 1.0
        except: pass
    
    return 0.0  # Wrong
```

#### Level 4: Anti-Hallucination Penalties (-2 to 0)
```python
def anti_hallucination_penalty(completion, puzzle_type):
    penalty = 0.0
    
    # Penalize "I identify the pattern" without showing work
    if 'i identify' in completion.lower() and 'bit[' not in completion:
        penalty -= 1.0
    
    # Penalize thrash markers
    thrash = ['hmm', 'let me try', 'actually', 'wait', 'no that']
    if sum(1 for t in thrash if t in completion.lower()) > 3:
        penalty -= 0.5
    
    # Penalize cross-type contamination (cipher template on binary)
    if puzzle_type == 'binary' and any(m in completion for m in ['LEN:', 'TABLE:', 'VOCAB']):
        penalty -= 1.5
    
    # Penalize copy-input (answer identical to input)
    if puzzle_type == 'equation':
        # Check if answer is just the target string repeated
        pass
    
    return max(penalty, -2.0)
```

#### Combined Reward
```python
def total_reward(completion, puzzle_type, predicted, expected):
    r = 0.0
    r += format_reward(completion)          # 0-1
    r += process_reward(completion, type)    # 0-2
    r += correctness_reward(pred, exp, type) # 0-5
    r += anti_hallucination_penalty(comp, type)  # -2 to 0
    return r  # Range: -2 to 8
```

### GRPO Training Data Selection (Goldilocks Zone)
- Run v38a adapter on 500 diverse prompts, 4 completions each
- Keep prompts where model gets 1-3 out of 4 correct (25-75%)
- These are the "learnable" problems — too easy = no signal, too hard = no signal
- Expected: ~200 Goldilocks prompts from 500

### Next GRPO Run Plan (v41)
1. Base: v38a adapter (0.72 SFT)
2. Data: 200 Goldilocks prompts (selected by running v38a inference)
3. Rewards: multi-level system above
4. Config: 100-200 steps × 27s = 45-90 min on RunPod
5. Expected delta: +0.02-0.05 → 0.74-0.77
