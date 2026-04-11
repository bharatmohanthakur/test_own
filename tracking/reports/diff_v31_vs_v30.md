# Diff: v31 vs v30
Samples: 60

## Per-type accuracy delta

| type     |    v31 |    v30 | delta  |
|----------|--------|--------|--------|
| binary   | 1/10   | 3/10   | +0.20  |
| cipher   | 9/10   | 0/10   | -0.90  |
| equation | 0/10   | 0/10   | +0.00  |
| gravity  | 10/10   | 10/10   | +0.00  |
| roman    | 10/10   | 10/10   | +0.00  |
| unit     | 10/10   | 10/10   | +0.00  |
| TOTAL    | 40/60   | 33/60   | -0.117 |

## Case breakdown
- both_right:  31
- both_wrong:  18
- regressions: 9  (v31→right, v30→wrong)
- improvements: 2  (v31→wrong, v30→right)

### Regressions by type & failure mode
- cipher: {'template_with_lying_ver': 2, 'hallucinated_vocab': 5, 'template_with_wrong_table': 1, 'no_boxed': 1}

### Improvements by type & prior failure mode
- binary: {'wrong_2gate': 1, 'lazy_constant_0': 1}

### Both wrong by type (hardest cases → next training data candidates)
- equation: 10
- binary: 7
- cipher: 1

## Sample Regression (top 3)

### Regression #1: cipher
**Expected:** `teacher chases the colorful key`
**v31 box:** `teacher chases the colorful key`  → ✓ (correct)
**v30 box:** `teacher potion the potions for`  → ✗ (template_with_lying_ver)
**Prompt:**
```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
xquxvl buldxlc eldu ndvdbl -> turtle creates near palace
xldbslu oudfc mduole -> teacher draws garden
xldbslu cxqoklc orru -> teacher studies door
xsl skoole udppkx grvvrfc -> the hidden rabbit follows
xldbslu ktdmkelc dprzl cbsrrv -> teacher imagines above school
Now, decrypt the following text: xldbslu bsdcl
```

### Regression #2: cipher
**Expected:** `turtle dreams story`
**v31 box:** `turtle dreams story`  → ✓ (correct)
**v30 box:** `turtle dreams tower`  → ✗ (hallucinated_vocab)
**Prompt:**
```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
amt mfyyts dfwy cwtxatv -> the hidden bird creates
atxcmtw vttv dtpnsy rfdwxwp -> teacher sees beyond library
gfsj hwfatv lsytw anhtw -> king writes under tower
Now, decrypt the following text: alwart ywtxov vanwp
```

### Regression #3: cipher
**Expected:** `the silver dragon sees`
**v31 box:** `the silver dragon sees`  → ✓ (correct)
**v30 box:** `the silver knight sees`  → ✗ (template_with_lying_ver)
**Prompt:**
```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
eoccst pqoupnsg pngpis uotisn -> hatter imagines inside garden
qfzgs mfzni ces afjftmzj bffv -> mouse found the colorful book
ces gpjdst aoc mfjjfrg -> the silver cat follows
rpxoti aeogsg ces qhgcstpfzg wzxxjs -> wizard chases the mysterious puzzle
cztcjs aeogsg pn aogcjs -> turtle chases in castle
Now, decry
```

## Sample Improvement (top 3)

### Improvement #1: binary
**Expected:** `10010010`
**v31 box:** `11100011`  → ✗ (wrong_2gate)
**v30 box:** `10010010`  → ✓ (correct)
**Prompt:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
11001000 -> 10000011
11101101 -> 11010011
01001101 -> 11010001
11010110 -> 01100011
11010101 -> 01010011
01011000 -> 10000001
01001010 -> 10
```

### Improvement #2: binary
**Expected:** `00000001`
**v31 box:** `00000000`  → ✗ (lazy_constant_0)
**v30 box:** `00000001`  → ✓ (correct)
**Prompt:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
11010101 -> 01000001
11000010 -> 10000001
11010001 -> 01000001
00101011 -> 11000000
00010000 -> 00000000
00011010 -> 10000000
11101010 -> 10
```

## Sample Both-wrong (top 3)

### Both-wrong #1: binary
**Expected:** `10010001`
**v31 box:** `11000101`  → ✗ (wrong_2gate)
**v30 box:** `11101000`  → ✗ (wrong_2gate)
**Prompt:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
11010011 -> 10011101
11010100 -> 10100101
10001000 -> 01000110
10000010 -> 00010110
01110110 -> 10110010
10011001 -> 11001110
00110000 -> 10
```

### Both-wrong #2: binary
**Expected:** `01111111`
**v31 box:** `11111011`  → ✗ (complex_gates)
**v30 box:** `01011101`  → ✗ (complex_gates)
**Prompt:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
00101100 -> 01011011
11110101 -> 11011111
01111101 -> 10110111
11000010 -> 10011110
00000110 -> 11100100
11101110 -> 00111110
10011100 -> 11
```

### Both-wrong #3: binary
**Expected:** `00011100`
**v31 box:** `11111110`  → ✗ (wrong_2gate)
**v30 box:** `10111101`  → ✗ (wrong_2gate)
**Prompt:**
```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
10000111 -> 00111100
00110000 -> 10000001
11001101 -> 01101110
01011011 -> 11011010
11110111 -> 10111111
00100001 -> 00001001
00001110 -> 01
```
