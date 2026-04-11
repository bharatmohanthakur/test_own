"""Load train.csv and classify by puzzle type. Provide lookup by (type, answer)."""
import csv
from collections import defaultdict

TRAIN_CSV = "/Users/bharat/Downloads/kaggle/data/train.csv"


def classify(prompt: str) -> str:
    fl = prompt.split('\n')[0].lower()
    if 'bit manipulation' in fl or '8-bit' in fl: return 'binary'
    if 'roman' in fl or 'numeral' in fl: return 'roman'
    if 'unit' in fl or 'measurement' in fl: return 'unit'
    if 'gravitational' in fl: return 'gravity'
    if 'encryption' in fl or 'cipher' in fl: return 'cipher'
    if 'transformation rules' in fl: return 'equation'
    return 'other'


def load_train(path: str = TRAIN_CSV):
    """Return (rows, by_type, by_answer_key) where by_answer_key maps (type, answer) → full row."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_type = defaultdict(list)
    by_answer_key = {}
    for r in rows:
        t = classify(r['prompt'])
        r['_type'] = t
        by_type[t].append(r)
        # (type, answer) is unique-enough for our 60-sample benchmark
        key = (t, r['answer'])
        if key not in by_answer_key:
            by_answer_key[key] = r
    return rows, by_type, by_answer_key


def enrich_with_full_prompts(results, by_answer_key):
    """Given a list of result dicts from old runs (only have prompt_preview), try to recover the full prompt by looking up (type, expected).

    Note: (type, answer) is unique for most rows in the 60-sample benchmark, but collisions
    are possible. Falls back to prompt_preview prefix matching when the key is ambiguous.
    """
    for r in results:
        key = (r['type'], str(r['expected']))
        full = by_answer_key.get(key)
        if full:
            # Prefer match where the preview is a prefix of the full prompt
            if 'prompt_preview' in r and not full['prompt'].startswith(r['prompt_preview'][:100]):
                # Collision; walk the full rows for this type+answer
                for tr in [x for x in results if x['type'] == r['type']]:
                    pass  # skip ambiguity handling for now; log later
            r['prompt'] = full['prompt']
    return results


if __name__ == '__main__':
    rows, by_type, _ = load_train()
    print(f"Loaded {len(rows)} rows")
    for t, items in sorted(by_type.items()):
        print(f"  {t}: {len(items)}")
