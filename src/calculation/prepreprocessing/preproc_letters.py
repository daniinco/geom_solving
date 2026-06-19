import re
import csv
import argparse
from pathlib import Path

SUBSCRIPT_DIGITS = '₀₁₂₃₄₅₆₇₈₉'
SUBSCRIPT_MAP = {c: str(i) for i, c in enumerate(SUBSCRIPT_DIGITS)}

def normalize(m):
    s = m.group(0)
    for sub, dig in SUBSCRIPT_MAP.items():
        s = s.replace(sub, dig)
    s = re.sub(r'[_{}\s]', '', s)
    return s

def reletter(question):
    pattern = re.compile(
        r'[A-Z]_\{\d+\}|[A-Z]_\d|[A-Z]\d|[A-Z][' + SUBSCRIPT_DIGITS + r']'
    )
    used = set(re.findall(r'[A-Z]', question))
    free = [c for c in 'ABCDEFGHIJKLMNPQRSTUVWXYZ' if c not in used]
    
    mapping = {}
    free_idx = 0

    def replace(m):
        nonlocal free_idx
        key = normalize(m)
        if key not in mapping:
            mapping[key] = free[free_idx]
            free_idx += 1
        return mapping[key]

    return pattern.sub(replace, question)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()

    in_path = Path(args.dataset)
    out_path = in_path.with_name(in_path.stem + '_lettered.csv')

    with open(in_path, newline='', encoding='utf-8') as fin, \
         open(out_path, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=['question', 'verifiable_answer'])
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            writer.writerow({
                'question': reletter(row['question']),
                'verifiable_answer': row['verifiable_answer'],
            })

if __name__ == '__main__':
    main()
