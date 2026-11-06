"""Evaluate the adversarial check: transfer rate against edit size.

Reads the inversion logs and puts the transfer rate next to the agreement
baseline. It additionally splits the flips into those that transferred and those
that did not -- the question behind it being whether sparse counterfactuals are
more model-specific than large ones.

Usage:
    python src/analyze_transfer.py logs2/inv_transfer_proteins.log 0.9425
"""
import re
import statistics as st
import sys


def parse(path):
    rows, cur = [], None
    for line in open(path):
        if 'Flipped at t=' in line:
            seg = line.split('Flipped at t=')[1]
            cur = {'t': int(seg.split(',')[0])}
            for key in ('GED', 'ModRate', 'NodeSparsity', 'EdgeSparsity'):
                m = re.search(rf'{key}=([-\d.]+)', seg)
                if m:
                    cur[key] = float(m.group(1))
        elif 'Transfer-Judge:' in line and cur is not None:
            cur['transfer'] = 0 if 'kein Flip' in line else 1
            rows.append(cur)
            cur = None
    return rows


def main():
    path = sys.argv[1]
    baseline = float(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = parse(path)
    if not rows:
        print(f'{path}: keine ausgewerteten Flips gefunden')
        return

    n = len(rows)
    k = sum(r['transfer'] for r in rows)
    print('=' * 76)
    print(path)
    print('=' * 76)
    print(f'  Flips mit Transfer-Auswertung : {n}')
    print(f'  Transfer-Flip-Rate            : {k}/{n} = {k / n:.3f}')
    if baseline is not None:
        print(f'  Agreement-Baseline (Originale): {baseline:.3f}')
        print(f'  Differenz                     : {k / n - baseline:+.3f}')
    print()
    print(f'  {"Gruppe":20s} {"n":>3s} {"tau":>7s} {"NodeSp":>8s} {"EdgeSp":>8s} {"GED":>7s}')
    for grp, lab in ((1, 'uebertragen'), (0, 'nicht uebertragen')):
        g = [r for r in rows if r['transfer'] == grp]
        if not g:
            continue
        print(f'  {lab:20s} {len(g):3d} '
              f'{st.mean(r["t"] for r in g):7.1f} '
              f'{st.mean(r["NodeSparsity"] for r in g):8.3f} '
              f'{st.mean(r["EdgeSparsity"] for r in g):8.3f} '
              f'{st.mean(r["GED"] for r in g):7.2f}')
    print('=' * 76)


if __name__ == '__main__':
    main()
