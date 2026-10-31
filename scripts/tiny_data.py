"""Very-small-data experiment: does the OU prior prevent failure when only 1-8 drifters are available?

For k in {1, 2, 4, 8} training drifters, draw `--draws` random sets of drifters from the pre-test pool.
Everything is fitted on those drifters only (realistic for a new region):
  - OU prior parameters (alpha, T, sigma) on their windows,
  - early stopping on their last 20% of windows in time (no data from other drifters),
then the OU prior, the direct LSTM and the LSTM + OU prior are evaluated on the full test year.

Usage:  python tiny_data.py <data_dir> <out_dir> [--years 2015 2022] [--draws 5]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows
from pilot_ou_lstm import make_samples, fit_ou, ou_predict, train_net, metrics
from pilot_ablation import LOC_COLS

K_LIST = (1, 2, 4, 8)
MIN_WINDOWS = 40            # a drifter must contribute at least this many windows to be drawn


def summarize(m):
    return dict(sep24=m['sep'][:, 2].mean(), sep72=m['sep'][:, 4].mean(),
                nll=m['nll'].mean(), cov72=m['cover'][:, 4].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--years', type=int, nargs=2, default=[2015, 2022])
    ap.add_argument('--test-start', default='2022-01-01')
    ap.add_argument('--draws', type=int, default=5)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / 'log.txt', 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    d = load(a.data_dir, years=tuple(a.years))
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    S['static'] = np.delete(S['static'], LOC_COLS, axis=1)
    t0_time = d['time'][t0]
    test = t0_time >= np.datetime64(a.test_start).astype('datetime64[s]').astype(float)
    ids = d['id'][t0]
    sub = lambda m: {k: v[m] for k, v in S.items()}
    te = sub(test)

    pool_ids, counts = np.unique(ids[~test], return_counts=True)
    pool_ids = pool_ids[counts >= MIN_WINDOWS]
    log(f'{a.data_dir}: test {test.sum():,} windows; pool {len(pool_ids)} drifters with >= {MIN_WINDOWS} windows')

    rows = []
    rng = np.random.default_rng(2026)
    for k in K_LIST:
        for draw in range(a.draws):
            chosen = rng.choice(pool_ids, size=k, replace=False)
            m = ~test & np.isin(ids, chosen)
            idx = np.flatnonzero(m)
            idx = idx[np.argsort(t0_time[idx], kind='stable')]
            n_val = max(1, int(0.2 * len(idx)))
            tr_m = np.zeros(len(t0), bool); tr_m[idx[:-n_val]] = True
            va_m = np.zeros(len(t0), bool); va_m[idx[-n_val:]] = True
            tr, va = sub(tr_m), sub(va_m)
            p = fit_ou(tr)
            r_ou = summarize(metrics(te, ou_predict(te, p)))
            rms = np.sqrt((tr['y'] ** 2).mean((0, 2)))
            bases = {'LSTM direct': lambda s: (np.zeros_like(s['y']), np.broadcast_to(rms, s['y'].shape[:2]).copy()),
                     'LSTM + OU prior': lambda s: ou_predict(s, p)[:2]}
            row = dict(k=k, draw=draw, n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
                       alpha=p['alpha'], T=p['T'], **{f'OU_{x}': v for x, v in r_ou.items()})
            for name, base in bases.items():
                t_ = time.time()
                predict, n_ep, _ = train_net(tr, va, base(tr), base(va), seed=draw, epochs=100, patience=10, bs=128)
                r = summarize(metrics(te, predict(te, base(te))))
                tag = 'direct' if name == 'LSTM direct' else 'hybrid'
                row.update({f'{tag}_{x}': v for x, v in r.items()}); row[f'{tag}_epochs'] = n_ep
                log(f'  k={k} draw={draw} n_train={tr_m.sum()} {name}: sep24 {r["sep24"]:.2f} '
                    f'(OU {r_ou["sep24"]:.2f}), {n_ep} ep, {time.time() - t_:.0f}s')
            rows.append(row)

    cols = list(rows[0].keys())
    with open(out_dir / 'tiny_results.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')

    log('\n k | windows (median) | 24h OU prior | 24h LSTM direct | 24h LSTM+OU | direct worse than OU (>10%) | hybrid worse than OU (>10%)')
    for k in K_LIST:
        R = [r for r in rows if r['k'] == k]
        f = lambda key: np.array([r[key] for r in R])
        ou, di, hy = f('OU_sep24'), f('direct_sep24'), f('hybrid_sep24')
        fmt = lambda x: f'{np.median(x):.2f} [{x.min():.2f}-{x.max():.2f}]'
        log(f' {k} | {int(np.median(f("n_train")))} | {fmt(ou)} | {fmt(di)} | {fmt(hy)} | '
            f'{int((di > 1.1 * ou).sum())}/{len(R)} | {int((hy > 1.1 * ou).sum())}/{len(R)}')


if __name__ == '__main__':
    main()
