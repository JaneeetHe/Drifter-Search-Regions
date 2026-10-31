"""Where does the OU prior help?  Three pilot experiments on GDP hourly (NW Pacific).

  A  location ablation   : LSTM with vs without (lat0, lon0, f) inputs, test 2022
  B  spatial holdout     : train 2020-21 west of 150E -> test 2022 east, and the reverse
  C  small training sets : 2/5/10/25% of training drifters (no location inputs), test 2022

Each setting compares OU prior alone, LSTM direct and LSTM + OU prior (mean over seeds).

Usage:  python pilot_ablation.py <data_dir> <out_dir> --exp A|B|C [--seeds 2] [--threads 8]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows, TEST_START, PILOT_YEARS
from pilot_ou_lstm import make_samples, fit_ou, ou_predict, train_net, metrics

LOC_COLS = [0, 1, 4]                  # lat0, lon0, f in make_samples' static block
SPLIT_LON = 150.0


def subset(s, m, drop_loc):
    out = {k: v[m] for k, v in s.items()}
    if drop_loc:
        out['static'] = np.delete(out['static'], LOC_COLS, axis=1)
    return out


def summarize(m):
    return dict(sep24=m['sep'][:, 2].mean(), sep48=m['sep'][:, 3].mean(), sep72=m['sep'][:, 4].mean(),
                nll=m['nll'].mean(), cov72=m['cover'][:, 4].mean(), area72=np.median(m['area'][:, 4]))


def run_setting(S, tr_m, va_m, te_m, drop_loc, seeds, tag, log):
    tr, va, te = (subset(S, m, drop_loc) for m in (tr_m, va_m, te_m))
    p = fit_ou(tr)
    rows = [dict(tag=tag, model='OU prior', n_train=len(tr['y']), seeds=0,
                 **summarize(metrics(te, ou_predict(te, p))))]
    small = len(tr['y']) < 20000
    kw = dict(epochs=80, patience=8) if small else dict(epochs=30, patience=4)
    rms = np.sqrt((tr['y'] ** 2).mean((0, 2)))
    bases = {'LSTM direct': lambda s: (np.zeros_like(s['y']), np.broadcast_to(rms, s['y'].shape[:2]).copy()),
             'LSTM + OU prior': lambda s: ou_predict(s, p)[:2]}
    for name, base in bases.items():
        per_seed = []
        for seed in range(seeds):
            t_ = time.time()
            predict, n_ep, _ = train_net(tr, va, base(tr), base(va), seed, **kw)
            per_seed.append(summarize(metrics(te, predict(te, base(te)))))
            log(f'  [{tag}] {name} seed {seed}: {n_ep} ep, {time.time() - t_:.0f}s, '
                f'sep24 {per_seed[-1]["sep24"]:.2f}')
        agg = {k: np.mean([r[k] for r in per_seed]) for k in per_seed[0]}
        agg['sep24_range'] = np.ptp([r['sep24'] for r in per_seed])
        rows.append(dict(tag=tag, model=name, n_train=len(tr['y']), seeds=seeds, **agg))
    log(f'  [{tag}] OU fit: alpha={p["alpha"]} T={p["T"]}h')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--exp', required=True, choices=list('ABC'))
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--years', type=int, nargs=2, default=list(PILOT_YEARS), metavar=('FIRST', 'LAST'))
    ap.add_argument('--test-start', default='2022-01-01')
    ap.add_argument('--split-lon', type=float, default=SPLIT_LON)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / f'log_{a.exp}.txt', 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    d = load(a.data_dir, years=tuple(a.years))
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    test = d['time'][t0] >= np.datetime64(a.test_start).astype('datetime64[s]').astype(float)
    ids = d['id'][t0]
    west = S['lon0'] < a.split_lon
    rng = np.random.default_rng(0)

    def split_ids(pool_mask, val_frac=0.2):
        uid = np.unique(ids[pool_mask])
        val = set(rng.choice(uid, size=max(1, int(len(uid) * val_frac)), replace=False))
        is_val = pool_mask & np.isin(ids, list(val))
        return pool_mask & ~is_val, is_val

    rows = []
    if a.exp == 'A':
        tr_m, va_m = split_ids(~test)
        for drop in (False, True):
            rows += run_setting(S, tr_m, va_m, test, drop, a.seeds,
                                'A ' + ('no location' if drop else 'with location'), log)
    elif a.exp == 'B':
        for name, src in [('W->E', west), ('E->W', ~west)]:
            tr_m, va_m = split_ids(~test & src)
            te_m = test & ~src
            rows += run_setting(S, tr_m, va_m, te_m, True, a.seeds, f'B {name} no location', log)
            rows += run_setting(S, tr_m, va_m, te_m, False, a.seeds, f'B {name} with location', log)
            # in-distribution reference: same target region, trained on target-region drifters
            tr_i, va_i = split_ids(~test & ~src)
            rows += run_setting(S, tr_i, va_i, te_m, True, a.seeds, f'B {name} in-region ref', log)
    else:
        tr_full, va_m = split_ids(~test)
        uid = np.unique(ids[tr_full])
        for frac in (0.02, 0.05, 0.10, 0.25):
            keep = rng.choice(uid, size=max(2, int(round(len(uid) * frac))), replace=False)
            tr_m = tr_full & np.isin(ids, keep)
            rows += run_setting(S, tr_m, va_m, test, True, a.seeds,
                                f'C {int(frac * 100)}% drifters ({len(keep)})', log)

    cols = ['tag', 'model', 'n_train', 'seeds', 'sep24', 'sep24_range', 'sep48', 'sep72',
            'nll', 'cov72', 'area72']
    with open(out_dir / f'results_{a.exp}.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(c, '')) if isinstance(r.get(c, ''), str) or c in ('n_train', 'seeds')
                             else f'{r.get(c, np.nan):.4f}' for c in cols) + '\n')
    log('\n' + f'{"setting":<34}{"model":<17}{"n_train":>8}{"24h":>7}{"±":>6}{"48h":>7}{"72h":>7}'
        f'{"NLL":>7}{"cov72":>7}{"area72":>8}')
    for r in rows:
        log(f'{r["tag"]:<34}{r["model"]:<17}{r["n_train"]:>8}{r["sep24"]:7.2f}'
            f'{r.get("sep24_range", np.nan):6.2f}{r["sep48"]:7.2f}{r["sep72"]:7.2f}'
            f'{r["nll"]:7.2f}{r["cov72"]:7.3f}{r["area72"]:8.0f}')


if __name__ == '__main__':
    main()
