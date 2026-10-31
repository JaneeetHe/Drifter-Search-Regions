"""Upper bound on how much position-smoothing leakage could help the LSTM.

The hourly product position at t0 is fitted from fixes around t0, so it can carry
information from (t0, t0+w].  Shifting the issue time does not remove this (the leak
shifts with it).  Instead we give the LSTM the TRUE positions at t0+1..t0+w explicitly:
any implicit leak carries at most this information, so the error reduction from the
explicit future positions bounds the leak's contribution.

Usage:  python pilot_leak_oracle.py <data_dir> <out_dir> [--seeds 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows, to_local, TEST_START, PILOT_YEARS
from pilot_ou_lstm import make_samples, fit_ou, ou_predict, train_net, metrics
from pilot_ablation import LOC_COLS, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--years', type=int, nargs=2, default=list(PILOT_YEARS), metavar=('FIRST', 'LAST'))
    ap.add_argument('--test-start', default='2022-01-01')
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    d = load(a.data_dir, years=tuple(a.years))
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    S['static'] = np.delete(S['static'], LOC_COLS, axis=1)          # no location inputs
    test = d['time'][t0] >= np.datetime64(a.test_start).astype('datetime64[s]').astype(float)
    ids = d['id'][t0]
    rng = np.random.default_rng(0)
    uid = np.unique(ids[~test])
    val = np.isin(ids, rng.choice(uid, size=len(uid) // 5, replace=False)) & ~test
    tr_m, va_m = ~test & ~val, val

    # true future positions (km, relative to t0) at +1..+6h
    lat0, lon0 = d['lat'][t0], d['lon'][t0]
    Fh = t0[:, None] + np.arange(1, 7)
    fx, fy = to_local(d['lat'][Fh], d['lon'][Fh], lat0[:, None], lon0[:, None])
    fut = np.stack([fx, fy], -1) / 1000                                  # (n, 6, 2)

    sub = lambda s, m: {k: v[m] for k, v in s.items()}
    rows = []
    for w in (0, 1, 3, 6):
        Sw = dict(S)
        if w:
            Sw['static'] = np.concatenate([S['static'], fut[:, :w].reshape(len(t0), -1)], 1).astype('float32')
        tr, va, te = sub(Sw, tr_m), sub(Sw, va_m), sub(Sw, test)
        p = fit_ou(tr)
        if w == 0:
            rows.append(('OU prior (no leak)', summarize(metrics(te, ou_predict(te, p)))))
        base = lambda s: ou_predict(s, p)[:2]
        res = []
        for seed in range(a.seeds):
            predict, n_ep, _ = train_net(tr, va, base(tr), base(va), seed, epochs=30, patience=4)
            res.append(summarize(metrics(te, predict(te, base(te)))))
            print(f'  w={w}h seed {seed}: {n_ep} ep, sep24 {res[-1]["sep24"]:.2f}', flush=True)
        agg = {k: np.mean([r[k] for r in res]) for k in res[0]}
        agg['sep24_range'] = np.ptp([r['sep24'] for r in res])
        label = 'LSTM + OU (standard, history to t0)' if w == 0 else f'LSTM + OU + true pos. t0+1..t0+{w}h'
        rows.append((label, agg))

    lines = [f'{"model":<40}{"24h":>7}{"±":>6}{"48h":>7}{"72h":>7}{"NLL":>7}{"cov72":>7}']
    for name, r in rows:
        lines.append(f'{name:<40}{r["sep24"]:7.2f}{r.get("sep24_range", np.nan):6.2f}{r["sep48"]:7.2f}'
                     f'{r["sep72"]:7.2f}{r["nll"]:7.2f}{r["cov72"]:7.3f}')
    text = '\n'.join(lines)
    print('\n' + text)
    (out_dir / 'leak_oracle_report.txt').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
