"""Conformal calibration of the probabilistic drift forecasts (90% prediction ellipses).

Nonconformity score: squared Mahalanobis distance d^2 of the true displacement under the
predicted bivariate Gaussian, per horizon.  The calibrated region is {d^2 <= q_h}, i.e. the
predicted ellipse rescaled; its area is pi * q_h * sx * sy * sqrt(1 - rho^2).

Calibration variants
  raw       uncalibrated Gaussian 90% ellipse (q = chi2_2(0.9) = 4.605)
  static    split conformal on a held-out calibration set of 2020-21 drifters
  rolling   operational recalibration: for an issue on day D, use the scores of forecasts in the
            same target stream whose verification time is within the 30 days before D
            (falls back to `static` while fewer than 100 verified forecasts exist)

Settings: in-distribution (test 2022 same region) and the two cross-region transfers.
Models:   OU prior and the LSTM + OU prior ensemble (no location inputs), both calibrated,
          so search areas are compared at equal coverage.

Usage:  python pilot_conformal.py <data_dir> <out_dir> [--seeds 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows, TEST_START, PILOT_YEARS
from pilot_ou_lstm import (make_samples, fit_ou, ou_predict, train_net,
                           mixture_to_gaussian, HORIZONS, CHI2_90)
from pilot_ablation import LOC_COLS, SPLIT_LON

ALPHA = 0.10
ROLL_DAYS = 30
MIN_POOL = 100


def scores(s, pred):
    """d^2 and unit ellipse area (area of {d^2 <= 1}) per sample and horizon."""
    mu, sx, sy, rho = pred
    e = s['y'] - mu
    one_r2 = 1 - rho ** 2
    d2 = ((e[..., 0] / sx) ** 2 + (e[..., 1] / sy) ** 2
          - 2 * rho * e[..., 0] * e[..., 1] / (sx * sy)) / one_r2
    return d2, np.pi * sx * sy * np.sqrt(one_r2)


def conformal_q(d2_cal):
    n = len(d2_cal)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return np.quantile(d2_cal, level, axis=0, method='higher')       # per horizon


LAT_BANDS = [(20, 30), (30, 38), (38, 45.01)]


def rolling_q(d2, t0_time, q_fallback, groups=None):
    """Per-sample threshold from verified forecasts of the previous ROLL_DAYS days.

    With `groups` (Mondrian), the pool is restricted to forecasts of the same group; if a
    group has < MIN_POOL verified forecasts, the window widens to 3*ROLL_DAYS, and if that is
    still too small the sample keeps `q_fallback` (per-sample, e.g. the marginal rolling q)."""
    q_fallback = np.broadcast_to(q_fallback, d2.shape)
    q = np.array(q_fallback, dtype=float)
    day = np.floor(t0_time / 86400).astype(np.int64)
    if groups is None:
        groups = np.zeros(len(d2), dtype=int)
    for g in np.unique(groups):
        gi = np.flatnonzero(groups == g)
        gday = day[gi]
        for k, h in enumerate(HORIZONS):
            verify = t0_time[gi] + h * 3600
            for D in np.unique(gday):
                start = D * 86400
                m = gi[gday == D]
                for width in (ROLL_DAYS, 3 * ROLL_DAYS):
                    pool = (verify <= start) & (verify > start - width * 86400)
                    if pool.sum() >= MIN_POOL:
                        q[m, k] = conformal_q(d2[gi[pool], k][:, None])[0]
                        break
    return q


def group_labels(te):
    lat = te['lat0']
    band = np.select([(lat >= lo) & (lat < hi) for lo, hi in LAT_BANDS], [0, 1, 2], default=2)
    drog = (te['group'] == 'drogued').astype(int)          # unknown drogue state -> not drogued
    return band, drog, band * 2 + drog


GROUP_NAMES = [f'{lo}-{int(hi)}N {d}' for lo, hi in LAT_BANDS for d in ('undrogued', 'drogued')]


def set_lat_bands(edges):
    """Mondrian latitude bands from 4 edges, e.g. [20, 30, 38, 45] (last band includes its top)."""
    global LAT_BANDS, GROUP_NAMES
    LAT_BANDS = [(edges[0], edges[1]), (edges[1], edges[2]), (edges[2], edges[3] + 0.01)]
    GROUP_NAMES = [f'{lo:g}-{hi:.0f}N {d}' for lo, hi in LAT_BANDS for d in ('undrogued', 'drogued')]


def add_region_args(ap):
    """Data-range options shared by the final run and the calibration experiment.
    Defaults reproduce the pilot (NW Pacific 2020-2022, test 2022)."""
    ap.add_argument('--years', type=int, nargs=2, default=list(PILOT_YEARS), metavar=('FIRST', 'LAST'))
    ap.add_argument('--test-start', default='2022-01-01')
    ap.add_argument('--split-lon', type=float, default=SPLIT_LON, help='west/east split for transfer tests')
    ap.add_argument('--lat-bands', type=float, nargs=4, default=[20, 30, 38, 45], help='Mondrian band edges')


def apply_region_args(a):
    set_lat_bands(a.lat_bands)
    return (tuple(a.years), np.datetime64(a.test_start).astype('datetime64[s]').astype(float), a.split_lon)


def evaluate(te, d2, unit_area, q):
    cover = d2 <= q
    area = unit_area * q
    r = dict(cov24=cover[:, 2].mean(), cov72=cover[:, 4].mean(),
             area24=np.median(area[:, 2]), area72=np.median(area[:, 4]))
    _, _, g6 = group_labels(te)
    per = {}
    for g in range(6):
        m = g6 == g
        if m.sum() >= 200:
            per[GROUP_NAMES[g]] = (cover[m, 2].mean(), cover[m, 4].mean(), int(m.sum()))
    r['per_group'] = per
    c24 = [v[0] for v in per.values()]; c72 = [v[1] for v in per.values()]
    r.update(g24_min=min(c24), g24_max=max(c24), g72_min=min(c72), g72_max=max(c72))
    return r


def run(S, t0_time, tr_m, va_m, cal_m, te_m, seeds, tag, log):
    sub = lambda m: {k: v[m] for k, v in S.items()}
    tr, va, cal, te = sub(tr_m), sub(va_m), sub(cal_m), sub(te_m)
    tt = t0_time[te_m]
    log(f'[{tag}] train {len(tr["y"]):,}  val {len(va["y"]):,}  cal {len(cal["y"]):,}  test {len(te["y"]):,}')
    p = fit_ou(tr)
    base = lambda s: ou_predict(s, p)[:2]

    preds_cal, preds_te = [], []
    for seed in range(seeds):
        predict, n_ep, _ = train_net(tr, va, base(tr), base(va), seed, epochs=30, patience=4)
        preds_cal.append(predict(cal, base(cal))); preds_te.append(predict(te, base(te)))
        log(f'  seed {seed}: {n_ep} epochs')
    models = {'OU prior': (ou_predict(cal, p), ou_predict(te, p)),
              'LSTM + OU (ens)': (mixture_to_gaussian(preds_cal), mixture_to_gaussian(preds_te))}

    rows = []
    for name, (pc, pt) in models.items():
        d2c, _ = scores(cal, pc)
        d2t, ua = scores(te, pt)
        q_static = conformal_q(d2c)
        sep72 = np.linalg.norm(te['y'][:, 4] - pt[0][:, 4], axis=-1).mean()
        q_roll = rolling_q(d2t, tt, q_static)
        band, drog, g6 = group_labels(te)
        variants = [('raw', np.full_like(d2t, CHI2_90)),
                    ('static', np.broadcast_to(q_static, d2t.shape)),
                    ('rolling', q_roll),
                    ('M-lat', rolling_q(d2t, tt, q_roll, band)),
                    ('M-drogue', rolling_q(d2t, tt, q_roll, drog)),
                    ('M-lat×drg', rolling_q(d2t, tt, q_roll, g6))]
        for cal_name, q in variants:
            rows.append(dict(tag=tag, model=name, calib=cal_name, sep72=sep72,
                             **evaluate(te, d2t, ua, q)))
        log(f'  {name}: static q/chi2 per horizon = {np.round(q_static / CHI2_90, 2)}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, default=3)
    add_region_args(ap)
    a = ap.parse_args()
    years, test_start, split_lon = apply_region_args(a)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / 'log.txt', 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    log(f'data {a.data_dir} years {years} test from {a.test_start} split_lon {split_lon} lat bands {LAT_BANDS}')
    d = load(a.data_dir, years=years)
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    S['static'] = np.delete(S['static'], LOC_COLS, axis=1)
    t0_time = d['time'][t0]
    test = t0_time >= test_start
    ids = d['id'][t0]
    west = S['lon0'] < split_lon
    rng = np.random.default_rng(0)

    def three_way(pool):
        """Split pool by drifter: 70% train, 15% early-stopping val, 15% conformal calibration."""
        uid = rng.permutation(np.unique(ids[pool]))
        n = len(uid)
        va_ids, cal_ids = uid[:int(0.15 * n)], uid[int(0.15 * n):int(0.30 * n)]
        va = pool & np.isin(ids, va_ids)
        cal = pool & np.isin(ids, cal_ids)
        return pool & ~va & ~cal, va, cal

    rows = []
    tr, va, cal = three_way(~test)
    rows += run(S, t0_time, tr, va, cal, test, a.seeds, 'in-region (all NWP)', log)
    for name, src in [('W->E', west), ('E->W', ~west)]:
        tr, va, cal = three_way(~test & src)
        rows += run(S, t0_time, tr, va, cal, test & ~src, a.seeds, f'cross-region {name}', log)

    cols = ['tag', 'model', 'calib', 'sep72', 'cov24', 'cov72', 'g24_min', 'g24_max',
            'g72_min', 'g72_max', 'area24', 'area72']
    with open(out_dir / 'conformal_results.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(r[c] if isinstance(r[c], str) else f'{r[c]:.4f}' for c in cols) + '\n')
    log(f'\n{"setting":<22}{"model":<17}{"calib":<11}{"cov24":>7}{"cov72":>7}'
        f'{"grp24 min-max":>15}{"grp72 min-max":>15}{"area24":>9}{"area72":>9}')
    for r in rows:
        log(f'{r["tag"]:<22}{r["model"]:<17}{r["calib"]:<11}{r["cov24"]:7.3f}{r["cov72"]:7.3f}'
            f'{r["g24_min"]:9.3f}-{r["g24_max"]:.3f}{r["g72_min"]:9.3f}-{r["g72_max"]:.3f}'
            f'{r["area24"]:9.0f}{r["area72"]:9.0f}')
    log('\nPer-group coverage 24h/72h (n) for LSTM + OU (ens):')
    for r in rows:
        if r['model'] == 'LSTM + OU (ens)' and r['calib'] in ('rolling', 'M-lat×drg'):
            log(f'  {r["tag"]} [{r["calib"]}]: ' + '; '.join(
                f'{g} {c24:.2f}/{c72:.2f} ({n})' for g, (c24, c72, n) in r['per_group'].items()))


if __name__ == '__main__':
    main()
