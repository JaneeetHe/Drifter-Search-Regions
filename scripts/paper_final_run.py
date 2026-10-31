"""Final pilot experiment for the paper: one consistent split, all models, saved predictions.

Data      GDP hourly v2.01, GPS drifters (v2.01.1 flag), NW Pacific 20-45N 120-180E, 2020-01..2022-10
Split     2020-21 drifters -> 70% train / 15% early-stopping val / 15% calibration ; 2022 = test
Models    persistence, constant velocity (past-only), damped CV, OU prior,
          LSTM direct (ensemble of 3), LSTM + OU prior (ensemble of 3); no location inputs
Calib.    Mondrian rolling conformal (latitude band x drogue state) for OU and LSTM + OU

Output    <out_dir>/final_predictions.npz  (everything the figure script needs)

Usage:  python paper_final_run.py <data_dir> <out_dir> [--seeds 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows, to_local
from pilot_ou_lstm import (make_samples, fit_ou, ou_predict, ou_mean, train_net,
                           mixture_to_gaussian, HORIZONS)
from pilot_ablation import LOC_COLS
from pilot_conformal import scores, conformal_q, rolling_q, group_labels, add_region_args, apply_region_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, default=3)
    add_region_args(ap)
    a = ap.parse_args()
    years, test_start, _ = apply_region_args(a)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'data {a.data_dir} years {years} test from {a.test_start}')

    d = load(a.data_dir, years=years)
    starts, lens = segments(d)
    t0, n_drop = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    hist_xy = S['seq'][:, :, :2].copy()                       # km, every 3h, for case-study plots
    S['static'] = np.delete(S['static'], LOC_COLS, axis=1)
    t0_time = d['time'][t0]
    test = t0_time >= test_start
    ids = d['id'][t0]

    rng = np.random.default_rng(0)
    uid = rng.permutation(np.unique(ids[~test]))
    n = len(uid)
    va_ids, cal_ids = uid[:int(0.15 * n)], uid[int(0.15 * n):int(0.30 * n)]
    va = ~test & np.isin(ids, va_ids)
    cal = ~test & np.isin(ids, cal_ids)
    tr = ~test & ~va & ~cal
    sub = lambda m: {k: v[m] for k, v in S.items()}
    Str, Sva, Scal, Ste = sub(tr), sub(va), sub(cal), sub(test)
    print(f'windows: train {tr.sum():,} val {va.sum():,} cal {cal.sum():,} test {test.sum():,} '
          f'(dropped {n_drop:,} with gaps)')

    # ---- physics-style baselines
    p = fit_ou(Str)
    T_grid = [12, 24, 36, 48, 72, 96, 144]
    T_d = min(T_grid, key=lambda T: np.linalg.norm(
        ou_mean(Str['u0'], Str['w48'], 0.0, T) - Str['y'], axis=-1).mean())
    tau = HORIZONS[None, :, None].astype(float)
    det = {'Persistence': np.zeros_like(Ste['y']),
           'Constant velocity': Ste['u0'][:, None, :] * tau,
           'Damped velocity': ou_mean(Ste['u0'], Ste['w48'], 0.0, T_d)}
    print(f'OU prior: alpha={p["alpha"]} T={p["T"]}h sigma={np.sqrt(p["sigma2"]) / 3.6:.3f} m/s ; damped T={T_d}h')

    # ---- networks
    base_ou = lambda s: ou_predict(s, p)[:2]
    rms = np.sqrt((Str['y'] ** 2).mean((0, 2)))
    base_direct = lambda s: (np.zeros_like(s['y']), np.broadcast_to(rms, s['y'].shape[:2]).copy())
    prob = {'OU prior': (ou_predict(Scal, p), ou_predict(Ste, p))}
    for name, base in [('LSTM direct', base_direct), ('LSTM + OU prior', base_ou)]:
        pc, pt = [], []
        for seed in range(a.seeds):
            predict, n_ep, vb = train_net(Str, Sva, base(Str), base(Sva), seed, epochs=30, patience=4)
            pc.append(predict(Scal, base(Scal))); pt.append(predict(Ste, base(Ste)))
            print(f'  {name} seed {seed}: {n_ep} epochs, val NLL {vb:.3f}')
        prob[name] = (mixture_to_gaussian(pc), mixture_to_gaussian(pt))

    # ---- conformal: Mondrian rolling (lat band x drogue)
    tt = t0_time[test]
    _, _, g6 = group_labels(Ste)
    out = dict(HORIZONS=HORIZONS, y=Ste['y'], t0_time=tt, lat0=Ste['lat0'], lon0=Ste['lon0'],
               group=Ste['group'].astype('U10'), group6=g6, id=ids[test],
               hist_xy=hist_xy[test], ou_alpha=p['alpha'], ou_T=p['T'], ou_sigma2=p['sigma2'],
               damped_T=T_d)
    for name, arr in det.items():
        out[f'{name}|mu'] = arr
    for name, (pc, pt) in prob.items():
        d2c, _ = scores(Scal, pc)
        d2t, _ = scores(Ste, pt)
        q_static = conformal_q(d2c)
        q_roll = rolling_q(d2t, tt, q_static)
        q_mond = rolling_q(d2t, tt, q_roll, g6)
        for k, v in zip(('mu', 'sx', 'sy', 'rho'), pt):
            out[f'{name}|{k}'] = v
        out[f'{name}|q_mondrian'] = q_mond
        out[f'{name}|q_rolling'] = q_roll

    # hourly future truth (km) for case-study plots
    F = t0[test][:, None] + np.arange(1, 73)
    fx, fy = to_local(d['lat'][F], d['lon'][F], Ste['lat0'][:, None], Ste['lon0'][:, None])
    out['future_xy_hourly'] = np.stack([fx, fy], -1).astype('float32') / 1000

    # drifter tracks + split label for the study-area map
    split_of = {i: 'test' for i in np.unique(ids[test])}
    split_of.update({i: 'calibration' for i in cal_ids}); split_of.update({i: 'validation' for i in va_ids})
    split_of.update({i: 'train' for i in np.unique(ids[tr])})
    out.update(track_id=d['id'], track_lat=d['lat'].astype('float32'), track_lon=d['lon'].astype('float32'),
               track_time=d['time'], split_ids=np.array(list(split_of.keys())),
               split_labels=np.array(list(split_of.values())))
    np.savez_compressed(out_dir / 'final_predictions.npz', **out)
    print('saved', out_dir / 'final_predictions.npz')


if __name__ == '__main__':
    main()
