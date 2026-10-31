"""Comparison models on exactly the same data, inputs, split and calibration as paper_final_run.py.

  MLP                 flattened 48-h history + static features -> 2 x 256 ReLU -> Gaussian head
  Seq2Seq-Att         LSTM encoder + autoregressive LSTM decoder over the 5 horizons with Luong (general)
                      attention on the encoder states (a common design in drifter/trajectory forecasting)
  Gradient boosting   HistGradientBoostingRegressor per horizon and component (point forecast only)

Each is run 'direct' and '+ OU prior' (Gaussian/point correction relative to the OU prior, zero-initialised
output layer for the networks).  Networks: ensemble of --seeds, Gaussian NLL, Mondrian rolling calibration.

Usage:  python compare_models.py <data_dir> <out_dir> --years 2015 2022 [--split-lon ... --lat-bands ...]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import load, segments, build_windows
from pilot_ou_lstm import (make_samples, fit_ou, ou_predict, train_net, mixture_to_gaussian,
                           HORIZONS, HIST_STEP, CHI2_90)
from pilot_ablation import LOC_COLS
from pilot_conformal import (scores, conformal_q, rolling_q, group_labels, add_region_args,
                             apply_region_args)

N_STEPS = 48 // HIST_STEP + 1
NH = len(HORIZONS)


class MLPNet(nn.Module):
    def __init__(self, n_seq, n_static, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_STEPS * n_seq + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, NH * 5))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, seq, static):
        return self.net(torch.cat([seq.flatten(1), static], -1)).view(-1, NH, 5)


class Seq2SeqAtt(nn.Module):
    def __init__(self, n_seq, n_static, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.enc = nn.LSTM(n_seq, hidden, layers, batch_first=True, dropout=dropout)
        self.h_emb = nn.Embedding(NH, 16)
        self.dec = nn.LSTMCell(16 + n_static + 5, hidden)
        self.att = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 5))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, seq, static):
        H, (h, c) = self.enc(seq)
        hd, cd = h[-1], c[-1]
        prev = torch.zeros(seq.shape[0], 5, device=seq.device)
        outs = []
        for k in range(NH):
            emb = self.h_emb.weight[k].expand(seq.shape[0], -1)
            hd, cd = self.dec(torch.cat([emb, static, prev], -1), (hd, cd))
            a = torch.softmax(torch.bmm(H, self.att(hd).unsqueeze(2)).squeeze(2), 1)
            ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)
            prev = self.out(torch.cat([hd, ctx], -1))
            outs.append(prev)
        return torch.stack(outs, 1)


def gbm_fit_predict(Xtr, Ytr, Xte_list, seed, max_train):
    """One regressor per target column; returns predictions for each array in Xte_list."""
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, Ytr = Xtr[idx], Ytr[idx]
    preds = [np.zeros((len(X), Ytr.shape[1])) for X in Xte_list]
    for j in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                          random_state=seed)
        m.fit(Xtr, Ytr[:, j])
        for p, X in zip(preds, Xte_list):
            p[:, j] = m.predict(X)
    return preds


def summarize(te, mu, tt, g6, pc=None, pt=None, cal=None):
    sep = np.linalg.norm(mu - te['y'], axis=-1)
    r = {f's{h}': sep[:, i].mean() for i, h in enumerate(HORIZONS)}
    r['med72'] = np.median(sep[:, -1])
    if pt is not None:
        d2c, _ = scores(cal, pc); d2t, ua = scores(te, pt)
        q_static = conformal_q(d2c)
        q_m = rolling_q(d2t, tt, rolling_q(d2t, tt, q_static), g6)
        _, sx, sy, rho = pt
        r['nll'] = (np.log(2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2)) + d2t / 2).mean()
        r['cov_raw72'] = (d2t[:, -1] <= CHI2_90).mean()
        cov = d2t <= q_m
        r['cov_m72'] = cov[:, -1].mean()
        gc = [cov[g6 == g, -1].mean() for g in range(6) if (g6 == g).sum() >= 200]
        r['g72_min'], r['g72_max'] = min(gc), max(gc)
        r['area72'] = np.median(ua[:, -1] * q_m[:, -1]) / 1e3
        r['area24'] = np.median(ua[:, 2] * q_m[:, 2]) / 1e3
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--gbm-max-train', type=int, default=300_000)
    ap.add_argument('--models', nargs='+', default=['MLP', 'Seq2Seq-Att', 'GBM'])
    add_region_args(ap)
    a = ap.parse_args()
    years, test_start, _ = apply_region_args(a)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / 'log.txt', 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    d = load(a.data_dir, years=years)
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    S = make_samples(d, t0)
    S['static'] = np.delete(S['static'], LOC_COLS, axis=1)
    t0_time = d['time'][t0]
    test = t0_time >= test_start
    ids = d['id'][t0]
    rng = np.random.default_rng(0)                         # identical split to paper_final_run.py
    uid = rng.permutation(np.unique(ids[~test]))
    n = len(uid)
    va_ids, cal_ids = uid[:int(0.15 * n)], uid[int(0.15 * n):int(0.30 * n)]
    va = ~test & np.isin(ids, va_ids); cal = ~test & np.isin(ids, cal_ids); tr = ~test & ~va & ~cal
    sub = lambda m: {k: v[m] for k, v in S.items()}
    Str, Sva, Scal, Ste = sub(tr), sub(va), sub(cal), sub(test)
    tt = t0_time[test]
    _, _, g6 = group_labels(Ste)
    log(f'{a.data_dir} years {years}: train {tr.sum():,} val {va.sum():,} cal {cal.sum():,} test {test.sum():,}')

    p = fit_ou(Str)
    rms = np.sqrt((Str['y'] ** 2).mean((0, 2)))
    bases = {'direct': lambda s: (np.zeros_like(s['y']), np.broadcast_to(rms, s['y'].shape[:2]).copy()),
             '+ OU prior': lambda s: ou_predict(s, p)[:2]}
    rows, saved = [], {}

    for model in [m for m in a.models if m != 'GBM']:
        cls = MLPNet if model == 'MLP' else Seq2SeqAtt
        for variant, base in bases.items():
            pc, pt = [], []
            for seed in range(a.seeds):
                t_ = time.time()
                predict, n_ep, vb = train_net(Str, Sva, base(Str), base(Sva), seed, epochs=30, patience=4, net_cls=cls)
                pc.append(predict(Scal, base(Scal))); pt.append(predict(Ste, base(Ste)))
                log(f'  {model} {variant} seed {seed}: {n_ep} ep, val NLL {vb:.3f}, {time.time() - t_:.0f}s')
            gc_, gt_ = mixture_to_gaussian(pc), mixture_to_gaussian(pt)
            r = summarize(Ste, gt_[0], tt, g6, gc_, gt_, Scal)
            rows.append(dict(model=model, variant=variant, **r)); saved[f'{model}|{variant}|mu'] = gt_[0]
            log(f'  -> {model} {variant}: 24h {r["s24"]:.2f} 72h {r["s72"]:.2f} cov {r["cov_m72"]:.3f} area72 {r["area72"]:.1f}')

    if 'GBM' in a.models:
        feat = lambda s: np.concatenate([s['seq'].reshape(len(s['y']), -1), s['static']], 1)
        Xtr, Xte = feat(Str), feat(Ste)
        mu_tr, sd_tr = ou_predict(Str, p)[:2]; mu_te, sd_te = ou_predict(Ste, p)[:2]
        for variant in bases:
            t_ = time.time()
            if variant == 'direct':
                (pred,) = gbm_fit_predict(Xtr, Str['y'].reshape(len(Xtr), -1), [Xte], 0, a.gbm_max_train)
                mu = pred.reshape(-1, NH, 2)
            else:
                Ytr = ((Str['y'] - mu_tr) / sd_tr[..., None]).reshape(len(Xtr), -1)
                (pred,) = gbm_fit_predict(Xtr, Ytr, [Xte], 0, a.gbm_max_train)
                mu = mu_te + sd_te[..., None] * pred.reshape(-1, NH, 2)
            r = summarize(Ste, mu, tt, g6)
            rows.append(dict(model='Gradient boosting', variant=variant, **r)); saved[f'GBM|{variant}|mu'] = mu
            log(f'  -> GBM {variant}: 24h {r["s24"]:.2f} 72h {r["s72"]:.2f} ({time.time() - t_:.0f}s)')

    cols = ['model', 'variant'] + [f's{h}' for h in HORIZONS] + ['med72', 'nll', 'cov_raw72', 'cov_m72',
                                                                 'g72_min', 'g72_max', 'area24', 'area72']
    with open(out_dir / 'compare_results.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) if isinstance(r.get(c), str) else (f'{r[c]:.4f}' if c in r else '')
                             for c in cols) + '\n')
    np.savez_compressed(out_dir / 'compare_predictions.npz', **saved)
    log('saved ' + str(out_dir / 'compare_results.csv'))


if __name__ == '__main__':
    main()
