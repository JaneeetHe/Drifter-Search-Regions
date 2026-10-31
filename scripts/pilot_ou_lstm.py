"""Pilot: OU (Langevin) prior + LSTM residual with bivariate-Gaussian output.

All inputs are past-only (positions up to t0, backward-difference velocities);
the product ve/vn is not used because it contains t0+1h information (see T1).

Models compared on the 2022 test year:
  damped CV          deterministic mean of the OU prior with background velocity 0
  OU prior           mean u_b*tau + (u0-u_b)*T(1-e^{-tau/T}), analytic isotropic variance
  LSTM direct        LSTM predicts displacement + Gaussian (no physics prior)
  LSTM + OU prior    LSTM predicts a residual and variance multipliers on top of the OU prior

Usage:  python pilot_ou_lstm.py <data_dir> <out_dir> [--seeds 3]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_t1_t2 import (load, segments, build_windows, to_local, to_latlon,
                         haversine_km, OMEGA, TEST_START, PILOT_YEARS)

HORIZONS = np.array([6, 12, 24, 48, 72])            # hours
HIST_STEP = 3                                         # history sampled every 3h over 48h
CHI2_90 = 4.605                                       # chi^2_2 quantile for 90% ellipse


# == Windows and features (km, local tangent plane at t0) ==
def make_samples(d, t0):
    lat0, lon0 = d['lat'][t0], d['lon'][t0]
    offs = np.arange(-48, 1, HIST_STEP)
    H = t0[:, None] + offs
    hx, hy = to_local(d['lat'][H], d['lon'][H], lat0[:, None], lon0[:, None])
    hx, hy = hx / 1000, hy / 1000
    F = t0[:, None] + HORIZONS
    fx, fy = to_local(d['lat'][F], d['lon'][F], lat0[:, None], lon0[:, None])
    y = np.stack([fx / 1000, fy / 1000], -1)                          # (n, 5, 2) km

    vx = np.diff(hx, axis=1) / HIST_STEP; vy = np.diff(hy, axis=1) / HIST_STEP   # km/h
    vx = np.c_[vx[:, :1], vx]; vy = np.c_[vy[:, :1], vy]
    seq = np.stack([hx, hy, vx, vy, np.hypot(vx, vy)], -1)            # (n, 17, 5)

    t = d['time'][t0]
    doy = 2 * np.pi * ((t / 86400) % 365.25) / 365.25
    dld = d['dld'][t0]
    unknown = np.isnan(dld) | (dld <= 0)
    drogued = ~unknown & (t < dld)
    undrogued = ~unknown & ~drogued
    f = 2 * OMEGA * np.sin(np.radians(lat0)) * 1e4
    static = np.stack([lat0, lon0, np.sin(doy), np.cos(doy), f,
                       drogued, undrogued, unknown], -1).astype(float)

    # past-only velocity for the prior: 3h backward difference (km/h)
    u0 = np.stack([vx[:, -1], vy[:, -1]], -1)
    # 48h least-squares mean velocity (km/h)
    th = offs.astype(float)
    P = np.linalg.pinv(np.stack([np.ones_like(th), th], 1))
    w48 = np.stack([hx @ P[1], hy @ P[1]], -1)
    group = np.where(drogued, 'drogued', np.where(undrogued, 'undrogued', 'unknown'))
    return dict(seq=seq.astype('float32'), static=static.astype('float32'), y=y,
                u0=u0, w48=w48, lat0=lat0, lon0=lon0, group=group)


# == OU prior ==
def ou_mean(u0, w48, alpha, T):
    tau = HORIZONS[None, :, None].astype(float)
    ub = alpha * w48[:, None, :]
    return ub * tau + (u0[:, None, :] - ub) * T * (1 - np.exp(-tau / T))


def ou_var_shape(T):
    """Var of integrated OU velocity given u0, per unit stationary variance (km^2 per (km/h)^2)."""
    tau = HORIZONS.astype(float)
    return T ** 2 * (2 * tau / T - 3 + 4 * np.exp(-tau / T) - np.exp(-2 * tau / T))


def fit_ou(s):
    """Grid-search T and background fraction alpha on train (mean error over horizons),
    then MLE of the stationary velocity variance per horizon-shape."""
    best = None
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for T in [12, 24, 36, 48, 72, 96, 144]:
            m = ou_mean(s['u0'], s['w48'], alpha, T)
            err = np.linalg.norm(m - s['y'], axis=-1).mean()
            if best is None or err < best[0]:
                best = (err, alpha, T)
    _, alpha, T = best
    m = ou_mean(s['u0'], s['w48'], alpha, T)
    g = ou_var_shape(T)
    sigma2 = np.mean(((s['y'] - m) ** 2).sum(-1) / (2 * g[None, :]))
    return dict(alpha=alpha, T=T, sigma2=sigma2)


def ou_predict(s, p):
    mu = ou_mean(s['u0'], s['w48'], p['alpha'], p['T'])
    sd = np.sqrt(p['sigma2'] * ou_var_shape(p['T']))
    sd = np.broadcast_to(sd[None, :], mu.shape[:2])
    return mu, sd, sd, np.zeros(mu.shape[:2])


# == Network ==
class Net(nn.Module):
    def __init__(self, n_seq, n_static, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 128), nn.ReLU(),
                                  nn.Linear(128, len(HORIZONS) * 5))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq, static):
        h, _ = self.lstm(seq)
        return self.head(torch.cat([h[:, -1], static], -1)).view(-1, len(HORIZONS), 5)


def gaussian_params(out, base_mu, base_sd):
    """mu = base_mu + base_sd*dm ;  sigma = base_sd*exp(a) ;  rho = 0.95 tanh(c)."""
    mu = base_mu + base_sd[..., None] * out[..., :2]
    sx = base_sd * torch.exp(out[..., 2].clamp(-5, 5))
    sy = base_sd * torch.exp(out[..., 3].clamp(-5, 5))
    rho = 0.95 * torch.tanh(out[..., 4])
    return mu, sx, sy, rho


def nll(y, mu, sx, sy, rho):
    ex, ey = (y - mu).unbind(-1)
    one_r2 = 1 - rho ** 2
    z = (ex / sx) ** 2 + (ey / sy) ** 2 - 2 * rho * ex * ey / (sx * sy)
    return torch.log(2 * np.pi * sx * sy * torch.sqrt(one_r2)) + z / (2 * one_r2)


def train_net(tr, va, base_tr, base_va, seed, epochs=40, patience=5, bs=512, lr=2e-3, net_cls=None):
    torch.manual_seed(seed); np.random.seed(seed)
    flat = tr['seq'].reshape(-1, tr['seq'].shape[-1])
    # features that are (near) constant in a small training set are left unscaled
    floor = lambda s: np.where(s < 1e-3, 1.0, s)
    mean_s, std_s = flat.mean(0), floor(flat.std(0))
    mean_c, std_c = tr['static'].mean(0), floor(tr['static'].std(0))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype='float32')).to(dev)
    norm = lambda s: (T((s['seq'] - mean_s) / std_s), T((s['static'] - mean_c) / std_c))
    Xs, Xc = norm(tr); Vs, Vc = norm(va)
    Y, VY = T(tr['y']), T(va['y'])
    Bm, Bs = T(base_tr[0]), T(base_tr[1]); VBm, VBs = T(base_va[0]), T(base_va[1])

    net = (net_cls or Net)(Xs.shape[-1], Xc.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 50)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Y), device=dev)
        for i in range(0, len(Y), bs):
            j = perm[i:i + bs]
            loss = nll(Y[j], *gaussian_params(net(Xs[j], Xc[j]), Bm[j], Bs[j])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = torch.cat([nll(VY[i:i + 8192], *gaussian_params(
                net(Vs[i:i + 8192], Vc[i:i + 8192]), VBm[i:i + 8192], VBs[i:i + 8192])).mean(1)
                for i in range(0, len(VY), 8192)]).mean().item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(s, base):
        cs, cc = norm(s)
        bm, bsd = T(base[0]), T(base[1])
        outs = []
        with torch.no_grad():
            for i in range(0, len(cs), 8192):
                p_ = gaussian_params(net(cs[i:i + 8192], cc[i:i + 8192]), bm[i:i + 8192], bsd[i:i + 8192])
                outs.append([x.cpu().numpy() for x in p_])
        return tuple(np.concatenate(z) for z in zip(*outs))
    return predict, ep + 1, best


def mixture_to_gaussian(preds):
    """Moment-match an equal-weight ensemble of bivariate Gaussians."""
    mus = np.stack([p[0] for p in preds])
    mu = mus.mean(0)
    cov = np.zeros(mu.shape[:-1] + (2, 2))
    for m, sx, sy, r in preds:
        c = np.stack([np.stack([sx ** 2, r * sx * sy], -1), np.stack([r * sx * sy, sy ** 2], -1)], -2)
        dm = (m - mu)[..., :, None]
        cov += c + dm @ np.swapaxes(dm, -1, -2)
    cov /= len(preds)
    sx, sy = np.sqrt(cov[..., 0, 0]), np.sqrt(cov[..., 1, 1])
    return mu, sx, sy, cov[..., 0, 1] / (sx * sy)


# == Metrics ==
def metrics(s, pred):
    mu, sx, sy, rho = pred
    qlat, qlon = to_latlon(mu[..., 0] * 1000, mu[..., 1] * 1000, s['lat0'][:, None], s['lon0'][:, None])
    tlat, tlon = to_latlon(s['y'][..., 0] * 1000, s['y'][..., 1] * 1000, s['lat0'][:, None], s['lon0'][:, None])
    sep = haversine_km(qlat, qlon, tlat, tlon)
    ex, ey = (s['y'] - mu)[..., 0], (s['y'] - mu)[..., 1]
    one_r2 = 1 - rho ** 2
    d2 = ((ex / sx) ** 2 + (ey / sy) ** 2 - 2 * rho * ex * ey / (sx * sy)) / one_r2
    nll_ = np.log(2 * np.pi * sx * sy * np.sqrt(one_r2)) + d2 / 2
    area = np.pi * CHI2_90 * sx * sy * np.sqrt(one_r2)
    return dict(sep=sep, nll=nll_, cover=d2 <= CHI2_90, area=area)


def table(results, mask, title):
    hs = ''.join(f'{f"{h}h":>8}' for h in HORIZONS)
    lines = [f'\n{title}  (n={mask.sum():,})',
             f'{"model":<26}{"| mean separation (km)":<40}| med72  | NLL   | cov90@24/72 | area90@72 (km2)',
             f'{"":<26}{hs}']
    for name, m in results.items():
        line = f'{name:<26}' + ''.join(f'{m["sep"][mask, k].mean():8.2f}' for k in range(len(HORIZONS)))
        line += f'  {np.median(m["sep"][mask, -1]):6.2f}'
        if m.get('prob', True):
            line += (f'  {m["nll"][mask].mean():6.2f}  {m["cover"][mask, 2].mean():.3f}/{m["cover"][mask, -1].mean():.3f}'
                     f'  {np.median(m["area"][mask, -1]):10.0f}')
        lines.append(line)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = load(a.data_dir, years=PILOT_YEARS)
    starts, lens = segments(d)
    t0, _ = build_windows(d, starts, lens)
    test = d['time'][t0] >= TEST_START
    # validation: 20% of 2020-21 drifters (by ID) for early stopping
    ids = d['id'][t0]
    rng = np.random.default_rng(0)
    uid = np.unique(ids[~test])
    val_ids = set(rng.choice(uid, size=len(uid) // 5, replace=False))
    is_val = ~test & np.array([i in val_ids for i in ids])
    is_tr = ~test & ~is_val

    sub = lambda s, m: {k: v[m] for k, v in s.items()}
    S = make_samples(d, t0)
    tr, va, te = sub(S, is_tr), sub(S, is_val), sub(S, test)
    print(f'windows train {len(tr["y"]):,}  val {len(va["y"]):,}  test {len(te["y"]):,}')

    p = fit_ou(tr)
    print(f'OU prior: alpha={p["alpha"]}  T={p["T"]}h  sigma={np.sqrt(p["sigma2"]) / 3.6:.3f} m/s')
    pd_ = dict(p, alpha=0.0)
    damped_T = min([12, 24, 36, 48, 72, 96, 144], key=lambda T: np.linalg.norm(
        ou_mean(tr['u0'], tr['w48'], 0.0, T) - tr['y'], axis=-1).mean())
    pd_['T'] = damped_T

    results = {}
    results['damped CV'] = dict(metrics(te, ou_predict(te, pd_)), prob=False)
    results['OU prior'] = metrics(te, ou_predict(te, p))

    # direct LSTM: base mean 0, base sd = train RMS displacement per horizon (per component)
    rms = np.sqrt((tr['y'] ** 2).mean((0, 2)))
    base_direct = lambda s: (np.zeros_like(s['y']), np.broadcast_to(rms, s['y'].shape[:2]).copy())
    base_ou = lambda s: ou_predict(s, p)[:2]

    for name, base in [('LSTM direct', base_direct), ('LSTM + OU prior', base_ou)]:
        preds = []
        for seed in range(a.seeds):
            t_ = time.time()
            predict, n_ep, vbest = train_net(tr, va, base(tr), base(va), seed)
            preds.append(predict(te, base(te)))
            print(f'  {name} seed {seed}: {n_ep} epochs, val NLL {vbest:.3f}, {time.time() - t_:.0f}s')
        results[f'{name} (1 model)'] = metrics(te, preds[0])
        if a.seeds > 1:
            results[f'{name} (ens{a.seeds})'] = metrics(te, mixture_to_gaussian(preds))

    report = [table(results, np.ones(len(te['y']), bool), 'TEST 2022, all')]
    for g in ('drogued', 'undrogued'):
        report.append(table(results, te['group'] == g, f'TEST 2022, {g}'))
    for lo, hi in [(20, 30), (30, 38), (38, 45)]:
        report.append(table(results, (te['lat0'] >= lo) & (te['lat0'] < hi), f'TEST 2022, lat {lo}-{hi}N'))
    text = (f'OU prior fit: alpha={p["alpha"]}, T={p["T"]}h, sigma={np.sqrt(p["sigma2"]) / 3.6:.3f} m/s; '
            f'damped CV T={damped_T}h\n' + '\n'.join(report))
    print(text)
    (out_dir / 'pilot_ou_lstm_report.txt').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
