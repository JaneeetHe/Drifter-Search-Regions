"""Pilot feasibility checks on GDP hourly drifters (NW Pacific, 2020-01..2022-10).

T1  Leakage: is the product velocity (ve, vn) closer to backward / centered / forward
    position differences?  Does it forecast better than any past-only velocity?
T2  Physics prior: constant velocity vs 48h least-squares mean velocity vs
    mean velocity + inertial oscillation (frequency f = 2*Omega*sin(lat)).

Usage:  py -I pilot_t1_t2.py <data_dir> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file

R = 6_371_000.0
OMEGA = 7.2921e-5
HIST, FUT, STRIDE = 48, 72, 6            # hours
HORIZONS = [6, 12, 24, 48, 72]
MAX_GAP_S = 6 * 3600                     # drop windows whose fixes are >6h apart
TEST_START = np.datetime64('2022-01-01').astype('datetime64[s]').astype(float)
PILOT_YEARS = (2020, 2022)               # pilot data; 2015-2019 files in the same folder are ignored
TD_GRID = [12, 24, 48, 96, 192]          # damping e-folding times (h), tuned on train years


# == Data ==
# v2.01.1 location_type per drifter (ERDDAP's v2.01 wrongly flags every drifter as GPS)
LOCATION_TYPE_CSV = Path(__file__).resolve().parent.parent / 'data' / 'meta' / 'gdp_v2.01.1_location_type.csv'


def gps_ids():
    t = np.loadtxt(LOCATION_TYPE_CSV, delimiter=',', comments='#', skiprows=2, dtype=np.int64)
    return t[t[:, 1] == 1, 0]


def load(data_dir, gps_only=True, years=None):
    """Read all monthly files gdp1h_*.nc in data_dir.  gps_only keeps GPS-tracked drifters
    (per v2.01.1); years=(first, last) optionally restricts by observation year."""
    cols = {k: [] for k in ('id', 'time', 'lat', 'lon', 've', 'vn', 'gap', 'dld')}
    names = dict(time='time', lat='latitude', lon='longitude', ve='ve', vn='vn',
                 gap='gap', dld='drogue_lost_date')
    for f in sorted(Path(data_dir).glob('gdp1h_*.nc')):
        if years and not years[0] <= int(f.stem.split('_')[-1][:4]) <= years[1]:
            continue
        with netcdf_file(f, 'r', mmap=False) as nc:
            v = nc.variables
            raw = np.ascontiguousarray(v['ID'][:])
            cols['id'].append(raw.view(f'S{raw.shape[1]}').ravel().astype(np.int64))
            for k, n in names.items():
                cols[k].append(v[n][:].astype(float))
    d = {k: np.concatenate(x) for k, x in cols.items()}
    if gps_only:
        keep = np.isin(d['id'], gps_ids())
        d = {k: x[keep] for k, x in d.items()}
    o = np.lexsort((d['time'], d['id']))
    return {k: x[o] for k, x in d.items()}


def segments(d):
    """Start index and length of contiguous exactly-hourly runs of one drifter."""
    brk = np.r_[True, (d['id'][1:] != d['id'][:-1]) | (np.diff(d['time']) != 3600)]
    starts = np.flatnonzero(brk)
    return starts, np.diff(np.r_[starts, len(d['time'])])


def to_local(lat, lon, lat0, lon0):
    """Equirectangular projection (m) around the issue position."""
    x = R * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
    y = R * np.radians(lat - lat0)
    return x, y


def to_latlon(x, y, lat0, lon0):
    return lat0 + np.degrees(y / R), lon0 + np.degrees(x / (R * np.cos(np.radians(lat0))))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R / 1000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# == T1: velocity vs finite differences ==
def t1_velocity_check(d, starts, lens):
    seg_id = np.repeat(np.arange(len(starts)), lens)
    i = np.arange(1, len(d['time']) - 1)
    i = i[(seg_id[i - 1] == seg_id[i]) & (seg_id[i + 1] == seg_id[i])]   # interior points
    lat0, lon0 = d['lat'][i], d['lon'][i]
    xm, ym = to_local(d['lat'][i - 1], d['lon'][i - 1], lat0, lon0)
    xp, yp = to_local(d['lat'][i + 1], d['lon'][i + 1], lat0, lon0)
    diffs = {'backward': (-xm / 3600, -ym / 3600),
             'centered': ((xp - xm) / 7200, (yp - ym) / 7200),
             'forward': (xp / 3600, yp / 3600)}
    rows = []
    for name, (u, v) in diffs.items():
        rmsd = np.sqrt(np.mean((d['ve'][i] - u) ** 2 + (d['vn'][i] - v) ** 2))
        cu = np.corrcoef(d['ve'][i], u)[0, 1]
        cv = np.corrcoef(d['vn'][i], v)[0, 1]
        rows.append((name, cu, cv, rmsd))
    return rows, len(i)


# == T2: windows and forecasts ==
def build_windows(d, starts, lens):
    t0 = [s + np.arange(HIST, L - FUT, STRIDE) for s, L in zip(starts, lens) if L > HIST + FUT]
    t0 = np.concatenate(t0)
    # any fix gap > MAX_GAP_S inside [t0-HIST, t0+FUT]?  (prefix sum: O(n) memory)
    bad = np.r_[0, np.cumsum(d['gap'] > MAX_GAP_S)]
    ok = bad[t0 + FUT + 1] - bad[t0 - HIST] == 0
    return t0[ok], int((~ok).sum())


def fit_io(x, y, f_h, th):
    """LS fit z = c + W t + B exp(-i f t) to past positions (complex z = x + i y).
    Returns (cx, cy, wx, wy, br, bi) per window."""
    n, T = x.shape
    c, s = np.cos(f_h[:, None] * th), np.sin(f_h[:, None] * th)
    one, tt, zero = np.ones((n, T)), np.broadcast_to(th, (n, T)), np.zeros((n, T))
    # x = cx + wx t + br cos + bi sin ;  y = cy + wy t + bi cos - br sin
    Ax = np.stack([one, zero, tt, zero, c, s], -1)
    Ay = np.stack([zero, one, zero, tt, -s, c], -1)
    A = np.concatenate([Ax, Ay], 1)
    b = np.concatenate([x, y], 1)
    AtA = np.einsum('nti,ntj->nij', A, A) + 1e-6 * np.eye(6)
    Atb = np.einsum('nti,nt->ni', A, b)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]


def forecasts(d, t0):
    """Yield (name, px, py) with px, py of shape (n, FUT) for hours 1..FUT."""
    lat0, lon0 = d['lat'][t0], d['lon'][t0]
    H = t0[:, None] + np.arange(-HIST, 1)
    hx, hy = to_local(d['lat'][H], d['lon'][H], lat0[:, None], lon0[:, None])
    x1, y1 = to_local(d['lat'][t0 + 1], d['lon'][t0 + 1], lat0, lon0)
    tau = np.arange(1, FUT + 1)[None, :].astype(float)          # hours
    tau_s = tau * 3600

    cv = lambda u, v: (u[:, None] * tau_s, v[:, None] * tau_s)
    yield 'persistence', np.zeros((len(t0), FUT)), np.zeros((len(t0), FUT))
    yield 'CV product ve/vn (t0)', *cv(d['ve'][t0], d['vn'][t0])
    yield 'CV centered diff (uses t0+1h)', *cv((x1 - hx[:, -2]) / 7200, (y1 - hy[:, -2]) / 7200)
    # emulates the 6-hourly product velocity (centered difference over +/-6h, uses t0+6h)
    x6, y6 = to_local(d['lat'][t0 + 6], d['lon'][t0 + 6], lat0, lon0)
    yield 'CV centered 6h (uses t0+6h)', *cv((x6 - hx[:, -7]) / 43200, (y6 - hy[:, -7]) / 43200)
    yield 'CV backward 1h (past-only)', *cv(-hx[:, -2] / 3600, -hy[:, -2] / 3600)
    u3, v3 = -hx[:, -4] / 10800, -hy[:, -4] / 10800
    yield 'CV backward 3h (past-only)', *cv(u3, v3)
    for td in TD_GRID:
        k = td * 3600 * (1 - np.exp(-tau / td))
        yield f'damped CV back3h Td={td}h', u3[:, None] * k, v3[:, None] * k

    th = np.arange(-HIST, 1).astype(float)
    A = np.stack([np.ones_like(th), th], 1)
    P = np.linalg.pinv(A)                                         # (2, T)
    cx, wx = hx @ P.T[:, 0], hx @ P.T[:, 1]
    cy, wy = hy @ P.T[:, 0], hy @ P.T[:, 1]
    # anchor at the observed t0 position, extrapolate with the 48h LS mean velocity
    yield 'LS mean velocity 48h', wx[:, None] * tau, wy[:, None] * tau

    f_h = 2 * OMEGA * np.sin(np.radians(lat0)) * 3600           # rad/h
    p = fit_io(hx, hy, f_h, th)
    ccx, ccy, wwx, wwy, br, bi = p.T
    c, s = np.cos(f_h[:, None] * tau), np.sin(f_h[:, None] * tau)
    oscx, oscy = br[:, None] * c + bi[:, None] * s, bi[:, None] * c - br[:, None] * s
    osc0x, osc0y = br, bi                                         # oscillation at t=0
    for td in [None] + TD_GRID:
        damp = 1.0 if td is None else np.exp(-tau / td)
        # anchored at observed t0 position: z(tau) = z0 + W tau + (osc(tau)*damp - osc(0))
        px = wwx[:, None] * tau + oscx * damp - osc0x[:, None]
        py = wwy[:, None] * tau + oscy * damp - osc0y[:, None]
        yield ('mean + inertial osc.' if td is None else f'mean + inertial osc. damped Td={td}h'), px, py


def evaluate(d, t0, chunk=20000):
    """Separation (km) at HORIZONS and Liu-Weisberg skill (72h) per window, per model."""
    out = {}
    for a in range(0, len(t0), chunk):
        tc = t0[a:a + chunk]
        lat0, lon0 = d['lat'][tc], d['lon'][tc]
        F = tc[:, None] + np.arange(1, FUT + 1)
        tlat, tlon = d['lat'][F], d['lon'][F]
        # cumulative length of the observed path from t0 (for LW skill)
        plat = np.c_[lat0, tlat]; plon = np.c_[lon0, tlon]
        seg = haversine_km(plat[:, :-1], plon[:, :-1], plat[:, 1:], plon[:, 1:])
        l_cum = np.cumsum(seg, 1)
        for name, px, py in forecasts(d, tc):
            qlat, qlon = to_latlon(px, py, lat0[:, None], lon0[:, None])
            sep = haversine_km(qlat, qlon, tlat, tlon)
            ss = np.maximum(0, 1 - sep.sum(1) / np.maximum(l_cum.sum(1), 1e-9))
            rec = out.setdefault(name, {h: [] for h in HORIZONS} | {'skill72': []})
            for h in HORIZONS:
                rec[h].append(sep[:, h - 1])
            rec['skill72'].append(ss)
    return {m: {k: np.concatenate(v) for k, v in r.items()} for m, r in out.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir'); ap.add_argument('out_dir')
    ap.add_argument('--years', type=int, nargs=2, default=list(PILOT_YEARS))
    ap.add_argument('--test-start', default='2022-01-01')
    a = ap.parse_args()
    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    test_start = np.datetime64(a.test_start).astype('datetime64[s]').astype(float)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load(data_dir, years=tuple(a.years))
    starts, lens = segments(d)
    print(f'rows {len(d["time"]):,}  drifters {len(np.unique(d["id"]))}  segments {len(starts)}')

    # ---- T1
    rows, n = t1_velocity_check(d, starts, lens)
    print(f'\n[T1] product (ve,vn) vs position differences, {n:,} interior hourly points')
    print(f'{"difference":<10} {"corr u":>8} {"corr v":>8} {"RMSD m/s":>9}')
    for name, cu, cv_, r in rows:
        print(f'{name:<10} {cu:8.4f} {cv_:8.4f} {r:9.4f}')

    # ---- T2
    t0, n_drop = build_windows(d, starts, lens)
    test = d['time'][t0] >= test_start
    if (~test).sum() > 150_000:          # damping times are tuned on a random subset of training windows
        keep_tr = np.random.default_rng(0).choice(np.flatnonzero(~test), 150_000, replace=False)
        t0 = t0[np.sort(np.r_[keep_tr, np.flatnonzero(test)])]
        test = d['time'][t0] >= test_start
    dld = d['dld'][t0]
    drog = np.where(np.isnan(dld) | (dld <= 0), 'unknown',
                    np.where(d['time'][t0] < dld, 'drogued', 'undrogued'))
    lat0 = d['lat'][t0]
    print(f'\n[T2] windows: {len(t0):,} (dropped {n_drop:,} with fix gaps >6h); '
          f'train(2020-21) {(~test).sum():,}  test(2022) {test.sum():,}')
    res = evaluate(d, t0)

    # pick damping times on train years (mean 48h separation)
    def best(prefix):
        cands = [m for m in res if m.startswith(prefix)]
        return min(cands, key=lambda m: res[m][48][~test].mean())
    keep = ['persistence', 'CV product ve/vn (t0)', 'CV centered diff (uses t0+1h)',
            'CV centered 6h (uses t0+6h)', 'CV backward 1h (past-only)', 'CV backward 3h (past-only)',
            best('damped CV back3h'), 'LS mean velocity 48h',
            'mean + inertial osc.', best('mean + inertial osc. damped')]

    def table(mask, title):
        lines = [f'\n{title}  (n={mask.sum():,})',
                 f'{"model":<38}' + ''.join(f'{f"{h}h":>8}' for h in HORIZONS)
                 + f'{"med72":>8}{"LW72":>7}']
        for m in keep:
            r = res[m]
            lines.append(f'{m:<38}' + ''.join(f'{r[h][mask].mean():8.2f}' for h in HORIZONS)
                         + f'{np.median(r[72][mask]):8.2f}{r["skill72"][mask].mean():7.3f}')
        return '\n'.join(lines)

    report = [table(test, 'TEST 2022, all drifters — mean separation (km)')]
    for g in ('drogued', 'undrogued'):
        report.append(table(test & (drog == g), f'TEST 2022, {g}'))
    for lo, hi in [(20, 30), (30, 38), (38, 45)]:
        report.append(table(test & (lat0 >= lo) & (lat0 < hi), f'TEST 2022, lat {lo}-{hi}N'))
    report.append(table(~test, 'TRAIN 2020-2021 (sanity), all'))
    text = '\n'.join(report)
    print(text)

    with open(out_dir / 'pilot_t1_t2_report.txt', 'w', encoding='utf-8') as f:
        f.write('[T1]\n' + '\n'.join(f'{a:<10} {b:.4f} {c:.4f} {e:.4f}' for a, b, c, e in rows))
        f.write('\n\n[T2]\n' + text + '\n')
    np.savez_compressed(out_dir / 'pilot_t2_errors.npz', t0_time=d['time'][t0], lat0=lat0,
                        drogue=drog, test=test,
                        **{f'{m}|{k}': v for m, r in res.items() if m in keep for k, v in r.items()})


if __name__ == '__main__':
    main()
