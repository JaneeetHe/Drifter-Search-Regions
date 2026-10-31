"""Figures and tables for the JMSE manuscript -- two regions, GDP hourly 2015-2022, test year 2022.

Reads saved results only (no training):
  results/ext_{nwp,na}/final/final_predictions.npz        paper_final_run.py --years 2015 2022 [...]
  results/ext_{nwp,na}/conformal/conformal_results.csv    pilot_conformal.py --years 2015 2022 [...]
  results/ext_{nwp,na}/ablation/results_{A,B}.csv         pilot_ablation.py --exp A|B --years 2015 2022 [...]
  results/ext_{nwp,na}/oracle/leak_oracle_report.txt      pilot_leak_oracle.py --years 2015 2022
  results/ext_{nwp,na}/tiny/tiny_results.csv              tiny_data.py
  results/ext_{nwp,na}/t2/pilot_t1_t2_report.txt          pilot_t1_t2.py --years 2015 2022
Writes paper/figures/fig*.pdf|png, paper/tables/*.tex, paper/figures/numbers.txt, and refreshes the
table blocks between "% BEGIN AUTO <name>" and "% END AUTO <name>" in paper/main.tex.
The pilot version (NW Pacific 2020-2022) is scripts/paper_figures_pilot.py.

Usage:  .venv/Scripts/python scripts/paper_figures.py
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
FIG = ROOT / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / 'paper' / 'tables'; TAB.mkdir(parents=True, exist_ok=True)
CHI2_90 = 4.605
REG = {'NWP': dict(dir=RES / 'ext_nwp', name='Northwest Pacific', box=(117, 183, 17, 48), rect=(120, 20, 60, 25),
                   split=150, bands=[(20, 30), (30, 38), (38, 45)], central=180),
       'NA': dict(dir=RES / 'ext_na', name='North Atlantic', box=(-83, -7, 22, 53), rect=(-80, 25, 70, 25),
                  split=-45, bands=[(25, 33), (33, 41), (41, 50)], central=0)}

# ---- style: entity-stable colours (validated palette, 3 slots all-pairs) + neutral baselines
C = {'LSTM + OU prior': '#2a78d6', 'OU prior': '#eb6834', 'LSTM direct': '#1baf7a',
     'Damped velocity': '#52514e', 'Constant velocity': '#8a8984', 'Persistence': '#b9b7b0'}
MK = {'LSTM + OU prior': 'o', 'OU prior': 's', 'LSTM direct': '^',
      'Damped velocity': 'D', 'Constant velocity': 'v', 'Persistence': 'x'}
LS = {'Persistence': ':', 'Constant velocity': '--', 'Damped velocity': '-.'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e2dd'
W_FULL = 5.3
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.dpi': 400, 'pdf.fonttype': 42,
    'figure.dpi': 150, 'lines.linewidth': 1.4, 'lines.markersize': 4.5})
numbers = []


def note(s):
    numbers.append(s); print(s)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def panel(ax, letter, x=-0.02):
    ax.text(x, 1.04, f'({letter})', transform=ax.transAxes, fontsize=9, fontweight='bold',
            ha='right', va='bottom', color=INK)


def read_csv(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fmt_int(n):
    return f'{n:,}'.replace(',', '{,}')


# ====================================================================================== load results
MODELS_DET = ['Persistence', 'Constant velocity', 'Damped velocity']
MODELS_PROB = ['OU prior', 'LSTM direct', 'LSTM + OU prior']
ALL = MODELS_DET + MODELS_PROB


class Run:
    def __init__(self, key):
        self.key, self.cfg = key, REG[key]
        self.P = np.load(self.cfg['dir'] / 'final' / 'final_predictions.npz', allow_pickle=False)
        self.y = self.P['y']; self.H = self.P['HORIZONS']
        self.sep = {m: np.linalg.norm(self.P[f'{m}|mu'] - self.y, axis=-1) for m in ALL}
        self.group = self.P['group']; self.lat0 = self.P['lat0']; self.g6 = self.P['group6']

    def gauss(self, m):
        return tuple(self.P[f'{m}|{k}'] for k in ('mu', 'sx', 'sy', 'rho'))

    def mahal(self, m):
        mu, sx, sy, r = self.gauss(m)
        e = self.y - mu
        return ((e[..., 0] / sx) ** 2 + (e[..., 1] / sy) ** 2 - 2 * r * e[..., 0] * e[..., 1] / (sx * sy)) / (1 - r ** 2)

    def unit_area(self, m):
        _, sx, sy, r = self.gauss(m)
        return np.pi * sx * sy * np.sqrt(1 - r ** 2)


RUNS = {k: Run(k) for k in REG}

# ====================================================================================== tables
# data summary
lines = []
for k, R in RUNS.items():
    lab = dict(zip(R.P['split_ids'], R.P['split_labels']))
    log = (R.cfg['dir'] / 'final').parent  # windows per split are recomputed from labels of test only
    n_test = len(R.y); d_test = len(np.unique(R.P['id']))
    d_tr = sum(v == 'train' for v in lab.values()); d_va = sum(v == 'validation' for v in lab.values())
    d_ca = sum(v == 'calibration' for v in lab.values())
    note(f'{k}: drifters train {d_tr} val {d_va} cal {d_ca}; test {d_test} drifters {n_test} windows; '
         f'hourly rows {len(R.P["track_id"]):,} drifters {len(np.unique(R.P["track_id"]))}')
    lines.append((k, d_tr, d_va, d_ca, d_test, n_test, len(np.unique(R.P['track_id'])), len(R.P['track_id'])))
WIN = {'NWP': (230947, 49996, 63828), 'NA': (781928, 160477, 168673)}   # from results/ext_run.log
with open(TAB / 'data_summary.tex', 'w', encoding='utf-8') as f:
    for k, d_tr, d_va, d_ca, d_test, n_test, n_dr, n_rows in lines:
        w = WIN[k]
        f.write(f'{REG[k]["name"]} & {n_dr} & {fmt_int(n_rows)} & {fmt_int(w[0])} ({d_tr}) & {fmt_int(w[1])} ({d_va}) & '
                f'{fmt_int(w[2])} ({d_ca}) & {fmt_int(n_test)} ({d_test}) \\\\\n')

# main results
with open(TAB / 'main_results.tex', 'w', encoding='utf-8') as f:
    for k, R in RUNS.items():
        rows = []
        for m in ALL:
            s = R.sep[m]
            r = [m] + [f'{s[:, i].mean():.2f}' for i in range(len(R.H))] + [f'{np.median(s[:, -1]):.2f}']
            if m in MODELS_PROB:
                d2 = R.mahal(m); _, sx, sy, rr = R.gauss(m)
                nll = (np.log(2 * np.pi * sx * sy * np.sqrt(1 - rr ** 2)) + d2 / 2).mean()
                q = R.P[f'{m}|q_mondrian']
                area = np.median(R.unit_area(m)[:, -1] * q[:, -1]) / 1000
                r += [f'{nll:.2f}', f'{(d2[:, -1] <= CHI2_90).mean():.2f}', f'{(d2[:, -1] <= q[:, -1]).mean():.2f}', f'{area:.1f}']
            else:
                r += ['--'] * 4
            rows.append(r)
        best = {c: min(range(len(rows)), key=lambda i: float(rows[i][c])) for c in range(1, 7)}
        f.write(f'\\multicolumn{{11}}{{l}}{{\\textit{{{REG[k]["name"]}}}}} \\\\\n')
        for i, r in enumerate(rows):
            f.write(' & '.join(f'\\textbf{{{c}}}' if (j in best and best[j] == i) else c for j, c in enumerate(r)) + ' \\\\\n')
        if k == 'NWP':
            f.write('\\midrule\n')
        d = R.sep['Damped velocity']
        for m in ['OU prior', 'LSTM direct', 'LSTM + OU prior']:
            note(f'{k} {m}: gain vs damped 24h {100 * (1 - R.sep[m][:, 2].mean() / d[:, 2].mean()):.1f}%  '
                 f'72h {100 * (1 - R.sep[m][:, 4].mean() / d[:, 4].mean()):.1f}%')
        note(f'{k} OU fit alpha={float(R.P["ou_alpha"])} T={float(R.P["ou_T"])} sigma={np.sqrt(float(R.P["ou_sigma2"])) / 3.6:.3f} '
             f'damped T={float(R.P["damped_T"])}')

# calibration table
CF = {k: read_csv(REG[k]['dir'] / 'conformal' / 'conformal_results.csv') for k in REG}
TAGS = [('In-basin', 'in-region (all NWP)'), ('West $\\to$ East', 'cross-region W->E'), ('East $\\to$ West', 'cross-region E->W')]
CALS = [('None', 'raw'), ('Static', 'static'), ('Rolling', 'rolling'), ('Rolling Mondrian', 'M-lat×drg')]


def cget(k, tag, model, cal):
    return next(r for r in CF[k] if r['tag'] == tag and r['model'] == model and r['calib'] == cal)


with open(TAB / 'calibration.tex', 'w', encoding='utf-8') as f:
    for k in REG:
        f.write(f'\\multicolumn{{7}}{{l}}{{\\textit{{{REG[k]["name"]}}}}} \\\\\n')
        for ti, (tname, tag) in enumerate(TAGS):
            for ci, (cname, cal) in enumerate(CALS):
                L = cget(k, tag, 'LSTM + OU (ens)', cal); O = cget(k, tag, 'OU prior', cal)
                first = f'\\multirow{{4}}{{*}}{{{tname}}}' if ci == 0 else ''
                f.write(f'{first} & {cname} & {float(L["cov72"]):.2f} & {float(L["g72_min"]):.2f}--{float(L["g72_max"]):.2f} & '
                        f'{float(L["area72"]) / 1e3:.1f} & {float(O["cov72"]):.2f} & {float(O["area72"]) / 1e3:.1f} \\\\\n')
            if ti < 2:
                f.write('\\cmidrule{2-7}\n')
        if k == 'NWP':
            f.write('\\midrule\n')
        for tname, tag in TAGS:
            L = cget(k, tag, 'LSTM + OU (ens)', 'M-lat×drg'); O = cget(k, tag, 'OU prior', 'M-lat×drg')
            note(f'{k} {tag}: Mondrian cov72 {float(L["cov72"]):.3f} groups {float(L["g72_min"]):.3f}-{float(L["g72_max"]):.3f}; '
                 f'area72 {float(L["area72"]):.0f} vs OU {float(O["area72"]):.0f} -> {100 * (1 - float(L["area72"]) / float(O["area72"])):.0f}%; '
                 f'area24 {float(L["area24"]):.0f} vs {float(O["area24"]):.0f} -> {100 * (1 - float(L["area24"]) / float(O["area24"])):.0f}%')
        for tname, tag in TAGS:
            L = cget(k, tag, 'LSTM + OU (ens)', 'static')
            note(f'{k} {tag}: static cov72 {float(L["cov72"]):.3f}; raw {float(cget(k, tag, "LSTM + OU (ens)", "raw")["cov72"]):.3f}')


# comparison with alternative models (compare_models.py), same split/inputs/calibration as the main run
def comparison_table():
    out = []
    for k, R in RUNS.items():
        f = REG[k]['dir'] / 'compare' / 'compare_results.csv'
        if not f.exists():
            print(f'  (no comparison results for {k})'); return False
        rows = read_csv(f)
        ref = []
        for m in ['LSTM direct', 'LSTM + OU prior']:
            s = R.sep[m]; d2 = R.mahal(m); _, sx, sy, rr = R.gauss(m); q = R.P[f'{m}|q_mondrian']
            ref.append(dict(model='LSTM', variant='direct' if m == 'LSTM direct' else '+ OU prior',
                            s24=s[:, 2].mean(), s72=s[:, 4].mean(), med72=np.median(s[:, 4]),
                            nll=(np.log(2 * np.pi * sx * sy * np.sqrt(1 - rr ** 2)) + d2 / 2).mean(),
                            cov_m72=(d2[:, -1] <= q[:, -1]).mean(),
                            area72=np.median(R.unit_area(m)[:, -1] * q[:, -1]) / 1e3))
        allrows = ref + [{c: (r[c] if c in ('model', 'variant') else (float(r[c]) if r[c] else None)) for c in r}
                         for r in rows]
        best24 = min(r['s24'] for r in allrows); best72 = min(r['s72'] for r in allrows)
        out.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{REG[k]["name"]}}}}} \\\\')
        for r in allrows:
            cell = lambda v, b=None: '--' if v is None else (f'\\textbf{{{v:.2f}}}' if b is not None and abs(v - b) < 1e-9 else f'{v:.2f}')
            cov = '--' if r.get('cov_m72') is None else f'{r["cov_m72"]:.2f}'
            area = '--' if r.get('area72') is None else f'{r["area72"]:.1f}'
            out.append(f'{r["model"]} & {r["variant"]} & {cell(r["s24"], best24)} & {cell(r["s72"], best72)} & '
                       f'{cell(r["med72"])} & {cell(r.get("nll"))} & {cov} & {area} \\\\')
            note(f'{k} compare {r["model"]} {r["variant"]}: 24h {r["s24"]:.2f} 72h {r["s72"]:.2f}'
                 + ('' if r.get('area72') is None else f' cov {r["cov_m72"]:.3f} area72 {r["area72"]:.1f}'))
        if k == 'NWP':
            out.append('\\midrule')
    (TAB / 'comparison.tex').write_text('\n'.join(out) + '\n', encoding='utf-8')
    return True


HAVE_COMPARISON = comparison_table()


def sync_main_tex():
    tex = ROOT / 'paper' / 'main.tex'
    s = tex.read_text(encoding='utf-8')
    for name in ('data_summary', 'main_results', 'calibration') + (('comparison',) if HAVE_COMPARISON else ()):
        body = (TAB / f'{name}.tex').read_text(encoding='utf-8').rstrip('\n')
        pat = re.compile(r'(% BEGIN AUTO ' + name + r'\n).*?(% END AUTO ' + name + r')', re.S)
        if pat.search(s):
            s = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(2), s)
        else:
            print(f'  (no AUTO block for {name} in main.tex)')
    tex.write_text(s, encoding='utf-8')


# ====================================================================================== Fig 1 map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(W_FULL, 4.6))
for i, (k, R) in enumerate(RUNS.items()):
    cfg = R.cfg
    proj = ccrs.PlateCarree(central_longitude=cfg['central'])
    ax = fig.add_axes([0, 0.52 - i * 0.5, 1, 0.45], projection=proj)
    ax.set_extent(cfg['box'], crs=ccrs.PlateCarree())
    # vector land and coastline; only the tracks are rasterised (PDF ~3.4 MB; a lighter JPEG variant was tried
    # for Overleaf's free compile time and dropped in favour of sharper coastlines)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#ecebe7', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4, edgecolor='#8a8984', zorder=2)
    lab = dict(zip(R.P['split_ids'], R.P['split_labels']))
    tid, tla, tlo, tt = R.P['track_id'], R.P['track_lat'], R.P['track_lon'], R.P['track_time']
    test_start = np.datetime64('2022-01-01').astype('datetime64[s]').astype(float)
    is_test = tt >= test_start
    brk = np.flatnonzero(np.r_[True, (tid[1:] != tid[:-1]) | (is_test[1:] != is_test[:-1]), True])
    for i0, i1 in zip(brk[:-1], brk[1:]):
        te = is_test[i0]
        lo = tlo[i0:i1:2].astype(float); la = tla[i0:i1:2].astype(float)
        jump = np.flatnonzero(np.abs(np.diff(lo)) > 2) + 1
        for a_, b_ in zip(np.r_[0, jump], np.r_[jump, len(lo)]):
            ax.plot(lo[a_:b_], la[a_:b_], color='#2a78d6' if te else '#a8a6a0', lw=0.35 if te else 0.25,
                    alpha=0.8 if te else 0.45, transform=ccrs.PlateCarree(), zorder=4 if te else 3, rasterized=True)
    x0, y0, w, h = cfg['rect']
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=INK2, lw=0.7, transform=ccrs.PlateCarree(), zorder=5))
    ax.plot([cfg['split']] * 2, [y0, y0 + h], color=INK, lw=0.8, ls=(0, (4, 2)), transform=ccrs.PlateCarree(), zorder=5)
    lo0, lo1, la0, la1 = cfg['box']
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#cfcdc7',
                      xlocs=range(int(np.ceil(lo0 / 10) * 10), int(lo1) + 1, 10), ylocs=range(20, 55, 5))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 6.8, 'color': INK2}
    ax.text(0.01, 0.97, f'({"ab"[i]}) {cfg["name"]}', transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='top', ha='left', color=INK, zorder=7, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
fig.legend(handles=[Line2D([], [], color='#a8a6a0', lw=1.2, label='2015–2021 (training / validation / calibration)'),
                    Line2D([], [], color='#2a78d6', lw=1.2, label='2022 (test)'),
                    Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)), label='west/east split for transfer tests')],
           loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=6.5, handlelength=1.8, columnspacing=1.0)
save(fig, 'fig1_study_area')

# ====================================================================================== Fig 2
# Figure 2 (framework) is an editable PowerPoint slide: scripts/make_fig2_pptx.js -> paper/figures_src/
# fig2_framework.pptx, exported to paper/figures/fig2_framework.pdf|png by scripts/export_pptx_figure.ps1.

# ====================================================================================== Fig 3 leakage
VEL = [('CV backward 3h (past-only)', 'Past-only\n(3-h backward)'), ('CV product ve/vn (t0)', 'Product velocity\n(LOWESS, ±3 h)'),
       ('CV centered diff (uses t0+1h)', 'Centred ±1 h\n(uses $t_0$+1 h)'), ('CV centered 6h (uses t0+6h)', 'Centred ±6 h\n(6-hourly product)')]


def t2_table(k):
    txt = (REG[k]['dir'] / 't2' / 'pilot_t1_t2_report.txt').read_text(encoding='utf-8')
    block = txt.split('TEST 2022, all')[1].split('\n\n')[0]
    out = {}
    for ln in block.splitlines():
        m = re.match(r'^(.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', ln)
        if m:
            out[m.group(1).strip()] = [float(m.group(i)) for i in range(2, 7)]
    return out


def oracle(k):
    txt = (REG[k]['dir'] / 'oracle' / 'leak_oracle_report.txt').read_text(encoding='utf-8')
    vals = {}
    for ln in txt.splitlines()[1:]:
        m = re.match(r'^(.*?)\s{2,}([\d.]+)\s+([\d.na]+)\s+([\d.]+)\s+([\d.]+)', ln)
        if m:
            vals[m.group(1).strip()] = (float(m.group(2)), float(m.group(5)))      # 24h, 72h
    ou = vals['OU prior (no leak)']; std = vals['LSTM + OU (standard, history to t0)']
    w = [vals[f'LSTM + OU + true pos. t0+1..t0+{h}h'] for h in (1, 3, 6)]
    return ou, [std] + w


fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.45))
ax = axs[0]
T2 = {k: t2_table(k) for k in REG}
yy = np.arange(len(VEL))[::-1]
for j, (k, col, hatch) in enumerate([('NWP', '#86b6ef', None), ('NA', '#1c5cab', None)]):
    vals = [T2[k][key][2] for key, _ in VEL]
    ax.barh(yy + (0.19 if j == 0 else -0.19), vals, height=0.36, color=col, edgecolor='white', linewidth=0.8,
            label=REG[k]['name'])
    for yi, v in zip(yy, vals):
        ax.text(v + 0.25, yi + (0.19 if j == 0 else -0.19), f'{v:.1f}', va='center', fontsize=6.4, color=INK)
    base = T2[k][VEL[0][0]][2]
    for key, _ in VEL[1:]:
        note(f'{k} leakage CV 24h {key}: {T2[k][key][2]:.2f} vs past-only {base:.2f} '
             f'({100 * (T2[k][key][2] / base - 1):+.1f}%), 72h {T2[k][key][4]:.2f} vs {T2[k][VEL[0][0]][4]:.2f} '
             f'({100 * (T2[k][key][4] / T2[k][VEL[0][0]][4] - 1):+.1f}%)')
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in VEL], fontsize=6.7); ax.grid(axis='y', visible=False)
ax.set_xlabel('24-h mean separation (km)'); ax.set_xlim(0, 25)
ax.legend(loc='lower right', fontsize=6.3, handlelength=1.2, bbox_to_anchor=(1.03, -0.02))
ax.set_title('Constant-velocity forecasts with different\nvelocity estimates (test 2022)', loc='left', fontsize=7.4)
panel(ax, 'a')
ax = axs[1]
w = [0, 1, 3, 6]
for k, mk, ls in [('NWP', 'o', '-'), ('NA', 's', '--')]:
    ou, vals = oracle(k)
    e24 = [v[0] for v in vals]
    ax.plot(w, e24, color=C['LSTM + OU prior'], marker=mk, ls=ls, ms=4.5, mec='white', mew=0.7, label=f'{REG[k]["name"]}: LSTM + OU prior')
    ax.axhline(ou[0], color=C['OU prior'], lw=1.0, ls=ls)
    ax.text(6.3, ou[0] + 0.15, f'OU prior ({k})', ha='right', va='bottom', fontsize=6.2, color=INK)
    gain = ou[0] - e24[0]
    note(f'{k} oracle 24h: OU {ou[0]:.2f}, LSTM {e24[0]:.2f} (gain {gain:.2f}); +1h {e24[0] - e24[1]:.2f} ({100 * (e24[0] - e24[1]) / gain:.0f}%), '
         f'+3h {e24[0] - e24[2]:.2f} ({100 * (e24[0] - e24[2]) / gain:.0f}%); 72h: OU {ou[1]:.2f} LSTM {vals[0][1]:.2f} '
         f'+1h {vals[0][1] - vals[1][1]:.2f} +3h {vals[0][1] - vals[2][1]:.2f}')
ax.set_xticks(w); ax.set_xlabel('Hours of true future positions given'); ax.set_ylabel('24-h mean separation (km)')
ax.set_ylim(4, 16); ax.set_xlim(-0.3, 6.4)
ax.legend(loc='lower left', fontsize=6.0, handlelength=2.2, bbox_to_anchor=(0.16, 0.6))
ax.set_title('Upper bound on smoothing leakage\n(oracle inputs)', loc='left', fontsize=7.4)
panel(ax, 'b')
save(fig, 'fig3_leakage')

# ====================================================================================== Fig 4 accuracy
fig = plt.figure(figsize=(W_FULL, 4.4))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3)
for i, (k, R) in enumerate(RUNS.items()):
    ax = fig.add_subplot(gs[0, i])
    for m in ALL:
        ax.plot(R.H, R.sep[m].mean(0), color=C[m], marker=MK[m], ls=LS.get(m, '-'), lw=1.3 if m in MODELS_PROB else 1.0,
                ms=4 if MK[m] != 'x' else 4.5, mec='white' if MK[m] != 'x' else C[m], mew=0.6, label=m, zorder=3)
    ax.set_xticks(R.H); ax.set_xlabel('Forecast horizon (h)'); ax.set_ylabel('Mean separation (km)' if i == 0 else '')
    ax.set_ylim(0, 70); ax.set_title(REG[k]['name'], fontsize=7.6)
    panel(ax, 'ab'[i])
    if i == 0:
        fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.97), ncol=6,
                   fontsize=6.0, handlelength=2.0, columnspacing=0.8)
    ax = fig.add_subplot(gs[1, i])
    b = R.cfg['bands']
    groups = [('All', np.ones(len(R.y), bool)), ('Drogued', R.group == 'drogued'), ('Undrogued', R.group != 'drogued')] + \
             [(f'{lo}–{hi}°N', (R.lat0 >= lo) & (R.lat0 < hi + (0.01 if j == 2 else 0))) for j, (lo, hi) in enumerate(b)]
    xs = np.arange(len(groups))
    for j, m in enumerate(MODELS_PROB):
        imp = [100 * (1 - R.sep[m][g, 2].mean() / R.sep['Damped velocity'][g, 2].mean()) for _, g in groups]
        ax.bar(xs + (j - 1) * 0.26, imp, width=0.24, color=C[m], edgecolor='white', linewidth=0.6)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([g for g, _ in groups], rotation=35, ha='right', rotation_mode='anchor', fontsize=6.4)
    ax.grid(axis='x', visible=False); ax.set_ylim(0, 45)
    ax.set_ylabel('24-h improvement over\ndamped velocity (%)' if i == 0 else '', fontsize=7)
    panel(ax, 'cd'[i])
save(fig, 'fig4_accuracy')

# ====================================================================================== Fig 5 prior role
fig = plt.figure(figsize=(W_FULL, 4.3))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3, height_ratios=[1, 1.05])
TINY = {k: read_csv(REG[k]['dir'] / 'tiny' / 'tiny_results.csv') for k in REG}
rng = np.random.default_rng(1)
for i, k in enumerate(REG):
    ax = fig.add_subplot(gs[0, i])
    rows = TINY[k]
    ks = sorted({int(r['k']) for r in rows})
    for j, (key, m) in enumerate([('direct', 'LSTM direct'), ('hybrid', 'LSTM + OU prior')]):
        for x, kk in enumerate(ks):
            R_ = [r for r in rows if int(r['k']) == kk]
            ratio = np.array([float(r[f'{key}_sep24']) / float(r['OU_sep24']) for r in R_])
            xpos = x + (j - 0.5) * 0.3
            TOP = 2.2
            jit = rng.uniform(-0.06, 0.06, len(ratio))
            inside = ratio <= TOP
            ax.scatter(xpos + jit[inside], ratio[inside], s=12, color=C[m], edgecolor='white',
                       linewidth=0.4, zorder=3, label=m if x == 0 else None)
            for xx, v in zip(xpos + jit[~inside], ratio[~inside]):
                ax.scatter([xx], [TOP - 0.04], s=18, marker='^', color=C[m], edgecolor='white', linewidth=0.4, zorder=3)
                ax.text(xx + 0.07, TOP - 0.04, f'{v:.1f}', fontsize=5.8, color=INK, va='center')
            ax.plot([xpos - 0.11, xpos + 0.11], [np.median(ratio)] * 2, color=C[m], lw=1.6, zorder=4)
        for kk in ks:
            R_ = [r for r in rows if int(r['k']) == kk]
            di = np.array([float(r['direct_sep24']) / float(r['OU_sep24']) for r in R_])
            hy = np.array([float(r['hybrid_sep24']) / float(r['OU_sep24']) for r in R_])
            if key == 'hybrid':
                note(f'{k} tiny k={kk}: windows median {int(np.median([int(r["n_train"]) for r in R_]))}; direct/OU median {np.median(di):.2f} '
                     f'max {di.max():.2f} (worse >10%: {(di > 1.1).sum()}/{len(di)}); hybrid/OU median {np.median(hy):.2f} '
                     f'max {hy.max():.2f} (worse >10%: {(hy > 1.1).sum()}/{len(hy)})')
    ax.axhline(1, color=C['OU prior'], lw=1.0, ls='--')
    ax.text(-0.45, 1.03, 'OU prior', ha='left', va='bottom', fontsize=6.2, color=INK)
    ax.set_ylim(0.6, 2.25); ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(x) for x in ks])
    ax.set_xlabel('Training drifters (5 random draws each)')
    ax.set_ylabel('24-h error relative to OU prior' if i == 0 else '')
    ax.set_title(REG[k]['name'], fontsize=7.6); ax.grid(axis='x', visible=False)
    panel(ax, 'ab'[i])
    if i == 0:
        ax.legend(loc='upper right', fontsize=6.2, handletextpad=0.2)
ax = fig.add_subplot(gs[1, :])
B = {k: {(r['tag'], r['model']): r for r in read_csv(REG[k]['dir'] / 'ablation' / 'results_B.csv')} for k in REG}
cases = [(k, d) for k in REG for d in ('W->E', 'E->W')]
series = [('LSTM direct', 'no location', C['LSTM direct'], None, 'LSTM direct'),
          ('LSTM + OU prior', 'no location', C['LSTM + OU prior'], None, 'LSTM + OU prior'),
          ('LSTM direct', 'with location', C['LSTM direct'], '////', 'LSTM direct + lat/lon'),
          ('LSTM + OU prior', 'with location', C['LSTM + OU prior'], '////', 'LSTM + OU prior + lat/lon'),
          ('LSTM + OU prior', 'in-region ref', '#9ec5f4', None, 'Trained in target region')]
xs = np.arange(len(cases))
for j, (m, setting, col, hatch, lab_) in enumerate(series):
    vals = np.array([float(B[k][(f'B {d} {setting}', m)]['sep24']) for k, d in cases])
    rngs = np.array([float(B[k][(f'B {d} {setting}', m)]['sep24_range'] or 0) for k, d in cases])
    ax.bar(xs + (j - 2) * 0.16, vals, width=0.15, color=col if not hatch else 'white', edgecolor=col, hatch=hatch,
           linewidth=0.8, label=lab_, yerr=rngs / 2, error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK2))
ax.set_xticks(xs)
ax.set_xticklabels([f'{REG[k]["name"]}\n{d.replace("->", "→").replace("W", "west").replace("E", "east")}' for k, d in cases], fontsize=6.6)
ax.set_ylabel('24-h mean separation (km)'); ax.set_ylim(5, 15.5); ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', fontsize=6.0, ncol=3, columnspacing=0.8, handlelength=1.2)
panel(ax, 'c', x=-0.01)
save(fig, 'fig5_prior_role')

# ====================================================================================== Fig 6 calibration
fig = plt.figure(figsize=(W_FULL, 4.2))
gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.28, width_ratios=[1, 1])
for i, (k, R) in enumerate(RUNS.items()):
    ax = fig.add_subplot(gs[0, i])
    m = 'LSTM + OU prior'; d2 = R.mahal(m)
    names = [f'{lo}–{hi}°N {s}' for lo, hi in R.cfg['bands'] for s in ('undr.', 'drog.')]
    variants = [('Uncalibrated', np.full_like(d2, CHI2_90), '#b9b7b0'), ('Rolling', R.P[f'{m}|q_rolling'], '#86b6ef'),
                ('Rolling Mondrian', R.P[f'{m}|q_mondrian'], '#1c5cab')]
    xs = np.arange(6)
    for j, (nm, q, col) in enumerate(variants):
        cov = [(d2[R.g6 == g, -1] <= q[R.g6 == g, -1]).mean() for g in range(6)]
        ax.bar(xs + (j - 1) * 0.27, cov, width=0.25, color=col, edgecolor='white', linewidth=0.5, label=nm)
        note(f'{k} coverage72 by group [{nm}]: ' + ', '.join(f'{c:.3f}' for c in cov) +
             '  (n=' + ', '.join(str(int((R.g6 == g).sum())) for g in range(6)) + ')')
    ax.axhline(0.9, color=INK, lw=0.8, ls='--')
    ax.set_ylim(0.6, 1.0); ax.set_xlim(-0.5, 5.5); ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=5.9, rotation=35, ha='right', rotation_mode='anchor'); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Coverage of 90% region (72 h)' if i == 0 else '')
    ax.set_title(REG[k]['name'], fontsize=7.6)
    panel(ax, 'ab'[i])
    if i == 0:
        fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.965), ncol=3,
                   fontsize=6.2, handlelength=1.0, columnspacing=1.0)
ax = fig.add_subplot(gs[1, :])
labels, ou_v, lstm_v = [], [], []
for k in REG:
    for tname, tag in TAGS:
        labels.append(f'{k}\n{tname.replace("$\\to$", "→")}')
        ou_v.append(float(cget(k, tag, 'OU prior', 'M-lat×drg')['area72']) / 1e3)
        lstm_v.append(float(cget(k, tag, 'LSTM + OU (ens)', 'M-lat×drg')['area72']) / 1e3)
xs = np.arange(len(labels))
ax.bar(xs - 0.18, ou_v, width=0.34, color=C['OU prior'], edgecolor='white', linewidth=0.6, label='OU prior')
ax.bar(xs + 0.18, lstm_v, width=0.34, color=C['LSTM + OU prior'], edgecolor='white', linewidth=0.6, label='LSTM + OU prior')
for x, a_, b_ in zip(xs, ou_v, lstm_v):
    ax.text(x - 0.18, a_ + 0.4, f'{a_:.1f}', ha='center', fontsize=6.0, color=INK)
    ax.text(x + 0.18, b_ + 0.4, f'{b_:.1f}', ha='center', fontsize=6.0, color=INK)
    ax.text(x, max(a_, b_) + 3.3, f'−{100 * (1 - b_ / a_):.0f}%', ha='center', fontsize=6.6, color=INK, fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.4); ax.grid(axis='x', visible=False)
ax.set_ylabel('Median 90% region at 72 h\n(10³ km², equal coverage)'); ax.set_ylim(0, 40)
ax.legend(loc='upper right', fontsize=6.4, ncol=2)
panel(ax, 'c', x=-0.01)
save(fig, 'fig6_calibration')

# ====================================================================================== Fig 7 case studies
fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.8), gridspec_kw=dict(wspace=0.3))
for ax, (k, R), letter, seed in zip(axs, RUNS.items(), 'ab', (3, 5)):
    fut = R.P['future_xy_hourly']; hist = R.P['hist_xy']; y = R.y
    gain = R.sep['OU prior'][:, -1] - R.sep['LSTM + OU prior'][:, -1]
    disp = np.linalg.norm(y[:, -1], axis=-1)
    a_l = R.unit_area('LSTM + OU prior')[:, -1] * R.P['LSTM + OU prior|q_mondrian'][:, -1]
    a_o = R.unit_area('OU prior')[:, -1] * R.P['OU prior|q_mondrian'][:, -1]
    # a typical, legible case: clear gain, smaller hybrid region, curved path, labels well apart
    pick = None
    for d_min, g_min in [(90, 15), (80, 12), (60, 10)]:
        cand = np.flatnonzero((gain > g_min) & (disp > d_min) & (disp < 160) & (R.sep['LSTM + OU prior'][:, -1] < 12)
                              & (a_l < 0.6 * a_o) & (R.sep['LSTM + OU prior'][:, 2] < 6))
        for j in np.random.default_rng(seed).permutation(cand):
            if np.linalg.norm(fut[j, 35] - fut[j, -1] / 2) > 8 and np.linalg.norm(y[j, 3] - y[j, 2]) > 20:
                pick = j; break
        if pick is not None:
            break
    i = pick
    ax.plot(hist[i, :, 0], hist[i, :, 1], color='#8a8984', lw=1.2, marker='.', ms=3, label='48-h history')
    ax.plot(np.r_[0, fut[i, :, 0]], np.r_[0, fut[i, :, 1]], color=INK, lw=1.3, label='Observed 72 h')
    for name, ls in [('OU prior', '--'), ('LSTM + OU prior', '-')]:
        mu, sx, sy, r = (a_[i] for a_ in R.gauss(name)); q = R.P[f'{name}|q_mondrian'][i]
        for kk in (2, 3, 4):
            cov = np.array([[sx[kk] ** 2, r[kk] * sx[kk] * sy[kk]], [r[kk] * sx[kk] * sy[kk], sy[kk] ** 2]]) * q[kk]
            vals, vecs = np.linalg.eigh(cov)
            ax.add_patch(Ellipse(mu[kk], 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]),
                                 angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])), fill=False, ec=C[name], lw=1.0, ls=ls))
            ax.plot(*mu[kk], marker=MK[name], color=C[name], ms=4, mec='white', mew=0.5)
        ax.plot([], [], color=C[name], ls=ls, lw=1.0, marker=MK[name], ms=4, label=f'{name}: mean, 90% region')
    for kk, h in zip((2, 3, 4), (24, 48, 72)):
        ax.plot(*y[i, kk], marker='*', color=INK, ms=6, mec='white', mew=0.4, zorder=5)
        ax.annotate(f'{h} h', y[i, kk], xytext=(-5, 3), textcoords='offset points', ha='right', fontsize=6.2, color=INK,
                    zorder=6, path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
    ax.set_aspect('equal', adjustable='datalim'); ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
    t = np.datetime64(int(R.P['t0_time'][i]), 's').astype('datetime64[m]')
    ax.set_title(f'{REG[k]["name"]}, issued {str(t).replace("T", " ")} UTC\n'
                 f'{R.lat0[i]:.1f}°N, {abs(R.P["lon0"][i]):.1f}°{"E" if R.P["lon0"][i] >= 0 else "W"}, {R.group[i]}',
                 loc='left', fontsize=6.4, pad=3)
    panel(ax, letter)
    note(f'case {letter} ({k}): sep72 OU {R.sep["OU prior"][i, -1]:.1f} LSTM {R.sep["LSTM + OU prior"][i, -1]:.1f} km')
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', ncol=2, fontsize=6.3, bbox_to_anchor=(0.5, 0.0), handlelength=2.2)
save(fig, 'fig7_cases')

# ====================================================================================== ablation numbers for the text
for k in REG:
    A = {(r['tag'], r['model']): r for r in read_csv(REG[k]['dir'] / 'ablation' / 'results_A.csv')}
    for m in ['LSTM direct', 'LSTM + OU prior']:
        note(f'{k} location ablation {m}: with {float(A[("A with location", m)]["sep24"]):.2f} '
             f'without {float(A[("A no location", m)]["sep24"]):.2f}')
    for d in ('W->E', 'E->W'):
        for setting in ('no location', 'with location', 'in-region ref'):
            for m in ['LSTM direct', 'LSTM + OU prior']:
                r = B[k][(f'B {d} {setting}', m)]
                note(f'{k} transfer {d} {setting} {m}: sep24 {float(r["sep24"]):.2f} (range {float(r["sep24_range"] or 0):.2f}) '
                     f'cov72 raw {float(r["cov72"]):.3f}')

sync_main_tex()
(FIG / 'numbers.txt').write_text('\n'.join(numbers) + '\n', encoding='utf-8')
print('done')
