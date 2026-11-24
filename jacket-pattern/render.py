"""Render block sheet, drafting diagram, pattern layout (1:1 PDF, A4 tiles, PNG previews)."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Polygon, LineString, box  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft import M, J, build_block, outline  # noqa: E402
from geom import P, unit, plen, rotv, tan_start, point_at_len  # noqa: E402

for f in font_manager.findSystemFonts():
    if 'wqy-zenhei' in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
CM = 1 / 2.54
SA = 1.0

R, BLK = build_block()
K = R['K']

INFO = {
    'yoke': ('뒤요크', '겉감 1장 (골선)', 'fold_cb'),
    'cb_panel': ('뒤중심판', '겉감 1장 (골선)', 'fold_cb'),
    'mid_back': ('뒤중간판', '겉감 2장', None),
    'side_back': ('뒤옆판', '겉감 2장', None),
    'side_front': ('앞옆판', '겉감 2장', None),
    'front': ('앞판', '겉감 2장', None),
    'front_facing': ('앞안단', '겉감 2장 + 심지', None),
    'back_facing': ('뒷목안단', '겉감 1장 (골선) + 심지', 'fold_cb'),
    'sleeve': ('소매', '겉감 2장', None),
    'collar': ('스탠드칼라', '겉감 2장 (골선) + 심지 1', 'fold_cb'),
    'ruffle': ('소매 러플', '겉감 2장 (안단 2)', None),
    'tab': ('탭', '10장 (5쌍) + 심지', None),
    'belt': ('벨트', '2장 (겉/안) + 심지', None),
}


# ------------------------------------------------------------------ derived facing pieces
def polys():
    d = {}
    for n, pc in R['pieces'].items():
        d[n] = Polygon(outline(pc))
    fr = d['front']
    shp, _ = point_at_len(outline([('s', R['pieces']['front'][0][1])]), 4.0)   # 4 cm from SNP on shoulder
    a, b = shp, P(K['CF'] - 7.0, -30)
    dirv = unit(b - a)
    big = Polygon([a - dirv * 80, b + dirv * 80, b + dirv * 80 + P(80, 0), a - dirv * 80 + P(80, 0)])
    d['front_facing'] = fr.intersection(big)
    yk = d['yoke']
    d['back_facing'] = yk.intersection(LineString(R['curves']['neck_b']).buffer(6.0))
    return d


POLY = polys()


def sa_poly(name):
    p = POLY[name].buffer(SA, join_style=2, mitre_limit=2.5)
    if INFO[name][2] == 'fold_cb':
        p = p.intersection(box(0, -500, 500, 500))
    return p


def draw_piece(ax, name, dx=0, dy=0, lw=0.9, label=True, sa=True, color='k'):
    poly = POLY[name]
    x, y = poly.exterior.xy
    ax.plot(np.array(x) + dx, np.array(y) + dy, color=color, lw=lw)
    if sa:
        sp = sa_poly(name)
        sx, sy = sp.exterior.xy
        ax.plot(np.array(sx) + dx, np.array(sy) + dy, color='0.35', lw=0.5, ls=(0, (4, 2)))
    if INFO[name][2] == 'fold_cb':
        ys = np.array(poly.exterior.xy[1])
        xs = np.array(poly.exterior.xy[0])
        yy = ys[np.abs(xs) < 1e-6]
        if len(yy):
            ax.plot([dx - 0.25, dx - 0.25], [yy.min() + dy, yy.max() + dy], color='tab:blue', lw=2.2)
            ax.text(dx - 1.0, (yy.min() + yy.max()) / 2 + dy, '골선 (CB)', rotation=90, ha='center', va='center',
                    fontsize=7, color='tab:blue')
    if label:
        c = poly.representative_point()
        cx, cy = c.x + dx, c.y + dy
        if name in ('belt',):
            ax.annotate('', xy=(cx + 12, cy), xytext=(cx - 12, cy),
                        arrowprops=dict(arrowstyle='<->', lw=0.8))
        else:
            hgt = min(14, (poly.bounds[3] - poly.bounds[1]) * 0.45)
            ax.annotate('', xy=(cx, cy + hgt / 2), xytext=(cx, cy - hgt / 2),
                        arrowprops=dict(arrowstyle='<->', lw=0.8))
        fs = 8 if name not in ('tab',) else 4
        ax.text(cx + 0.6, cy + 0.6, INFO[name][0], fontsize=fs + 2, weight='bold', ha='left', va='bottom')
        ax.text(cx + 0.6, cy - 0.6, INFO[name][1], fontsize=fs, ha='left', va='top')


NOTCH_MAP = {
    'side_back': [('side_waist', 0), ('o_waist', 1)], 'side_front': [('side_waist', 1), ('p_waist', 1), ('bp', 1)],
    'cb_panel': [('v_waist', 0), ('v_chest', 0)], 'mid_back': [('v_waist', 1), ('v_chest', 1), ('o_waist', 0), ('o_chest', 0)],
    'front': [('p_waist', 0), ('bp', 0)], 'yoke': [('yoke_O', 0)],
    'sleeve': [('sleeve_sh', 0), ('sleeve_yoke', 0), ('sleeve_pr', 0), ('sleeve_elbow', 0)],
    'collar': [('collar_snp', 0)],
}
NOTCH_MAP['side_back'].append(('o_chest', 1))


def draw_notches(ax, name, dx=0, dy=0):
    if name not in NOTCH_MAP:
        return
    ring = np.array(POLY[name].exterior.coords)
    for key, idx in NOTCH_MAP[name]:
        p = np.asarray(R['notches'][key][idx], float)
        i = int(np.argmin(np.linalg.norm(ring - p, axis=1)))
        t = unit(ring[(i + 1) % len(ring)] - ring[i - 1])
        n = rotv(t, 90)
        a, b = p - n * 0.9, p + n * 0.9
        ax.plot([a[0] + dx, b[0] + dx], [a[1] + dy, b[1] + dy], color='crimson', lw=1.1)
    if name == 'sleeve':
        v0, v1 = R['notches']['vent']
        ax.plot([v0[0] + dx, v1[0] + dx], [v0[1] + dy, v1[1] + dy], color='crimson', lw=1.0, ls='--')
        ax.text(v1[0] + dx + 0.4, v1[1] + dy, '트임 8cm', fontsize=6, color='crimson')
        ap, e1, e2 = K['apex'], K['E1'], K['E2']
        ax.text(ap[0] + dx + 0.4, ap[1] + dy, '팔꿈치다트', fontsize=6)
        ax.plot([0 + dx, 0 + dx], [0 + dy, -M['SL'] + dy], color='0.5', lw=0.5, ls='-.')
        ax.plot([K['Bb'][0] + dx, K['Bf'][0] + dx], [K['Bb'][1] + dy] * 2, color='0.5', lw=0.5, ls='-.')
        ax.plot([K['Bb'][0] + dx - 1, K['Bf'][0] + dx + 1], [K['y_el'] + dy] * 2, color='0.5', lw=0.5, ls='-.')
    if name == 'front':
        for y in K['tab_y']:
            ax.plot([K['CF'] - 0.8 + dx, K['CF'] + dx], [y + dy, y + dy], color='tab:green', lw=1.3)
        ax.text(K['CF'] - 3 + dx, K['tab_y'][0] + 1.2 + dy, '탭 위치', fontsize=6, color='tab:green')
        ax.plot([K['BP'][0] + dx], [K['BP'][1] + dy], 'o', ms=3, mfc='none', color='k')


# ------------------------------------------------------------------ layout
ROW1 = ['yoke', 'cb_panel', 'mid_back', 'side_back', 'side_front', 'front', 'front_facing']
ROW2 = ['sleeve', 'collar', 'back_facing', 'ruffle']
GAP = 5.0


def layout():
    pos = {}
    x = 0.0
    top1 = max(sa_poly(n).bounds[3] for n in ROW1)
    for n in ROW1:
        b = sa_poly(n).bounds
        pos[n] = (x - b[0] + 2, -top1)
        x += b[2] - b[0] + GAP
    w1 = x
    low1 = min(sa_poly(n).bounds[1] + pos[n][1] for n in ROW1) - GAP - 3
    x = 0.0
    hmax = 0
    for n in ROW2:
        b = sa_poly(n).bounds
        pos[n] = (x - b[0] + 2, low1 - b[3])
        x += b[2] - b[0] + GAP
        hmax = max(hmax, b[3] - b[1])
    # tabs and belt beside ruffle
    b = sa_poly('tab').bounds
    tx = x + 2
    for i in range(5):
        pos[f'tab{i}'] = (tx - b[0], low1 - 3 - i * 4.5)
    bb = sa_poly('belt').bounds
    pos['belt'] = (2 - bb[0], low1 - hmax - GAP - bb[3])
    return pos, max(w1, x + 15, sa_poly('belt').bounds[2] + 6)


def draw_layout(ax, pos):
    for n, (dx, dy) in pos.items():
        if n.startswith('tab'):
            draw_piece(ax, 'tab', dx, dy, label=(n == 'tab0'))
            continue
        draw_piece(ax, n, dx, dy)
        draw_notches(ax, n, dx, dy)


def scale_bar(ax, x, y):
    ax.add_patch(plt.Rectangle((x, y), 10, 10, fill=False, lw=1.2))
    ax.text(x + 5, y + 5, '10 × 10 cm\n인쇄 확인용', ha='center', va='center', fontsize=8)


def sheet_title(ax, x, y, t, sub):
    ax.text(x, y, t, fontsize=18, weight='bold', va='top')
    ax.text(x, y - 3.2, sub, fontsize=9, va='top')


def pattern_figure(scale=1.0, dpi=None):
    pos, width = layout()
    allb = [sa_poly('tab' if n.startswith('tab') else n).bounds for n in pos]
    xs0 = min(b[0] + pos[n][0] for b, n in zip(allb, pos)) - 4
    xs1 = max(b[2] + pos[n][0] for b, n in zip(allb, pos)) + 4
    ys0 = min(b[1] + pos[n][1] for b, n in zip(allb, pos)) - 4
    ys1 = 14
    fig = plt.figure(figsize=((xs1 - xs0) * CM * scale, (ys1 - ys0) * CM * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xs0, xs1)
    ax.set_ylim(ys0, ys1)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_layout(ax, pos)
    sheet_title(ax, xs0 + 3, ys1 - 1, '페플럼 피티드 자켓 — 패턴 (실물 1:1)',
                f"B{M['B']:.0f} W{M['W']:.0f} H{M['H']:.0f} 등길이{M['BL']:.0f} 소매길이{M['SL']:.0f}  |  "
                "실선 = 완성선, 점선 = 시접 1cm (골선 제외), 빨간 표시 = 너치, 양화살표 = 식서(올방향)")
    scale_bar(ax, xs1 - 16, ys1 - 13)
    return fig, ax, (xs0, xs1, ys0, ys1), pos


# ------------------------------------------------------------------ block sheet
def block_figure():
    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.set_aspect('equal')
    ax.axis('off')
    B, BL = M['B'], M['BL']
    CF, yC, xbw, xfw, xs, ysb, yE = K['CF'], K['yC'], K['xbw'], K['xfw'], K['xs'], K['ysb'], K['yE']
    g = dict(color='0.6', lw=0.6, ls='--')
    ax.plot([0, CF, CF, 0, 0], [0, 0, K['yFtop'], K['yFtop'], 0], **g)
    ax.plot([0, CF], [yC, yC], **g)
    ax.plot([xbw, xbw], [yC, BL], **g)
    ax.plot([xfw, xfw], [yC, K['yFtop']], **g)
    ax.plot([0, xbw], [ysb, ysb], **g)
    ax.plot([xbw, xs], [yE, yE], **g)
    ax.plot([xfw - B / 32, xfw - B / 32], [yC - 1, yC + 1], **g)
    k = dict(color='k', lw=1.4)
    ax.plot([0, 0], [0, BL], **k)
    ax.plot(*BLK['neck_b'].T, **k)
    ax.plot([K['SNPb'][0], K['P1'][0]], [K['SNPb'][1], K['P1'][1]], **k)
    ax.plot([K['P2'][0], K['SPb'][0]], [K['P2'][1], K['SPb'][1]], **k)
    ax.plot([K['P1'][0], K['D'][0], K['P2'][0]], [K['P1'][1], K['D'][1], K['P2'][1]], color='k', lw=0.9)
    ax.plot(*BLK['arm_b'].T, **k)
    ax.plot([xs, xs], [yC, 0], **k)
    ax.plot([0, CF], [0, 0], **k)
    ax.plot([CF, CF], [0, K['CFn'][1]], **k)
    ax.plot(*BLK['neck_f'].T, **k)
    ax.plot([K['SNPf'][0], K['SPf_blk'][0]], [K['SNPf'][1], K['SPf_blk'][1]], **k)
    ax.plot(*BLK['up_f'].T, **k)
    ax.plot(*BLK['low_f'].T, **k)
    ax.plot([K['G'][0], K['BP'][0], K['G2'][0]], [K['G'][1], K['BP'][1], K['G2'][1]], color='k', lw=0.9)
    ax.plot(*K['BP'], 'o', ms=5, mfc='white', color='k')
    t = lambda x, y, s, **kw: ax.text(x, y, s, fontsize=8, **kw)
    t(CF / 2, -1.5, f'B/2 + 6 = {B / 2 + 6:.2f}', ha='center', va='top')
    t(-0.8, (BL + yC) / 2, f'B/12+13.7\n= {B / 12 + 13.7:.2f}', ha='right', va='center')
    t(-0.8, yC / 2, f'등길이 {BL:.0f}', ha='right', va='center')
    t(xbw / 2, yC + 0.4, f'등품 B/8+7.4 = {xbw:.2f}', ha='center')
    t(CF - 0.5, yC + 0.4, f'앞품 B/8+6.2 = {B / 8 + 6.2:.2f}', ha='right')
    t(CF + 0.8, (yC + K['yFtop']) / 2, f'B/5+8.3\n= {B / 5 + 8.3:.2f}', va='center')
    t(K['BP'][0], yC - 1.0, f'BP\n(앞품/2+0.7)', ha='center', va='top')
    t(xbw / 2, ysb + 0.4, '견갑골선 (뒷목점 -8)', ha='center')
    t(xs + 0.4, yC / 2, f'옆선\n{xs:.2f}', va='center')
    t(K['nb'] / 2, BL + 3.2, f'◎ = B/24+3.4 = {K["nb"]:.2f}', ha='center')
    t((K['SNPb'][0] + K['SPb'][0]) / 2 + 1, (K['SNPb'][1] + K['SPb'][1]) / 2 + 0.8,
      f'18°  ▲+(B/32-0.8) = {np.linalg.norm(K["SPb"] - K["SNPb"]):.2f}', rotation=-18)
    t((K['SNPf'][0] + K['SPf_blk'][0]) / 2 - 2, (K['SNPf'][1] + K['SPf_blk'][1]) / 2 + 0.8,
      f'22°  ▲ = {K["tri"]:.2f}', rotation=22)
    t(K['G'][0] - 0.5, K['G'][1] + 0.8, f'가슴다트 (B/4-2.5)° = {K["dartf"]:.1f}°', ha='right')
    t(K['D'][0], K['D'][1] - 1.0, f'어깨다트 B/32-0.8 = {K["dartb"]:.3f}', ha='center', va='top')
    t(CF - 6, K['CFn'][1] - 1.5, f"앞목 ◎-0.2 × ◎+0.5", ha='center')
    ax.text(0, K['yFtop'] + 9, '문화식 신원형 — 길 원형 & 소매 원형', fontsize=17, weight='bold')
    ax.text(0, K['yFtop'] + 6.2, f"B{B:.0f} / W{M['W']:.0f} / 등길이 {BL:.0f} / 소매길이 {M['SL']:.0f} cm   "
            f"뒤AH {BLK['AHb']:.2f} + 앞AH {BLK['AHf']:.2f} = AH {BLK['AHb'] + BLK['AHf']:.2f} cm",
            fontsize=10)
    # sleeve block
    ox, oy = CF + 34, K['yFtop'] - 1
    sb = lambda a: (a[:, 0] + ox, a[:, 1] + oy)
    ax.plot(*sb(BLK['capb']), **k)
    ax.plot(*sb(BLK['capf']), **k)
    Bf, Bb, L = BLK['Bf'], BLK['Bb'], BLK['SL']
    ax.plot([Bb[0] + ox, Bb[0] + ox], [Bb[1] + oy, -L + oy], **k)
    ax.plot([Bf[0] + ox, Bf[0] + ox], [Bf[1] + oy, -L + oy], **k)
    ax.plot([Bb[0] + ox, Bf[0] + ox], [-L + oy, -L + oy], **k)
    ax.plot([ox, ox], [oy, -L + oy], **g)
    ax.plot([Bb[0] + ox, Bf[0] + ox], [Bb[1] + oy] * 2, **g)
    ax.plot([Bb[0] + ox, Bf[0] + ox], [BLK['y_el'] + oy] * 2, **g)
    ax.plot([ox, Bf[0] + ox], [oy, Bf[1] + oy], **g)
    ax.plot([ox, Bb[0] + ox], [oy, Bb[1] + oy], **g)
    ax.text(ox - 0.5, oy - BLK['cap_h'] * 0.75, f'산높이 = 평균진동깊이×5/6\n= {BLK["cap_h"]:.2f}', fontsize=8, ha='right')
    ax.text(ox + Bf[0] / 2 + 1, oy + Bf[1] / 2 + 1, f'앞AH = {BLK["AHf"]:.2f}', fontsize=8, rotation=-40)
    ax.text(ox + Bb[0] / 2 - 6, oy + Bb[1] / 2 + 1, f'뒤AH+1 = {BLK["AHb"] + 1:.2f}', fontsize=8, rotation=40)
    ax.text(ox, oy + BLK['y_el'] + 0.5, f'팔꿈치선 (소매길이/2+2.5)', fontsize=8, ha='center')
    ax.text(ox, -L + oy - 1.5, f'소매 원형  (소매산 이즈 {BLK["cap_ease"]:.2f} cm)', fontsize=10, ha='center', va='top')
    ax.text(ox, oy + 1.2, '소매산 꼭지점', fontsize=8, ha='center')
    ax.set_xlim(-12, ox + 26)
    ax.set_ylim(-L + oy - 5, K['yFtop'] + 11)
    return fig


# ------------------------------------------------------------------ drafting diagram (body frame)
def draft_figure():
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.set_aspect('equal')
    ax.axis('off')
    cols = dict(yoke='tab:blue', cb_panel='tab:orange', mid_back='tab:green', side_back='tab:red',
                side_front='tab:brown', front='tab:purple')
    gk = dict(color='0.75', lw=0.6, ls='--')
    ax.plot([0, K['CF']], [K['yC']] * 2, **gk)
    ax.plot([0, K['CF']], [0, 0], **gk)
    ax.plot([K['xs'], K['xs']], [K['yC'], J['hem']['side']], **gk)
    for n, c in cols.items():
        o = outline(R['pieces'][n])
        ax.fill(o[:, 0], o[:, 1], color=c, alpha=0.08)
        ax.plot(o[:, 0], o[:, 1], color=c, lw=1.2)
        p = POLY[n].representative_point()
        ax.text(p.x, p.y, INFO[n][0], color=c, fontsize=10, weight='bold', ha='center')
    ax.plot(*R['curves']['arm_b'].T, color='0.5', lw=0.6, ls=':')
    ax.text(K['D'][0], K['D'][1] + 0.5, 'D (어깨다트 끝 → 요크로 이동)', fontsize=7)
    ax.text(K['CF'] + 0.6, 0, '허리선', fontsize=8, va='center')
    ax.text(K['CF'] + 0.6, K['yC'], '가슴선', fontsize=8, va='center')
    ax.plot(*K['BP'], 'ko', ms=3)
    ax.text(0, K['yFtop'] + 4, '자켓 제도도 — 원형 위 전개 (요크 / 뒤 3판 / 앞 프린세스)', fontsize=15, weight='bold')
    ax.text(0, K['yFtop'] + 1.8,
            f"허리흡수 {K['supp']:.1f}cm 문화식 배분: V솔기 e+f {K['sV']:.2f} · 바깥솔기 d {K['sO']:.2f} · 옆선 c {K['sS']:.2f} · "
            f"프린세스 a+b {K['sP']:.2f} (+가슴다트 {K['dartf']:.1f}°)   |   페플럼 플레어 솔기당 {J['flare']:.0f}cm", fontsize=9)
    return fig


def assembled_figure():
    from geom import rot
    C = R['curves']
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    ax = axs[0]
    f_up, f_lo = C['up_f'], rot(C['low_f'], K['BP'], -K['dartf'])
    b_y = rot(C['arm_y'], K['D'], -K['thd'])
    ax.plot(*f_up.T, 'tab:purple', lw=2, label='앞판 진동')
    ax.plot(*f_lo.T, 'tab:brown', lw=2, label='앞옆판 진동 (가슴다트 닫음)')
    ax.plot(*b_y.T, 'tab:blue', lw=2, label='요크 진동 (몸판 위치로)')
    ax.plot(*C['low_b'].T, 'tab:red', lw=2, label='뒤옆판 진동')
    for p in (f_up[-1], b_y[-1]):
        ax.plot(*p, 'ko', ms=4)
    ax.set_title('진동둘레 — 봉제 상태 (이음점에서 꺾임 0)', fontsize=12)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    ax = axs[1]
    cap = C['cap']
    ax.plot(*cap.T, 'k', lw=2)
    for key, lab in (('sleeve_sh', '어깨노치'), ('sleeve_yoke', '요크노치'), ('sleeve_pr', '프린세스노치')):
        p = R['notches'][key][0]
        ax.plot(*p, 'o', color='crimson'); ax.text(p[0] + 0.4, p[1] + 0.4, lab, fontsize=8, color='crimson')
    ax.set_title(f"소매산  {K['cap_len']:.2f}cm = AH {K['AH']:.2f} + 이즈 {K['ease']:.2f}", fontsize=12)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    return fig


def main():
    f = assembled_figure()
    f.savefig(os.path.join(OUT, '05_봉제상태_곡선확인.png'), dpi=100)
    plt.close(f)
    # 1) block sheet
    f = block_figure()
    f.savefig(os.path.join(OUT, '01_원형_bodice_sleeve_block.png'), dpi=110)
    f.savefig(os.path.join(OUT, '01_원형_bodice_sleeve_block.pdf'))
    plt.close(f)
    # 2) drafting diagram
    f = draft_figure()
    f.savefig(os.path.join(OUT, '02_자켓_제도도.png'), dpi=110)
    f.savefig(os.path.join(OUT, '02_자켓_제도도.pdf'))
    plt.close(f)
    # 3) pattern layout preview + full scale
    f, ax, (x0, x1, y0, y1), pos = pattern_figure(scale=1.0)
    f.savefig(os.path.join(OUT, '03_패턴_실물크기_1to1.pdf'))
    f.savefig(os.path.join(OUT, '03_패턴_미리보기.png'), dpi=40)
    # 4) A4 tiles
    tw, th = 19.0, 27.7
    nx, ny = int(np.ceil((x1 - x0) / tw)), int(np.ceil((y1 - y0) / th))
    union = [sa_poly('tab' if n.startswith('tab') else n) for n in pos]
    from shapely.affinity import translate
    union = [translate(p, *pos[n]) for p, n in zip(union, pos)]
    pages = 0
    with PdfPages(os.path.join(OUT, '04_패턴_A4분할인쇄.pdf')) as pdf:
        for j in range(ny):
            for i in range(nx):
                tx0, ty1 = x0 + i * tw, y1 - j * th
                tile = box(tx0, ty1 - th, tx0 + tw, ty1)
                if not (i == 0 and j == 0) and not any(p.intersects(tile) for p in union):
                    continue
                g = plt.figure(figsize=(21 * CM, 29.7 * CM))
                a = g.add_axes([1 / 21, 1 / 29.7, tw / 21, th / 29.7])
                a.set_xlim(tx0, tx0 + tw)
                a.set_ylim(ty1 - th, ty1)
                a.set_aspect('equal')
                a.set_xticks([])
                a.set_yticks([])
                for s in a.spines.values():
                    s.set_lw(0.3)
                    s.set_color('0.6')
                draw_layout(a, pos)
                if i == 0 and j == 0:
                    sheet_title(a, x0 + 1, y1 - 1, '자켓 패턴', '100% / 실제 크기로 인쇄')
                    scale_bar(a, tx0 + 2, ty1 - th + 2)
                g.text(0.5, 0.985, f'열 {i + 1}/{nx}  ·  행 {j + 1}/{ny}   (테두리선 맞춰 붙이기)', ha='center',
                       va='top', fontsize=8)
                pdf.savefig(g)
                plt.close(g)
                pages += 1
    plt.close(f)
    print('A4 pages', pages, 'grid', nx, ny, 'sheet cm', round(x1 - x0, 1), round(y1 - y0, 1))


if __name__ == '__main__':
    main()
