"""원형 (Bunka block) verification + real-size outputs."""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from shapely.geometry import Polygon, box, LineString  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402,F401  (fonts)
import block as BK  # noqa: E402
from draft import M, outline  # noqa: E402
from geom import P, unit, plen, rot, rotv, tan_start, tan_end, angdeg, inflections, max_turn, min_radius  # noqa: E402
from verify import ang  # noqa: E402

OUT = os.path.join(HERE, 'out')
K, D, PC = BK.K, BK.DATA, BK.PIECES
CM = 1 / 2.54
B, W, BL = M['B'], M['W'], M['BL']
TL, TA = 0.01, 0.10
rounds = []


def rnd(t):
    rounds.append((t, []))


def add(name, v, tgt, tol, u=''):
    rounds[-1][1].append((name, f'{v:.4f}{u}', f'{tgt:.4f}{u} ±{tol}', abs(v - tgt) <= tol))


def cond(name, ok, info=''):
    rounds[-1][1].append((name, info, '', bool(ok)))


def sg(p, e):
    return [c for n, c in PC[p] if n == e][0]


def segs(p, e):
    return [c for n, c in PC[p] if n == e]


# ------------------------------------------------------------ 1 formulas
rnd('1. 문화식 공식 재계산')
add('가슴선 B/12+13.7', BL - K['yC'], B / 12 + 13.7, 1e-9, 'cm')
add('몸판폭 B/2+6', K['half'], B / 2 + 6, 1e-9, 'cm')
add('등품 B/8+7.4', K['xbw'], B / 8 + 7.4, 1e-9, 'cm')
add('앞품 B/8+6.2', K['CF'] - K['xfw'], B / 8 + 6.2, 1e-9, 'cm')
add('앞중심 B/5+8.3', K['yFtop'] - K['yC'], B / 5 + 8.3, 1e-9, 'cm')
add('뒷목너비 B/24+3.4', K['nb'], B / 24 + 3.4, 1e-9, 'cm')
add('가슴다트 (B/4-2.5)°', math.degrees(math.atan2(*(K['G2'] - K['BP'])[::-1]) - math.atan2(*(K['G'] - K['BP'])[::-1])),
    B / 4 - 2.5, 1e-9, '°')
add('어깨다트 B/32-0.8', np.linalg.norm(K['P1'] - K['P2']), B / 32 - 0.8, 1e-9, 'cm')
add('허리다트 합 = B/2+6 - (W/2+3)', sum(D['amt'].values()), K['half'] - (W / 2 + 3), 1e-9, 'cm')

# ------------------------------------------------------------ 2 shoulder / neck
rnd('2. 어깨 · 목둘레')
bsh = plen(sg('back', 'shoulder_in')) + plen(sg('back', 'shoulder_out'))
add('뒤어깨(다트 제외) = 앞어깨', bsh - plen(sg('front', 'shoulder')), 0, TL, 'cm')
add('어깨다트 두 다리 길이차', plen(sg('back', 'sdart_out')) - plen(sg('back', 'sdart_in')), 0, 1e-9, 'cm')
add('뒷목 ∠뒤중심', ang(sg('back', 'neck'), 'e', sg('back', 'cb_upper'), 's'), 90, TA, '°')
a1 = ang(sg('back', 'shoulder_in'), 'e', sg('back', 'neck'), 's')
a2 = ang(sg('front', 'neck'), 'e', sg('front', 'shoulder'), 's')
add('옆목점 뒤 ∠', a1, 90, TA, '°')
add('옆목점 앞 ∠', a2, 90, TA, '°')
add('옆목점 합 180°', a1 + a2, 180, 2 * TA, '°')
add('앞목 ∠앞중심', ang(sg('front', 'cf'), 'e', sg('front', 'neck'), 's'), 90, TA, '°')
add('목둘레 변곡 0 (뒤)', inflections(D['neck_b']), 0, 0)
add('목둘레 변곡 0 (앞)', inflections(D['neck_f']), 0, 0)

# ------------------------------------------------------------ 3 armhole
rnd('3. 진동둘레')
b1 = ang(sg('back', 'armhole'), 'e', sg('back', 'shoulder_out'), 's')
b2 = ang(sg('front', 'shoulder'), 'e', sg('front', 'armhole_up'), 's')
add('어깨끝점 뒤 ∠', b1, 90, TA, '°')
add('어깨끝점 앞 ∠', b2, 90, TA, '°')
add('어깨끝점 합 180°', b1 + b2, 180, 2 * TA, '°')
c1 = ang(sg('back', 'side'), 'e', sg('back', 'armhole'), 's')
c2 = ang(sg('front', 'armhole_lo'), 'e', sg('front', 'side'), 's')
add('옆선 위 뒤 ∠', c1, 90, TA, '°')
add('옆선 위 앞 ∠', c2, 90, TA, '°')
add('옆선 위 합 180°', c1 + c2, 180, 2 * TA, '°')
lowc = rot(D['low_f'], K['BP'], -K['dartf'])
add('가슴다트 닫음: 진동 이음 간격', np.linalg.norm(lowc[0] - D['up_f'][-1]), 0, 1e-6, 'cm')
add('가슴다트 닫음: 진동 접선차', math.degrees(math.acos(min(1, float(tan_end(D['up_f']) @ tan_start(lowc))))), 0, TA, '°')
for nm, c in (('뒤 진동', D['arm_b']), ('앞 진동 위', D['up_f']), ('앞 진동 아래', D['low_f']), ('앞 진동(다트 닫음)', D['arm_fc'])):
    add(f'{nm} 변곡 0', inflections(c), 0, 0)
cond('앞 진동 최소 곡률반경 ≥ 1.5cm', min_radius(D['arm_fc']) >= 1.5, f"{min_radius(D['arm_fc']):.2f}cm")
cond('앞품 유지 (앞품선 안쪽 ≤ 0.2cm)', D['up_f'][:, 0].max() - K['xfw'] <= 0.2, f"{D['up_f'][:, 0].max() - K['xfw']:.2f}cm")
add('진동둘레 AH (참고)', D['AHb'] + D['AHf'], D['AHb'] + D['AHf'], 0, 'cm')

# ------------------------------------------------------------ 4 side + waist
rnd('4. 옆선 · 허리선 · 허리다트')
add('옆선 뒤 = 앞', plen(sg('back', 'side')) - plen(sg('front', 'side')), 0, 1e-9, 'cm')
bw, fw = segs('back', 'waist'), segs('front', 'waist')
add('허리 ∠뒤중심', ang(sg('back', 'cb_lower'), 'e', bw[0], 's'), 90, TA, '°')
add('허리 ∠앞중심', ang(fw[-1], 'e', sg('front', 'cf'), 's'), 90, TA, '°')
w1 = ang(bw[-1], 'e', sg('back', 'side'), 's')
w2 = ang(sg('front', 'side'), 'e', fw[0], 's')
add('허리 옆선 합 180°', w1 + w2, 180, 2 * TA, '°')
for pn, wl, nm in (('back', bw, ('e', 'd')), ('front', fw, ('b', 'a'))):
    for i in range(2):
        L, Rr = sg(pn, f'dart{i}_L'), sg(pn, f'dart{i}_R')
        add(f'허리다트 {nm[i]} 두 다리 길이차', plen(L) - plen(Rr), 0, 1e-6, 'cm')
        x1 = ang(wl[i], 'e', L, 's')
        x2 = ang(Rr, 'e', wl[i + 1], 's')
        add(f'허리다트 {nm[i]} 접으면 허리선 꺾임 0 (합 180°)', x1 + x2, 180, 2 * TA, '°')
wt = 2 * (sum(plen(s) for s in bw) + sum(plen(s) for s in fw))
cond('완성 허리둘레(다트 접음) ≈ W+6 (허리선 바로잡기로 생긴 차 ≤ 0.2cm)', abs(wt - (W + 6)) <= 0.2,
     f'{wt:.2f}cm (목표 {W + 6:.0f})')

# ------------------------------------------------------------ 5 sleeve
rnd('5. 소매원형')
cb, cf = sg('sleeve', 'cap_back'), sg('sleeve', 'cap_front')
add('소매산 높이 = 평균 진동깊이 × 5/6', D['cap_h'], D['depth'] * 5 / 6, 1e-9, 'cm')
add('뒤 사선 = 뒤AH+1', np.linalg.norm(D['Bb']), D['AHb'] + 1, 1e-9, 'cm')
add('앞 사선 = 앞AH', np.linalg.norm(D['Bf']), D['AHf'], 1e-9, 'cm')
s1 = ang(sg('sleeve', 'underarm_back'), 'e', cb, 's')
s2 = ang(cf, 'e', sg('sleeve', 'underarm_front'), 's')
add('소매산 ∠뒤 소매밑선', s1, 90, TA, '°')
add('소매산 ∠앞 소매밑선', s2, 90, TA, '°')
add('소매산 꼭지 접선 수평', abs(angdeg(tan_end(cb))), 0, TA, '°')
add('소매산 뒤 변곡 1 (S자)', inflections(cb), 1, 0)
add('소매산 앞 변곡 1 (S자)', inflections(cf), 1, 0)
add('소매밑선 앞 = 뒤', plen(sg('sleeve', 'underarm_front')) - plen(sg('sleeve', 'underarm_back')), 0, 1e-9, 'cm')
add('소매부리 ∠', ang(sg('sleeve', 'underarm_front'), 'e', sg('sleeve', 'wrist'), 's'), 90, TA, '°')
cond('소매산 이즈 (참고: 원형은 디자인 때 원단에 맞춰 조정)', True, f"{D['cap_ease']:.2f}cm")

# ------------------------------------------------------------ 6 integrity
rnd('6. 외곽선 폐합 · 자기교차 · 꺾임')
worst = 0
for pn, piece in PC.items():
    gaps = [np.linalg.norm(piece[i][1][-1] - piece[(i + 1) % len(piece)][1][0]) for i in range(len(piece))]
    add(f'[{pn}] 폐합 간격', max(gaps), 0, 1e-6, 'cm')
    cond(f'[{pn}] 자기교차 없음', Polygon(outline(piece)).is_valid)
    for en, c in piece:
        worst = max(worst, max_turn(c))
cond('모든 곡선 꺾임 없음 (< 1°)', worst < 1.0, f'{worst:.3f}°')

# ------------------------------------------------------------ render
NAMES = dict(back=('뒤판 원형', '1장 (반신)'), front=('앞판 원형', '1장 (반신)'), sleeve=('소매 원형', '1장'))


def layout():
    pos, x = {}, 0.0
    top = 0.0
    for n in ('back', 'front', 'sleeve'):
        o = outline(PC[n])
        dx = x - o[:, 0].min() + 2
        dy = top - o[:, 1].max() - 12
        pos[n] = (dx, dy)
        x += o[:, 0].max() - o[:, 0].min() + 6
    return pos


def draw(ax, pos, fs=1.0):
    for n, (dx, dy) in pos.items():
        o = outline(PC[n])
        ax.plot(o[:, 0] + dx, o[:, 1] + dy, 'k', lw=1.0)
        c = Polygon(o).representative_point()
        ax.text(c.x + dx, c.y + dy + 3, NAMES[n][0], fontsize=12 * fs, weight='bold', ha='center')
        ax.text(c.x + dx, c.y + dy + 1, NAMES[n][1] + ' · 시접 없음 (완성선)', fontsize=8 * fs, ha='center')
        ax.annotate('', xy=(c.x + dx, c.y + dy - 2), xytext=(c.x + dx, c.y + dy - 16),
                    arrowprops=dict(arrowstyle='<->', lw=0.8))
    dxb, dyb = pos['back']
    dxf, dyf = pos['front']
    g = dict(color='0.55', lw=0.6, ls='-.')
    for (dx, dy), x0_, x1_ in (((dxb, dyb), 0, K['xs'] + 1), ((dxf, dyf), K['xs'] - 1, K['CF'])):
        ax.plot([x0_ + dx, x1_ + dx], [K['yC'] + dy] * 2, **g)
        ax.text(x1_ + dx - 0.5, K['yC'] + dy + 0.3, '가슴선', fontsize=7 * fs, ha='right')
    ax.plot(K['BP'][0] + dxf, K['BP'][1] + dyf, 'o', ms=4, mfc='white', color='k')
    ax.text(K['BP'][0] + dxf + 0.6, K['BP'][1] + dyf - 0.8, 'BP', fontsize=8 * fs)
    ax.text(K['CBn'][0] + dxb + 0.5, K['CBn'][1] + dyb - 2, '뒤중심', fontsize=7 * fs, rotation=90)
    ax.text(K['CF'] + dxf - 1.2, 20 + dyf, '앞중심', fontsize=7 * fs, rotation=90)
    for (cx, w, ap), lab in zip(D['back_darts'], 'ed'):
        ax.text(cx + dxb, 1.0 + dyb, lab, fontsize=8 * fs, ha='center')
    for (cx, w, ap), lab in zip(D['front_darts'], 'ba'):
        ax.text(cx + dxf, 1.0 + dyf, lab, fontsize=8 * fs, ha='center')
    dxs, dys = pos['sleeve']
    ax.plot([dxs, dxs], [dys, -D['SL'] + dys], **g)
    ax.plot([D['Bb'][0] + dxs, D['Bf'][0] + dxs], [-D['cap_h'] + dys] * 2, **g)
    ax.plot([D['Bb'][0] + dxs, D['Bf'][0] + dxs], [D['y_el'] + dys] * 2, **g)
    ax.text(D['Bf'][0] + dxs - 0.5, D['y_el'] + dys + 0.3, '팔꿈치선', fontsize=7 * fs, ha='right')
    def tick(curve, p_, dx, dy, double=False):
        i = int(np.argmin(np.linalg.norm(curve - p_, axis=1)))
        t_ = unit(curve[min(i + 3, len(curve) - 1)] - curve[max(i - 3, 0)])
        n_ = rotv(t_, 90)
        offs = (-0.25, 0.25) if double else (0.0,)
        for o_ in offs:
            q = p_ + t_ * o_
            ax.plot([q[0] - n_[0] * 0.8 + dx, q[0] + n_[0] * 0.8 + dx], [q[1] - n_[1] * 0.8 + dy, q[1] + n_[1] * 0.8 + dy],
                    color='crimson', lw=1.2)
    cap = D['cap']
    tick(cap, D['sh_notch'], dxs, dys)
    tick(cap, D['fr_notch'], dxs, dys)
    tick(cap, D['bk_notch'], dxs, dys, double=True)
    tick(D['arm_b'], D['bw_notch_arm'], dxb, dyb, double=True)
    ax.text(K['G'][0] + dxf - 0.5, K['G'][1] + dyf + 0.6, '앞너치=다트끝', fontsize=6 * fs, ha='right', color='crimson')
    tick(D['up_f'], D['up_f'][-1], dxf, dyf)


def bounds(pos):
    xs, ys = [], []
    for n, (dx, dy) in pos.items():
        o = outline(PC[n])
        xs += [o[:, 0].min() + dx, o[:, 0].max() + dx]
        ys += [o[:, 1].min() + dy, o[:, 1].max() + dy]
    return min(xs) - 4, max(xs) + 4, min(ys) - 4, max(ys) + 16


def main():
    pos = layout()
    x0, x1, y0, y1 = bounds(pos)
    fig = plt.figure(figsize=((x1 - x0) * CM, (y1 - y0) * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.axis('off')
    draw(ax, pos)
    ax.text(x0 + 3, y1 - 2, '문화식 신원형 — 실물 1:1 (55: B84 W66 등길이38 소매길이58)', fontsize=18, weight='bold', va='top')
    ax.add_patch(plt.Rectangle((x1 - 16, y1 - 13), 10, 10, fill=False, lw=1.2))
    ax.text(x1 - 11, y1 - 8, '10 × 10 cm', ha='center', va='center', fontsize=8)
    fig.savefig(os.path.join(OUT, '06_원형_실물크기_1to1.pdf'))
    fig.savefig(os.path.join(OUT, '06_원형_미리보기.png'), dpi=45)
    plt.close(fig)
    tw, th = 19.0, 27.7
    nx, ny = int(np.ceil((x1 - x0) / tw)), int(np.ceil((y1 - y0) / th))
    polys = [Polygon(outline(PC[n]) + np.array(pos[n])) for n in pos]
    pages = 0
    with PdfPages(os.path.join(OUT, '07_원형_A4분할인쇄.pdf')) as pdf:
        for j in range(ny):
            for i in range(nx):
                tx0, ty1 = x0 + i * tw, y1 - j * th
                tile = box(tx0, ty1 - th, tx0 + tw, ty1)
                if not (i == 0 and j == 0) and not any(p.buffer(0.5).intersects(tile) for p in polys):
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
                draw(a, pos)
                if i == 0 and j == 0:
                    a.add_patch(plt.Rectangle((tx0 + 2, ty1 - th + 2), 10, 10, fill=False, lw=1.2))
                    a.text(tx0 + 7, ty1 - th + 7, '10 × 10 cm', ha='center', va='center', fontsize=8)
                g.text(0.5, 0.985, f'원형  열 {i + 1}/{nx} · 행 {j + 1}/{ny}  (100% 인쇄, 테두리 맞춰 붙이기)', ha='center',
                       va='top', fontsize=8)
                pdf.savefig(g)
                plt.close(g)
                pages += 1
    # report
    tot = sum(len(r[1]) for r in rounds)
    nf = sum(1 for r in rounds for c in r[1] if not c[3])
    L = ['# 원형 검증 보고서 (문화식 신원형, 55)', '', f'**전체 {tot}개 항목 중 불합격 {nf}개**', '']
    for t, cs in rounds:
        f_ = sum(1 for c in cs if not c[3])
        L += [f'## {t} — {"✅" if f_ == 0 else f"❌ {f_}건"}', '', '| 항목 | 측정값 | 기준 | 판정 |', '|---|---:|---:|:---:|']
        L += [f'| {c[0]} | {c[1]} | {c[2]} | {"✅" if c[3] else "❌"} |' for c in cs]
        L.append('')
    open(os.path.join(OUT, 'block_report.md'), 'w').write('\n'.join(L))
    for t, cs in rounds:
        for c in cs:
            if not c[3]:
                print('FAIL', t, c)
    print('TOTAL', tot, 'FAIL', nf, 'A4 pages', pages, 'sheet', round(x1 - x0, 1), round(y1 - y0, 1))


if __name__ == '__main__':
    main()
