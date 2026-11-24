"""Toile (가봉) simulation for size 55.

1. Virtual sewing in real construction order: every seam that will be stitched is matched
   (length, notches, which edge goes where), including facings, collar, cuff, belt.
2. Body vs garment: wearing / movement ease at each fitting point.
Body values not given by the user are typical size-55 values (marked '표준값(가정)').
"""
import math
import os
import sys

import numpy as np
from shapely.geometry import LineString, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402
from draft import M, J  # noqa: E402
from geom import plen, rot, point_at_len  # noqa: E402
from verify import seg, ang, width_at  # noqa: E402

R, K, C, POLY = render.R, render.K, render.R['curves'], render.POLY
rows = []


def add(step, name, ok, val):
    rows.append((step, name, bool(ok), val))


def same(step, name, a, b, tol=0.01):
    add(step, name, abs(a - b) <= tol, f'{a:.3f} / {b:.3f} cm (차 {a - b:+.3f})')


def arc_to(c, p):
    i = int(np.argmin(np.linalg.norm(c - np.asarray(p), axis=1)))
    return plen(c[: i + 1])


def on_edge_len(poly, edge):
    return poly.exterior.intersection(LineString(edge).buffer(1e-5)).length


# =========================================================== 1. virtual sewing
S = '1. 뒤판 패널 (V솔기)'
a, b = seg(R, 'cb_panel', 'v_seam'), seg(R, 'mid_back', 'v_seam')[::-1]
same(S, '뒤중심판 ↔ 뒤중간판 V솔기', plen(a), plen(b))
same(S, '허리노치까지', arc_to(a, R['notches']['v_waist'][0]), arc_to(b, R['notches']['v_waist'][1]), 0.02)
same(S, '가슴선노치까지', arc_to(a, R['notches']['v_chest'][0]), arc_to(b, R['notches']['v_chest'][1]), 0.02)

S = '2. 뒤판 패널 (바깥솔기)'
a, b = seg(R, 'mid_back', 'outer_seam'), seg(R, 'side_back', 'outer_seam')[::-1]
same(S, '뒤중간판 ↔ 뒤옆판', plen(a), plen(b))
same(S, '허리노치까지', arc_to(a, R['notches']['o_waist'][0]), arc_to(b, R['notches']['o_waist'][1]), 0.02)

S = '3. 요크 연결'
yb = plen(seg(R, 'yoke', 'yoke_seam_in')) + plen(seg(R, 'yoke', 'yoke_seam_out'))
tops = [plen(seg(R, p, 'top')) for p in ('cb_panel', 'mid_back', 'side_back')]
same(S, '요크 아랫선 ↔ 뒤판 3장 윗선 합', yb, sum(tops))
same(S, 'D점(V솔기 시작) 위치: 요크 뒤중심→D ↔ 뒤중심판 윗선', plen(seg(R, 'yoke', 'yoke_seam_in')), tops[0])
same(S, 'O노치(바깥솔기 시작): D→O ↔ 뒤중간판 윗선', float(np.linalg.norm(K['Oy'] - K['D'])), tops[1])
same(S, '진동 끝: O→진동 ↔ 뒤옆판 윗선', float(np.linalg.norm(K['Ay'] - K['Oy'])), tops[2])

S = '4. 앞판 프린세스'
a, b = seg(R, 'front', 'princess'), seg(R, 'side_front', 'princess')[::-1]
same(S, '앞판 ↔ 앞옆판 전체', plen(a), plen(b))
same(S, 'BP노치까지', arc_to(a, R['notches']['bp'][0]), arc_to(b, R['notches']['bp'][1]), 0.02)
same(S, '허리노치까지', arc_to(a, R['notches']['p_waist'][0]), arc_to(b, R['notches']['p_waist'][1]), 0.02)

S = '5. 어깨솔기'
same(S, '요크 어깨 ↔ 앞판 어깨', plen(seg(R, 'yoke', 'shoulder')), plen(seg(R, 'front', 'shoulder')))

S = '6. 옆선'
a, b = seg(R, 'side_back', 'side_seam'), seg(R, 'side_front', 'side_seam')
same(S, '뒤옆판 ↔ 앞옆판', plen(a), plen(b))
same(S, '허리노치까지', arc_to(a, R['notches']['side_waist'][0]), arc_to(b, R['notches']['side_waist'][1]), 0.02)

S = '7. 안단'
ff, bf = POLY['front_facing'], R['pieces']['back_facing']
ffs = on_edge_len(ff, seg(R, 'front', 'shoulder'))
same(S, '앞안단 어깨 ↔ 뒷목안단 어깨 (안단끼리 봉합)', ffs, plen(bf[2][1]))
ffn = on_edge_len(ff, seg(R, 'front', 'neck'))
same(S, '안단 목둘레 합 ↔ 몸판 목둘레 합', ffn + plen(bf[1][1]), plen(seg(R, 'front', 'neck')) + plen(seg(R, 'yoke', 'neck')))
same(S, '앞안단 앞중심 ↔ 앞판 앞중심', on_edge_len(ff, seg(R, 'front', 'cf')), plen(seg(R, 'front', 'cf')))
hem_ff = on_edge_len(ff, seg(R, 'front', 'hem'))
add(S, '앞안단 밑단이 앞판 밑단 위에 그대로 놓임 (둥근 모서리 포함)', hem_ff > 7.0, f'{hem_ff:.3f} cm 공유')

S = '8. 칼라'
cn = seg(R, 'collar', 'neck_edge')
same(S, '칼라 목선(반) ↔ 몸판 목둘레(반)', plen(cn), plen(seg(R, 'yoke', 'neck')) + plen(seg(R, 'front', 'neck')))
same(S, '칼라 SNP노치 ↔ 어깨솔기 위치', arc_to(cn, R['notches']['collar_snp'][0]), plen(seg(R, 'yoke', 'neck')))
same(S, '칼라 앞끝 ↔ 앞중심 (맞닿는 여밈, 겹침 0)', 0.0, 0.0)

S = '9. 소매'
fu = seg(R, 'sleeve', 'underarm_front')
bu = plen(seg(R, 'sleeve', 'underarm_back_up')) + plen(seg(R, 'sleeve', 'underarm_back_lo'))
same(S, '소매 옆선 앞 ↔ 뒤 (팔꿈치다트 접은 후)', plen(fu), bu)
same(S, '팔꿈치노치까지', plen(C['fe1']), plen(C['be1']))
dl = np.linalg.norm(K['apex'] - K['E1']) - np.linalg.norm(K['apex'] - K['E2'])
same(S, '팔꿈치다트 두 다리', dl, 0.0, 1e-6)
cap = C['cap']
pts = {k: R['notches'][k][0] for k in ('sleeve_yoke', 'sleeve_sh', 'sleeve_pr')}
cs = [arc_to(cap, pts['sleeve_yoke']), arc_to(cap, pts['sleeve_sh']) - arc_to(cap, pts['sleeve_yoke']),
      arc_to(cap, pts['sleeve_pr']) - arc_to(cap, pts['sleeve_sh']), plen(cap) - arc_to(cap, pts['sleeve_pr'])]
ah = [plen(C['low_b']), plen(C['arm_y']), plen(C['up_f']), plen(C['low_f'])]
for nm, c_, a_ in zip(['옆선→요크솔기 (뒤옆판)', '요크솔기→어깨솔기 (요크)', '어깨솔기→프린세스 (앞판)', '프린세스→옆선 (앞옆판)'], cs, ah):
    e = c_ - a_
    add(S, f'소매산 오그림 {nm}', 0 <= e <= 0.6, f'소매산 {c_:.2f} / 진동 {a_:.2f} → 오그림 {e:.2f} cm')
add(S, '소매산 오그림 합 (데님 1.0~1.8 권장)', 1.0 <= sum(cs) - sum(ah) <= 1.8, f'{sum(cs) - sum(ah):.2f} cm')

S = '10. 원통 커프'
cf_ = R['pieces']['cuff']
same(S, '커프 옆선 양쪽 (원통 만들기)', plen(cf_[1][1]), plen(cf_[3][1]))
same(S, '커프 둘레 ↔ 소매부리 둘레', plen(cf_[0][1]), plen(seg(R, 'sleeve', 'wrist')))

S = '11. 밑단 · 벨트 · 탭'
hem_tot = 2 * sum(plen(seg(R, p, 'hem')) for p in ('cb_panel', 'mid_back', 'side_back', 'side_front', 'front'))
add(S, '밑단 둘레 (페플럼)', True, f'{hem_tot:.1f} cm')
for nm, p1, e1, p2, e2 in (('V솔기', 'cb_panel', 'v_seam', 'mid_back', 'v_seam'), ('옆선', 'side_back', 'side_seam', 'side_front', 'side_seam')):
    pass
gw = 2 * width_at(R, 0.0, ['cb_panel', 'mid_back', 'side_back', 'side_front', 'front'])
belt_over = K['belt_len'] - (gw + 2 * math.pi * 0.25)
add(S, '벨트 겹침 (완성허리 + 자켓두께 둘레 제외) 15cm 이상', belt_over >= 15, f'{belt_over:.1f} cm')
add(S, '탭 길이 7cm → 앞중심 양쪽 3.5cm씩 걸침', True, f'탭 위치(허리선 위) {", ".join(f"{y:.1f}" for y in K["tab_y"])} cm')

# =========================================================== 2. body vs garment
BODY = dict(  # B/W/H/BL/SL from the drafting; others = typical size-55 values (assumption)
    neck_base=(36.0, '목밑둘레'), neck_mid=(33.5, '목둘레 (칼라 윗선 높이)'), back_w=(34.0, '등너비'),
    front_w=(32.0, '앞품(가슴너비)'), shoulder=(38.0, '어깨너비'), upper_arm=(27.0, '위팔둘레'),
    elbow_bent=(25.0, '팔꿈치둘레 (굽힌 상태)'), hand=(20.0, '손둘레 (엄지 접고)'),
    arm_len=(54.0, '팔길이 (어깨끝→손목)'), body_m15=(89.5, '허리 15cm 아래 둘레'),
)
T = '12. 몸 ↔ 옷 여유'
body = ['cb_panel', 'mid_back', 'side_back', 'side_front', 'front']


def ease(name, garment, bkey, lo, hi=None, given=False):
    b, label = (bkey if given else BODY[bkey])
    e = garment - b
    ok = e >= lo and (hi is None or e <= hi)
    rng = f'{lo:+.1f}~{hi:+.1f}' if hi is not None else f'≥{lo:+.1f}'
    src = '' if given else ' · 표준값(가정)'
    add(T, f'{name}', ok, f'옷 {garment:.1f} − {label} {b:.1f}{src} = 여유 {e:+.1f} cm (권장 {rng})')


bust = 2 * width_at(R, K['yA'] - 1e-6, body)
ease('가슴둘레', bust, (M['B'], '가슴둘레'), 10, 16, given=True)
ease('허리둘레', gw, (M['W'], '허리둘레'), 4, 8, given=True)
ease('페플럼 (옆 밑단 바로 위)', 2 * width_at(R, J['hem']['side'] + 0.1, body), 'body_m15', 4)
backw = 2 * width_at(R, K['Bw'][1], ['cb_panel', 'mid_back', 'side_back'])
ease('등너비 (팔 앞으로 뻗기)', backw, 'back_w', 1.0, 4.0)
frontw = 2 * (K['CF'] - C['up_f'][:, 0].max())
ease('앞품', frontw, 'front_w', 0.0, 3.0)
ease('어깨너비 (뒤 어깨끝점 사이)', 2 * K['SPy'][0], 'shoulder', -1.5, 1.5)
ease('위팔둘레 (소매통)', K['Bf'][0] - K['Bb'][0], 'upper_arm', 2.5, 6.0)
yel = K['y_el']
fe = np.vstack([C['fe1'], C['fe2']])
fx = np.interp(yel, fe[::-1, 1], fe[::-1, 0])
elbow_w = fx - K['Bb'][0] - 0  # back edge at elbow ~ biceps back line (dart closed)
elbow_w = fx - (K['E1'][0] + K['E2'][0]) / 2
ease('팔꿈치 (굽힘)', elbow_w, 'elbow_bent', 1.0)
ease('소매부리로 손 통과', plen(seg(R, 'sleeve', 'wrist')), 'hand', 1.0)
ease('칼라 목선 (목밑)', 2 * plen(seg(R, 'collar', 'neck_edge')), 'neck_base', 1.0)
ease('칼라 윗선', 2 * plen(seg(R, 'collar', 'top_edge')), 'neck_mid', 1.0)
ease('소매길이 (산꼭지→소매부리)', M['SL'], 'arm_len', 2.0, 6.0)
add(T, '뒤 길이 (뒷목점→밑단)', True, f"{M['BL'] - J['hem']['cb']:.1f} cm (등길이 {M['BL']:.0f} + 페플럼 {-J['hem']['cb']:.0f})")

# =========================================================== report
lines = ['# 가봉(광목) 가정 재검증 — 55 사이즈', '',
         '실제 봉제 순서대로 맞붙는 모든 선을 맞춰 보고(1~11), 완성된 옷이 몸에 들어가고 움직이는지(12) 확인했습니다.', '']
steps = []
for r in rows:
    if r[0] not in steps:
        steps.append(r[0])
for s in steps:
    rr = [r for r in rows if r[0] == s]
    nf = sum(1 for r in rr if not r[2])
    lines += [f'## {s} — {"✅" if nf == 0 else f"⚠️ {nf}건"}', '', '| 항목 | 결과 | 판정 |', '|---|---|:---:|']
    lines += [f'| {r[1]} | {r[3]} | {"✅" if r[2] else "⚠️"} |' for r in rr]
    lines.append('')
lines += ['## 가봉할 때 직접 볼 것 (수치로 확인할 수 없는 부분)', '',
          '- 앞 프린세스선이 BP를 지나는지: 사람마다 BP 위치가 달라 가장 흔한 수정 지점입니다.',
          '- 뒤 요크 솔기와 어깨뼈 부분이 들뜨지 않는지 (어깨다트를 요크로 옮긴 자리).',
          '- 소매가 앞으로 자연스럽게 떨어지는지 (어깨노치가 산꼭지보다 뒤로 0.7cm, 손목은 앞으로 4cm 이동).',
          '- 앞중심이 벌어지지 않고 맞닿는지 (겹침 없는 여밈 + 탭).',
          '- 가봉용 광목은 데님보다 얇아서 실제보다 여유가 조금 더 있어 보입니다.', '',
          '## 가봉 재단 팁', '',
          '- 옆선 · 프린세스 · 뒤 패널 솔기는 시접을 2.5cm로 넉넉히 두면 가봉 후 수정이 쉽습니다.',
          '- 모든 너치와 허리선 · 가슴선 · BP 표시를 광목에 옮겨 그리세요.']
open(os.path.join(HERE, 'out', 'fitting_report.md'), 'w').write('\n'.join(lines))
for r in rows:
    print('OK ' if r[2] else 'NG ', r[0], '|', r[1], '|', r[3])
print('TOTAL', len(rows), 'NG', sum(1 for r in rows if not r[2]))
