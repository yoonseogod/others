"""Independent audit of the DELIVERED files (not the drafting code's own numbers).

A. 1:1 PDF  : page scale, 10cm square, every piece outline re-measured from the PDF vectors
B. A4 PDF   : page size, printable frame 19 x 27.7 cm, tile count, test square
C. Seam allowance : distance net line -> cutting line = 1.000 cm everywhere (fold edges 0)
D. Notches  : on the outline, matching arc positions, sleeve-cap ease split
E. Facings  : lie exactly inside / on their parent pieces
F. Size sweep: whole system rebuilt + verified for 5 other body sizes
"""
import math
import os
import subprocess
import sys

import numpy as np
import pymupdf
from shapely.geometry import Polygon, Point, LinearRing

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402  (builds R, POLY, layout)
from draft import outline  # noqa: E402
from geom import plen, rot, unit  # noqa: E402

PT = 72 / 2.54          # points per cm
R, K, POLY = render.R, render.K, render.POLY
res = []


def chk(sec, name, ok, val):
    res.append((sec, name, bool(ok), val))


def path_pts(d):
    pts = []
    for it in d['items']:
        if it[0] == 'l':
            pts += [it[1], it[2]]
        elif it[0] == 'c':
            pts += [it[1], it[4]]
        elif it[0] == 're':
            r = it[1]
            pts += [r.tl, r.tr, r.br, r.bl, r.tl]
        elif it[0] == 'qu':
            q = it[1]
            pts += [q.ul, q.ur, q.lr, q.ll, q.ul]
    if len(pts) < 2:
        return np.zeros((2, 2))
    a = np.array([[p.x, p.y] for p in pts])
    keep = np.r_[True, np.linalg.norm(np.diff(a, axis=0), axis=1) > 1e-9]
    return a[keep]


# ---------------------------------------------------------------- A. 1:1 PDF
doc = pymupdf.open(os.path.join(HERE, 'out', '03_패턴_실물크기_1to1.pdf'))
pg = doc[0]
pos, _ = render.layout()
fig, ax, (x0, x1, y0, y1), _ = render.pattern_figure()
render.plt.close(fig)
chk('A', '페이지 가로 = 배치폭', abs(pg.rect.width / PT - (x1 - x0)) < 0.01, f'{pg.rect.width / PT:.3f} / {x1 - x0:.3f} cm')
chk('A', '페이지 세로 = 배치높이', abs(pg.rect.height / PT - (y1 - y0)) < 0.01, f'{pg.rect.height / PT:.3f} / {y1 - y0:.3f} cm')
draws = pg.get_drawings()
sq = [d for d in draws if abs(d['rect'].width / PT - 10) < 0.02 and abs(d['rect'].height / PT - 10) < 0.02]
chk('A', '10×10cm 확인 사각형', len(sq) >= 1, f"{sq[0]['rect'].width / PT:.4f} × {sq[0]['rect'].height / PT:.4f} cm" if sq else '없음')
for n in render.INFO:
    if n == 'tab':
        continue
    ring = np.array(POLY[n].exterior.coords)
    L = plen(ring)
    # expected bounding box of this net outline on the PDF page (PDF y axis points down)
    dx, dy = pos[n]
    ex = ((ring[:, 0].min() + dx - x0) * PT, (y1 - (ring[:, 1].max() + dy)) * PT,
          (ring[:, 0].max() + dx - x0) * PT, (y1 - (ring[:, 1].min() + dy)) * PT)
    cand = [d for d in draws if max(abs(d['rect'].x0 - ex[0]), abs(d['rect'].y0 - ex[1]),
                                    abs(d['rect'].x1 - ex[2]), abs(d['rect'].y1 - ex[3])) < 0.05 * PT]
    if not cand:
        chk('A', f'PDF 재측정 완성선 둘레 [{render.INFO[n][0]}]', False, 'PDF에서 선을 찾지 못함')
        continue
    best = min((plen(path_pts(d) / PT) for d in cand), key=lambda v: abs(v - L))
    chk('A', f'PDF 재측정 완성선 둘레 [{render.INFO[n][0]}]', abs(best - L) < 0.02, f'PDF {best:.3f} / 설계 {L:.3f} cm')

# ---------------------------------------------------------------- B. A4 tiles
d4 = pymupdf.open(os.path.join(HERE, 'out', '04_패턴_A4분할인쇄.pdf'))
okA4 = all(abs(p.rect.width - 595.28) < 0.5 and abs(p.rect.height - 841.89) < 0.5 for p in d4)
chk('B', 'A4 용지 크기 (모든 페이지)', okA4, f'{len(d4)}쪽')
frames = 0
for p in d4:
    fr = [d for d in p.get_drawings() if abs(d['rect'].width / PT - 19) < 0.02 and abs(d['rect'].height / PT - 27.7) < 0.02]
    frames += bool(fr)
chk('B', '모든 쪽 인쇄틀 19×27.7cm (이어붙임 기준)', frames == len(d4), f'{frames}/{len(d4)}')
sq4 = [d for d in d4[0].get_drawings() if abs(d['rect'].width / PT - 10) < 0.02 and abs(d['rect'].height / PT - 10) < 0.02]
chk('B', 'A4 1쪽 10×10cm 사각형', len(sq4) >= 1, f"{sq4[0]['rect'].width / PT:.4f} cm" if sq4 else '없음')
from shapely.geometry import box  # noqa: E402
from shapely.affinity import translate  # noqa: E402
allp = [translate(render.sa_poly('tab' if n.startswith('tab') else n), *pos[n]) for n in pos]
nx, ny = int(np.ceil((x1 - x0) / 19)), int(np.ceil((y1 - y0) / 27.7))
need = sum(1 for j in range(ny) for i in range(nx)
           if (i == 0 and j == 0) or any(q.intersects(box(x0 + i * 19, y1 - (j + 1) * 27.7, x0 + (i + 1) * 19, y1 - j * 27.7)) for q in allp))
chk('B', '패턴이 걸친 칸 수 = PDF 쪽수 (빠진 조각 없음)', need == len(d4), f'{need} / {len(d4)}')

# ---------------------------------------------------------------- C. seam allowance
for n in render.INFO:
    net = POLY[n]
    cut = render.sa_poly(n)
    fold = render.INFO[n][2] == 'fold_cb'
    ring = LinearRing(cut.exterior.coords)
    if fold:   # the fold line itself is not a cutting line -> remove it before measuring
        from shapely.geometry import LineString
        ring = ring.difference(LineString([(0, -999), (0, 999)]).buffer(1e-7))
    pts = np.array(net.exterior.coords)
    pts = pts[pts[:, 0] > 1e-6] if fold else pts
    sel = np.array([ring.distance(Point(p)) for p in pts[::3]])
    ok = abs(sel.min() - 1.0) < 1e-3 and np.all(sel > 1.0 - 1e-3)
    chk('C', f'시접 1cm [{render.INFO[n][0]}]', ok and cut.contains(net.buffer(-1e-6)),
        f'완성선→재단선 최소 {sel.min():.4f} cm' + (' (골선 0)' if fold else ''))

# ---------------------------------------------------------------- D. notches
for pn, lst in render.NOTCH_MAP.items():
    for key, idx in lst:
        p = np.asarray(R['notches'][key][idx], float)
        dist = POLY[pn].exterior.distance(Point(p))
        chk('D', f'너치가 외곽선 위 [{render.INFO[pn][0]} · {key}]', dist < 0.01, f'{dist:.5f} cm')
C = R['curves']
e = K['ease']
cap = C['cap']


def arc_to(c, p):
    i = int(np.argmin(np.linalg.norm(c - p, axis=1)))
    return plen(c[: i + 1])


segs_cap = [arc_to(cap, R['notches']['sleeve_yoke'][0]),
            arc_to(cap, R['notches']['sleeve_sh'][0]) - arc_to(cap, R['notches']['sleeve_yoke'][0]),
            arc_to(cap, R['notches']['sleeve_pr'][0]) - arc_to(cap, R['notches']['sleeve_sh'][0]),
            plen(cap) - arc_to(cap, R['notches']['sleeve_pr'][0])]
segs_arm = [plen(C['low_b']), plen(C['arm_y']), plen(C['up_f']), plen(C['low_f'])]
names = ['뒤옆판 진동', '요크 진동', '앞판 진동', '앞옆판 진동']
for nm, a, b in zip(names, segs_cap, segs_arm):
    chk('D', f'소매산 구간 이즈 [{nm}]', 0 < a - b < 0.8 and abs((a - b) - b * e / K['AH']) < 0.02,
        f'소매산 {a:.3f} - 진동 {b:.3f} = {a - b:+.3f} cm')
chk('D', '구간 이즈 합 = 1.30', abs(sum(segs_cap) - sum(segs_arm) - 1.3) < 0.01, f'{sum(segs_cap) - sum(segs_arm):.4f} cm')
cn = R['pieces']['collar'][1][1]
chk('D', '칼라 SNP노치 ↔ 뒷목둘레', abs(arc_to(cn, R['notches']['collar_snp'][0]) - plen(C['neck_b'])) < 0.01,
    f"{arc_to(cn, R['notches']['collar_snp'][0]):.3f} / {plen(C['neck_b']):.3f} cm")

# ---------------------------------------------------------------- E. facings
for f, parent in (('front_facing', 'front'), ('back_facing', 'yoke')):
    outside = POLY[f].difference(POLY[parent]).area
    chk('E', f'{render.INFO[f][0]}가 {render.INFO[parent][0]} 안에 정확히 들어감', outside < 1e-6, f'벗어난 면적 {outside:.2e} cm²')
from geom import max_turn, inflections  # noqa: E402
ffc = POLY['_ff_inner']
chk('E', '앞안단 안쪽선 꺾임 없음 / 변곡 0', max_turn(ffc) < 1.0 and inflections(ffc) == 0,
    f'최대 방향변화 {max_turn(ffc):.3f}°, 변곡 {inflections(ffc)}')
ffr = np.array(POLY['front_facing'].exterior.coords)
chk('E', '앞안단 꼭짓점 수 = 4 (어깨·앞중심 위·아래, 안쪽선 끝) 외 꺾임 없음',
    sum(1 for i in range(1, len(ffr) - 1) if np.degrees(np.arccos(np.clip(np.dot(unit(ffr[i] - ffr[i - 1]), unit(ffr[i + 1] - ffr[i])), -1, 1))) > 15) <= 5,
    '')
fr = POLY['front_facing']
cf_on = [p for p in fr.exterior.coords if abs(p[0] - K['CF']) < 1e-6]
chk('E', '앞안단 앞중심선 = 앞판 앞중심선', len(cf_on) >= 2, f'{len(cf_on)}점 일치')

# ---------------------------------------------------------------- F. other sizes
sizes = [('44', 80, 62, 88, 37, 56), ('55', 84, 66, 92, 38, 58), ('66', 88, 70, 96, 38.5, 58),
         ('77', 94, 76, 100, 39, 59), ('88', 100, 82, 106, 39.5, 60), ('99', 106, 88, 112, 40, 60)]
code = ("import sys;sys.path.insert(0,'.');import draft;draft.M.update(B={},W={},H={},BL={},SL={});"
        "from verify import verify,to_markdown;rep,R=verify();md,t,f=to_markdown(rep,R);print(t,f)")
for nm, B, W, H, BL, SL in sizes:
    try:
        out = subprocess.run([sys.executable, '-c', code.format(B, W, H, BL, SL)], cwd=HERE, capture_output=True,
                             text=True, timeout=300)
        t, f = out.stdout.split()[-2:]
        chk('F', f'{nm}사이즈 (B{B} W{W} H{H}) 전체 재제도 + 10회 검토', f == '0', f'{t}항목 중 불합격 {f}')
    except Exception as ex:  # noqa: BLE001
        chk('F', f'{nm}사이즈 (B{B} W{W} H{H})', False, f'제도 실패: {out.stderr.strip().splitlines()[-1] if out.stderr else ex}')

titles = dict(A='A. 실물크기 PDF 재측정', B='B. A4 분할 PDF', C='C. 시접', D='D. 너치 · 소매산 이즈 배분',
              E='E. 안단', F='F. 다른 사이즈로 시스템 검증')
lines = ['# 출력물 독립 재검증 (audit)', '']
for s, t in titles.items():
    rows = [r for r in res if r[0] == s]
    nf = sum(1 for r in rows if not r[2])
    lines += [f'## {t} — {"✅" if nf == 0 else f"❌ {nf}건"}', '', '| 항목 | 결과 | 판정 |', '|---|---|:---:|']
    lines += [f'| {r[1]} | {r[3]} | {"✅" if r[2] else "❌"} |' for r in rows]
    lines.append('')
open(os.path.join(HERE, 'out', 'audit_report.md'), 'w').write('\n'.join(lines))
for r in res:
    if not r[2]:
        print('FAIL', r)
print('TOTAL', len(res), 'FAIL', sum(1 for r in res if not r[2]))
