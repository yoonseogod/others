"""10-round independent verification of the jacket pattern (lengths, angles, curves)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
from shapely.geometry import LineString, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft import M, J, PCT, build, outline  # noqa: E402
from geom import (P, unit, plen, tan_start, tan_end, rotv, angdeg, inflections, max_turn, rot)  # noqa: E402

TOL_L = 0.01    # cm  (0.1 mm)
TOL_A = 0.10    # degrees


def leave(c, end):
    return tan_start(c) if end == 's' else -tan_end(c)


def ang(c1, e1, c2, e2):
    u, v = leave(c1, e1), leave(c2, e2)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))


class Report:
    def __init__(self):
        self.rounds = []

    def round(self, title):
        self.rounds.append(dict(title=title, checks=[]))

    def add(self, name, value, target, tol, unit_=''):
        ok = abs(value - target) <= tol
        self.rounds[-1]['checks'].append(dict(name=name, value=float(value), target=float(target), tol=tol,
                                              unit=unit_, ok=bool(ok)))
        return ok

    def cond(self, name, ok, info=''):
        self.rounds[-1]['checks'].append(dict(name=name, value=info, target='', tol='', unit='', ok=bool(ok)))

    def info(self, name, value, unit_=''):
        self.rounds[-1]['checks'].append(dict(name=name, value=value if isinstance(value, str) else float(value),
                                              target='(참고)', tol='', unit=unit_, ok=True))


def seg(R, piece, edge):
    for n, c in R['pieces'][piece]:
        if n == edge:
            return c
    raise KeyError(piece + '.' + edge)


def width_at(R, y, names):
    tot = 0.0
    ln = LineString([(-50, y), (150, y)])
    for n in names:
        poly = Polygon(outline(R['pieces'][n]))
        g = poly.intersection(ln)
        tot += g.length
    return tot


def remeasure_dense():
    """Rebuild everything with 4x denser sampling in a separate process -> lengths must converge."""
    code = ("import sys,json;sys.path.insert(0,'.');from draft import build;from geom import plen;R=build();C=R['curves'];"
            "K=R['K'];print(json.dumps(dict(AH=K['AH'],cap=K['cap_len'],neck=K['neck_len'],pc=plen(C['pc']),"
            "ps=plen(C['ps']),wrist=K['wrist_len'],sideb=plen(C['side'].edge(-1)))))")
    out = {}
    for n in (1200, 4800):
        env = dict(os.environ, PATTERN_N=str(n))
        r = subprocess.run([sys.executable, '-c', code], cwd=os.path.dirname(os.path.abspath(__file__)),
                           env=env, capture_output=True, text=True, check=True)
        out[n] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


def verify(R=None):
    R = R or build()
    K, C, rep = R['K'], R['curves'], Report()
    B, W, BL = M['B'], M['W'], M['BL']

    # ---------------------------------------------------------------- R1
    rep.round('1차 검토 - 문화식 원형 기준선·공식 재계산')
    rep.add('가슴선 깊이 B/12+13.7', BL - K['yC'], B / 12 + 13.7, 1e-9, 'cm')
    rep.add('몸판 폭 B/2+6', K['half'], B / 2 + 6, 1e-9, 'cm')
    rep.add('등품 B/8+7.4', K['xbw'], B / 8 + 7.4, 1e-9, 'cm')
    rep.add('앞품 B/8+6.2', K['CF'] - K['xfw'], B / 8 + 6.2, 1e-9, 'cm')
    rep.add('앞중심 높이 B/5+8.3 (가슴선→)', K['yFtop'] - K['yC'], B / 5 + 8.3, 1e-9, 'cm')
    rep.add('BP: 앞품/2+0.7', K['CF'] - K['BP'][0], (B / 8 + 6.2) / 2 + 0.7, 1e-9, 'cm')
    rep.add('옆선 = (등품점+(앞품점-B/32))/2', K['xs'], (K['xbw'] + K['xfw'] - B / 32) / 2, 1e-9, 'cm')
    rep.add('뒷목너비 B/24+3.4', K['nb'], B / 24 + 3.4, 1e-9, 'cm')
    rep.add('뒷목높이 ◎/3', K['SNPb'][1] - BL, K['nb'] / 3, 1e-9, 'cm')
    rep.add('앞목너비 ◎-0.2', K['CF'] - K['SNPf'][0], K['nb'] - 0.2, 1e-9, 'cm')
    rep.add('앞목깊이 ◎+0.5', K['yFtop'] - K['CFn'][1], K['nb'] + 0.5, 1e-9, 'cm')
    rep.add('앞어깨 기울기 22°', -angdeg(K['SPf_blk'] - K['SNPf']) - 180 + 360 if False else
            math.degrees(math.atan2(K['SNPf'][1] - K['SPf_blk'][1], K['SNPf'][0] - K['SPf_blk'][0])), 22, 1e-9, '°')
    rep.add('뒤어깨 기울기 18°', math.degrees(math.atan2(K['SNPb'][1] - K['SPb'][1], K['SPb'][0] - K['SNPb'][0])),
            18, 1e-9, '°')
    rep.add('뒤어깨 = 앞어깨▲ + (B/32-0.8)', np.linalg.norm(K['SPb'] - K['SNPb']), K['tri'] + B / 32 - 0.8, 1e-9, 'cm')
    rep.add('가슴다트 각 (B/4-2.5)°', math.degrees(math.atan2(*(K['G2'] - K['BP'])[::-1])) -
            math.degrees(math.atan2(*(K['G'] - K['BP'])[::-1])), B / 4 - 2.5, 1e-9, '°')
    rep.add('가슴다트 두 다리 길이 차', np.linalg.norm(K['G'] - K['BP']) - np.linalg.norm(K['G2'] - K['BP']), 0, 1e-9, 'cm')
    rep.add('어깨다트 두 다리 길이 차', np.linalg.norm(K['P1'] - K['D']) - np.linalg.norm(K['P2'] - K['D']), 0, 1e-9, 'cm')
    rep.add('어깨다트 분량 B/32-0.8', np.linalg.norm(K['P1'] - K['P2']), B / 32 - 0.8, 1e-9, 'cm')

    # ---------------------------------------------------------------- R2
    rep.round('2차 검토 - 어깨선 · 목둘레선 (길이 + 각)')
    ysh, fsh = seg(R, 'yoke', 'shoulder'), seg(R, 'front', 'shoulder')
    rep.add('어깨솔기: 요크 - 앞', plen(ysh) - plen(fsh), 0, TOL_L, 'cm')
    rep.add('뒷목둘레 ∠뒤중심 (직각)', ang(seg(R, 'yoke', 'fold_cb'), 'e', seg(R, 'yoke', 'neck'), 's'), 90, TOL_A, '°')
    a1 = ang(seg(R, 'yoke', 'neck'), 'e', ysh, 's')
    a2 = ang(fsh, 's', seg(R, 'front', 'neck'), 'e')
    rep.add('뒷목둘레 ∠어깨 (요크)', a1, 90, TOL_A, '°')
    rep.add('앞목둘레 ∠어깨', a2, 90, TOL_A, '°')
    rep.add('목둘레 어깨점 합 (앞+뒤 = 180°, 꺾임 없음)', a1 + a2, 180, 2 * TOL_A, '°')
    rep.add('앞목둘레 ∠앞중심 (직각)', ang(seg(R, 'front', 'cf'), 'e', seg(R, 'front', 'neck'), 's'), 90, TOL_A, '°')
    rep.add('목둘레선 변곡점 수 (뒤)', inflections(C['neck_b']), 0, 0)
    rep.add('목둘레선 변곡점 수 (앞)', inflections(C['neck_f']), 0, 0)
    rep.info('뒷목둘레 (요크)', plen(C['neck_b']), 'cm')
    rep.info('앞목둘레', plen(C['neck_f']), 'cm')

    # ---------------------------------------------------------------- R3
    rep.round('3차 검토 - 진동둘레 교차각 4곳 + 곡선 매끄러움')
    ya, fa = seg(R, 'yoke', 'armhole'), seg(R, 'front', 'armhole')
    b1, b2 = ang(ysh, 'e', ya, 's'), ang(fsh, 'e', fa, 's')
    rep.add('어깨끝점 요크 ∠', b1, 90, TOL_A, '°')
    rep.add('어깨끝점 앞 ∠', b2, 90, TOL_A, '°')
    rep.add('어깨끝점 합 = 180°', b1 + b2, 180, 2 * TOL_A, '°')
    c1 = ang(ya, 'e', seg(R, 'yoke', 'yoke_seam_out'), 's')
    sba = seg(R, 'side_back', 'armhole')
    c2 = ang(seg(R, 'side_back', 'top'), 'e', sba, 's')
    rep.add('요크솔기 위치 진동 합 = 180°', c1 + c2, 180, 2 * TOL_A, '°')
    fsa = seg(R, 'side_front', 'armhole')
    d1 = ang(fa, 'e', seg(R, 'front', 'princess'), 's')
    d2 = ang(seg(R, 'side_front', 'princess'), 'e', fsa, 's')
    rep.add('프린세스 위치 진동 합 = 180°', d1 + d2, 180, 2 * TOL_A, '°')
    e1 = ang(sba, 'e', seg(R, 'side_back', 'side_seam'), 's')
    e2 = ang(fsa, 'e', seg(R, 'side_front', 'side_seam'), 's')
    rep.add('옆선 위치 진동 뒤 ∠', e1, 90, TOL_A, '°')
    rep.add('옆선 위치 진동 앞 ∠', e2, 90, TOL_A, '°')
    rep.add('옆선 위치 진동 합 = 180°', e1 + e2, 180, 2 * TOL_A, '°')
    for nm, c in (('요크 진동', ya), ('뒤옆판 진동', sba), ('앞판 진동', fa), ('앞옆판 진동', fsa)):
        rep.add(f'{nm} 변곡점 수', inflections(c), 0, 0)
        rep.cond(f'{nm} 꺾임 없음 (샘플간 최대 방향변화 < 1°)', max_turn(c) < 1.0, f'{max_turn(c):.3f}°')
    # garment armhole assembled with dart closed -> check continuous tangent at G
    lowc = rot(C['low_f'], K['BP'], -K['dartf'])
    rep.add('가슴다트 닫은 상태 진동 이음점 간격', np.linalg.norm(lowc[0] - C['up_f'][-1]), 0, 1e-6, 'cm')
    rep.add('가슴다트 닫은 상태 진동 접선차', math.degrees(math.acos(min(1, float(tan_end(C['up_f']) @ tan_start(lowc))))),
            0, TOL_A, '°')
    def minrad(c):
        d1 = np.gradient(c, axis=0)
        d2 = np.gradient(d1, axis=0)
        kk = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        return 1 / kk.max()
    rb = min(minrad(rot(C['arm_y'], K['D'], -K['thd'])), minrad(C['low_b']))
    rf = minrad(C['arm_fc'])
    rep.cond('앞 진동(다트 닫은 착용상태) 최소 곡률반경 ≥ 2.5cm', rf >= 2.5, f'{rf:.2f}cm')
    rep.cond('뒤 진동 최소 곡률반경 ≥ 2.5cm', rb >= 2.5, f'{rb:.2f}cm')
    rep.add('앞 진동(다트 닫은 상태) 변곡점 수', inflections(C['arm_fc']), 0, 0)
    rep.info('앞품 감소 (진동이 앞품선 안쪽으로 들어간 양)', C['up_f'][:, 0].max() - K['xfw'], 'cm')
    rep.info('뒤 진동둘레 AH(b)', K['AHb'], 'cm')
    rep.info('앞 진동둘레 AH(f)', K['AHf'], 'cm')

    # ---------------------------------------------------------------- R4
    rep.round('4차 검토 - 세로 솔기 길이 (전체 / 노치 구간별)')
    N_ = R['notches']
    pairs = [('옆선', seg(R, 'side_back', 'side_seam'), seg(R, 'side_front', 'side_seam'), 'side_waist'),
             ('V솔기', seg(R, 'cb_panel', 'v_seam'), rev_(seg(R, 'mid_back', 'v_seam')), 'v_waist'),
             ('바깥 뒤솔기', seg(R, 'mid_back', 'outer_seam'), rev_(seg(R, 'side_back', 'outer_seam')), 'o_waist'),
             ('프린세스', seg(R, 'front', 'princess'), rev_(seg(R, 'side_front', 'princess')), 'p_waist')]
    for nm, a, b, wn in pairs:
        rep.add(f'{nm} 전체 길이차', plen(a) - plen(b), 0, TOL_L, 'cm')
        wa, wb = N_[wn]
        ia = int(np.argmin(np.linalg.norm(a - wa, axis=1)))
        ib = int(np.argmin(np.linalg.norm(b - wb, axis=1)))
        rep.add(f'{nm} 위→허리노치 길이차', plen(a[: ia + 1]) - plen(b[: ib + 1]), 0, 0.02, 'cm')
        rep.add(f'{nm} 허리노치→밑단 길이차', plen(a[ia:]) - plen(b[ib:]), 0, 0.02, 'cm')
        rep.info(f'{nm} 길이', plen(a), 'cm')
    a, b = seg(R, 'front', 'princess'), rev_(seg(R, 'side_front', 'princess'))
    ia = int(np.argmin(np.linalg.norm(a - K['BP'], axis=1)))
    ib = int(np.argmin(np.linalg.norm(b - K['BP'], axis=1)))
    rep.add('프린세스 위→BP노치 길이차', plen(a[: ia + 1]) - plen(b[: ib + 1]), 0, 0.02, 'cm')
    pc = C['pc']
    m = (pc[:, 1] > 0) & (pc[:, 1] < K['BP'][1])
    rep.cond('앞판 프린세스: BP~허리 사이 허리점보다 튀어나옴 없음', pc[m, 0].max() <= K['Wc'][0] + 1e-3,
             f'{pc[m, 0].max() - K["Wc"][0]:+.3f}cm')
    rep.add('허리 흡수량 합 = 몸판폭 - (W+6)/2', K['sV'] + K['sO'] + K['sS'] + K['sP'], K['half'] - (W + 6) / 2, 1e-9, 'cm')

    # ---------------------------------------------------------------- R5
    rep.round('5차 검토 - 밑단 교차각 + 밑단 곡선')
    h = {p: seg(R, p, 'hem') for p in ('cb_panel', 'mid_back', 'side_back', 'side_front', 'front')}
    j = []
    j.append(('뒤중심 밑단 ∠ (직각)', ang(h['cb_panel'], 'e', seg(R, 'cb_panel', 'fold_cb'), 's'), 90))
    x1 = ang(seg(R, 'cb_panel', 'v_seam'), 'e', h['cb_panel'], 's')
    x2 = ang(h['mid_back'], 'e', seg(R, 'mid_back', 'v_seam'), 's')
    y1 = ang(seg(R, 'mid_back', 'outer_seam'), 'e', h['mid_back'], 's')
    y2 = ang(h['side_back'], 'e', seg(R, 'side_back', 'outer_seam'), 's')
    z1 = ang(seg(R, 'side_back', 'side_seam'), 'e', h['side_back'], 's')
    z2 = ang(seg(R, 'side_front', 'side_seam'), 'e', h['side_front'], 's')
    w1 = ang(h['side_front'], 'e', seg(R, 'side_front', 'princess'), 's')
    w2 = ang(seg(R, 'front', 'princess'), 'e', h['front'], 's')
    for nm, u1, u2 in (('V솔기', x1, x2), ('바깥 뒤솔기', y1, y2), ('옆선', z1, z2), ('프린세스', w1, w2)):
        rep.add(f'{nm} 밑단 각(좌)', u1, 90, TOL_A, '°')
        rep.add(f'{nm} 밑단 각(우)', u2, 90, TOL_A, '°')
        rep.add(f'{nm} 밑단 합 = 180°', u1 + u2, 180, 2 * TOL_A, '°')
    rep.add(*j[0], TOL_A, '°')
    rep.add('앞중심 둥근 모서리 접선 연속 (180°)', ang(h['front'], 'e', seg(R, 'front', 'cf'), 's'), 180, TOL_A, '°')
    for p, c in h.items():
        rep.cond(f'{p} 밑단 꺾임 없음', max_turn(c) < 1.0, f'{max_turn(c):.3f}°')

    # ---------------------------------------------------------------- R6
    rep.round('6차 검토 - 요크 솔기 · 어깨다트 처리')
    yb = plen(seg(R, 'yoke', 'yoke_seam_out')) + plen(seg(R, 'yoke', 'yoke_seam_in'))
    bt = sum(plen(seg(R, p, 'top')) for p in ('cb_panel', 'mid_back', 'side_back'))
    rep.add('요크 아랫선 = 뒤판 3장 윗선 합', yb - bt, 0, TOL_L, 'cm')
    rep.add('어깨다트 닫힘 (P2 회전 → P1)', np.linalg.norm(rot(K['P2'], K['D'], K['thd']) - K['P1']), 0, 1e-9, 'cm')
    rep.add('요크 뒤중심 ∠목둘레 (직각)', ang(seg(R, 'yoke', 'fold_cb'), 'e', seg(R, 'yoke', 'neck'), 's'), 90, TOL_A, '°')
    rep.add('요크 뒤중심 ∠아랫선 (직각)', ang(seg(R, 'yoke', 'yoke_seam_in'), 'e', seg(R, 'yoke', 'fold_cb'), 's'), 90, TOL_A, '°')
    rep.add('뒤중심판 윗선 ∠뒤중심 (직각)', ang(seg(R, 'cb_panel', 'fold_cb'), 'e', seg(R, 'cb_panel', 'top'), 's'), 90, TOL_A, '°')
    rep.add('요크 O노치 = 중간판 윗선 길이', np.linalg.norm(K['Oy'] - K['D']) - plen(seg(R, 'mid_back', 'top')), 0, 1e-9, 'cm')
    rep.info('어깨다트 닫힘 회전각 (D점 기준)', K['thd'], '°')
    rep.info('요크 어깨선 길이', K['Lsh'], 'cm')

    # ---------------------------------------------------------------- R7
    rep.round('7차 검토 - 소매산 (이즈 · 각 · 곡)')
    rep.add('소매산 곡선 - 진동둘레 = 목표 이즈', K['cap_len'] - K['AH'], J['cap_ease'], 0.005, 'cm')
    rep.add('소매산 높이 = 평균 진동깊이 × 5/6', K['cap_h'], K['depth'] * 5 / 6, 1e-9, 'cm')
    cap = C['cap']
    iT = int(np.argmin(np.linalg.norm(cap, axis=1)))
    tT = unit(cap[iT + 1] - cap[iT - 1])
    rep.add('소매산 꼭지점 접선 수평', abs(angdeg(tT)), 0, TOL_A, '°')
    s1 = ang(seg(R, 'sleeve', 'underarm_back_up'), 'e', seg(R, 'sleeve', 'cap_back'), 's')
    s2 = ang(seg(R, 'sleeve', 'cap_front'), 'e', seg(R, 'sleeve', 'underarm_front'), 's')
    rep.add('소매산 뒤 겨드랑점 ∠ (직각)', s1, 90, TOL_A, '°')
    rep.add('소매산 앞 겨드랑점 ∠ (직각)', s2, 90, TOL_A, '°')
    rep.add('겨드랑 봉합 시 소매산 합 = 180°', s1 + s2, 180, 2 * TOL_A, '°')
    rep.add('소매산 뒤 변곡점 1개 (S자)', inflections(C['cap_b']), 1, 0)
    rep.add('소매산 앞 변곡점 1개 (S자)', inflections(C['cap_f']), 1, 0)
    rep.cond('소매산 꺾임 없음', max_turn(cap) < 1.0, f'{max_turn(cap):.3f}°')
    eb = plen(C['cap'][: int(np.argmin(np.linalg.norm(cap - R['notches']['sleeve_sh'][0], axis=1))) + 1])
    rep.add('어깨노치까지 뒤소매산 - 뒤AH', eb - K['AHb'], J['cap_ease'] * K['AHb'] / K['AH'], 0.01, 'cm')
    rep.info('소매산 길이', K['cap_len'], 'cm')
    rep.info('어깨노치 위치 (꼭지점에서 앞쪽 +)', K['sh_offset'], 'cm')

    # ---------------------------------------------------------------- R8
    rep.round('8차 검토 - 소매 옆선 · 팔꿈치다트 · 소매부리')
    fu = seg(R, 'sleeve', 'underarm_front')
    bu = plen(seg(R, 'sleeve', 'underarm_back_up')) + plen(seg(R, 'sleeve', 'underarm_back_lo'))
    rep.add('소매 옆선 앞 - 뒤(다트 제외)', plen(fu) - bu, 0, TOL_L, 'cm')
    rep.add('팔꿈치노치 위 구간 앞-뒤', plen(C['fe1']) - plen(C['be1']), 0, TOL_L, 'cm')
    rep.add('팔꿈치다트 다리 길이차', np.linalg.norm(K['apex'] - K['E1']) - np.linalg.norm(K['apex'] - K['E2']), 0, 1e-9, 'cm')
    rep.add('팔꿈치다트 분량', np.linalg.norm(K['E1'] - K['E2']), J['elbow_dart'], 1e-6, 'cm')
    w = seg(R, 'sleeve', 'wrist')
    q1 = ang(fu, 'e', w, 's')
    q2 = ang(w, 'e', seg(R, 'sleeve', 'underarm_back_lo'), 's')
    rep.add('소매부리 앞 ∠ (직각)', q1, 90, TOL_A, '°')
    rep.add('소매부리 뒤 ∠ (직각)', q2, 90, TOL_A, '°')
    rep.add('소매부리 봉합 시 합 = 180°', q1 + q2, 180, 2 * TOL_A, '°')
    rep.add('소매부리 폭', np.linalg.norm(K['Wf'] - K['Wb']), J['sleeve_wrist'], 1e-9, 'cm')
    rep.add('소매길이 (산꼭지→소매부리선)', -K['Wf'][1], M['SL'], 1e-9, 'cm')

    # ---------------------------------------------------------------- R9
    rep.round('9차 검토 - 칼라 · 러플 · 탭/벨트')
    cn = seg(R, 'collar', 'neck_edge')
    rep.add('칼라 목선 = 몸판 목둘레(앞+뒤)', plen(cn) - (plen(C['neck_b']) + plen(C['neck_f'])), 0, 1e-6, 'cm')
    rep.add('칼라 뒤중심 ∠목선 (직각)', ang(seg(R, 'collar', 'fold_cb'), 'e', cn, 's'), 90, TOL_A, '°')
    rep.add('칼라 앞끝 ∠목선 (직각)', ang(cn, 'e', seg(R, 'collar', 'cf_edge'), 's'), 90, TOL_A, '°')
    rep.add('칼라 앞끝 ∠윗선 (직각)', ang(seg(R, 'collar', 'cf_edge'), 'e', seg(R, 'collar', 'top_edge'), 's'), 90, TOL_A, '°')
    rep.add('칼라 윗선 ∠뒤중심 (직각)', ang(seg(R, 'collar', 'top_edge'), 'e', seg(R, 'collar', 'fold_cb'), 's'), 90, TOL_A, '°')
    sn = R['notches']['collar_snp'][0]
    i = int(np.argmin(np.linalg.norm(cn - sn, axis=1)))
    rep.add('칼라 SNP노치 = 뒷목둘레', plen(cn[: i + 1]) - plen(C['neck_b']), 0, 0.01, 'cm')
    rep.add('칼라 높이', np.linalg.norm(seg(R, 'collar', 'cf_edge')[-1] - seg(R, 'collar', 'cf_edge')[0]), J['collar_h'], 1e-9, 'cm')
    ri = seg(R, 'ruffle', 'inner')
    rep.add('러플 안쪽 호 = 소매부리 길이', plen(ri) - K['wrist_len'], 0, 0.005, 'cm')
    rep.add('러플 끝선 ∠안쪽호 (직각)', ang(ri, 'e', seg(R, 'ruffle', 'end2'), 's'), 90, TOL_A, '°')
    rep.add('러플 폭', np.linalg.norm(seg(R, 'ruffle', 'end1')[0] - seg(R, 'ruffle', 'end1')[-1]), J['ruffle_w'], 1e-9, 'cm')
    rep.add('벨트 길이 = 완성허리 + 1 + 18(겹침)', K['belt_len'], (W + J['waist_ease']) + 1 + 18, 1e-9, 'cm')
    rep.cond('탭 5개가 앞중심선 위, 허리선 위에 위치', all(0 < y < K['CFnj'][1] for y in K['tab_y']),
             ', '.join(f'{y:.1f}' for y in K['tab_y']))

    # ---------------------------------------------------------------- R10
    rep.round('10차 검토 - 완성치수 · 패턴 폐합 · 꺾임 전수검사 · 독립 재측정')
    body = ['cb_panel', 'mid_back', 'side_back', 'side_front', 'front']
    bust = 2 * width_at(R, K['yA'] - 1e-6, body)
    waist = 2 * width_at(R, 0.0, body)
    rep.cond('완성 가슴둘레(진동 밑) ≥ B + 12 (문화식 기본여유)', bust >= B + 12, f'{bust:.2f}cm')
    chest_blk = 2 * width_at(R, K['yC'], ['cb_panel', 'mid_back'])
    rep.add('뒤판 가슴선 폭 = 원형 등쪽 폭 (가슴선 위 흡수 0)', chest_blk / 2 + 0,
            float(np.interp(K['yC'], *_xedge(R))), 0.005, 'cm')
    rep.add('완성 허리둘레 = W + 6', waist, W + J['waist_ease'], 0.02, 'cm')
    lvl = J['hem']['side'] + 1
    hipw = 2 * width_at(R, lvl, body)
    rep.cond(f'페플럼 둘레(y={lvl}) ≥ 엉덩이둘레 H', hipw >= M['H'], f'{hipw:.2f}cm (H={M["H"]})')
    worst = 0.0
    for pn, piece in R['pieces'].items():
        gaps = [np.linalg.norm(piece[i][1][-1] - piece[(i + 1) % len(piece)][1][0]) for i in range(len(piece))]
        rep.add(f'[{pn}] 외곽선 폐합 (이음 간격 최대)', max(gaps), 0, 1e-6, 'cm')
        rep.cond(f'[{pn}] 자기교차 없음', Polygon(outline(piece)).is_valid)
        for en, c in piece:
            if en in ('elbow_dart',) or pn in ('tab', 'belt') or len(c) < 5:
                continue
            worst = max(worst, max_turn(c))
    rep.cond('모든 곡선 꺾임 없음 (샘플간 최대 방향변화 < 1°)', worst < 1.0, f'{worst:.3f}°')
    dense = remeasure_dense()
    for k in dense[1200]:
        rep.add(f'4배 조밀 재측정 수렴: {k}', dense[4800][k] - dense[1200][k], 0, 0.005, 'cm')
    return rep, R


def _xedge(R):
    # x of the outer-seam centre line on the left panel side at chest level (no suppression there)
    oL = seg(R, 'mid_back', 'outer_seam')
    o = oL[np.argsort(oL[:, 1])]
    return o[:, 1], o[:, 0]


def rev_(c):
    return c[::-1].copy()


def to_markdown(rep, R):
    K = R['K']
    lines = ['# 자켓 패턴 검증 보고서 (10회 재검토)', '',
             f"치수: B{M['B']:.0f} / W{M['W']:.0f} / H{M['H']:.0f} / 등길이 {M['BL']:.0f} / 소매길이 {M['SL']:.0f} cm "
             f"(한국 여성 55/M 표준)", '',
             '허용오차: 길이 ±0.01cm(0.1mm), 각 ±0.1° — 실제값은 대부분 1e-6 이하', '']
    total = sum(len(r['checks']) for r in rep.rounds)
    fails = sum(1 for r in rep.rounds for c in r['checks'] if not c['ok'])
    lines += [f'**전체 {total}개 항목 중 불합격 {fails}개**', '']
    for i, r in enumerate(rep.rounds, 1):
        nf = sum(1 for c in r['checks'] if not c['ok'])
        lines += [f'## {r["title"]}  —  {"✅ 통과" if nf == 0 else f"❌ {nf}건"}', '',
                  '| 항목 | 측정값 | 기준 | 판정 |', '|---|---:|---:|:---:|']
        for c in r['checks']:
            v = c['value']
            vs = f'{v:.4f}{c["unit"]}' if isinstance(v, float) else str(v)
            t = c['target']
            ts = f'{t:.4f}{c["unit"]} ±{c["tol"]}' if isinstance(t, float) else str(t)
            lines.append(f'| {c["name"]} | {vs} | {ts} | {"✅" if c["ok"] else "❌"} |')
        lines.append('')
    return '\n'.join(lines), total, fails


if __name__ == '__main__':
    rep, R = verify()
    md, total, fails = to_markdown(rep, R)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'verification_report.md'), 'w').write(md)
    for r in rep.rounds:
        bad = [c for c in r['checks'] if not c['ok']]
        print(f"{r['title']}: {len(r['checks'])} checks, {len(bad)} fail")
        for c in bad:
            print('   FAIL', c['name'], c['value'], c['target'])
    print('TOTAL', total, 'FAIL', fails)
