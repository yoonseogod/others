# -*- coding: utf-8 -*-
"""
밀리터리 롱코트 패턴 제도 + A4 타일 PDF 출력 (실물 1:1)
단위: cm, 좌표계: x→오른쪽, y→아래쪽
치수를 바꾸려면 MEAS 를 수정 후 실행:  python3 draft_coat.py
"""
import math, sys, os
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPoint
from shapely.ops import substring, nearest_points
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ------------------------------------------------------------------ 치수
MEAS = dict(
    size_name="여성 M (55~66)",
    bust=88, waist=70, hip=94,
    back_len=38,      # 뒷목점~허리
    shoulder=39,      # 어깨너비
    arm=56,           # 팔길이
    coat_len=115,     # 뒷목점~밑단
    sleeve_len=60,    # 소매산~소매끝 (커프스 포함 길이)
    bust_dart_deg=8,  # 가슴 다트 각(남성은 2~3 정도로)
)
EASE = dict(bust=14, waist=12, hip=14)
SA, HEM_SA = 1.0, 4.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

# ------------------------------------------------------------------ 기하 도구
P = lambda x, y: np.array([x, y], float)

def bez(p0, p1, p2, p3, n=40):
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def cr(pts, n=30):
    """centripetal Catmull-Rom (점들을 지나는 부드러운 곡선)"""
    pts = [np.asarray(p, float) for p in pts]
    ext = [2*pts[0]-pts[1]] + pts + [2*pts[-1]-pts[-2]]
    out = []
    for i in range(1, len(ext)-2):
        P0, P1, P2, P3 = ext[i-1], ext[i], ext[i+1], ext[i+2]
        t0 = 0
        t1 = t0 + np.linalg.norm(P1-P0)**0.5
        t2 = t1 + np.linalg.norm(P2-P1)**0.5
        t3 = t2 + np.linalg.norm(P3-P2)**0.5
        t = np.linspace(t1, t2, n)[:, None]
        A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
        A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
        A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
        B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
        B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
        C = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
        out.append(C if i == 1 else C[1:])
    return np.vstack(out)

def seg(a, b):
    return np.array([a, b], float)

def join(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        out.append(np.asarray(p, float)[1:])
    return np.vstack(out)

def rev(a):
    return np.asarray(a)[::-1]

def L(a):
    a = np.asarray(a)
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))

def rot(p, c, deg):
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return c + R @ (np.asarray(p) - c)

def reflect(p, a, b):
    a, b, p = map(np.asarray, (a, b, p))
    u = (b-a)/np.linalg.norm(b-a)
    d = p-a
    return a + 2*np.dot(d, u)*u - d

def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)

# ------------------------------------------------------------------ 조각 클래스
class Piece:
    def __init__(self, code, name, cut, edges, wl=None, hl=None, extra=None):
        """edges: [(pts, sa, kind)]  kind: 'seam' | 'fold' | 'hem' """
        self.code, self.name, self.cut = code, name, cut
        self.edges = [(np.asarray(e[0], float), e[1], e[2] if len(e) > 2 else 'seam') for e in edges]
        for i in range(len(self.edges)):
            a = self.edges[i][0][-1]; b = self.edges[(i+1) % len(self.edges)][0][0]
            assert np.linalg.norm(a-b) < 1e-6, (code, i, a, b)
        self.stitch = np.vstack([e[0][:-1] for e in self.edges] + [self.edges[0][0][:1]])
        self.poly = Polygon(self.stitch).buffer(0)
        self.wl, self.hl = wl, hl
        self.extra = extra or {}           # lines/marks
        self.notch_pts = []                # (point, edge_index)
        self.cut_line = self._sa_outline()
        self.cut_poly = Polygon(self.cut_line).buffer(0)
        self.offset = P(0, 0)

    def _sa_outline(self):
        offs = []
        for pts, sa, kind in self.edges:
            ls = LineString(pts)
            if sa == 0:
                o = np.asarray(ls.coords)
            else:
                cand = []
                for d in (sa, -sa):
                    oc = ls.offset_curve(d, quad_segs=8, join_style='round')
                    if oc.geom_type != 'LineString' or oc.is_empty:
                        continue
                    c = np.asarray(oc.coords)
                    if np.linalg.norm(c[0]-pts[0]) > np.linalg.norm(c[-1]-pts[0]):
                        c = c[::-1]
                    mid = LineString(c).interpolate(0.5, normalized=True)
                    cand.append((self.poly.contains(mid), c))
                o = [c for inside, c in cand if not inside][0]
            E = 6.0
            s = o[0] - E*unit(o[1]-o[0]) if len(o) > 1 else o[0]
            e = o[-1] + E*unit(o[-1]-o[-2]) if len(o) > 1 else o[-1]
            offs.append(np.vstack([s, o, e]))
        n = len(offs)
        corners = []
        for i in range(n):
            a = LineString(offs[i]); b = LineString(offs[(i+1) % n])
            v = Point(self.edges[i][0][-1])
            x = a.intersection(b)
            pts = []
            if not x.is_empty:
                geoms = getattr(x, 'geoms', [x])
                for g in geoms:
                    if g.geom_type == 'Point':
                        pts.append(g)
                    else:
                        pts.extend(Point(c) for c in g.coords)
            if pts:
                c = min(pts, key=lambda q: q.distance(v))
            else:
                c = Point((offs[i][-2] + offs[(i+1) % n][1])/2)
            corners.append(c)
        # 너무 뾰족한 모서리는 시접을 잘라(bevel) 처리
        bevel = []
        for i in range(n):
            sa = max(self.edges[i][1], self.edges[(i+1) % n][1])
            v = Point(self.edges[i][0][-1])
            bevel.append(sa > 0 and corners[i].distance(v) > 2.5*sa)
        out = []
        for i in range(n):
            ls = LineString(offs[i])
            E = 6.0
            d0 = E if bevel[i-1] else ls.project(corners[i-1])
            d1 = ls.length - E if bevel[i] else ls.project(corners[i])
            if d1 <= d0:
                d0, d1 = 0, ls.length
            sub = substring(ls, d0, d1)
            out.append(np.asarray(sub.coords))
        return np.vstack(out)

    def bbox(self):
        x0, y0, x1, y1 = self.cut_poly.bounds
        return x0, y0, x1, y1

    def add_notch_y(self, y, edge_idx_list):
        hl = LineString([(-500, y), (500, y)])
        for i in edge_idx_list:
            x = LineString(self.edges[i][0]).intersection(hl)
            for g in getattr(x, 'geoms', [x]):
                if g.geom_type == 'Point':
                    self.notch_pts.append(np.array(g.coords[0]))

    def add_notch(self, p):
        self.notch_pts.append(np.asarray(p, float))

# ------------------------------------------------------------------ 제도
def draft(M):
    hb = (M['bust']+EASE['bust'])/2      # 반 가슴둘레
    hw = (M['waist']+EASE['waist'])/2
    hh = (M['hip']+EASE['hip'])/2
    bk_chest = hb/2 - 0.5
    fr_chest = hb/2 + 0.5
    nw = M['bust']/20 + 3.1               # 목너비
    hs = M['shoulder']/2 + 1.0            # 반 어깨너비(코트)
    WLb = 2.5 + M['back_len']             # 뒤 허리선 y
    HLb = WLb + 20
    HEMb = 2.5 + M['coat_len']
    WLf, HLf = WLb + 2, HLb + 2
    HEMf = HEMb + 2
    chest_b = 26.5
    chest_f = chest_b + 2
    info = {}

    # ---------------- 뒤판
    CBn, HNPb, SPb = P(0, 2.5), P(nw, 0), P(hs, 4.5)
    Pb = P(hs-1.1, 14.5)
    SCb = P(bk_chest, chest_b)
    neck_b = bez(CBn, (3, 2.5), (nw-1, 1.6), HNPb)
    sh_b = seg(HNPb, SPb)
    ah_b1 = bez(SPb, SPb+(-0.5, 3.5), Pb+(0.1, -3), Pb)
    ah_b2 = bez(Pb, Pb+(0.1, 5), SCb+(-3.5, -0.4), SCb)
    # 허리 줄임: CB 1.5, 프린세스 2.0, 옆 1.0
    cb_w, cb_h, cb_hem = 1.5, 1.2, 1.0
    side_w = bk_chest - 1.0
    side_h = hh/2 + cb_h
    pw = cb_w + (side_w - cb_w - hw/2) * 0 + 0   # placeholder
    dart_b = side_w - cb_w - hw/2
    pwc = 11.0; pws = pwc + dart_b
    ph = (pwc+pws)/2
    flare_p, flare_s = 6.5, 8.0
    pr_bc = cr([Pb, (15.0, 25), (pwc, WLb), (ph, HLb), (ph+flare_p, HEMb)])
    pr_bs = cr([Pb, (16.0, 25), (pws, WLb), (ph, HLb), (ph-flare_p, HEMb)])
    side_b = cr([SCb, (side_w, WLb), (side_h, HLb), (side_h+flare_s, HEMb)])
    cb = cr([CBn, (0, 20), (cb_w, WLb), (cb_h, HLb), (cb_hem, HEMb)])
    # 벤트: y=80 에서 시작, 폭 5
    vent_top = 80.0
    cbl = LineString(cb)
    k = int(np.argmin(np.abs(cb[:, 1]-vent_top)))
    cb_upper = cb[:k+1]
    vtx = cb[k]
    v1 = P(vtx[0]-5, vtx[1]+5)
    v2 = P(vtx[0]-5, HEMb)
    back_c = Piece('1', '뒤판 중심', '겉감 2장 (좌우 대칭)', [
        (neck_b, SA), (sh_b, SA), (ah_b1, SA), (pr_bc, SA),
        (seg(pr_bc[-1], v2), HEM_SA, 'hem'), (seg(v2, v1), SA), (seg(v1, vtx), SA),
        (rev(cb_upper), SA)], wl=WLb, hl=HLb)
    back_c.add_notch_y(WLb, [3, 7]); back_c.add_notch_y(HLb, [3, 7])
    back_c.extra['marks'] = [('벤트 트임 시작', vtx + P(1.0, 0))]

    back_s = Piece('2', '뒤판 옆', '겉감 2장 (좌우 대칭)', [
        (ah_b2, SA), (side_b, SA), (seg(side_b[-1], pr_bs[-1]), HEM_SA, 'hem'), (rev(pr_bs), SA)],
        wl=WLb, hl=HLb)
    back_s.add_notch_y(WLb, [1, 3]); back_s.add_notch_y(HLb, [1, 3])

    # ---------------- 앞판
    HNPf = P(nw, 0)
    shl = L(sh_b) - 0.7
    drop = 5.5
    SPf = HNPf + P(math.sqrt(shl**2-drop**2), drop)
    BP = P(9.5, 27)
    AP1 = P(SPf[0]-1.35, 15.5)
    th = M['bust_dart_deg']
    AP2 = rot(AP1, BP, th)
    SCf = P(fr_chest, chest_f)
    t1 = P(-0.1, 3.0)
    ah_f1 = bez(SPf, SPf+(-0.4, 3.5), AP1-t1, AP1)
    t2 = rot(AP1+t1, BP, th) - AP2
    ah_f2 = bez(AP2, AP2+t2*1.6, SCf+(-3.8, -0.3), SCf)
    sh_f = seg(HNPf, SPf)
    # 앞 허리 줄임
    side_wf = fr_chest - 2.0
    dart_f = side_wf - hw/2
    fwc = 9.0; fws = fwc + dart_f
    fph = (fwc+fws)/2 - 0.25
    C1 = P(12.0, 21.5)
    pr_fc = cr([AP1, C1, (BP[0]-0.3, BP[1]+1), (fwc, WLf), (fph, HLf), (fph+flare_p, HEMf)])
    pr_fs = cr([AP2, rot(C1, BP, th), (BP[0]+0.3, BP[1]+1), (fws, WLf), (fph, HLf), (fph-flare_p, HEMf)])
    side_hf = hh/2
    side_f = cr([SCf, (side_wf, WLf), (side_hf, HLf), (side_hf+flare_s, HEMf)])

    # 라펠 / 앞여밈
    ext = 7.0
    B = P(-ext, 24.0)
    R = HNPf + 2.5*unit(HNPf - SPf)
    def roll_x(y):
        return B[0] + (R[0]-B[0])*(B[1]-y)/(B[1]-R[1])
    T = P(8.5, 14.0)
    G = P(2.6, 10.2)
    Gr = P(roll_x(7.0), 7.0)
    n = unit(P(10, 15.5))*1.2
    lap_worn = bez(T, T+(-4, 2.6)+n, B+(4, -2.6)+n, B)
    Tp, Gp = reflect(T, B, R), reflect(G, B, R)
    lap = rev(np.array([reflect(q, B, R) for q in lap_worn]))   # B -> T'
    neck_f = bez(Gr, Gr+(2.2, -2.2), HNPf+(-0.3, 2.8), HNPf)   # Gr -> HNP
    gorge = seg(Gp, Gr)
    tab_y = 85.0
    fe1 = seg(P(-1, tab_y), P(-ext, tab_y))
    fe0 = seg(P(4, HEMf), P(-1, tab_y))
    fe2 = seg(P(-ext, tab_y), B)
    front_c = Piece('3', '앞판 중심 (라펠 포함)', '겉감 2장 (좌우 대칭)', [
        (sh_f, SA), (ah_f1, SA), (pr_fc, SA), (seg(pr_fc[-1], P(4, HEMf)), HEM_SA, 'hem'),
        (fe0, SA), (fe1, SA), (fe2, SA), (lap, SA), (seg(Tp, Gp), SA), (gorge, SA), (neck_f, SA)],
        wl=WLf, hl=HLf)
    front_c.add_notch_y(WLf, [2]); front_c.add_notch_y(HLf, [2])
    front_c.add_notch(B)
    front_c.extra['roll'] = (B, R)
    front_c.extra['cf'] = True
    front_c.extra['marks'] = [('BP', BP)]

    front_s = Piece('4', '앞판 옆', '겉감 2장 (좌우 대칭)', [
        (ah_f2, SA), (side_f, SA), (seg(side_f[-1], pr_fs[-1]), HEM_SA, 'hem'), (rev(pr_fs), SA)],
        wl=WLf, hl=HLf)
    front_s.add_notch_y(WLf, [1, 3]); front_s.add_notch_y(HLf, [1, 3])
    pk_y = HLf - 5
    front_s.extra['pocket'] = (P(fph+1.0, pk_y), P(fph+15.0, pk_y-0.6))

    # 안단
    Fs = HNPf + 4.0*unit(SPf-HNPf)
    inner = cr([Fs, (8.0, 14), (5.0, 30), (4.0, 50), (4.0, tab_y), (11.0, HEMf)])
    facing = Piece('5', '앞 안단', '겉감 2장 (좌우 대칭)', [
        (seg(HNPf, Fs), SA), (inner, SA), (seg(inner[-1], P(4, HEMf)), HEM_SA, 'hem'),
        (fe0, SA), (fe1, SA), (fe2, SA), (lap, SA), (seg(Tp, Gp), SA), (gorge, SA), (neck_f, SA)],
        wl=WLf, hl=HLf)
    facing.add_notch(B)
    facing.extra['roll'] = (B, R)
    facing.extra['cf'] = True

    # 뒤 목안단
    Fb = HNPb + 4.0*unit(SPb-HNPb)
    bn_in = bez(Fb, Fb+(-1.5, 5), (5, 10.5), (0, 10.5))
    back_facing = Piece('6', '뒤 목안단', '겉감 1장 (골)', [
        (neck_b, SA), (seg(HNPb, Fb), SA), (bn_in, SA), (seg(P(0, 10.5), CBn), 0, 'fold')])

    # ---------------- 소매 (두 장 소매)
    AH = L(ah_b1)+L(ah_b2)+L(ah_f1)+L(ah_f2)
    info['AH'] = AH

    def sleeve(h):
        SL = M['sleeve_len']; EL = 34
        top_cap = cr([(-12, 0.70*h), (-7.5, 0.30*h), (0.5, 0), (6.5, 0.22*h), (11, 0.55*h)])
        und_cap = cr([(-6, 0.70*h), (-3.5, 0.97*h), (0.5, h), (5, 0.88*h), (7, 0.55*h)])
        top_f = cr([(-12, 0.70*h), (-12.3, h), (-12.6, EL), (-12.0, SL)])
        top_b = cr([(11, 0.55*h), (11.5, h), (9.4, EL), (7.0, SL)])
        und_f = cr([(-6, 0.70*h), (-6.3, h), (-6.6, EL), (-6.0, SL)])
        und_b = cr([(7, 0.55*h), (7.5, h), (6.4, EL), (4.0, SL+0.3)])
        return top_cap, und_cap, top_f, top_b, und_f, und_b
    lo, hi = 10, 25
    target = AH + 3.0
    for _ in range(60):
        h = (lo+hi)/2
        tc, uc, *_ = sleeve(h)
        if L(tc)+L(uc) < target: lo = h
        else: hi = h
    h = (lo+hi)/2
    top_cap, und_cap, top_f, top_b, und_f, und_b = sleeve(h)
    info['cap_h'] = h; info['cap_len'] = L(top_cap)+L(und_cap)
    top_sl = Piece('7', '큰 소매 (윗소매)', '겉감 2장 (좌우 대칭)', [
        (top_cap, SA), (top_b, SA), (seg(top_b[-1], top_f[-1]), 3.0, 'hem'), (rev(top_f), SA)])
    top_sl.add_notch(P(0.5, 0)); top_sl.add_notch_y(34, [1, 3])
    und_sl = Piece('8', '작은 소매 (밑소매)', '겉감 2장 (좌우 대칭)', [
        (rev(und_cap), SA), (und_f, SA), (seg(und_f[-1], und_b[-1]), 3.0, 'hem'), (rev(und_b), SA)])
    und_sl.add_notch_y(34, [1, 3])

    # ---------------- 칼라
    bnl = L(neck_b); fnl = L(neck_f); gl = L(gorge)
    info['neck'] = (bnl, fnl, gl)
    # 스탠드 칼라 (CB 골)
    sh_ = 6.5
    s_len = bnl + fnl
    s_neck = join(seg(P(0, 0), P(bnl, 0)), bez(P(bnl, 0), P(bnl+fnl*0.4, 0), P(bnl+fnl*0.75, -0.4), P(s_len-0.05, -1.5)))
    # 실제 길이로 보정
    sc = s_len / L(s_neck)
    s_neck[:, 0] *= sc
    se = s_neck[-1]
    s_front = bez(se, se+(0.15, -1.5), se+(0.2, -3.5), se+P(-0.3, -sh_))
    st = s_front[-1]
    s_top = bez(st, st+(-1.5, 0.4), P(bnl, -sh_), P(0, -sh_))
    stand = Piece('9', '스탠드 칼라', '겉감 2장 (골, 겉/안)', [
        (s_neck, SA), (s_front, SA), (s_top, SA), (seg(P(0, -sh_), P(0, 0)), 0, 'fold')])
    stand.add_notch(P(bnl*sc, 0))

    # 윗칼라 (테일러드, CB 골)
    c_len = bnl + fnl + gl
    c_neck = join(seg(P(0, 0), P(bnl, 0)), bez(P(bnl, 0), P(bnl+5, 0), P(c_len-4, -1.0), P(c_len, -2.2)))
    sc2 = c_len / L(c_neck); c_neck[:, 0] *= sc2
    ce = c_neck[-1]
    CP = ce + P(3.0, -4.8)                     # 칼라 끝
    c_end = seg(ce, CP)
    c_out = bez(CP, CP+(-4, -1.6), P(bnl+2, -8.4), P(0, -8.4))
    collar = Piece('10', '칼라 (윗칼라)', '겉감 2장 (골, 겉/안)', [
        (c_neck, SA), (c_end, SA), (c_out, SA), (seg(P(0, -8.4), P(0, 0)), 0, 'fold')])
    collar.add_notch(P(bnl*sc2, 0)); collar.add_notch(c_neck[np.argmin(np.abs(np.cumsum(np.r_[0, np.linalg.norm(np.diff(c_neck, axis=0), axis=1)])-(bnl+fnl)))])
    collar.extra['marks'] = [('꺾임선(접히는 선)', P(2, -3.2))]
    collar.extra['rollline'] = cr([(0, -3.5), (bnl, -3.5), (c_len*0.9, -4.2)])

    # ---------------- 커프스
    sh_hem = (L(seg(top_b[-1], top_f[-1])) + L(seg(und_f[-1], und_b[-1])))
    cw_b = sh_hem + 2.0; cw_t = cw_b + 4.0; chh = 12.0
    c0 = P(0, chh); c1 = P(cw_b, chh); c2 = P(cw_b+2, 0); c3 = P(-2, 0)
    cuff = Piece('11', '커프스', '겉감 4장 (겉 2, 안 2)', [
        (seg(c0, c1), SA), (seg(c1, c2), SA), (bez(c2, c2+(-8, -0.6), c3+(8, -0.6), c3), SA), (seg(c3, c0), SA)])
    cuff.extra['marks'] = [('소매 뒷솔기 맞춤 ↔ 커프스 양끝', P(cw_b/2-7, chh-2))]

    # ---------------- 포켓 플랩
    fw, fh = 14.0, 6.0
    flap = Piece('12', '포켓 플랩', '겉감 4장 (겉 2, 안 2)', [
        (seg(P(0, 0), P(fw, 0)), SA), (seg(P(fw, 0), P(fw+0.3, fh)), SA),
        (seg(P(fw+0.3, fh), P(-0.3, fh)), SA), (seg(P(-0.3, fh), P(0, 0)), SA)])

    pieces = [front_c, front_s, back_c, back_s, facing, top_sl, und_sl, back_facing, stand, collar, cuff, flap]
    # 검증용 길이
    info['checks'] = [
        ('뒤 프린세스 (중심/옆)', L(pr_bc), L(pr_bs)),
        ('앞 프린세스 (중심/옆)', L(pr_fc), L(pr_fs)),
        ('옆선 (뒤/앞)', L(side_b), L(side_f)),
        ('어깨 (뒤/앞, 뒤 0.7 이즈)', L(sh_b), L(sh_f)),
        ('소매 앞솔기 (큰/작은)', L(top_f), L(und_f)),
        ('소매 뒷솔기 (큰/작은)', L(top_b), L(und_b)),
        ('진동둘레 / 소매산 둘레', AH, info['cap_len']),
        ('스탠드칼라 목둘레 / 몸판 목둘레(반)', L(s_neck), bnl+fnl),
        ('칼라 목둘레 / 목+고지(반)', L(c_neck), c_len),
    ]
    # 완성 치수
    info['finished'] = dict(
        bust=2*(bk_chest+fr_chest), waist=2*(side_w-cb_w-dart_b + side_wf-dart_f),
        hip=2*(side_h-cb_h + side_hf),
        hem=2*((pr_bc[-1][0]-v2[0]-5) + (side_b[-1][0]-pr_bs[-1][0]) + (pr_fc[-1][0]-4) + (side_f[-1][0]-pr_fs[-1][0])),
        length=HEMb-2.5, sleeve=M['sleeve_len'], cap_h=h)
    info['rects'] = [
        ('R1', '벨트 (선택: 원단 벨트일 때)', f'{11} × {int(round(M["waist"]+EASE["waist"]+28))} cm', '1장 (길면 뒤중심에서 이어 2장)', '4번 접어 완성 폭 약 4.5cm'),
        ('R2', '벨트 고리', '4 × 9 cm', '4장', '3번 접어 완성 폭 1cm'),
        ('R3', '플랩 포켓 주머니(안감)', '18 × 40 cm', '안감 2장', '반 접어 주머니'),
    ]
    return pieces, info

# ------------------------------------------------------------------ 배치 (skyline)
def pack(pieces, W, gap=1.6):
    items = sorted(pieces, key=lambda p: -(p.bbox()[3]-p.bbox()[1]))
    sky = [(0.0, W, 0.0)]   # (x0,x1,y)
    for p in items:
        x0, y0, x1, y1 = p.bbox()
        w, h = x1-x0+gap, y1-y0+gap
        best = None
        xs = sorted(set([s[0] for s in sky]))
        for x in xs:
            if x+w > W+1e-6: continue
            top = max(s[2] for s in sky if s[1] > x+1e-9 and s[0] < x+w-1e-9)
            if best is None or (top, x) < best[:2]:
                best = (top, x)
        top, x = best
        p.offset = P(x - x0 + gap/2, top - y0 + gap/2)
        new = []
        for s in sky:
            if s[1] <= x or s[0] >= x+w:
                new.append(s)
            else:
                if s[0] < x: new.append((s[0], x, s[2]))
                if s[1] > x+w: new.append((x+w, s[1], s[2]))
        new.append((x, x+w, top+h))
        sky = sorted(new)
    Hb = max(p.cut_poly.bounds[3]+p.offset[1] for p in pieces) + gap/2
    return Hb

# ------------------------------------------------------------------ PDF
PW, PH = 21.0, 29.7
MX, MY = 1.0, 1.0
TW, TH = PW-2*MX, PH-2*MY      # 19 x 27.7

def render(pieces, info, M, path):
    pdfmetrics.registerFont(TTFont('K', FONT, subfontIndex=0))
    best = None
    for cols in (6, 7, 8):
        for pc in pieces: pc.offset = P(0, 0)
        Hb = pack(pieces, cols*TW)
        rows = int(math.ceil(Hb/TH))
        used = []
        for r in range(rows):
            for c in range(cols):
                box = Polygon([(c*TW, r*TH), ((c+1)*TW, r*TH), ((c+1)*TW, (r+1)*TH), (c*TW, (r+1)*TH)])
                if any(box.intersects(Polygon(pc.cut_line + pc.offset)) for pc in pieces):
                    used.append((r, c))
        if best is None or len(used) < best[0]:
            best = (len(used), cols, rows, used, [pc.offset.copy() for pc in pieces])
    n_used, cols, rows, used, offs = best
    for pc, o in zip(pieces, offs): pc.offset = o
    print(f'board {cols}x{rows} tiles, printed pages {n_used}')

    cv = canvas.Canvas(path, pagesize=A4)
    cv.setTitle('Military Long Coat Pattern A4')
    rowname = lambda r: chr(ord('A')+r)

    # ---------- 1쪽: 안내
    def T(x, y, s, size=10, font='K', color=(0, 0, 0)):
        cv.setFont(font, size); cv.setFillColorRGB(*color); cv.drawString(x*cm, (PH-y)*cm, s)
    T(1.5, 2.2, '밀리터리 롱코트 패턴  (A4 타일 · 실물 1:1)', 18)
    T(1.5, 3.0, f'기준 사이즈: {M["size_name"]}   ·   시접 포함 (기본 {SA:g}cm, 밑단 {HEM_SA:g}cm, 소매단 3cm)', 10)
    y = 4.0
    T(1.5, y, '■ 기준 신체 치수 / 완성 치수', 12); y += 0.7
    f = info['finished']
    rows_t = [('가슴둘레', M['bust'], f['bust']), ('허리둘레', M['waist'], f['waist']), ('엉덩이둘레', M['hip'], f['hip']),
              ('등길이', M['back_len'], None), ('어깨너비', M['shoulder'], M['shoulder']+2), ('팔길이', M['arm'], f['sleeve']),
              ('총장(뒷목점~밑단)', None, f['length']), ('밑단 둘레', None, f['hem'])]
    T(1.8, y, '항목', 9); T(6.5, y, '신체', 9); T(9.0, y, '완성', 9); y += 0.5
    for nm, a, b in rows_t:
        T(1.8, y, nm, 9); T(6.5, y, '-' if a is None else f'{a:g} cm', 9); T(9.0, y, '-' if b is None else f'{b:.1f} cm', 9); y += 0.45
    # 테스트 사각형
    cv.setStrokeColorRGB(0, 0, 0); cv.setLineWidth(0.8)
    cv.rect(13.5*cm, (PH-4.0-5)*cm, 5*cm, 5*cm)
    T(13.7, 4.6, '테스트 사각형', 9); T(13.7, 5.1, '5 cm × 5 cm', 9)
    T(13.5, 9.6, '인쇄 후 자로 꼭 확인하세요.', 8)
    y += 0.5
    T(1.5, y, '■ 패턴 조각 (원단 재단 부분만, 부자재·장식 제외)', 12); y += 0.6
    for pc in sorted(pieces, key=lambda q: int(q.code)):
        T(1.8, y, f'{pc.code:>2}. {pc.name}', 9); T(9.0, y, pc.cut, 9); y += 0.45
    y += 0.3
    T(1.5, y, '■ 직사각형 재단 (패턴 없이 치수로 재단, 시접 포함)', 12); y += 0.6
    for code, nm, size, cnt, note in info['rects']:
        T(1.8, y, f'{code}. {nm}', 9); T(9.0, y, f'{size}  ·  {cnt}  ·  {note}', 8); y += 0.45
    y += 0.3
    T(1.5, y, '■ 안감', 12); y += 0.6
    for s in ['1·2·4·7·8번을 그대로 사용, 3번(앞판 중심)은 안단(5번) 안쪽선 + 시접 2cm 까지만 사용.',
              '1번은 목안단(6번) 아래부터. 안감 밑단은 겉감보다 2~3cm 짧게 자르세요.']:
        T(1.8, y, s, 8.5); y += 0.45
    y += 0.3
    T(1.5, y, '■ 인쇄 · 붙이기', 12); y += 0.6
    for s in ['① 인쇄 배율 "실제 크기 / 100%" (페이지에 맞춤 해제), A4 세로.',
              '② 각 페이지의 회색 테두리 선을 기준으로 위·왼쪽 여백을 잘라내고, 페이지 번호(A1, A2 …)대로',
              '    테두리 선끼리 맞춰 테이프로 붙입니다. 가장자리의 ◆ 표시와 → 이웃 페이지 번호를 참고하세요.',
              f'③ 전체 배치도는 2쪽. 빈 페이지는 인쇄에서 뺐습니다 (총 {n_used}장).',
              '④ 실선 = 재단선, 점선 = 완성선(박음선). 노치(짧은 눈금)끼리 맞춰 봉제. WL=허리선, HL=엉덩이선.',
              '⑤ "골" 표시는 원단을 접은 선에 대고 재단. 결 방향 화살표는 원단 식서 방향과 평행.',
              '⑥ 이미지 기반 제도라 실제 몸에 맞추려면 광목 가봉(토일)을 먼저 해보길 권장합니다.']:
        T(1.8, y, s, 8.5); y += 0.45
    y += 0.3
    T(1.5, y, '■ 원단 요척 (추정)', 12); y += 0.6
    area = sum(Polygon(pc.cut_line).area * int(pc.cut.split('장')[0].split()[-1]) for pc in pieces)
    need = area/150/0.72 + 30
    T(1.8, y, f'150cm 폭 겉감 약 {need/100:.1f} m  (원단 벨트 포함 시 +0.2m)  ·  안감 110cm 폭 약 {need*1.3/100:.1f} m', 9); y += 0.45
    T(1.8, y, '접착심지: 칼라·스탠드·안단·커프스·플랩·라펠 부분 (약 1m)', 9)
    cv.showPage()

    # ---------- 2쪽: 배치도
    BW, BH = cols*TW, rows*TH
    s = min(18.5/BW, 25.0/BH)
    ox, oy = (PW - BW*s)/2, 3.0
    T(1.5, 2.0, '전체 배치도 (페이지 번호대로 이어 붙이기)', 14)
    def mp(x, y): return ((ox + x*s)*cm, (PH - (oy + y*s))*cm)
    cv.setLineWidth(0.3)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = mp(c*TW, r*TH); x1, y1 = mp((c+1)*TW, (r+1)*TH)
            if (r, c) in used:
                cv.setFillColorRGB(1, 1, 1); cv.setStrokeColorRGB(0.6, 0.6, 0.6)
            else:
                cv.setFillColorRGB(0.92, 0.92, 0.92); cv.setStrokeColorRGB(0.8, 0.8, 0.8)
            cv.rect(x0, y1, x1-x0, y0-y1, fill=1, stroke=1)
    for pc in pieces:
        pts = pc.cut_line + pc.offset
        pth = cv.beginPath(); pth.moveTo(*mp(*pts[0]))
        for q in pts[1:]: pth.lineTo(*mp(*q))
        pth.close()
        cv.setFillColorRGB(0.85, 0.9, 1.0); cv.setStrokeColorRGB(0.1, 0.2, 0.5); cv.setLineWidth(0.6)
        cv.drawPath(pth, fill=1, stroke=1)
        c = Polygon(pts).representative_point()
        cv.setFillColorRGB(0.1, 0.2, 0.5); cv.setFont('K', 9)
        cv.drawCentredString(*mp(c.x, c.y), pc.code)
    cv.setFillColorRGB(0.4, 0.4, 0.4); cv.setFont('K', 7)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in used:
                x, y = mp(c*TW+1.2, r*TH+3.5)
                cv.drawString(x, y, f'{rowname(r)}{c+1}')
    yy = oy + BH*s + 1.0
    cv.setFont('K', 8); cv.setFillColorRGB(0, 0, 0)
    leg = '  '.join(f'{pc.code}.{pc.name}' for pc in sorted(pieces, key=lambda q: int(q.code)))
    import textwrap
    for line in textwrap.wrap(leg, 70):
        cv.drawString(1.5*cm, (PH-yy)*cm, line); yy += 0.45
    cv.showPage()

    # ---------- 타일 페이지
    page_no = 3
    for (r, c) in used:
        X0, Y0 = c*TW, r*TH
        def mpt(x, y): return ((MX + x - X0)*cm, (PH - MY - (y - Y0))*cm)
        cv.saveState()
        clip = cv.beginPath(); clip.rect(MX*cm, MY*cm, TW*cm, TH*cm)
        cv.clipPath(clip, stroke=0, fill=0)
        # 큰 페이지 번호(배경)
        cv.setFillColorRGB(0.9, 0.9, 0.9); cv.setFont('Helvetica-Bold', 90)
        cv.drawCentredString(PW/2*cm, PH/2*cm - 30, f'{rowname(r)}{c+1}')
        for pc in pieces:
            o = pc.offset
            bx = Polygon(pc.cut_line + o).bounds
            if bx[2] < X0 or bx[0] > X0+TW or bx[3] < Y0 or bx[1] > Y0+TH:
                continue
            def path(pts, close=False):
                pth = cv.beginPath(); pth.moveTo(*mpt(*(pts[0]+o)))
                for q in pts[1:]: pth.lineTo(*mpt(*(q+o)))
                if close: pth.close()
                return pth
            # 재단선
            cv.setStrokeColorRGB(0, 0, 0); cv.setLineWidth(1.1); cv.setDash()
            cv.drawPath(path(pc.cut_line, True), stroke=1, fill=0)
            # 완성선
            cv.setLineWidth(0.5); cv.setDash(4, 3); cv.setStrokeColorRGB(0.25, 0.25, 0.25)
            for pts, sa, kind in pc.edges:
                if kind == 'fold': continue
                cv.drawPath(path(pts), stroke=1, fill=0)
            # 골선
            for pts, sa, kind in pc.edges:
                if kind == 'fold':
                    cv.setDash(10, 3, ); cv.setLineWidth(1.6); cv.setStrokeColorRGB(0.1, 0.3, 0.7)
                    cv.drawPath(path(pts), stroke=1, fill=0)
                    mid = (pts[0]+pts[-1])/2 + o
                    cv.setFillColorRGB(0.1, 0.3, 0.7); cv.setFont('K', 10)
                    x, y = mpt(mid[0]+0.4, mid[1])
                    cv.drawString(x, y, '◀ 골 (원단 접힌 선에 대기)')
            cv.setDash(); cv.setStrokeColorRGB(0, 0, 0)
            # 노치
            cv.setLineWidth(1.0)
            cl = LineString(np.vstack([pc.cut_line, pc.cut_line[:1]]))
            for p in pc.notch_pts:
                q = np.array(nearest_points(cl, Point(p))[0].coords[0])
                d = p - q; dl = np.linalg.norm(d)
                if dl < 1e-6: continue
                e = q + d/dl*min(0.8, dl+0.2)
                cv.drawPath(path(np.array([q, e])), stroke=1, fill=0)
            # WL / HL
            cv.setLineWidth(0.4); cv.setDash(1, 2); cv.setStrokeColorRGB(0.5, 0.1, 0.1)
            for yv, lab in ((pc.wl, 'WL 허리선'), (pc.hl, 'HL 엉덩이선')):
                if yv is None: continue
                x = pc.poly.intersection(LineString([(-500, yv), (500, yv)]))
                for g in getattr(x, 'geoms', [x]):
                    if g.is_empty or g.geom_type != 'LineString': continue
                    a = np.array(g.coords)
                    cv.drawPath(path(a), stroke=1, fill=0)
                    cv.setFillColorRGB(0.5, 0.1, 0.1); cv.setFont('K', 8)
                    xx, yy = mpt(*(a[:, 0].min()+o[0]+0.5, yv+o[1]-0.2))
                    cv.drawString(xx, yy, lab)
            # CF / 라펠 꺾임선
            if pc.extra.get('cf'):
                x = pc.poly.intersection(LineString([(0, -50), (0, 200)]))
                for g in getattr(x, 'geoms', [x]):
                    if g.geom_type == 'LineString':
                        a = np.array(g.coords)
                        cv.setDash([6, 2, 1, 2]); cv.setStrokeColorRGB(0.1, 0.4, 0.1)
                        cv.drawPath(path(a), stroke=1, fill=0)
                        cv.setFillColorRGB(0.1, 0.4, 0.1); cv.setFont('K', 8)
                        xx, yy = mpt(0.3+o[0], 40+o[1]); cv.drawString(xx, yy, 'CF 앞중심')
            if 'roll' in pc.extra:
                Bp, Rp = pc.extra['roll']
                ls = pc.poly.intersection(LineString([Bp, Bp + (Rp-Bp)*1.0]))
                if not ls.is_empty and ls.geom_type == 'LineString':
                    a = np.array(ls.coords)
                    cv.setDash([6, 2, 1, 2]); cv.setStrokeColorRGB(0.6, 0.3, 0)
                    cv.drawPath(path(a), stroke=1, fill=0)
                    m = (Bp+Rp)/2 + o
                    cv.setFillColorRGB(0.6, 0.3, 0); cv.setFont('K', 8)
                    xx, yy = mpt(m[0]+0.4, m[1]); cv.drawString(xx, yy, '라펠 꺾임선')
            if 'rollline' in pc.extra:
                cv.setDash([6, 2, 1, 2]); cv.setStrokeColorRGB(0.6, 0.3, 0)
                cv.drawPath(path(pc.extra['rollline']), stroke=1, fill=0)
            if 'pocket' in pc.extra:
                a, b = pc.extra['pocket']
                cv.setDash(); cv.setLineWidth(0.8); cv.setStrokeColorRGB(0.3, 0.3, 0.3)
                cv.drawPath(path(np.array([a, b])), stroke=1, fill=0)
                for e in (a, b):
                    cv.drawPath(path(np.array([e+(0, -0.6), e+(0, 0.6)])), stroke=1, fill=0)
                cv.setFillColorRGB(0.3, 0.3, 0.3); cv.setFont('K', 8)
                xx, yy = mpt(*(a+o+(0.5, -0.4))); cv.drawString(xx, yy, '포켓 플랩 위치 (플랩 윗선)')
            cv.setDash()
            for txt, p in pc.extra.get('marks', []):
                cv.setFillColorRGB(0.2, 0.2, 0.2); cv.setFont('K', 8)
                x, y = mpt(*(p+o)); cv.circle(x, y, 2, fill=1)
                cv.drawString(x+4, y-3, txt)
            # 결 방향 + 라벨
            minx, miny, maxx, maxy = pc.poly.bounds
            best_g = None
            for xx in np.linspace(minx+0.5, maxx-0.5, 40):
                g = pc.poly.intersection(LineString([(xx, miny-1), (xx, maxy+1)]))
                if g.geom_type == 'LineString' and (best_g is None or g.length > best_g[1] * 1.02 or
                                                     (g.length > best_g[1]*0.9 and abs(xx-(minx+maxx)/2) < abs(best_g[0]-(minx+maxx)/2))):
                    best_g = (xx, g.length, g.bounds)
            gx, glen, gb = best_g
            gl = min(glen*0.7, 45)
            gm = (gb[1]+gb[3])/2
            ga, gz = P(gx, gm-gl/2), P(gx, gm+gl/2)
            cv.setStrokeColorRGB(0, 0, 0); cv.setLineWidth(0.9)
            cv.drawPath(path(np.array([ga, gz])), stroke=1, fill=0)
            for tip, sgn in ((ga, 1), (gz, -1)):
                cv.drawPath(path(np.array([tip + (-0.35, sgn*0.9), tip, tip + (0.35, sgn*0.9)])), stroke=1, fill=0)
            xx, yy = mpt(*(P(gx-0.35, gm+4)+o))
            cv.saveState(); cv.translate(xx, yy); cv.rotate(90)
            cv.setFillColorRGB(0, 0, 0); cv.setFont('K', 8); cv.drawString(0, 0, '결 방향 (식서)')
            cv.restoreState()
            # 이름표 (조각 안쪽, 결선 옆 가장 넓은 구간에 맞춤)
            lines = [f'{pc.code}. {pc.name}', pc.cut, f'사이즈 {M["size_name"]} · 시접 포함']
            rel = [1.0, 0.65, 0.55]
            cands = []
            for yy_ in np.linspace(miny+1.5, maxy-2.5, 60):
                g = pc.poly.buffer(-0.6).intersection(LineString([(minx-1, yy_), (maxx+1, yy_)]))
                for sg in getattr(g, 'geoms', [g]):
                    if sg.is_empty or sg.geom_type != 'LineString': continue
                    a0, a1 = sg.bounds[0], sg.bounds[2]
                    # 결 방향 선과 겹치지 않는 쪽
                    for b0, b1 in ((a0, min(a1, gx-0.4)) if a0 < gx-0.4 else (0, 0), (max(a0, gx+0.4), a1) if a1 > gx+0.4 else (0, 0)):
                        if b1-b0 > 2: cands.append((b1-b0, yy_, b0))
            # 세 줄이 들어가도록 아래쪽 2.2cm 도 같은 폭인지 확인
            best_l = None
            for w_, yy_, b0 in sorted(cands, reverse=True)[:80]:
                g2 = pc.poly.buffer(-0.6).intersection(LineString([(b0, yy_+1.6), (b0+w_, yy_+1.6)]))
                if g2.is_empty or abs(g2.length - w_) > 0.3: continue
                fs = min(14, 14*(w_*cm)/max(pdfmetrics.stringWidth(lines[0], 'K', 14), pdfmetrics.stringWidth(lines[2], 'K', 14*0.55)/0.55*0.55*1.0+1))
                fs = min(fs, 14*(w_*cm)/pdfmetrics.stringWidth(lines[2], 'K', 14*0.55))
                if best_l is None or fs > best_l[0] + 0.3:
                    best_l = (fs, yy_, b0)
            if best_l is None:
                c0 = pc.poly.representative_point(); best_l = (8, c0.y, c0.x - 3)
            fs, ly, lx = best_l
            yoff = 0
            for s_, r_ in zip(lines, rel):
                x, y = mpt(*(P(lx, ly)+o))
                cv.setFillColorRGB(0, 0, 0); cv.setFont('K', fs*r_); cv.drawString(x, y - yoff, s_)
                yoff += fs*r_*1.35
        cv.restoreState()
        # 테두리, 맞춤 표시
        cv.setStrokeColorRGB(0.55, 0.55, 0.55); cv.setLineWidth(0.4); cv.setDash()
        cv.rect(MX*cm, MY*cm, TW*cm, TH*cm)
        cv.setFillColorRGB(0.55, 0.55, 0.55)
        def diamond(x, y, s=4):
            p_ = cv.beginPath(); p_.moveTo(x, y+s); p_.lineTo(x+s, y); p_.lineTo(x, y-s); p_.lineTo(x-s, y); p_.close()
            cv.drawPath(p_, fill=1, stroke=0)
        for fx in (0.25, 0.75):
            diamond((MX + TW*fx)*cm, MY*cm); diamond((MX + TW*fx)*cm, (MY+TH)*cm)
        for fy in (0.25, 0.75):
            diamond(MX*cm, (MY + TH*fy)*cm); diamond((MX+TW)*cm, (MY + TH*fy)*cm)
        cv.setFont('K', 7); cv.setFillColorRGB(0.3, 0.3, 0.3)
        nb = lambda rr, cc: f'{rowname(rr)}{cc+1}' if 0 <= rr < rows and 0 <= cc < cols else ''
        if nb(r-1, c): cv.drawCentredString(PW/2*cm, (PH-MY)*cm + 3, f'▲ {nb(r-1, c)}')
        if nb(r+1, c): cv.drawCentredString(PW/2*cm, MY*cm - 9, f'▼ {nb(r+1, c)}')
        if nb(r, c-1):
            cv.saveState(); cv.translate(MX*cm - 4, PH/2*cm); cv.rotate(90); cv.drawCentredString(0, 0, f'◀ {nb(r, c-1)}'); cv.restoreState()
        if nb(r, c+1):
            cv.saveState(); cv.translate((MX+TW)*cm + 9, PH/2*cm); cv.rotate(90); cv.drawCentredString(0, 0, f'{nb(r, c+1)} ▶'); cv.restoreState()
        cv.setFont('K', 7)
        cv.drawString(MX*cm, 0.35*cm, f'밀리터리 롱코트 · {M["size_name"]} · 실물 1:1 (100% 인쇄)')
        cv.drawRightString((PW-MX)*cm, 0.35*cm, f'{rowname(r)}{c+1}  ·  p.{page_no}')
        # 1cm 눈금 (배율 확인용)
        cv.setStrokeColorRGB(0.3, 0.3, 0.3)
        x0 = (PW-MX-5)*cm; y0 = (PH-0.55)*cm
        cv.line(x0, y0, x0+5*cm, y0)
        for i in range(6): cv.line(x0+i*cm, y0, x0+i*cm, y0+(4 if i % 5 else 6))
        cv.drawString(x0-38, y0-2, '5 cm')
        cv.showPage(); page_no += 1
    cv.save()
    return n_used

if __name__ == '__main__':
    pieces, info = draft(MEAS)
    for nm, a, b in info['checks']:
        print(f'{nm:34s} {a:7.2f} {b:7.2f}  diff {a-b:+.2f}')
    print('finished', {k: round(v, 1) for k, v in info['finished'].items()})
    out = os.path.join(OUT_DIR, 'military_coat_pattern_A4.pdf')
    render(pieces, info, MEAS, out)
    print('saved', out)
