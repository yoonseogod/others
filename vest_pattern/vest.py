# Steampunk gear-button vest — UNLINED, A4 tiled sewing pattern (1:1), women's size M
# Body: bust 88 / waist 70 / hip 94.  Geometry units: cm, y up, waist line y = 0.
# Edges are finished with facings (front edge / V-neck / back neck) and bias facing (armholes, hem).
import math, os
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
from shapely.affinity import rotate
from shapely.ops import unary_union

# ---------- fonts ----------
F = FB = "Helvetica"
for path_, idx in (("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", None),
                   ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0)):
    if os.path.exists(path_):
        pdfmetrics.registerFont(TTFont("KR", path_, subfontIndex=idx or 0))
        b = path_.replace("NanumGothic.ttf", "NanumGothicBold.ttf")
        pdfmetrics.registerFont(TTFont("KRB", b if (idx is None and os.path.exists(b)) else path_, subfontIndex=idx or 0))
        F, FB = "KR", "KRB"
        break

# ---------- geometry helpers ----------
def bez(p0, p1, p2, p3, n=40):
    return [((1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t*t*p2[0] + t**3*p3[0],
             (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t*t*p2[1] + t**3*p3[1])
            for t in (i/n for i in range(1, n+1))]

def path(start, segs):
    pts = [start]
    for s in segs:
        pts += [s[0]] if len(s) == 1 else bez(pts[-1], *s)
    return pts

def plen(pts): return sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

def largest(g):
    return max(g.geoms, key=lambda q: q.area) if isinstance(g, MultiPolygon) else g

def smooth_interp(y, table):  # cosine interpolation -> zero slope at every node (no kinks)
    if y >= table[0][0]: return table[0][1]
    if y <= table[-1][0]: return table[-1][1]
    for (y0, v0), (y1, v1) in zip(table, table[1:]):
        if y1 <= y <= y0:
            t = (y0-y)/(y0-y1); t = (1-math.cos(math.pi*t))/2
            return v0 + (v1-v0)*t

def resample(ring, step=0.25):
    ls = LineString(ring); n = max(int(ls.length/step), 8)
    return [ls.interpolate(i*ls.length/n).coords[0] for i in range(n)]

def fillet(ring, corner, r, n=24):
    """Round the corner of a closed ring (list of pts, no repeat) nearest to `corner` with a curve of reach r."""
    ring = resample(ring)
    i = min(range(len(ring)), key=lambda k: math.dist(ring[k], corner))
    ring = ring[i:] + ring[:i]                   # corner at index 0
    c0 = ring[0]
    def walk(seq):
        acc = 0
        for a, b in zip(seq, seq[1:]):
            d = math.dist(a, b)
            if acc + d >= r:
                t = (r-acc)/d; return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t), seq.index(b)
            acc += d
    fwd, jf = walk(ring)
    back_seq = [ring[0]] + ring[::-1][:-1]
    bwd, jb = walk(back_seq)
    keep = ring[jf:len(ring)-jb+1]
    curve = [((1-t)**2*bwd[0] + 2*(1-t)*t*c0[0] + t*t*fwd[0], (1-t)**2*bwd[1] + 2*(1-t)*t*c0[1] + t*t*fwd[1])
             for t in (k/n for k in range(n+1))]
    return curve + keep

def ang(v1, v2):
    return math.degrees(math.acos(max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/math.hypot(*v1)/math.hypot(*v2)))))
sub = lambda a, b: (a[0]-b[0], a[1]-b[1])
def perp_down(a, b, L):   # control point leaving b perpendicular to the line a->b, pointing downward
    dx, dy = b[0]-a[0], b[1]-a[1]; n = math.hypot(dx, dy); px, py = -dy/n, dx/n
    if py > 0: px, py = -px, -py
    return (b[0]+px*L, b[1]+py*L)

# ================= FRONT =================
E_t = (-2.0, 8.0)      # V bottom on front edge (2 cm overlap beyond CF)
N   = (7.5, 44.0)      # neck point
Sh  = (12.0, 43.3)     # shoulder point (strap 4.5 cm)
U   = (23.5, 22.0)     # underarm
S_w = (21.0, 0.0)      # side waist
S_b = (24.0, -17.0)    # side hem
T   = (4.0, -22.5)     # hem point
E_b = (-2.0, -17.5)    # front edge bottom corner

NECK_F = [(0.3, 18.0), perp_down(Sh, N, 10.0), N]               # leaves N at 90° to shoulder
ARM_F  = [perp_down(N, Sh, 8.0), (15.0, 22.0), U]               # 90° at shoulder, horizontal at underarm
SIDE_F_UP = [(23.4, 14.0), (21.0, 5.0), S_w]                     # vertical tangent at waist (no kink)
SIDE_F_LO = [(21.0, -5.0), (23.3, -12.0), S_b]

front_raw = path(E_b, [[E_t], NECK_F, [Sh], ARM_F, SIDE_F_UP, SIDE_F_LO, [T]])
front_ring = front_raw
for corner, r in ((T, 2.5), (E_b, 1.5), (E_t, 2.5)):            # free corners -> curves
    front_ring = fillet(front_ring, corner, r)
front_poly = Polygon(front_ring).buffer(0)
front_arm = path(Sh, [ARM_F]); front_neck = path(E_t, [NECK_F])

# ----- princess line (armhole -> hem), smooth waist suppression -----
A = min(front_arm, key=lambda p: abs(p[1]-24.5))
_ai = front_arm.index(A); _t = sub(front_arm[_ai+1], front_arm[_ai-1])
A_ext = (A[0]+_t[1]*-1.5/math.hypot(*_t)*-1 if False else A[0]+2.5, A[1]+2.0)
C_w = (10.75, 0.0)
centre = [A_ext] + path(A, [[(13.5, 18.0), (10.75, 9.0), C_w], [(10.75, -8.0), (12.0, -16.0), (12.8, -27.0)]])
W_TAB = [(A[1], 0.0), (20.0, 0.2), (0.0, 1.25), (-17.0, 0.15), (-27.0, 0.0)]
def _normal(i):
    a = centre[max(i-1, 0)]; b = centre[min(i+1, len(centre)-1)]
    tx, ty = b[0]-a[0], b[1]-a[1]; n = math.hypot(tx, ty); nx, ny = ty/n, -tx/n
    return (nx, ny) if nx > 0 else (-nx, -ny)
def _bounds(k):
    L, R = [], []
    for i, (x, y) in enumerate(centre):
        w = smooth_interp(y, W_TAB); nx, ny = _normal(i)
        L.append((x - nx*w*(1-k), y - ny*w*(1-k))); R.append((x + nx*w*(1+k), y + ny*w*(1+k)))
    L[0] = R[0] = A_ext
    cf = largest(front_poly.intersection(Polygon(L + [(-40, -40), (-40, 60), (A_ext[0], 60)])))
    sf = largest(front_poly.intersection(Polygon(R + [(60, -40), (60, 60), (A_ext[0], 60)])))
    return L, R, cf, sf
def seam_line(bound, poly):
    g = LineString(bound).intersection(poly.buffer(0.05))
    g = max(getattr(g, "geoms", [g]), key=lambda q: q.length)
    return g if g.coords[0][1] > g.coords[-1][1] else LineString(list(g.coords)[::-1])
# symmetric suppression; the CF panel is ~1 cm longer (outside of the bend) — that excess goes
# into the ruching zone, so notches are walked from the top and from the hem.
left_b, right_b, cf_poly, sf_poly = _bounds(0.0)
CF_SEAM, SF_SEAM = seam_line(left_b, cf_poly), seam_line(right_b, sf_poly)
SPLIT_K = 0.0
GZ_TOP, GZ_BOT = 10.0, -7.0
_top = SF_SEAM.project(Point(min(right_b, key=lambda q: abs(q[1]-GZ_TOP))))
_bot = SF_SEAM.project(Point(min(right_b, key=lambda q: abs(q[1]-GZ_BOT))))
_bot_cf = CF_SEAM.length - (SF_SEAM.length - _bot)
NOTCH_SF = [SF_SEAM.interpolate(_top).coords[0], SF_SEAM.interpolate(_bot).coords[0]]
NOTCH_CF = [CF_SEAM.interpolate(_top).coords[0], CF_SEAM.interpolate(_bot_cf).coords[0]]
RUCH_LEN = _bot - _top
ELASTIC = round(RUCH_LEN * 0.75)

BUTTONS = [(0.0, y) for y in (4.5, -1.9, -8.3, -14.7)]
POCKET = ((9.0, -11.5), (17.5, -8.0))
POCKET_LEN = math.dist(*POCKET)

# ================= BACK (centre back on fold) =================
B_cb = (0.0, 41.0); B_n = (7.5, 43.5); B_sh = (12.0, 43.0)
U_b = (22.0, 22.0); B_w = (19.5, 0.0); B_sb = (22.5, -17.0); B_cbb = (0.0, -17.0)
NECK_B = [perp_down(B_sh, B_n, 1.8), (3.5, 41.0), B_cb]          # 90° at shoulder, 90° at CB
ARM_B = [perp_down(B_n, B_sh, 9.0), (13.5, 22.0), U_b]
SIDE_B_UP = [(21.9, 14.0), (19.5, 5.0), B_w]                     # = front side shifted 1.5 cm
SIDE_B_LO = [(19.5, -5.0), (21.8, -12.0), B_sb]
back_ring = path(B_cbb, [[B_cb]])[:-1] + path(B_cb, [list(reversed([B_n] + NECK_B[:2]))[1:] + [B_n]]) if False else None
back_neck = path(B_n, [NECK_B])                                   # B_n -> CB
back_outline = [B_cbb] + list(reversed(back_neck)) + path(B_sh, [ARM_B, SIDE_B_UP, SIDE_B_LO, [B_cbb]])
back_poly = Polygon(back_outline).buffer(0)
back_arm = path(B_sh, [ARM_B])
BACK_DART = (9.5, 1.0, 16.0, -9.0)

# ================= FACINGS =================
FACE_W = 6.0
_ring = list(front_poly.exterior.coords)[:-1]
def _idx(ring, p): return min(range(len(ring)), key=lambda k: math.dist(ring[k], p))
iN, iT = _idx(_ring, N), _idx(_ring, (T[0]+1.5, T[1]+0.6))
seq = _ring[iN:] + _ring[:iN]                                   # start at N
k = seq.index(_ring[iT])
edgeA, edgeB = seq[:k+1], [seq[0]] + seq[k:][::-1]
edge_line = edgeA if any(p[0] < -1.5 for p in edgeA) else edgeB  # the run that passes the front edge
face_f = largest(front_poly.intersection(LineString(edge_line).buffer(FACE_W, cap_style=1)))
face_f = largest(face_f.intersection(Polygon(left_b + [(-40, -40), (-40, 60), (A_ext[0], 60)])))
face_b = largest(back_poly.intersection(LineString(back_neck + [B_sh]).buffer(FACE_W, cap_style=1)))

SA = 1.0
def cutline(poly, fold=False):
    c = poly.buffer(SA, join_style=2, mitre_limit=2.5)
    return c.intersection(box(0, -100, 100, 100)) if fold else c

# ================= drawing primitives =================
class Piece:
    def __init__(self, name):
        self.name = name; self.ops = []; self.dx = self.dy = 0
    def poly(self, pts, w=0.8, dash=None, close=True): self.ops.append(("poly", list(pts), w, dash, close))
    def text(self, p, s, size=9, bold=False, rot=0): self.ops.append(("text", p, s, size, bold, rot))
    def circle(self, p, r): self.ops.append(("circle", p, r))
    def bounds(self):
        xs, ys = [], []
        for op in self.ops:
            if op[0] == "poly":
                xs += [p[0] for p in op[1]]; ys += [p[1] for p in op[1]]
        return min(xs), min(ys), max(xs), max(ys)
    def place(self, x_mm, y_mm):
        x0, y0, _, _ = self.bounds(); self.dx = x_mm - x0*10; self.dy = y_mm - y0*10
    def size_mm(self):
        x0, y0, x1, y1 = self.bounds(); return (x1-x0)*10, (y1-y0)*10
    def draw(self, c, ox, oy):
        T_ = lambda p: ((p[0]*10 + self.dx - ox)*mm, (p[1]*10 + self.dy - oy)*mm)
        for op in self.ops:
            if op[0] == "poly":
                _, pts, w, dash, close = op
                pth = c.beginPath(); x, y = T_(pts[0]); pth.moveTo(x, y)
                for p in pts[1:]:
                    x, y = T_(p); pth.lineTo(x, y)
                if close: pth.close()
                c.setLineWidth(w); c.setDash(dash) if dash else c.setDash()
                c.drawPath(pth, stroke=1, fill=0); c.setDash()
            elif op[0] == "text":
                _, p, s, size, bold, rot = op
                c.saveState(); x, y = T_(p); c.translate(x, y); c.rotate(rot)
                c.setFont(FB if bold else F, size); c.drawCentredString(0, 0, s); c.restoreState()
            elif op[0] == "circle":
                x, y = T_(op[1]); c.setLineWidth(0.6); c.circle(x, y, op[2]*10*mm)

def coords(g): return list(g.exterior.coords)

def add_grain(pc, x, y0, y1, label="식서 방향 GRAIN"):
    pc.poly([(x, y0), (x, y1)], 0.6, close=False)
    for y, d in ((y1, -1), (y0, 1)):
        pc.poly([(x-0.2, y+d*0.4), (x, y), (x+0.2, y+d*0.4)], 0.6, close=False)
    pc.text((x+0.35, (y0+y1)/2), label, 7, rot=90)

def outward_normal(poly, p):
    ring = LineString(poly.exterior.coords); s = ring.project(Point(p))
    a = ring.interpolate(max(s-0.3, 0)); b = ring.interpolate(min(s+0.3, ring.length))
    tx, ty = b.x-a.x, b.y-a.y; n1 = (ty, -tx)
    test = (p[0]+n1[0]*0.05, p[1]+n1[1]*0.05)
    return n1 if not poly.contains(Point(test)) else (-ty, tx)

def add_notch(pc, poly, p):
    p = LineString(poly.exterior.coords).interpolate(LineString(poly.exterior.coords).project(Point(p))).coords[0]
    d = outward_normal(poly, p); n = math.hypot(*d)
    pc.poly([p, (p[0]+d[0]/n*SA, p[1]+d[1]/n*SA)], 0.9, close=False)

def welt_box():
    (x0, y0), (x1, y1) = POCKET
    L = math.dist(*POCKET); ux, uy = (x1-x0)/L, (y1-y0)/L; nx, ny = -uy, ux
    return [(x0, y0), (x1, y1), (x1+nx*2.5, y1+ny*2.5), (x0+nx*2.5, y0+ny*2.5)]
def draw_welt_box(pc, poly):
    a, b, c_, d_ = welt_box()
    for seg_, w in (([a, b], 1.3), ([b, c_], 0.9), ([c_, d_], 0.9), ([d_, a], 0.9)):
        g = LineString(seg_).intersection(poly)
        for q in getattr(g, "geoms", [g]):
            if not q.is_empty and q.length > 0.05: pc.poly(list(q.coords), w, close=False)

def side_notch_pts(top, ctrl):    # point 12 cm down the side seam from the underarm, and the waist
    seg_ = path(top, [ctrl]); acc = 0
    for a, b in zip(seg_, seg_[1:]):
        acc += math.dist(a, b)
        if acc >= 12: return b

# ---------- ① centre front ----------
P1 = Piece("cf")
P1.poly(coords(cutline(cf_poly)), 1.0)
P1.poly(coords(cf_poly), 0.5, [4, 2])
P1.poly([(0, 7), (0, -19)], 0.5, [6, 2, 1, 2], close=False)
P1.text((-0.9, -4), "앞 중심 CF", 6.5, rot=90)
P1.poly([(-2, 0), (CF_SEAM.interpolate(CF_SEAM.project(Point(C_w))).x, 0)], 0.4, [1, 2], close=False)
P1.text((4.5, 0.4), "허리선 WAIST", 6.5)
for bx, by in BUTTONS:
    P1.circle((bx, by), 0.15); P1.poly([(bx-0.2, by), (bx+1.8, by)], 1.3, close=False)
P1.text((2.6, -5), "단춧구멍 2cm · 단추 1.8cm", 6.5, rot=90)
for p in NOTCH_CF: add_notch(P1, cf_poly, p)
P1.text((CF_SEAM.interpolate(_top + 6).x - 1.0, CF_SEAM.interpolate(_top + 6).y), "노치 사이 러칭 (고무줄)", 6.5, rot=88)
draw_welt_box(P1, cf_poly)
P1.text((6.5, -13.2), "웰트 포켓 (굵은 선 = 박음 위치)", 6.5)
add_grain(P1, 4.5, -12, 20)
P1.text((4.2, 31), "① 앞 중심판", 13, True)
P1.text((4.2, 29.2), "CENTRE FRONT", 8, True)
P1.text((4.2, 27.6), "겉감 2장 (좌우 대칭)", 8)
P1.text((4.2, 26.2), "시접 1cm 포함", 7)

# ---------- ② side front ----------
P2 = Piece("sf")
P2.poly(coords(cutline(sf_poly)), 1.0)
P2.poly(coords(sf_poly), 0.5, [4, 2])
for p in NOTCH_SF: add_notch(P2, sf_poly, p)
add_notch(P2, sf_poly, side_notch_pts(U, SIDE_F_UP)); add_notch(P2, sf_poly, S_w)
P2.poly([S_w, (SF_SEAM.interpolate(SF_SEAM.project(Point(C_w))).x, 0)], 0.4, [1, 2], close=False)
P2.text((17.0, 0.4), "허리선 WAIST", 6.5)
draw_welt_box(P2, sf_poly)
P2.text((17.0, -11.0), "웰트 포켓 위치", 6.5)
gp = min(right_b, key=lambda q: abs(q[1]-3))
P2.text((gp[0]+0.9, gp[1]), "노치 사이 러칭 (고무줄)", 6.5, rot=88)
add_grain(P2, 18.0, -15, -2)
P2.text((18.5, 13), "② 옆 앞판", 13, True)
P2.text((18.5, 11.2), "SIDE FRONT", 8, True)
P2.text((18.5, 9.6), "겉감 2장 (좌우 대칭)", 8)
P2.text((18.5, 8.2), "시접 1cm 포함", 7)

# ---------- ③ back ----------
P3 = Piece("back")
P3.poly(coords(cutline(back_poly, fold=True)), 1.0)
P3.poly(coords(back_poly), 0.5, [4, 2])
cx, hw, top, bot = BACK_DART
dart_l = path((cx, top), [[(cx-hw*0.9, 8), (cx-hw, 3), (cx-hw, 0)], [(cx-hw, -3), (cx-hw*0.6, -6), (cx, bot)]])
dart_r = [(2*cx-x, y) for x, y in dart_l]
P3.poly(dart_l, 0.6, close=False); P3.poly(dart_r, 0.6, close=False)
P3.poly([(0, 43), (0, -18)], 2.2, close=False)
P3.text((0.6, 10), "골선 FOLD — 원단을 접어서 재단", 8, True, rot=90)
P3.poly([(0, 0), B_w], 0.4, [1, 2], close=False)
add_notch(P3, back_poly, side_notch_pts(U_b, SIDE_B_UP)); add_notch(P3, back_poly, B_w)
add_grain(P3, 15.5, -14, 12)
P3.text((10.5, 30), "③ 뒤판 BACK", 13, True)
P3.text((10.5, 28.2), "겉감 1장 (골선)", 8)
P3.text((10.5, 26.8), "시접 1cm 포함", 7)

# ---------- ④ front facing ----------
P4 = Piece("face_f")
P4.poly(coords(cutline(face_f)), 1.0)
P4.poly(coords(face_f), 0.5, [4, 2])
for bx, by in BUTTONS:
    P4.circle((bx, by), 0.15); P4.poly([(bx-0.2, by), (bx+1.8, by)], 1.3, close=False)
P4.poly([(0, 7), (0, -19)], 0.5, [6, 2, 1, 2], close=False)
add_grain(P4, 2.0, -10, 14, "식서")
P4.text((2.6, 33), "④ 앞 안단", 12, True, rot=-72)
P4.text((4.4, 30.5), "FRONT FACING · 겉감 2장 + 심지", 7, rot=-72)
P4.text((3.6, -20.0), "안쪽 가장자리는 오버록", 6)

# ---------- ⑤ back neck facing ----------
P5 = Piece("face_b")
P5.poly(coords(cutline(face_b, fold=True)), 1.0)
P5.poly(coords(face_b), 0.5, [4, 2])
P5.poly([(0, 44.5), (0, 33)], 2.2, close=False)
P5.text((0.6, 38.0), "골선", 7, True, rot=90)
P5.text((5.5, 39.0), "⑤ 뒷목 안단", 10, True)
P5.text((5.5, 37.8), "BACK NECK FACING", 6.5, True)
P5.text((5.5, 36.8), "겉감 1장 + 심지 (골선)", 6.5)

# ---------- ⑥ welt, ⑦ pocket bag ----------
WL = POCKET_LEN + 2*SA
P6 = Piece("welt")
P6.poly([(0, 0), (WL, 0), (WL, 7), (0, 7)], 1.0)
P6.poly([(0, 3.5), (WL, 3.5)], 0.4, [1, 2], close=False)
P6.text((WL/2, 5.0), "⑥ 웰트 WELT", 9, True)
P6.text((WL/2, 2.2), "겉감 2장 + 심지 2장", 7)
P6.text((WL/2, 1.0), "····· 접는 선 / 시접 포함", 6.5)
BW = POCKET_LEN + 2 + 2*SA
BH = 8.0 + 2*SA
P7 = Piece("bag")
P7.poly(path((0, BH), [[(0, 3)], [(0, 0.8), (1.2, 0), (3, 0)], [(BW-3, 0)], [(BW-1.2, 0), (BW, 0.8), (BW, 3)], [(BW, BH)], [(0, BH)]]), 1.0, close=False)
add_grain(P7, 2.5, 1.5, BH-1.5, "식서")
P7.text((BW/2+1, 7.0), "⑦ 주머니감", 9, True)
P7.text((BW/2+1, 5.7), "POCKET BAG", 7, True)
P7.text((BW/2+1, 4.3), "면/안감 원단 4장 · 시접 포함", 7)

# ================= layout on big sheet =================
COLS, ROWS = 4, 4
PW, PH, MARG = 297, 210, 10            # pattern tiles: A4 landscape
CPW, CPH = 210, 297                    # cover: A4 portrait
TW, TH = PW - 2*MARG, PH - 2*MARG
SHEET_W, SHEET_H = COLS*TW, ROWS*TH
GAP = 22
P1.place(10, 15)
x = 10 + P1.size_mm()[0] + GAP
P2.place(x, SHEET_H - 15 - P2.size_mm()[1])
P7.place(x + 10, 15)
P6.place(x + 10, 15 + P7.size_mm()[1] + 20)
x += P2.size_mm()[0] + GAP
P3.place(x, 15)
x += P3.size_mm()[0] + GAP
P4.place(x, 15)
x += P4.size_mm()[0] + GAP
P5.place(x, 15)
PIECES = [P1, P2, P3, P4, P5, P6, P7]
for p in PIECES:
    x0, y0, x1, y1 = p.bounds()
    assert 0 <= x0*10+p.dx and x1*10+p.dx <= SHEET_W and 0 <= y0*10+p.dy and y1*10+p.dy <= SHEET_H, (p.name, x1*10+p.dx, y1*10+p.dy)

# ================= pages =================
def test_square(c, x, y):
    c.setLineWidth(0.8); c.rect(x*mm, y*mm, 50*mm, 50*mm)
    c.setFont(FB, 9); c.drawCentredString((x+25)*mm, (y+28)*mm, "5 cm × 5 cm")
    c.setFont(F, 7); c.drawCentredString((x+25)*mm, (y+21)*mm, "인쇄 확인용 테스트 사각형")

def tile_page(c, col, row):
    ox = col*TW - MARG; oy = (ROWS-1-row)*TH - MARG
    c.saveState()
    pth = c.beginPath(); pth.rect(MARG*mm, MARG*mm, TW*mm, TH*mm); c.clipPath(pth, stroke=0, fill=0)
    for p in PIECES: p.draw(c, ox, oy)
    c.restoreState()
    c.setLineWidth(0.3); c.setStrokeColorRGB(0.55, 0.55, 0.55); c.setDash([2, 2])
    c.rect(MARG*mm, MARG*mm, TW*mm, TH*mm); c.setDash(); c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0.6, 0.6, 0.6)
    for (x, y) in ((PW/2, MARG), (PW/2, PH-MARG), (MARG, PH/2), (PW-MARG, PH/2)):
        pp = c.beginPath(); s = 5
        pp.moveTo(x*mm, (y+s)*mm); pp.lineTo((x+s)*mm, y*mm); pp.lineTo(x*mm, (y-s)*mm); pp.lineTo((x-s)*mm, y*mm); pp.close()
        c.drawPath(pp, stroke=0, fill=1)
    lab = f"{chr(65+row)}{col+1}"
    c.setFont(FB, 22); c.setFillColorRGB(0.82, 0.82, 0.82); c.drawCentredString(PW/2*mm, PH/2*mm, lab)
    c.setFillColorRGB(0, 0, 0); c.setFont(F, 7)
    c.drawString(MARG*mm, 4*mm, f"페이지 {lab}  — 회색 점선을 따라 여백을 자르고 ◆ 표시를 맞춰 붙이세요")
    c.drawRightString((PW-MARG)*mm, 4*mm, "실제 크기 100% 인쇄")
    if (col, row) == (0, 0):
        test_square(c, MARG+8, PH-MARG-60)

def cover(c):
    c.setFont(FB, 21); c.drawString(18*mm, 274*mm, "스팀펑크 기어 단추 조끼 패턴 (안감 없음)")
    c.setFont(F, 10); c.drawString(18*mm, 267*mm, "프린세스 라인 + 레이스 + 러칭 · A4 타일 (실물 크기 1:1) · 여성 M 사이즈")
    y = 256
    def line(s, size=8.8, bold=False, dy=5.0):
        nonlocal y
        c.setFont(FB if bold else F, size); c.drawString(18*mm, y*mm, s); y -= dy
    line("■ 인쇄", 10.5, True, 5.8)
    line("A4 가로 방향 · '실제 크기 / 100%'로 인쇄 ('페이지에 맞춤' 금지) → A1 페이지의 5cm 사각형을 자로 확인")
    line("회색 점선 바깥 여백을 자르고 ◆ 표시와 선을 맞춰 아래 배치도 순서대로 테이프로 붙이기")
    y -= 1.5
    line("■ 치수 (M)", 10.5, True, 5.8)
    line("몸 치수: 가슴 88 · 허리 70 · 엉덩이 94cm  →  완성 치수: 가슴 약 91 · 허리 약 72 · 밑단 둘레 약 93cm")
    line(f"앞 길이(옆목점~앞 끝) 약 {N[1]-T[1]:.0f}cm · 뒤 길이(뒷목~밑단) 약 {B_cb[1]-B_cbb[1]:.0f}cm · 포켓 입구 약 {POCKET_LEN:.1f}cm")
    y -= 1.5
    line("■ 재단 목록 (시접 1cm 포함: 바깥 실선 = 재단선, 안쪽 점선 = 박음선)", 10.5, True, 5.8)
    line("① 앞 중심판 — 2장    ② 옆 앞판 — 2장    ③ 뒤판 — 1장 (골선)    ④ 앞 안단 — 2장 + 심지")
    line("⑤ 뒷목 안단 — 1장 (골선) + 심지    ⑥ 웰트 — 2장 + 심지    ⑦ 주머니감 — 면 4장    · ①②④는 좌우 대칭")
    line("부자재: 기어 단추 4개(1.8cm) · 레이스 트림 1.2m · 바이어스 테이프(2.5cm) 2.5m · 고무줄(0.6cm) 30cm · 접착심지")
    line("원단: 겉감 코튼 트윌 / 스트레치 코튼 약 1m (150cm 폭) — 안감 필요 없음")
    y -= 1.5
    line("■ 만드는 순서 (안감 없이 안단 + 바이어스 처리)", 10.5, True, 5.8)
    for s in [
        "1. ④⑤ 안단과 ⑥웰트에 심지를 붙입니다. V넥 박음선에 늘림 방지 테이프. 각 조각 가장자리는 오버록(지그재그).",
        "2. ①+② 사이에 레이스를 끼워 프린세스 선 박기 (노치끼리 맞춤) → 두 시접을 함께 오버록, 옆으로 넘기기.",
        f"3. 러칭: 폭 0.6cm 고무줄 {ELASTIC}cm를 노치 사이({RUCH_LEN:.0f}cm) 시접에 늘려가며 지그재그로 박기 → 사진처럼 솔기 양쪽에 주름.",
        "4. 웰트 포켓 만들기 (프린세스 선을 비스듬히 가로지름).   5. ③뒤판 다트 박기.",
        "6. 겉감 어깨선 박기 / ④앞 안단 + ⑤뒷목 안단 어깨선 박기 → 안단 바깥(안쪽) 가장자리 오버록.",
        "7. 안단을 겉감에 겉끼리 맞대고 앞단 → V넥 → 뒷목 → 앞 끝 밑단까지 박기 → 가위집 → 뒤집어 상침 0.5cm.",
        "8. 옆선 박기 (앞+뒤) → 시접 가름.",
        "9. 진동둘레와 나머지 밑단: 바이어스 테이프를 겉에 박고 안으로 완전히 넘겨 상침 (테이프가 겉에 안 보이게).",
        "10. 오른쪽 앞판(입었을 때)에 단춧구멍 2cm × 4, 왼쪽에 기어 단추. 브로치를 달면 완성!",
    ]:
        line(s, 8.2, dy=4.6)
    y -= 3
    c.setFont(FB, 10.5); c.drawString(18*mm, y*mm, f"■ 페이지 배치도 ({COLS} × {ROWS} = {COLS*ROWS}장)"); y -= 3
    s = 0.085; bx = 18; by = y - SHEET_H*s
    for r in range(ROWS):
        for col in range(COLS):
            x0 = bx + col*TW*s; y0 = by + (ROWS-1-r)*TH*s
            c.setLineWidth(0.4); c.rect(x0*mm, y0*mm, TW*s*mm, TH*s*mm)
            c.setFont(FB, 8); c.setFillColorRGB(0.6, 0.6, 0.6)
            c.drawCentredString((x0+TW*s/2)*mm, (y0+TH*s/2)*mm, f"{chr(65+r)}{col+1}"); c.setFillColorRGB(0, 0, 0)
    c.saveState(); c.translate(bx*mm, by*mm); c.scale(s, s)
    for p in PIECES:
        pts = p.ops[0][1]
        pth = c.beginPath(); pth.moveTo((pts[0][0]*10+p.dx)*mm, (pts[0][1]*10+p.dy)*mm)
        for q in pts[1:]: pth.lineTo((q[0]*10+p.dx)*mm, (q[1]*10+p.dy)*mm)
        pth.close(); c.setLineWidth(3); c.drawPath(pth)
    c.restoreState()
    tx = bx + SHEET_W*s + 6
    c.setFont(F, 7.5)
    for i, s_ in enumerate([
        "※ 사진을 보고 비율을 재서 제도한 패턴입니다.",
        "   뒤판은 사진에 보이지 않아 일반 조끼 뒤판으로",
        "   제도했습니다. 본 원단 전 광목 가봉을 권장합니다.",
        "",
        f"앞 진동둘레 {plen(front_arm):.1f}cm · 뒤 진동둘레 {plen(back_arm):.1f}cm",
        f"러칭 구간 {RUCH_LEN:.0f}cm → 고무줄 {ELASTIC}cm × 2",
        "모든 이음 지점은 앞·뒤를 붙였을 때 곡선이",
        "꺾이지 않도록 각도를 맞췄습니다.",
    ]):
        c.drawString(tx*mm, (by + SHEET_H*s - 4 - i*4.2)*mm, s_)
    test_square(c, bx, by - 62)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steampunk_vest_pattern_A4.pdf")
c = canvas.Canvas(out, pagesize=(CPW*mm, CPH*mm)); c.setTitle("Steampunk Vest Pattern A4")
cover(c); c.showPage()
c.setPageSize((PW*mm, PH*mm))
for r in range(ROWS):
    for col in range(COLS):
        tile_page(c, col, r); c.showPage()
c.save()

if __name__ == "__main__":
    print(out)
    def xs_at(poly, y):
        g = LineString([(-50, y), (60, y)]).intersection(poly.exterior)
        return sorted(round(q.x, 2) for q in getattr(g, "geoms", [g]))
    print("seam CF", round(CF_SEAM.length, 2), "SF", round(SF_SEAM.length, 2), "split k", round(SPLIT_K, 3), "ruch", round(RUCH_LEN, 1), "elastic", ELASTIC)
    print("bust y=20  F", xs_at(front_poly, 20), " B", xs_at(back_poly, 20))
    print("waist y=0  F", xs_at(front_poly, 0), " B", xs_at(back_poly, 0))
    print("hem  y=-16 F", xs_at(front_poly, -16), " B", xs_at(back_poly, -16))
    print("side F", round(plen(path(U, [SIDE_F_UP])), 2), round(plen(path(S_w, [SIDE_F_LO])), 2),
          " B", round(plen(path(U_b, [SIDE_B_UP])), 2), round(plen(path(B_w, [SIDE_B_LO])), 2))
    print("shoulder F", round(math.dist(N, Sh), 2), " B", round(math.dist(B_n, B_sh), 2))
    J = {
     "N":  (ang(sub(Sh, N), sub(NECK_F[1], N)), ang(sub(B_sh, B_n), sub(NECK_B[0], B_n))),
     "Sh": (ang(sub(N, Sh), sub(ARM_F[0], Sh)), ang(sub(B_n, B_sh), sub(ARM_B[0], B_sh))),
     "U":  (ang(sub(ARM_F[1], U), sub(SIDE_F_UP[0], U)), ang(sub(ARM_B[1], U_b), sub(SIDE_B_UP[0], U_b))),
     "hem":(ang(sub(SIDE_F_LO[1], S_b), sub(T, S_b)), ang(sub(SIDE_B_LO[1], B_sb), sub(B_cbb, B_sb))),
     "CBneck": (None, ang(sub(B_cbb, B_cb), sub(NECK_B[1], B_cb))),
    }
    for k_, (f_, b_) in J.items(): print("junction", k_, f_ and round(f_, 1), round(b_, 1), "sum", f_ and round(f_+b_, 1))
    for p in PIECES: print(p.name, [round(v) for v in p.size_mm()])
