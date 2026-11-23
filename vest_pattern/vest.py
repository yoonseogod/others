# Steampunk gear-button vest — A4 tiled sewing pattern (1:1), women's size M
# Body: bust 88 / waist 70 / hip 94.  Units in geometry: cm, y up, waist line y=0.
import math, os, glob
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

def plen(pts):
    return sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

def largest(g):
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda q: q.area)
    return g

def interp(y, table):  # table: [(y, v)] descending y
    if y >= table[0][0]: return table[0][1]
    if y <= table[-1][0]: return table[-1][1]
    for (y0, v0), (y1, v1) in zip(table, table[1:]):
        if y1 <= y <= y0:
            return v0 + (v1-v0)*(y0-y)/(y0-y1)

# ================= FRONT (whole, seam lines) =================
E_t = (-2.0, 6.0)      # V bottom on front edge (overlap 2 cm beyond CF)
N   = (7.5, 44.0)      # neck point (≥ back neck point: front needs bust length)
Sh  = (12.0, 43.3)     # shoulder point (strap 4.5 cm)
U   = (23.5, 22.0)     # underarm
S_w = (21.0, 0.0)      # side waist
S_b = (24.0, -17.0)    # side hem
T   = (4.0, -22.5)     # hem point
E_b = (-2.0, -17.5)    # front edge bottom corner

ARM_F  = [(12.8, 34.0), (16.0, 23.5), U]
SIDE_F_UP = [(23.4, 14.0), (21.5, 6.0), S_w]
SIDE_F_LO = [(21.5, -6.0), (23.3, -12.0), S_b]
NECK_F = [(0.5, 18.5), (5.0, 33.5), N]

front_outline = path(E_b, [[E_t], NECK_F, [Sh], ARM_F, SIDE_F_UP, SIDE_F_LO, [T], [E_b]])
front_poly = Polygon(front_outline)
front_arm  = path(Sh, [ARM_F])
front_neck = path(E_t, [NECK_F])

# ----- princess line (armhole -> hem) with waist suppression -----
# point A on armhole where the lace line starts (y ≈ 24.5)
A = min(front_arm, key=lambda p: abs(p[1]-24.5))
A_ext = (A[0]+3.0, A[1]+1.8)                      # extend outward past the armhole
C_w = (10.75, 0.0)
centre = [A_ext] + path(A, [[(14.0, 19.0), (11.0, 9.0), C_w], [(10.9, -8.0), (12.0, -16.0), (12.8, -27.0)]])
W_TAB = [(A[1], 0.0), (20.0, 0.25), (10.0, 0.9), (0.0, 1.25), (-8.0, 0.6), (-17.0, 0.15), (-27.0, 0.0)]
def _normal(i):
    a = centre[max(i-1, 0)]; b = centre[min(i+1, len(centre)-1)]
    tx, ty = b[0]-a[0], b[1]-a[1]; n = math.hypot(tx, ty)
    nx, ny = ty/n, -tx/n            # right-hand normal of a downward-running line points to -x; fix sign below
    return (nx, ny) if nx > 0 else (-nx, -ny)
left_b, right_b = [], []
for i, (x, y) in enumerate(centre):
    w = interp(y, W_TAB); nx, ny = _normal(i)
    left_b.append((x - nx*w, y - ny*w)); right_b.append((x + nx*w, y + ny*w))
left_b[0] = right_b[0] = A_ext

cf_region = Polygon(left_b + [(-40, -40), (-40, 60), (A_ext[0], 60)])
sf_region = Polygon(right_b + [(60, -40), (60, 60), (A_ext[0], 60)])
cf_poly = largest(front_poly.intersection(cf_region))
sf_flat = largest(front_poly.intersection(sf_region))

# ----- side front: slash at waist & spread 1.5 cm at the princess edge for ruching -----
GATHER = 2.5
seam_w0 = (C_w[0] + interp(0, W_TAB), 0.0)            # princess edge at waist on side panel
r = math.dist(seam_w0, S_w)
theta = math.degrees(2*math.asin(GATHER/2/r))
upper = largest(sf_flat.intersection(box(-50, 0, 60, 60)))
lower = largest(sf_flat.intersection(box(-50, -60, 60, 0)))
upper_r = rotate(upper, -theta, origin=S_w)
def rot_pt(p):  # same rotation for points
    a = math.radians(-theta); dx, dy = p[0]-S_w[0], p[1]-S_w[1]
    return (S_w[0] + dx*math.cos(a) - dy*math.sin(a), S_w[1] + dx*math.sin(a) + dy*math.cos(a))
wedge = Polygon([S_w, seam_w0, rot_pt(seam_w0)])
sf_poly = largest(unary_union([upper_r, lower, wedge.buffer(0.001)]).buffer(0))
def sf_pt(p):  # map an original (unspread) side-front point
    return rot_pt(p) if p[1] > 0 else p

# princess seam lengths (for the record)
def seam_len(bound, y0, y1):
    pts = [p for p in bound if y1 <= p[1] <= y0]
    return plen(pts)
GZ_TOP, GZ_BOT = 10.0, -7.0      # gather zone between notches
def seam_line(bound, poly, tf=lambda p: p):
    g = LineString([tf(p) for p in bound]).intersection(poly.buffer(0.01))
    g = max(getattr(g, "geoms", [g]), key=lambda q: q.length)
    if g.coords[0][1] < g.coords[-1][1]: g = LineString(list(g.coords)[::-1])   # top -> bottom
    return g
CF_SEAM = seam_line(left_b, cf_poly)
SF_SEAM = seam_line(right_b, sf_poly, sf_pt)
_sf_top = SF_SEAM.project(Point(sf_pt(min(right_b, key=lambda q: abs(q[1]-GZ_TOP)))))
_sf_bot = SF_SEAM.project(Point(min(right_b, key=lambda q: abs(q[1]-GZ_BOT))))
NOTCH_SF = [SF_SEAM.interpolate(_sf_top).coords[0], SF_SEAM.interpolate(_sf_bot).coords[0]]
NOTCH_CF = [CF_SEAM.interpolate(_sf_top).coords[0],
            CF_SEAM.interpolate(CF_SEAM.length - (SF_SEAM.length - _sf_bot)).coords[0]]
NET_GATHER = (_sf_bot - _sf_top) - (CF_SEAM.length - (SF_SEAM.length - _sf_bot) - _sf_top)

# ----- buttons, pocket -----
BUTTONS = [(0.0, y) for y in (4.5, -1.9, -8.3, -14.7)]
POCKET = ((9.0, -11.5), (17.5, -8.0))      # inner (lower) -> outer (higher), crosses princess line
POCKET_LEN = math.dist(*POCKET)

# ================= BACK (centre back on fold) =================
B_cb = (0.0, 41.0); B_n = (7.5, 43.5); B_sh = (12.0, 43.0)
U_b = (22.0, 22.0); B_w = (19.5, 0.0); B_sb = (22.5, -17.0); B_cbb = (0.0, -17.0)
ARM_B = [(11.5, 33.0), (14.0, 23.5), U_b]
SIDE_B_UP = [(21.9, 14.0), (20.0, 6.0), B_w]          # = front side shifted 1.5 cm -> equal length
SIDE_B_LO = [(20.0, -6.0), (21.8, -12.0), B_sb]
back_outline = path(B_cbb, [[B_cb], [(3.5, 41.0), (6.5, 42.0), B_n], [B_sh], ARM_B, SIDE_B_UP, SIDE_B_LO, [B_cbb]])
back_poly = Polygon(back_outline)
back_arm = path(B_sh, [ARM_B])
BACK_DART = (9.5, 1.0, 16.0, -9.0)

SA = 1.0
def cutline(poly, fold=False):
    c = poly.buffer(SA, join_style=2, mitre_limit=3)
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
    def place(self, x_mm, y_mm):   # put bottom-left of bounds at (x_mm, y_mm)
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

def add_notch(pc, p, direction):
    n = math.hypot(*direction); d = (direction[0]/n*SA, direction[1]/n*SA)
    pc.poly([p, (p[0]+d[0], p[1]+d[1])], 0.9, close=False)

def outward_normal(poly, p):
    # approximate outward normal at boundary point p
    ring = LineString(poly.exterior.coords); s = ring.project(Point(p))
    a = ring.interpolate(max(s-0.3, 0)); b = ring.interpolate(min(s+0.3, ring.length))
    tx, ty = b.x-a.x, b.y-a.y; n1 = (ty, -tx); n2 = (-ty, tx)
    test = (p[0]+n1[0]*0.05, p[1]+n1[1]*0.05)
    return n1 if not poly.contains(Point(test)) else n2

def on_bound(bound, y):
    return min(bound, key=lambda p: abs(p[1]-y))


def welt_box():
    (x0, y0), (x1, y1) = POCKET
    L = math.dist(POCKET[0], POCKET[1]); ux, uy = (x1-x0)/L, (y1-y0)/L; nx, ny = -uy, ux
    h = 2.5
    a, b = (x0, y0), (x1, y1); c_, d_ = (x1+nx*h, y1+ny*h), (x0+nx*h, y0+ny*h)
    return [a, b, c_, d_]
def draw_welt_box(pc, poly):
    a, b, c_, d_ = welt_box()
    for seg_, w in (([a, b], 1.3), ([b, c_], 0.9), ([c_, d_], 0.9), ([d_, a], 0.9)):
        g = LineString(seg_).intersection(poly)
        for q in getattr(g, "geoms", [g]):
            if not q.is_empty and q.length > 0.05: pc.poly(list(q.coords), w, close=False)

# ---------- ① centre front ----------
P1 = Piece("cf")
P1.poly(coords(cutline(cf_poly)), 1.0)
P1.poly(coords(cf_poly), 0.5, [4, 2])
P1.poly([(0, 7), (0, -19)], 0.5, [6, 2, 1, 2], close=False)
P1.text((-0.9, -4), "앞 중심 CF", 6.5, rot=90)
P1.poly([(-2, 0), (on_bound(left_b, 0)[0], 0)], 0.4, [1, 2], close=False)
P1.text((4.5, 0.4), "허리선 WAIST", 6.5)
for bx, by in BUTTONS:
    P1.circle((bx, by), 0.15); P1.poly([(bx-0.2, by), (bx+1.8, by)], 1.3, close=False)
P1.text((2.6, -5), "단춧구멍 2cm · 단추 1.8cm", 6.5, rot=90)
for p in NOTCH_CF:
    add_notch(P1, p, outward_normal(cf_poly, p))
P1.text((on_bound(left_b, 2)[0]-1.0, 2.0), "개더 구간", 6.5, rot=90)
draw_welt_box(P1, cf_poly)
P1.text((7.0, -13.0), "웰트 포켓 (굵은 선 = 박음 위치)", 6.5)
add_grain(P1, 4.5, -12, 20)
P1.text((4.0, 31), "① 앞 중심판", 13, True)
P1.text((4.0, 29.2), "CENTRE FRONT", 8, True)
P1.text((4.0, 27.6), "겉감 2장 (좌우 대칭)", 8)
P1.text((4.0, 26.2), "심지 붙임 · 시접 1cm 포함", 7)

# ---------- ② side front (spread for ruching) ----------
P2 = Piece("sf")
P2.poly(coords(cutline(sf_poly)), 1.0)
P2.poly(coords(sf_poly), 0.5, [4, 2])
for p in NOTCH_SF:
    add_notch(P2, p, outward_normal(sf_poly, p))
add_notch(P2, sf_pt(min(path(U, [SIDE_F_UP]), key=lambda q: abs(q[1]-10))), (1, -0.15))
add_notch(P2, S_w, (1, 0))
P2.poly([S_w, rot_pt(seam_w0)], 0.3, [1, 2], close=False)
P2.poly([S_w, seam_w0], 0.3, [1, 2], close=False)
P2.text((17.0, 1.4), "허리선 WAIST", 6.5)
draw_welt_box(P2, sf_poly)
P2.text((17.0, -10.8), "웰트 포켓 위치", 6.5)
gp = sf_pt(on_bound(right_b, 2))
P2.text((gp[0]+0.9, gp[1]), "노치 사이 개더", 6.5, rot=80)
add_grain(P2, 17.5, -15, -2)
P2.text((18.5, 12), "② 옆 앞판", 13, True)
P2.text((18.5, 10.2), "SIDE FRONT", 8, True)
P2.text((18.5, 8.6), "겉감 2장 (좌우 대칭)", 8)
P2.text((18.5, 7.2), "시접 1cm 포함", 7)

# ---------- ③ back ----------
P3 = Piece("back")
P3.poly(coords(cutline(back_poly, fold=True)), 1.0)
P3.poly(coords(back_poly), 0.5, [4, 2])
cx, hw, top, bot = BACK_DART
P3.poly([(cx, top), (cx-hw, 0), (cx, bot), (cx+hw, 0), (cx, top)], 0.6, close=False)
P3.poly([(0, 43), (0, -18)], 2.2, close=False)
P3.text((0.6, 10), "골선 FOLD — 원단을 접어서 재단", 8, True, rot=90)
P3.poly([(0, 0), B_w], 0.4, [1, 2], close=False)
add_notch(P3, min(path(U_b, [SIDE_B_UP]), key=lambda q: abs(q[1]-10)), (1, -0.15))
add_notch(P3, B_w, (1, 0))
add_grain(P3, 15.5, -14, 12)
P3.text((10.5, 30), "③ 뒤판 BACK", 13, True)
P3.text((10.5, 28.2), "겉감 1장 · 안감 1장 (골선)", 8)
P3.text((10.5, 26.8), "시접 1cm 포함", 7)

# ---------- ④ front lining (whole front, princess suppression as long dart) ----------
P4 = Piece("lining")
P4.poly(coords(cutline(front_poly)), 1.0)
P4.poly(coords(front_poly), 0.5, [4, 2])
dl = [p for p in left_b if -19 < p[1] < 21]; dr = [p for p in right_b if -19 < p[1] < 21]
P4.poly(dl, 0.6, close=False); P4.poly(dr, 0.6, close=False)
P4.poly([dl[0], dr[0]], 0.6, close=False); P4.poly([dl[-1], dr[-1]], 0.6, close=False)
P4.poly([(-2, 0), (S_w[0], 0)], 0.4, [1, 2], close=False)
P4.poly([(0, 7), (0, -19)], 0.5, [6, 2, 1, 2], close=False)
add_notch(P4, min(path(U, [SIDE_F_UP]), key=lambda q: abs(q[1]-10)), (1, -0.15))
add_notch(P4, S_w, (1, 0))
add_grain(P4, 16.5, -12, 14)
P4.text((14.5, 33.5), "④ 앞 안감", 13, True)
P4.text((14.5, 31.7), "FRONT LINING", 8, True)
P4.text((14.5, 30.1), "안감 2장 (좌우 대칭)", 8)
P4.text((14.5, 28.7), "긴 다트 박음 · 시접 1cm 포함", 7)

# ---------- ⑤ welt, ⑥ pocket bag ----------
WL = POCKET_LEN + 2*SA
P5 = Piece("welt")
P5.poly([(0, 0), (WL, 0), (WL, 7), (0, 7)], 1.0)
P5.poly([(0, 3.5), (WL, 3.5)], 0.4, [1, 2], close=False)
P5.text((WL/2, 5.0), "⑤ 웰트 WELT", 9, True)
P5.text((WL/2, 2.2), "겉감 2장 + 심지 2장", 7)
P5.text((WL/2, 1.0), "····· 접는 선 / 시접 포함", 6.5)
BW = POCKET_LEN + 2 + 2*SA
BH = 8.0 + 2*SA                       # finished depth 8 cm (hem is ~9.6 cm below the pocket)
P6 = Piece("bag")
P6.poly(path((0, BH), [[(0, 3)], [(0, 0.8), (1.2, 0), (3, 0)], [(BW-3, 0)], [(BW-1.2, 0), (BW, 0.8), (BW, 3)], [(BW, BH)], [(0, BH)]]), 1.0, close=False)
add_grain(P6, 2.5, 1.5, BH-1.5, "식서")
P6.text((BW/2+1, 7.0), "⑥ 주머니감", 9, True)
P6.text((BW/2+1, 5.7), "POCKET BAG", 7, True)
P6.text((BW/2+1, 4.3), "안감 4장 · 시접 포함", 7)

# ================= layout on big sheet =================
COLS, ROWS = 4, 4
PW, PH, MARG = 297, 210, 10            # pattern tiles: A4 landscape
CPW, CPH = 210, 297                    # cover: A4 portrait
TW, TH = PW - 2*MARG, PH - 2*MARG
SHEET_W, SHEET_H = COLS*TW, ROWS*TH
GAP = 25
P1.place(10, 15)
x = 10 + P1.size_mm()[0] + GAP
P2.place(x, SHEET_H - 15 - P2.size_mm()[1])
P6.place(x + 10, 15)
P5.place(x + 10, 15 + P6.size_mm()[1] + 20)
x += P2.size_mm()[0] + GAP
P3.place(x, 15)
x += P3.size_mm()[0] + GAP
P4.place(x, 15)
PIECES = [P1, P2, P3, P4, P5, P6]
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
    c.setFont(FB, 21); c.drawString(18*mm, 274*mm, "스팀펑크 기어 단추 조끼 패턴")
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
    line(f"앞 길이(옆목점~앞 끝) 약 {N[1]-T[1]:.0f}cm · 뒤 길이(뒷목~밑단) 약 {B_cb[1]-B_cbb[1]:.0f}cm · 엉덩이를 덮는 길이 · 포켓 입구 약 {POCKET_LEN:.1f}cm")
    y -= 1.5
    line("■ 재단 목록 (시접 1cm 포함: 바깥 실선 = 재단선, 안쪽 점선 = 박음선)", 10.5, True, 5.8)
    line("① 앞 중심판 — 겉감 2장 + 심지 2장      ② 옆 앞판 — 겉감 2장      ③ 뒤판 — 겉감 1장, 안감 1장 (골선)")
    line("④ 앞 안감 — 안감 2장      ⑤ 웰트 — 겉감 2장 + 심지 2장      ⑥ 주머니감 — 안감 4장 (깊이 8cm)  · ①②④는 좌우 대칭")
    line("부자재: 기어 모양 금속 단추 4개(지름 1.8cm) · 검정 레이스 트림(폭 1~1.5cm) 약 1.2m · 접착심지 · 브로치(선택)")
    line("원단: 겉감 코튼 트윌/스트레치 코튼 약 1m (150cm 폭) · 안감 약 0.9m")
    y -= 1.5
    line("■ 만드는 순서", 10.5, True, 5.8)
    for s in [
        "1. ①앞 중심판과 ⑤웰트에 심지를 붙이고, V넥 목둘레 박음선을 따라 늘림 방지 테이프를 붙입니다 (바이어스라 늘어나기 쉬움).",
        "2. ②옆 앞판의 프린세스 선 노치 사이(개더 구간) 시접 안쪽에 큰 땀 2줄을 박고 실을 당겨 ①과 길이를 맞춥니다 (러칭).",
        "3. ①과 ②를 겉끼리 맞대고 레이스를 사이에 끼워 프린세스 선을 박습니다 → 레이스가 솔기에서 살짝 보이게.",
        "4. 웰트 포켓을 만듭니다 — 표시처럼 프린세스 선을 비스듬히 가로지릅니다 (솔기를 먼저 박은 뒤 작업).",
        "5. ③뒤판 다트, ④앞 안감의 긴 다트, 뒤 안감 다트를 박습니다.",
        "6. 겉감끼리, 안감끼리 어깨선을 박습니다 (옆선은 아직 남겨둠).",
        "7. 겉감·안감을 겉끼리 맞대고 앞단·밑단·목둘레·진동둘레를 박고, 곡선에 가위집 → 어깨 끈 사이로 뒤판을 빼 뒤집기.",
        "8. 옆선: 겉감 앞·뒤 + 안감 앞·뒤를 한 번에 링으로 박고, 안감 옆선에 남긴 창구로 정리해 공그르기.",
        "9. 입었을 때 오른쪽 앞판에 단춧구멍(2cm) 4개, 왼쪽에 기어 단추. 왼쪽 가슴에 브로치를 달면 사진과 같은 모양!",
    ]:
        line(s, 8.4, dy=4.7)
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
        f"개더 분량 약 {NET_GATHER:.1f}cm (② 옆 앞판, 노치 사이)",
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
    print("theta", round(theta, 1))
    print("bust y=20  F", xs_at(front_poly, 20), " B", xs_at(back_poly, 20))
    print("waist y=0  F", xs_at(front_poly, 0), " B", xs_at(back_poly, 0), " supp F 2.5 B 2.0")
    print("hem  y=-16 F", xs_at(front_poly, -16), " B", xs_at(back_poly, -16))
    print("side F", round(plen(path(U, [SIDE_F_UP])), 2), round(plen(path(S_w, [SIDE_F_LO])), 2),
          " B", round(plen(path(U_b, [SIDE_B_UP])), 2), round(plen(path(B_w, [SIDE_B_LO])), 2))
    print("shoulder F", round(math.dist(N, Sh), 2), " B", round(math.dist(B_n, B_sh), 2))
    print("princess CF len", round(plen([p for p in left_b if p[1] < A[1]+0.01]), 2), " SF(flat)", round(plen([p for p in right_b if p[1] < A[1]+0.01]), 2))
    print("seam CF", round(CF_SEAM.length,2), "SF", round(SF_SEAM.length,2), "net gather", round(NET_GATHER,2))
    print("A", A, "pocket", round(POCKET_LEN, 2))
    for p in PIECES: print(p.name, [round(v) for v in p.size_mm()])
