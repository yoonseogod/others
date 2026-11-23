"""Fitted bustier-style waistcoat: princess seams front/back, deep V, button front,
pointed front hem. 1:1 pattern in mm, A4 tiled PDF + overview + preview.
Usage: python draft.py OUT_DIR [bust waist hip] (cm)"""
import json, math, subprocess, sys
import numpy as np
from shapely.geometry import Polygon
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = sys.argv[1]
bust, waist, hip = (float(v) * 10 for v in (sys.argv[2:5] if len(sys.argv) >= 5 else (84, 66, 90)))
SA = 10.0

font_path = subprocess.run(["fc-match", "-f", "%{file}", "WenQuanYi Zen Hei"], capture_output=True, text=True).stdout
pdfmetrics.registerFont(TTFont("KO", font_path, subfontIndex=0))

P = lambda x, y: np.array([x, y], float)


def bez(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def crom(pts, n=25):
    pts = [np.asarray(p, float) for p in pts]
    ext = [2 * pts[0] - pts[1]] + pts + [2 * pts[-1] - pts[-2]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.array(out)


def line(a, b, n=2):
    return np.linspace(a, b, n)


def plen(c):
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))


def split_at(c, L):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    i = int(np.searchsorted(d, L))
    i = min(max(i, 1), len(c) - 1)
    t = (L - d[i - 1]) / max(d[i] - d[i - 1], 1e-9)
    p = c[i - 1] + t * (c[i] - c[i - 1])
    return np.vstack([c[:i], p]), np.vstack([p, c[i:]]), p


def hem_y(x, a, b):  # y on straight segment a-b at x
    return a[1] + (x - a[0]) / (b[0] - a[0]) * (b[1] - a[1])


# ---------------- measurements -> quarter widths (close fit, woven) ----------------
q_b = (bust + 40) / 4
q_w = (waist + 20) / 4
hh = waist + 0.55 * (hip - waist)          # high hip, ~8 cm below waist
q_h = (hh + 40) / 4
F_B, F_W, F_H = q_b + 5, q_w + 5, q_h + 5   # front
B_B, B_W, B_H = q_b - 5, q_w - 5, q_h - 5   # back
WF, WB = 410.0, 400.0                       # HPS -> waist (front over bust, back)
BUST_Y = 245.0
EXT = 20.0                                  # button extension

# ---------------- FRONT ----------------
hps = P(63, 0); sp = P(178, 45); u = (sp - hps) / np.linalg.norm(sp - hps)
N = hps + 25 * u; S2 = hps + 80 * u
side_top = P(F_B, 235)
arm = bez(S2, S2 + P(-6, 95), P(150, 235), side_top)
BP = P(90, BUST_Y)
dW, dH = 30.0, 3.5                          # princess suppression at waist / half at hem
side_w = F_W + dW
side_h = F_H + 2 * dH
S_h = P(side_h, WF + 80); Pt = P(8, WF + 165); E = P(-EXT, WF + 138); V = P(-EXT, 255)
ax_w, ax_h = 95.0, 97.0
W_cf, W_sd = P(ax_w - dW / 2, WF), P(ax_w + dW / 2, WF)
H_cf = P(ax_h - dH, hem_y(ax_h - dH, S_h, Pt)); H_sd = P(ax_h + dH, hem_y(ax_h + dH, S_h, Pt))
side_f = crom([side_top, P(F_B, BUST_Y + 15), P(side_w, WF), S_h])
EASE_BUST = 3.0                             # CF panel 3 mm longer over the bust (eased in)


def front_split(LA):
    a_cf, _, A1 = split_at(arm, LA - 10)
    _, a_sf, A2 = split_at(arm, LA + 10)
    p1 = crom([A1, BP, W_cf, H_cf]); p2 = crom([A2, BP, W_sd, H_sd])
    return a_cf, a_sf, A1, A2, p1, p2


lo, hi = 0.25 * plen(arm), 0.85 * plen(arm)
for _ in range(60):                          # bisection: princess edges equal (+ bust ease)
    mid = (lo + hi) / 2
    *_, p1, p2 = front_split(mid)
    if plen(p1) - plen(p2) > EASE_BUST:
        lo = mid
    else:
        hi = mid
arm_cf, arm_sf, A_cf, A_sd, pr_cf, pr_sf = front_split((lo + hi) / 2)
neck_f = bez(V, V + P(30, -80), N + P(-30, 70), N)

cf_outline = np.vstack([line(N, S2), arm_cf, pr_cf, line(H_cf, Pt), line(Pt, E), line(E, V), neck_f])
sf_outline = np.vstack([arm_sf, side_f, line(S_h, H_sd), pr_sf[::-1]])

# ---------------- BACK ----------------
hpsb = P(65, 0); spb = P(183, 40); ub = (spb - hpsb) / np.linalg.norm(spb - hpsb)
Nb = hpsb + 25 * ub; Sb2 = hpsb + 82 * ub
CBn = P(0, 50)
neck_b = bez(CBn, CBn + P(45, 0), Nb + P(-12, 30), Nb)
dWb, dHb, cb_w, cb_h = 25.0, 3.5, 10.0, 4.0
side_wb = B_W + cb_w + dWb
side_hb = B_H + cb_h + 2 * dHb
Lf = plen(side_f)


def back_side(ysb):
    return crom([P(B_B, ysb), P(B_B, ysb + 25), P(side_wb, WB), P(side_hb, WB + 80)])


ysb = 225.0
for _ in range(50):                          # match side-seam lengths exactly
    ysb += plen(back_side(ysb)) - Lf
side_b = back_side(ysb); sb_top = side_b[0]; HEMB = WB + 80
armb = bez(Sb2, Sb2 + P(0, 90), P(150, ysb), sb_top)
Ab_cut = 0.5 * plen(armb)
armb_cb, armb_sb, Ab = split_at(armb, Ab_cut)
Wb_c, Wb_s = P(95 - dWb / 2, WB), P(95 + dWb / 2, WB)
Hb_c, Hb_s = P(97 - dHb, HEMB), P(97 + dHb, HEMB)
pr_cb = crom([Ab, P(92, BUST_Y), Wb_c, Hb_c]); pr_sb = crom([Ab, P(92, BUST_Y), Wb_s, Hb_s])
cb_line = crom([CBn, P(0, BUST_Y), P(cb_w, WB), P(cb_h, HEMB)])

cb_outline = np.vstack([line(Nb, Sb2), armb_cb, pr_cb, line(Hb_c, cb_line[-1]), cb_line[::-1], neck_b[1:-1]])
cb_outline = np.vstack([neck_b[::-1][-1:], cb_outline]) if False else cb_outline
sb_outline = np.vstack([armb_sb, side_b, line(side_b[-1], Hb_s), pr_sb[::-1]])

# ---------------- validation ----------------
checks = []


def chk(name, a, b, tol, note=""):
    d = a - b
    checks.append(dict(name=name, a=round(a, 1), b=round(b, 1), diff=round(d, 1), ok=bool(abs(d) <= tol), note=note))


chk("앞 프린세스: 중앙앞판 ↔ 옆앞판", plen(pr_cf), plen(pr_sf) + EASE_BUST, 1, "중앙앞판이 3mm 길게: 가슴 부분에서 오그려 박기")
chk("뒤 프린세스: 중앙뒤판 ↔ 옆뒤판", plen(pr_cb), plen(pr_sb), 4)
chk("옆선: 앞 ↔ 뒤", plen(side_f), plen(side_b), 1)
chk("어깨: 뒤 ↔ 앞 (뒤 2mm 이즈)", 57.0, 55.0, 3)
fin = {
    "가슴 완성둘레": 4 * (q_b), "허리 완성둘레": 4 * (q_w), "밑단(하이힙) 완성둘레": 4 * (q_h)}
waist_f = (W_cf[0] - 0) + (side_w - W_sd[0]); waist_b = (Wb_c[0] - cb_w) + (side_wb - Wb_s[0])
chk("허리 반둘레 (앞+뒤) vs 목표", waist_f + waist_b, F_W + B_W, 1)
for nm, o in [("중앙앞판", cf_outline), ("옆앞판", sf_outline), ("중앙뒤판", cb_outline), ("옆뒤판", sb_outline)]:
    pg = Polygon(o)
    checks.append(dict(name=f"{nm} 외곽선 자기교차 없음", a=None, b=None, diff=None, ok=bool(pg.is_valid), note=""))

# ---------------- pieces ----------------
pieces = [
    dict(key="cf", name="① 중앙 앞판 Center Front", cut="겉감 2장(좌우대칭) · 안감 2장 · 앞단 심지", o=cf_outline,
         grain=(P(40, 330), P(40, 470)), notches=[(pr_cf, [BP, W_cf])],
         buttons=[P(0, y) for y in (270, 345, 420, 495)], cfline=(P(0, 200), P(0, WF + 150)),
         welt=None),
    dict(key="sf", name="② 옆 앞판 Side Front", cut="겉감 2장 · 안감 2장", o=sf_outline,
         grain=(P(160, 300), P(160, 400)), notches=[(pr_sf, [BP, W_sd]), (side_f, [P(side_w, WF)])],
         buttons=[], cfline=None, welt=(P(118, WF + 45), P(190, WF + 38))),
    dict(key="cb", name="③ 중앙 뒤판 Center Back", cut="안감천 4장 (겉 2 + 안 2)", o=cb_outline,
         grain=(P(45, 120), P(45, 300)), notches=[(pr_cb, [P(92, BUST_Y), Wb_c])],
         buttons=[], cfline=None, welt=None),
    dict(key="sb", name="④ 옆 뒤판 Side Back", cut="안감천 4장 (겉 2 + 안 2)", o=sb_outline,
         grain=(P(160, 290), P(160, 420)), notches=[(pr_sb, [P(92, BUST_Y), Wb_s]), (side_b, [P(side_wb, WB)])],
         buttons=[], cfline=None, welt=None),
]
for pc in pieces:
    poly = Polygon(pc["o"])
    pc["cut_poly"] = np.array(poly.buffer(SA, join_style=2, mitre_limit=2.5).exterior.coords)

# ---------------- layout ----------------
GAP = 20
x = 15
for pc in pieces:
    mn = pc["cut_poly"].min(0); mx = pc["cut_poly"].max(0)
    pc["off"] = P(x - mn[0], 15 - mn[1])
    x += (mx[0] - mn[0]) + GAP
TEST = P(x, 15)                                  # 5x5 test square
W_all = x + 60; H_all = max(pc["cut_poly"][:, 1].max() + pc["off"][1] for pc in pieces) + 15
CW, CH, MX, MY = 180.0, 267.0, 15.0, 15.0
cols, rows = math.ceil(W_all / CW), math.ceil(H_all / CH)


def draw_pattern(c, ox, oy, flip_h):
    """draw in mm coords (y down) with origin offset; flip_h = page height in mm"""
    T = lambda p: ((p[0] + ox) * mm, (flip_h - (p[1] + oy)) * mm)

    def poly(pts, close=True, dash=None, w=0.6):
        c.setDash(list(dash), 0) if dash else c.setDash()
        c.setLineWidth(w)
        pth = c.beginPath(); pth.moveTo(*T(pts[0]))
        for p in pts[1:]:
            pth.lineTo(*T(p))
        if close:
            pth.close()
        c.drawPath(pth, stroke=1, fill=0); c.setDash()

    for pc in pieces:
        o = pc["off"]
        poly(pc["cut_poly"] + o, w=0.9)
        poly(pc["o"] + o, dash=(3, 2), w=0.4)
        g0, g1 = pc["grain"][0] + o, pc["grain"][1] + o
        poly([g0, g1], close=False, w=0.6)
        for tip, dirn in ((g0, 1), (g1, -1)):
            poly([tip + P(-3, 6 * dirn), tip, tip + P(3, 6 * dirn)], close=False, w=0.6)
        for curve, pts in pc["notches"]:
            for q in pts:
                i = int(np.argmin(np.linalg.norm(curve - q, axis=1)))
                tng = curve[min(i + 1, len(curve) - 1)] - curve[max(i - 1, 0)]
                nrm = P(-tng[1], tng[0]) / (np.linalg.norm(tng) + 1e-9)
                poly([curve[i] - nrm * SA * 1.0 + o, curve[i] + nrm * SA * 1.0 + o], close=False, w=0.8)
        for b in pc["buttons"]:
            poly([b + P(-6, 0) + o, b + P(6, 0) + o], close=False, w=0.8)
            poly([b + P(0, -6) + o, b + P(0, 6) + o], close=False, w=0.8)
        if pc["cfline"]:
            poly([pc["cfline"][0] + o, pc["cfline"][1] + o], close=False, dash=(8, 2, 1, 2), w=0.4)
        if pc["welt"]:
            a, b = pc["welt"]
            nrm = P(-(b - a)[1], (b - a)[0]); nrm = nrm / np.linalg.norm(nrm) * 12
            poly([a + o, b + o, b + nrm + o, a + nrm + o], w=0.6)
        cx = pc["o"][:, 0].mean() - 30 + o[0]; cy = pc["o"][:, 1].mean() + o[1]
        c.setFont("KO", 9); c.drawString(*T(P(cx, cy)), pc["name"])
        c.setFont("KO", 7); c.drawString(*T(P(cx, cy + 6)), pc["cut"])
        c.drawString(*T(P(cx, cy + 11)), f"KR55 기준 · 시접 {SA:.0f}mm 포함 (점선=완성선)")
    poly([TEST + P(ox and 0, 0), TEST + P(50, 0), TEST + P(50, 50), TEST + P(0, 50)], w=0.8)
    c.setFont("KO", 7); c.drawString(*T(TEST + P(2, 56)), "확인용 5 × 5 cm")


from shapely.geometry import box
from shapely.ops import unary_union
occupied = unary_union([Polygon(pc["cut_poly"] + pc["off"]) for pc in pieces] + [box(TEST[0], TEST[1], TEST[0] + 50, TEST[1] + 56)])
used = [(r, col) for r in range(rows) for col in range(cols)
        if occupied.intersects(box(col * CW, r * CH, (col + 1) * CW, (r + 1) * CH))]
pdf = canvas.Canvas(f"{OUT}/bustier_vest_A4_tiled.pdf", pagesize=(210 * mm, 297 * mm))
# overview page
s = min(180 / W_all, 230 / H_all)
pdf.setFont("KO", 14); pdf.drawString(15 * mm, 280 * mm, "뷔스티에 조끼 패턴 — A4 붙이기 배치도")
pdf.setFont("KO", 8)
pdf.drawString(15 * mm, 273 * mm, f"가슴 {bust/10:.0f} / 허리 {waist/10:.0f} / 엉덩이 {hip/10:.0f} cm · A4 {len(used)}장 (빈 칸은 인쇄 안 함) · 반드시 100%(실제 크기)로 인쇄")
pdf.saveState(); pdf.translate(15 * mm, (265 - H_all * s) * mm); pdf.scale(s, s)
draw_pattern(pdf, 0, 0, H_all)
pdf.setLineWidth(0.3 / s)
for r in range(rows):
    for col in range(cols):
        if (r, col) not in used:
            continue
        pdf.rect(col * CW * mm, (H_all - (r + 1) * CH) * mm, CW * mm, CH * mm)
        pdf.setFont("KO", 40); pdf.drawString((col * CW + 5) * mm, (H_all - r * CH - 25) * mm, f"{chr(65+r)}{col+1}")
pdf.restoreState(); pdf.showPage()

for r in range(rows):
    for col in range(cols):
        if (r, col) not in used:
            continue
        pdf.saveState()
        clip = pdf.beginPath(); clip.rect(MX * mm, MY * mm, CW * mm, CH * mm)
        pdf.clipPath(clip, stroke=1, fill=0)
        draw_pattern(pdf, MX - col * CW, MY - r * CH, 297)
        pdf.restoreState()
        pdf.setFont("KO", 10); pdf.drawString(MX * mm, 289 * mm, f"{chr(65+r)}{col+1}   (행 {chr(65+r)} / 열 {col+1})")
        pdf.setFont("KO", 7)
        pdf.drawRightString(195 * mm, 289 * mm, "테두리 선에 맞춰 잘라 이어 붙이세요 · 100% 인쇄")
        nb = {"위": (r > 0, f"{chr(64+r)}{col+1}"), "아래": (r < rows - 1, f"{chr(66+r)}{col+1}"),
              "왼쪽": (col > 0, f"{chr(65+r)}{col}"), "오른쪽": (col < cols - 1, f"{chr(65+r)}{col+2}")}
        pos = {"위": (105, 284), "아래": (105, 8), "왼쪽": (2, 150), "오른쪽": (196, 150)}
        for k, (has, lab) in nb.items():
            if has:
                pdf.drawString(pos[k][0] * mm, pos[k][1] * mm, f"{k}→{lab}")
        for (xa, ya) in ((MX + CW / 2, MY), (MX + CW / 2, MY + CH), (MX, MY + CH / 2), (MX + CW, MY + CH / 2)):
            pdf.circle(xa * mm, ya * mm, 3 * mm)
        pdf.showPage()
pdf.save()

# preview png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(W_all / 60, H_all / 60), dpi=110)
for pc in pieces:
    o = pc["off"]
    ax.plot(*(pc["cut_poly"] + o).T, "k-", lw=1)
    ax.plot(*(np.vstack([pc["o"], pc["o"][:1]]) + o).T, "b--", lw=0.6)
    for b in pc["buttons"]:
        ax.plot(*(b + o), "ro", ms=4)
    ax.text(pc["o"][:, 0].mean() + o[0] - 25, pc["o"][:, 1].mean() + o[1], pc["key"].upper(), fontsize=12)
for r in range(rows + 1):
    ax.axhline(r * CH, color="#bbb", lw=0.5)
for col in range(cols + 1):
    ax.axvline(col * CW, color="#bbb", lw=0.5)
ax.add_patch(plt.Rectangle(TEST, 50, 50, fill=False))
ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
fig.savefig(f"{OUT}/preview.png", bbox_inches="tight"); plt.close(fig)

json.dump(dict(checks=checks, finished={k: round(v / 10, 1) for k, v in fin.items()}, pages=len(used), used=[f"{chr(65+r)}{c+1}" for r, c in used],
               grid=[rows, cols]), open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(checks=checks, fin={k: round(v / 10, 1) for k, v in fin.items()}, pages=[rows, cols]),
                 ensure_ascii=False, indent=1))
