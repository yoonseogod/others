# Steampunk-style fitted vest — A4 tiled sewing pattern (1:1), women's size M
import math, os, glob
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from shapely.geometry import Polygon, box, LineString
from shapely.affinity import translate, scale as sscale

# ---------- fonts ----------
F, FB = "Helvetica", "Helvetica-Bold"
cands = glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
if cands:
    pdfmetrics.registerFont(TTFont("KR", cands[0]))
    b = cands[0].replace("NanumGothic.ttf", "NanumGothicBold.ttf")
    pdfmetrics.registerFont(TTFont("KRB", b if os.path.exists(b) else cands[0]))
    F, FB = "KR", "KRB"
elif os.path.exists("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    pdfmetrics.registerFont(TTFont("KR", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", subfontIndex=0))
    F = FB = "KR"
else:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    F = FB = "HYGothic-Medium"

# ---------- geometry helpers (units: cm, y up, x outward from centre) ----------
def bez(p0, p1, p2, p3, n=30):
    out = []
    for i in range(1, n + 1):
        t = i / n; u = 1 - t
        out.append((u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                    u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]))
    return out

def path(start, segs):
    pts = [start]
    for s in segs:
        if len(s) == 1:
            pts.append(s[0])
        else:
            pts += bez(pts[-1], *s)
    return pts

def seg_len(pts):
    return sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

# ----- FRONT (right front as worn; x=0 centre front, overlap to x=-2) -----
# waist line y=0.  Bust 88 / waist 70 / hip 94 (size M)
F_neck   = (7.5, 42.0)
F_shoul  = (11.5, 41.0)
F_under  = (22.5, 18.5)
F_waist  = (21.0, 0.0)
F_side_b = (21.5, -6.0)
F_tip    = (3.0, -15.0)
F_edge_b = (-2.0, -9.0)
F_edge_t = (-2.0, 15.0)   # top button / start of V

front_outline = path(F_edge_b, [
    [F_edge_t],
    [(1.0, 24.0), (5.0, 35.0), F_neck],            # V neckline (gentle curve)
    [F_shoul],                                      # narrow shoulder strap
    [(13.0, 30.0), (16.0, 19.5), F_under],         # deep vest armhole
    [(21.6, 10.0), (20.8, 4.0), F_waist],          # side seam (shaped)
    [F_side_b],
    [F_tip],                                         # pointed hem
    [F_edge_b],
])
front_neck = path(F_edge_t, [[(1.0, 24.0), (5.0, 35.0), F_neck]])
front_arm  = path(F_shoul, [[(13.0, 30.0), (16.0, 19.5), F_under]])

# ----- BACK (centre back on fold, x=0) -----
B_cb_neck = (0.0, 41.0)
B_neck    = (7.5, 43.5)
B_shoul   = (11.5, 43.0)
B_under   = (21.0, 18.5)
B_waist   = (19.5, 0.0)
B_side_b  = (20.0, -5.0)
B_cb_b    = (0.0, -5.0)

back_outline = path(B_cb_b, [
    [B_cb_neck],
    [(3.5, 41.0), (6.5, 42.0), B_neck],
    [B_shoul],
    [(9.5, 33.0), (10.5, 22.0), B_under],          # racer-style cut-in armhole
    [(20.6, 10.0), (19.4, 4.0), B_waist],
    [B_side_b],
    [B_cb_b],
])
back_arm = path(B_shoul, [[(9.5, 33.0), (10.5, 22.0), B_under]])

# darts (fish darts) : centre x, half width, top y, bottom y
FRONT_DART = (10.5, 1.25, 17.0, -9.0)
BACK_DART  = (9.0, 1.0, 15.0, -3.0)

# welt pocket placement on front (two ends)
POCKET = ((11.0, -6.0), (19.0, -4.0))
BUTTONS = [(0.0, y) for y in (13.0, 6.5, 0.0, -6.5)]

SA = 1.0  # seam allowance cm

def with_sa(outline, fold=False):
    poly = Polygon(outline)
    cut = poly.buffer(SA, join_style=2, mitre_limit=3)
    if fold:
        cut = cut.intersection(box(0, -100, 100, 100))
    return poly, cut

# ---------- piece layout on the big sheet (mm, origin bottom-left) ----------
COLS, ROWS = 3, 3
PW, PH = 210, 297
MARG = 10
TW, TH = PW - 2*MARG, PH - 2*MARG   # tile = printable area 190 x 277
SHEET_W, SHEET_H = COLS*TW, ROWS*TH

pieces = []  # dict(name, poly(mm), cut(mm), ...)

def place(outline_cm, dx, dy, fold=False, mirror=False):
    pts = [((-x if mirror else x)*10 + dx, y*10 + dy) for x, y in outline_cm]
    return pts

front_dx, front_dy = 50, 180
back_dx,  back_dy  = 330, 300

def tf_front(p): return (p[0]*10 + front_dx, p[1]*10 + front_dy)
def tf_back(p):  return (p[0]*10 + back_dx,  p[1]*10 + back_dy)

f_poly, f_cut = with_sa(front_outline)
b_poly, b_cut = with_sa(back_outline, fold=True)

def poly_mm(poly, tf):
    return [tf(p) for p in poly.exterior.coords]

# small pieces: welt (finished 10.5 x 2.5, cut doubled on fold) and pocket bag
welt_cm = [(0,0),(11.5,0),(11.5,7.0),(0,7.0)]          # incl. SA
bag_cm  = path((0, 0), [[(12.5, 0)], [(12.5, 14)], [(0, 14)], [(0, 0)]])  # incl. SA, rounded bottom later
bag_cm  = path((0, 14), [[(0, 3)], [(0, 0.8), (1.2, 0), (3, 0)], [(9.5, 0)], [(11.3, 0), (12.5, 0.8), (12.5, 3)], [(12.5, 14)], [(0,14)]])
welt_dx, welt_dy = 300, 40
bag_dx,  bag_dy  = 430, 40

# ---------- drawing ----------
def draw_poly(c, pts, ox, oy, dash=None, width=0.8, close=True):
    p = c.beginPath()
    p.moveTo((pts[0][0]-ox)*mm, (pts[0][1]-oy)*mm)
    for x, y in pts[1:]:
        p.lineTo((x-ox)*mm, (y-oy)*mm)
    if close: p.close()
    c.setDash(*dash) if dash else c.setDash()
    c.setLineWidth(width)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()

def text(c, s, x, y, ox, oy, size=9, bold=False, rot=0, center=True):
    c.saveState()
    c.translate((x-ox)*mm, (y-oy)*mm); c.rotate(rot)
    c.setFont(FB if bold else F, size)
    (c.drawCentredString if center else c.drawString)(0, 0, s)
    c.restoreState()

def grain(c, x, y0, y1, ox, oy, label="식서 방향 (Grain)"):
    c.setLineWidth(0.6)
    X = (x-ox)*mm; Y0 = (y0-oy)*mm; Y1 = (y1-oy)*mm
    c.line(X, Y0, X, Y1)
    for Y, d in ((Y1, -1), (Y0, 1)):
        c.line(X, Y, X-2*mm, Y+d*4*mm); c.line(X, Y, X+2*mm, Y+d*4*mm)
    text(c, label, x+3, (y0+y1)/2, ox, oy, 8, rot=90)

def notch(c, pt, tf, nrm, ox, oy):
    x, y = tf(pt)
    c.setLineWidth(0.8)
    c.line((x-ox)*mm, (y-oy)*mm, (x+nrm[0]*SA*10-ox)*mm, (y+nrm[1]*SA*10-oy)*mm)

def draw_dart(c, d, tf, ox, oy):
    cx, hw, top, bot = d
    pts = [tf(p) for p in [(cx, top), (cx-hw, 0), (cx, bot), (cx+hw, 0), (cx, top)]]
    draw_poly(c, pts, ox, oy, width=0.6, close=False)
    x, y = tf((cx, 0)); c.setLineWidth(0.4)
    c.line((x-ox-hw*10-4)*mm, (y-oy)*mm, (x-ox+hw*10+4)*mm, (y-oy)*mm)

def draw_pieces(c, ox, oy):
    c.setStrokeColorRGB(0, 0, 0)
    # FRONT
    draw_poly(c, poly_mm(f_cut, tf_front), ox, oy, width=1.0)
    draw_poly(c, poly_mm(f_poly, tf_front), ox, oy, dash=(4, 2), width=0.5)
    draw_dart(c, FRONT_DART, tf_front, ox, oy)
    # center front line
    a, b = tf_front((0, 15)), tf_front((0, -11.5))
    c.setDash([6, 2, 1, 2]); c.setLineWidth(0.5)
    c.line((a[0]-ox)*mm, (a[1]-oy)*mm, (b[0]-ox)*mm, (b[1]-oy)*mm); c.setDash()
    text(c, "앞 중심선 CF", a[0]+3, a[1]-60, ox, oy, 7, rot=90)
    # waist line
    a, b = tf_front((-2, 0)), tf_front((21, 0))
    c.setDash(1, 2); c.line((a[0]-ox)*mm, (a[1]-oy)*mm, (b[0]-ox)*mm, (b[1]-oy)*mm); c.setDash()
    text(c, "허리선 WAIST", tf_front((4, 0.5))[0], tf_front((4, 0.5))[1], ox, oy, 7)
    # buttons / buttonholes
    for bx, by in BUTTONS:
        x, y = tf_front((bx, by))
        c.circle((x-ox)*mm, (y-oy)*mm, 1.5*mm)
        c.setLineWidth(1.2); c.line((x-ox-2)*mm, (y-oy)*mm, (x-ox+18)*mm, (y-oy)*mm); c.setLineWidth(0.6)
    text(c, "단추 / 단춧구멍 1.8cm (기어 단추)", tf_front((1, 9.5))[0]+5, tf_front((1, 9.5))[1], ox, oy, 7, rot=90)
    # pocket
    p0, p1 = tf_front(POCKET[0]), tf_front(POCKET[1])
    pocket_box = [p0, p1, (p1[0], p1[1]+25), (p0[0], p0[1]+25)]
    draw_poly(c, pocket_box, ox, oy, width=1.0)
    text(c, "웰트 포켓 위치", (p0[0]+p1[0])/2, (p0[1]+p1[1])/2-8, ox, oy, 7)
    # notches
    notch(c, (21.4, 10.0), tf_front, (1, 0), ox, oy)
    notch(c, (14.6, 25.0), tf_front, (0.95, 0.3), ox, oy)
    notch(c, (21.0, 0.0), tf_front, (1, 0), ox, oy)
    grain(c, tf_front((6, 0))[0], tf_front((6, -8))[1], tf_front((6, 28))[1], ox, oy)
    tx, ty = tf_front((11, 30))
    text(c, "① 앞판 FRONT", tx-15, ty, ox, oy, 14, bold=True)
    text(c, "겉감 2장 · 안감 2장 (좌우 대칭)", tx-15, ty-8, ox, oy, 9)
    text(c, "시접 1cm 포함 · 사이즈 M", tx-15, ty-15, ox, oy, 8)

    # BACK
    draw_poly(c, poly_mm(b_cut, tf_back), ox, oy, width=1.0)
    draw_poly(c, poly_mm(b_poly, tf_back), ox, oy, dash=(4, 2), width=0.5)
    draw_dart(c, BACK_DART, tf_back, ox, oy)
    a, b = tf_back((0, 42)), tf_back((0, -6))
    c.setLineWidth(2.0); c.line((a[0]-ox)*mm, (a[1]-oy)*mm, (b[0]-ox)*mm, (b[1]-oy)*mm); c.setLineWidth(0.6)
    text(c, "골선 FOLD — 원단을 접어서 재단", a[0]+4, (a[1]+b[1])/2, ox, oy, 9, bold=True, rot=90)
    a, b = tf_back((0, 0)), tf_back((19.5, 0))
    c.setDash(1, 2); c.line((a[0]-ox)*mm, (a[1]-oy)*mm, (b[0]-ox)*mm, (b[1]-oy)*mm); c.setDash()
    notch(c, (20.6, 10.0), tf_back, (1, 0), ox, oy)
    notch(c, (19.5, 0.0), tf_back, (1, 0), ox, oy)
    grain(c, tf_back((15, 0))[0], tf_back((15, -3))[1], tf_back((15, 14))[1], ox, oy)
    tx, ty = tf_back((5, 25))
    text(c, "② 뒤판 BACK", tx+15, ty, ox, oy, 14, bold=True)
    text(c, "겉감 1장 · 안감 1장 (골선)", tx+15, ty-8, ox, oy, 9)
    text(c, "시접 1cm 포함", tx+15, ty-15, ox, oy, 8)

    # WELT
    w = [(x*10+welt_dx, y*10+welt_dy) for x, y in welt_cm] + [(welt_dx, welt_dy)]
    draw_poly(c, w, ox, oy, close=False)
    c.setDash(1, 2); c.line((welt_dx-ox)*mm, (welt_dy+35-oy)*mm, (welt_dx+115-ox)*mm, (welt_dy+35-oy)*mm); c.setDash()
    text(c, "③ 웰트 WELT  4장", welt_dx+57, welt_dy+50, ox, oy, 10, bold=True)
    text(c, "(겉감 2 + 심지 2) 접는선 ·····", welt_dx+57, welt_dy+20, ox, oy, 7)
    text(c, "시접 1cm 포함", welt_dx+57, welt_dy+12, ox, oy, 7)
    # BAG
    bg = [(x*10+bag_dx, y*10+bag_dy) for x, y in bag_cm]
    draw_poly(c, bg, ox, oy, close=False)
    text(c, "④ 주머니감 POCKET BAG", bag_dx+62, bag_dy+105, ox, oy, 10, bold=True)
    text(c, "안감 4장", bag_dx+62, bag_dy+97, ox, oy, 8)
    text(c, "시접 1cm 포함", bag_dx+62, bag_dy+90, ox, oy, 7)
    grain(c, bag_dx+30, bag_dy+20, bag_dy+80, ox, oy, "식서")

def test_square(c, x, y):
    c.setLineWidth(0.8); c.rect(x*mm, y*mm, 50*mm, 50*mm)
    c.setFont(FB, 9); c.drawCentredString((x+25)*mm, (y+28)*mm, "5 cm × 5 cm")
    c.setFont(F, 7); c.drawCentredString((x+25)*mm, (y+21)*mm, "인쇄 확인용 테스트 사각형")
    c.drawCentredString((x+25)*mm, (y+16)*mm, "Test square")

def tile_page(c, col, row):
    ox = col*TW - MARG
    oy = (ROWS-1-row)*TH - MARG
    c.saveState()
    pth = c.beginPath(); pth.rect(MARG*mm, MARG*mm, TW*mm, TH*mm)
    c.clipPath(pth, stroke=0, fill=0)
    draw_pieces(c, ox, oy)
    c.restoreState()
    # tile frame (cut / tape line)
    c.setLineWidth(0.3); c.setStrokeColorRGB(0.55, 0.55, 0.55); c.setDash(2, 2)
    c.rect(MARG*mm, MARG*mm, TW*mm, TH*mm); c.setDash(); c.setStrokeColorRGB(0, 0, 0)
    # alignment diamonds at edge midpoints
    lab = f"{chr(65+row)}{col+1}"
    c.setFillColorRGB(0.6, 0.6, 0.6)
    for (x, y) in ((PW/2, MARG), (PW/2, PH-MARG), (MARG, PH/2), (PW-MARG, PH/2)):
        p = c.beginPath(); s = 5
        p.moveTo(x*mm, (y+s)*mm); p.lineTo((x+s)*mm, y*mm); p.lineTo(x*mm, (y-s)*mm); p.lineTo((x-s)*mm, y*mm); p.close()
        c.drawPath(p, stroke=0, fill=1)
    c.setFillColorRGB(0, 0, 0)
    # labels
    c.setFont(FB, 22); c.setFillColorRGB(0.8, 0.8, 0.8)
    c.drawCentredString(PW/2*mm, (PH/2)*mm, lab)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(F, 7)
    c.drawString(MARG*mm, 4*mm, f"페이지 {lab}  (행 {chr(65+row)} / 열 {col+1})  — 회색 점선을 따라 여백을 자르고 ◆ 표시를 맞춰 붙이세요")
    c.drawRightString((PW-MARG)*mm, 4*mm, "실제 크기 100% 인쇄")
    if (col, row) == (0, 0):
        test_square(c, MARG+8, PH-MARG-60)

def cover(c):
    c.setFont(FB, 22); c.drawString(20*mm, 270*mm, "스팀펑크 기어 단추 조끼 패턴")
    c.setFont(F, 11); c.drawString(20*mm, 262*mm, "Fitted Steampunk Vest — A4 타일 패턴 (실물 크기 1:1) · 여성 M 사이즈")
    y = 248
    def line(s, size=9.5, bold=False, dy=5.6):
        nonlocal y
        c.setFont(FB if bold else F, size); c.drawString(20*mm, y*mm, s); y -= dy
    line("■ 인쇄 방법", 11, True, 6.5)
    line("1. 프린터 설정에서 반드시 '실제 크기 / 100% / 크기 조절 없음' 으로 인쇄하세요. ('페이지에 맞춤' 금지)")
    line("2. A1 페이지의 5cm 테스트 사각형을 자로 재서 정확히 5cm인지 확인합니다.")
    line("3. 각 페이지의 회색 점선 바깥 여백을 자르고, ◆ 표시와 선을 맞춰 아래 배치도 순서로 테이프로 붙입니다.")
    y -= 2
    line("■ 치수 (완성 기준, M)", 11, True, 6.5)
    line("가슴둘레 88cm  ·  허리둘레 70cm  ·  앞 길이(어깨~앞 끝) 약 57cm  ·  뒤 길이 약 46cm")
    line("사이즈 조정: 옆선에서 1cm 늘리면 둘레 4cm 증가. 길이는 허리선 위아래에서 평행하게 늘리거나 줄이세요.")
    y -= 2
    line("■ 재단 목록 (시접 1cm 포함 — 바깥 실선 = 재단선, 안쪽 점선 = 박음선)", 11, True, 6.5)
    line("① 앞판  — 겉감 2장, 안감 2장 (좌우 대칭)")
    line("② 뒤판  — 겉감 1장, 안감 1장 (골선에 대고 재단)")
    line("③ 웰트  — 겉감 2장 + 접착심지 2장        ④ 주머니감 — 안감 4장")
    line("부자재: 기어 모양 금속 단추 4개(지름 1.8cm), 앞판용 접착심지, 브로치(선택)")
    line("            레이스 트림 약 1.2m (앞판 다트선 장식용)")
    line("권장 원단: 겉감 — 코튼 트윌/개버딘/스트레치 코튼 약 0.8m (150cm 폭) · 안감 — 0.7m")
    y -= 2
    line("■ 만드는 순서", 11, True, 6.5)
    for s in [
        "1. 앞판 겉감에 심지를 붙이고, 겉감·안감 모두 다트를 박아 앞판은 중심 쪽, 뒤판은 중심 쪽으로 다트를 넘깁니다.",
        "2. 앞판 겉감에 웰트 포켓을 만듭니다 (포켓 위치 표시 참고). 원하면 다트선 위에 레이스 트림을 상침합니다.",
        "3. 겉감끼리 어깨선을 박고, 안감끼리도 어깨선을 박습니다. (옆선은 아직 박지 않음)",
        "4. 겉감과 안감을 겉끼리 맞대고 앞단, 밑단, 목둘레, 진동둘레를 박습니다. 옆선만 남겨둡니다.",
        "5. 곡선 시접에 가위집을 넣고, 어깨 끈 속으로 뒤판을 빼내어 뒤집습니다 (bagging 방식).",
        "6. 옆선: 겉감 앞·뒤, 안감 앞·뒤를 이어 하나의 링으로 박고, 안감에 남긴 창구로 정리한 뒤 공그르기.",
        "7. 다림질 후 단춧구멍(1.8cm)을 만들고 기어 단추를 답니다. 왼쪽 가슴에 브로치를 달아 완성!",
    ]:
        line(s, 9)
    y -= 3
    # layout diagram
    c.setFont(FB, 11); c.drawString(20*mm, y*mm, "■ 페이지 배치도 (3 × 3 = 9장)"); y -= 4
    s = 0.12
    bx, by = 20, y - ROWS*TH*s
    for r in range(ROWS):
        for col in range(COLS):
            x0 = bx + col*TW*s; y0 = by + (ROWS-1-r)*TH*s
            c.setLineWidth(0.4); c.rect(x0*mm, y0*mm, TW*s*mm, TH*s*mm)
            c.setFont(FB, 9); c.setFillColorRGB(0.6, 0.6, 0.6)
            c.drawCentredString((x0+TW*s/2)*mm, (y0+TH*s/2)*mm, f"{chr(65+r)}{col+1}")
            c.setFillColorRGB(0, 0, 0)
    # mini pieces
    c.saveState()
    c.translate(bx*mm, by*mm); c.scale(s, s)
    def mini(pts):
        p = c.beginPath(); p.moveTo(pts[0][0]*mm, pts[0][1]*mm)
        for x, yy in pts[1:]: p.lineTo(x*mm, yy*mm)
        p.close(); c.setLineWidth(2); c.drawPath(p)
    mini(poly_mm(f_cut, tf_front)); mini(poly_mm(b_cut, tf_back))
    mini([(x*10+welt_dx, yy*10+welt_dy) for x, yy in welt_cm])
    mini([(x*10+bag_dx, yy*10+bag_dy) for x, yy in bag_cm])
    c.restoreState()
    c.setFont(F, 7.5)
    c.drawString((bx+COLS*TW*s+6)*mm, (by+TH*s*2.5)*mm, "※ 이 패턴은 사진을 참고해 제도한 기본 패턴입니다.")
    c.drawString((bx+COLS*TW*s+6)*mm, (by+TH*s*2.5-5)*mm, "   본 원단 재단 전 광목으로 가봉(머슬린)해서")
    c.drawString((bx+COLS*TW*s+6)*mm, (by+TH*s*2.5-10)*mm, "   핏을 확인하는 것을 권장합니다.")
    c.drawString((bx+COLS*TW*s+6)*mm, (by+TH*s*2.5-20)*mm, f"앞 진동둘레 {seg_len(front_arm):.1f}cm · 뒤 진동둘레 {seg_len(back_arm):.1f}cm")
    c.drawString((bx+COLS*TW*s+6)*mm, (by+TH*s*2.5-25)*mm, f"앞 목둘레(V) {seg_len(front_neck):.1f}cm")
    test_square(c, bx+COLS*TW*s+12, by+4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steampunk_vest_pattern_A4.pdf")
c = canvas.Canvas(out, pagesize=(PW*mm, PH*mm))
c.setTitle("Steampunk Vest Pattern A4")
cover(c); c.showPage()
for r in range(ROWS):
    for col in range(COLS):
        tile_page(c, col, r); c.showPage()
c.save()
print(out)
# sanity: everything inside sheet
for nm, pts in (("front", poly_mm(f_cut, tf_front)), ("back", poly_mm(b_cut, tf_back))):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    print(nm, min(xs), max(xs), min(ys), max(ys), "sheet", SHEET_W, SHEET_H)
