import math, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open('vest.py').read().split('out = os.path')[0])
import fitz
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
R = []
def rep(n, title, ok, detail): R.append((n, title, ok, detail)); print(f"[{n:2}] {'OK ' if ok else 'XX '} {title}: {detail}")

# 1 print scale
D = fitz.open('steampunk_vest_pattern_A4.pdf')
sq = [round(dr['rect'].width/72*25.4, 2) for pg in (D[0], D[1]) for dr in pg.get_drawings() if 140 < dr['rect'].width < 144 and abs(dr['rect'].width-dr['rect'].height) < .5]
sizes = {(round(p.rect.width/72*25.4), round(p.rect.height/72*25.4)) for p in D}
rep(1, "인쇄 축척/페이지", sq and all(abs(v-50) < .05 for v in sq) and len(D) == 17 and sizes == {(210, 297), (297, 210)},
    f"테스트 사각형 {sq} mm, {len(D)}쪽, 크기 {sorted(sizes)}")

# 2 layout: inside sheet, no overlaps between pieces
def placed(p):
    pts = p.ops[0][1]; return Polygon([(x*10+p.dx, y*10+p.dy) for x, y in pts]).buffer(0)
pl = {p.name: placed(p) for p in PIECES}
inside = all(g.bounds[0] >= 0 and g.bounds[1] >= 0 and g.bounds[2] <= SHEET_W and g.bounds[3] <= SHEET_H for g in pl.values())
ov = [(a, b) for i, a in enumerate(pl) for b in list(pl)[i+1:] if pl[a].intersection(pl[b]).area > 1]
sqbox = box(MARG+8-MARG, SHEET_H-60, MARG+8-MARG+50, SHEET_H-10)
ov_sq = [n for n, g in pl.items() if g.intersects(sqbox)]
rep(2, "배치 (용지 안 / 조각 겹침)", inside and not ov and not ov_sq, f"용지 안={inside}, 겹침={ov or '없음'}, 테스트사각형과 겹침={ov_sq or '없음'}")

# 3 ease
def xs_at(poly, y):
    g = LineString([(-50, y), (60, y)]).intersection(poly.exterior); return sorted(q.x for q in getattr(g, "geoms", [g]))
bust = 2*(xs_at(front_poly, 20)[-1] + xs_at(back_poly, 20)[-1])
waist = 2*((S_w[0] - 2*smooth_interp(0, W_TAB)) + (B_w[0] - 2*BACK_DART[1]))
hip = 2*(xs_at(front_poly, -16)[-1] - 2*smooth_interp(-16, W_TAB) + xs_at(back_poly, -16)[-1])
rep(3, "완성 둘레 vs 몸(88/70/94)", bust > 88+2 and waist > 70+1 and hip >= 94+1.5,
    f"가슴 {bust:.1f} (+{bust-88:.1f}) · 허리 {waist:.1f} (+{waist-70:.1f}) · 밑단(허리-16) {hip:.1f}")

# 4 seam lengths
sideF = (plen(path(U, [SIDE_F_UP])), plen(path(S_w, [SIDE_F_LO]))); sideB = (plen(path(U_b, [SIDE_B_UP])), plen(path(B_w, [SIDE_B_LO])))
shF, shB = math.dist(N, Sh), math.dist(B_n, B_sh)
top_cf = CF_SEAM.project(Point(NOTCH_CF[0])); top_sf = SF_SEAM.project(Point(NOTCH_SF[0]))
bot_cf = CF_SEAM.length - CF_SEAM.project(Point(NOTCH_CF[1])); bot_sf = SF_SEAM.length - SF_SEAM.project(Point(NOTCH_SF[1]))
ok4 = abs(sideF[0]-sideB[0]) < .1 and abs(sideF[1]-sideB[1]) < .1 and abs(shF-shB) < .1 and abs(top_cf-top_sf) < .1 and abs(bot_cf-bot_sf) < .1
rep(4, "솔기 길이 맞춤", ok4, f"옆선 앞{sideF[0]:.2f}/{sideF[1]:.2f} 뒤{sideB[0]:.2f}/{sideB[1]:.2f} · 어깨 {shF:.2f}/{shB:.2f} · 절개선 위구간 {top_cf:.2f}/{top_sf:.2f} 아래구간 {bot_cf:.2f}/{bot_sf:.2f}")

# 5 junction angles
J = {"옆목점": ang(sub(Sh, N), sub(NECK_F[1], N)) + ang(sub(B_sh, B_n), sub(NECK_B[0], B_n)),
     "어깨끝": ang(sub(N, Sh), sub(ARM_F[0], Sh)) + ang(sub(B_n, B_sh), sub(ARM_B[0], B_sh)),
     "겨드랑이": ang(sub(ARM_F[1], U), sub(SIDE_F_UP[0], U)) + ang(sub(ARM_B[1], U_b), sub(SIDE_B_UP[0], U_b)),
     "옆밑단": ang(sub(SIDE_F_LO[1], S_b), sub(T, S_b)) + ang(sub(SIDE_B_LO[1], B_sb), sub(B_cbb, B_sb)),
     "뒷목CB": 2*ang(sub(B_cbb, B_cb), sub(NECK_B[1], B_cb)), "뒤밑단CB": 2*ang(sub(B_cb, B_cbb), sub(B_sb, B_cbb))}
rep(5, "이음 지점 각도 (합 180°)", all(abs(v-180) < 3 for v in J.values()), " · ".join(f"{k} {v:.1f}°" for k, v in J.items()))

# 6 kinks along outlines (excluding intended corners)
def kinks(poly, allowed, thr=12):
    ring = resample(list(poly.exterior.coords)[:-1], 0.4); out = []
    for i in range(len(ring)):
        a, b, c = ring[i-1], ring[i], ring[(i+1) % len(ring)]
        t = 180 - ang(sub(a, b), sub(c, b))
        if t > thr and not any(math.dist(b, q) < 1.2 for q in allowed): out.append((round(b[0], 1), round(b[1], 1), round(t)))
    return out
cor_f = [N, Sh, U, S_b]
k_cf = kinks(cf_poly, cor_f + [CF_SEAM.coords[0], CF_SEAM.coords[-1]])
k_sf = kinks(sf_poly, cor_f + [SF_SEAM.coords[0], SF_SEAM.coords[-1]])
k_b = kinks(back_poly, [B_n, B_sh, U_b, B_sb, B_cb, B_cbb])
rep(6, "곡선 꺾임 (의도된 이음 모서리 제외, 12° 초과)", not (k_cf or k_sf or k_b), f"앞중심 {k_cf or '없음'} · 옆앞 {k_sf or '없음'} · 뒤 {k_b or '없음'}")

# 7 geometry validity & SA
val = {n: g.is_valid for n, g in (("cf", cf_poly), ("sf", sf_poly), ("back", back_poly), ("face_f", face_f), ("face_b", face_b))}
sa_ok = all(cutline(g).buffer(0.01).contains(g) for g in (cf_poly, sf_poly, face_f)) and cutline(back_poly, True).buffer(0.01).contains(back_poly)
gap = cf_poly.intersection(sf_poly).area
union_vs_front = front_poly.area - (cf_poly.area + sf_poly.area)
rep(7, "도형 유효성/시접", all(val.values()) and sa_ok and gap < .05,
    f"유효 {val}, 시접선이 박음선 감쌈={sa_ok}, 두 앞판 겹침 {gap:.3f}㎠, 절개 분량(다트) {union_vs_front:.1f}㎠")

# 8 buttons
over = min(b[0] for b in BUTTONS) - (-2.0)
in_cf = all(cf_poly.contains(Point(b).buffer(0.9)) for b in BUTTONS) and all(face_f.contains(Point(b).buffer(0.9)) for b in BUTTONS)
sp = [round(BUTTONS[i][1]-BUTTONS[i+1][1], 1) for i in range(3)]
top_gap = E_t[1] - 2.5 - (BUTTONS[0][1]+0.9)   # fillet reach 2.5
bot_gap = (BUTTONS[-1][1]-0.9) - (E_b[1]+1.5)
rep(8, "단추/단춧구멍", in_cf and over >= 1.8 and top_gap >= 0 and bot_gap >= 0,
    f"앞판·안단 안에 들어감={in_cf}, 여밈 {over}cm, 간격 {sp}, V곡선 시작까지 {top_gap:.1f}cm, 밑 모서리 곡선까지 {bot_gap:.1f}cm")

# 9 pocket
wb = Polygon(welt_box()); bag_depth = BH - 2*SA
hem_line = LineString(list(front_poly.exterior.coords))
d_hem = min(Point(p).distance(LineString([q for q in front_poly.exterior.coords if q[1] < -14])) for p in POCKET)
cross = LineString(POCKET).intersects(CF_SEAM)
side_d = Point(POCKET[1]).distance(LineString(path(S_w, [SIDE_F_LO])))
face_hit = wb.intersection(face_f).area
bag_poly = Polygon([POCKET[0], POCKET[1], (POCKET[1][0], POCKET[1][1]-bag_depth), (POCKET[0][0], POCKET[0][1]-bag_depth)])
bag_face = bag_poly.intersection(face_f).area
rep(9, "웰트 포켓/주머니감", front_poly.contains(wb) and cross and bag_depth < d_hem - 0.5 and side_d > 2 and face_hit < .01 and bag_face < .01,
    f"절개선 가로지름={cross}, 밑단까지 {d_hem:.1f}cm vs 주머니 깊이 {bag_depth}cm, 옆선까지 {side_d:.1f}cm, 안단과 겹침 웰트 {face_hit:.2f}㎠ / 주머니감 {bag_face:.1f}㎠")

# 10 text / fonts
txt = "".join(p.get_text() for p in D)
fonts = {f[3] for p in D for f in p.get_fonts()}
need = ["앞 중심판", "옆 앞판", "뒤판", "앞 안단", "뒷목 안단", "웰트", "주머니감", "골선", "5 cm × 5 cm", "실제 크기"]
miss = [w for w in need if w not in txt]
fitz_ok = "�" not in txt
cover_w = max(dr['rect'].x1 for dr in []) if False else None
blocks = [b for b in D[0].get_text("blocks") if b[2] > D[0].rect.width - 5*72/25.4]
rep(10, "글자/폰트", not miss and fitz_ok and not blocks, f"폰트 {fonts}, 누락 라벨 {miss or '없음'}, 깨진 글자 없음={fitz_ok}, 표지 오른쪽 넘침 {len(blocks)}줄")
# 10b labels inside their pieces
from shapely.geometry import Polygon as _P
outside = []
for p in PIECES:
    g = _P(p.ops[0][1]).buffer(0)
    for op in p.ops:
        if op[0] == "text" and not g.contains(Point(op[1])): outside.append((p.name, op[2]))
rep(11, "글자가 조각 안에 있는지", not outside, outside or "전부 안쪽")
print("PASS", sum(r[2] for r in R), "/", len(R))
