"""Fitted peplum jacket - pattern drafting from Bunka (文化式 新原型) bodice block.

Units: cm. Coordinates: y up, waist line y=0, centre back x=0, centre front x=B/2+6.
All pieces are NET (finished) lines; seam allowance added only at render time.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import brentq
from geom import (min_radius, inflections, cluster, spline_through, P, unit, dirdeg, rot, rotv, plen, tan_start, tan_end, line, hermite,
                  chain, split_at_line, point_at_len, rev, N)

# ------------------------------------------------------------------ measurements
M = dict(B=84.0, W=66.0, H=92.0, BL=38.0, HL=18.0, SL=58.0, wrist=16.0,
         upper_arm=27.0)   # upper arm: typical size-55 value (not measured)

# ------------------------------------------------------------------ design params
J = dict(
    ah_drop=1.0,        # jacket armhole drop from block chest line
    side_ease=0.5,      # added at side seam, front and back each
    neck_widen=0.5,     # neckline widened at SNP (along shoulder)
    cf_drop=1.0,        # front neck lowered at CF
    waist_ease=6.0,     # total waist ease (W + 6)
    flare=2.0,          # peplum flare at hem for every vertical seam (cm, total per seam)
    cb_waist=3.5,       # CB panel width at waist
    hem=dict(cb=-20.0, v=-19.5, outer=-17.5, side=-15.0, princess=-15.5, cf=-13.0),
    k_yoke_arm=(0.8, 0.8),
    front_width_loss_max=0.2,  # front armhole may move inside the Bunka front-width line by <= 0.2 cm
    pr_beta=12.0, pr_bp_dir=-64.0, pr_k=0.6,
    cap_ease=1.3,
    biceps_ease=4.0,    # jacket sleeve: biceps width = upper arm + 4 cm       # sleeve-cap ease (denim/twill)
    sleeve_wrist=22.0,  # finished sleeve width at wrist (before cuff)
    wrist_fwd=4.0,      # wrist centre shifted forward (arm hangs forward)
    elbow_dart=1.5,
    collar_h=3.5, collar_rise=1.0,
    cuff_h=10.0,        # straight cylinder cuff (covered later)
)

PCT = dict(a=.14, b=.15, c=.11, d=.35, e=.18, f=.07)   # Bunka waist dart distribution


def root(f, lo, hi, n=24, xtol=1e-12):
    """brentq on the first sign change found by scanning [lo, hi] (robust bracket)."""
    def g(x):
        try:
            return f(x)
        except (ValueError, IndexError):
            return float('nan')
    xs = np.linspace(lo, hi, n + 1)
    fs = [g(x) for x in xs]
    for i in range(n):
        if fs[i] == 0:
            return xs[i]
        if fs[i] * fs[i + 1] < 0:      # nan compares False -> skipped
            return brentq(f, xs[i], xs[i + 1], xtol=xtol)
    raise ValueError(f'no root in [{lo}, {hi}]')


class ProfileSeam:
    """Straight centre line; the two seam edges are mirror images (offset +-s(t)/2).
    Mirror symmetry => both edges have *identical* length and mirrored end angles."""

    def __init__(self, top, through, knots, n=N, flat_top=False):
        self.top = top
        self.u = unit(through - top)
        nR = P(-self.u[1], self.u[0])
        self.nR = nR if nR[0] > 0 else -nR
        ty = lambda y: (top[1] - y) / (-self.u[1])
        self.ty = ty
        tk = [ty(y) for y, _ in knots]
        # offsets are perpendicular to the centre line (keeps both edges congruent); scale by cos(tilt)
        # so that the HORIZONTAL amount taken out at each level equals the specified suppression.
        cosphi = abs(self.u[1])
        sv = [v * cosphi for _, v in knots]
        if flat_top:   # zero slope at the top knot -> edges parallel to centre line there
            self.f = CubicSpline(tk, sv, bc_type=((1, 0.0), 'not-a-knot'))
        else:
            self.f = PchipInterpolator(tk, sv)
        self.t = np.linspace(0, tk[-1], n)

    def edge(self, sign):
        t = self.t
        return self.top + np.outer(t, self.u) + sign * np.outer(self.f(t) / 2, self.nR)

    def point(self, sign, y):
        t = self.ty(y)
        return self.top + t * self.u + sign * self.f(t) / 2 * self.nR


def build():
    B, W, BL = M['B'], M['W'], M['BL']
    K = {}   # key values / points
    # ============================ BUNKA BODICE BLOCK ============================
    half = B / 2 + 6
    CF = half
    yC = BL - (B / 12 + 13.7)               # chest (BL) line
    yFtop = yC + B / 5 + 8.3                # front neck top line
    xbw = B / 8 + 7.4                       # back width
    fw = B / 8 + 6.2
    xfw = CF - fw                           # front width line
    BP = P(CF - (fw / 2 + 0.7), yC)
    xF = xfw - B / 32
    xs = (xbw + xF) / 2                     # side line
    ysb = BL - 8                            # shoulder-blade line
    yE = (ysb + yC) / 2 - 0.5
    nb = B / 24 + 3.4                       # back neck width
    CBn = P(0, BL)
    SNPb = P(nb, BL + nb / 3)
    SNPf = P(CF - (nb - 0.2), yFtop - 0.5)
    CFn = P(CF, yFtop - (nb + 0.5))
    ush_f = dirdeg(180 + 22)
    SPf_blk = SNPf + ush_f * ((SNPf[0] - (xfw - 1.8)) / math.cos(math.radians(22)))
    tri = float(np.linalg.norm(SPf_blk - SNPf))
    dartb = B / 32 - 0.8
    ush_b = dirdeg(-18)
    SPb = SNPb + ush_b * (tri + dartb)
    G = P(xfw, yE)
    dartf = B / 4 - 2.5
    G2 = rot(G, BP, dartf)                  # lower leg of armhole dart
    D = P(xbw / 2 + 1, ysb)                 # back shoulder-dart apex
    P1 = SNPb + ush_b * ((D[0] - SNPb[0]) / ush_b[0])
    thd = math.degrees(2 * math.asin(dartb / 2 / np.linalg.norm(P1 - D)))
    P2 = rot(P1, D, -thd)
    K.update(half=half, CF=CF, yC=yC, yFtop=yFtop, xbw=xbw, xfw=xfw, BP=BP, xs=xs, ysb=ysb, yE=yE,
             nb=nb, CBn=CBn, SNPb=SNPb, SNPf=SNPf, CFn=CFn, SPf_blk=SPf_blk, tri=tri, dartb=dartb,
             SPb=SPb, G=G, G2=G2, dartf=dartf, D=D, P1=P1, P2=P2, thd=thd, ush_f=ush_f, ush_b=ush_b)

    # ============================ JACKET DEVELOPMENT ============================
    H = J['hem']
    fl = J['flare']
    yA = yC - J['ah_drop']
    waist_half = (W + J['waist_ease']) / 2
    supp = half - waist_half
    sV = supp * (PCT['e'] + PCT['f'])
    sO = supp * PCT['d']
    sS = supp * PCT['c']
    sP = supp * (PCT['a'] + PCT['b'])
    K.update(yA=yA, supp=supp, sV=sV, sO=sO, sS=sS, sP=sP, waist_half=waist_half)

    # ---- side seam (mirror)
    side = ProfileSeam(P(xs, yA), P(xs, yA - 1),
                       [(yA, -2 * J['side_ease']), (0, sS), (H['side'], -fl)])
    side_b, side_f = side.edge(-1), side.edge(+1)

    # ---- back armhole (body frame)
    Sb = side_b[0]
    Bw = P(xbw, yE + 0.8)
    arm_b = chain([SPb, Bw, Sb], [rotv(ush_b, -90), P(0, -1), rotv(tan_start(side_b), 90)],
                  ks=[(1.0, 0.9), (0.9, 1.25)])
    up_b, low_b = split_at_line(arm_b, P(0, ysb), P(1, 0))
    A = low_b[0].copy()
    O = P(A[0] - 2.5, ysb)

    # ---- yoke: close shoulder dart about D (apex on yoke line)
    SPy = rot(SPb, D, thd)
    Ay = rot(A, D, thd)
    Oy = rot(O, D, thd)
    SNPbj = SNPb + ush_b * J['neck_widen']
    ush_y = unit(SPy - SNPbj)
    Lsh = float(np.linalg.norm(SPy - SNPbj))
    # yoke armhole: pick the tension that gives a single-direction curve (no S) - size independent
    def yoke_arm(k0, k1):
        return hermite(SPy, rotv(ush_y, -90), Ay, rotv(tan_end(up_b), thd), k0, k1)
    cand = [(inflections(yoke_arm(k0, k1)), -min_radius(yoke_arm(k0, k1)), k0, k1)
            for k0 in np.arange(0.4, 1.41, 0.1) for k1 in np.arange(0.4, 1.41, 0.1)]
    _, _, ky0, ky1 = min(cand)
    arm_y = yoke_arm(ky0, ky1)
    neck_b = hermite(CBn, P(1, 0), SNPbj, rotv(ush_y, 90), 1.1, 0.45)
    yoke = [('fold_cb', line(P(0, ysb), CBn)), ('neck', neck_b), ('shoulder', line(SNPbj, SPy)),
            ('armhole', arm_y), ('yoke_seam_out', line(Ay, D)), ('yoke_seam_in', line(D, P(0, ysb)))]

    # ---- back panel seams (mirror)
    V = ProfileSeam(D, P(J['cb_waist'] + sV / 2, 0),
                    [(ysb, 0), (yC, 0.0), (0, sV), (H['v'], -fl)])
    Os = ProfileSeam(O, P(O[0] - 1.6, 0),
                     [(ysb, 0), (yC, 0.0), (0, sO), (H['outer'], -fl)])
    vL, vR = V.edge(-1), V.edge(+1)
    oL, oR = Os.edge(-1), Os.edge(+1)

    def hem_between(L, R, dL=None, dR=None, k=(1.0, 1.0)):
        d0 = rotv(tan_end(L), 90) if dL is None else dL
        d1 = rotv(tan_end(R), 90) if dR is None else dR
        return hermite(L[-1], d0, R[-1], d1, *k)

    cbline = line(P(0, ysb), P(0, H['cb']))
    hem_cb = hem_between(cbline, vL, dL=P(1, 0))
    hem_mid = hem_between(vR, oL)
    hem_sb = hem_between(oR, side_b)
    cbp = [('top', line(P(0, ysb), D)), ('v_seam', vL), ('hem', rev(hem_cb)), ('fold_cb', rev(cbline))]
    midb = [('top', line(D, O)), ('outer_seam', oL), ('hem', rev(hem_mid)), ('v_seam', rev(vR))]
    sideb = [('top', line(O, A)), ('armhole', low_b), ('side_seam', side_b), ('hem', rev(hem_sb)),
             ('outer_seam', rev(oR))]

    # ---- front shoulder / neck
    SNPfj = SNPf + ush_f * J['neck_widen']
    SPf = SNPfj + ush_f * Lsh
    CFnj = CFn - P(0, J['cf_drop'])
    neck_f = hermite(SNPfj, rotv(ush_f, 90), CFnj, P(1, 0), 0.75, 1.1)

    # ---- front armhole designed with bust dart CLOSED, then reopened
    Sf = side_f[0]
    Sfc = rot(Sf, BP, -dartf)
    efc = rotv(tan_start(side_f), -dartf)
    # front armhole (dart closed).  Passes the Bunka front-width point G (dart point); the tangent at
    # G and the tensions are searched: no S-curve anywhere, front width reduced by at most 0.2 cm
    # (fitting priority), then the largest minimum radius.
    def front_arm(T, ku, k, ke):
        return chain([SPf, G, Sfc], [rotv(ush_f, 90), dirdeg(T), rotv(efc, -90)], ks=[(1.0, ku), (k, ke)])
    best = None
    for T in np.arange(-90.0, -112.1, -2.0):
        for ku in (0.6, 0.8, 1.0):
            for k in (0.8, 1.0, 1.3, 1.6):
                for ke in (0.6, 0.8, 1.0):
                    c = front_arm(T, ku, k, ke)
                    up, lo = split_at_line(c, G, P(1, 0))
                    loss = up[:, 0].max() - xfw
                    key = (inflections(c) + inflections(up) + inflections(lo) > 0, loss > J['front_width_loss_max'],
                           -min_radius(c))
                    if best is None or key < best[0]:
                        best = (key, T, ku, k, ke)
    arm_fc = front_arm(*best[1:])
    up_f, lowc_f = split_at_line(arm_fc, G, P(1, 0))
    up_f[-1] = G
    lowc_f[0] = G
    Gj, G2j = G, G2
    K.update(front_arm_T=best[1], front_arm_k=best[2:], yoke_arm_k=(ky0, ky1), Gj=Gj, G2j=G2j)
    low_f = rot(lowc_f, BP, dartf)       # starts at G2j, ends at Sf

    # ---- princess seam
    pr = ProfileSeam(P(BP[0], 0), P(BP[0], -1),
                     [(0, sP), (-6, sP * 0.35), (H['princess'], -fl)], flat_top=True)
    pc_lo, ps_lo = pr.edge(+1), pr.edge(-1)
    Wc, Ws = pc_lo[0], ps_lo[0]
    # Princess: two Hermite segments each edge (top->BP, BP->waist).  Start tangents are the dart
    # legs rotated by the same beta, so with the dart closed both edges leave the armhole along one
    # line (armhole stays smooth).  Side-panel tensions are solved so that BOTH segments match the
    # centre-panel segments exactly (BP notch and waist notch coincide).
    beta, a_bp, ub_k = J['pr_beta'], J['pr_bp_dir'], J['pr_k']
    pc_u1 = hermite(Gj, rotv(unit(BP - Gj), beta), BP, dirdeg(a_bp), 1.0)
    pc_u2 = hermite(BP, dirdeg(a_bp), Wc, tan_start(pc_lo), ub_k, 0.9)
    s1 = lambda k: hermite(G2j, rotv(unit(BP - G2j), beta), BP, dirdeg(a_bp), k)
    s2 = lambda k: hermite(BP, dirdeg(a_bp), Ws, tan_start(ps_lo), k, 0.9)
    k1 = brentq(lambda k: plen(s1(k)) - plen(pc_u1), 0.2, 1.5, xtol=1e-13)
    k2 = brentq(lambda k: plen(s2(k)) - plen(pc_u2), 0.05, 1.5, xtol=1e-13)
    ps_u1, ps_u2 = s1(k1), s2(k2)
    pc_up = np.vstack([pc_u1, pc_u2[1:]])
    ps_up = np.vstack([ps_u1, ps_u2[1:]])
    pc = np.vstack([pc_up, pc_lo[1:]])
    ps = np.vstack([ps_up, ps_lo[1:]])
    bp_s = BP
    K.update(k_princess=(k1, k2))

    cf_end = P(CF, H['cf'])
    hem_fc = hermite(pc[-1], rotv(tan_end(pc), 90), cf_end, P(0, 1), 1.0, 1.0)
    hem_fs = hem_between(side_f, ps)
    frontc = [('shoulder', line(SNPfj, SPf)), ('armhole', up_f), ('princess', pc), ('hem', hem_fc),
              ('cf', line(cf_end, CFnj)), ('neck', rev(neck_f))]
    fronts = [('armhole', low_f), ('side_seam', side_f), ('hem', hem_fs), ('princess', rev(ps))]

    K.update(SNPbj=SNPbj, SPy=SPy, Ay=Ay, Oy=Oy, A=A, O=O, Lsh=Lsh, SNPfj=SNPfj, SPf=SPf, CFnj=CFnj,
             Sb=Sb, Sf=Sf, Bw=Bw, Wc=Wc, Ws=Ws)
    notches = dict(
        side_waist=(side.point(-1, 0), side.point(+1, 0)),
        v_chest=(V.point(-1, yC), V.point(1, yC)), v_waist=(V.point(-1, 0), V.point(1, 0)),
        o_chest=(Os.point(-1, yC), Os.point(1, yC)), o_waist=(Os.point(-1, 0), Os.point(1, 0)),
        bp=(BP, bp_s), p_waist=(Wc, Ws), yoke_O=(Oy, O),
    )

    pieces = dict(yoke=yoke, cb_panel=cbp, mid_back=midb, side_back=sideb, front=frontc, side_front=fronts)
    curves = dict(side=side, V=V, Os=Os, pr=pr, arm_b=arm_b, up_b=up_b, low_b=low_b, arm_y=arm_y,
                  arm_fc=arm_fc, up_f=up_f, low_f=low_f, lowc_f=lowc_f, neck_b=neck_b, neck_f=neck_f,
                  pc=pc, ps=ps, pc_up=pc_up, ps_up=ps_up)

    # ============================ SLEEVE ============================
    AHb = plen(arm_y) + plen(low_b)
    AHf = plen(up_f) + plen(low_f)
    AH = AHb + AHf
    depth = ((SPb[1] - yA) + (SPf[1] - yA)) / 2
    cap_h_bunka = depth * 5 / 6                # Bunka: 5/6 of average armhole depth
    cap_h = cap_h_bunka
    L = M['SL']
    y_el = -(L / 2 + 2.5)                      # elbow line
    T = P(0, 0)

    def sleeve(dl, cap_h=cap_h):
        """dl shortens both Bunka slope lines (front AH, back AH+1) equally."""
        wf = math.sqrt((AHf - dl) ** 2 - cap_h**2)
        wb = math.sqrt((AHb + 1 - dl) ** 2 - cap_h**2)
        Bf, Bb = P(wf, -cap_h), P(-wb, -cap_h)

        def edges(sh, off=J['wrist_fwd']):
            ww = J['sleeve_wrist']
            Wf, Wb = P(off + ww / 2, -L), P(off - ww / 2, -L)
            fx = Bf[0] + (Wf[0] - Bf[0]) * (y_el - Bf[1]) / (Wf[1] - Bf[1]) - 1.0
            Ef = P(fx, y_el)
            df = unit(Wf - Bf)
            fe1 = hermite(Bf, unit(Ef - Bf), Ef, df, 1.0)
            fe2 = hermite(Ef, df, Wf, unit(Wf - Ef), 1.0)
            Ebn = P(Bb[0] + 1.0 - sh, y_el)
            E1 = Bb + plen(fe1) * unit(Ebn - Bb)
            E2 = Wb + plen(fe2) * unit(Ebn - Wb)
            return dict(Wf=Wf, Wb=Wb, Ef=Ef, fe1=fe1, fe2=fe2, E1=E1, E2=E2, off=off, sh=sh)

        def gap(o):
            E = edges(o)
            return math.copysign(np.linalg.norm(E['E1'] - E['E2']), E['E1'][1] - E['E2'][1])
        sh = root(lambda o: gap(o) - J['elbow_dart'], -2.8, 6.0, n=12)
        S = edges(sh)
        fedge = np.vstack([S['fe1'], S['fe2'][1:]])
        sf, sb = Bf - T, Bb - T
        nf = rotv(unit(sf), 90)
        nbv = rotv(unit(sb), -90)
        Uf = T + 0.25 * sf + 1.8 * nf            # Bunka guide offsets
        If = T + (0.5 * np.linalg.norm(sf) + 1.0) * unit(sf)
        Df = T + 0.75 * sf - 1.3 * nf
        Ub = T + 0.25 * sb + 1.9 * nbv
        Ib = T + 0.5 * sb
        Db = T + 0.75 * sb - 0.5 * nbv
        pts = [Bb, Ib, Ub, T, Uf, If, Df, Bf]
        # C2 splines: back half / front half, horizontal at the cap top,
        # perpendicular to the underarm seams at both ends
        cb_ = spline_through([Bb, Ib, Ub, T], rotv(unit(S['E1'] - Bb), 90), P(1, 0))
        cf_ = spline_through([T, Uf, If, Df, Bf], P(1, 0), rotv(tan_start(fedge), 90))
        cap = np.vstack([cb_, cf_[1:]])
        S.update(Bf=Bf, Bb=Bb, wf=wf, wb=wb, fedge=fedge, cap=cap, cap_pts=pts, dl=dl)
        return S

    # Jacket sleeve: cap height lowered from the Bunka value until the biceps width gives the
    # target wearing ease over the upper arm; for every height the slopes are re-solved so the
    # cap ease stays exactly at J['cap_ease'].
    def solve_dl(h):
        return root(lambda x: plen(sleeve(x, h)['cap']) - AH - J['cap_ease'], -3.0, 4.0, n=7)

    def biceps(h):
        s = sleeve(solve_dl(h), h)
        return s['Bf'][0] - s['Bb'][0]
    target = M['upper_arm'] + J['biceps_ease']
    cap_h = root(lambda h: biceps(h) - target, cap_h_bunka * 0.7, cap_h_bunka, n=6, xtol=1e-9)
    dl = solve_dl(cap_h)
    S = sleeve(dl, cap_h)
    Bf, Bb, Wf, Wb, E1, E2, Ef = S['Bf'], S['Bb'], S['Wf'], S['Wb'], S['E1'], S['E2'], S['Ef']
    cap, fedge, fe1, fe2 = S['cap'], S['fedge'], S['fe1'], S['fe2']
    mid = (E1 + E2) / 2
    nin = unit(rotv(E1 - E2, -90))
    if nin[0] < 0:
        nin = -nin
    apex = mid + 10.0 * nin
    be1, be2 = line(Bb, E1), line(E2, Wb)
    iT = int(np.argmin(np.linalg.norm(cap - T, axis=1)))
    cap_b, cap_f = cap[: iT + 1], cap[iT:]
    wrist = hermite(Wb, rotv(unit(Wb - E2), 90), Wf, rotv(tan_end(fedge), 90), 1.0)
    sleeve_p = [('cap_back', cap_b), ('cap_front', cap_f), ('underarm_front', fedge), ('wrist', rev(wrist)),
                ('underarm_back_lo', rev(be2)), ('elbow_dart', np.vstack([line(E2, apex), line(apex, E1)[1:]])),
                ('underarm_back_up', rev(be1))]
    e = plen(cap) - AH
    capL = plen(cap)
    sh_notch, _ = point_at_len(cap, AHb * (1 + e / AH))
    yoke_notch, _ = point_at_len(cap, plen(low_b) * (1 + e / AH))
    pr_notch, _ = point_at_len(rev(cap), plen(low_f) * (1 + e / AH))
    K.update(AHb=AHb, AHf=AHf, AH=AH, cap_h=cap_h, cap_h_bunka=cap_h_bunka, depth=depth, wf=S['wf'], wb=S['wb'], dl=dl, off=S['off'], elbow_sh=S['sh'],
             cap_len=capL, ease=e, Bf=Bf, Bb=Bb, Wf=Wf, Wb=Wb, E1=E1, E2=E2, apex=apex, Ef=Ef, y_el=y_el,
             sh_notch=sh_notch, sh_offset=float(sh_notch[0]), cap_pts=S['cap_pts'])
    notches.update(sleeve_sh=(sh_notch,), sleeve_yoke=(yoke_notch,), sleeve_pr=(pr_notch,), sleeve_elbow=(Ef,))
    curves.update(cap=cap, cap_b=cap_b, cap_f=cap_f, fedge=fedge, be1=be1, be2=be2, wrist=wrist, fe1=fe1, fe2=fe2)
    pieces['sleeve'] = sleeve_p

    # ============================ COLLAR (stand) ============================
    Ln = plen(neck_b) + plen(neck_f)
    rise, ch = J['collar_rise'], J['collar_h']
    end_dir = dirdeg(14)

    def col_neck(x):
        return hermite(P(0, 0), P(1, 0), P(x, rise), end_dir, 1.0)

    xe = brentq(lambda x: plen(col_neck(x)) - Ln, Ln * 0.8, Ln * 1.05, xtol=1e-12)
    cn = col_neck(xe)
    Ef_c = cn[-1] + rotv(end_dir, 90) * ch
    ctop = hermite(P(0, ch), P(1, 0), Ef_c, end_dir, 1.0)
    collar = [('fold_cb', line(P(0, ch), P(0, 0))), ('neck_edge', cn), ('cf_edge', line(cn[-1], Ef_c)),
              ('top_edge', rev(ctop))]
    snp_notch, _ = point_at_len(cn, plen(neck_b))
    notches["collar_snp"] = (snp_notch,)
    pieces['collar'] = collar

    # ---- back neck facing: 6 cm at CB, 5 cm along the shoulder, lower edge perpendicular to
    #      CB (fold) and to the shoulder seam
    fw_cb, fw_sh = 6.0, 5.0
    shp = SNPbj + ush_y * fw_sh
    bf_low = hermite(P(0, CBn[1] - fw_cb), P(1, 0), shp, rotv(ush_y, 90), 1.0, 0.8)
    back_facing = [('fold_cb', line(P(0, CBn[1] - fw_cb), CBn)), ('neck', neck_b), ('shoulder', line(SNPbj, shp)),
                   ('lower', rev(bf_low))]
    pieces['back_facing'] = back_facing
    curves['bf_low'] = bf_low
    curves.update(collar_neck=cn, collar_top=ctop)
    K.update(neck_len=Ln, collar_x=xe)

    # ============================ CUFF TUBE (straight cylinder) ============================
    # Unrolled cylinder = rectangle: width = sleeve-hem length, height = cuff_h.
    # Joined into a tube with one seam that lines up with the sleeve underarm seam.
    Lw = plen(wrist)
    ch_ = J['cuff_h']
    cuff = [('top', line(P(0, ch_), P(Lw, ch_))), ('side2', line(P(Lw, ch_), P(Lw, 0))),
            ('bottom', line(P(Lw, 0), P(0, 0))), ('side1', line(P(0, 0), P(0, ch_)))]
    pieces['cuff'] = cuff
    K.update(wrist_len=Lw)

    # ============================ TAB / BELT ============================
    tw, tl, tp = 2.4, 7.0, 1.1
    tab = np.array([[-tl / 2, 0], [-tl / 2 + tp, tw / 2], [tl / 2 - tp, tw / 2], [tl / 2, 0],
                    [tl / 2 - tp, -tw / 2], [-tl / 2 + tp, -tw / 2], [-tl / 2, 0]])
    pieces['tab'] = [('edge', tab)]
    belt_len = 2 * waist_half + 1.0 + 18.0
    bw = 3.5
    pieces['belt'] = [('edge', np.array([[0, 0], [belt_len - 2, 0], [belt_len, bw / 2], [belt_len - 2, bw],
                                          [0, bw], [0, 0]]))]
    K.update(belt_len=belt_len)
    tab_y = np.linspace(CFnj[1] - 2.5, 6.0, 5)
    K['tab_y'] = tab_y

    return dict(K=K, pieces=pieces, curves=curves, notches=notches)


def outline(piece):
    return np.vstack([seg if i == 0 else seg[1:] for i, (_, seg) in enumerate(piece)])


def build_block():
    """Plain Bunka bodice block + sleeve block (for the 원형 sheet)."""
    R = build()
    K = R['K']
    B = M['B']
    yC, xs, BP, G, G2, dartf = K['yC'], K['xs'], K['BP'], K['G'], K['G2'], K['dartf']
    S = P(xs, yC)
    arm_b = chain([K['SPb'], P(K['xbw'], K['yE'] + 0.8), S], [rotv(K['ush_b'], -90), P(0, -1), P(1, 0)],
                  ks=[(1.0, 0.9), (0.9, 1.25)])
    Sc = rot(S, BP, -dartf)
    arm_fc = chain([K['SPf_blk'], G, Sc], [rotv(K['ush_f'], 90), dirdeg(-104), rotv(P(-1, 0), -dartf)])
    up_f, lowc = split_at_line(arm_fc, G, P(1, 0))
    low_f = rot(lowc, BP, dartf)
    neck_b = hermite(K['CBn'], P(1, 0), K['SNPb'], rotv(K['ush_b'], 90), 1.1, 0.45)
    neck_f = hermite(K['SNPf'], rotv(K['ush_f'], 90), K['CFn'], P(1, 0), 0.75, 1.1)
    blk = dict(arm_b=arm_b, up_f=up_f, low_f=low_f, neck_b=neck_b, neck_f=neck_f, S=S)
    # sleeve block (Bunka): cap height 5/6 avg armhole depth, slopes front AH / back AH+1
    AHb, AHf = plen(arm_b), plen(up_f) + plen(low_f)
    depth = ((K['SPb'][1] - yC) + (K['SPf_blk'][1] - yC)) / 2
    h = depth * 5 / 6
    wf, wb = math.sqrt(AHf**2 - h**2), math.sqrt((AHb + 1) ** 2 - h**2)
    T, Bf, Bb = P(0, 0), P(wf, -h), P(-wb, -h)
    sf, sb = Bf - T, Bb - T
    nf, nbv = rotv(unit(sf), 90), rotv(unit(sb), -90)
    capb = spline_through([Bb, T + 0.5 * sb, T + 0.25 * sb + 1.9 * nbv, T], P(1, 0), P(1, 0))
    capf = spline_through([T, T + 0.25 * sf + 1.8 * nf, T + (0.5 * np.linalg.norm(sf) + 1) * unit(sf),
                           T + 0.75 * sf - 1.3 * nf, Bf], P(1, 0), P(1, 0))
    L = M['SL']
    blk.update(AHb=AHb, AHf=AHf, cap_h=h, depth=depth, Bf=Bf, Bb=Bb, capb=capb, capf=capf, SL=L,
               y_el=-(L / 2 + 2.5), cap_ease=plen(capb) + plen(capf) - AHb - AHf)
    return R, blk
