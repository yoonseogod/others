"""문화식 신원형 (Bunka bodice block + sleeve block) as REAL-SIZE pattern pieces, size 55.

- back block : CB (f-amount at waist), back neck, shoulder with shoulder dart, armhole,
               side seam (c/2), waist darts e, d
- front block: CF, front neck, shoulder, armhole with bust (armhole) dart, side seam (c/2),
               waist darts b, a
- sleeve block: Bunka cap (front AH / back AH+1, height = 5/6 average armhole depth), straight sleeve
The waist line is TRUED: it is drawn as one smooth curve with every dart closed, then cut and
re-opened, so with the darts sewn the waist has no kink and every dart's two legs are equal.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from draft import M, PCT, build, outline  # noqa: E402
from geom import (P, unit, dirdeg, rot, rotv, plen, tan_start, tan_end, line, hermite, chain,  # noqa: E402
                  split_at_line, spline_through, point_at_len, rev, inflections, min_radius, max_turn)

R0 = build()
K = R0['K']
B, W, BL = M['B'], M['W'], M['BL']


def affine_rot(c, deg):
    a = math.radians(deg)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return Rm, c - Rm @ c          # x' = Rm x + t


def compose(A, Bt):                # A after Bt
    return A[0] @ Bt[0], A[0] @ Bt[1] + A[1]


def apply(T, pts):
    return np.asarray(pts, float) @ T[0].T + T[1]


def inv(T):
    Ri = T[0].T
    return Ri, -Ri @ T[1]


def trued_waist(W0, d0, Wn, dn, darts):
    """darts: list of (centre_x, width, apex) ordered left->right along the waist (y=0).
    W0/Wn: left/right waist ends, d0: waist direction leaving W0, dn: arrival direction at Wn
    (both in the OPEN frame).  Returns open-frame waist segments and dart feet."""
    Ts = [(np.eye(2), np.zeros(2))]
    feet = []
    for cx, w, ap in darts:
        L, Rf = P(cx - w / 2, 0), P(cx + w / 2, 0)
        th = math.degrees(math.atan2(*(L - ap)[::-1]) - math.atan2(*(Rf - ap)[::-1]))
        apc = apply(Ts[-1], ap)
        Ts.append(compose(affine_rot(apc, th), Ts[-1]))
        feet.append((L, Rf))
    Tn = Ts[-1]
    Wnc = apply(Tn, Wn)
    dnc = Tn[0] @ dn
    closed = hermite(W0, d0, Wnc, dnc, 1.0, 1.0)
    segs, rest = [], closed
    for i, (cx, w, ap) in enumerate(darts):
        apc = apply(Ts[i], ap)                       # dart i lives in frame of region i
        legc = Ts[i][0] @ unit(feet[i][0] - ap)      # its (closed) leg direction
        a, rest = split_at_line(rest, apc, legc)
        segs.append(a)
    segs.append(rest)
    open_segs = [apply(inv(Ts[k]), s) for k, s in enumerate(segs)]
    return open_segs, closed, Ts


def build_blocks():
    yC, xs, BP, G, G2, dartf = K['yC'], K['xs'], K['BP'], K['G'], K['G2'], K['dartf']
    supp = K['half'] - (W / 2 + 3)
    amt = {k: supp * v for k, v in PCT.items()}
    D, P1, P2, SNPb, SNPf, SPf = K['D'], K['P1'], K['P2'], K['SNPb'], K['SNPf'], K['SPf_blk']
    ush_b, ush_f = K['ush_b'], K['ush_f']
    # back shoulder: P2 -> SP so that (SNP->P1) + (P2->SP) == front shoulder exactly
    SPb = P2 + ush_b * (K['tri'] - np.linalg.norm(P1 - SNPb))
    u_out = unit(SPb - P2)
    # side seams (c split front/back)
    Sb_top = Sf_top = P(xs, yC)
    Sb_bot, Sf_bot = P(xs - amt['c'] / 2, 0), P(xs + amt['c'] / 2, 0)
    db, df_ = unit(Sb_bot - Sb_top), unit(Sf_bot - Sf_top)
    # back armhole: perpendicular to outer shoulder segment and to the back side seam
    arm_b = chain([SPb, P(K['xbw'], K['yE'] + 0.8), Sb_top], [rotv(u_out, -90), P(0, -1), rotv(db, 90)],
                  ks=[(1.0, 0.9), (0.9, 1.25)])
    # front armhole designed with the bust dart closed (same method as the jacket)
    Sc = rot(Sf_top, BP, -dartf)
    efc = rotv(df_, -dartf)
    best = None
    for T in np.arange(-90.0, -112.1, -2.0):
        for ku in (0.6, 0.8, 1.0):
            for k in (0.8, 1.0, 1.3, 1.6):
                for ke in (0.6, 0.8, 1.0):
                    c = chain([SPf, G, Sc], [rotv(ush_f, 90), dirdeg(T), rotv(efc, -90)], ks=[(1.0, ku), (k, ke)])
                    up, lo = split_at_line(c, G, P(1, 0))
                    loss = up[:, 0].max() - K['xfw']
                    key = (inflections(c) + inflections(up) + inflections(lo) > 0, loss > 0.2, -min_radius(c))
                    if best is None or key < best[0]:
                        best = (key, T, ku, k, ke)
    arm_fc = chain([SPf, G, Sc], [rotv(ush_f, 90), dirdeg(best[1]), rotv(efc, -90)],
                   ks=[(1.0, best[2]), (best[3], best[4])])
    up_f, lowc = split_at_line(arm_fc, G, P(1, 0))
    up_f[-1] = G
    lowc[0] = G
    low_f = rot(lowc, BP, dartf)
    neck_b = hermite(K['CBn'], P(1, 0), SNPb, rotv(ush_b, 90), 1.1, 0.45)
    neck_f = hermite(SNPf, rotv(ush_f, 90), K['CFn'], P(1, 0), 0.75, 1.1)
    # back: CB straight to chest line, then f taken in to the waist
    CBc, CBw = P(0, yC), P(amt['f'], 0)
    dcb = unit(CBw - CBc)
    back_darts = [(D[0] + 0.5, amt['e'], P(D[0] + 0.5, K['ysb'] - 2.0)),
                  (K['xbw'] - 1.0, amt['d'], P(K['xbw'] - 1.0, yC + 2.0))]
    bsegs, bclosed, _ = trued_waist(CBw, rotv(dcb, 90), Sb_bot, rotv(db, 90), back_darts)
    xF = K['xfw'] - B / 32
    front_darts = [(xF + 1.5, amt['b'], P(xF + 1.5, yC)), (BP[0], amt['a'], BP - P(0, 2.0))]
    fsegs, fclosed, _ = trued_waist(Sf_bot, rotv(df_, 90), P(K['CF'], 0), P(1, 0), front_darts)

    def waist_with_darts(segs, darts):
        out = [('waist', segs[0])]
        for i, (cx, w, ap) in enumerate(darts):
            out.append((f'dart{i}_L', line(segs[i][-1], ap)))
            out.append((f'dart{i}_R', line(ap, segs[i + 1][0])))
            out.append(('waist', segs[i + 1]))
        return out

    back = ([('cb_upper', line(K['CBn'], CBc)), ('cb_lower', line(CBc, CBw))] + waist_with_darts(bsegs, back_darts)
            + [('side', line(Sb_bot, Sb_top)), ('armhole', rev(arm_b)), ('shoulder_out', line(SPb, P2)),
               ('sdart_out', line(P2, D)), ('sdart_in', line(D, P1)), ('shoulder_in', line(P1, SNPb)),
               ('neck', rev(neck_b))])
    front = (waist_with_darts(fsegs, front_darts)
             + [('cf', line(P(K['CF'], 0), K['CFn'])), ('neck', rev(neck_f)), ('shoulder', line(SNPf, SPf)),
                ('armhole_up', up_f), ('bdart_up', line(G, BP)), ('bdart_lo', line(BP, G2)),
                ('armhole_lo', low_f), ('side', line(Sf_top, Sf_bot))])
    # front waist runs side -> CF; reorder so the outline is closed: start at side bottom

    # ---------------- sleeve block
    AHb, AHf = plen(arm_b), plen(up_f) + plen(low_f)
    depth = ((SPb[1] - yC) + (SPf[1] - yC)) / 2
    h = depth * 5 / 6
    wf, wb = math.sqrt(AHf**2 - h**2), math.sqrt((AHb + 1) ** 2 - h**2)
    T0, Bf, Bb = P(0, 0), P(wf, -h), P(-wb, -h)
    sf, sb = Bf - T0, Bb - T0
    nf, nbv = rotv(unit(sf), 90), rotv(unit(sb), -90)
    capb = spline_through([Bb, T0 + 0.5 * sb, T0 + 0.25 * sb + 1.9 * nbv, T0], P(1, 0), P(1, 0))
    capf = spline_through([T0, T0 + 0.25 * sf + 1.8 * nf, T0 + (0.5 * np.linalg.norm(sf) + 1) * unit(sf),
                           T0 + 0.75 * sf - 1.3 * nf, Bf], P(1, 0), P(1, 0))
    L = M['SL']
    Wf, Wb = P(wf, -L), P(-wb, -L)
    sleeve = [('cap_back', capb), ('cap_front', capf), ('underarm_front', line(Bf, Wf)),
              ('wrist', line(Wf, Wb)), ('underarm_back', line(Wb, Bb))]
    cap = np.vstack([capb, capf[1:]])
    e = plen(cap) - (AHb + AHf)
    sh_notch, _ = point_at_len(cap, AHb * (1 + e / (AHb + AHf)))
    fr_notch, _ = point_at_len(rev(cap), plen(low_f) * (1 + e / (AHb + AHf)))
    bw_arm = split_at_line(rev(arm_b), P(0, K['yE'] + 0.8), P(1, 0))[0]      # side -> back-width level
    bw_notch_arm = bw_arm[-1]
    bk_notch, _ = point_at_len(cap, plen(bw_arm) * (1 + e / (AHb + AHf)))
    data = dict(SPb=SPb, arm_b=arm_b, up_f=up_f, low_f=low_f, arm_fc=arm_fc, neck_b=neck_b, neck_f=neck_f,
                back_darts=back_darts, front_darts=front_darts, bsegs=bsegs, fsegs=fsegs, bclosed=bclosed,
                fclosed=fclosed, amt=amt, AHb=AHb, AHf=AHf, cap_h=h, depth=depth, Bf=Bf, Bb=Bb, Wf=Wf, Wb=Wb,
                capb=capb, capf=capf, cap=cap, cap_ease=e, SL=L, y_el=-(L / 2 + 2.5), wf=wf, wb=wb,
                sh_notch=sh_notch, fr_notch=fr_notch, bk_notch=bk_notch, bw_notch_arm=bw_notch_arm,
                front_arm=best[1:], Sb_bot=Sb_bot, Sf_bot=Sf_bot, CBw=CBw, CBc=CBc)
    return dict(back=back, front=front, sleeve=sleeve), data


PIECES, DATA = build_blocks()
