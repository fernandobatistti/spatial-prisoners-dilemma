"""Núcleo NumPy do modelo (imitar o melhor, síncrono, vizinhança de Moore, buracos = 2).

Equivalente, passo a passo, ao engine.py original e ao engine_v2.py com K_FERMI = 0
(verificado bit a bit). Usado pelos scripts de figuras por não depender do Numba.
Convenção: 0 = cooperador, 1 = desertor, 2 = buraco."""
import numpy as np
from scipy.ndimage import label as nd_label
from collections import deque

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
S8 = np.ones((3, 3), dtype=int)


def roll(a, dr, dc):
    # valor do vizinho em (r+dr, c+dc) -> np.roll com sinal invertido
    return np.roll(np.roll(a, -dr, axis=0), -dc, axis=1)


def passo_racional(grid, R, S, T, P):
    L = grid.shape[0]
    isC = grid == 0
    isD = grid == 1
    nc = np.zeros((L, L), np.int32)
    nd = np.zeros((L, L), np.int32)
    for dr, dc in MOORE:
        nc += roll(isC, dr, dc)
        nd += roll(isD, dr, dc)
    pay = np.zeros((L, L))
    pay[isC] = nc[isC] * R + nd[isC] * S
    pay[isD] = nc[isD] * T + nd[isD] * P
    # imitar o melhor com desempate por inércia (mesma semântica do engine)
    M = np.full((L, L), -np.inf)
    for dr, dc in MOORE:
        vs = roll(grid, dr, dc)
        vp = roll(pay, dr, dc)
        M = np.where(vs != 2, np.maximum(M, vp), M)
    mine_at_max = np.zeros((L, L), bool)
    for dr, dc in MOORE:
        vs = roll(grid, dr, dc)
        vp = roll(pay, dr, dc)
        mine_at_max |= (vs != 2) & (vp == M) & (vs == grid)
    troca = (M > pay) & (~mine_at_max) & (grid != 2)
    prop = grid.copy()
    prop[troca] = 1 - grid[troca]
    return prop, nc, nd, pay


class TorusLabeler:
    """Rotulagem de componentes (Moore) no toro, com deslocamentos de imagem
    (union-find com offsets) -> coordenadas desembrulhadas e flags de wrap."""

    def __init__(self, L):
        self.L = L
        a_idx, b_idx, sx, sy = [], [], [], []
        for r in range(L):
            for c in range(L):
                for dr, dc in MOORE:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < L and 0 <= cc < L:
                        continue
                    a_idx.append(r * L + c)
                    b_idx.append((rr % L) * L + (cc % L))
                    sx.append(1 if cc >= L else (-1 if cc < 0 else 0))
                    sy.append(1 if rr >= L else (-1 if rr < 0 else 0))
        self.a = np.array(a_idx); self.b = np.array(b_idx)
        self.sx = np.array(sx); self.sy = np.array(sy)

    def __call__(self, isC):
        L = self.L
        lab, n = nd_label(isC, structure=S8)
        if n == 0:
            return lab, 0, None
        flat = lab.ravel()
        m = (flat[self.a] > 0) & (flat[self.b] > 0)
        la, lb = flat[self.a[m]], flat[self.b[m]]
        sxs, sys_ = self.sx[m], self.sy[m]
        parent = list(range(n + 1))
        ox = [0] * (n + 1); oy = [0] * (n + 1)
        wx = {}; wy = {}

        def find(x):
            # retorna raiz e offset de x relativo à raiz
            dx = dy = 0
            path = []
            while parent[x] != x:
                path.append(x)
                dx += ox[x]; dy += oy[x]
                x = parent[x]
            return x, dx, dy

        for A, B, s_x, s_y in zip(la, lb, sxs, sys_):
            ra, oax, oay = find(A)
            rb, obx, oby = find(B)
            if ra == rb:
                if obx != oax + s_x: wx[ra] = True
                if oby != oay + s_y: wy[ra] = True
            else:
                parent[rb] = ra
                ox[rb] = oax + s_x - obx
                oy[rb] = oay + s_y - oby
                if wx.get(rb): wx[ra] = True
                if wy.get(rb): wy[ra] = True
        roots = np.zeros(n + 1, int); offx = np.zeros(n + 1, int); offy = np.zeros(n + 1, int)
        for k in range(1, n + 1):
            r, dx, dy = find(k)
            roots[k] = r; offx[k] = dx; offy[k] = dy
        # renumera clusters toroidais pela ordem do menor rótulo planar (= ordem do BFS original)
        uniq = []
        seen = {}
        for k in range(1, n + 1):
            r = roots[k]
            if r not in seen:
                seen[r] = len(uniq) + 1; uniq.append(r)
        newlab = np.zeros(n + 1, int)
        for k in range(1, n + 1):
            newlab[k] = seen[roots[k]]
        tl = newlab[lab]
        info = dict(ox=offx[lab], oy=offy[lab],
                    wrapx=np.array([False] + [bool(wx.get(r, False)) for r in uniq]),
                    wrapy=np.array([False] + [bool(wy.get(r, False)) for r in uniq]))
        return tl, len(uniq), info


def cinematica_original(tl, ncl, info, L, cx_prev=np.nan, cy_prev=np.nan):
    """Cópia fiel de calcular_cinematica_completa (toroide) usando os rótulos rápidos."""
    if ncl == 0:
        return (np.nan,) * 4 + (0, False, False, -1)
    tam = np.bincount(tl.ravel(), minlength=ncl + 1); tam[0] = 0
    val = np.where(tam >= 4)[0]
    if len(val) == 0:
        return (np.nan,) * 4 + (0, False, False, -1)
    melhor = val[np.argmax(tam[val])]
    p_x = info['wrapx'][melhor]; p_y = info['wrapy'][melhor]
    ys, xs = np.where(tl > 0)
    rp = tl[ys, xs]
    o = np.argsort(rp)
    ys, xs, rp = ys[o], xs[o], rp[o]
    lim = np.searchsorted(rp, val)
    if not np.isnan(cx_prev):
        best = np.inf
        for k, rot in enumerate(val):
            px = xs[lim[k]:lim[k] + tam[rot]]; py = ys[lim[k]:lim[k] + tam[rot]]
            tx = 2 * np.pi * px / L; ty = 2 * np.pi * py / L
            cmx = (np.arctan2(np.mean(np.sin(tx)), np.mean(np.cos(tx))) * L / (2 * np.pi)) % L
            cmy = (np.arctan2(np.mean(np.sin(ty)), np.mean(np.cos(ty))) * L / (2 * np.pi)) % L
            dx = min(abs(cmx - cx_prev), L - abs(cmx - cx_prev))
            dy = min(abs(cmy - cy_prev), L - abs(cmy - cy_prev))
            sc = (dx ** 2 + dy ** 2 + 1.0) / tam[rot] ** 2
            if sc < best:
                best = sc; melhor = rot
    k = np.where(val == melhor)[0][0]
    px = xs[lim[k]:lim[k] + tam[melhor]]; py = ys[lim[k]:lim[k] + tam[melhor]]
    tx = 2 * np.pi * px / L; ty = 2 * np.pi * py / L
    cmx = (np.arctan2(np.mean(np.sin(tx)), np.mean(np.cos(tx))) * L / (2 * np.pi)) % L
    cmy = (np.arctan2(np.mean(np.sin(ty)), np.mean(np.cos(ty))) * L / (2 * np.pi)) % L
    dx = px - cmx; dy = py - cmy
    dx = dx - L * np.round(dx / L); dy = dy - L * np.round(dy / L)
    rg2 = np.mean(dx ** 2) + np.mean(dy ** 2)
    return cmx, cmy, rg2, np.nan, tam[melhor], p_x, p_y, melhor


def rg2_desembrulhado(tl, info, rot, L):
    ys, xs = np.where(tl == rot)
    X = xs + L * info['ox'][ys, xs]
    Y = ys + L * info['oy'][ys, xs]
    return np.var(X) + np.var(Y)


def bfs_quimica(active, r0, c0):
    """Distância química (Moore) a partir de (r0,c0) no substrato + coords desembrulhadas."""
    L = active.shape[0]
    ell = np.full((L, L), -1, int)
    ux = np.zeros((L, L), int); uy = np.zeros((L, L), int)
    q = deque([(r0, c0)]); ell[r0, c0] = 0
    ux[r0, c0] = 0; uy[r0, c0] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in MOORE:
            rr, cc = (r + dr) % L, (c + dc) % L
            if active[rr, cc] and ell[rr, cc] < 0:
                ell[rr, cc] = ell[r, c] + 1
                ux[rr, cc] = ux[r, c] + dc; uy[rr, cc] = uy[r, c] + dr
                q.append((rr, cc))
    return ell, ux, uy
