"""
Figura 2.15 — Protocolo de espalhamento aplicado à queima (espalhamento_queima.pdf).

Moore, no limiar de percolação, 5336 realizações em redes 801x801, t <= 200. Imprime os
ajustes em 20 <= t <= 200 citados na legenda da figura e no Apêndice A.
Dados em cache (dados_espalhamento_queima.json); recalcular leva alguns minutos.
"""
import os

import numpy as np
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

import estilo as E

E.aplicar()

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
PARAM = dict(L=801, tmax=200, n=5336, semente=23, pc=0.592746)
DMIN = 1.1307
ETA = (91 / 48) / DMIN - 1 - (5 / 48) / DMIN          # d_ell - 1 - beta/(nu d_min)
DELTA = (5 / 48) / DMIN                                 # beta/(nu d_min)
DOIS_Z = 2 / DMIN


def dados():
    arq = os.path.join(E.AQUI, 'dados_espalhamento_queima.json')
    d = E.cache_valido(arq, PARAM)
    if d is not None:
        return np.array(d['viva']), np.array(d['nsh']), np.array(d['r2s'])
    print("  calculando a queima (alguns minutos)...")
    L, T, c = PARAM['L'], PARAM['tmax'], PARAM['L'] // 2
    rng = np.random.default_rng(PARAM['semente'])
    rr, cc = np.indices((L, L)); R2 = (rr - c) ** 2.0 + (cc - c) ** 2.0
    viva, nsh, r2s = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1)
    for _ in range(PARAM['n']):
        occ = rng.random((L, L)) >= PARAM['pc']; occ[c, c] = True
        lab, _ = label(occ, structure=np.ones((3, 3)))
        cl = lab == lab[c, c]
        ys, xs = np.where(cl)
        if len(ys) == 1:
            nsh[0] += 1; viva[0] += 1
            continue
        idx = -np.ones((L, L), int); idx[ys, xs] = np.arange(len(ys))
        lin, col = [], []
        for dr, dc in MOORE:
            y2, x2 = ys + dr, xs + dc
            ok = (y2 >= 0) & (y2 < L) & (x2 >= 0) & (x2 < L)
            ok[ok] = cl[y2[ok], x2[ok]]
            lin.append(idx[ys[ok], xs[ok]]); col.append(idx[y2[ok], x2[ok]])
        A = coo_matrix((np.ones(sum(map(len, lin))), (np.concatenate(lin), np.concatenate(col))),
                       shape=(len(ys),) * 2).tocsr()
        dist = shortest_path(A, unweighted=True, indices=idx[c, c], directed=False)
        dd = dist[np.isfinite(dist)].astype(int); r2 = R2[ys, xs][np.isfinite(dist)]
        m = dd <= T
        np.add.at(nsh, dd[m], 1.0); np.add.at(r2s, dd[m], r2[m])
        viva[:min(int(dd.max()), T) + 1] += 1
    E.gravar_cache(arq, PARAM, dict(viva=viva.tolist(), nsh=nsh.tolist(), r2s=r2s.tolist(),
                                    n_realizacoes=PARAM['n']))
    return viva, nsh, r2s


viva, nsh, r2s = dados()
n = PARAM['n']
t = np.arange(len(viva))
N = nsh / n                                   # sobre todas as realizações
Ps = viva / n
R2 = np.where(nsh > 0, r2s / np.maximum(nsh, 1), np.nan)   # só as sobreviventes contribuem

m = (t >= 20) & (t <= 200)
aj = lambda y: np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0]
print(f"  ajustes em 20 <= t <= 200:  eta = {aj(N):.3f} (teoria {ETA:.3f})   "
      f"delta = {-aj(Ps):.3f} (teoria {DELTA:.3f})   R^2 ~ t^{aj(R2):.3f} (teoria {DOIS_Z:.3f})")

fig, axs = E.figura(1, 3, altura=5.2)
paineis = [
    (axs[0], N, r'$\langle N(t)\rangle$', ETA, r'(a) Massa queimada por camada'),
    (axs[1], Ps, r'$P_s(t)$', -DELTA, r'(b) Sobrevivência do fogo'),
    (axs[2], R2, r'$R^2(t)$', DOIS_Z, r'(c) Alcance quadrático médio'),
]
tt = np.array([20, PARAM['tmax']])
i0 = np.argmin(abs(t - 60))
for ax, y, rot, ex, tit in paineis:
    ax.loglog(t[1:], y[1:], 'o', color=E.AZUL, ms=2.2, mew=0, alpha=0.8)
    ax.loglog(tt, y[i0] * (tt / t[i0]) ** ex, color=E.VERMELHO, ls='--', lw=1.1,
              label=rf'$\propto t^{{{E.num(round(ex, 3))}}}$ (teórico)')
    ax.set_xlabel(r'Tempo $t$')
    ax.set_ylabel(rot)
    ax.set_title(tit)
    ax.grid(ls=':', which='major')
    ax.legend(loc='upper left' if ex > 0 else 'lower left')
E.log_decimal(axs[1], 'y')
import matplotlib.ticker as mt
axs[0].yaxis.set_major_locator(mt.FixedLocator([3, 5, 10, 20, 40]))
axs[0].yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: f'${v:g}$'))
axs[0].yaxis.set_minor_formatter(mt.NullFormatter())

E.salvar(fig, 'espalhamento_queima')
