"""
Figura 2.13 — Distância química em redes diluídas (distancia_quimica.pdf).

(a) queima a partir do centro de um aglomerado de Moore no limiar (janela 301x301 de uma
    rede 1201x1201); (b) distância química contra a distância euclidiana média da camada,
    médias sobre 40 aglomerados que alcançam ell = 400.

Imprime também a medida de d_min citada na Seção 2.5.1 e no Apêndice A (ajuste de
<r> ~ ell^(1/d_min) em 20 <= ell <= 400). Dados em cache (dados_quimica.json e
dados_quimica_img.npz); recalcular leva alguns minutos.
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

import estilo as E

E.aplicar()

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
VN = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CONFIG = [('Moore, $p=p_c$', 0.592746, MOORE),
          ('von Neumann, $p=p_c$', 0.407254, VN),
          ('Moore, $p=0{,}50$', 0.50, MOORE)]
PARAM = dict(L=1201, n=40, lmax=400, semente=3, p=[c[1] for c in CONFIG])
DMIN = 1.1307
rng = np.random.default_rng(PARAM['semente'])


def dist_quimica(occ, semente, viz):
    """Busca em largura (grafo não ponderado) a partir de um sítio, bordas abertas."""
    L = occ.shape[0]
    idx = -np.ones(occ.shape, int)
    ys, xs = np.where(occ)
    idx[ys, xs] = np.arange(len(ys))
    lin, col = [], []
    for dr, dc in viz:
        r2, c2 = ys + dr, xs + dc
        ok = (r2 >= 0) & (r2 < L) & (c2 >= 0) & (c2 < L)
        ok[ok] = occ[r2[ok], c2[ok]]
        lin.append(idx[ys[ok], xs[ok]]); col.append(idx[r2[ok], c2[ok]])
    A = coo_matrix((np.ones(sum(len(r) for r in lin)), (np.concatenate(lin), np.concatenate(col))),
                   shape=(len(ys), len(ys))).tocsr()
    d = shortest_path(A, unweighted=True, indices=idx[semente], directed=False)
    out = np.full(occ.shape, np.inf)
    out[ys, xs] = d
    return out


def medir(p, viz):
    L, n, lmax = PARAM['L'], PARAM['n'], PARAM['lmax']
    c = L // 2
    rr, cc = np.indices((L, L))
    R = np.hypot(rr - c, cc - c)
    soma, cont = np.zeros(lmax + 1), np.zeros(lmax + 1)
    feitos, ex = 0, None
    while feitos < n:
        occ = rng.random((L, L)) >= p
        occ[c, c] = True
        d = dist_quimica(occ, (c, c), viz)
        ok = np.isfinite(d)
        if d[ok].max() < lmax:                 # só aglomerados que alcançam ell = lmax
            continue
        if ex is None:
            ex = (d.copy(), occ.copy())
        dd = d[ok].astype(int); m = dd <= lmax
        np.add.at(soma, dd[m], R[ok][m]); np.add.at(cont, dd[m], 1)
        feitos += 1
    ell = np.arange(lmax + 1); v = (cont > 0) & (ell > 0)
    return ell[v], soma[v] / cont[v], ex


def dados():
    arq_j = os.path.join(E.AQUI, 'dados_quimica.json')
    arq_n = os.path.join(E.AQUI, 'dados_quimica_img.npz')
    res = E.cache_valido(arq_j, PARAM)
    if res is not None and os.path.exists(arq_n):
        img = np.load(arq_n)
        return {k: v for k, v in res.items() if k != '_parametros'}, img['d'], img['occ']
    print("  calculando distâncias químicas (alguns minutos)...")
    res, d_ex, occ_ex = {}, None, None
    for nome, p, viz in CONFIG:
        l, r, ex = medir(p, viz)
        res[nome] = {'r': r.tolist(), 'l': l.tolist()}
        if nome == CONFIG[0][0]:
            d_ex, occ_ex = ex
    E.gravar_cache(arq_j, PARAM, res)
    np.savez_compressed(arq_n, d=d_ex, occ=occ_ex)
    return res, d_ex, occ_ex


res, d_img, occ_img = dados()

# a medida de d_min citada no texto
print("  d_min medido (ajuste de <r> ~ ell^(1/d_min) em 20 <= ell <= 400):")
for nome, v in res.items():
    l, r = np.array(v['l']), np.array(v['r'])
    m = (l >= 20) & (l <= 400)
    s = np.polyfit(np.log(l[m]), np.log(r[m]), 1)[0]
    print(f"     {nome:<24} d_min = {1 / s:.3f}")
print(f"     teoria: d_min = {DMIN}")

fig, axs = E.figura(1, 2, altura=6.6)

# (a) queima a partir do centro
c, w = d_img.shape[0] // 2, 150
sd = d_img[c - w:c + w + 1, c - w:c + w + 1]
so = occ_img[c - w:c + w + 1, c - w:c + w + 1]
img = np.zeros((2 * w + 1, 2 * w + 1, 3))
img[so & ~np.isfinite(sd)] = [0.8, 0.8, 0.8]              # outros aglomerados: cinza claro
cmap = plt.get_cmap('viridis')
norm = matplotlib.colors.Normalize(vmin=0, vmax=300)
foc = np.isfinite(sd)
img[foc] = cmap(norm(sd[foc]))[:, :3]
axs[0].imshow(img, interpolation='none')
axs[0].plot(w, w, '*', color='white', ms=7, mec=E.PRETO, mew=0.6)
axs[0].set_xticks([]); axs[0].set_yticks([])
axs[0].set_title(r'(a) Queima a partir do centro (Moore, $p=p_c$)')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axs[0], fraction=0.05, pad=0.02, extend='max')
cb.set_label(r'Distância química $\ell$')

# (b) crescimento das camadas químicas
estilos = {CONFIG[0][0]: (E.VERMELHO, 'o'), CONFIG[1][0]: (E.AZUL, 's'),
           CONFIG[2][0]: (E.CINZA, 'D')}
for nome, v in res.items():
    cor, mk = estilos[nome]
    axs[1].loglog(v['r'], v['l'], mk, color=cor, ms=2.4, alpha=0.8, mew=0, label=nome)
rr = np.array([10, 300.])
axs[1].loglog(rr, 1.15 * rr ** DMIN, color=E.PRETO, lw=1.3,
              label=rf'fractal $\propto r^{{{E.num(DMIN)}}}$')
axs[1].loglog(rr, 1.05 * rr, color=E.CINZA_ESCURO, ls='--', lw=1.1, label=r'euclidiano $\propto r$')
axs[1].set_xlabel(r'Distância euclidiana média $\langle r\rangle$')
axs[1].set_title(r'(b) Crescimento das camadas químicas')
axs[1].yaxis.tick_right(); axs[1].yaxis.set_label_position('right')
axs[1].set_ylabel(r'Distância química $\ell$')
axs[1].legend(loc='upper left')
axs[1].grid(ls=':', which='major')

E.salvar(fig, 'distancia_quimica')
