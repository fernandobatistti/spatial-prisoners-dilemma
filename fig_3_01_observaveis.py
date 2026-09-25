"""
Figura 3.1 — Observáveis da colônia (observaveis_colonia.pdf).

Substrato de Moore com p = 0,407 (o mesmo conjunto de parâmetros do diagnóstico do Apêndice D),
L = 101, b = 6/5+ (b = 1,2 com eps = 0,01), mu = 1e-4, 400 passos.
(a) mapa estático de distância química a partir da colônia inicial;
(b) estado em t = 400: colônia por contato, cooperadores fora dela, desertores e buracos.

Imprime os cinco números citados na legenda. Usa nucleo_numpy (sem Numba).
Dados em cache (dados_observaveis.npz).
"""
import json
import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Patch

import estilo as E

E.aplicar()

PARAM = dict(L=101, p=0.407254, b=1.2, eps=0.01, mu=1e-4, tfim=400, lado=9, semente=12)
L = PARAM['L']
c = L // 2
MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bfs(ativo, colonia):
    """Distância química e coordenadas desembrulhadas a partir da colônia inicial."""
    ell = np.full((L, L), -1, int)
    ux, uy = np.zeros((L, L), int), np.zeros((L, L), int)
    q = deque()
    for r, cc in zip(*np.where(colonia)):
        ell[r, cc] = 0; ux[r, cc] = cc - c; uy[r, cc] = r - c
        q.append((r, cc))
    while q:
        r, cc = q.popleft()
        for dr, dc in MOORE:
            r2, c2 = (r + dr) % L, (cc + dc) % L
            if ativo[r2, c2] and ell[r2, c2] < 0:
                ell[r2, c2] = ell[r, cc] + 1
                ux[r2, c2] = ux[r, cc] + dc; uy[r2, c2] = uy[r, cc] + dr
                q.append((r2, c2))
    return ell, ux, uy


def simular():
    from nucleo_numpy import passo_racional, TorusLabeler
    rng = np.random.default_rng(PARAM['semente'])
    occ = np.ones(L * L, bool)
    occ[rng.choice(L * L, int(round(L * L * PARAM['p'])), replace=False)] = False
    ativo = occ.reshape(L, L)
    grid = np.where(ativo, 1, 2).astype(np.int32)                 # 0 = C, 1 = D, 2 = buraco
    h = PARAM['lado'] // 2
    bloco = np.zeros((L, L), bool); bloco[c - h:c + h + 1, c - h:c + h + 1] = True
    grid[bloco & ativo] = 0
    colonia = grid == 0
    lab = TorusLabeler(L)
    for _ in range(PARAM['tfim']):
        prop, *_ = passo_racional(grid, 1.0, 0.0, PARAM['b'], PARAM['eps'])
        # duas chamadas ao gerador, como no código original: mantém a sequência aleatória
        m = (((grid == 0) & (rng.random((L, L)) < PARAM['mu']))
             | ((grid == 1) & (rng.random((L, L)) < PARAM['mu'])))
        grid = np.where(m, (prop + 1) % 2, prop)
        grid[~ativo] = 2
        tl, _, _ = lab(grid == 0)
        dil = colonia.copy()
        for dr, dc in MOORE:
            dil |= np.roll(np.roll(colonia, dr, 0), dc, 1)
        colonia = np.isin(tl, np.unique(tl[dil & (tl > 0)])) & (tl > 0)
    return ativo, grid, colonia, bloco & ativo


def dados():
    arq = os.path.join(E.AQUI, 'dados_observaveis.npz')
    if os.path.exists(arq):
        d = np.load(arq)
        if '_parametros' in d and json.loads(str(d['_parametros'])) == PARAM:
            return d['ativo'], d['grid'], d['colonia'], d['inicial']
        print("  [cache] dados_observaveis.npz tem outros parâmetros: recalculando")
    print("  simulando 400 passos...")
    ativo, grid, colonia, inicial = simular()
    np.savez_compressed(arq, ativo=ativo, grid=grid, colonia=colonia, inicial=inicial,
                        _parametros=json.dumps(PARAM))
    return ativo, grid, colonia, inicial


ativo, grid, colonia, inicial = dados()
ell, ux, uy = bfs(ativo, inicial)

sel = colonia & (ell >= 0)
X, Y = ux[sel] * 1.0, uy[sel] * 1.0
fora = (grid == 0) & ~colonia
print(f"  N_col = {sel.sum()}   R2 = {np.mean(X**2 + Y**2):.0f}   Rg2 = {X.var() + Y.var():.0f}   "
      f"ell_max = {ell[sel].max()}   alcance = {max(abs(X).max(), abs(Y).max()):.0f}   "
      f"cooperadores fora da colônia = {fora.sum()}   sítios ativos inalcançáveis = "
      f"{(ativo & (ell < 0)).sum()}")

fig, axs = E.figura(1, 2, altura=8.3)

# (a) mapa estático: buracos em preto, fora do aglomerado da semente em cinza
img = np.where(ativo & (ell >= 0), ell, np.nan).astype(float)
cmap = plt.get_cmap('viridis').copy()
cmap.set_bad(E.PRETO)
im = axs[0].imshow(img, cmap=cmap, vmin=0, interpolation='none')
inalc = np.ma.masked_where(~(ativo & (ell < 0)), np.ones((L, L)))
axs[0].imshow(inalc, cmap=mc.ListedColormap([E.CINZA_CLARO]), interpolation='none')
axs[0].set_title(r'(a) Mapa estático do substrato ($\ell$)')
cb = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.02)
cb.set_label(r'Distância química $\ell$')

# (b) estado em t = tfim
estado = np.where(grid == 0, np.where(colonia, 0, 3), grid)       # 0 col, 1 D, 2 buraco, 3 C fora
cores = [E.COOPERADOR, E.DESERTOR, E.BURACO, E.AZUL_PALIDO]
axs[1].imshow(estado, cmap=mc.ListedColormap(cores), vmin=0, vmax=3, interpolation='none')
axs[1].set_title(rf"(b) Estado da simulação em $t={PARAM['tfim']}$")
axs[1].legend(handles=[Patch(facecolor=cores[0], edgecolor='k', lw=0.4, label='Colônia'),
                       Patch(facecolor=cores[3], edgecolor='k', lw=0.4, label='Cooperador fora da colônia'),
                       Patch(facecolor=cores[1], edgecolor='k', lw=0.4, label='Desertor'),
                       Patch(facecolor=cores[2], edgecolor='k', lw=0.4, label='Buraco')],
              loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2)

for ax in axs:
    ax.plot(c, c, '*', color='white', ms=7, mec=E.PRETO, mew=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
E.salvar(fig, 'observaveis_colonia')
