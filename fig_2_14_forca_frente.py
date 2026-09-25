"""
Figura 2.14 — Resposta de uma frente reta a um único passo (forca_local_frente.pdf).

Cooperadores à esquerda, desertores à direita, substrato de Moore com fração de buracos p,
rede 200x200, seis realizações por ponto. (a) fração dos desertores da fronteira que passam
a cooperar; (b) fração dos cooperadores da fronteira que passam a desertar.
Usa nucleo_numpy (sem Numba). Dados em cache (dados_forca.json).
"""
import os

import numpy as np

import estilo as E

E.aplicar()

PARAM = dict(L=200, n=6, semente=5, eps=1e-4, p=[0.0, 0.6, 25],
             b=[1.15, 1.30, 1.45, 1.55, 1.65])
PC = 0.592746


def um_passo(p, b, rng, passo_racional):
    L, c = PARAM['L'], PARAM['L'] // 2
    av, rec = [], []
    for _ in range(PARAM['n']):
        g = np.ones((L, L), np.int32)            # 1 = D
        g[:, :c] = 0                              # 0 = C à esquerda: frente vertical em c
        g[:, :2] = 1                              # fecha o toro com desertores, longe da frente
        g[rng.random((L, L)) < p] = 2             # 2 = buraco
        g2, *_ = passo_racional(g, 1.0, 0.0, b, PARAM['eps'])
        fD, fC = g[:, c] == 1, g[:, c - 1] == 0
        av.append((g2[:, c][fD] == 0).mean())
        rec.append((g2[:, c - 1][fC] == 1).mean())
    return float(np.mean(av)), float(np.mean(rec))


def dados():
    arq = os.path.join(E.AQUI, 'dados_forca.json')
    d = E.cache_valido(arq, PARAM)
    if d is not None:
        return d
    from nucleo_numpy import passo_racional
    rng = np.random.default_rng(PARAM['semente'])
    ps = np.linspace(*PARAM['p'])
    print("  calculando a resposta da frente (alguns minutos)...")
    d = {str(b): [um_passo(p, b, rng, passo_racional) for p in ps] for b in PARAM['b']}
    E.gravar_cache(arq, PARAM, d)
    return d


d = dados()
ps = np.linspace(*PARAM['p'])
cores = E.serie(len(PARAM['b']))
marcas = ['o', 's', 'D', '^', 'v']

fig, axs = E.figura(1, 2, altura=6.4)
for b, cor, mk in zip(PARAM['b'], cores, marcas):
    v = np.array(d[str(b)])
    for k in (0, 1):
        axs[k].plot(ps, v[:, k], marker=mk, color=cor, ms=2.8, label=rf'$b={E.num(b, 2)}$')
for ax, tit in zip(axs, (r'(a) Desertores da fronteira que passam a cooperar',
                         r'(b) Cooperadores da fronteira que passam a desertar')):
    ax.axvline(PC, color=E.REFERENCIA, ls='--', lw=0.8)
    ax.text(PC - 0.008, 0.98, rf'$p_c={E.num(PC)}$', rotation=90, va='top', ha='right',
            fontsize=E.TAM_ANOTACAO)
    ax.set_xlabel(r'Fração de buracos $p$')
    ax.set_ylabel(r'Fração da fronteira, num passo')
    ax.set_title(tit)
    ax.set_xlim(-0.02, 0.62); ax.set_ylim(-0.02, 1.02)
    ax.grid(ls=':')
    E.virgula(ax)
axs[0].legend(loc='lower left')
E.salvar(fig, 'forca_local_frente')
