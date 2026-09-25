"""
Figura 5.5 e Tabela 5.4 — a frente comparada à queima do mesmo substrato (espalhamento_1_series.pdf).

b = 1,07, mu = 0, L = 401 (60 realizações), seis valores de p. Lê
espalhamento_v1_b1.0700_moore1_n09/espalhamento_p*.json, gravados por espalhamento_frente.py,
e imprime os números da Tabela 5.4 e da Seção 5.3. Não simula nada.

(a) massa da colônia (cheio) e da queima (pontilhado); (b) alcance quadrático, até o corte de
validade geométrica; (c) razão colônia/queima, tracejada depois que a queima satura; (d) número
médio de componentes da colônia, tracejado depois do corte de validade (é uma grandeza
topológica, que o rotulador periódico mede corretamente mesmo depois que a colônia dá a volta).
"""
import glob
import json
import os
import warnings

import numpy as np

import estilo as E

E.aplicar()

PASTA = os.path.join(E.AQUI, 'espalhamento_v1_b1.0700_moore1_n09')
B, L, T_MAX, N_CHECK = 1.07, 401, 2000, 60
TS = np.array(sorted(set(np.unique(np.geomspace(1, T_MAX, N_CHECK).astype(int)).tolist())), float)


def media(ent, campo, apenas_validas=False):
    """Média sobre realizações, instante a instante, ignorando ausentes."""
    M = []
    for sr in ent['series']:
        v = [np.nan if x is None else float(x) for x in sr[campo]]
        if apenas_validas:
            v = [a if bv == 1 else np.nan for a, bv in zip(v, sr['valido_geom'])]
        M.append(v)
    M = np.array(M, float)
    with warnings.catch_warnings():                 # instantes sem nenhuma realização válida
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(M, axis=0), np.sum(np.isfinite(M), axis=0)


dados = {}
for arq in sorted(glob.glob(os.path.join(PASTA, 'espalhamento_p*.json'))):
    p = float(os.path.basename(arq)[len('espalhamento_p'):-5])
    d = json.load(open(arq))
    if f'L{L}' in d:
        dados[p] = d[f'L{L}']
ps = sorted(dados)
cores = E.serie(len(ps))

fig, axs = E.figura(2, 2, altura=11.0)
(a, b), (c, dd) = axs
print(f"  {'p':>5} {'veloc.':>7} {'razao t=100':>11} {'razao final':>11} {'comp. fim':>9} "
      f"{'comp. max':>9} {'n real.':>7}")
i100 = int(np.argmin(abs(TS - 100)))
for p, cor in zip(ps, cores):
    ent = dados[p]
    N, _ = media(ent, 'N_col')
    Nq, _ = media(ent, 'N_queima')
    R2, _ = media(ent, 'R2_origem', apenas_validas=True)
    R2q, _ = media(ent, 'R2_queima')
    em, _ = media(ent, 'ell_max')
    nc, _ = media(ent, 'n_comp_colonia')
    val, _ = media(ent, 'valido_geom')
    dentro = np.where(val >= 0.9)[0]
    jc = int(dentro[-1]) + 1 if len(dentro) else len(TS)       # corte de validade (90%)
    js = int(np.argmax(Nq >= 0.999 * Nq[-1])) + 1               # a queima saturou
    razao = N / np.maximum(Nq, 1)
    rot = rf'$p={E.num(p, 2)}$'
    a.plot(TS, N, color=cor, label=rot)
    a.plot(TS, Nq, color=cor, ls=':', lw=0.8)
    b.plot(TS[:jc], R2[:jc], color=cor)
    b.plot(TS[:min(jc, js)], R2q[:min(jc, js)], color=cor, ls=':', lw=0.8)   # a queima também dá a volta
    c.plot(TS[:js], razao[:js], color=cor)
    c.plot(TS[js - 1:], razao[js - 1:], color=cor, ls='--', lw=0.8)
    dd.plot(TS[:jc], nc[:jc], color=cor)                     # topológico: vale após o corte,
    dd.plot(TS[jc - 1:], nc[jc - 1:], color=cor, ls='--', lw=0.8)  # tracejado para distinguir
    print(f"  {p:5.2f} {em[i100] / TS[i100]:7.3f} {razao[i100]:11.3f} {razao[-1]:11.3f} "
          f"{nc[-1]:9.2f} {np.nanmax(nc):9.2f} {len(ent['series']):7d}")

a.set_yscale('log'); b.set_yscale('log')
for ax in axs.ravel():
    ax.set_xscale('log')
    ax.set_xlabel(r'Tempo $t$')
    ax.grid(ls=':', which='major')
a.set_ylabel(r'$\langle N\rangle$, $N_{\mathrm{queima}}$')
a.set_title(r'(a) Massa: frente (cheio) e queima (pontilhado)')
b.set_ylabel(r'$R^2(t)$')
b.set_title(r'(b) Alcance quadrático, até o corte de validade')
c.set_ylabel(r'$\langle N\rangle/N_{\mathrm{queima}}$')
c.set_title(r'(c) Razão frente/queima')
dd.set_ylabel(r'Componentes da colônia')
dd.set_title(r'(d) Fragmentação da colônia')
E.virgula(c, dd, eixos='y')
a.legend(loc='upper left', ncol=2)
E.salvar(fig, 'espalhamento_1_series')


# ==============================================================================
# EXPOENTE kappa de <N> ~ t^kappa (Tabela 5.4), com incerteza
# ==============================================================================
# Ajuste em 20 <= t <= 200 sobre a média das realizações. Incerteza estatística por bootstrap
# sobre as 60 realizações; incerteza de janela = maior desvio ao trocar a janela por
# 10-100, 20-100 ou 40-200. A segunda domina: kappa serve para comparar valores de p entre
# si, não como expoente assintótico (ver as ressalvas da Seção 5.3).
def kappa_bootstrap(n_bs=1000, semente=7, janela=(20, 200),
                    alternativas=((10, 100), (20, 100), (40, 200))):
    rng = np.random.default_rng(semente)

    def ajuste(M, a, b):
        j = (TS >= a) & (TS <= b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m = np.nanmean(M, axis=0)
        return np.polyfit(np.log(TS[j]), np.log(m[j]), 1)[0]

    print(f"\n  kappa de <N> ~ t^kappa, ajuste em {janela[0]} <= t <= {janela[1]}")
    print(f"  {'p':>5} | {'kappa':>6} {'EP':>6} | janelas alternativas")
    for p in ps:
        M = np.array([[np.nan if x is None else float(x) for x in sr['N_col']]
                      for sr in dados[p]['series']], float)
        k = ajuste(M, *janela)
        n = len(M)
        ep = np.std([ajuste(M[rng.integers(0, n, n)], *janela) for _ in range(n_bs)])
        alt = [ajuste(M, a, b) for a, b in alternativas]
        print(f"  {p:5.2f} | {k:6.3f} {ep:6.3f} | " +
              "  ".join(f"{a}-{b}: {v:5.2f}" for (a, b), v in zip(alternativas, alt)) +
              f"   (maior desvio {max(abs(v - k) for v in alt):.2f})")


kappa_bootstrap()
