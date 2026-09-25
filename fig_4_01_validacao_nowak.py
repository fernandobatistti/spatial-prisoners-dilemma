"""
Figura 4.1 — Validação com condição inicial aleatória (validacao_rho_b.pdf).

Lê dados/resumo_validacao.csv (uma linha por semente; gerado por resumo_validacao.py a partir
das rodadas de validação) e imprime todos os números citados na Seção 4.2: média sobre as 30
sementes, fração de sementes extintas e média (com erro padrão) entre as sobreviventes.
"""
import os

import numpy as np
import pandas as pd

import estilo as E

E.aplicar()

LIM = [(8/7, '8/7'), (7/6, '7/6'), (6/5, '6/5'), (5/4, '5/4'), (4/3, '4/3'), (7/5, '7/5'),
       (3/2, '3/2'), (8/5, '8/5'), (5/3, '5/3'), (7/4, '7/4')]
EXTINTA = 0.01                      # rho abaixo disso: cooperação extinta

d = pd.read_csv(os.path.join(E.AQUI, 'dados', 'resumo_validacao.csv'), sep=';')
g = d.groupby('b').rho_medio
bs = np.array(sorted(d.b.unique()))
med = g.mean().values
sem = (g.std() / np.sqrt(g.size())).values
ext = g.apply(lambda x: (x < EXTINTA).mean()).values
sob = g.apply(lambda x: x[x >= EXTINTA].mean() if (x >= EXTINTA).any() else np.nan).values
sob_ep = g.apply(lambda x: x[x >= EXTINTA].std() / np.sqrt((x >= EXTINTA).sum())
                 if (x >= EXTINTA).sum() > 1 else np.nan).values

print("  b       rho (30 sementes)     extintas   sobreviventes")
for x, m, s, e, so, se in zip(bs, med, sem, ext, sob, sob_ep):
    txt = f"{so:.3f} +- {se:.3f}" if np.isfinite(so) else "-"
    print(f"  {x:.3f}   {m:.3f} +- {s:.3f}      {e:4.0%}      {txt}")

fig, axs = E.figura(1, 2, altura=6.2, gridspec_kw={'width_ratios': [1.6, 1]})

a = axs[0]
for x, lab in LIM:
    a.axvline(x, color='#e0e0e0', lw=0.7, zorder=0)
    a.text(x, 0.015, rf'$\tfrac{{{lab.split("/")[0]}}}{{{lab.split("/")[1]}}}$', ha='center',
           va='bottom', fontsize=E.TAM_ANOTACAO, color=E.CINZA_ESCURO)
rng = np.random.default_rng(0)
a.plot(d.b + rng.normal(0, 0.005, len(d)), d.rho_medio, 'o', ms=1.8, mew=0, color=E.CINZA,
       alpha=0.6, zorder=1, label='sementes individuais')
a.errorbar(bs, med, yerr=sem, fmt='o-', ms=3, color=E.AZUL, zorder=3, label='média (30 sementes)')
a.plot(bs, sob, 's--', ms=2.8, color=E.VERMELHO, zorder=2, label='média (sobreviventes)')
a.axhline(0.30, color=E.PRETO, ls=':', lw=0.9, zorder=0)
a.text(1.0, 0.32, r'$\rho_{\mathsf{C}}\approx0{,}30$', fontsize=E.TAM_ANOTACAO)
a.set_xlabel(r'Tentação $b$')
a.set_ylabel(r'Fração estacionária $\rho_{\mathsf{C}}$')
a.set_xlim(0.98, 1.95); a.set_ylim(-0.03, 1.03)
a.set_title(r'(a) Condição inicial aleatória ($L=100$, $\rho_{\mathsf{C}}(0)=0{,}5$)')
a.legend(loc='center left')
a.grid(ls=':')
E.virgula(a)

b_ = axs[1]
b_.bar(bs[ext > 0], ext[ext > 0], width=0.015, color=E.VERMELHO, edgecolor=E.PRETO, lw=0.4)
for x, lab, ha, dx in ((8/5, r'$\tfrac{8}{5}$', 'right', -0.015), (5/3, r'$\tfrac{5}{3}$', 'left', 0.015)):
    b_.axvline(x, color=E.CINZA_ESCURO, lw=0.8, ls='--')
    b_.text(x + dx, 0.04, lab, ha=ha, va='bottom', fontsize=E.TAM_NUMERO, color=E.CINZA_ESCURO)
b_.set_xlabel(r'Tentação $b$')
b_.set_ylabel(r'Fração de sementes extintas')
b_.set_xlim(0.98, 1.95); b_.set_ylim(-0.02, 1.05)
b_.set_title(r'(b) Extinção da cooperação')
b_.grid(ls=':')
E.virgula(b_)

E.salvar(fig, 'validacao_rho_b')
