"""
Figura 5.4 — Razões críticas decisivas (mecanismo_1_razoes.pdf).

Lê mecanismo_v1_moore1_eps1e-05_n09/mecanismo_resumo.json, gravado por
mecanismo_fronteira.py, e desenha as ONZE razões da Tabela 5.3, na mesma ordem --- inclusive
2/1, que decide de 13% a 30% das mudanças. (A versão anterior da figura omitia 2/1: a
tabela iterava sobre as razões de (1,2) mais 2/1, a figura só sobre as de (1,2).)
Não simula nada.
"""
import json
import os
from fractions import Fraction

import numpy as np

import estilo as E

E.aplicar()

ARQ = os.path.join(E.AQUI, 'mecanismo_v1_moore1_eps1e-05_n09', 'mecanismo_resumo.json')
d = json.load(open(ARQ))
bs = list(d)                                     # na ordem gravada: b crescente
RAZOES = sorted((Fraction(r) for r in d[bs[0]] if not r.startswith('_')))

# por razão: [decisões, realizações em que decidiu, sítios D, sítios C]; _totais = [nreal, nsit, ndec]
dec = {b: [100 * d[b][str(r)][0] / d[b]['_totais'][2] for r in RAZOES] for b in bs}
rea = {b: [100 * d[b][str(r)][1] / d[b]['_totais'][0] for r in RAZOES] for b in bs}

print("  % das decisões  (compare com a Tabela 5.3)")
print("  razão " + "".join(f"{'b=' + b:>9}" for b in bs))
for i, r in enumerate(RAZOES):
    print(f"  {str(r):>5} " + "".join(f"{dec[b][i]:9.2f}" for b in bs))
cinco = [RAZOES.index(Fraction(*f)) for f in ((3, 2), (2, 1), (4, 3), (5, 3), (5, 4))]
print("  3/2+2/1+4/3+5/3+5/4: " + "  ".join(f"{sum(dec[b][i] for i in cinco):.1f}%" for b in bs))

x = np.arange(len(RAZOES))
larg = 0.82 / len(bs)
cores = E.serie(len(bs))
fig, axs = E.figura(2, 1, altura=10.5, sharex=True)
for i, (b, cor) in enumerate(zip(bs, cores)):
    for ax, v in ((axs[0], dec[b]), (axs[1], rea[b])):
        ax.bar(x + (i - (len(bs) - 1) / 2) * larg, v, width=larg, color=cor,
               edgecolor=E.PRETO, lw=0.25, label=rf'$b={E.num(float(b), 3)}$')
for ax in axs:
    ax.grid(axis='y', ls=':')
    E.virgula(ax, eixos='y')
axs[0].set_ylabel(r'\% das decisões de mudança')
axs[1].set_ylabel(r'\% das realizações afetadas')
axs[1].set_xticks(x)
axs[1].set_xticklabels([rf'$\dfrac{{{r.numerator}}}{{{r.denominator}}}$' for r in RAZOES])
axs[1].set_xlabel(r'Razão crítica $r_x=\hat{k}_x/\hat{m}_x$')
axs[0].legend(ncol=len(bs), loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
E.salvar(fig, 'mecanismo_1_razoes')
