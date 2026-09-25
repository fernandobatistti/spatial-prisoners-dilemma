"""
Figura 2.10 — Limiares k/m da vizinhança de Moore e destino de um único desertor (limiares_km.pdf).

Os regimes vêm de `verificacoes_apendices.py B1` (e do Apêndice B.1): período 2 abaixo de
6/5, período 3 até 7/5, quadrado 3x3 estável até 8/5, crescimento irregular até 5/3,
invasão até 7/4 e cruz diagonal até 2. Nenhuma simulação: roda em segundos.
"""
import matplotlib.patches as patches

import estilo as E

E.aplicar()

LIMIARES = [(1.0, '1'), (8/7, r'$\dfrac{8}{7}$'), (7/6, r'$\dfrac{7}{6}$'),
            (6/5, r'$\dfrac{6}{5}$'), (5/4, r'$\dfrac{5}{4}$'), (4/3, r'$\dfrac{4}{3}$'),
            (7/5, r'$\dfrac{7}{5}$'), (3/2, r'$\dfrac{3}{2}$'), (8/5, r'$\dfrac{8}{5}$'),
            (5/3, r'$\dfrac{5}{3}$'), (7/4, r'$\dfrac{7}{4}$'), (2.0, '2')]

# (início, fim, rótulo, cor de fundo): o esquema azul -> vermelho do autor
REGIOES = [
    (1.0, 6/5, r'Ciclo\\(período 2)', E.AZUL_CLARO),
    (6/5, 7/5, r'Ciclo\\(período 3)', E.AZUL_MEDIO),
    (7/5, 8/5, r'Quadrado\\$3\times3$ estável', E.AZUL),
    (8/5, 5/3, r'Cresc.\\irreg.', E.VERMELHO_CLARO),
    (5/3, 7/4, r'Invasão', E.VERMELHO),
    (7/4, 2.0, r'Cruz diagonal\\expansiva', E.VERMELHO_ESCURO),
]
CLARAS = {E.AZUL_CLARO, E.VERMELHO_CLARO}      # fundo claro: texto preto, para ter contraste

fig, ax = E.figura(altura=3.0)
y0, y1 = 0.42, 0.88
for x0, x1, txt, cor in REGIOES:
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=cor,
                                   edgecolor=E.PRETO, lw=0.7))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, r'\shortstack{' + txt + '}', ha='center',
            va='center', fontsize=E.TAM_ANOTACAO,
            color=E.PRETO if cor in CLARAS else 'white')

ax.plot([0.99, 2.01], [y0, y0], color=E.PRETO, lw=0.9)
for v, rot in LIMIARES:
    ax.plot([v, v], [y0, y0 - 0.04], color=E.PRETO, lw=0.7)
    ax.text(v, y0 - 0.06, rot, ha='center', va='top', fontsize=E.TAM_NUMERO)
ax.text(1.5, 0.10, r'Tentação $b$', ha='center', va='top', fontsize=E.TAM_ROTULO)

ax.set_xlim(0.98, 2.02)
ax.set_ylim(0.0, 0.92)
ax.axis('off')
E.salvar(fig, 'limiares_km')
