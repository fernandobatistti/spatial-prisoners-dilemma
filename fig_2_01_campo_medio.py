"""
Figura 2.1 — Aproximação de campo médio (campo_medio.pdf).

Esquerda: o indivíduo focal interage com oito vizinhos concretos (cinco cooperadores, três
desertores). Direita: na aproximação de campo médio, ele interage com o estado médio da
população, desenhado como um anel contínuo cuja cor é a mistura das duas estratégias NA
MESMA PROPORÇÃO dos vizinhos da esquerda (5/8 de azul, 3/8 de vermelho).

Esquemático; não usa dados. Substitui scripts_antigos/cap2_1/figuras_teoria.py
(gerar_campo_medio). Correções: aspecto 1:1 (o anel saía elíptico) e paleta da dissertação.
"""
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

import estilo as E

E.aplicar()

VIZINHOS = [E.AZUL, E.AZUL, E.VERMELHO, E.AZUL, E.VERMELHO, E.VERMELHO, E.AZUL, E.AZUL]
frac_c = VIZINHOS.count(E.AZUL) / len(VIZINHOS)
MISTURA = tuple(frac_c * np.array(to_rgb(E.AZUL)) + (1 - frac_c) * np.array(to_rgb(E.VERMELHO)))

fig, ax = E.figura(altura=5.0, largura=0.8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-5.6, 5.6)
ax.set_ylim(-2.75, 2.1)

# esquerda: interações locais
c0 = np.array([-3.5, 0.0])
for k, cor in enumerate(VIZINHOS):
    a = k * np.pi / 4
    p = c0 + 1.6 * np.array([np.cos(a), np.sin(a)])
    ax.plot(*zip(c0, p), color=E.PRETO, lw=0.9, zorder=1)
    ax.plot(*p, 'o', color=cor, ms=8, zorder=2)
ax.plot(*c0, 'o', color=E.PRETO, ms=9, zorder=3)
ax.text(c0[0], -2.2, 'vizinhos concretos', ha='center', va='top')

# direita: estado médio
c1 = np.array([3.5, 0.0])
ax.add_patch(patches.Wedge(c1, 1.8, 0, 360, width=0.4, color=MISTURA, lw=0, zorder=1))
ax.plot(*c1, 'o', color=E.PRETO, ms=9, zorder=3)
ax.text(c1[0], -2.2, r'estado médio da população', ha='center', va='top')

ax.annotate('', xy=(1.0, 0), xytext=(-1.0, 0),
            arrowprops=dict(arrowstyle='-|>', color=E.PRETO, lw=1.4, mutation_scale=12))
E.salvar(fig, 'campo_medio', recortar=False)
