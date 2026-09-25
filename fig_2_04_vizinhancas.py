"""
Figura 2.4 — Vizinhanças de von Neumann e de Moore, alcances 1 e 2 (vizinhancas.pdf).

Esquemático; não usa dados. Substitui scripts_antigos/cap2_3/gerar_imagens.py (mesmo
desenho, agora no tamanho final e com a paleta de estilo.py).
"""
import matplotlib.patches as patches

import estilo as E

E.aplicar()

N, C = 5, 2                                   # grade 5x5, foco no centro


def grade(ax, tipo, rmax):
    for i in range(N):
        for j in range(N):
            dx, dy = abs(i - C), abs(j - C)
            d = dx + dy if tipo == 'vn' else max(dx, dy)     # Manhattan / Chebyshev
            if d == 0:
                cor, txt = E.CINZA_ESCURO, r'\textbf{Foco}'
            elif d == 1:
                cor, txt = E.AZUL, r'$h=1$'
            elif d == 2 and rmax == 2:
                cor, txt = E.AZUL_MEDIO, r'$h=2$'
            else:
                cor, txt = None, ''
            if cor:
                ax.add_patch(patches.Rectangle((i, j), 1, 1, facecolor=cor, lw=0, zorder=2))
                ax.text(i + .5, j + .5, txt, ha='center', va='center', color='white',
                        fontsize=E.TAM_ANOTACAO, zorder=4)
    for k in range(N + 1):
        ax.plot([0, N], [k, k], color=E.PRETO, lw=0.7, zorder=3)
        ax.plot([k, k], [0, N], color=E.PRETO, lw=0.7, zorder=3)
    ax.set_xlim(-0.02, N + 0.02); ax.set_ylim(-0.02, N + 0.02)
    ax.set_aspect('equal'); ax.axis('off')


fig, axs = E.figura(2, 2, altura=12.0, largura=0.8)
for lin, tipo in enumerate(('vn', 'moore')):
    for col, r in enumerate((1, 2)):
        grade(axs[lin, col], tipo, r)
axs[0, 0].set_title(r'(a) Alcance $h=1$')
axs[0, 1].set_title(r'(b) Alcance $h=2$')
axs[0, 0].text(-0.35, 2.5, 'von Neumann', rotation=90, ha='center', va='center')
axs[1, 0].text(-0.35, 2.5, 'Moore', rotation=90, ha='center', va='center')
E.salvar(fig, 'vizinhancas')
