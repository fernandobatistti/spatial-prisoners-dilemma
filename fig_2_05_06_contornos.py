"""
Figuras 2.5 e 2.6 — Condições de contorno.

    2.5  Figura_Toroide.pdf   (a) identificação das bordas opostas; (b) o toro resultante
    2.6  Figura_Parede.pdf    parede: vizinhança de Moore truncada junto à borda inativa

Esquemáticos; não usam dados. Substitui scripts_antigos/cap2_3/gerar_contornos.py (mesmo
desenho, no tamanho final e com a paleta de estilo.py). As setas de (a) indicam para onde vai
quem sai por cada borda; as cores seguem a figura original (azul na vertical, vermelho na
horizontal) — aqui elas NÃO significam cooperador/desertor.
"""
import numpy as np
import matplotlib.patches as patches

import estilo as E

E.aplicar()


# ------------------------------------------------------------------ Figura 2.5
def toroide():
    fig = E.figura(altura=5.6, largura=0.85)[0]
    fig.set_layout_engine('none')
    fig.clf()
    ax1 = fig.add_axes([0.0, 0.0, 0.5, 0.9])
    ax2 = fig.add_axes([0.5, 0.0, 0.5, 0.9], projection='3d')
    L = 6
    for i in range(L + 1):
        ax1.plot([0, L], [i, i], color=E.CINZA_MEDIO, lw=0.6, ls='--', zorder=1)
        ax1.plot([i, i], [0, L], color=E.CINZA_MEDIO, lw=0.6, ls='--', zorder=1)
    ax1.add_patch(patches.Rectangle((0, 0), L, L, facecolor='none', edgecolor=E.PRETO, lw=1.2,
                                    zorder=3))
    seta = dict(arrowstyle='-|>', lw=1.1, mutation_scale=8)
    ax1.annotate('', xy=(L / 2, L + 1), xytext=(L / 2, L), arrowprops=dict(seta, color=E.AZUL))
    ax1.annotate('', xy=(L / 2, 0), xytext=(L / 2, -1), arrowprops=dict(seta, color=E.AZUL))
    ax1.annotate('', xy=(L + 1, L / 2), xytext=(L, L / 2), arrowprops=dict(seta, color=E.VERMELHO))
    ax1.annotate('', xy=(0, L / 2), xytext=(-1, L / 2), arrowprops=dict(seta, color=E.VERMELHO))
    kw = dict(fontsize=E.TAM_ANOTACAO)
    ax1.text(L / 2, L + 1.15, r'topo $\to$ base', ha='center', va='bottom', color=E.AZUL, **kw)
    ax1.text(L / 2, -1.15, r'base $\to$ topo', ha='center', va='top', color=E.AZUL, **kw)
    ax1.text(L + 0.25, L / 2 + 0.25, r'dir. $\to$' '\n' r'esq.', ha='left', va='bottom',
             color=E.VERMELHO, **kw)
    ax1.text(-0.25, L / 2 + 0.25, r'esq. $\to$' '\n' r'dir.', ha='right', va='bottom',
             color=E.VERMELHO, **kw)
    ax1.set_xlim(-2.0, L + 2.0); ax1.set_ylim(-2.0, L + 2.0)
    ax1.set_aspect('equal'); ax1.axis('off')

    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, 2 * np.pi, 25))
    R, r = 3, 1
    ax2.plot_wireframe((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u),
                       r * np.sin(v), color=E.CINZA_MEDIO, linewidth=0.3)
    ax2.set_xlim(-3.2, 3.2); ax2.set_ylim(-3.2, 3.2); ax2.set_zlim(-2.2, 2.2)
    ax2.set_box_aspect((1, 1, 0.69), zoom=1.12)
    ax2.view_init(elev=35, azim=50)
    ax2.axis('off')
    fig.canvas.draw()
    for ax, t in ((ax1, '(a) Bordas opostas identificadas'), (ax2, '(b) A superfície resultante: um toro')):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.985, t, ha='center', va='top')
    E.salvar(fig, 'Figura_Toroide', recortar=False)


# ------------------------------------------------------------------ Figura 2.6
def parede():
    fig, ax = E.figura(altura=9.2, largura=0.5)
    L = 7                                   # borda inativa em volta de um núcleo 5x5
    for i in range(L):
        for j in range(L):
            if i in (0, L - 1) or j in (0, L - 1):
                ax.add_patch(patches.Rectangle((i, j), 1, 1, facecolor=E.CINZA_CLARO,
                                               hatch='////', edgecolor=E.PRETO, lw=0.6, zorder=2))
            else:
                ax.add_patch(patches.Rectangle((i, j), 1, 1, facecolor='none',
                                               edgecolor=E.PRETO, lw=0.6, zorder=2))
    cx, cy = 1, 3
    ax.add_patch(patches.Rectangle((cx, cy), 1, 1, facecolor=E.CINZA_ESCURO, edgecolor=E.PRETO,
                                   lw=0.6, zorder=3))
    ax.text(cx + .5, cy + .5, r'\textbf{Foco}', ha='center', va='center', color='white',
            fontsize=E.TAM_ANOTACAO, zorder=4)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            perdido = nx in (0, L - 1) or ny in (0, L - 1)
            ax.add_patch(patches.Rectangle((nx, ny), 1, 1, lw=0.6, edgecolor=E.PRETO, zorder=3,
                                           facecolor=E.VERMELHO_MEDIO if perdido else E.AZUL_MEDIO))
            ax.text(nx + .5, ny + .5, 'Perda' if perdido else 'Válido', ha='center',
                    va='center', color='white' if not perdido else E.PRETO,
                    fontsize=E.TAM_ANOTACAO, zorder=4)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect('equal'); ax.axis('off')
    alcas = [patches.Patch(facecolor=E.CINZA_CLARO, hatch='////', edgecolor=E.PRETO, lw=0.6,
                           label='Parede (inativa)'),
             patches.Patch(facecolor=E.CINZA_ESCURO, edgecolor=E.PRETO, lw=0.6, label='Foco'),
             patches.Patch(facecolor=E.AZUL_MEDIO, edgecolor=E.PRETO, lw=0.6,
                           label='Vizinho válido'),
             patches.Patch(facecolor=E.VERMELHO_MEDIO, edgecolor=E.PRETO, lw=0.6,
                           label='Vizinho perdido')]
    ax.legend(handles=alcas, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2)
    E.salvar(fig, 'Figura_Parede')


toroide()
parede()
