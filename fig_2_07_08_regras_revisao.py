"""
Figuras 2.7 e 2.8 — Regras de revisão de estratégia.

    2.7  Figura_Imitate_Best.pdf   imitar o melhor: o foco copia o vizinho de maior ganho
    2.8  Figura_Regra_Fermi.pdf    W(s_i <- s_j) = 1/(1 + exp[(Pi_i - Pi_j)/K]), Eq. (2.x)

Esquemáticos; não usam dados. Substitui scripts_antigos/cap2_4/gerar_regras_atualizacao.py.
Correções: a anotação "Empate Topológico: P = 0,5" virou "W = 1/2" (a grandeza no eixo é W, e
o empate é de ganhos, não topológico); números negativos com sinal de menos tipográfico;
tamanho final.
"""
import numpy as np
import matplotlib.patches as patches

import estilo as E

E.aplicar()


def imitar_melhor():
    fig, ax = E.figura(altura=6.72, largura=0.42)
    ganhos = {(0, 2): 1.2, (1, 2): 0.5, (2, 2): 1.8,
              (0, 1): 2.1, (1, 1): 2.0, (2, 1): 3.2,
              (0, 0): 0.0, (1, 0): 1.5, (2, 0): 2.8}
    for (i, j), g in ganhos.items():
        v = E.num(g, 1)
        lw, z = 0.7, 2
        if (i, j) == (1, 1):
            cor, txt, ct = E.CINZA_ESCURO, r'\textbf{Foco} $i$' '\n' rf'$\Pi_i={v}$', 'white'
        elif (i, j) == (2, 1):
            cor, txt, ct = E.AZUL_MEDIO, r'\textbf{Maior} $j$' '\n' rf'$\Pi_j={v}$', 'white'
            lw, z = 2.0, 4
        else:
            cor, txt, ct = E.AZUL, rf'$\Pi={v}$', 'white'
        ax.add_patch(patches.Rectangle((i, j), 1, 1, facecolor=cor, edgecolor=E.PRETO, lw=lw,
                                       zorder=z))
        ax.text(i + .5, j + .42, txt, ha='center', va='center', color=ct,
                fontsize=E.TAM_ANOTACAO, zorder=z + 1, linespacing=1.3)
    ax.annotate('', xy=(1.62, 1.82), xytext=(2.38, 1.82),
                arrowprops=dict(arrowstyle='-|>', color=E.VERMELHO, lw=1.4, mutation_scale=10,
                                connectionstyle='arc3,rad=0.35'), zorder=6)
    ax.text(2.0, 1.97, 'copia', color=E.VERMELHO, ha='center', va='bottom',
            fontsize=E.TAM_ANOTACAO, zorder=7, bbox=dict(fc='white', ec='none', pad=0.4))
    ax.set_xlim(-0.01, 3.01); ax.set_ylim(-0.01, 3.01)
    ax.set_aspect('equal'); ax.axis('off')
    E.salvar(fig, 'Figura_Imitate_Best', recortar=False)


def fermi():
    fig, ax = E.figura(altura=6.0, largura=0.6)
    d = np.linspace(-3, 3, 1000)
    for K, cor, ls, rot in ((0.01, E.PRETO, '-', r'$K\to0$ (seleção forte)'),
                            (0.3, E.AZUL, '--', rf'$K={E.num(0.3)}$'),
                            (2.0, E.VERMELHO, ':', rf'$K={E.num(2.0, 1)}$ (seleção fraca)')):
        with np.errstate(over='ignore', under='ignore'):
            ax.plot(d, 1 / (1 + np.exp(-d / K)), color=cor, ls=ls, lw=1.3, label=rot, zorder=3)
    ax.axhline(0.5, color=E.CINZA, ls='--', lw=0.6, zorder=1)
    ax.axvline(0, color=E.CINZA, ls='--', lw=0.6, zorder=1)
    ax.plot([0], [0.5], 'o', color=E.PRETO, ms=4, zorder=4)
    ax.text(0.12, 0.46, r'ganhos iguais: $W=1/2$', va='top', fontsize=E.TAM_ANOTACAO,
            color=E.CINZA_ESCURO)
    ax.set_xlabel(r'Diferença de ganho $\Pi_j-\Pi_i$')
    ax.set_ylabel(r'$W(s_i\leftarrow s_j)$')
    ax.set_xlim(-3, 3); ax.set_ylim(-0.03, 1.03)
    ax.spines[['top', 'right']].set_visible(False)
    E.virgula(ax)
    ax.legend(loc='upper left')
    E.salvar(fig, 'Figura_Regra_Fermi')


imitar_melhor()
fermi()
