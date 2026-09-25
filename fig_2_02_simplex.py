"""
Figura 2.2 — Simplex S_3 e o fluxo do replicador (figura_simplex.pdf).

As curvas são trajetórias VERDADEIRAS da Eq. (2.x) (equação do replicador), integradas
numericamente para o jogo com aptidões f_i = 1 - x_i (matriz A = J - I: cada estratégia ganha 1
contra as outras e 0 contra si mesma). Esse jogo tem exatamente a estrutura que a figura
antiga desenhava à mão: vértices instáveis, pontos médios das arestas que atraem ao longo da
aresta e repelem para o interior (selas), e um atrator no centro (1/3, 1/3, 1/3). As retas
tracejadas que ligam os pontos médios ao centro também são trajetórias (as separatrizes: a
reta x_i = x_j é invariante por simetria).

Substitui scripts_antigos/cap2_1/gerar_figuras_teoria.py (gerar_simplex), cujas setas eram
arcos desenhados, não o fluxo de um jogo.
"""
import numpy as np
from scipy.integrate import solve_ivp

import estilo as E

E.aplicar()

A = np.ones((3, 3)) - np.eye(3)
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])        # vértices: estratégias 1, 2, 3


def replicador(t, x):
    f = A @ x
    return x * (f - x @ f)


def xy(x):
    return np.asarray(x) @ V


def trajetoria(x0, T=12):
    s = solve_ivp(replicador, (0, T), x0, rtol=1e-9, atol=1e-12, dense_output=True)
    return s.sol(np.linspace(0, T, 600)).T


fig, ax = E.figura(altura=8.0, largura=0.6)
ax.set_aspect('equal')
ax.axis('off')
ax.fill(*V.T, facecolor='none', edgecolor=E.PRETO, lw=1.0, zorder=1)

centro = np.ones(3) / 3
# separatrizes: dos pontos médios das arestas ao centro
for k in range(3):
    m = np.full(3, 0.5); m[k] = 0.0
    x = trajetoria(m + 1e-4 * (centro - m) / np.linalg.norm(centro - m), T=60)
    ax.plot(*xy(x).T, color=E.CINZA_MEDIO, lw=0.8, ls='--', zorder=2)
# trajetórias a partir de perto dos vértices (e de perto das arestas), nos dois sentidos
for k in range(3):
    for desv in (0.03, -0.03):
        x0 = np.full(3, 0.04); x0[k] = 0.92
        x0[(k + 1) % 3] += desv; x0 /= x0.sum()
        x = trajetoria(x0)
        P = xy(x)
        ax.plot(*P.T, color=E.AZUL, lw=1.0, zorder=3)
        # seta a meio caminho (em comprimento de arco)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        i = int(np.searchsorted(s, 0.45 * s[-1]))
        ax.annotate('', xy=P[i + 1], xytext=P[i - 1],
                    arrowprops=dict(arrowstyle='-|>', color=E.AZUL, lw=0, mutation_scale=9))

for p in V:
    ax.plot(*p, 'o', color=E.PRETO, ms=4.5, zorder=5)
pc = xy(centro)
ax.plot(*pc, 'o', color=E.VERMELHO_ESCURO, ms=5.5, zorder=6)
for k in range(3):
    m = np.full(3, 0.5); m[k] = 0.0
    ax.plot(*xy(m), 'o', color=E.CINZA_MEDIO, ms=3.5, zorder=5)

bx = r'\boldsymbol{x}'
ax.text(*(V[0] + [0, -0.04]), rf'Estratégia 1' '\n' rf'${bx}=(1,0,0)$', ha='center', va='top')
ax.text(*(V[1] + [0, -0.04]), rf'Estratégia 2' '\n' rf'${bx}=(0,1,0)$', ha='center', va='top')
ax.text(*(V[2] + [0, 0.04]), rf'Estratégia 3' '\n' rf'${bx}=(0,0,1)$', ha='center', va='bottom')
ax.text(0.5, -0.03, r'$(1/2,1/2,0)$', ha='center', va='top', fontsize=E.TAM_ANOTACAO)
ax.text(*(xy([0, .5, .5]) + [0.03, 0]), r'$(0,1/2,1/2)$', ha='left', va='center',
        fontsize=E.TAM_ANOTACAO)
ax.text(*(xy([.5, 0, .5]) - [0.03, 0]), r'$(1/2,0,1/2)$', ha='right', va='center',
        fontsize=E.TAM_ANOTACAO)
ax.text(pc[0] + 0.035, pc[1] - 0.035, r'$(1/3,1/3,1/3)$', ha='left', va='top',
        color=E.VERMELHO_ESCURO, fontsize=E.TAM_ANOTACAO, zorder=7,
        bbox=dict(fc='white', ec='none', pad=0.6))
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.17, 1.03)
E.salvar(fig, 'figura_simplex', recortar=False)
