"""
Figura 2.9 — Esquemas de atualização (sincrono_assincrono_real.pdf).

(a) síncrono: todos os sítios revisam nos instantes inteiros; (b) assíncrono em tempo
contínuo: cada sítio revisa em instantes próprios, independentes dos demais (tipicamente, um
processo de Poisson de taxa 1; aqui, instantes ilustrativos fixos); (c) fração aleatória:
a cada passo, só uma parte sorteada dos sítios é atualizada.

Esquemático. Substitui scripts_antigos/cap2_3/gerar_relogio_simulacao.py. Correção: no
painel (b) a versão antiga punha os cinco sítios revisando juntos em t, o que é justamente o
que o esquema assíncrono não faz; aqui nenhum instante coincide.
"""
from matplotlib.lines import Line2D

import estilo as E

E.aplicar()

SITIOS = [1, 2, 3, 4, 5]
T_MAX = 2.25
KW = dict(s=28, edgecolors=E.PRETO, linewidths=0.6, zorder=3)


def eixo(ax, titulo):
    ax.set_yticks(SITIOS)
    ax.set_yticklabels([f'Sítio {k}' for k in SITIOS])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$t$', r'$t+1$', r'$t+2$'])
    for y in SITIOS:
        ax.axhline(y, color=E.CINZA, ls='--', lw=0.5, zorder=1)
    for t in (0, 1, 2):
        ax.axvline(t, color=E.PRETO, ls=':', lw=0.8, zorder=2)
    ax.set_xlim(-0.3, T_MAX + 0.05)
    ax.set_ylim(0.4, 5.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_title(titulo)


fig, axs = E.figura(1, 3, altura=5.6, sharey=True)
a, b, c = axs
eixo(a, '(a) Síncrono')
for t in (0, 1, 2):
    a.scatter([t] * 5, SITIOS, color=E.AZUL, **KW)

eixo(b, '(b) Assíncrono contínuo')
# instantes ilustrativos, escolhidos à mão: independentes entre sítios, sem coincidências,
# em média um por sítio por unidade de tempo
INSTANTES = {1: [0.35, 1.55], 2: [-0.12, 0.80, 1.92], 3: [0.55, 1.28, 2.15], 4: [0.15, 1.05],
             5: [0.68, 1.72]}
for y, ts in INSTANTES.items():
    b.scatter(ts, [y] * len(ts), color=E.VERMELHO, **KW)

eixo(c, '(c) Fração aleatória')
atual = {0: SITIOS, 1: [1, 3, 4], 2: [2, 3, 4, 5]}
for t, lista in atual.items():
    ign = [k for k in SITIOS if k not in lista]
    c.scatter([t] * len(lista), lista, color=E.AZUL, **KW)
    if ign:
        c.scatter([t] * len(ign), ign, color='white', hatch='//////', **KW)
alcas = [Line2D([], [], marker='o', ls='', mfc=E.AZUL, mec=E.PRETO, mew=0.6, ms=5, label='atualizado'),
         Line2D([], [], marker='o', ls='', mfc='white', mec=E.PRETO, mew=0.6, ms=5, label='ignorado')]
c.legend(handles=alcas, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
         handletextpad=0.2, columnspacing=0.8)
E.salvar(fig, 'sincrono_assincrono_real')
