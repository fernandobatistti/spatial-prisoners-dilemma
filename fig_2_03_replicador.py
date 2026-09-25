"""
Figura 2.3 — Dilema do Prisioneiro no campo médio (replicador_dp.pdf).

(a) campo de velocidades x' = -x(1-x)[(b-1)x + eps(1-x)], Eq. (eq:replicador-nowak);
(b) integração numérica a partir de x(0) = 0,99, com a assíntota 1/[(b-1)t] do dilema fraco.
Não depende de simulação: roda em segundos.
"""
import numpy as np
from scipy.integrate import solve_ivp

import estilo as E

E.aplicar()

B = 1.65
X0 = 0.99
CONDICOES = [
    dict(eps=0.0, ls='-', cor=E.AZUL, lab=r'$\varepsilon=0$ (dilema fraco)'),
    dict(eps=0.1, ls='--', cor=E.VERMELHO, lab=r'$\varepsilon=0{,}1$ (dilema estrito)'),
]


def xdot(x, eps):
    return -x * (1 - x) * ((B - 1) * x + eps * (1 - x))


fig, (ax1, ax2) = E.figura(1, 2, altura=6.2)

# (a) campo de velocidades
x = np.linspace(0, 1, 400)
ax1.axhline(0, color=E.CINZA, lw=0.7, zorder=1)
for c in CONDICOES:
    ax1.plot(x, xdot(x, c['eps']), ls=c['ls'], color=c['cor'], label=c['lab'], zorder=2)
for xx in (0.25, 0.5, 0.75):                           # fluxo: do repulsor para o atrator
    ax1.annotate('', xy=(xx - 0.08, 0), xytext=(xx, 0), zorder=3,
                 arrowprops=dict(arrowstyle='->', color=E.VERMELHO_ESCURO, lw=1.2))
ax1.plot([0], [0], 'o', color=E.PRETO, ms=5, zorder=4)                     # atrator
ax1.plot([1], [0], 'o', mfc='white', mec=E.PRETO, ms=5, mew=0.8, zorder=4)  # repulsor
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\dot{x}$')
ax1.set_title(rf'(a) Campo de velocidades ($b={E.num(B)}$)')
ax1.set_ylim(-0.128, 0.008)
ax1.legend(loc='lower left')
ax1.grid(ls=':')
E.virgula(ax1)

# (b) extinção no tempo
t = np.logspace(-1, 3, 400)
for c in CONDICOES:
    sol = solve_ivp(lambda tt, y: xdot(y, c['eps']), (0, t[-1]), [X0],
                    t_eval=t, rtol=1e-10, atol=1e-14)
    ax2.loglog(sol.t, sol.y[0], ls=c['ls'], color=c['cor'], label=c['lab'])
ta = t[t > 20]
ax2.loglog(ta, 1 / ((B - 1) * ta), ':', color=E.CINZA_ESCURO, lw=1.3,
           label=r'assíntota $1/[(b-1)t]$')
ax2.set_ylim(1e-4, 1.5)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$x(t)$')
ax2.set_title(rf'(b) Extinção dos cooperadores ($x_0={E.num(X0)}$)')
ax2.legend(loc='lower left')
ax2.grid(ls=':')

E.salvar(fig, 'replicador_dp')
