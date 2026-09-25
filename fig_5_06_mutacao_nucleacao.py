"""
Figura 5.6 — Probabilidade de a frente escapar antes da nucleação (mutacao_3_diagrama.pdf).

Lê mutacao_v1_moore1_eps1e-05_n09/resultados_b*_mu*.json, gravados por varredura_mutacao.py
(L = 101, trinta sementes por ponto: os valores citados na Seção 5.4 são estes). Não simula nada.

Correção em relação à versão anterior: os valores de p NÃO são igualmente espaçados
(0,42 0,44 0,48 0,52 0,56 em b = 1,07; 0,30 0,32 0,36 0,42 0,48 em b = 1,45), e o imshow
com 'extent' os desenhava como se fossem, deslocando as células em até 0,03 no eixo p. Aqui
cada célula vai de um ponto médio ao seguinte (pcolormesh), centrada no p medido.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import estilo as E

E.aplicar()

PASTA = os.path.join(E.AQUI, 'mutacao_v1_moore1_eps1e-05_n09')
B_VALS = [1.07, 1.45]
MU_VALS = [1e-3, 3e-3, 1e-2, 3e-2]
PC_MU0 = {1.07: 0.3548, 1.45: 0.2090}
PC = 0.592746
L = 101                     # 30 sementes por ponto (L = 201 tem 10); é o L citado no texto


def bordas(v, log=False):
    """Bordas de células centradas nos valores v (pontos médios; extremos por simetria)."""
    v = np.log10(v) if log else np.asarray(v, float)
    m = (v[1:] + v[:-1]) / 2
    e = np.concatenate([[v[0] - (m[0] - v[0])], m, [v[-1] + (v[-1] - m[-1])]])
    return 10 ** e if log else e


def mu_tex(m):
    ex = int(np.floor(np.log10(m) + 1e-9)); c = round(m / 10 ** ex)
    return rf'$10^{{{ex}}}$' if c == 1 else rf'${c}\times10^{{{ex}}}$'


fig, axs = E.figura(1, 2, altura=6.2)
for ax, b in zip(axs, B_VALS):
    tab = {}
    for mu in MU_VALS:
        arq = os.path.join(PASTA, f'resultados_b{b:.4f}_mu{mu:g}.json')
        for k, e in json.load(open(arq)).items():
            if not k.startswith('_') and e.get('L') == L:
                tab[(mu, round(e['p'], 4))] = e['P_esc_antes_nuc']
    ps = sorted({p for _, p in tab})
    M = np.array([[tab.get((mu, p), np.nan) for p in ps] for mu in MU_VALS])
    print(f"  b = {b}: p = {ps}")
    for mu, lin in zip(MU_VALS, M):
        print(f"     mu = {mu:.0e}: " + "  ".join(f"{x:4.2f}" for x in lin))
    im = ax.pcolormesh(bordas(ps), bordas(MU_VALS, log=True), M, cmap='viridis', vmin=0, vmax=1,
                       edgecolor='white', lw=0.4)
    ax.set_yscale('log')
    ax.set_yticks(MU_VALS)
    ax.set_yticklabels([mu_tex(m) for m in MU_VALS])
    ax.minorticks_off()
    ax.axvline(PC_MU0[b], color=E.PRETO, ls=':', lw=1.0)
    ax.axvline(PC, color=E.CINZA_ESCURO, ls='--', lw=0.9)
    ax.set_xlim(PC_MU0[b] - 0.03, PC + 0.02)
    ax.set_xlabel(r'Fração de buracos $p$')
    ax.set_title(rf'$b={E.num(b, 2)}$, $L={L}$')
    E.virgula(ax, eixos='x')
axs[0].set_ylabel(r'Taxa de mutação $\mu$')
cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
cb.set_label(r'$P$(escape antes da nucleação)')
E.virgula(cb.ax, eixos='y')
E.salvar(fig, 'mutacao_3_diagrama')


# ==============================================================================
# EXPOENTE theta de t_esc ~ mu^(-theta) (Seção 5.4.4), com incerteza por bootstrap
# ==============================================================================
# Células (b, p, L) em que ao menos 90% das realizações escapam em três ou mais taxas;
# o ajuste usa a mediana de t_esc entre as que escaparam, nessas taxas. As sementes são as
# mesmas em todas as taxas, então a reamostragem sorteia índices de semente uma vez e os
# aplica a todas as taxas da célula (reamostragem pareada).
def theta_bootstrap(n_bs=1000, semente=7, p_min=0.9, n_taxas=3):
    cel = {}
    for b in B_VALS:
        for mu in MU_VALS:
            arq = os.path.join(PASTA, f'resultados_b{b:.4f}_mu{mu:g}.json')
            for k, e in json.load(open(arq)).items():
                if not k.startswith('_'):
                    cel.setdefault((b, round(e['p'], 2), e['L']), {})[mu] = e
    rng = np.random.default_rng(semente)
    print("\n  theta de t_esc ~ mu^-theta (mediana de t_esc entre as que escapam; bootstrap pareado)")
    print(f"  {'b':>5} {'p':>5} {'L':>4} | {'taxas':>22} | {'theta':>6} {'EP':>5} | {'16%':>5} {'84%':>5} | n")
    for (b, p, L), dd in sorted(cel.items()):
        ms = [m for m in MU_VALS if m in dd and dd[m]['P_esc'] >= p_min]
        if len(ms) < n_taxas:
            continue
        T = np.array([dd[m]['t_esc'] for m in ms], float)
        lx = np.log(ms)

        def ajuste(Tk):
            med = [np.median(r[r > 0]) if (r > 0).any() else np.nan for r in Tk]
            return -np.polyfit(lx, np.log(med), 1)[0] if np.all(np.isfinite(med)) else np.nan

        th = ajuste(T)
        n = T.shape[1]
        bs = np.array([ajuste(T[:, rng.integers(0, n, n)]) for _ in range(n_bs)])
        lo, hi = np.nanpercentile(bs, [16, 84])
        print(f"  {b:5.2f} {p:5.2f} {L:4d} | {' '.join(f'{m:g}' for m in ms):>22} | "
              f"{th:6.2f} {np.nanstd(bs):5.2f} | {lo:5.2f} {hi:5.2f} | {n}")


theta_bootstrap()
