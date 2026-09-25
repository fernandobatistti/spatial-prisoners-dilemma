"""
Figura D.1 — Diagnóstico da medida anterior (diagnostico_semente44_L201.pdf).

L = 201, semente 44, b = 1,2 (logo acima de 6/5), p = 0,407 (Moore); uma única realização,
apenas ilustrativa. Versão em estilo.py de diagnostico/fig_diag.py (mesmos painéis e
mesmos dados), agora vetorial, no tamanho final, com vírgula decimal e sem o título geral
(que passou para a legenda da figura).

PRECISA DE DADOS QUE NÃO ESTÃO NO REPOSITÓRIO (ver LEIA-ME, seção "Diagnóstico"):
    diagnostico/diag_s44.npz        <- diagnostico/run_diag.py  (engine.py ORIGINAL)
    diagnostico/diag_mu0_p0.npz     <- diagnostico/run_mu.py 0 300 0.0
    PASTA_V2/v2_b1.2000_eps0.01_mutC{mu}_mutD{mu}_pbur0.4073*/data/L_201/log_semente_44.csv
                                    <- diagnostico/run_v2.py (engine_v2.py, mu = 0, 1e-5, 1e-4, 1e-3)
Por padrão PASTA_V2 é esta pasta (scripts_figuras/), onde diagnostico/run_v2.py grava.
"""
import glob
import os

import numpy as np
import pandas as pd

import estilo as E

E.aplicar()

DIAG = os.path.join(E.AQUI, 'diagnostico')
PASTA_V2 = os.environ.get('PASTA_V2', E.AQUI)   # onde diagnostico/run_v2.py grava
MUS = [('0', 0.0), ('1e-05', 1e-5), ('0.0001', 1e-4), ('0.001', 1e-3)]
COR_MU = [E.PRETO, E.CINZA_MEDIO, E.AZUL, E.VERMELHO]


def novo(tag):
    padrao = os.path.join(PASTA_V2, f'v2_b1.2000_eps0.01_mutC{tag}_mutD{tag}_pbur0.4073*',
                          'data', 'L_201', 'log_semente_44.csv')
    arqs = glob.glob(padrao)
    if not arqs:
        raise SystemExit(f"não encontrei {padrao}\n(ajuste PASTA_V2; ver o cabeçalho do script)")
    return pd.read_csv(arqs[0], sep=';')


def mu_tex(mu):
    if mu == 0:
        return r'$\mu=0$'
    ex = int(np.floor(np.log10(mu) + 1e-9))
    return rf'$\mu=10^{{{ex}}}$'


for arq in ('diag_s44.npz', 'diag_mu0_p0.npz'):
    if not os.path.exists(os.path.join(DIAG, arq)):
        print(f"  PULADO: falta diagnostico/{arq} (ver o cabeçalho deste script)")
        raise SystemExit(0)
antigo = np.load(os.path.join(DIAG, 'diag_s44.npz'))
p0 = np.load(os.path.join(DIAG, 'diag_mu0_p0.npz'))
N4 = novo('0.0001')

fig, axs = E.figura(2, 3, altura=11.5)
kw_leg = dict(fontsize=E.TAM_ANOTACAO, handlelength=1.4)

# (a) mesma rodada, observável antigo e novo
a = axs[0, 0]
a.loglog(antigo['t'], antigo['rg2'], '.', ms=1.2, color=E.CINZA, label=r'$R_g^2$ antigo')
a.loglog(N4['step'], N4['rg2'], color=E.AZUL, lw=0.9, label=r'$R_g^2$ novo (colônia)')
a.loglog(N4['step'], N4['R2_origem'], color=E.VERMELHO, lw=0.9, label=r'$R^2$ (colônia)')
t_inv = int(N4['step'][N4['valido_geom'] == 0].min())
a.axvline(t_inv, color=E.PRETO, ls='--', lw=0.8)
a.text(t_inv * 1.2, 0.03, rf'alcança $L/2$' '\n' rf'($t={t_inv}$)', transform=a.get_xaxis_transform(),
       fontsize=E.TAM_ANOTACAO, va='bottom')
a.set_title('(a) Observáveis antigo e novo')
a.set_xlabel(r'$t$'); a.set_ylabel('Tamanho quadrático')
a.set_ylim(top=a.get_ylim()[1] * 30)                 # espaço para a legenda
a.legend(loc='upper left', **kw_leg)

# (b) as quedas eram fragmentação
a = axs[0, 1]
m = antigo['t'] <= 4000
a.plot(antigo['t'][m], antigo['m_trk'][m], color=E.CINZA, lw=0.6, label='aglomerado rastreado')
a.plot(N4['step'][:4000], N4['N_col'][:4000], color=E.AZUL, lw=1.0, label=r'$N_{\mathrm{col}}$')
a.plot(N4['step'][:4000], N4['massa_maior_comp'][:4000], color=E.VERMELHO_MEDIO, lw=0.5,
       label='maior componente')
a.set_title('(b) Fragmentação')
a.set_xlabel(r'$t$'); a.set_ylabel('Sítios')
a.set_ylim(top=a.get_ylim()[1] * 1.35)               # espaço para a legenda
a.legend(loc='upper left', **kw_leg)

# (c) controle sem buracos
a = axs[0, 2]
tt = p0['t'][p0['t'] < 96]
a.loglog(tt, p0['rg2'][:len(tt)], 'o', ms=2, mfc='none', color=E.AZUL, label='simulação')
a.loglog(tt, ((9 + 2 * tt) ** 2 - 1) / 6, color=E.PRETO, lw=0.8, label=r'$[(9+2t)^2-1]/6$')
a.loglog(tt, (2 * tt) ** 2 / 6, color=E.PRETO, ls=':', lw=0.8, label=r'$\propto t^2$')
a.set_title(r'(c) Controle sem buracos ($p=0$)')
a.set_xlabel(r'$t$'); a.set_ylabel(r'$R_g^2$')
a.legend(**kw_leg)

# (d)-(f) as quatro taxas
for (tag, mu), cor in zip(MUS, COR_MU):
    d = novo(tag)
    axs[1, 0].loglog(d['step'], d['N_col'], color=cor, lw=0.9, label=mu_tex(mu))
    if mu > 0:
        axs[1, 1].loglog(mu * d['step'], d['N_col'], color=cor, lw=0.9, label=mu_tex(mu))
    axs[1, 2].semilogx(d['step'], d['ell_max'].where(d['valido_geom'] == 1), color=cor, lw=0.9,
                       label=mu_tex(mu))
axs[1, 2].plot([1, 60], [5, 64], color=E.PRETO, ls=':', lw=0.8, label=r'$\propto t$')
axs[1, 0].set_title(r'(d) Com $\mu=0$ a colônia congela')
axs[1, 1].set_title(r'(e) Contra $\mu t$')
axs[1, 2].set_title(r'(f) A frente química trava')
axs[1, 0].set_xlabel(r'$t$'); axs[1, 1].set_xlabel(r'$\mu t$'); axs[1, 2].set_xlabel(r'$t$')
axs[1, 0].set_ylabel(r'$N_{\mathrm{col}}$'); axs[1, 1].set_ylabel(r'$N_{\mathrm{col}}$')
axs[1, 2].set_ylabel(r'$\ell_{\mathrm{frente}}$')
for a in axs[1]:
    a.legend(**kw_leg)
for a in axs.ravel():
    a.grid(which='major', ls=':')
E.virgula(axs[0, 1], axs[1, 2], eixos='y')
E.salvar(fig, 'diagnostico_semente44_L201')
