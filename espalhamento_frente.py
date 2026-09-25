"""
espalhamento_frente.py — Experimento D: como a frente cresce, e o quanto ela fica atrás da
queima no mesmo substrato.

Coloque este arquivo na MESMA PASTA do engine_v2.py e rode:

    python espalhamento_frente.py               # roda (ou continua)
    python espalhamento_frente.py --piloto      # versão curta
    python espalhamento_frente.py --so-plotar   # refaz figuras e tabela dos JSON
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import engine_v2 as motor

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
PILOTO = "--piloto" in sys.argv

B = 1.07                     # primeiro patamar; p*_c = 0.3548 (Experimento A)
PC_DIN = 0.3548
P_VALS = [0.00, 0.20, 0.30, 0.34, 0.36, 0.40]    # limpo, longe, perto (abaixo), no limiar, acima
TAMANHOS = [201, 401]
N_SEMENTES = {201: 120, 401: 60}
T_MAX = 2000
N_CHECK = 60                 # instantes log-espaçados guardados por realização
SEMENTE_BASE = 2000
LADO_COLONIA = 9
EPSILON = 1e-5
BLOCO = 10
USAR_LATEX = True
VERSAO = 1

if PILOTO:
    P_VALS = [0.00, 0.30, 0.40]
    TAMANHOS = [201]
    N_SEMENTES = {201: 10}
    T_MAX = 800
    BLOCO = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, ("piloto_" if PILOTO else "") +
                     f"espalhamento_v{VERSAO}_b{B:.4f}_moore1_n0{LADO_COLONIA}")


def configurar_motor():
    motor.TIPO_VIZINHANCA = "moore"
    motor.PROFUNDIDADE_VIZINHANCA = 1
    motor.TOPOLOGIA_TABULEIRO = "toroide"
    motor.ESTRATEGIA_MAR = motor.INDICE_D
    motor.ESTRATEGIA_INVASOR = motor.INDICE_C
    motor.CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": LADO_COLONIA}]
    motor.CONFIG_BURACOS = []
    motor.PROPORCAO_RANDOM_COOPERADORES = 0.0
    motor.TAXA_MUTACAO_C = motor.TAXA_MUTACAO_D = 0.0
    motor.K_FERMI = 0.0
    motor.PROBABILIDADE_ATUALIZACAO = 1.0
    motor.EPSILON_P = EPSILON
    motor.b = B


def meta():
    return dict(versao=VERSAO, b=B, semente_base=SEMENTE_BASE, t_max=T_MAX, n_check=N_CHECK,
                lado_colonia=LADO_COLONIA, epsilon=EPSILON, vizinhanca="moore1",
                topologia="toroide")


def instantes():
    return sorted(set(np.unique(np.geomspace(1, T_MAX, N_CHECK).astype(int)).tolist()))


# ==============================================================================
# UMA REALIZAÇÃO
# ==============================================================================
CAMPOS = ('N_col', 'R2_origem', 'rg2', 'anisotropia', 'ell_max', 'ell_medio', 'alcance',
          'valido_geom', 'percola', 'n_comp_colonia', 'massa_maior_comp')


def uma_rodada(L, p, semente, directions, buffers):
    labels, qr, qc, ox, oy, px, py, keep = buffers
    np.random.seed(semente)
    grid, paredes, colonia, origem = motor.inicializar_universo(L, p_buracos=p, p_coop=0.0)
    ativo, ell, ux, uy = motor.construir_substrato(grid, colonia, origem, L, directions)

    # --- a queima, de graça: uma camada química por passo, no mesmo substrato ---
    dentro = ell >= 0
    r2 = (ux.astype(np.float64) ** 2 + uy.astype(np.float64) ** 2)
    ells = ell[dentro]
    r2s = r2[dentro]
    ordem = np.argsort(ells, kind='stable')
    ells, r2s = ells[ordem], r2s[ordem]
    N_ac = np.arange(1, len(ells) + 1, dtype=np.float64)
    R2_ac = np.cumsum(r2s) / N_ac

    PM = motor.matriz_payoff()
    C = motor.INDICE_C
    di = np.zeros((1, 1), np.int32)
    du = np.zeros((1, 1))
    checks = instantes()
    serie = {c: [] for c in CAMPOS}
    serie['N_queima'] = []
    serie['R2_queima'] = []
    vistos = set()
    congelado = None
    t = 0
    for t in range(1, T_MAX + 1):
        if congelado is None:
            _, _, pay = motor.censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0],
                                                   PM[1, 1], directions)
            grid = motor.escolha_racional_numba(grid, pay, L, 0.0, 0.0, directions, di, du)
            grid[~ativo] = motor.INDICE_BURACO
            nlab = motor.rotular_toro(grid == C, L, directions, labels, qr, qc, ox, oy, px, py)
            colonia = motor.atualizar_colonia(labels, nlab, colonia, L, directions, keep)
            h = hash(grid.tobytes())
            if h in vistos or not (grid == C).any():
                tam = np.bincount(labels.ravel(), minlength=nlab + 1)
                tam[0] = 0
                congelado = motor.observaveis_colonia(colonia, ell, ux, uy, labels, tam,
                                                      px, py, L)
            vistos.add(h)
        if t in checks:
            if congelado is not None:
                obs = congelado
            else:
                tam = np.bincount(labels.ravel(), minlength=nlab + 1)
                tam[0] = 0
                obs = motor.observaveis_colonia(colonia, ell, ux, uy, labels, tam, px, py, L)
            for c in CAMPOS:
                v = obs.get(c, np.nan)
                serie[c].append(None if (isinstance(v, float) and not np.isfinite(v))
                                else (round(float(v), 4) if isinstance(v, float) else int(v)))
            j = int(np.searchsorted(ells, t, side='right'))
            serie['N_queima'].append(int(j))
            serie['R2_queima'].append(round(float(R2_ac[j - 1]), 4) if j > 0 else 0.0)
    return serie


# ==============================================================================
# ARQUIVOS
# ==============================================================================
def arquivo(p):
    return os.path.join(PASTA, f"espalhamento_p{p:.4f}.json")


def carregar(p):
    if not os.path.exists(arquivo(p)):
        return {"_meta": meta()}
    d = json.load(open(arquivo(p)))
    if d.get("_meta") != meta():
        raise SystemExit(f"[ERRO] {os.path.basename(arquivo(p))} tem outros parâmetros. "
                         f"Mude VERSAO ou apague a pasta.")
    return d


def gravar(p, dados):
    os.makedirs(PASTA, exist_ok=True)
    tmp = arquivo(p) + '.tmp'
    json.dump(dados, open(tmp, 'w'))
    os.replace(tmp, arquivo(p))


def rodar():
    os.makedirs(PASTA, exist_ok=True)
    configurar_motor()
    directions = motor.gerar_vizinhanca("moore", 1)
    t0 = time.time()
    for p in P_VALS:
        dados = carregar(p)
        for L in TAMANHOS:
            buffers = (np.zeros((L, L), np.int32), np.zeros(L * L, np.int32),
                       np.zeros(L * L, np.int32), np.zeros((L, L), np.int32),
                       np.zeros((L, L), np.int32), np.zeros(L * L + 1, np.bool_),
                       np.zeros(L * L + 1, np.bool_), np.zeros(L * L + 1, np.bool_))
            chave = f"L{L}"
            ent = dados.get(chave) or dict(L=L, p=p, b=B, series=[])
            for s in range(len(ent['series']), N_SEMENTES[L]):
                ent['series'].append(uma_rodada(L, p, SEMENTE_BASE + s, directions, buffers))
                if (s + 1) % BLOCO == 0 or s + 1 == N_SEMENTES[L]:
                    dados[chave] = ent
                    gravar(p, dados)
            dados[chave] = ent
            gravar(p, dados)
            fim = [sr['N_col'][-1] for sr in ent['series']]
            vivos = sum(1 for v in fim if v and v > 0)
            print(f"  p={p:.2f} L={L:4d}: {len(ent['series'])} realizações | "
                  f"{vivos} com colônia viva no fim | massa final média {np.mean(fim):.0f} "
                  f"({(time.time() - t0) / 60:.1f} min)")
    resumo()


# ==============================================================================
# ANÁLISE E PLOTAGEM VETORIZADA
# ==============================================================================
def media_serie(ent, campo, apenas_validas=False):
    """Média sobre realizações, instante a instante, ignorando ausentes."""
    M = []
    for sr in ent['series']:
        v = [np.nan if x is None else float(x) for x in sr[campo]]
        if apenas_validas:
            ok = [np.nan if x is None else float(x) for x in sr['valido_geom']]
            v = [a if (b == 1) else np.nan for a, b in zip(v, ok)]
        M.append(v)
    M = np.array(M, float)
    with np.errstate(invalid='ignore'):
        return np.nanmean(M, axis=0), np.sum(np.isfinite(M), axis=0)


def formatar_virgula(x, pos):
    """Substitui pontos por vírgulas nos decimais dos eixos."""
    if isinstance(x, float) and abs(x) > 0 and abs(x - round(x)) < 1e-10:
        x = round(x)
    return f"{x:g}".replace('.', '{,}')

def formatar_potencia(x, pos):
    """Força o formato LaTeX 10^x rigoroso para eixos logarítmicos."""
    if x > 0:
        expoente = int(np.round(np.log10(x)))
        if abs(np.log10(x) - expoente) < 1e-5:
            return rf"$10^{{{expoente}}}$"
    return ""


def tipografia():
    plt.rcParams.update({
        'text.usetex': USAR_LATEX,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.constrained_layout.use': True
    })

def salvar(fig, nome):
    os.makedirs(PASTA, exist_ok=True)
    # Salvamos direto em PDF com qualidade de publicação
    fig.savefig(os.path.join(PASTA, f'{nome}.pdf'), format='pdf', transparent=True, dpi=600)
    plt.close(fig)
    print(f"  figura: {nome}.pdf")


def resumo():
    tipografia()
    ts = np.array(instantes(), float)
    Lmax = max(TAMANHOS)
    dados = {p: carregar(p) for p in P_VALS if os.path.exists(arquivo(p))}
    dados = {p: d for p, d in dados.items() if f"L{Lmax}" in d}
    if not dados:
        return
        
    # Gradiente de cores usando a paleta 'viridis' otimizada
    cores = plt.get_cmap('viridis')(np.linspace(0, 0.9, len(dados)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    linhas = ["=" * 78, f"ESPALHAMENTO DA FRENTE (b = {B}, L = {Lmax})", "=" * 78,
              f"{'p':>6} {'N_col(fim)':>12} {'N_queima(fim)':>14} {'razão':>8} "
              f"{'R2(val)':>10} {'t(val)':>7} {'ell_max':>9} {'n_comp':>8}"]
              
    for (p, d), cor in zip(sorted(dados.items()), cores):
        ent = d[f"L{Lmax}"]
        N, _ = media_serie(ent, 'N_col')
        Nq, _ = media_serie(ent, 'N_queima')
        R2, nval = media_serie(ent, 'R2_origem', apenas_validas=True)
        ok = np.where(nval >= 0.5 * len(ent['series']))[0]
        jv = int(ok[-1]) if len(ok) else -1
        R2q, _ = media_serie(ent, 'R2_queima')
        em, _ = media_serie(ent, 'ell_max')
        nc, _ = media_serie(ent, 'n_comp_colonia')
        
        val, _ = media_serie(ent, 'valido_geom')
        dentro = np.where(val >= 0.9)[0]
        jc = int(dentro[-1]) + 1 if len(dentro) else len(ts)
        js = int(np.argmax(Nq >= 0.999 * Nq[-1])) + 1
        
        str_p = f"{p:.2f}".replace('.', '{,}')
        
        # Painel A
        axs[0, 0].plot(ts, N, color=cor, lw=2.5, label=fr'$p = {str_p}$')
        axs[0, 0].plot(ts, Nq, color=cor, ls=':', lw=1.5, alpha=0.7)
        # Painel B
        axs[0, 1].plot(ts[:jc], R2[:jc], color=cor, lw=2.5, label=fr'$p = {str_p}$')
        axs[0, 1].plot(ts[:jc], R2q[:jc], color=cor, ls=':', lw=1.5, alpha=0.7)
        # Painel C
        axs[1, 0].plot(ts[:js], (np.array(N) / np.maximum(Nq, 1))[:js], color=cor, lw=2.5, label=fr'$p = {str_p}$')
        axs[1, 0].plot(ts[js - 1:], (np.array(N) / np.maximum(Nq, 1))[js - 1:], color=cor, ls='--', lw=1.5, alpha=0.5)
        # Painel D
        axs[1, 1].plot(ts[:jc], nc[:jc], color=cor, lw=2.5, label=fr'$p = {str_p}$')
        
        linhas.append(f"{p:>6.2f} {N[-1]:>12.0f} {Nq[-1]:>14.0f} "
                      f"{N[-1] / max(Nq[-1], 1):>8.3f} {R2[jv]:>10.0f} {ts[jv]:>7.0f} "
                      f"{em[-1]:>9.1f} {nc[-1]:>8.2f}")

    # Organização de Títulos e Eixos
    paineis = [
        (axs[0, 0], r'(a) Massa $\langle N(t) \rangle$', r'$\langle N \rangle$ (linhas cheias)', True, True),
        (axs[0, 1], r'(b) Alcance quadrático $R^2(t)$', r'$R^2(t)$', True, True),
        (axs[1, 0], r'(c) Defasagem frente vs queima', r'$\langle N \rangle \,/\, N_{\mathrm{queima}}$', True, False),
        (axs[1, 1], r'(d) Fragmentação da colônia', r'Componentes conexos', False, False)
    ]

    for ax, titulo, y_label, log_x, log_y in paineis:
        if log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatar_potencia))
        if log_y:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_potencia))
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))
            
        ax.set_xlabel(r'Tempo $t$')
        ax.set_ylabel(y_label)
        ax.set_title(titulo, pad=12)
        ax.grid(True, which='major', ls='-', alpha=0.2)
        ax.grid(True, which='minor', ls=':', alpha=0.1)

    # Legenda em apenas um painel para não poluir
    axs[0, 0].legend(frameon=False, loc='upper left')
    
    str_B = f"{B:.2f}".replace('.', '{,}')
    fig.suptitle(fr'Espalhamento da frente e limite de queima ($b = {str_B}$, malha $L = {Lmax}$)', fontsize=16)
    
    salvar(fig, 'espalhamento_1_series')

    linhas.append("=" * 78)
    txt = "\n".join(linhas)
    print("\n" + txt)
    open(os.path.join(PASTA, 'espalhamento_tabela.txt'), 'w').write(txt + "\n")


if __name__ == "__main__":
    if "--so-plotar" in sys.argv:
        resumo()
    else:
        rodar()