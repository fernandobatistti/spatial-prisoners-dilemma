"""
varredura_mutacao.py — Experimento B: a mutação desaprisiona a frente?

Coloque este arquivo na MESMA PASTA do engine_v2.py e rode:

    python varredura_mutacao.py               # roda (ou continua) a varredura
    python varredura_mutacao.py --piloto      # versão curta e barata, para calibrar
    python varredura_mutacao.py --so-plotar   # refaz figuras e resumo a partir dos JSON
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

B_VALS = [1.07, 1.45]

P_POR_B = {
    1.07: [0.42, 0.44, 0.48, 0.52, 0.56],
    1.45: [0.30, 0.32, 0.36, 0.42, 0.48],
}
PC_MU0 = {1.07: 0.3548, 1.45: 0.2090}

MU_VALS = [1e-3, 3e-3, 1e-2, 3e-2]
RAZAO_MU = 1.0

TAMANHOS = [101, 201]
N_SEMENTES = {101: 30, 201: 10}

K_TEMPO = 30.0
T_TETO = 20000
T_POS = 2000
INTERVALO = 10
N_CHECK = 40
S_NUCLEO = 9
SEMENTE_BASE = 2000
SEMENTE_MUTACAO = 900000
LADO_COLONIA = 9
EPSILON = 1e-5
BLOCO = 5
VERSAO = 1

if PILOTO:
    B_VALS = [1.07]
    P_POR_B = {1.07: [0.42, 0.50, 0.58]}
    MU_VALS = [3e-3, 3e-2]
    TAMANHOS = [101]
    N_SEMENTES = {101: 8}
    T_TETO = 10000
    BLOCO = 2

def t_max(mu):
    return int(min(T_TETO, max(1000, K_TEMPO / mu)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, ("piloto_" if PILOTO else "") +
                     f"mutacao_v{VERSAO}_moore1_eps{EPSILON:g}_n0{LADO_COLONIA}")

def configurar_motor(b, mu):
    motor.TIPO_VIZINHANCA = "moore"
    motor.PROFUNDIDADE_VIZINHANCA = 1
    motor.TOPOLOGIA_TABULEIRO = "toroide"
    motor.ESTRATEGIA_MAR = motor.INDICE_D
    motor.ESTRATEGIA_INVASOR = motor.INDICE_C
    motor.CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": LADO_COLONIA}]
    motor.CONFIG_BURACOS = []
    motor.PROPORCAO_RANDOM_COOPERADORES = 0.0
    motor.K_FERMI = 0.0
    motor.PROBABILIDADE_ATUALIZACAO = 1.0
    motor.EPSILON_P = EPSILON
    motor.TAXA_MUTACAO_C = RAZAO_MU * mu
    motor.TAXA_MUTACAO_D = mu
    motor.b = b

def meta():
    return dict(versao=VERSAO, semente_base=SEMENTE_BASE, semente_mutacao=SEMENTE_MUTACAO,
                k_tempo=K_TEMPO, t_teto=T_TETO, t_pos=T_POS, intervalo=INTERVALO,
                s_nucleo=S_NUCLEO, razao_mu=RAZAO_MU, lado_colonia=LADO_COLONIA,
                epsilon=EPSILON, vizinhanca="moore1", topologia="toroide")

# ==============================================================================
# UMA REALIZAÇÃO
# ==============================================================================
def uma_rodada(L, p, mu, semente, directions, buffers):
    labels, qr, qc, ox, oy, px, py, keep = buffers
    np.random.seed(semente)
    grid, paredes, colonia, origem = motor.inicializar_universo(L, p_buracos=p, p_coop=0.0)
    ativo, ell, ux, uy = motor.construir_substrato(grid, colonia, origem, L, directions)
    comp = ell >= 0
    alvo = L // 2 - 1
    raio = np.maximum(np.abs(ux), np.abs(uy))
    n_ativos = max(int(ativo.sum()), 1)

    rng = np.random.default_rng(SEMENTE_MUTACAO + semente)
    PM = motor.matriz_payoff()
    C, D = motor.INDICE_C, motor.INDICE_D
    di = np.zeros((1, 1), np.int32)
    du = np.zeros((1, 1))
    mu_c, mu_d = RAZAO_MU * mu, mu

    t_esc = t_nuc = -1
    serie_rho, serie_perc = [], []
    T = t_max(mu)
    fim = T
    checks = sorted(set(np.unique(np.geomspace(INTERVALO, T, N_CHECK).astype(int)
                                  // INTERVALO * INTERVALO).tolist()))
    raio_serie = []
    for t in range(1, T + 1):
        _, _, pay = motor.censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0], PM[1, 1], directions)
        prop = motor.escolha_racional_numba(grid, pay, L, 0.0, 0.0, directions, di, du)
        r = rng.random((L, L))
        mut = ((grid == C) & (r < mu_c)) | ((grid == D) & (r < mu_d))
        grid = np.where(mut, (prop + 1) % motor.NUM_STRATEGIES, prop).astype(np.int32)
        grid[~ativo] = motor.INDICE_BURACO

        eh_c = grid == C
        nlab = motor.rotular_toro(eh_c, L, directions, labels, qr, qc, ox, oy, px, py)
        colonia = motor.atualizar_colonia(labels, nlab, colonia, L, directions, keep)
        sel = colonia & comp
        
        if t_esc < 0 and sel.any() and raio[sel].max() >= alvo:
            t_esc = t
        if t_nuc < 0 and nlab > 0:
            tam = np.bincount(labels.ravel(), minlength=nlab + 1)
            tam[0] = 0
            fora = np.ones(nlab + 1, bool)
            fora[0] = False
            rot_col = np.unique(labels[colonia])
            fora[rot_col[rot_col > 0]] = False
            if fora.any() and tam[fora].max() >= S_NUCLEO:
                t_nuc = t
        if t % INTERVALO == 0:
            serie_rho.append(float(eh_c.sum()) / n_ativos)
            labs_grandes = np.unique(labels[eh_c])
            labs_grandes = labs_grandes[labs_grandes > 0]
            serie_perc.append(float(bool(px[labs_grandes].any() or py[labs_grandes].any()))
                              if labs_grandes.size else 0.0)
            if t in checks:
                raio_serie.append([t, float(raio[sel].max()) / alvo if sel.any() else 0.0])
                
        if t_esc > 0 and t >= t_esc + T_POS:
            fim = t
            break

    n = len(serie_rho)
    j0, j1 = int(0.8 * n), int(0.6 * n)
    sel = colonia & comp
    return dict(t_esc=t_esc, t_nuc=t_nuc, t_fim=fim,
                rho_est=float(np.mean(serie_rho[j0:])) if n else np.nan,
                rho_ant=float(np.mean(serie_rho[j1:j0])) if j0 > j1 else np.nan,
                perc=float(np.mean(serie_perc[j0:])) if n else np.nan,
                alcance=float(raio[sel].max()) / alvo if sel.any() else 0.0,
                n_col=int(sel.sum()), raio_serie=raio_serie)

# ==============================================================================
# ARQUIVOS
# ==============================================================================
def arquivo(b, mu):
    return os.path.join(PASTA, f"resultados_b{b:.4f}_mu{mu:g}.json")

def carregar(b, mu):
    arq = arquivo(b, mu)
    if not os.path.exists(arq):
        return {"_meta": meta()}
    d = json.load(open(arq))
    if d.get("_meta") != meta():
        raise SystemExit(f"\n[ERRO] {os.path.basename(arq)} foi gerado com outros parâmetros.\n")
    return d

def gravar(b, mu, dados):
    os.makedirs(PASTA, exist_ok=True)
    tmp = arquivo(b, mu) + ".tmp"
    json.dump(dados, open(tmp, "w"), indent=1)
    os.replace(tmp, arquivo(b, mu))

CAMPOS = ('t_esc', 't_nuc', 't_fim', 'rho_est', 'rho_ant', 'perc', 'alcance', 'n_col', 'raio_serie')

def resumir(e, n_sem):
    te = np.array(e['t_esc'], float)
    tn = np.array(e['t_nuc'], float)
    esc = te > 0
    e['n'] = len(te)
    e['P_esc'] = float(esc.mean())
    e['t_esc_mediano'] = float(np.median(te[esc])) if esc.any() else np.nan
    e['t_fim_medio'] = float(np.mean(e['t_fim']))
    e['P_esc_antes_nuc'] = float(np.mean(esc & ((tn < 0) | (te < tn))))
    e['P_nuc'] = float(np.mean(tn > 0))
    e['t_nuc_mediano'] = float(np.median(tn[tn > 0])) if (tn > 0).any() else np.nan
    e['raio_mediano'] = float(np.median([r[-1][1] if r else 0.0 for r in e['raio_serie']]))
    for c in ('rho_est', 'rho_ant', 'perc', 'alcance'):
        v = np.array(e[c], float)
        e[c + '_media'] = float(np.nanmean(v))
        e[c + '_erro'] = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
    e['estacionario'] = bool(abs(e['rho_est_media'] - e['rho_ant_media']) <=
                             2 * max(e['rho_est_erro'], 1e-9) + 0.01)
    return e

# ==============================================================================
# EXECUÇÃO E ANÁLISE DE DADOS
# ==============================================================================
def rodar():
    os.makedirs(PASTA, exist_ok=True)
    directions = motor.gerar_vizinhanca("moore", 1)
    t_inicio = time.time()
    
    for b in B_VALS:
        for mu in MU_VALS:
            configurar_motor(b, mu)
            dados = carregar(b, mu)
            for L in TAMANHOS:
                buffers = (np.zeros((L, L), np.int32), np.zeros(L * L, np.int32),
                           np.zeros(L * L, np.int32), np.zeros((L, L), np.int32),
                           np.zeros((L, L), np.int32), np.zeros(L * L + 1, np.bool_),
                           np.zeros(L * L + 1, np.bool_), np.zeros(L * L + 1, np.bool_))
                for p in P_POR_B[b]:
                    chave = f"L{L}_p{p:.4f}"
                    ent = dados.get(chave) or dict(L=L, p=p, mu=mu, b=b,
                                                   **{c: [] for c in CAMPOS})
                    feitas = len(ent['t_esc'])
                    alvo_n = N_SEMENTES[L]
                    if feitas >= alvo_n:
                        continue
                    t0 = time.time()
                    for s in range(feitas, alvo_n):
                        r = uma_rodada(L, p, mu, SEMENTE_BASE + s, directions, buffers)
                        for c in CAMPOS:
                            ent[c].append(r[c])
                        if (s + 1) % BLOCO == 0 or s + 1 == alvo_n:
                            dados[chave] = ent
                            gravar(b, mu, dados)
                    dados[chave] = resumir(ent, alvo_n)
                    gravar(b, mu, dados)
                    
            plotar(b, dados, mu)
    resumo_geral()

# ==============================================================================
# FIGURAS (VETORIZADAS, COM PADRÃO LATEX)
# ==============================================================================
def formatar_virgula(x, pos):
    if isinstance(x, float) and abs(x) > 0 and abs(x - round(x)) < 1e-10:
        x = round(x)
    return f"{x:g}".replace('.', '{,}')

def formatar_potencia(x, pos):
    if x > 0:
        expoente = int(np.round(np.log10(x)))
        if abs(np.log10(x) - expoente) < 1e-5:
            return rf"$10^{{{expoente}}}$"
    return ""

def tipografia():
    plt.rcParams.update({
        'text.usetex': True,
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
    fig.savefig(os.path.join(PASTA, f'{nome}.pdf'), format='pdf', transparent=True, dpi=600)
    plt.close(fig)
    print(f"  figura: {nome}.pdf gerada com sucesso!")

def pontos(dados, L=None):
    v = [e for k, e in dados.items() if not k.startswith('_') and 'n' in e]
    if L is not None:
        v = [e for e in v if e['L'] == L]
    return sorted(v, key=lambda e: (e['L'], e['p']))


def plotar(b, dados, mu):
    """Gera o sumário por p para um dado par (b, mu)."""
    tipografia()
    v = pontos(dados)
    if not v: return
    Ls = sorted({e['L'] for e in v})
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    
    cor_b = '#0000ff' if b == 1.07 else '#ff0000' # Azul para baixo b, vermelho para alto b
    
    for L in Ls:
        w = pontos(dados, L)
        ps = [e['p'] for e in w]
        
        # O L define a transparência e tipo da linha
        alpha_l = 1.0 if L == max(Ls) else 0.5
        ls_l = '-' if L == max(Ls) else '--'
        
        axs[0].plot(ps, [e['P_esc'] for e in w], marker='o', ls=ls_l, color=cor_b, alpha=alpha_l, lw=2, label=fr'$L = {L}$')
        axs[0].plot(ps, [e['P_esc_antes_nuc'] for e in w], marker='s', mfc='white', ls=ls_l, color=cor_b, alpha=alpha_l, lw=1.5, label=fr'$L = {L}$ (antes da nuc.)')
        
        axs[1].errorbar(ps, [e['rho_est_media'] for e in w], yerr=[e['rho_est_erro'] for e in w], fmt='o', ls=ls_l, color=cor_b, alpha=alpha_l, capsize=3, lw=2, label=fr'$L = {L}$')
        axs[2].plot(ps, [e['perc_media'] for e in w], marker='o', ls=ls_l, color=cor_b, alpha=alpha_l, lw=2, label=fr'$L = {L}$')

    for ax, titulo in zip(axs, (r'Probabilidade $P(\mathrm{escape})$', r'Fração $\rho_C$ estacionária', r'Fração do tempo percolando')):
        
        # Limiares Teóricos
        ax.axvline(PC_MU0.get(b, np.nan), color='black', ls=':', lw=1.5)
        ax.axvline(0.592746, color='dimgrey', ls='--', lw=1.5)
        
        ax.set_xlabel(r'Fração de buracos $p$')
        ax.set_title(titulo, pad=10)
        ax.grid(True, ls=':', alpha=0.5)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))

    axs[0].legend(fontsize=10, frameon=False)
    
    str_b, str_mu = f"{b:.3f}".replace('.', '{,}'), f"{mu:g}".replace('.', '{,}')
    fig.suptitle(fr'Efeito da mutação local ($b = {str_b}$, $\mu = {str_mu}$) | Linha Pontilhada = $p^*_c$ original', y=1.05)
    salvar(fig, f'mutacao_1_p_b{b:.4f}_mu{mu:g}')


def resumo_geral():
    tipografia()
    linhas = []
    tabela = {}
    for b in B_VALS:
        for mu in MU_VALS:
            if not os.path.exists(arquivo(b, mu)): continue
            d = carregar(b, mu)
            for e in pontos(d):
                tabela[(b, mu, e['L'], e['p'])] = e
    if not tabela: return
    
    cab = (f"{'b':>6} {'mu':>8} {'L':>5} {'p':>6} {'P_esc':>7} {'antes nuc':>10} "
           f"{'t_esc':>8} {'t_nuc':>8} {'rho_C':>7} {'perc':>6} {'estac.':>7}")
    linhas += ["=" * len(cab), "EXPERIMENTO B — MUTAÇÃO COMO TAXA POR SÍTIO POR PASSO", "=" * len(cab), cab]
    for (b, mu, L, p), e in sorted(tabela.items()):
        linhas.append(f"{b:>6} {mu:>8.0e} {L:>5} {p:>6.2f} {e['P_esc']:>7.2f} "
                      f"{e['P_esc_antes_nuc']:>10.2f} {e['t_esc_mediano']:>8.0f} "
                      f"{e['t_nuc_mediano']:>8.0f} {e['rho_est_media']:>7.3f} "
                      f"{e['perc_media']:>6.2f} {'sim' if e['estacionario'] else 'NÃO':>7}")
    linhas.append("=" * len(cab))
    txt = "\n".join(linhas)
    open(os.path.join(PASTA, 'mutacao_tabela.txt'), 'w').write(txt + "\n")

    # --------------------------------------------------------------------------
    # Figura 2: Tempo de escape contra \mu (Assinatura do Mecanismo)
    # --------------------------------------------------------------------------
    Lmax = max(TAMANHOS)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    cores = plt.get_cmap('coolwarm')(np.linspace(0, 1, sum(len(P_POR_B[b]) for b in B_VALS)))
    mks = ['o', 's', 'D', '^', 'v'] * 3
    idx_cor = 0
    
    for b in B_VALS:
        for p in P_POR_B[b]:
            xs, ys = [], []
            for mu in MU_VALS:
                e = tabela.get((b, mu, Lmax, p))
                if e and np.isfinite(e['t_esc_mediano']) and e['P_esc'] > 0.5:
                    xs.append(mu); ys.append(e['t_esc_mediano'])
            if len(xs) > 1:
                str_b, str_p = f"{b:.2f}".replace('.', '{,}'), f"{p:.2f}".replace('.', '{,}')
                ax.plot(xs, ys, marker=mks[idx_cor], color=cores[idx_cor], ls='-', lw=2, ms=6, label=fr'$b = {str_b}$, $p = {str_p}$')
                idx_cor += 1
                
    if ax.lines:
        mus = np.array(MU_VALS, float)
        ref = ax.lines[0].get_ydata()[0] * (mus / ax.lines[0].get_xdata()[0]) ** -1.0
        ax.plot(mus, ref, color='black', ls='--', lw=2, label=r'Teórico $\propto \mu^{-1}$')
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatar_potencia))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_potencia))
    
    ax.set_xlabel(r'Taxa de mutação $\mu$')
    ax.set_ylabel(r'Tempo mediano de escape $t_{\mathrm{esc}}$')
    ax.set_title(fr'Escala de tempo do desaprisionamento ($L = {Lmax}$)', pad=15)
    
    ax.grid(True, which='major', ls='-', alpha=0.2)
    ax.grid(True, which='minor', ls=':', alpha=0.1)
    ax.legend(fontsize=10, loc='upper right', frameon=False)
    salvar(fig, 'mutacao_2_tempo_escape')

    # --------------------------------------------------------------------------
    # Figura 3: Diagrama de Calor (P_escape antes da Nucleação)
    # --------------------------------------------------------------------------
    fig, axs = plt.subplots(1, len(B_VALS), figsize=(6.5 * len(B_VALS), 5.5), squeeze=False)
    for ax, b in zip(axs[0], B_VALS):
        M = np.full((len(MU_VALS), len(P_POR_B[b])), np.nan)
        for i, mu in enumerate(MU_VALS):
            for j, p in enumerate(P_POR_B[b]):
                e = tabela.get((b, mu, Lmax, p))
                if e: M[i, j] = e['P_esc_antes_nuc']
                
        # Usamos interpolation='none' para os pixels do heatmap não borrarem
        im = ax.imshow(M, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis', interpolation='none',
                       extent=[min(P_POR_B[b]) - .02, max(P_POR_B[b]) + .02, -0.5, len(MU_VALS) - 0.5])
                       
        ax.set_yticks(range(len(MU_VALS)))
        
        # Formata \mu em notação científica no eixo Y para ficar limpo
        y_labels = []
        for m in MU_VALS:
            exp = int(np.round(np.log10(m)))
            num = int(m / (10**exp))
            if num == 1: y_labels.append(rf'$10^{{{exp}}}$')
            else: y_labels.append(rf'${num} \times 10^{{{exp}}}$')
        ax.set_yticklabels(y_labels)
        
        ax.axvline(PC_MU0.get(b, np.nan), color='black', ls=':', lw=2)
        ax.axvline(0.592746, color='dimgrey', ls='--', lw=2)
        ax.set_xlim(min(PC_MU0.get(b, 0.3), min(P_POR_B[b])) - 0.02, 0.61)
        
        ax.set_xlabel(r'Fração de buracos $p$')
        ax.set_ylabel(r'Taxa de mutação $\mu$')
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))
        
        str_b = f"{b:.3f}".replace('.', '{,}')
        ax.set_title(rf'Tentação $b = {str_b}$', pad=10)
        
        # Barra de cor customizada
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'Probabilidade (escape antes da nucleação)')
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))

    fig.suptitle(r'Desaprisionamento por mutação (Linha Pontilhada: $p^*_c$ em $\mu = 0$)', fontsize=15, y=1.05)
    salvar(fig, 'mutacao_3_diagrama')


if __name__ == "__main__":
    if "--so-plotar" in sys.argv:
        for b in B_VALS:
            for mu in MU_VALS:
                if os.path.exists(arquivo(b, mu)):
                    plotar(b, carregar(b, mu), mu)
        resumo_geral()
    else:
        rodar()