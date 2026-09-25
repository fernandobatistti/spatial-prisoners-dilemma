import os
import csv
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from numba import njit, prange

# ==============================================================================
# 1. PARÂMETROS
# ==============================================================================
GERAR_FRAMES = False
FRAME_SKIP = 25
CONFIG_SEMENTES = 30          # int -> sorteia N sementes | lista -> usa as sementes dadas

CONTINUAR_SIMULACAO = False
CHECKPOINT_SKIP = 10000
TAMANHOS_REDE = [100]
STEPS = 5000
PARADA_ANTECIPADA = True        # só atua se a dinâmica for determinística
PERIODO_MAX = 64                # novo parâmetro recomendado

# Matriz de payoff (Nowak, eq. 9.1): R=1, S=0, T=b, P=epsilon
b = 1.29                       # ATENÇÃO: 1.2 = 6/5 é ponto de transição (ver checar_parametros)
EPSILON_P = 0.0001

TOPOLOGIA_TABULEIRO = "toroide"  # "toroide" ou "parede"
TIPO_VIZINHANCA = "moore"        # "moore" ou "neumann"
PROFUNDIDADE_VIZINHANCA = 1
PROBABILIDADE_ATUALIZACAO = 1.0

TAXA_MUTACAO_C = 0 * 10e-4            # prob. de um sítio atualmente C sofrer erro de cópia
TAXA_MUTACAO_D = 0 * 10e-4            # prob. de um sítio atualmente D sofrer erro de cópia

ESTRATEGIA_MAR = 1
ESTRATEGIA_INVASOR = 0           # a colônia rastreada é sempre a estratégia invasora
CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": 5}]
CONFIG_BURACOS = []

# Fração de BURACOS no limiar de percolação de sítios (profundidade 1):
#   von Neumann: 1 - 0.592746 = 0.407254  |  Moore: 1 - 0.407254 = 0.592746
LIMIAR_BURACOS = {"neumann": 0.407254, "moore": 0.592746}
PROPORCAO_RANDOM_BURACOS = 0.35
PROPORCAO_RANDOM_COOPERADORES = 0.0

K_FERMI = 0.0
TAU_FERMI = 0.0

INDICE_C, INDICE_D, INDICE_BURACO = 0, 1, 2
NUM_STRATEGIES = 2

COLUNAS = ['step', 'frac_c', 'frac_d', 'porosidade_p', 'rho_c', 'atividade', 'n_mutacoes',
           'N_col', 'R2_origem', 'rg2', 'anisotropia', 'msd_cm', 'ell_max', 'ell_medio',
           'alcance', 'valido_geom', 'percola', 'n_comp_colonia', 'massa_maior_comp',
           'n_clusters_estrategia']

COLORS = np.array([
    [0.12, 0.46, 0.70, 1.0],   # C
    [0.83, 0.15, 0.15, 1.0],   # D
    [0.00, 0.00, 0.00, 1.0],   # buraco
    [0.62, 0.80, 0.93, 1.0],   # estratégia invasora FORA da colônia (mutantes/núcleos)
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def nome_experimento():
    return (f"v2_b{b:.4f}_eps{EPSILON_P:g}_mutC{TAXA_MUTACAO_C:g}_mutD{TAXA_MUTACAO_D:g}"
            f"_pbur{PROPORCAO_RANDOM_BURACOS:.4f}_pcoop{PROPORCAO_RANDOM_COOPERADORES:.3f}"
            f"_{TIPO_VIZINHANCA}{PROFUNDIDADE_VIZINHANCA}_upd{PROBABILIDADE_ATUALIZACAO:.2f}"
            f"_K{K_FERMI:g}_tau{TAU_FERMI:g}_{TOPOLOGIA_TABULEIRO}")


def matriz_payoff():
    return np.array([[1.0, 0.0], [b, EPSILON_P]], dtype=np.float64)


# ==============================================================================
# 2. GEOMETRIA
# ==============================================================================
def gerar_vizinhanca(tipo, profundidade):
    directions = []
    for dx in range(-profundidade, profundidade + 1):
        for dy in range(-profundidade, profundidade + 1):
            if dx == 0 and dy == 0:
                continue
            if tipo == "moore" and max(abs(dx), abs(dy)) <= profundidade:
                directions.append((dx, dy))
            elif tipo == "neumann" and abs(dx) + abs(dy) <= profundidade:
                directions.append((dx, dy))
    return np.array(directions, dtype=np.int32)


def arredondar(v):
    """Arredonda .5 sempre para cima. O round() do Python usa arredondamento bancário
    (round(12.5) = 12, round(11.5) = 12), o que fazia dois blocos com x diferentes
    caírem no mesmo lugar."""
    return int(np.floor(v + 0.5))


def coordenadas_para_matriz(config, L):
    """Retorna (linha_inicial, coluna_inicial, linha_centro, coluna_centro).
    Convenção: eixo x para a direita, eixo y para cima, (0, 0) no centro do tabuleiro.
    Bloco de lado ímpar: centrado em (x, y).
    Bloco de lado par: (x, y) é a célula superior-esquerda do quadrado 2x2 central,
    e o bloco se estende para a direita e para baixo. Deslocar x (ou y) em 1 sempre
    desloca o bloco em exatamente 1 sítio."""
    centro = (L - 1) / 2.0
    if config["tipo"] == "celulas":
        xs = [x for x, y in config["celulas"]]
        ys = [y for x, y in config["celulas"]]
        return None, None, centro - np.mean(ys), centro + np.mean(xs)
    if config["tipo"] == "cartesiano":
        x, y = config["x"], config["y"]
    else:
        th = np.radians(config["theta"])
        x, y = config["r"] * np.cos(th), config["r"] * np.sin(th)
    t = config["tamanho"]
    start_c = arredondar(centro + x - (t - 1) / 2.0)
    start_r = arredondar(centro - y - (t - 1) / 2.0)
    return start_r, start_c, centro - y, centro + x


def celulas_da_config(config, L):
    """Lista de (linha, coluna) ocupadas por um item de CONFIG_INVASORES/CONFIG_BURACOS.
    tipo "celulas": {"tipo": "celulas", "celulas": [(x1, y1), (x2, y2), ...]}
    (mesmo sistema de eixos do tipo cartesiano, uma célula por par)."""
    centro = (L - 1) / 2.0
    if config["tipo"] == "celulas":
        return [(arredondar(centro - y) % L, arredondar(centro + x) % L) for x, y in config["celulas"]]
    r0, c0, _, _ = coordenadas_para_matriz(config, L)
    t = config["tamanho"]
    return [((r0 + i) % L, (c0 + j) % L) for i in range(t) for j in range(t)]


def desenhar_bloco(arr, r0, c0, tamanho, valor, L):
    for r in range(r0, r0 + tamanho):
        for c in range(c0, c0 + tamanho):
            arr[r % L, c % L] = valor
    return arr


def inicializar_universo(L, p_buracos=None, p_coop=None):
    """Mesma sequência de números aleatórios do engine.py original (reprodutível)."""
    p_buracos = PROPORCAO_RANDOM_BURACOS if p_buracos is None else p_buracos
    p_coop = PROPORCAO_RANDOM_COOPERADORES if p_coop is None else p_coop
    grid = np.full((L, L), ESTRATEGIA_MAR, dtype=np.int32)
    hole = np.zeros((L, L), dtype=bool)
    paredes = np.zeros((L, L), dtype=bool)
    colonia = np.zeros((L, L), dtype=bool)

    if p_coop == 0.0:
        for inv in CONFIG_INVASORES:
            for r, c in celulas_da_config(inv, L):
                grid[r, c] = ESTRATEGIA_INVASOR
                colonia[r, c] = True
    for bur in CONFIG_BURACOS:
        for r, c in celulas_da_config(bur, L):
            hole[r, c] = True
    if p_buracos > 0:
        livres = np.where(~hole)
        qtd = min(int(round(L * L * p_buracos)), len(livres[0]))
        idx = np.random.choice(len(livres[0]), qtd, replace=False)
        hole[livres[0][idx], livres[1][idx]] = True
    grid[hole] = INDICE_BURACO
    if p_coop > 0.0:
        validas = np.where(~hole)
        qtd = int(round(len(validas[0]) * p_coop))
        idx = np.random.choice(len(validas[0]), qtd, replace=False)
        grid[validas[0][idx], validas[1][idx]] = ESTRATEGIA_INVASOR
    if TOPOLOGIA_TABULEIRO == "parede":
        d = PROFUNDIDADE_VIZINHANCA
        paredes[:d, :] = paredes[-d:, :] = True
        paredes[:, :d] = paredes[:, -d:] = True
        grid[paredes] = INDICE_BURACO
    colonia &= (grid == ESTRATEGIA_INVASOR)
    if p_coop > 0.0 or not CONFIG_INVASORES:
        return grid, paredes, None, None
    _, _, rc, cc = coordenadas_para_matriz(CONFIG_INVASORES[0], L)
    origem = (arredondar(rc) % L, arredondar(cc) % L)
    return grid, paredes, colonia, origem


# ==============================================================================
# 3. NÚCLEOS NUMBA
# ==============================================================================
@njit(parallel=True, cache=True)
def censo_e_payoff_numba(grid, L, R, S, T, P, directions):
    payoffs = np.zeros((L, L), dtype=np.float64)
    n_c_m = np.zeros((L, L), dtype=np.int32)
    n_d_m = np.zeros((L, L), dtype=np.int32)
    nv = directions.shape[0]
    for r in prange(L):
        for c in range(L):
            n_c = 0
            n_d = 0
            for i in range(nv):
                s = grid[(r + directions[i, 0]) % L, (c + directions[i, 1]) % L]
                if s == 0:
                    n_c += 1
                elif s == 1:
                    n_d += 1
            n_c_m[r, c] = n_c
            n_d_m[r, c] = n_d
            st = grid[r, c]
            if st == 0:
                payoffs[r, c] = n_c * R + n_d * S
            elif st == 1:
                payoffs[r, c] = n_c * T + n_d * P
    return n_c_m, n_d_m, payoffs


@njit(parallel=True, cache=True)
def escolha_racional_numba(grid, payoffs, L, K, TAU, directions, idx_viz, u_aceite):
    """K<=0: imitar o melhor com desempate por inércia.
    K>0 : regra de Fermi com vizinho idx_viz[r,c] e sorteio u_aceite[r,c] (gerados em NumPy)."""
    prop = np.copy(grid)
    nv = directions.shape[0]
    for r in prange(L):
        for c in range(L):
            if grid[r, c] == 2:
                continue
            if K <= 0.0:
                best_p = payoffs[r, c]
                minha = grid[r, c]
                best_s = minha
                for i in range(nv):
                    nr = (r + directions[i, 0]) % L
                    nc = (c + directions[i, 1]) % L
                    vs = grid[nr, nc]
                    if vs == 2:
                        continue
                    vp = payoffs[nr, nc]
                    if vp > best_p:
                        best_p = vp
                        best_s = vs
                    elif vp == best_p and vs == minha:
                        best_s = minha
                prop[r, c] = best_s
            else:
                i = idx_viz[r, c]
                nr = (r + directions[i, 0]) % L
                nc = (c + directions[i, 1]) % L
                vs = grid[nr, nc]
                if vs != 2:
                    ex = (payoffs[r, c] - payoffs[nr, nc] + TAU) / K
                    if ex > 50.0:
                        pt = 0.0
                    elif ex < -50.0:
                        pt = 1.0
                    else:
                        pt = 1.0 / (1.0 + np.exp(ex))
                    if u_aceite[r, c] < pt:
                        prop[r, c] = vs
    return prop


@njit(cache=True)
def rotular_toro(is_x, L, directions, labels, qr, qc, off_x, off_y, perc_x, perc_y):
    """Componentes conexos no toro (conectividade = vizinhança do jogo) + flags de percolação."""
    labels[:, :] = 0
    off_x[:, :] = 0
    off_y[:, :] = 0
    perc_x[:] = False
    perc_y[:] = False
    nv = directions.shape[0]
    cur = 0
    for r in range(L):
        for c in range(L):
            if is_x[r, c] and labels[r, c] == 0:
                cur += 1
                qr[0] = r
                qc[0] = c
                labels[r, c] = cur
                head = 0
                tail = 1
                while head < tail:
                    cr = qr[head]
                    cc = qc[head]
                    head += 1
                    for i in range(nv):
                        rr = cr + directions[i, 0]
                        ccc = cc + directions[i, 1]
                        nr = rr % L
                        nc = ccc % L
                        if not is_x[nr, nc]:
                            continue
                        sx = 1 if ccc >= L else (-1 if ccc < 0 else 0)
                        sy = 1 if rr >= L else (-1 if rr < 0 else 0)
                        dox = off_x[cr, cc] + sx
                        doy = off_y[cr, cc] + sy
                        if labels[nr, nc] == 0:
                            labels[nr, nc] = cur
                            off_x[nr, nc] = dox
                            off_y[nr, nc] = doy
                            qr[tail] = nr
                            qc[tail] = nc
                            tail += 1
                        else:
                            if off_x[nr, nc] != dox:
                                perc_x[cur] = True
                            if off_y[nr, nc] != doy:
                                perc_y[cur] = True
    return cur


@njit(cache=True)
def atualizar_colonia(labels, n_labels, colonia_prev, L, directions, keep):
    """Colônia(t) = componentes que tocam a colônia(t-1) dilatada por uma vizinhança."""
    for k in range(n_labels + 1):
        keep[k] = False
    nv = directions.shape[0]
    for r in range(L):
        for c in range(L):
            if colonia_prev[r, c]:
                lab = labels[r, c]
                if lab > 0:
                    keep[lab] = True
                for i in range(nv):
                    lab = labels[(r + directions[i, 0]) % L, (c + directions[i, 1]) % L]
                    if lab > 0:
                        keep[lab] = True
    col = np.zeros((L, L), dtype=np.bool_)
    for r in range(L):
        for c in range(L):
            lab = labels[r, c]
            if lab > 0 and keep[lab]:
                col[r, c] = True
    return col


@njit(cache=True)
def bfs_quimica(ativo, fontes_r, fontes_c, fontes_ux, fontes_uy, L, directions):
    """BFS multifonte no substrato: distância química e coordenadas desembrulhadas."""
    ell = np.full((L, L), -1, dtype=np.int32)
    ux = np.zeros((L, L), dtype=np.int32)
    uy = np.zeros((L, L), dtype=np.int32)
    qr = np.zeros(L * L, dtype=np.int32)
    qc = np.zeros(L * L, dtype=np.int32)
    tail = 0
    for k in range(fontes_r.shape[0]):
        r = fontes_r[k]
        c = fontes_c[k]
        if ativo[r, c] and ell[r, c] < 0:
            ell[r, c] = 0
            ux[r, c] = fontes_ux[k]
            uy[r, c] = fontes_uy[k]
            qr[tail] = r
            qc[tail] = c
            tail += 1
    head = 0
    nv = directions.shape[0]
    while head < tail:
        r = qr[head]
        c = qc[head]
        head += 1
        for i in range(nv):
            nr = (r + directions[i, 0]) % L
            nc = (c + directions[i, 1]) % L
            if ativo[nr, nc] and ell[nr, nc] < 0:
                ell[nr, nc] = ell[r, c] + 1
                ux[nr, nc] = ux[r, c] + directions[i, 1]
                uy[nr, nc] = uy[r, c] + directions[i, 0]
                qr[tail] = nr
                qc[tail] = nc
                tail += 1
    return ell, ux, uy


# ==============================================================================
# 4. MAPA ESTÁTICO DO SUBSTRATO E OBSERVÁVEIS
# ==============================================================================
def construir_substrato(grid, colonia, origem, L, directions):
    ativo = grid != INDICE_BURACO
    fr, fc = np.where(colonia)
    dx = fc - origem[1]
    dx = dx - L * np.round(dx / L)
    dy = fr - origem[0]
    dy = dy - L * np.round(dy / L)
    ell, ux, uy = bfs_quimica(ativo, fr.astype(np.int32), fc.astype(np.int32),
                              dx.astype(np.int32), dy.astype(np.int32), L, directions)
    return ativo, ell, ux, uy


def observaveis_colonia(colonia, ell, ux, uy, labels, tamanhos, perc_x, perc_y, L):
    nan = np.nan
    obs = dict(N_col=0, R2_origem=nan, rg2=nan, anisotropia=nan, msd_cm=nan, ell_max=nan,
               ell_medio=nan, alcance=nan, valido_geom=0, percola=0, n_comp_colonia=0,
               massa_maior_comp=0)
    sel = colonia & (ell >= 0)
    N = int(sel.sum())
    obs['N_col'] = N
    if N == 0:
        return obs
    X = ux[sel].astype(np.float64)
    Y = uy[sel].astype(np.float64)
    cx, cy = X.mean(), Y.mean()
    dX, dY = X - cx, Y - cy
    sxx, syy, sxy = np.mean(dX * dX), np.mean(dY * dY), np.mean(dX * dY)
    rg2 = sxx + syy
    obs['R2_origem'] = float(np.mean(X * X + Y * Y))
    obs['rg2'] = float(rg2)
    if rg2 > 0:
        lam = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]]))
        obs['anisotropia'] = float(1.0 - 4.0 * lam[0] * lam[1] / rg2 ** 2)
    obs['msd_cm'] = float(cx * cx + cy * cy)
    obs['_cx'], obs['_cy'] = float(cx), float(cy)
    e = ell[sel]
    obs['ell_max'] = int(e.max())
    obs['ell_medio'] = float(e.mean())
    alc = float(max(np.abs(X).max(), np.abs(Y).max()))
    obs['alcance'] = alc
    labs = np.unique(labels[sel])
    labs = labs[labs > 0]
    perc = bool(perc_x[labs].any() or perc_y[labs].any())
    obs['percola'] = int(perc)
    obs['valido_geom'] = int((alc < L // 2 - 1) and not perc)
    obs['n_comp_colonia'] = int(len(labs))
    obs['massa_maior_comp'] = int(tamanhos[labs].max())
    return obs


def checar_parametros():
    avisos = []
    z = len(gerar_vizinhanca(TIPO_VIZINHANCA, PROFUNDIDADE_VIZINHANCA))
    gaps = []
    for m in range(1, z + 1):
        for k in range(0, z + 1):
            g = abs(b * m - k)
            if g < 1e-12:
                avisos.append(f"b = {b} = {k}/{m} é um PONTO DE TRANSIÇÃO: empates exatos entre "
                              f"C com {k} vizinhos C e D com {m} vizinhos C. Com epsilon>0 o "
                              f"empate pende para D (na prática b = {k}/{m}+).")
            else:
                gaps.append(g)
    if gaps and EPSILON_P * z >= min(gaps):
        avisos.append(f"epsilon={EPSILON_P} é grande demais: epsilon*z={EPSILON_P * z:.3g} >= "
                      f"menor folga |b m - k| = {min(gaps):.3g}. O epsilon altera a ordem dos payoffs.")
    if PROFUNDIDADE_VIZINHANCA == 1:
        outro = "neumann" if TIPO_VIZINHANCA == "moore" else "moore"
        if abs(PROPORCAO_RANDOM_BURACOS - LIMIAR_BURACOS[outro]) < 1e-3:
            avisos.append(f"p = {PROPORCAO_RANDOM_BURACOS} é o limiar de percolação da vizinhança "
                          f"'{outro}', mas o jogo usa '{TIPO_VIZINHANCA}' "
                          f"(limiar = {LIMIAR_BURACOS[TIPO_VIZINHANCA]}). O substrato NÃO é crítico.")
    for a in avisos:
        print("  [AVISO] " + a)
    return avisos


# ==============================================================================
# 5. RENDERIZAÇÃO
# ==============================================================================
def renderizar_frame(grid, colonia, t, obs, L, pasta, origem):
    img = grid.copy()
    if colonia is not None:
        # Pinta mutantes isolados de azul claro (cor 3)
        img[(grid == ESTRATEGIA_INVASOR) & (~colonia)] = 3
        
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2])
    
    ax_board = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    
    # ==========================================================
    # 1. RENDERIZAÇÃO DO TABULEIRO (ESQUERDA)
    # ==========================================================
    cmap = mcolors.ListedColormap(COLORS)
    ax_board.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation='nearest', 
                    aspect='equal', extent=(-0.5, L-0.5, L-0.5, -0.5))
                    
    # Restaura as linhas de grade preta fininha no tabuleiro
    ax_board.set_xticks(np.arange(-0.5, L, 1))
    ax_board.set_yticks(np.arange(-0.5, L, 1))
    ax_board.set_xticklabels([])
    ax_board.set_yticklabels([])
    ax_board.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax_board.set_xlim(-0.5, L - 0.5)
    ax_board.set_ylim(L - 0.5, -0.5)
    
    if origem is not None:
        # Cruz amarela na Origem (Semente)
        ax_board.plot(origem[1], origem[0], 'y+', ms=18, mew=2)
        if obs['N_col'] > 0:
            cx = obs.get('_cx', np.nan)
            cy = obs.get('_cy', np.nan)
            if np.isfinite(cx):
                # X branco no Centro de Massa
                ax_board.plot((origem[1] + cx) % L, (origem[0] + cy) % L, 'wx', ms=14, mew=2)
                
    # ==========================================================
    # 2. PAINEL DE INFORMAÇÕES (DIREITA)
    # ==========================================================
    ax_text.axis('off') 
    fonte_padrao = 'serif'
    
    # Títulos
    ax_text.text(0.0, 0.95, "DILEMA DO PRISIONEIRO", 
                 fontsize=18, fontweight='bold', ha='left', va='top', family=fonte_padrao)
    ax_text.text(0.0, 0.90, f"Tempo (MCS): {t:05d}", 
                 fontsize=16, color='darkred', fontweight='bold', ha='left', va='top', family=fonte_padrao)
    
    # Bloco 1: Abundância Relativa
    texto_relativo = (
        "Abundância Relativa:\n"
        f"  • Cooperadores Ativos ($\\rho_C$): {obs.get('rho_c', 0)*100:05.2f}%\n"
        f"  • Sítios na Colônia ($N_{{col}}$): {obs['N_col']}\n"
        f"  • Fragmentos Conexos: {obs['n_comp_colonia']}\n"
    )
    ax_text.text(0.0, 0.75, texto_relativo, fontsize=14, ha='left', va='top', family=fonte_padrao)

    # Bloco 2: Cinemática da Colônia
    valido_str = "Sim" if obs['valido_geom'] else "Não (Saturado/Percolado)"
    texto_cinematica = (
        "Cinemática e Topologia:\n"
        f"  • $R^2$ (a partir da Origem): {obs['R2_origem']:.1f}\n"
        f"  • Raio de Giração ($R_g^2$): {obs['rg2']:.1f}\n"
        f"  • Alcance Máximo ($r_{{max}}$): {obs['alcance']:.1f}\n"
        f"  • Frente Química ($\\ell_{{max}}$): {obs['ell_max']}\n"
        f"  • Geometria Válida: {valido_str}\n"
    )
    ax_text.text(0.0, 0.55, texto_cinematica, fontsize=14, ha='left', va='top', family=fonte_padrao)

    # Bloco 3: Parâmetros Físicos
    texto_params = (
        "Parâmetros Físicos:\n"
        f"  • Tamanho ($L$): {L}$\\times${L}\n"
        f"  • Tentação ($b$): {b}\n"
        f"  • Vacâncias ($p$): {PROPORCAO_RANDOM_BURACOS*100:.2f}%\n"
        f"  • Mutação ($\\mu_C$): {TAXA_MUTACAO_C}\n"
        f"  • Mutação ($\\mu_D$): {TAXA_MUTACAO_D}\n"
    )
    ax_text.text(0.0, 0.30, texto_params, fontsize=12, ha='left', va='top', color='dimgrey', family=fonte_padrao)

    # Legendas
    nome_inv = 'Cooperador ($C$)' if ESTRATEGIA_INVASOR == INDICE_C else 'Desertor ($D$)'
    legend_elements = [
        Patch(facecolor=COLORS[INDICE_C], edgecolor='black',
              label='Cooperador ($C$)' + (' - colônia' if ESTRATEGIA_INVASOR == INDICE_C else '')),
        Patch(facecolor=COLORS[INDICE_D], edgecolor='black',
              label='Desertor ($D$)' + (' - colônia' if ESTRATEGIA_INVASOR == INDICE_D else '')),
        Patch(facecolor=COLORS[3], edgecolor='black', label=nome_inv + ' - fora da colônia'),
        Patch(facecolor=COLORS[INDICE_BURACO], edgecolor='black', label='Buraco (vazio)'),
    ]
    ax_text.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.0), 
                   fontsize=12, frameon=False, prop={'family': fonte_padrao, 'size': 12})
    
    os.makedirs(pasta, exist_ok=True)
    # Voltei o DPI de 100 para 120 para garantir a alta resolução
    plt.savefig(os.path.join(pasta, f"frame_{t:06d}_L{L}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)


def sanitizar_csv_resume(caminho_csv, n_col):
    with open(caminho_csv, 'r+', newline='') as f:
        linhas = f.readlines()
        if linhas and linhas[-1].count(';') + 1 != n_col:
            print(f"    [!] Última linha truncada em {os.path.basename(caminho_csv)} — removendo.")
            f.seek(0)
            f.truncate()
            f.writelines(linhas[:-1])


def _replace_atomico(tmp, dst):
    try:
        os.replace(tmp, dst)
    except PermissionError:
        time.sleep(0.1)
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(tmp, dst)


# ==============================================================================
# 6. LAÇO PRINCIPAL
# ==============================================================================
def executar_simulacao():
    print(f"[{time.strftime('%H:%M:%S')}] engine_v2 | experimento: {nome_experimento()}")
    checar_parametros()
    raiz = os.path.join(BASE_DIR, nome_experimento())
    cam_dados = os.path.join(raiz, 'data')
    cam_chk = os.path.join(raiz, 'checkpoints')
    cam_frames = os.path.join(raiz, 'frames')
    os.makedirs(cam_dados, exist_ok=True)
    os.makedirs(cam_chk, exist_ok=True)

    if isinstance(CONFIG_SEMENTES, int):
        np.random.seed()
        sementes = np.random.randint(1, 9999999, size=CONFIG_SEMENTES).tolist()
    else:
        sementes = list(CONFIG_SEMENTES)

    PM = matriz_payoff()
    directions = gerar_vizinhanca(TIPO_VIZINHANCA, PROFUNDIDADE_VIZINHANCA)
    nv = directions.shape[0]
    deterministico = (TAXA_MUTACAO_C == 0 and TAXA_MUTACAO_D == 0 and K_FERMI <= 0
                      and PROBABILIDADE_ATUALIZACAO >= 1.0)
    rastrear = PROPORCAO_RANDOM_COOPERADORES == 0.0 and len(CONFIG_INVASORES) > 0

    for L in TAMANHOS_REDE:
        pasta_L = os.path.join(cam_dados, f'L_{L}')
        os.makedirs(pasta_L, exist_ok=True)
        with open(os.path.join(pasta_L, 'sementes_usadas.txt'), 'w') as f:
            f.write(f"Tamanho do Ensemble: {len(sementes)}\nSementes: {sementes}\n")
        labels = np.zeros((L, L), np.int32)
        qr = np.zeros(L * L, np.int32)
        qc = np.zeros(L * L, np.int32)
        offx = np.zeros((L, L), np.int32)
        offy = np.zeros((L, L), np.int32)
        percx = np.zeros(L * L + 1, np.bool_)
        percy = np.zeros(L * L + 1, np.bool_)
        keep = np.zeros(L * L + 1, np.bool_)
        dummy_i = np.zeros((1, 1), np.int32)
        dummy_u = np.zeros((1, 1), np.float64)

        for n_sim, semente in enumerate(sementes):
            print(f"\n---> L={L} | rodada {n_sim + 1}/{len(sementes)} | semente {semente}")
            csv_path = os.path.join(pasta_L, f'log_semente_{semente}.csv')
            chk_path = os.path.join(cam_chk, f'chk_L{L}_semente_{semente}.npz')
            sub_path = os.path.join(pasta_L, f'substrato_semente_{semente}.npz')
            passo0 = 0
            if CONTINUAR_SIMULACAO and os.path.exists(chk_path) and os.path.exists(csv_path):
                chk = np.load(chk_path, allow_pickle=True)
                passo0 = int(chk['step'])
                if passo0 >= STEPS:
                    print("    ✔ já concluída.")
                    continue
                grid = chk['grid']
                paredes = chk['paredes_mask']
                colonia = chk['colonia'] if rastrear else None
                np.random.set_state(tuple(chk['rng_state']))
                chk.close()
                if rastrear:
                    sub = np.load(sub_path)
                    ativo, ell, ux, uy = sub['ativo'], sub['ell'], sub['ux'], sub['uy']
                    origem = tuple(sub['origem'])
                else:
                    ativo = grid != INDICE_BURACO
                    origem = None
                sanitizar_csv_resume(csv_path, len(COLUNAS))
                modo = 'a'
            else:
                np.random.seed(semente)
                grid, paredes, colonia, origem = inicializar_universo(L)
                ativo = grid != INDICE_BURACO
                if rastrear:
                    ativo, ell, ux, uy = construir_substrato(grid, colonia, origem, L, directions)
                    np.savez_compressed(sub_path, ativo=ativo, ell=ell, ux=ux, uy=uy,
                                        origem=np.array(origem), colonia0=colonia)
                modo = 'w'

            n_ativos = int(ativo.sum())
            total_uteis = L * L - int(paredes.sum())
            hist = deque(maxlen=PERIODO_MAX)
            ult_linhas = deque(maxlen=PERIODO_MAX)
            buffer = []
            with open(csv_path, modo, newline='') as fcsv:
                wr = csv.writer(fcsv, delimiter=';')
                if modo == 'w':
                    wr.writerow(COLUNAS)
                    # ---- estado inicial (t = 0): registrado antes de qualquer atualização ----
                    n_c = int(np.sum(grid == INDICE_C))
                    n_d = int(np.sum(grid == INDICE_D))
                    row = dict(step=0, frac_c=n_c / total_uteis, frac_d=n_d / total_uteis,
                               porosidade_p=float(np.sum((grid == INDICE_BURACO) & ~paredes)) / total_uteis,
                               rho_c=n_c / n_ativos if n_ativos else np.nan,
                               atividade=0, n_mutacoes=0)
                    if rastrear:
                        is_x = grid == ESTRATEGIA_INVASOR
                        nlab = rotular_toro(is_x, L, directions, labels, qr, qc, offx, offy, percx, percy)
                        tam = np.bincount(labels.ravel(), minlength=nlab + 1)
                        tam[0] = 0
                        obs = observaveis_colonia(colonia, ell, ux, uy, labels, tam, percx, percy, L)
                        obs['n_clusters_estrategia'] = int(nlab)
                    else:
                        obs = {k: np.nan for k in COLUNAS[7:]}
                        obs['N_col'] = 0
                    row.update(obs)
                    linha = [row[k] for k in COLUNAS]
                    buffer.append(linha)
                    ult_linhas.append(linha)
                    if PARADA_ANTECIPADA and deterministico:
                        assinatura = n_c
                        hist.append((assinatura, grid.copy(), None if colonia is None else colonia.copy()))
                    if GERAR_FRAMES:
                        renderizar_frame(grid, colonia, 0, row, L,
                                         os.path.join(cam_frames, f'semente_{semente}'), origem)
                t = passo0
                while t < STEPS:
                    t += 1
                    g_ant = grid
                    _, _, pay = censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0], PM[1, 1],
                                                     directions)
                    if K_FERMI > 0:
                        idx_viz = np.random.randint(0, nv, size=(L, L)).astype(np.int32)
                        u_ac = np.random.rand(L, L)
                    else:
                        idx_viz, u_ac = dummy_i, dummy_u
                    prop_rac = escolha_racional_numba(grid, pay, L, K_FERMI, TAU_FERMI, directions,
                                                      idx_viz, u_ac)
                    rnd = np.random.rand(L, L)
                    mut = ((grid == INDICE_C) & (rnd < TAXA_MUTACAO_C)) | \
                          ((grid == INDICE_D) & (rnd < TAXA_MUTACAO_D))
                    prop = np.where(mut, (prop_rac + 1) % NUM_STRATEGIES, prop_rac)
                    if PROBABILIDADE_ATUALIZACAO < 1.0:
                        cel = np.where(grid != INDICE_BURACO)
                        q = int(round(len(cel[0]) * PROBABILIDADE_ATUALIZACAO))
                        sel = np.zeros((L, L), dtype=bool)
                        if q > 0:
                            ii = np.random.choice(len(cel[0]), q, replace=False)
                            sel[cel[0][ii], cel[1][ii]] = True
                        prop = np.where(sel, prop, grid)
                        mut &= sel
                    prop[~ativo] = INDICE_BURACO
                    prop[paredes] = INDICE_BURACO
                    grid = prop.astype(np.int32)

                    n_c = int(np.sum(grid == INDICE_C))
                    n_d = int(np.sum(grid == INDICE_D))
                    row = dict(step=t, frac_c=n_c / total_uteis, frac_d=n_d / total_uteis,
                               porosidade_p=float(np.sum((grid == INDICE_BURACO) & ~paredes)) / total_uteis,
                               rho_c=n_c / n_ativos if n_ativos else np.nan,
                               atividade=int(np.sum(grid != g_ant)),
                               n_mutacoes=int(np.sum(mut & ativo)))
                    if rastrear:
                        is_x = grid == ESTRATEGIA_INVASOR
                        nlab = rotular_toro(is_x, L, directions, labels, qr, qc, offx, offy, percx, percy)
                        colonia = atualizar_colonia(labels, nlab, colonia, L, directions, keep)
                        tam = np.bincount(labels.ravel(), minlength=nlab + 1)
                        tam[0] = 0
                        obs = observaveis_colonia(colonia, ell, ux, uy, labels, tam, percx, percy, L)
                        obs['n_clusters_estrategia'] = int(nlab)
                    else:
                        obs = {k: np.nan for k in COLUNAS[7:]}
                        obs['N_col'] = 0
                    row.update(obs)
                    linha = [row[k] for k in COLUNAS]
                    buffer.append(linha)
                    ult_linhas.append(linha)

                    # --- parada antecipada (ciclo determinístico) ---
                    periodo = 0
                    if PARADA_ANTECIPADA and deterministico:
                        assinatura = n_c
                        for k_back, (assin, gh, ch) in enumerate(reversed(hist), start=1):
                            if assin == assinatura and np.array_equal(gh, grid) and \
                               (colonia is None or np.array_equal(ch, colonia)):
                                periodo = k_back
                                break
                        hist.append((assinatura, grid.copy(), None if colonia is None else colonia.copy()))
                    if periodo:
                        base = list(ult_linhas)[-periodo:]
                        for s in range(t + 1, STEPS + 1):
                            ln = list(base[(s - t - 1) % periodo])
                            ln[0] = s
                            buffer.append(ln)
                        print(f"    ciclo de período {periodo} em t={t}: série completada até {STEPS}.")
                        # avança o estado para a fase correta de t = STEPS (checkpoint e frame coerentes)
                        j = (periodo - (STEPS - t) % periodo) % periodo
                        _, g_h, c_h = hist[-1 - j]
                        grid = g_h.copy()
                        if colonia is not None:
                            colonia = c_h.copy()
                        row = dict(zip(COLUNAS, buffer[-1]))
                        t = STEPS

                    if t % CHECKPOINT_SKIP == 0 or t == STEPS:
                        wr.writerows(buffer)
                        buffer.clear()
                        fcsv.flush()
                        tmp = chk_path.replace('.npz', '_temp.npz')
                        np.savez(tmp, grid=grid, paredes_mask=paredes,
                                 colonia=colonia if colonia is not None else np.zeros(1, bool),
                                 step=t, rng_state=np.array(np.random.get_state(), dtype=object))
                        _replace_atomico(tmp, chk_path)
                        print(f"    t={t:6d} | rho_C={row['rho_c']:.3f} | N_col={row['N_col']} | "
                              f"R2_origem={row['R2_origem']:.1f} | válido={row['valido_geom']}")
                    if GERAR_FRAMES and t % FRAME_SKIP == 0:
                        renderizar_frame(grid, colonia, t, row, L,
                                         os.path.join(cam_frames, f'semente_{semente}'), origem)
        print(f"✔ L={L} concluído.")
    print(f"\n[{time.strftime('%H:%M:%S')}] fim.")


if __name__ == "__main__":
    executar_simulacao()