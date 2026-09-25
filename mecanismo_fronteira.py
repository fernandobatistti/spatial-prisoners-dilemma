"""
mecanismo_fronteira.py — Experimento C: por que só alguns limiares k/m importam?

Coloque este arquivo na MESMA PASTA do engine_v2.py e rode:

    python mecanismo_fronteira.py               # roda (ou continua) o catálogo
    python mecanismo_fronteira.py --so-plotar   # refaz figuras e tabelas dos JSON
"""
import json
import os
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import engine_v2 as motor

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Um b por patamar do Experimento A, com o p logo ACIMA do respectivo p*_c.
CASOS = [
    (1.07,  [0.38, 0.42]),
    (1.29,  [0.38, 0.42]),
    (1.367, [0.22, 0.26]),
    (1.45,  [0.24, 0.28]),
    (1.55,  [0.16, 0.20]),
    (1.633, [0.18, 0.22]),
]
TAMANHOS = [101]
N_SEMENTES = {101: 100, 201: 40}
SEMENTE_BASE = 2000
T_MAX = 3000
LADO_COLONIA = 9
EPSILON = 1e-5
BLOCO = 20
VERSAO = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, f"mecanismo_v{VERSAO}_moore1_eps{EPSILON:g}_n0{LADO_COLONIA}")

# Razões k/m possíveis no intervalo (1, 2) com k, m <= 8
RAZOES = sorted({Fraction(k, m) for m in range(1, 9) for k in range(1, 9)
                 if 1 < Fraction(k, m) < 2})


def configurar_motor(b):
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
    motor.b = b


def meta():
    return dict(versao=VERSAO, semente_base=SEMENTE_BASE, t_max=T_MAX,
                lado_colonia=LADO_COLONIA, epsilon=EPSILON, vizinhanca="moore1",
                topologia="toroide")


# ==============================================================================
# O CATÁLOGO DE UMA CONFIGURAÇÃO
# ==============================================================================
def vizinhos_deslocados(mascara, directions):
    return [np.roll(np.roll(mascara, int(d[0]), axis=0), int(d[1]), axis=1)
            for d in directions]


def campos_KM(grid, ativo, directions):
    C, D = motor.INDICE_C, motor.INDICE_D
    eh_c = (grid == C) & ativo
    eh_d = (grid == D) & ativo
    n_c = np.zeros(grid.shape, np.int16)
    for m in vizinhos_deslocados(eh_c.astype(np.int8), directions):
        n_c += m
    K = np.where(eh_c, n_c, -1).astype(np.int16)
    M = np.where(eh_d, n_c, 0).astype(np.int16)
    for d in directions:
        viz_nc = np.roll(np.roll(n_c, int(d[0]), axis=0), int(d[1]), axis=1)
        viz_c = np.roll(np.roll(eh_c, int(d[0]), axis=0), int(d[1]), axis=1)
        viz_d = np.roll(np.roll(eh_d, int(d[0]), axis=0), int(d[1]), axis=1)
        K = np.where(viz_c & (viz_nc > K), viz_nc, K)
        M = np.where(viz_d & (viz_nc > M), viz_nc, M)
    return K, M, eh_c, eh_d


def razoes_da_fronteira(grid, ativo, comp, directions):
    K, M, eh_c, eh_d = campos_KM(grid, ativo, directions)
    zona = comp & (K >= 0) & (M > 0)
    ks = K[zona].astype(int).tolist()
    ms = M[zona].astype(int).tolist()
    es = eh_c[zona].astype(int).tolist()
    return list(zip(ks, ms, es))


def uma_rodada(L, p, b, semente, directions):
    np.random.seed(semente)
    grid, paredes, colonia, origem = motor.inicializar_universo(L, p_buracos=p, p_coop=0.0)
    ativo, ell, ux, uy = motor.construir_substrato(grid, colonia, origem, L, directions)
    comp = ell >= 0
    alvo = L // 2 - 1
    raio = np.maximum(np.abs(ux), np.abs(uy))
    PM = motor.matriz_payoff()
    C = motor.INDICE_C
    di = np.zeros((1, 1), np.int32)
    du = np.zeros((1, 1))
    vistos = set()
    desfecho = 'indeciso'
    decisivas = Counter()
    for t in range(1, T_MAX + 1):
        K, M, eh_c, eh_d = campos_KM(grid, ativo, directions)
        _, _, pay = motor.censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0], PM[1, 1],
                                               directions)
        novo = motor.escolha_racional_numba(grid, pay, L, 0.0, 0.0, directions, di, du)
        novo[~ativo] = motor.INDICE_BURACO
        mudou = comp & (novo != grid) & ativo & (M > 0) & (K >= 0)
        if mudou.any():
            for kk, mm in zip(K[mudou].astype(int).tolist(), M[mudou].astype(int).tolist()):
                decisivas[(kk, mm)] += 1
        grid = novo
        X = (grid == C) & comp
        if X.any() and raio[X].max() >= alvo:
            desfecho = 'invadiu'
            break
        h = hash(grid.tobytes())
        if not X.any() or h in vistos:
            desfecho = 'ancorou'
            break
        vistos.add(h)
    pares = razoes_da_fronteira(grid, ativo, comp, directions) if desfecho == 'ancorou' else []
    return desfecho, t, pares, [[k, m, n] for (k, m), n in sorted(decisivas.items())]


# ==============================================================================
# ARQUIVOS
# ==============================================================================
def arquivo(b):
    return os.path.join(PASTA, f"mecanismo_b{b:.4f}.json")


def carregar(b):
    if not os.path.exists(arquivo(b)):
        return {"_meta": meta()}
    d = json.load(open(arquivo(b)))
    if d.get("_meta") != meta():
        raise SystemExit(f"[ERRO] {os.path.basename(arquivo(b))} tem outros parâmetros. ")
    return d


def gravar(b, dados):
    os.makedirs(PASTA, exist_ok=True)
    tmp = arquivo(b) + '.tmp'
    json.dump(dados, open(tmp, 'w'), indent=1)
    os.replace(tmp, arquivo(b))


def rodar():
    os.makedirs(PASTA, exist_ok=True)
    directions = motor.gerar_vizinhanca("moore", 1)
    t0 = time.time()
    for b, ps in CASOS:
        configurar_motor(b)
        print(f"\n=== b = {b} ===")
        dados = carregar(b)
        for L in TAMANHOS:
            for p in ps:
                chave = f"L{L}_p{p:.4f}"
                ent = dados.get(chave) or dict(L=L, p=p, b=b, desfecho=[], t=[], pares=[],
                                               decisivas=[])
                feitas = len(ent['desfecho'])
                if feitas >= N_SEMENTES[L]:
                    continue
                for s in range(feitas, N_SEMENTES[L]):
                    d, t, pares, dec = uma_rodada(L, p, b, SEMENTE_BASE + s, directions)
                    ent['desfecho'].append(d)
                    ent['t'].append(t)
                    ent['pares'].append(pares)
                    ent['decisivas'].append(dec)
                    if (s + 1) % BLOCO == 0 or s + 1 == N_SEMENTES[L]:
                        dados[chave] = ent
                        gravar(b, dados)
                anc = sum(x == 'ancorou' for x in ent['desfecho'])
                nsit = sum(len(pp) for pp in ent['pares'])
                print(f"  L={L:4d} p={p:.2f}: {anc}/{len(ent['desfecho'])} ancoraram | "
                      f"{nsit} sítios catalogados ({(time.time() - t0) / 60:.1f} min)")
        gravar(b, dados)
    resumo()


# ==============================================================================
# ANÁLISE E PLOTAGEM VETORIZADA
# ==============================================================================
def coletar(b):
    dados = carregar(b)
    sitio_c, sitio_d = Counter(), Counter()
    dec_sitios, dec_real = Counter(), Counter()
    n_real = n_sitios = n_dec = 0
    for k, ent in dados.items():
        if k.startswith('_'):
            continue
        for pares, dec in zip(ent['pares'], ent.get('decisivas', [])):
            n_real += 1
            for item in pares:
                K, M, eh_c = item
                if M <= 0 or K < 0:
                    continue
                n_sitios += 1
                r = Fraction(int(K), int(M))
                (sitio_c if eh_c else sitio_d)[r] += 1
            presentes = set()
            for K, M, n in dec:
                if M <= 0 or K < 0:
                    continue
                r = Fraction(int(K), int(M))
                dec_sitios[r] += n
                n_dec += n
                presentes.add(r)
            for r in presentes:
                dec_real[r] += 1
    return sitio_c, sitio_d, dec_sitios, dec_real, n_real, n_sitios, n_dec


def formatar_virgula(x, pos):
    """Substitui pontos por vírgulas nos decimais dos eixos y."""
    if isinstance(x, float) and abs(x) > 0 and abs(x - round(x)) < 1e-10:
        x = round(x)
    return f"{x:g}".replace('.', '{,}')


def tipografia():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.constrained_layout.use': True
    })

def salvar(fig, nome):
    fig.savefig(os.path.join(PASTA, f'{nome}.pdf'), format='pdf', transparent=True, dpi=600)
    plt.close(fig)
    print(f"  figura: {nome}.pdf gerada com sucesso!")


def resumo():
    tipografia()
    tudo = {b: coletar(b) for b, _ in CASOS if os.path.exists(arquivo(b))}
    if not tudo:
        return
    bs = list(tudo)
    
    # --------------------------------------------------------------------------
    # Gerações de Textos de Tabela Original (Preservados e Identados)
    # --------------------------------------------------------------------------
    linhas = []
    cab = f"{'razão':>7} " + " ".join(f"{'b=' + str(b):>15}" for b in bs)
    linhas += ["=" * len(cab),
               "RAZÕES CRÍTICAS DECISIVAS AO LONGO DA TRAJETÓRIA",
               "para cada razão: % das decisões de mudança | % das realizações em que decidiu",
               "=" * len(cab), cab, "-" * len(cab)]
    for r in RAZOES + [Fraction(2, 1)]:
        cels = []
        for b in bs:
            _, _, ds, dr, nreal, _, ndec = tudo[b]
            cels.append(f"{100 * ds[r] / max(ndec, 1):6.2f}% {100 * dr[r] / max(nreal, 1):5.1f}%")
        if any(float(c.split('%')[0]) > 0.005 for c in cels):
            linhas.append(f"{r.numerator:>3}/{r.denominator:<3} " + " ".join(f"{c:>15}" for c in cels))
    linhas.append("-" * len(cab))
    linhas.append(f"{'decisões':>8} " + " ".join(f"{tudo[b][6]:>15}" for b in bs))
    linhas.append(f"{'realiz.':>8} " + " ".join(f"{tudo[b][4]:>15}" for b in bs))
    linhas.append("=" * len(cab))

    cab2 = f"{'razão':>7} " + " ".join(f"{'b=' + str(b):>15}" for b in bs)
    linhas += ["", "=" * len(cab2),
               "RAZÕES PRESENTES NA ZONA DE FRONTEIRA DO ESTADO ANCORADO",
               "para cada razão: % dos sítios que eram D (avanço) | % dos que eram C (retirada)",
               "=" * len(cab2), cab2, "-" * len(cab2)]
    for r in RAZOES + [Fraction(2, 1)]:
        cels = []
        for b in bs:
            sc, sd, _, _, _, nsit, _ = tudo[b]
            cels.append(f"{100 * sd[r] / max(nsit, 1):6.2f}% {100 * sc[r] / max(nsit, 1):6.2f}%")
        if any(float(c.split('%')[0]) > 0.005 or float(c.split('%')[1]) > 0.005 for c in cels):
            linhas.append(f"{r.numerator:>3}/{r.denominator:<3} " + " ".join(f"{c:>15}" for c in cels))
    linhas.append("=" * len(cab2))

    txt = "\n".join(linhas)
    print("\n" + txt)
    os.makedirs(PASTA, exist_ok=True)
    open(os.path.join(PASTA, 'mecanismo_tabela.txt'), 'w').write(txt + "\n")
    json.dump({str(b): {str(r): [tudo[b][2][r], tudo[b][3][r], tudo[b][1][r], tudo[b][0][r]]
                        for r in RAZOES + [Fraction(2, 1)]}
               | {'_totais': [tudo[b][4], tudo[b][5], tudo[b][6]]} for b in bs},
              open(os.path.join(PASTA, 'mecanismo_resumo.json'), 'w'), indent=1)

    # --------------------------------------------------------------------------
    # Plotagem Vetorizada do Catálogo (Gráfico de Barras)
    # --------------------------------------------------------------------------
    rs = RAZOES
    x = np.arange(len(rs))
    larg = 0.8 / max(len(bs), 1)
    
    # Nossa paleta termodinâmica de cores
    cores = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(bs)))
    
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    
    for i, (b, cor) in enumerate(zip(bs, cores)):
        _, _, ds, dr, nreal, _, ndec = tudo[b]
        
        # Formatando legendas com vírgula para pt-br
        str_b = f"{b:.3f}".replace('.', '{,}')
        
        axs[0].bar(x + i * larg, [100 * ds[r] / max(ndec, 1) for r in rs], 
                   width=larg, color=cor, edgecolor='black', lw=0.4, label=rf'$b = {str_b}$')
                   
        axs[1].bar(x + i * larg, [100 * dr[r] / max(nreal, 1) for r in rs], 
                   width=larg, color=cor, edgecolor='black', lw=0.4, label=rf'$b = {str_b}$')
                   
    # Eixo Y e Grid
    for ax in axs:
        ax.grid(True, axis='y', ls='-', alpha=0.2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_virgula))
        
    axs[0].set_ylabel(r'\% das decisões de mudança')
    axs[1].set_ylabel(r'\% das realizações afetadas')
    
    # Eixo X com Frações Matemáticas Verticais
    axs[1].set_xticks(x + 0.4 - larg / 2)
    # Usando o mathmode do LaTeX com rotação sutil para não engavetar denominadores
    axs[1].set_xticklabels([rf'$\frac{{{r.numerator}}}{{{r.denominator}}}$' for r in rs], rotation=0)
    axs[1].set_xlabel(r'Razão crítica $r = K/M$ da fronteira', labelpad=10)
    
    axs[0].legend(ncol=len(bs), loc='upper left', frameon=False, bbox_to_anchor=(0.1, 1.10))
    axs[0].set_title(r'Catálogo de razões que decidem o travamento de uma frente', pad=25)
    
    salvar(fig, 'mecanismo_1_razoes')


if __name__ == "__main__":
    if "--so-plotar" in sys.argv:
        resumo()
    else:
        rodar()