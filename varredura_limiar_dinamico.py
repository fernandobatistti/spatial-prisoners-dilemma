"""
varredura_limiar_dinamico.py — Experimento A: a invasão determinística percola?

Coloque este arquivo na MESMA PASTA do engine_v2.py e rode:

    python varredura_limiar_dinamico.py              # roda (ou continua) a varredura
    python varredura_limiar_dinamico.py --so-plotar  # só refaz figuras e resumo a partir dos JSON

Protocolo (Seção 3.4.1 da dissertação)
--------------------------------------
mu = 0, K = 0, atualização síncrona: a dinâmica é determinística e a única aleatoriedade
é o sorteio do substrato. Para cada (b, L, p, semente):
  1. gera o substrato e uma colônia de LADO_COLONIA x LADO_COLONIA cooperadores no centro
     (os buracos também atingem o bloco inicial);
  2. evolui até (i) um cooperador alcançar a distância L/2 - 1 da origem, medida nas
     coordenadas desembrulhadas -> "alcançou"; (ii) o estado repetir um estado anterior
     (ciclo de QUALQUER período, detectado por hash) ou a colônia morrer -> "travou";
     (iii) T_MAX passos -> "indeciso" (transiente longo; deve ser raríssimo).
Grava, por semente: o desfecho, a fração do aglomerado da semente ocupada no fim e o
instante da decisão. Com isso calcula P_alcance (intervalo de Wilson), o território final
e estima o limiar dinâmico de duas formas: cruzamento das curvas de território entre os dois
maiores L, e o p em que P_alcance = 1/2 para cada L (com bandas por bootstrap nas sementes).

Sementes: a realização s usa a semente SEMENTE_BASE + s, a MESMA para todos os b e p.
  - Entre valores de p, os substratos ficam aninhados (buracos de p menor estão contidos nos
    de p maior), o que deixa as curvas suaves; pontos vizinhos em p NÃO são independentes.
  - Entre valores de b, o substrato é idêntico: se dois b estão no mesmo intervalo entre
    limiares k/m, os resultados devem ser IDÊNTICOS semente a semente. O script verifica.

Retomada: os resultados são gravados por semente, em blocos. Interromper e rodar de novo
continua de onde parou; aumentar N_SEMENTES só roda as sementes que faltam.
"""
import hashlib
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine_v2 as motor

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Um b no interior de cada intervalo entre limiares k/m abaixo de 5/3 (os mesmos da
# validação), mais 1.24 (mesmo intervalo de 1.225: teste exato de constância por partes)
# e 1.70 (acima de 5/3: controle, não deve haver invasão em nenhum p).
# Ordem: os mais informativos primeiro; se a varredura for interrompida, já há resultado útil.
B_VALS = [1.225, 1.24, 1.29, 1.367, 1.45, 1.55, 1.633, 1.70, 1.07, 1.155, 1.183]


def _grade(a, b, passo=0.02):
    return [round(x, 4) for x in np.arange(a, b + 1e-9, passo)]


# Grade de p (fração de buracos) por valor de b, centrada no piloto (L = 51 e 101), que deu:
#   b <= 1.24 : cruzamento do território ~0.33 ; P_alcance = 1/2 em ~0.42 (L=51), ~0.39 (L=101)
#   b = 1.29  : ~0.32 ; ~0.42 / ~0.39
#   b = 1.367 e 1.45 : ~0.16 ; ~0.30 / ~0.25
#   b = 1.55  : ~0.10 ; ~0.18 / ~0.16
#   b = 1.633 : território não cruza (regime de coexistência) ; ~0.19 / ~0.16
#   b = 1.70  : nenhuma invasão, nem em p = 0
P_POR_B = {
    1.07: _grade(0.28, 0.44), 1.155: _grade(0.28, 0.44), 1.183: _grade(0.28, 0.44),
    1.225: _grade(0.28, 0.44), 1.24: _grade(0.28, 0.44),
    1.29: _grade(0.26, 0.44),
    1.367: _grade(0.12, 0.32), 1.45: _grade(0.12, 0.32),
    1.55: _grade(0.06, 0.22),
    1.633: _grade(0.08, 0.24),
    1.70: [0.0, 0.1, 0.2],
}
P_PADRAO = _grade(0.0, 0.50, 0.05)   # para um b fora da tabela: grade larga, exploratória

TAMANHOS = [51, 101, 201, 401]
N_SEMENTES = {51: 400, 101: 200, 201: 100, 401: 50}
SEMENTE_BASE = 2000
T_MAX = 3000
LADO_COLONIA = 9
EPSILON = 1e-5               # abaixo de epsilon_max para todos os b acima
BLOCO = 25                   # grava o JSON a cada BLOCO sementes
USAR_LATEX = True            # cai para mathtext se o LaTeX falhar

VERSAO = 3                   # muda o nome da pasta: resultados antigos não são misturados

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE_DIR, f"limiar_dinamico_v{VERSAO}_moore1_eps{EPSILON:g}_n0{LADO_COLONIA}")

LIMIAR_GEOMETRICO = 0.592746  # Moore, em fração de buracos
LIMIARES_KM = sorted({Fraction(k, m) for m in range(1, 9) for k in range(9) if 1 < Fraction(k, m) < 2})


def configurar_motor(b):
    """Fixa TODOS os parâmetros do engine usados aqui, sem depender do que estiver no arquivo."""
    motor.TIPO_VIZINHANCA = "moore"
    motor.PROFUNDIDADE_VIZINHANCA = 1
    motor.TOPOLOGIA_TABULEIRO = "toroide"
    motor.ESTRATEGIA_MAR = motor.INDICE_D          # mar de desertores
    motor.ESTRATEGIA_INVASOR = motor.INDICE_C      # colônia de cooperadores
    motor.CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": LADO_COLONIA}]
    motor.CONFIG_BURACOS = []
    motor.PROPORCAO_RANDOM_COOPERADORES = 0.0
    motor.TAXA_MUTACAO_C = motor.TAXA_MUTACAO_D = 0.0
    motor.K_FERMI = 0.0
    motor.PROBABILIDADE_ATUALIZACAO = 1.0
    motor.EPSILON_P = EPSILON
    motor.b = b


def meta():
    return dict(versao=VERSAO, semente_base=SEMENTE_BASE, t_max=T_MAX, deteccao_ciclos='hash',
                lado_colonia=LADO_COLONIA, epsilon=EPSILON, vizinhanca="moore1", topologia="toroide")


def grade_p(b):
    return P_POR_B.get(b, P_PADRAO)


# ==============================================================================
# SIMULAÇÃO
# ==============================================================================
def _hash(grid):
    """Impressão digital de 128 bits do estado (colisão acidental: ~1e-38 por par)."""
    return hashlib.blake2b(grid.astype(np.int8).tobytes(), digest_size=16).digest()


def uma_rodada(L, p, semente, directions):
    np.random.seed(semente)
    grid, paredes, colonia, origem = motor.inicializar_universo(L, p_buracos=p, p_coop=0.0)
    ativo, ell, ux, uy = motor.construir_substrato(grid, colonia, origem, L, directions)
    comp = ell >= 0
    n_comp = max(int(comp.sum()), 1)
    borda = comp & (np.maximum(np.abs(ux), np.abs(uy)) >= L // 2 - 1)
    PM = motor.matriz_payoff()
    inv = motor.ESTRATEGIA_INVASOR
    di = np.zeros((1, 1), np.int32)
    du = np.zeros((1, 1))
    X = (grid == inv) & comp
    vistos = {_hash(grid)}
    res = 'indeciso'
    t = 0
    for t in range(1, T_MAX + 1):
        _, _, pay = motor.censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0], PM[1, 1], directions)
        grid = motor.escolha_racional_numba(grid, pay, L, 0.0, 0.0, directions, di, du)
        grid[~ativo] = motor.INDICE_BURACO
        X = (grid == inv) & comp
        if (X & borda).any():
            res = 'alcancou'
            break
        if not X.any():
            res = 'travou'
            break
        h = _hash(grid)
        if h in vistos:          # estado já visto: a dinâmica determinística entrou num ciclo
            res = 'travou'
            break
        vistos.add(h)
    codigo = {'travou': 0, 'alcancou': 1, 'indeciso': 2}[res]
    return codigo, round(float(X.sum()) / n_comp, 6), t


# ==============================================================================
# ESTATÍSTICA
# ==============================================================================
def wilson(k, n, z=1.0):
    """Intervalo de Wilson (z = 1: ~68%). Não colapsa em P = 0 ou 1."""
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    # em P = 0 ou 1 o arredondamento pode deixar c - h > P ou c + h < P por ~1e-16
    return ph, min(max(c - h, 0.0), ph), max(min(c + h, 1.0), ph)


def resumir(ent):
    alc = np.array(ent['alc']) == 1
    fr = np.array(ent['frac'])
    n = len(alc)
    P, lo, hi = wilson(int(alc.sum()), n)
    ent.update(n=n, P=P, P_lo=lo, P_hi=hi, frac_media=float(fr.mean()),
               frac_erro=float(fr.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
               indecisos=int((np.array(ent['alc']) == 2).sum()),
               t_medio=float(np.mean(ent['t'])))
    return ent


def pontos(dados, L=None, p=None):
    v = [e for k, e in dados.items() if not k.startswith('_')]
    if L is not None:
        v = [e for e in v if e['L'] == L]
    if p is not None:
        v = [e for e in v if abs(e['p'] - p) < 1e-9]
    return sorted(v, key=lambda e: (e['L'], e['p']))


def cruzamento(ps, y1, y2):
    """p onde y2 - y1 troca de sinal (de + para -), por interpolação linear. None se não houver."""
    d = np.asarray(y2) - np.asarray(y1)
    for i in range(len(d) - 1):
        if d[i] > 0 >= d[i + 1]:
            return float(ps[i] + d[i] * (ps[i + 1] - ps[i]) / (d[i] - d[i + 1]))
    return None


def meia_altura(ps, y):
    """Primeiro p em que y cai abaixo de 1/2, por interpolação linear. None se não houver."""
    for i in range(len(y) - 1):
        if y[i] >= 0.5 > y[i + 1]:
            return float(ps[i] + (y[i] - 0.5) * (ps[i + 1] - ps[i]) / (y[i] - y[i + 1]))
    return None


def _banda(valores):
    v = [x for x in valores if x is not None]
    return (float(np.percentile(v, 16)), float(np.percentile(v, 84))) if len(v) > 10 else (np.nan, np.nan)


def estimar_limiar(dados, n_boot=500, semente=1):
    """Duas estimativas de tamanho finito do limiar dinâmico, com bootstrap sobre sementes
    (o MESMO conjunto reamostrado em todos os p de um L, pois os substratos são acoplados):
      - territorio: cruzamento das curvas de território entre os dois maiores L
                    (e entre todos os pares consecutivos, como dispersão sistemática);
      - meia_altura: p em que P_alcance = 1/2, para cada L. Deve se aproximar de p*_c
                    com L crescente; a tendência com L é mais informativa que um valor só.
    Nenhuma das duas é o limite L -> infinito; a análise final deve comparar ambas."""
    Ls = sorted({e['L'] for e in pontos(dados)})
    if not Ls:
        return None
    ps = sorted(set.intersection(*[{round(e['p'], 6) for e in pontos(dados, L)} for L in Ls]))
    if ps and all(max(e['alc']) != 1 for e in pontos(dados)):      # controle (ex.: b = 1.70)
        return dict(ps=ps, Ls=Ls, sem_invasao=True, meia_altura={}, territorio=None)
    if len(ps) < 3:
        return None
    F, A = {}, {}
    for L in Ls:
        n = min(len(pontos(dados, L, p)[0]['frac']) for p in ps)
        F[L] = np.array([pontos(dados, L, p)[0]['frac'][:n] for p in ps])
        A[L] = np.array([np.array(pontos(dados, L, p)[0]['alc'][:n]) == 1 for p in ps], float)
    rng = np.random.default_rng(semente)
    idx = {L: [rng.integers(0, F[L].shape[1], F[L].shape[1]) for _ in range(n_boot)] for L in Ls}
    res = dict(ps=ps, Ls=Ls, sem_invasao=bool(all(A[L].sum() == 0 for L in Ls)))
    res['meia_altura'] = {}
    for L in Ls:
        c = meia_altura(ps, A[L].mean(1))
        lo, hi = _banda([meia_altura(ps, A[L][:, i].mean(1)) for i in idx[L]])
        res['meia_altura'][str(L)] = dict(p=c, lo=lo, hi=hi)
    res['territorio'] = None
    if len(Ls) >= 2:
        a, b = Ls[-2], Ls[-1]
        c = cruzamento(ps, F[a].mean(1), F[b].mean(1))
        lo, hi = _banda([cruzamento(ps, F[a][:, i].mean(1), F[b][:, j].mean(1))
                         for i, j in zip(idx[a], idx[b])])
        pares = {f'{x}-{y}': cruzamento(ps, F[x].mean(1), F[y].mean(1)) for x, y in zip(Ls[:-1], Ls[1:])}
        res['territorio'] = dict(p=c, lo=lo, hi=hi, par=[a, b], pares=pares)
    return res


# ==============================================================================
# EXECUÇÃO
# ==============================================================================
def arquivo_resultados(b):
    return os.path.join(PASTA, f"resultados_b{b:.4f}.json")


def carregar(b):
    arq = arquivo_resultados(b)
    if not os.path.exists(arq):
        return {'_meta': meta()}
    with open(arq) as f:
        dados = json.load(f)
    if dados.get('_meta') != meta():
        raise SystemExit(f"[ERRO] {arq} foi gerado com outra configuração:\n  arquivo: {dados.get('_meta')}\n"
                         f"  atual:   {meta()}\nMude VERSAO ou apague o arquivo.")
    return dados


def gravar(b, dados):
    arq = arquivo_resultados(b)
    tmp = arq + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dados, f)
    os.replace(tmp, arq)


def rodar():
    os.makedirs(PASTA, exist_ok=True)
    directions = motor.gerar_vizinhanca("moore", 1)
    inicio = time.time()
    for b in B_VALS:
        configurar_motor(b)
        motor.PROPORCAO_RANDOM_BURACOS = grade_p(b)[0]
        print(f"\n=== b = {b} ===")
        graves = [a for a in motor.checar_parametros() if 'TRANSIÇÃO' in a or 'epsilon' in a]
        if graves:
            raise SystemExit("[ERRO] b sobre um limiar k/m ou epsilon grande demais: corrija antes de rodar.")
        dados = carregar(b)
        for L in TAMANHOS:
            for p in grade_p(b):
                chave = f"L{L}_p{p:.4f}"
                ent = dados.get(chave, dict(L=L, p=p, alc=[], frac=[], t=[]))
                faltam = range(len(ent['alc']), N_SEMENTES[L])
                if len(faltam) == 0:
                    continue
                t0 = time.time()
                for s in faltam:
                    c, fr, t = uma_rodada(L, p, SEMENTE_BASE + s, directions)
                    ent['alc'].append(c)
                    ent['frac'].append(fr)
                    ent['t'].append(t)
                    if len(ent['alc']) % BLOCO == 0:
                        dados[chave] = resumir(ent)
                        gravar(b, dados)
                dados[chave] = resumir(ent)
                gravar(b, dados)
                print(f"  L={L:4d} p={p:.3f}: P_alcance={ent['P']:.3f} [{ent['P_lo']:.3f}, {ent['P_hi']:.3f}] | "
                      f"território={ent['frac_media']:.3f} | indecisos={ent['indecisos']} | "
                      f"{time.time() - t0:.0f}s (total {(time.time() - inicio) / 60:.1f} min)", flush=True)
        dados['_limiar'] = estimar_limiar(dados)
        gravar(b, dados)
        plotar(b, dados)
    resumo_geral()


# ==============================================================================
# FIGURAS E RESUMO
# ==============================================================================
def configurar_tipografia():
    base = {'font.family': 'serif', 'axes.labelsize': 14, 'font.size': 12, 'legend.fontsize': 10,
            'savefig.bbox': 'tight'}
    if USAR_LATEX:
        try:
            plt.rcParams.update({**base, 'text.usetex': True, 'font.serif': ['Computer Modern Roman'],
                                 'text.latex.preamble': r'\usepackage{amsmath}'})
            fig = plt.figure()
            fig.text(0.5, 0.5, r'$p^*_c$ teste')
            fig.savefig(os.devnull, format='png')
            plt.close(fig)
            return
        except Exception:
            plt.close('all')
            print("[aviso] LaTeX indisponível; usando mathtext.")
    plt.rcParams.update({**base, 'text.usetex': False, 'mathtext.fontset': 'cm',
                         'font.serif': ['DejaVu Serif']})


def salvar(fig, nome):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PASTA, f'{nome}.{ext}'), dpi=200)
    plt.close(fig)


def plotar(b, dados):
    configurar_tipografia()
    Ls = sorted({e['L'] for e in pontos(dados)})
    ps = sorted({e['p'] for e in pontos(dados)})
    lim = dados.get('_limiar')

    fig, ax = plt.subplots(figsize=(8, 6))
    for L in Ls:
        v = pontos(dados, L)
        P = np.array([e['P'] for e in v])
        err = np.clip([P - np.array([e['P_lo'] for e in v]), np.array([e['P_hi'] for e in v]) - P], 0, None)
        ax.errorbar([e['p'] for e in v], P, yerr=err, marker='o', capsize=3, label=fr'$L = {L}$')
    ax.set_xlabel(r'fração de buracos $p$')
    ax.set_ylabel(r'$P(\mathrm{alcance})$')
    ax.set_title(fr'(a) probabilidade de alcance, $b = {b}$')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend()
    salvar(fig, f'limiar_1_alcance_b{b:.4f}')

    fig, ax = plt.subplots(figsize=(8, 6))
    for L in Ls:
        v = pontos(dados, L)
        ax.errorbar([e['p'] for e in v], [e['frac_media'] for e in v], yerr=[e['frac_erro'] for e in v],
                    marker='o', capsize=3, label=fr'$L = {L}$')
    ter = lim and lim.get('territorio')
    if ter and ter['p'] is not None:
        ax.axvline(ter['p'], color='k', ls='--', lw=1)
        if np.isfinite(ter['lo']):
            ax.axvspan(ter['lo'], ter['hi'], color='k', alpha=0.08)
        ax.text(ter['p'], ax.get_ylim()[1] * 0.95, fr' cruzamento $\approx {ter["p"]:.3f}$', va='top')
    ax.set_xlabel(r'fração de buracos $p$')
    ax.set_ylabel('fração invadida do aglomerado da semente')
    ax.set_title(fr'(b) território final, $b = {b}$')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend()
    salvar(fig, f'limiar_2_territorio_b{b:.4f}')

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    for j, p in enumerate(ps):
        v = [e for e in pontos(dados, p=p) if e['P'] > 0]
        if len(v) > 1:
            P = np.array([e['P'] for e in v])
            err = np.clip([P - np.array([e['P_lo'] for e in v]), np.array([e['P_hi'] for e in v]) - P], 0, None)
            ax.errorbar([e['L'] for e in v], P, yerr=err, marker='o', capsize=3,
                        color=cmap(j / max(len(ps) - 1, 1)), label=fr'$p = {p:.2f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'tamanho da rede $L$')
    ax.set_ylabel(r'$P(\mathrm{alcance})$')
    ax.set_title(fr'(c) dependência com $L$: satura = invade, cai = não invade ($b = {b}$)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        ax.text(0.5, 0.5, 'nenhuma invasão em nenhum $p$', transform=ax.transAxes, ha='center')
    salvar(fig, f'limiar_3_tamanho_b{b:.4f}')


def intervalo_km(b):
    """Índice do intervalo entre limiares k/m que contém b."""
    return sum(1 for x in LIMIARES_KM if float(x) < b)


def verificar_constancia():
    """Para pares de b no mesmo intervalo k/m, compara os resultados semente a semente."""
    grupos = {}
    for b in B_VALS:
        if os.path.exists(arquivo_resultados(b)):
            grupos.setdefault(intervalo_km(b), []).append(b)
    linhas = []
    for bs in grupos.values():
        for b2 in bs[1:]:
            d1, d2 = carregar(bs[0]), carregar(b2)
            comuns = [k for k in d1 if not k.startswith('_') and k in d2]
            iguais = sum(d1[k]['alc'][:min(len(d1[k]['alc']), len(d2[k]['alc']))] ==
                         d2[k]['alc'][:min(len(d1[k]['alc']), len(d2[k]['alc']))] for k in comuns)
            linhas.append(f"  b = {bs[0]} e b = {b2} (mesmo intervalo k/m): {iguais}/{len(comuns)} pontos "
                          f"idênticos semente a semente" + ("  ✔" if iguais == len(comuns) else "  ✘ VERIFICAR"))
    return linhas


def _fmt(d):
    if d is None or d.get('p') is None:
        return '   --  '
    return f"{d['p']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def resumo_geral():
    configurar_tipografia()
    tab = []
    for b in B_VALS:
        if os.path.exists(arquivo_resultados(b)):
            dados = carregar(b)
            lim = estimar_limiar(dados)
            if lim is not None:
                tab.append((b, lim))
    if not tab:
        return
    Ls = sorted({L for _, lim in tab for L in lim['Ls']})
    print("\n" + "=" * 96)
    print("RESUMO DO LIMIAR DINÂMICO (estimativas de tamanho finito; bandas 16-84% por bootstrap)")
    print("=" * 96)
    print(f"{'b':>7} | {'cruzamento do território':^28} | " + " | ".join(f"P=1/2 em L={L:<4}" for L in Ls))
    for b, lim in tab:
        if lim['sem_invasao']:
            print(f"{b:>7} | nenhuma invasão em nenhum p da grade (p*_c = 0)")
            continue
        ter = lim['territorio']
        faltou = (ter is None or ter['p'] is None)
        linha = f"{b:>7} | {_fmt(ter):^28} | " + " | ".join(f"{_fmt(lim['meia_altura'].get(str(L))):>16}" for L in Ls)
        print(linha + ("   <- sem cruzamento: amplie P_POR_B" if faltou and lim['meia_altura'] else ""))
    for l in verificar_constancia():
        print(l)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for x in LIMIARES_KM:
        if float(x) < 5 / 3 + 1e-9:
            ax.axvline(float(x), color='0.88', lw=0.8, zorder=0)
            ax.text(float(x), 0.61, f'{x.numerator}/{x.denominator}', ha='center', fontsize=8, color='0.45')
    Lmax = str(Ls[-1])

    def serie(chave):
        xs, ys, lo, hi = [], [], [], []
        for b, lim in tab:
            d = lim['territorio'] if chave == 'ter' else lim['meia_altura'].get(Lmax)
            if lim['sem_invasao']:
                xs.append(b); ys.append(0.0); lo.append(0.0); hi.append(0.0)
            elif d and d.get('p') is not None:
                xs.append(b); ys.append(d['p'])
                # a estimativa pontual pode cair fora da banda bootstrap (percentis): recorta em 0
                lo.append(max(np.nan_to_num(d['p'] - d['lo']), 0.0))
                hi.append(max(np.nan_to_num(d['hi'] - d['p']), 0.0))
        return xs, ys, [lo, hi]
    x, y, e = serie('ter')
    ax.errorbar(x, y, yerr=e, fmt='o', capsize=4, color='C0', label='cruzamento do território (dois maiores $L$)')
    x, y, e = serie('meia')
    ax.errorbar(np.array(x) + 0.004, y, yerr=e, fmt='s', mfc='white', capsize=4, color='C3',
                label=f'$P_{{\\mathrm{{alcance}}}} = 1/2$ em $L = {Lmax}$')
    ax.axhline(LIMIAR_GEOMETRICO, color='k', ls='--', lw=1.2, label='limiar geométrico (Moore)')
    ax.set_xlabel('tentação $b$')
    ax.set_ylabel('fração de buracos $p$')
    ax.set_ylim(-0.02, 0.65)
    ax.set_xlim(1.0, max(b for b, _ in tab) + 0.04)
    ax.set_title('Limiar dinâmico de invasão (estimativas de tamanho finito)')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9, frameon=False)
    salvar(fig, 'limiar_4_diagrama_pc_b')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--so-plotar":
        for b in B_VALS:
            if os.path.exists(arquivo_resultados(b)):
                d = carregar(b)
                d['_limiar'] = estimar_limiar(d)
                plotar(b, d)
        resumo_geral()
    else:
        rodar()
