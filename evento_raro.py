"""
evento_raro.py — Experimento E: desaprisionamento por mutação única (limite de evento raro).

Coloque este arquivo na MESMA PASTA do engine_v2.py e rode:

    python evento_raro.py                # roda (ou retoma de onde parou)
    python evento_raro.py --piloto       # versão curta, para conferir que roda
    python evento_raro.py --so-plotar    # refaz tabela e figuras a partir dos JSON

Por que este experimento existe
-------------------------------
A Seção 5.4 mediu a mutação como taxa e NÃO conseguiu testar a hipótese H3 (crescimento na
escala 1/mu). O motivo ficou claro na própria medida: mesmo na menor taxa simulada a rede
sofre entre 6 e 23 mutações por passo, isto é, o que se observou foi o regime de ruído
permanente, não o de evento raro. O Apêndice ap:escala-mutacao mostra que a lei 1/mu vale
quando Gamma * tau_a << 1, com

    Gamma = mu * N_f * phi      (taxa de mutações DESTRAVADORAS por passo)
    tau_a = duração da avalanche que uma delas dispara

e nem phi nem tau_a haviam sido medidos em lugar nenhum da dissertação.

Este experimento mede os dois diretamente, sem pagar o custo de esperar. Em vez de sortear
mutações no tempo, a frente é levada ao aprisionamento com mu = 0 e então recebe UMA mutação
por vez, num sítio da fronteira sorteado uniformemente, deixando a dinâmica determinística
relaxar até um novo estado ancorado antes da mutação seguinte. Isso é o limite mu -> 0 exato,
e cada avalanche custa dezenas de passos em vez de 1/mu.

O que se mede
-------------
  phi     : fração das mutações que efetivamente mudam o estado ancorado da frente;
  tau_a   : passos de transiente até reancorar (a duração da avalanche);
  dN      : sítios ganhos pela colônia numa avalanche;
  dr      : avanço do raio numa avalanche;
  N_f     : número de sítios candidatos na fronteira (o N_f de Gamma);
  n_esc   : quantas mutações foram necessárias para atravessar o sistema.

Com isso fecham-se duas contas que a dissertação deixou em aberto:
  (i)  a taxa abaixo da qual a lei 1/mu vale:   mu << 1 / (N_f * phi * tau_a);
  (ii) a previsão t_esc = n_esc / (mu * N_f * phi), confrontável com o t_esc do Experimento B.

Detalhe técnico que importa: o estado ancorado pode ser um ciclo de período maior que 1.
Comparar o grid final com o inicial contaria a oscilação do ciclo como destravamento e
inflaria phi. Por isso o estado ancorado é canonizado pelo ciclo inteiro (menor hash entre os
estados do ciclo), e a massa e o raio são tomados como média e máximo sobre o ciclo.

Sementes: as mesmas dos demais experimentos (SEMENTE_BASE + s) para o substrato; a escolha do
sítio mutante usa um gerador independente, para não correlacionar com a desordem.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine_v2 as motor

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
PILOTO = "--piloto" in sys.argv

# (b, [p, ...]) — os mesmos patamares da Seção 5.4, em p ACIMA do limiar determinístico
CASOS = [
    (1.07, [0.42, 0.44, 0.48]),
    (1.45, [0.30, 0.36, 0.42]),
]
TAMANHOS = [101, 201]
N_SEMENTES = {101: 60, 201: 15}
# teto de mutações por realização: precisa bastar para atravessar (raio L/2), e o avanço por
# mutação destravadora é da ordem de 0,3 sítio, daí a escala ~ 2 L
N_EVENTOS = {101: 600, 201: 2000}
P_POR_L = {101: None, 201: 2}   # em L=201, usa apenas os 2 primeiros p de cada b (custo)
T_RELAX = 2000               # teto de passos por avalanche
T_ANCORA = 3000              # teto para o aprisionamento inicial
SEMENTE_BASE = 2000
SEMENTE_MUTANTE = 700000
LADO_COLONIA = 9
EPSILON = 1e-5
BLOCO = 10                   # grava o JSON a cada BLOCO sementes
USAR_LATEX = True
VERSAO = 1

if PILOTO:
    CASOS = [(1.07, [0.44]), (1.45, [0.36])]
    TAMANHOS = [101]
    N_SEMENTES = {101: 8}
    N_EVENTOS = {101: 150}
    P_POR_L = {101: None}
    BLOCO = 4

BASE_DIR = os.environ.get("PD_DADOS") or os.path.dirname(os.path.abspath(__file__))
# PD_DADOS: se definida, diz onde ficam (ou vão ficar) as pastas de dados. Sem ela, o
# comportamento e o de sempre: ao lado deste arquivo.
PASTA = os.path.join(BASE_DIR, ("piloto_" if PILOTO else "") +
                     f"evento_raro_v{VERSAO}_moore1_eps{EPSILON:g}_n0{LADO_COLONIA}")


def configurar_motor(b):
    motor.TIPO_VIZINHANCA = "moore"
    motor.PROFUNDIDADE_VIZINHANCA = 1
    motor.TOPOLOGIA_TABULEIRO = "toroide"
    motor.ESTRATEGIA_MAR = motor.INDICE_D
    motor.ESTRATEGIA_INVASOR = motor.INDICE_C
    motor.CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": LADO_COLONIA}]
    motor.CONFIG_BURACOS = []
    motor.PROPORCAO_RANDOM_COOPERADORES = 0.0
    motor.TAXA_MUTACAO_C = motor.TAXA_MUTACAO_D = 0.0   # a mutação é colocada à mão
    motor.K_FERMI = 0.0
    motor.PROBABILIDADE_ATUALIZACAO = 1.0
    motor.EPSILON_P = EPSILON
    motor.b = b


def meta():
    return dict(versao=VERSAO, semente_base=SEMENTE_BASE, semente_mutante=SEMENTE_MUTANTE,
                n_eventos={str(k): v for k, v in N_EVENTOS.items()},
                t_relax=T_RELAX, t_ancora=T_ANCORA,
                lado_colonia=LADO_COLONIA, epsilon=EPSILON,
                vizinhanca="moore1", topologia="toroide")


# ==============================================================================
# DINÂMICA
# ==============================================================================
def _passo(grid, ativo, L, PM, directions, di, du):
    _, _, pay = motor.censo_e_payoff_numba(grid, L, PM[0, 0], PM[0, 1], PM[1, 0], PM[1, 1],
                                           directions)
    grid = motor.escolha_racional_numba(grid, pay, L, 0.0, 0.0, directions, di, du)
    grid[~ativo] = motor.INDICE_BURACO
    return grid


def relaxar(grid, ativo, comp, raio, alvo, L, PM, directions, di, du, teto):
    """Evolui a dinâmica determinística até ancorar, escapar, morrer ou esgotar o teto.

    Ao ancorar, o estado é canonizado varrendo o ciclo inteiro: `canon` é o menor hash entre
    os estados do ciclo, `n_medio` a massa média da colônia ao longo dele e `raio` o raio
    máximo atingido nele. `tau` é o transiente — passos desde a mutação até ENTRAR no ciclo.
    """
    C = motor.INDICE_C
    vistos = {}
    hs, Ns, Rs = [], [], []     # histórico por passo, para canonizar sem repercorrer o ciclo
    for t in range(1, teto + 1):
        grid = _passo(grid, ativo, L, PM, directions, di, du)
        X = (grid == C) & comp
        if not X.any():
            return grid, dict(desfecho='morreu', tau=t, periodo=0, canon=0, n_medio=0.0,
                              raio=-1.0)
        rmax = float(raio[X].max())
        if rmax >= alvo:
            return grid, dict(desfecho='escapou', tau=t, periodo=0, canon=0,
                              n_medio=float(X.sum()), raio=rmax)
        h = hash(grid.tobytes())
        if h in vistos:
            t0 = vistos[h]                     # passo da 1a visita: o ciclo é t0 .. t-1
            sl = slice(t0 - 1, t - 1)
            return grid, dict(desfecho='ancorou', tau=int(t0), periodo=int(t - t0),
                              canon=int(min(hs[sl])), n_medio=float(np.mean(Ns[sl])),
                              raio=float(max(Rs[sl])))
        vistos[h] = t
        hs.append(h)
        Ns.append(float(X.sum()))
        Rs.append(rmax)
    return grid, dict(desfecho='teto', tau=teto, periodo=0, canon=0,
                      n_medio=Ns[-1] if Ns else 0.0, raio=Rs[-1] if Rs else -1.0)


def candidatos(grid, ativo, comp, directions):
    """Desertores do aglomerado da origem que tocam a frente cooperativa: os sítios onde uma
    mutação D -> C pode, em princípio, alimentar o avanço."""
    C, D = motor.INDICE_C, motor.INDICE_D
    eh_c = (grid == C) & ativo
    viz_c = np.zeros(grid.shape, bool)
    for d in directions:
        viz_c |= np.roll(np.roll(eh_c, int(d[0]), axis=0), int(d[1]), axis=1)
    return np.argwhere((grid == D) & ativo & comp & viz_c)


def uma_rodada(L, p, b, semente, directions):
    """Leva a frente ao aprisionamento e então aplica mutações únicas, uma de cada vez."""
    np.random.seed(semente)
    grid, paredes, colonia, origem = motor.inicializar_universo(L, p_buracos=p, p_coop=0.0)
    ativo, ell, ux, uy = motor.construir_substrato(grid, colonia, origem, L, directions)
    comp = ell >= 0
    alvo = L // 2 - 1
    raio = np.maximum(np.abs(ux), np.abs(uy)).astype(np.float64)
    PM = motor.matriz_payoff()
    di = np.zeros((1, 1), np.int32)
    du = np.zeros((1, 1))
    rng = np.random.default_rng(SEMENTE_MUTANTE + semente)

    grid, est = relaxar(grid, ativo, comp, raio, alvo, L, PM, directions, di, du, T_ANCORA)
    t_ancora, periodo0 = est['tau'], est['periodo']
    if est['desfecho'] != 'ancorou':
        # a frente resolveu sozinha, sem precisar de mutação: fora do escopo deste experimento
        return dict(inicial=est['desfecho'], t_ancora=t_ancora, periodo0=periodo0,
                    eventos=[], n_esc=-1)

    eventos = []   # [destravou, tau_a, dN, dr, N_f, periodo]
    n_esc = -1
    for k in range(N_EVENTOS[L]):
        cand = candidatos(grid, ativo, comp, directions)
        if len(cand) == 0:
            break
        n_f = len(cand)
        r, c = cand[rng.integers(len(cand))]
        antes = est
        grid = grid.copy()
        grid[r, c] = motor.INDICE_C                       # a mutação: um único sítio
        grid, est = relaxar(grid, ativo, comp, raio, alvo, L, PM, directions, di, du, T_RELAX)
        destravou = int(est['canon'] != antes['canon'])
        eventos.append([destravou, int(est['tau']), float(est['n_medio'] - antes['n_medio']),
                        float(est['raio'] - antes['raio']), int(n_f), int(est['periodo'])])
        if est['desfecho'] == 'escapou':
            n_esc = k + 1
            break
        if est['desfecho'] in ('morreu', 'teto'):
            break
    return dict(inicial='ancorou', t_ancora=t_ancora, periodo0=periodo0, eventos=eventos,
                n_esc=n_esc)


# ==============================================================================
# ARQUIVOS (com retomada)
# ==============================================================================
def arquivo(b):
    return os.path.join(PASTA, f"evento_raro_b{b:.4f}.json")


def carregar(b):
    if not os.path.exists(arquivo(b)):
        return {"_meta": meta()}
    d = json.load(open(arquivo(b)))
    if d.get("_meta") != meta():
        raise SystemExit(f"[ERRO] {os.path.basename(arquivo(b))} foi gerado com outros "
                         f"parâmetros. Mude VERSAO ou apague a pasta {PASTA}.")
    return d


def gravar(b, dados):
    os.makedirs(PASTA, exist_ok=True)
    tmp = arquivo(b) + '.tmp'
    json.dump(dados, open(tmp, 'w'))
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
            for p in ps[:P_POR_L.get(L)]:
                chave = f"L{L}_p{p:.4f}"
                ent = dados.get(chave) or dict(L=L, p=p, b=b, rodadas=[])
                for s in range(len(ent['rodadas']), N_SEMENTES[L]):
                    ent['rodadas'].append(uma_rodada(L, p, b, SEMENTE_BASE + s, directions))
                    if (s + 1) % BLOCO == 0:
                        dados[chave] = ent
                        gravar(b, dados)
                dados[chave] = ent
                gravar(b, dados)
                ev = [e for r in ent['rodadas'] for e in r['eventos']]
                anc = [r for r in ent['rodadas'] if r['inicial'] == 'ancorou']
                esc = [r['n_esc'] for r in anc if r['n_esc'] > 0]
                phi = float(np.mean([e[0] for e in ev])) if ev else float('nan')
                med = float(np.median(esc)) if esc else float('nan')
                print(f"  L={L:4d} p={p:.2f}: {len(anc)}/{len(ent['rodadas'])} ancoraram | "
                      f"{len(ev)} mutações | phi={phi:.3f} | {len(esc)} escaparam "
                      f"(mediana {med:.0f} mutações) | {(time.time() - t0) / 60:.1f} min")
    resumo()


# ==============================================================================
# ANÁLISE
# ==============================================================================
def coletar(b, L, p):
    d = carregar(b)
    ent = d.get(f"L{L}_p{p:.4f}")
    if not ent or not ent['rodadas']:
        return None
    ev = np.array([e for r in ent['rodadas'] for e in r['eventos']], float).reshape(-1, 6)
    if len(ev) == 0:
        return None
    anc = [r for r in ent['rodadas'] if r['inicial'] == 'ancorou']
    esc = np.array([r['n_esc'] for r in anc if r['n_esc'] > 0], float)
    d_ = ev[:, 0] > 0
    return dict(n_ev=len(ev), phi=float(d_.mean()),
                tau=float(np.mean(ev[d_, 1])) if d_.any() else np.nan,
                tau_max=float(np.max(ev[d_, 1])) if d_.any() else np.nan,
                dN=ev[d_, 2] if d_.any() else np.array([]),
                dr=float(np.mean(ev[d_, 3])) if d_.any() else np.nan,
                n_f=float(np.mean(ev[:, 4])), n_esc=esc,
                n_anc=len(anc), n_tot=len(ent['rodadas']))


def tipografia():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
                         'axes.labelsize': 13, 'font.size': 11, 'legend.fontsize': 9})
    plt.rcParams['text.usetex'] = USAR_LATEX
    if USAR_LATEX:
        try:
            f = plt.figure(); f.text(.5, .5, r'$x$'); f.canvas.draw(); plt.close(f)
        except Exception:
            plt.rcParams['text.usetex'] = False
            print("[aviso] LaTeX indisponível; usando mathtext.")


def salvar(fig, nome):
    os.makedirs(PASTA, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PASTA, f'{nome}.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura: {nome}.pdf/.png")


def resumo():
    tipografia()
    cab = (f"{'b':>6} {'p':>6} {'L':>5} {'mut':>7} {'phi':>7} {'tau_a':>7} {'<dN>':>7} "
           f"{'dN max':>7} {'<dr>':>6} {'N_f':>6} {'n_esc':>7} {'esc':>8} {'mu <<':>9}")
    linhas = ["=" * 112,
              "EXPERIMENTO E — DESAPRISIONAMENTO POR MUTACAO UNICA (limite de evento raro)",
              "=" * 112, cab, "-" * 112]
    dados_fig = []
    for b, ps in CASOS:
        for L in TAMANHOS:
            for p in ps[:P_POR_L.get(L)]:
                c = coletar(b, L, p)
                if not c:
                    continue
                tau = c['tau'] if np.isfinite(c['tau']) else 1.0
                mu_lim = 1.0 / max(c['n_f'] * c['phi'] * tau, 1e-12)
                linhas.append(
                    f"{b:>6} {p:>6.2f} {L:>5} {c['n_ev']:>7} {c['phi']:>7.3f} "
                    f"{c['tau']:>7.1f} "
                    f"{np.mean(c['dN']) if len(c['dN']) else np.nan:>7.1f} "
                    f"{np.max(c['dN']) if len(c['dN']) else np.nan:>7.0f} "
                    f"{c['dr']:>6.2f} {c['n_f']:>6.0f} "
                    f"{np.median(c['n_esc']) if len(c['n_esc']) else np.nan:>7.0f} "
                    f"{len(c['n_esc']):>3d}/{c['n_anc']:<4d} {mu_lim:>9.2e}")
                dados_fig.append((b, p, L, c))
    linhas += ["-" * 112,
               "phi    fracao das mutacoes que mudam o estado ancorado",
               "tau_a  passos de transiente ate reancorar (duracao da avalanche)",
               "dN,dr  massa e raio ganhos por avalanche destravadora",
               "N_f    sitios candidatos na fronteira | n_esc mediana de mutacoes ate atravessar",
               "mu <<  teto de taxa para valer o regime de evento raro: 1/(N_f phi tau_a)",
               "=" * 112]
    txt = "\n".join(linhas)
    print("\n" + txt)
    os.makedirs(PASTA, exist_ok=True)
    open(os.path.join(PASTA, 'evento_raro_tabela.txt'), 'w').write(txt + "\n")
    if not dados_fig:
        return

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    Lmax = max(L for _, _, L, _ in dados_fig)
    for b, p, L, c in dados_fig:
        if L != Lmax:
            continue
        dN = c['dN'][c['dN'] > 0]
        rot = fr'$b={b}$, $p={p:.2f}$'
        if len(dN) >= 10:
            bins = np.geomspace(max(dN.min(), 0.5), max(dN.max(), 2.0), 16)
            h, e = np.histogram(dN, bins=bins, density=True)
            ctr = np.sqrt(e[1:] * e[:-1])
            axs[0].plot(ctr[h > 0], h[h > 0], 'o-', ms=4, label=rot)
        if len(c['n_esc']):
            x = np.sort(c['n_esc'])
            axs[1].plot(x, np.arange(1, len(x) + 1) / c['n_anc'], drawstyle='steps-post',
                        label=rot)
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel(r'tamanho da avalanche $\Delta N$')
    axs[0].set_ylabel(r'densidade de probabilidade')
    axs[0].set_title(r'Avalanches disparadas por uma mutação')
    axs[1].set_xlabel(r'mutações até atravessar o sistema')
    axs[1].set_ylabel(r'fração acumulada de realizações')
    axs[1].set_title(r'Custo do desaprisionamento')
    for ax in axs:
        ax.grid(True, which='both', ls=':', alpha=.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    salvar(fig, 'evento_raro_1_avalanches')


def comparar_com_B(pasta_b=None):
    """Confronta a previsão de evento raro com o t_esc medido no Experimento B.

    A dedução do Apêndice ap:escala-mutacao dá, no limite de mutação rara,

        t_esc = n_esc / Gamma,     Gamma = mu * N_f * phi,

    com n_esc, N_f e phi medidos AQUI e t_esc medido LÁ. Como os dois experimentos usam a
    mesma SEMENTE_BASE, a comparação é feita semente a semente: a realização s é a mesma
    rede nos dois. A razão previsto/medido diz quanto o desaprisionamento real se afasta de
    uma sequência de eventos independentes — se for muito maior que 1, o destravamento já é
    coletivo naquela taxa, e a lei 1/mu não vale ali.
    """
    if pasta_b is None:
        pasta_b = os.path.join(BASE_DIR, f"mutacao_v1_moore1_eps{EPSILON:g}_n0{LADO_COLONIA}")
    if not os.path.isdir(pasta_b):
        print(f"\n[comparação com o Experimento B pulada: não achei {pasta_b}]")
        return
    linhas = ["", "=" * 96,
              "PREVISAO DO EVENTO RARO  x  t_esc MEDIDO NO EXPERIMENTO B (pareado por semente)",
              "=" * 96,
              f"{'b':>6} {'p':>6} {'L':>5} {'mu':>8} {'pares':>6} {'t_esc med':>10} "
              f"{'t_esc prev':>11} {'prev/med':>9}", "-" * 96]
    for b, ps in CASOS:
        for L in TAMANHOS:
            for p in ps[:P_POR_L.get(L)]:
                c = coletar(b, L, p)
                if not c or not np.isfinite(c['phi']) or c['phi'] <= 0:
                    continue
                dE = carregar(b)[f"L{L}_p{p:.4f}"]['rodadas']
                nE = np.array([r['n_esc'] for r in dE], float)
                for mu in (1e-3, 3e-3, 1e-2, 3e-2):
                    f = os.path.join(pasta_b, f"resultados_b{b:.4f}_mu{mu:g}.json")
                    if not os.path.exists(f):
                        continue
                    ent = json.load(open(f)).get(f"L{L}_p{p:.4f}")
                    if not ent:
                        continue
                    tB = np.array(ent['t_esc'], float)
                    n = min(len(tB), len(nE))
                    ok = (tB[:n] > 0) & (nE[:n] > 0)
                    if ok.sum() < 3:
                        continue
                    gama = mu * c['n_f'] * c['phi']
                    prev = nE[:n][ok] / gama
                    med = tB[:n][ok]
                    linhas.append(f"{b:>6} {p:>6.2f} {L:>5} {mu:>8.0e} {ok.sum():>6} "
                                  f"{np.median(med):>10.0f} {np.median(prev):>11.0f} "
                                  f"{np.median(prev / med):>9.2f}")
    linhas += ["-" * 96,
               "prev/med ~ 1  : o desaprisionamento e mesmo uma sequencia de eventos raros",
               "prev/med >> 1 : a rede escapa mais rapido que isso, ou seja, ja e coletivo",
               "Apenas as sementes que atravessaram nos DOIS experimentos entram na conta.",
               "=" * 96]
    txt = "\n".join(linhas)
    print(txt)
    os.makedirs(PASTA, exist_ok=True)
    with open(os.path.join(PASTA, 'evento_raro_tabela.txt'), 'a') as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if "--so-plotar" in sys.argv:
        resumo()
        comparar_com_B()
    else:
        rodar()
        comparar_com_B()
