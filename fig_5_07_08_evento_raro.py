"""
Figuras 5.7 e 5.8 e Tabela 5.7 — a mutação como evento raro.

    5.7  evento_raro_1_intensivas.pdf   phi e tau_a contra p, para os dois tamanhos
    5.8  evento_raro_2_avalanches.pdf   (a) distribuição das avalanches; (b) mediana da razão
                                         entre o tempo de escape previsto pelo limite de evento
                                         raro, n_esc/(mu N_f), e o medido na varredura em mu,
                                         semente a semente

Lê evento_raro_v1_moore1_eps1e-05_n09/evento_raro_b*.json (gravados por evento_raro.py) e,
para a comparação, mutacao_v1_moore1_eps1e-05_n09/ (de varredura_mutacao.py). Imprime a
Tabela 5.7, os expoentes ajustados às avalanches e a tabela da comparação pareada.
Não simula nada.
"""
import json
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

import estilo as E

E.aplicar()

PASTA_E = os.path.join(E.AQUI, 'evento_raro_v1_moore1_eps1e-05_n09')
PASTA_B = os.path.join(E.AQUI, 'mutacao_v1_moore1_eps1e-05_n09')
CASOS = [(1.07, [0.42, 0.44, 0.48]), (1.45, [0.30, 0.36, 0.42])]
P_POR_L = {101: None, 201: 2}
MUS = (1e-3, 3e-3, 1e-2, 3e-2)

# cor = (b, p): azul para b = 1,07 e vermelho para b = 1,45, mais escuro quanto mais diluído
COR = {(1.07, 0.42): E.AZUL_MEDIO, (1.07, 0.44): E.AZUL, (1.07, 0.48): E.AZUL_ESCURO,
       (1.45, 0.30): E.VERMELHO_MEDIO, (1.45, 0.36): E.VERMELHO, (1.45, 0.42): E.VERMELHO_ESCURO}
TRACO = {101: '-', 201: '--'}
MARCA = {101: 'o', 201: 's'}


def celula(b, L, p):
    arq = os.path.join(PASTA_E, f'evento_raro_b{b:.4f}.json')
    if not os.path.exists(arq):
        return None
    d = json.load(open(arq)).get(f'L{L}_p{p:.4f}')
    if not d:
        return None
    rod = d['rodadas']
    anc = [r for r in rod if r['inicial'] == 'ancorou']
    ev = np.array([e for r in rod for e in r['eventos']], float).reshape(-1, 6)
    if len(ev) == 0 or not anc:
        return None
    m = ev[:, 0] > 0
    nesc = np.array([r['n_esc'] for r in anc if r['n_esc'] > 0], float)
    return dict(b=b, L=L, p=p, rod=rod, ev=ev, m=m, phi=float(m.mean()),
                tau=float(ev[m, 1].mean()), dN=ev[m, 2], dr=float(ev[m, 3].mean()),
                Nf=float(ev[:, 4].mean()), nesc=nesc, n_anc=len(anc),
                frac_esc=len(nesc) / len(anc))


CEL = [c for b, ps in CASOS for L in (101, 201) for p in ps[:P_POR_L[L]]
       if (c := celula(b, L, p))]
if not CEL:
    raise SystemExit(f"nenhum dado do Experimento E em {PASTA_E}")


def incerteza(c, qual, n_bs=1000, semente=7):
    """Erro padrão de phi ou tau_a por bootstrap sobre REALIZAÇÕES (sementes).

    As mutações de uma mesma realização não são independentes: acontecem na mesma rede e na
    mesma frente, uma depois da outra. Por isso se sorteiam realizações inteiras, com
    reposição, e recalcula-se a grandeza com todos os eventos delas (Seção 3.6). Sortear
    eventos isolados (a versão anterior) subestima a incerteza.
    """
    rng = np.random.default_rng(semente)
    blocos = [np.array(r['eventos'], float).reshape(-1, 6) for r in c['rod'] if r['eventos']]
    n = len(blocos)
    est = []
    for _ in range(n_bs):
        ev = np.concatenate([blocos[i] for i in rng.integers(0, n, n)])
        m = ev[:, 0] > 0
        est.append(m.mean() if qual == 'phi' else (ev[m, 1].mean() if m.any() else np.nan))
    return float(np.nanstd(est))


def incerteza_eventos(x, n_bs=1000, semente=7):
    """Bootstrap sobre eventos isolados (versão anterior; só para comparação)."""
    rng = np.random.default_rng(semente)
    x = np.asarray(x, float)
    return float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_bs)]))


# ==============================================================================
# LEI DE POTÊNCIA: estimador de Hill com corte escolhido por Kolmogorov--Smirnov
# ==============================================================================
def hill(x, xmin):
    x = x[x >= xmin]
    if len(x) < 30:
        return np.nan, np.nan, 0
    a = 1.0 + len(x) / np.sum(np.log(x / xmin))
    return a, (a - 1) / np.sqrt(len(x)), len(x)


def ks(x, xmin, a):
    x = np.sort(x[x >= xmin])
    emp = np.arange(1, len(x) + 1) / len(x)
    return float(np.max(np.abs(emp - (1 - (x / xmin) ** (1 - a)))))


def ajustar_potencia(dN):
    """Devolve (xmin, alpha, erro, n_cauda, D_KS) com o xmin que minimiza a distância KS."""
    dN = dN[dN > 0]
    melhor = (np.inf, None)
    for xm in (2, 3, 5, 8, 12, 20, 30, 50):
        a, e, n = hill(dN, xm)
        if np.isnan(a):
            continue
        D = ks(dN, xm, a)
        if D < melhor[0]:
            melhor = (D, (xm, a, e, n, D))
    return melhor[1]


# ==============================================================================
# COMPARAÇÃO COM O EXPERIMENTO B
# ==============================================================================
def comparar(c):
    """[(mu, Gamma*tau_a, mediana de previsto/medido, n de pares), ...]"""
    out = []
    nE = np.array([r['n_esc'] for r in c['rod']], float)
    for mu in MUS:
        arq = os.path.join(PASTA_B, f"resultados_b{c['b']:.4f}_mu{mu:g}.json")
        if not os.path.exists(arq):
            continue
        ent = json.load(open(arq)).get(f"L{c['L']}_p{c['p']:.4f}")
        if not ent:
            continue
        tB = np.array(ent['t_esc'], float)
        n = min(len(tB), len(nE))
        ok = (tB[:n] > 0) & (nE[:n] > 0)
        if ok.sum() < 3:
            continue
        # n_esc conta TODAS as mutações de fronteira aplicadas (as que destravam e as que não
        # destravam), e elas chegam à taxa mu*N_f; logo t_prev = n_esc/(mu N_f). Dividir também
        # por phi contaria phi duas vezes (erro da versão anterior, que superestimava por 1/phi).
        taxa = mu * c['Nf']
        gamma = taxa * c['phi']                      # taxa de mutações destravadoras
        out.append((mu, gamma * c['tau'],
                    float(np.median((nE[:n][ok] / taxa) / tB[:n][ok])), int(ok.sum())))
    return out


# ==============================================================================
# TABELA
# ==============================================================================
def tabela():
    L1 = ["=" * 118,
          "EXPERIMENTO E — DESAPRISIONAMENTO POR MUTAÇÃO ÚNICA (limite de evento raro)",
          "=" * 118,
          f"{'b':>5} {'p':>5} {'L':>4} | {'mutações':>9} {'phi':>14} {'tau_a':>13} | "
          f"{'recuo':>6} {'med dN':>6} {'<dN>':>6} {'max':>6} {'top1%':>6} | {'N_f':>5} "
          f"{'n_esc':>6} {'esc':>7} | {'mu_max':>8}",
          "-" * 118]
    for c in CEL:
        dp = incerteza(c, 'phi')
        dt = incerteza(c, 'tau')
        rec = float(np.mean(c['dN'] <= 0))
        dN = c['dN'][c['dN'] > 0]
        top = np.sort(dN)[::-1][:max(1, len(dN) // 100)].sum() / dN.sum()
        mu_max = 1.0 / (c['Nf'] * c['phi'] * c['tau'])
        L1.append(
            f"{c['b']:>5} {c['p']:>5.2f} {c['L']:>4} | {len(c['ev']):>9} "
            f"{c['phi']:.3f}±{dp:.3f} {c['tau']:>7.2f}±{dt:<5.2f} | "
            f"{rec * 100:>5.0f}% {np.median(dN):>6.0f} {dN.mean():>6.1f} {dN.max():>6.0f} "
            f"{top * 100:>5.0f}% | "
            f"{c['Nf']:>5.0f} "
            f"{np.median(c['nesc']) if len(c['nesc']) else np.nan:>6.0f} "
            f"{len(c['nesc']):>3d}/{c['n_anc']:<3d} | {mu_max:>8.1e}")
    L1 += ["-" * 118,
           "phi    fração das mutações de fronteira que mudam o estado ancorado",
           "tau_a  passos de transiente até reancorar",
           "recuo  fração das avalanches destravadoras que ENCOLHEM a colônia",
           "dN     massa ganha, só nas avalanches que aumentam a colônia",
           "top1%  fração de toda a massa movida que cabe no 1% maiores avalanches",
           "n_esc  mediana de mutações até atravessar (só entre as que atravessaram)",
           "mu_max teto para o regime de evento raro, 1/(N_f phi tau_a)",
           "=" * 118, "",
           "EXPOENTES DA DISTRIBUIÇÃO DE AVALANCHES   P(dN) ~ dN^(-alpha),  dN >= dN_min",
           "-" * 118,
           f"{'b':>5} {'p':>5} {'L':>4} | {'dN_min':>7} {'alpha':>14} {'n na cauda':>11} "
           f"{'D_KS':>7}"]
    for c in CEL:
        r = ajustar_potencia(c['dN'])
        if r:
            xm, a, e, n, D = r
            L1.append(f"{c['b']:>5} {c['p']:>5.2f} {c['L']:>4} | {xm:>7.0f} "
                      f"{a:.2f}±{e:.2f}     {n:>11} {D:>7.3f}")
    L1 += ["-" * 118, "",
           "PREVISÃO DO EVENTO RARO  ÷  t_esc MEDIDO NO EXPERIMENTO B (pareado por semente)",
           "-" * 118,
           f"{'b':>5} {'p':>5} {'L':>4} | " + "".join(f"{f'mu={m:g}':>13}" for m in MUS)]
    for c in CEL:
        cmp = {m: (r, n) for m, _, r, n in comparar(c)}
        if not cmp:
            continue
        cel = "".join(f"{cmp[m][0]:>8.2f} ({cmp[m][1]:>2d})" if m in cmp else f"{'—':>13}"
                      for m in MUS)
        L1.append(f"{c['b']:>5} {c['p']:>5.2f} {c['L']:>4} | {cel}")
    todos = [(m, r, n) for c in CEL for m, _, r, n in comparar(c)]
    L1.append("-" * 118)
    L1.append("mediana por taxa: " + "   ".join(
        f"mu={m:g}: {np.median([r for mm, r, _ in todos if mm == m]):.2f}"
        for m in MUS if any(mm == m for mm, _, _ in todos)))
    L1 += ["entre parênteses, o número de sementes que atravessaram nos DOIS experimentos",
           "=" * 118]
    txt = "\n".join(L1)
    print(txt)
    os.makedirs(PASTA_E, exist_ok=True)
    open(os.path.join(PASTA_E, 'evento_raro_tabela.txt'), 'w').write(txt + "\n")




# ==============================================================================
# FIGURAS
# ==============================================================================
def rot_bp(b, p):
    return rf'$b={E.num(b, 2)}$, $p={E.num(p, 2)}$'




def fig_intensivas():
    """Figura 5.7: phi e tau_a contra p. Cor = b, traço/marcador = L."""
    cor_b = {1.07: E.AZUL, 1.45: E.VERMELHO}
    fig, axs = E.figura(1, 2, altura=5.8)
    for b, _ in CASOS:
        for L in (101, 201):
            cs = [c for c in CEL if c['b'] == b and c['L'] == L]
            if not cs:
                continue
            x = [c['p'] for c in cs]
            for ax, ch, err in ((axs[0], 'phi', lambda c: incerteza(c, 'phi')),
                                (axs[1], 'tau', lambda c: incerteza(c, 'tau'))):
                ax.errorbar(x, [c[ch] for c in cs], yerr=[err(c) for c in cs],
                            marker=MARCA[L], color=cor_b[b], ls=TRACO[L],
                            mfc=cor_b[b] if L == 101 else 'white',
                            label=rf'$b={E.num(b, 2)}$, $L={L}$')
    axs[0].set_ylabel(r'$\varphi$')
    axs[0].set_title(r'(a) Fração de mutações que destravam a frente')
    axs[1].set_ylabel(r'$\tau_a$ (passos)')
    axs[1].set_title(r'(b) Duração da avalanche')
    for ax in axs:
        ax.set_xlabel(r'Fração de buracos $p$')
        ax.grid(ls=':')
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        E.virgula(ax)
    axs[0].legend(loc='best')
    E.salvar(fig, 'evento_raro_1_intensivas')


def fig_avalanches():
    """Figura 5.8: CCDF das avalanches e previsão/medida contra mu.

    Cor = (b, p) (azul para b = 1,07, vermelho para b = 1,45; mais escuro = mais diluído);
    traço e marcador = L (cheio 101, tracejado/vazado 201). Uma legenda só, acima dos painéis.
    """
    fig, axs = E.figura(1, 2, altura=7.4)
    a, d = axs
    # acumulada complementar: não depende de binagem e mostra a cauda sem artefato
    for c in CEL:
        dN = np.sort(c['dN'][c['dN'] > 0])
        if len(dN) < 100:
            continue
        ccdf = 1.0 - np.arange(len(dN)) / len(dN)
        a.plot(dN, ccdf, ls=TRACO[c['L']], lw=0.9, color=COR[(c['b'], c['p'])])
    xx = np.geomspace(3, 200, 10)                    # guia de inclinação, não um ajuste
    a.plot(xx, 0.42 * (xx / 3.0) ** -1.0, color=E.PRETO, ls=':', lw=1.0, zorder=0)
    a.annotate(r'guia $\propto\Delta N^{-1}$', xy=(30, 0.42 * (30 / 3.0) ** -1.0),
               xytext=(1.2, 2.5e-3), fontsize=E.TAM_ANOTACAO,
               arrowprops=dict(arrowstyle='->', lw=0.6, color=E.PRETO))
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlim(0.8, 3000); a.set_ylim(5e-4, 1.6)
    a.set_xlabel(r'Tamanho da avalanche $\Delta N$')
    a.set_ylabel(r'$P(\geq \Delta N)$')
    a.set_title(r'(a) Avalanches disparadas por uma mutação')

    for c in CEL:
        r = comparar(c)
        if len(r) < 2:
            continue
        cor = COR[(c['b'], c['p'])]
        d.plot([x[0] for x in r], [x[2] for x in r], marker=MARCA[c['L']], ls=TRACO[c['L']],
               color=cor, mfc=cor if c['L'] == 101 else 'white')
    d.axhline(1.0, color=E.PRETO, lw=0.9, ls=':')
    d.text(1.0e-2, 1.07, r'previsto $=$ medido', va='bottom', fontsize=E.TAM_ANOTACAO,
           color=E.CINZA_ESCURO)
    d.set_xscale('log')
    d.set_xticks(MUS)
    d.set_xticklabels([r'$10^{-3}$', r'$3\times10^{-3}$', r'$10^{-2}$', r'$3\times10^{-2}$'])
    d.minorticks_off()
    d.set_yscale('log')
    d.set_ylim(0.12, 3.0)
    d.set_yticks([0.2, 0.5, 1, 2])
    d.set_yticklabels([r'$0{,}2$', r'$0{,}5$', r'$1$', r'$2$'])
    d.yaxis.set_minor_formatter(plt.NullFormatter())
    d.set_xlabel(r'Taxa de mutação $\mu$')
    d.set_ylabel(r'$t_{\mathrm{esc}}^{\mathrm{previsto}}\,/\,t_{\mathrm{esc}}^{\mathrm{medido}}$')
    d.set_title(r'(b) Previsão do evento raro contra a medida')
    a.grid(which='major', ls=':'); d.grid(ls=':')

    alcas = [Line2D([], [], color=COR[(b, p)], lw=1.4, label=rot_bp(b, p))
             for b, ps in CASOS for p in ps]
    alcas += [Line2D([], [], color=E.CINZA_ESCURO, ls=TRACO[L], marker=MARCA[L],
                     mfc=E.CINZA_ESCURO if L == 101 else 'white', label=rf'$L={L}$')
              for L in (101, 201)]
    fig.legend(handles=alcas, loc='outside upper center', ncol=4, columnspacing=1.2)
    E.salvar(fig, 'evento_raro_2_avalanches')


if __name__ == "__main__":
    tabela()
    fig_intensivas()
    fig_avalanches()
