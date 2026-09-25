"""
Figuras 5.1, 5.2 e 5.3 e Tabela 5.1 — o limiar dinâmico de invasão p*_c(b).

    5.1  analise_3_alcance.pdf         probabilidade de alcance P_alc(p) para três valores de b
    5.2  analise_1_convergencia_L.pdf  convergência das duas cotas contra 1/L
    5.3  analise_2_diagrama_pc_b.pdf   a escada p*_c(b)

Lê os JSON de limiar_dinamico_v3_moore1_eps1e-05_n09/ (gerados por
varredura_limiar_dinamico.py) e não simula nada. A análise --- cruzamento, meia altura,
extrapolação e bootstrap acoplado --- é a do antigo analise_experimento_A.py, que este
script substitui; a tabela sai em analise_tabela.txt, dentro da pasta de dados.
Leva cerca de dois minutos (bootstrap).
"""
import glob
import json
import os
from fractions import Fraction

import numpy as np

from matplotlib.ticker import MaxNLocator

import estilo as E

E.aplicar()

N_BOOT = 400
SEMENTE_BOOT = 7
LIMIAR_GEOMETRICO = 0.592746          # percolação de sítios, vizinhança de Moore (buracos)
LIMIARES_KM = sorted({Fraction(k, m) for m in range(1, 9) for k in range(9) if 1 < Fraction(k, m) < 2})
PASTA = os.path.join(E.AQUI, 'limiar_dinamico_v3_moore1_eps1e-05_n09')


# ==============================================================================
# LEITURA
# ==============================================================================
def carregar(pasta):
    """{b: {'ps': [...], 'Ls': [...], 'alc': {(L,p): array}, 'frac': {(L,p): array}}}"""
    dados = {}
    for arq in sorted(glob.glob(os.path.join(pasta, 'resultados_b*.json'))):
        b = float(os.path.basename(arq).split('_b')[1][:-5])
        d = json.load(open(arq))
        ent = [e for k, e in d.items() if not k.startswith('_')]
        Ls = sorted({e['L'] for e in ent})
        ps = sorted(set.intersection(*[{round(e['p'], 6) for e in ent if e['L'] == L} for L in Ls]))
        alc, frac = {}, {}
        for e in ent:
            p = round(e['p'], 6)
            if p in ps:
                alc[(e['L'], p)] = np.array(e['alc'])
                frac[(e['L'], p)] = np.array(e['frac'])
        dados[b] = dict(ps=ps, Ls=Ls, alc=alc, frac=frac, meta=d.get('_meta', {}))
    return dados


# ==============================================================================
# ESTIMADORES
# ==============================================================================
def cruzamento(ps, y1, y2):
    """p onde y2 - y1 troca de sinal (+ -> -), por interpolação linear."""
    d = np.asarray(y2) - np.asarray(y1)
    for i in range(len(d) - 1):
        if d[i] > 0 >= d[i + 1]:
            return float(ps[i] + d[i] * (ps[i + 1] - ps[i]) / (d[i] - d[i + 1]))
    return None


def meia_altura(ps, y):
    """primeiro p em que y cai abaixo de 1/2, por interpolação linear."""
    for i in range(len(y) - 1):
        if y[i] >= 0.5 > y[i + 1]:
            return float(ps[i] + (y[i] - 0.5) * (ps[i + 1] - ps[i]) / (y[i] - y[i + 1]))
    return None


def banda(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (float(np.percentile(v, 16)), float(np.percentile(v, 84))) if len(v) > 10 else (np.nan, np.nan)


def lei_potencia(L, pc, a, x):
    return pc + a * np.asarray(L, float) ** (-x)


X_GRADE = np.arange(0.20, 3.001, 0.005)


def extrapolar(Ls, y):
    """Ajusta y(L) = p*_c + a L^(-x)."""
    L = np.asarray(Ls, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)) or len(y) < 3:
        return (np.nan, np.nan, np.nan)
    melhor = (np.inf, np.nan, np.nan, np.nan)
    for x in X_GRADE:
        M = np.column_stack([np.ones_like(L), L ** (-x)])
        (pc, a), *_ = np.linalg.lstsq(M, y, rcond=None)
        r = float(np.sum((M @ [pc, a] - y) ** 2))
        if r < melhor[0] and 0.0 <= pc <= y.min() and a > 0:
            melhor = (r, float(pc), float(a), float(x))
    return melhor[1:] if np.isfinite(melhor[1]) else (np.nan, np.nan, np.nan)


def analisar_b(d, indecisos_como_alcance=False):
    """Estimadores para um valor de b, com bandas por bootstrap nas sementes."""
    ps, Ls = d['ps'], d['Ls']
    alvo = (lambda a: (a == 1) | (a == 2)) if indecisos_como_alcance else (lambda a: a == 1)
    A, F = {}, {}
    for L in Ls:
        n = min(len(d['alc'][(L, p)]) for p in ps)
        A[L] = np.array([alvo(d['alc'][(L, p)][:n]) for p in ps], float)
        F[L] = np.array([d['frac'][(L, p)][:n] for p in ps])
    if all(A[L].sum() == 0 for L in Ls):
        return dict(sem_invasao=True, Ls=Ls, ps=ps)

    rng = np.random.default_rng(SEMENTE_BOOT)
    idx = {L: [rng.integers(0, A[L].shape[1], A[L].shape[1]) for _ in range(N_BOOT)] for L in Ls}

    meia = {}
    for L in Ls:
        c = meia_altura(ps, A[L].mean(1))
        lo, hi = banda([meia_altura(ps, A[L][:, i].mean(1)) for i in idx[L]])
        meia[L] = dict(p=c, lo=lo, hi=hi)

    cruz = {}
    for a, c in zip(Ls[:-1], Ls[1:]):
        x = cruzamento(ps, F[a].mean(1), F[c].mean(1))
        lo, hi = banda([cruzamento(ps, F[a][:, i].mean(1), F[c][:, j].mean(1))
                        for i, j in zip(idx[a], idx[c])])
        cruz[(a, c)] = dict(p=x, lo=lo, hi=hi)

    y = [meia[L]['p'] for L in Ls]
    sup, amp, expo = (extrapolar(Ls, y) if all(v is not None for v in y) else (np.nan, np.nan, np.nan))
    sup_boot, x_boot = [], []
    if all(v is not None for v in y):
        for k in range(N_BOOT):
            yk = [meia_altura(ps, A[L][:, idx[L][k]].mean(1)) for L in Ls]
            if all(v is not None for v in yk):
                pk, _, xk = extrapolar(Ls, yk)
                sup_boot.append(pk)
                x_boot.append(xk)
    sup_lo, sup_hi = banda(sup_boot)
    # incerteza do expoente x: mesma reamostragem de sementes (percentis 16 e 84); a fração de
    # reamostragens em que x bate no limite da grade (0,2 ou 3,0) indica x mal determinado
    x_lo, x_hi = banda(x_boot)
    xb = np.array([v for v in x_boot if np.isfinite(v)])
    x_borda = float(np.mean((xb <= X_GRADE[0] + 1e-9) | (xb >= X_GRADE[-1] - 1e-9))) if len(xb) else np.nan

    inf = cruz[(Ls[-2], Ls[-1])]
    res = dict(sem_invasao=False, Ls=Ls, ps=ps, meia=meia, cruz=cruz, expoente=expo, amplitude=amp,
               expoente_lo=x_lo, expoente_hi=x_hi, expoente_borda=x_borda,
               inferior=inf, superior=dict(p=sup, lo=sup_lo, hi=sup_hi))
    if inf['p'] is not None and np.isfinite(sup):
        res['pc'] = 0.5 * (inf['p'] + sup)
        res['sist'] = 0.5 * abs(sup - inf['p'])
        est = [x for x in [inf['hi'] - inf['p'] if np.isfinite(inf['hi']) else np.nan,
                           sup_hi - sup if np.isfinite(sup_hi) else np.nan] if np.isfinite(x)]
        res['estat'] = float(np.max(est)) if est else np.nan
    else:
        res['pc'] = sup
        res['sist'] = np.nan
        res['estat'] = sup_hi - sup if np.isfinite(sup_hi) else np.nan
    res['indecisos'] = float(np.mean([np.mean(d['alc'][(L, p)] == 2) for L in Ls for p in ps]))
    return res


def intervalo_km(b):
    """(k/m abaixo, k/m acima) que delimitam b."""
    ab = max([x for x in LIMIARES_KM if x < b], default=Fraction(1))
    ac = min([x for x in LIMIARES_KM if x > b], default=Fraction(2))
    return ab, ac



# ==============================================================================
# FIGURA 5.2 — as duas cotas contra 1/L
# ==============================================================================
def fig_convergencia(res):
    bs = [b for b in sorted(res) if not res[b]['sem_invasao']]
    ncol = 4
    nlin = int(np.ceil(len(bs) / ncol))
    fig, axs = E.figura(nlin, ncol, altura=4.4 * nlin, sharex=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, b in zip(axs, bs):
        r = res[b]
        Ls = np.array(r['Ls'], float)
        y = np.array([r['meia'][L]['p'] for L in r['Ls']], float)
        e = np.clip([[y[i] - r['meia'][L]['lo'], r['meia'][L]['hi'] - y[i]]
                     for i, L in enumerate(r['Ls'])], 0, None).T
        ax.errorbar(1 / Ls, y, yerr=e, fmt='s', mfc='white', color=E.VERMELHO, ms=3,
                    label=r'$p_{1/2}(L)$')
        if np.isfinite(r['expoente']):
            xx = np.linspace(1e-4, 1 / Ls.min() * 1.05, 200)
            ax.plot(xx, lei_potencia(1 / xx, r['superior']['p'], r['amplitude'], r['expoente']),
                    color=E.VERMELHO, lw=0.9, ls='--')
        chaves = list(r['cruz'])
        xs = [2 / (a + c) for a, c in chaves]                  # 1/L médio do par
        ys = [r['cruz'][k]['p'] for k in chaves]
        ok = [i for i, v in enumerate(ys) if v is not None]
        if ok:
            ee = np.clip([[ys[i] - r['cruz'][chaves[i]]['lo'], r['cruz'][chaves[i]]['hi'] - ys[i]]
                          for i in ok], 0, None).T
            ax.errorbar([xs[i] for i in ok], [ys[i] for i in ok], yerr=ee, fmt='o',
                        color=E.AZUL, ms=3, label='cruzamento do território')
        if np.isfinite(r['pc']):
            ax.axhline(r['pc'], color=E.PRETO, lw=0.8, ls=':')
            if np.isfinite(r['sist']):
                ax.axhspan(r['pc'] - r['sist'], r['pc'] + r['sist'], color=E.CINZA, alpha=0.3, lw=0)
        ax.set_title(rf'$b={E.num(b)}$')          # o intervalo k/m está na Tabela 5.1
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(ls=':')
        ax.set_xlim(left=0)
        E.virgula(ax)
    for ax in axs[len(bs):]:
        ax.axis('off')
    for i, ax in enumerate(axs[:len(bs)]):
        if i >= len(bs) - ncol:
            ax.set_xlabel(r'$1/L$')
            ax.tick_params(labelbottom=True)
        if i % ncol == 0:
            ax.set_ylabel(r'Fração de buracos $p$')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    alcas = [Line2D([], [], marker='s', mfc='white', color=E.VERMELHO, ls='', ms=3,
                    label=r'$p_{1/2}(L)$ (cota superior)'),
             Line2D([], [], color=E.VERMELHO, lw=0.9, ls='--',
                    label=r'ajuste $p_{1/2}=p^*_c+aL^{-x}$'),
             Line2D([], [], marker='o', color=E.AZUL, ls='', ms=3,
                    label=r'cruzamento do território' '\n' r'(cota inferior)'),
             Line2D([], [], color=E.PRETO, lw=0.8, ls=':', label=r'$p^*_c(b)$'),
             Patch(color=E.CINZA, alpha=0.3, lw=0, label=r'incerteza sistemática')]
    livres = axs[len(bs):]
    if len(livres):
        # a legenda ocupa o espaço dos painéis vazios da última linha; é posta depois do
        # layout (fig.legend fora do constrained_layout) para não deformar a grade
        fig.canvas.draw()
        x0 = min(a.get_position().x0 for a in livres)
        x1 = max(a.get_position().x1 for a in livres)
        y0 = min(a.get_position().y0 for a in livres)
        y1 = max(a.get_position().y1 for a in livres)
        fig.legend(handles=alcas, loc='center', bbox_to_anchor=((x0 + x1) / 2, (y0 + y1) / 2),
                   fontsize=E.TAM_NUMERO, ncol=1)
    else:
        axs[0].legend(handles=alcas, loc='lower right')
    E.salvar(fig, 'analise_1_convergencia_L')


# ==============================================================================
# FIGURA 5.3 — a escada p*_c(b)
# ==============================================================================
def fig_diagrama(res):
    fig, ax = E.figura(altura=8.0)
    for x in LIMIARES_KM:
        ax.axvline(float(x), color='#e0e0e0', lw=0.7, zorder=0)
        ax.text(float(x), 0.612, rf'$\dfrac{{{x.numerator}}}{{{x.denominator}}}$', ha='center',
                va='bottom', fontsize=E.TAM_NUMERO, color=E.CINZA_ESCURO)
    bs = sorted(res)
    xs, ys, es = [], [], []
    for b in bs:
        r = res[b]
        if r['sem_invasao']:
            xs.append(b); ys.append(0.0); es.append(0.0)
        elif np.isfinite(r['pc']):
            xs.append(b); ys.append(r['pc']); es.append(np.nansum([r['sist'], r['estat']]))
        y = 0.0 if r['sem_invasao'] else r['pc']
        if np.isfinite(y):
            ab, ac = intervalo_km(b)
            ax.hlines(y, float(ab), float(ac), color=E.PRETO, lw=2.2, alpha=0.3, zorder=1)
    prim = True
    for b in bs:
        r = res[b]
        if r['sem_invasao']:
            ax.annotate('sem invasão', (b, 0.0), textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=E.TAM_ANOTACAO, color=E.CINZA_ESCURO)
            continue
        if r['inferior']['p'] is not None:
            ax.plot(b, r['inferior']['p'], '_', color=E.AZUL, ms=8, mew=1.4, zorder=3,
                    label='cota inferior (cruzamento)' if prim else None)
        if np.isfinite(r['superior']['p']):
            ax.plot(b, r['superior']['p'], '_', color=E.VERMELHO, ms=8, mew=1.4, zorder=3,
                    label=r'cota superior ($p_{1/2}$ extrapolado)' if prim else None)
        prim = False
    ax.errorbar(xs, ys, yerr=es, fmt='o', color=E.PRETO, ms=3.5, zorder=4,
                label=r'$p_c^*(b)$ (barra larga: intervalo $k/m$)')
    ax.axhline(LIMIAR_GEOMETRICO, color=E.CINZA_ESCURO, ls='--', lw=0.9, zorder=1,
               label=rf'limiar geométrico (Moore), $p_c={E.num(LIMIAR_GEOMETRICO)}$')
    ax.set_xlabel(r'Tentação $b$')
    ax.set_ylabel(r'Fração de buracos $p$')
    ax.set_xlim(0.98, 1.78)
    ax.set_ylim(-0.02, 0.715)
    ax.grid(ls=':')
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='white')
    E.virgula(ax)
    E.salvar(fig, 'analise_2_diagrama_pc_b')


# ==============================================================================
# FIGURA 5.1 — probabilidade de alcance
# ==============================================================================
def fig_alcance(res, dados, bs):
    fig, axs = E.figura(1, len(bs), altura=5.8, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, bs):
        d, r = dados[b], res[b]
        cores, marcas = E.cores_tamanhos(r['Ls']), E.marcas_tamanhos(r['Ls'])
        for L in r['Ls']:
            P = [np.mean(d['alc'][(L, p)] == 1) for p in r['ps']]
            ax.plot(r['ps'], P, marker=marcas[L], color=cores[L], ms=2.8, label=rf'$L={L}$')
        if np.isfinite(r['pc']):
            if np.isfinite(r['sist']):
                ax.axvspan(r['pc'] - r['sist'], r['pc'] + r['sist'], color=E.CINZA, alpha=0.3, lw=0)
            ax.axvline(r['pc'], color=E.PRETO, ls=':', lw=0.8)
        ax.set_xlabel(r'Fração de buracos $p$')
        ax.set_title(rf'$b={E.num(b)}$')
        ax.grid(ls=':')
        E.virgula(ax)
    axs[0].set_ylabel(r'Probabilidade de alcance $P_{\mathrm{alc}}$')
    axs[0].legend(loc='lower left')
    E.salvar(fig, 'analise_3_alcance')


# ==============================================================================
# TABELA 5.1
# ==============================================================================
def tabela(res, res_alt):
    cab = (f"{'b':>7} | {'intervalo k/m':>13} | {'inferior':>9} | {'superior':>9} | "
           f"{'p*_c':>15} | {'x':>5} | {'x (16%-84%)':>13} | {'borda':>5} | {'indec.':>7} | "
           f"{'p*_c (indec=alc)':>16}")
    linhas = ["=" * len(cab), "LIMIAR DINÂMICO DE INVASÃO — ESTIMATIVAS FINAIS", "=" * len(cab),
              cab, "-" * len(cab)]
    for b in sorted(res):
        r, ra = res[b], res_alt[b]
        ab, ac = intervalo_km(b)
        km = f"{ab.numerator}/{ab.denominator}–{ac.numerator}/{ac.denominator}"
        if r['sem_invasao']:
            linhas.append(f"{b:>7} | {km:>13} |   sem invasão em nenhum p da grade  ->  p*_c = 0")
            continue
        inf = r['inferior']['p']
        sinf = f"{inf:.4f}" if inf is not None else "--"
        linhas.append(f"{b:>7} | {km:>13} | {sinf:>9} | {r['superior']['p']:>9.4f} | "
                      f"{r['pc']:.4f} ± {np.nansum([r['sist'], r['estat']]):.4f} | "
                      f"{r['expoente']:5.2f} | {r['expoente_lo']:5.2f}--{r['expoente_hi']:5.2f} | "
                      f"{100 * r['expoente_borda']:4.0f}% | {100 * r['indecisos']:6.2f}% | {ra['pc']:16.4f}")
    linhas += ["-" * len(cab),
               "inferior = cruzamento do território entre os dois maiores L (sobe com L)",
               "superior = extrapolação de p_{1/2}(L) = p*_c + a L^(-x) (desce com L)",
               "p*_c = ponto médio; incerteza = sistemática (metade da distância) + estatística",
               "x (16%-84%): percentis do expoente sobre 400 reamostragens de sementes;",
               "borda: fração das reamostragens em que x bate no limite da grade [0,2; 3,0]",
               "última coluna: mesma estimativa contando os indecisos como alcance (cota)",
               "=" * len(cab)]
    txt = "\n".join(linhas)
    print(txt)
    open(os.path.join(PASTA, 'analise_tabela.txt'), 'w').write(txt + "\n")


if __name__ == "__main__":
    dados = carregar(PASTA)
    print(f"  {len(dados)} valores de b, tamanhos {dados[min(dados)]['Ls']}")
    res = {b: analisar_b(d) for b, d in dados.items()}
    res_alt = {b: analisar_b(d, indecisos_como_alcance=True) for b, d in dados.items()}
    invadem = [b for b in sorted(res) if not res[b]['sem_invasao']]
    fig_alcance(res, dados, [invadem[0], invadem[len(invadem) // 2], invadem[-1]])
    fig_convergencia(res)
    fig_diagrama(res)
    tabela(res, res_alt)
