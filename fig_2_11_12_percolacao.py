"""
Figuras 2.11 e 2.12 — percolação de sítios.

    2.11  aglomerados_moore.pdf       maior aglomerado de Moore abaixo, no e acima do limiar
    2.12  cruzamento_percolacao.pdf   probabilidade de cruzamento R_L (em função de p = 1 - q), von Neumann e Moore

A 2.12 usa 2000 realizações por ponto e fica em cache (percolacao_cruzamento.json, ~10 min
para recalcular). O cache guarda os parâmetros com que foi gerado e é refeito sozinho se
algum deles mudar.
"""
import os

import numpy as np
from scipy.ndimage import label

import estilo as E

E.aplicar()

S8 = np.ones((3, 3), int)                                  # Moore
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])           # von Neumann
SEMENTE = 7
PC = {'moore': 0.592746, 'neumann': 0.407254}              # em fração de buracos

rng = np.random.default_rng(SEMENTE)


# ==============================================================================
# 2.11 — aglomerados
# ==============================================================================
def fig_aglomerados(L=150, ps=(0.50, 0.592746, 0.66)):
    U = rng.random((L, L))
    fig, axs = E.figura(1, 3, altura=5.6)
    for ax, p in zip(axs, ps):
        occ = U >= p                                       # buraco com probabilidade p
        lab, _ = label(occ, structure=S8)
        tam = np.bincount(lab.ravel()); tam[0] = 0
        img = np.zeros((L, L, 3))
        img[occ] = [0.6, 0.6, 0.6]                         # ocupado: cinza
        img[lab == tam.argmax()] = [1.0, 0.0, 0.0]         # maior aglomerado: vermelho
        ax.imshow(img, interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])
        txt = r'p_c\approx' + E.num(round(p, 4)) if abs(p - PC['moore']) < 1e-6 else E.num(p, 2)
        ax.set_title(rf'$p={txt}$')
    E.salvar(fig, 'aglomerados_moore')


# ==============================================================================
# 2.12 — probabilidade de cruzamento
# ==============================================================================
def cruza(occ, S):
    """Existe aglomerado ligando a borda superior à inferior (bordas abertas)?"""
    lab, _ = label(occ, structure=S)
    return len(np.intersect1d(lab[0][lab[0] > 0], lab[-1][lab[-1] > 0])) > 0


PARAM = dict(Ls=[16, 32, 64, 128], n=2000, semente=SEMENTE,
             p_moore=[0.54, 0.64, 21], p_neumann=[0.36, 0.46, 21])


def dados_cruzamento():
    arq = os.path.join(E.AQUI, 'percolacao_cruzamento.json')
    res = E.cache_valido(arq, PARAM)
    if res is not None:
        return res
    print("  calculando a probabilidade de cruzamento (alguns minutos)...")
    res = {}
    for viz, S, (a, b, k) in (('moore', S8, PARAM['p_moore']),
                              ('neumann', S4, PARAM['p_neumann'])):
        ps = np.linspace(a, b, k)
        res[viz] = {str(L): [float(np.mean([cruza(rng.random((L, L)) >= p, S)
                                            for _ in range(PARAM['n'])])) for p in ps]
                    for L in PARAM['Ls']}
        res[viz]['p'] = ps.tolist()
    E.gravar_cache(arq, PARAM, res)
    return res


def fig_cruzamento():
    res = dados_cruzamento()
    cores = E.cores_tamanhos(PARAM['Ls'])
    marcas = E.marcas_tamanhos(PARAM['Ls'])
    fig, axs = E.figura(1, 2, altura=6.4)
    for ax, viz, tit in ((axs[0], 'neumann', r'(a) von Neumann ($z=4$)'),
                         (axs[1], 'moore', r'(b) Moore ($z=8$)')):
        for L in PARAM['Ls']:
            ax.plot(res[viz]['p'], res[viz][str(L)], marker=marcas[L], color=cores[L],
                    ms=3, label=rf'$L={L}$')
        pc = PC[viz]
        ax.axvline(pc, color=E.REFERENCIA, ls='--', lw=0.8)
        ax.text(pc - 0.0015, 0.03, rf'$p_c={E.num(pc)}$', rotation=90, va='bottom',
                ha='right', fontsize=E.TAM_ANOTACAO)
        ax.set_xlabel(r'Fração de buracos $p$')
        ax.set_ylabel(r'Probabilidade de cruzamento $R_L$')
        ax.set_title(tit)
        ax.grid(ls=':')
        ax.legend(loc='upper right')
        E.virgula(ax)
    E.salvar(fig, 'cruzamento_percolacao')


if __name__ == "__main__":
    fig_aglomerados()
    fig_cruzamento()
