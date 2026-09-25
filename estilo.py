"""
estilo.py — padrão visual único de todas as figuras da dissertação.

Todo script de figura começa com

    import estilo as E
    E.aplicar()

e desenha a figura com `E.figura(...)`, que já cria a figura NO TAMANHO EM QUE ELA APARECE
NA PÁGINA. Isso é o que garante tamanho de texto uniforme: 9 pt no código são 9 pt no PDF
impresso, em qualquer figura. (Desenhar a 14 polegadas e deixar o LaTeX reduzir para 16 cm
faz a fonte 13 virar 5 pt — era o que acontecia antes.)

Tamanhos (a legenda das figuras está em \\small, ~11 pt; o corpo do texto em 12 pt):
    rótulos de eixo e títulos de painel   9 pt
    números dos eixos e legendas          8 pt
    anotações dentro do gráfico           7,5 pt (mínimo)

Paleta: a do autor. Azul = cooperação / o que favorece a cooperação; vermelho = deserção /
o que a desfavorece; cinzas para o que é neutro ou de referência.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==============================================================================
# GEOMETRIA DA PÁGINA
# ==============================================================================
CM = 1 / 2.54
LARGURA = 16.0 * CM           # \textwidth: A4, margens de 3 cm e 2 cm  (= 6,30 pol)

# ==============================================================================
# PALETA
# ==============================================================================
AZUL = '#0000ff'
AZUL_MEDIO = '#6666ff'
AZUL_CLARO = '#b3b3ff'
AZUL_PALIDO = '#80b3ff'
AZUL_ESCURO = '#000080'
VERMELHO = '#ff0000'
VERMELHO_MEDIO = '#ff6666'
VERMELHO_CLARO = '#ffb3b3'
VERMELHO_ESCURO = '#800000'
PRETO = '#000000'
CINZA_ESCURO = '#333333'
CINZA = '#999999'
CINZA_MEDIO = '#888888'
CINZA_CLARO = '#cccccc'

COOPERADOR = AZUL
DESERTOR = VERMELHO
BURACO = PRETO
REFERENCIA = PRETO              # linhas de referência (p_c, previsões teóricas)

# Séries ordenadas de um parâmetro que favorece (início) ou desfavorece (fim) a cooperação:
# b crescente, p crescente. É o esquema da Figura 2.14, generalizado para n curvas.
_DIVERGENTE = {
    2: [AZUL, VERMELHO],
    3: [AZUL, CINZA, VERMELHO],
    4: [AZUL, AZUL_MEDIO, VERMELHO_MEDIO, VERMELHO],
    5: [AZUL, AZUL_MEDIO, CINZA, VERMELHO_MEDIO, VERMELHO],
    6: [AZUL, AZUL_MEDIO, '#8080c0', '#c08080', VERMELHO_MEDIO, VERMELHO],
    7: [AZUL, AZUL_MEDIO, '#8080c0', CINZA, '#c08080', VERMELHO_MEDIO, VERMELHO],
}


def serie(n):
    """n cores de azul a vermelho, passando pelo cinza — para b ou p crescentes."""
    if n not in _DIVERGENTE:
        raise ValueError(f"serie() definida para 2 a 7 curvas, pedido {n}")
    return list(_DIVERGENTE[n])


# Tamanhos de rede: sempre as mesmas cores para o mesmo L, em todas as figuras
# (é o esquema da Figura 2.12: tamanhos pequenos em cinza, os maiores em azul e vermelho).
_TAMANHOS = [CINZA_CLARO, CINZA_MEDIO, AZUL, VERMELHO]
MARCAS_TAMANHO = ['o', 's', 'D', '^']


def cores_tamanhos(Ls):
    """Cores para uma lista crescente de tamanhos de rede (até quatro)."""
    Ls = sorted(Ls)
    if len(Ls) > 4:
        raise ValueError("até quatro tamanhos de rede")
    return {L: _TAMANHOS[4 - len(Ls) + i] for i, L in enumerate(Ls)}


def marcas_tamanhos(Ls):
    Ls = sorted(Ls)
    return {L: MARCAS_TAMANHO[4 - len(Ls) + i] for i, L in enumerate(Ls)}


# ==============================================================================
# TIPOGRAFIA
# ==============================================================================
TAM_ROTULO = 9
TAM_NUMERO = 8
TAM_ANOTACAO = 7.5


def aplicar():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{amsmath}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': TAM_ROTULO,
        'axes.labelsize': TAM_ROTULO,
        'axes.titlesize': TAM_ROTULO,
        'figure.titlesize': TAM_ROTULO,
        'legend.fontsize': TAM_NUMERO,
        'legend.title_fontsize': TAM_NUMERO,
        'xtick.labelsize': TAM_NUMERO,
        'ytick.labelsize': TAM_NUMERO,
        'axes.titlepad': 4,
        'axes.labelpad': 2,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.1,
        'lines.markersize': 3.2,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        'legend.frameon': False,
        'legend.handlelength': 1.8,
        'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3,
        'legend.labelspacing': 0.3,
        'legend.columnspacing': 1.0,
        'errorbar.capsize': 1.8,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
        'figure.constrained_layout.hspace': 0.04,
        'figure.constrained_layout.wspace': 0.04,
    })


def figura(nlin=1, ncol=1, altura=5.5, largura=1.0, **kw):
    """Cria a figura no tamanho final.

    altura  em cm;  largura  como fração de \\textwidth (a mesma fração usada no
    \\includegraphics). Devolve (fig, axs), como plt.subplots.
    """
    return plt.subplots(nlin, ncol, figsize=(LARGURA * largura, altura * CM), **kw)


# ==============================================================================
# NÚMEROS COM VÍRGULA DECIMAL
# ==============================================================================
def num(x, casas=None):
    """Número formatado para LaTeX com vírgula decimal: num(0.355, 3) -> '0{,}355'."""
    s = f"{x:.{casas}f}" if casas is not None else f"{x:g}"
    return s.replace('.', '{,}').replace('-', '-')


class _Virgula(mticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', '{,}')


def virgula(*axs, eixos='xy'):
    """Troca o ponto decimal por vírgula nos números dos eixos lineares."""
    for ax in axs:
        for e in eixos:
            eixo = ax.xaxis if e == 'x' else ax.yaxis
            if eixo.get_scale() == 'linear':
                f = _Virgula(useMathText=True)
                f.set_useOffset(False)
                eixo.set_major_formatter(f)


def log_decimal(ax, eixo='y'):
    """Em eixo log que cobre menos de uma década, escreve 0,7 0,8 0,9 em vez de 7x10^-1."""
    e = ax.yaxis if eixo == 'y' else ax.xaxis
    e.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'${num(v)}$'))
    e.set_minor_formatter(mticker.FuncFormatter(lambda v, p: f'${num(v)}$'))


# ==============================================================================
# GRAVAÇÃO
# ==============================================================================
AQUI = os.path.dirname(os.path.abspath(__file__))
# Dentro da pasta do LaTeX (há um Dissertacao.tex ao lado), as figuras vão para ../figuras,
# que é onde o LaTeX as procura; no repositório, para ./figuras. PD_FIGURAS força outro lugar.
if os.environ.get('PD_FIGURAS'):
    FIGURAS = os.path.abspath(os.environ['PD_FIGURAS'])
elif os.path.exists(os.path.join(AQUI, '..', 'Dissertacao.tex')):
    FIGURAS = os.path.normpath(os.path.join(AQUI, '..', 'figuras'))
else:
    FIGURAS = os.path.join(AQUI, 'figuras')


def salvar(fig, nome, recortar=True):
    """Grava figuras/<nome>.pdf, vetorial, e fecha a figura.

    recortar=False grava a página exatamente do tamanho da figura (sem 'bbox tight'). Use nos
    esquemas com aspecto 1:1: o recorte justo os deixaria mais estreitos que a largura pedida,
    e o LaTeX, ao esticá-los de volta, aumentaria também o texto.
    """
    os.makedirs(FIGURAS, exist_ok=True)
    caminho = os.path.join(FIGURAS, f'{nome}.pdf')
    with plt.rc_context({'savefig.bbox': 'tight' if recortar else 'standard'}):
        fig.savefig(caminho, format='pdf')
    plt.close(fig)
    print(f"  -> {os.path.relpath(caminho)}")
    return caminho


# ==============================================================================
# CACHE COM CONTRATO
# ==============================================================================
def cache_valido(caminho_json, parametros):
    """Lê um cache JSON só se ele foi gravado com exatamente os mesmos parâmetros.

    Evita o erro silencioso de mudar um parâmetro no script e continuar desenhando os
    dados antigos. Devolve os dados, ou None se for preciso recalcular.
    """
    import json
    if not os.path.exists(caminho_json):
        return None
    d = json.load(open(caminho_json))
    if d.get('_parametros') != parametros:
        print(f"  [cache] {os.path.basename(caminho_json)} tem outros parâmetros: recalculando")
        return None
    return d


def gravar_cache(caminho_json, parametros, dados):
    import json
    dados = dict(dados)
    dados['_parametros'] = parametros
    json.dump(dados, open(caminho_json, 'w'))
