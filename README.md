# Dilema do Prisioneiro espacial em redes bidimensionais no limiar da percolação sob mutação

Código e dados da dissertação de mestrado de **Fernando José Zardinello Batistti**
(Programa de Pós-Graduação em Física Aplicada — PPGFISA, UNILA, 2026).
Orientador: Prof. Dr. Luciano Calheiros Lapas.

Este repositório tem tudo o que é preciso para reproduzir os resultados da dissertação:
- o modelo;
- os cinco experimentos numéricos;
- um script para cada figura do texto;
- os dados brutos que esses scripts leem.

## Instalação

```bash
git clone https://github.com/fernandobatistti/spatial-prisoners-dilemma.git
cd spatial-prisoners-dilemma
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

As figuras usam LaTeX para o texto (`text.usetex` do matplotlib), então é preciso ter uma
distribuição LaTeX instalada (TeX Live, MiKTeX ou MacTeX).

## Refazer as figuras e as tabelas

```bash
python gerar_figuras.py
```

As figuras são gravadas em `figuras/`. Os scripts dos experimentos do Capítulo 5 não simulam
nada: eles leem os dados gravados nas pastas `*_v1_*` e `*_v3_*`. Além da figura, cada script
imprime os números que o texto cita a partir dela. A tabela figura → script → arquivo está em
[`FIGURAS.md`](FIGURAS.md).

## Refazer os dados

Cada experimento é um programa que usa o motor `engine_v2.py` (acelerado com Numba) e grava
um arquivo JSON por valor da tentação `b`, com o desfecho de cada semente. Os programas podem
ser interrompidos e retomados.

| experimento | programa | seção | pasta de dados |
|---|---|---|---|
| A — limiar dinâmico de invasão | `varredura_limiar_dinamico.py` | 5.1 | `limiar_dinamico_v3_moore1_eps1e-05_n09/` |
| C — razões críticas da fronteira | `mecanismo_fronteira.py` | 5.2 | `mecanismo_v1_moore1_eps1e-05_n09/` |
| D — frente comparada à queima | `espalhamento_frente.py` | 5.3 | `espalhamento_v1_b1.0700_moore1_n09/` |
| B — mutação como taxa | `varredura_mutacao.py` | 5.4 | `mutacao_v1_moore1_eps1e-05_n09/` |
| E — mutação única (evento raro) | `evento_raro.py` | 5.5 | `evento_raro_v1_moore1_eps1e-05_n09/` |

Cada programa tem, no cabeçalho, a descrição do protocolo, dos parâmetros e do formato dos
arquivos que grava. As figuras que esses programas desenham dentro das pastas de dados são de
trabalho; as figuras da dissertação vêm sempre dos scripts `fig_*.py`.

## Outros arquivos

| arquivo | o que é |
|---|---|
| `nucleo_numpy.py` | implementação independente do modelo, só com NumPy; é equivalente passo a passo ao motor com `K_FERMI = 0` e é usada pelos scripts de figura do Capítulo 2 |
| `estilo.py` | paleta, tipografia e tamanho das figuras |
| `verificacoes_apendices.py` | verificações numéricas e simbólicas dos Apêndices A e B |
| `sobrevivencia_queima.py` | expoente de sobrevivência da queima (Apêndice A) |
| `resumo_validacao.py`, `dados/` | validação contra Nowak e May (Capítulo 4) |
| `diagnostico/` | diagnóstico da medida anterior (Apêndice D) |
| `dados_*.json`, `dados_*.npz` | caches das figuras do Capítulo 2; os scripts recalculam se os parâmetros mudarem |

## Como citar

Veja [`CITATION.cff`](CITATION.cff). O código está sob a licença MIT ([`LICENSE`](LICENSE)).
