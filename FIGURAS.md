# Scripts das figuras

Cada figura da dissertação tem um script com o **número da figura no nome**:
`fig_<capítulo>_<número>_<assunto>.py`. Todos importam `estilo.py` (paleta, tipografia,
tamanho final na página) e gravam em `../figuras/`, a pasta lida pelo LaTeX.

    python gerar_figuras.py            # refaz todas as figuras
    python gerar_figuras.py 2_1 5_0    # só as que casam com os trechos dados
    python gerar_figuras.py evento     # idem

Dependências: `numpy`, `scipy`, `matplotlib` com LaTeX instalado (as figuras usam `usetex`),
`pandas` (Fig. 4.1 e D.1) e `sympy` (só `verificacoes_apendices.py`). Nenhum script de
figura depende do Numba: os que simulam usam `nucleo_numpy.py`, equivalente passo a passo ao
motor com K = 0.

## Figura -> script -> arquivo

| figura | script | arquivo gerado | tempo |
|---|---|---|---|
| 2.1 campo médio | `fig_2_01_campo_medio.py` | `campo_medio.pdf` | s |
| 2.2 simplex e fluxo do replicador | `fig_2_02_simplex.py` | `figura_simplex.pdf` | s |
| 2.3 replicador no DP | `fig_2_03_replicador.py` | `replicador_dp.pdf` | s |
| 2.4 vizinhanças | `fig_2_04_vizinhancas.py` | `vizinhancas.pdf` | s |
| 2.5 contorno periódico | `fig_2_05_06_contornos.py` | `Figura_Toroide.pdf` | s |
| 2.6 contorno de parede | `fig_2_05_06_contornos.py` | `Figura_Parede.pdf` | (mesmo) |
| 2.7 imitar o melhor | `fig_2_07_08_regras_revisao.py` | `Figura_Imitate_Best.pdf` | s |
| 2.8 regra de Fermi | `fig_2_07_08_regras_revisao.py` | `Figura_Regra_Fermi.pdf` | (mesmo) |
| 2.9 esquemas de atualização | `fig_2_09_atualizacao.py` | `sincrono_assincrono_real.pdf` | s |
| 2.10 limiares k/m | `fig_2_10_limiares_km.py` | `limiares_km.pdf` | s |
| 2.11 aglomerados de Moore | `fig_2_11_12_percolacao.py` | `aglomerados_moore.pdf` | ~2 min* |
| 2.12 probabilidade de cruzamento | `fig_2_11_12_percolacao.py` | `cruzamento_percolacao.pdf` | (mesmo) |
| 2.13 distância química | `fig_2_13_distancia_quimica.py` | `distancia_quimica.pdf` | ~2 min* |
| 2.14 força local na frente | `fig_2_14_forca_frente.py` | `forca_local_frente.pdf` | ~2 min* |
| 2.15 espalhamento da queima | `fig_2_15_espalhamento_queima.py` | `espalhamento_queima.pdf` | ~3 min* |
| 3.1 observáveis da colônia | `fig_3_01_observaveis.py` | `observaveis_colonia.pdf` | ~1 min* |
| 4.1 validação (Nowak–May) | `fig_4_01_validacao_nowak.py` | `validacao_rho_b.pdf` | s |
| 5.1 probabilidade de alcance | `fig_5_01_03_limiar_dinamico.py` | `analise_3_alcance.pdf` | ~2 min |
| 5.2 convergência das cotas | `fig_5_01_03_limiar_dinamico.py` | `analise_1_convergencia_L.pdf` | (mesmo) |
| 5.3 diagrama p\*_c(b) | `fig_5_01_03_limiar_dinamico.py` | `analise_2_diagrama_pc_b.pdf` | (mesmo) |
| 5.4 razões críticas | `fig_5_04_razoes_criticas.py` | `mecanismo_1_razoes.pdf` | s |
| 5.5 frente contra queima | `fig_5_05_frente_queima.py` | `espalhamento_1_series.pdf` | s |
| 5.6 escape antes da nucleação | `fig_5_06_mutacao_nucleacao.py` | `mutacao_3_diagrama.pdf` | s |
| 5.7 φ e τ_a | `fig_5_07_08_evento_raro.py` | `evento_raro_1_intensivas.pdf` | s |
| 5.8 avalanches e previsão | `fig_5_07_08_evento_raro.py` | `evento_raro_2_avalanches.pdf` | (mesmo) |
| D.1 diagnóstico | `fig_D_01_diagnostico.py` | `diagnostico_semente44_L201.pdf` | s (dados externos) |

\* só na primeira vez: os scripts que simulam guardam o resultado em `dados_*.json`/`.npz`
junto com os parâmetros usados, e só recalculam se algum parâmetro mudar.

Os números das figuras podem mudar se o texto ganhar figuras novas; nesse caso, basta
renomear o script. O nome do **arquivo gerado** é o que o LaTeX usa e não deve mudar.

## Números que os scripts imprimem

Além da figura, cada script imprime os números que o texto cita a partir dela:

| script | imprime |
|---|---|
| `fig_2_13` | d_min medido (Seção 2.5.1, Apêndice A.8.5) |
| `fig_2_15` | expoentes η, δ e de R² da queima |
| `fig_3_01` | N_col, R², R_g², ℓ_max, alcance da legenda da Fig. 3.1 |
| `fig_4_01` | ρ_C médio, fração de sementes extintas, média ± EP das sobreviventes (Seção 4.2) |
| `fig_5_01_03` | Tabela 5.1 (grava também `analise_tabela.txt` na pasta de dados) |
| `fig_5_04` | Tabela 5.3 |
| `fig_5_05` | Tabela 5.4 e velocidades da Seção 5.3 |
| `fig_5_06` | P(escape antes da nucleação) por (p, μ) |
| `fig_5_07_08` | Tabela 5.7, expoentes das avalanches, previsão/medida (grava `evento_raro_tabela.txt`) |

Outras verificações: `verificacoes_apendices.py` (Apêndices A e B; `python
verificacoes_apendices.py A5`, por exemplo, roda só uma seção) e `sobrevivencia_queima.py`
(δ da queima, Apêndice A.8.3).

## Dados

| pasta | gerada por | usada por |
|---|---|---|
| `limiar_dinamico_v3_moore1_eps1e-05_n09/` | `varredura_limiar_dinamico.py` | Figs. 5.1–5.3 |
| `mecanismo_v1_moore1_eps1e-05_n09/` | `mecanismo_fronteira.py` | Fig. 5.4 |
| `espalhamento_v1_b1.0700_moore1_n09/` | `espalhamento_frente.py` | Fig. 5.5 |
| `mutacao_v1_moore1_eps1e-05_n09/` | `varredura_mutacao.py` | Figs. 5.6 e 5.8 |
| `evento_raro_v1_moore1_eps1e-05_n09/` | `evento_raro.py` | Figs. 5.7 e 5.8 |
| `dados/resumo_validacao.csv` | `resumo_validacao.py` | Fig. 4.1 |

Os programas de experimento (`varredura_*.py`, `mecanismo_fronteira.py`,
`espalhamento_frente.py`, `evento_raro.py`) rodam junto com `engine_v2.py` e também gravam
figuras **de trabalho** dentro das próprias pastas de dados. Essas não são as da
dissertação; as da dissertação são sempre as dos scripts `fig_*.py`.

## Diagnóstico do Apêndice D

`fig_D_01_diagnostico.py` precisa de dados que não estão nesta pasta:

1. copie o `engine.py` **original** para `diagnostico/` e rode `python diagnostico/run_diag.py`
   (semente 44, L = 201, 20000 passos) -> `diagnostico/diag_s44.npz`;
2. `python diagnostico/run_mu.py 0 300 0.0` -> `diagnostico/diag_mu0_p0.npz`;
3. rode o `engine_v2.py` com semente 44, L = 201, `STEPS = 20000`, `b = 1.2`,
   `EPSILON_P = 0.01`, `PROPORCAO_RANDOM_BURACOS = 0.407254` e μ = 0, 1e-5, 1e-4, 1e-3;
   aponte `PASTA_V2` (no script ou como variável de ambiente) para onde ficaram as pastas
   `v2_...`.

Sem esses arquivos o script avisa e é pulado; o LaTeX continua usando o
`diagnostico_semente44_L201.png` antigo. Depois de gerar o PDF, apague o PNG.

## Pasta `scripts_antigos/`

Scripts anteriores ao módulo de estilo. Nenhuma figura da dissertação depende deles; os que
geravam figuras usadas foram reescritos como `fig_2_01` a `fig_2_09`.
