"""
resumo_validacao.py — condensa as rodadas de validação num único arquivo pequeno.

Coloque na mesma pasta onde estão as pastas de experimento (v2_...) e rode:

    python resumo_validacao.py

Gera 'resumo_validacao.csv' com uma linha por semente, contendo os parâmetros lidos do
nome da pasta, a fração estacionária de cooperadores e o instante em que a série passou a
se repetir (quando houve parada antecipada). É esse arquivo, de poucos kB, que deve ser
enviado.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

JANELA = 500          # passos finais usados na média estacionária
AQUI = os.path.dirname(os.path.abspath(__file__))


def parametros(nome):
    def pega(chave):
        m = re.search(chave + r'([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)', nome)
        return float(m.group(1)) if m else np.nan
    return dict(b=pega('_b'), eps=pega('_eps'), mu=pega('_mutC'),
                p=pega('_pbur'), pcoop=pega('_pcoop'))


def periodo_da_serie(x, pmax=64):
    """Menor P tal que os últimos valores se repetem com período P (None se não houver)."""
    n = len(x)
    if n < 4 * pmax:
        return None
    cauda = x[-2 * pmax:]
    for P in range(1, pmax + 1):
        if np.allclose(cauda[P:], cauda[:-P], atol=1e-12):
            return P
    return None


linhas = []
for pasta in sorted(glob.glob(os.path.join(AQUI, 'v2_*'))):
    par = parametros(os.path.basename(pasta))
    for arq in sorted(glob.glob(os.path.join(pasta, 'data', 'L_*', 'log_semente_*.csv'))):
        L = int(re.search(r'L_(\d+)', arq).group(1))
        semente = int(re.search(r'log_semente_(\d+)', arq).group(1))
        try:
            d = pd.read_csv(arq, sep=';')
        except Exception as e:                                  # noqa: BLE001
            print(f'  [!] ignorado {os.path.basename(arq)}: {e}')
            continue
        rho = d['rho_c'].values
        janela = rho[-min(JANELA, len(rho)):]
        # instante a partir do qual a série é constante (ciclo detectado e repetido)
        P = periodo_da_serie(rho)
        linhas.append(dict(**par, L=L, semente=semente, passos=len(rho) - 1,
                           rho_medio=janela.mean(), rho_dp=janela.std(),
                           rho_min=janela.min(), rho_max=janela.max(),
                           periodo_final=P if P else np.nan,
                           atividade_final=d['atividade'].values[-min(JANELA, len(d)):].mean()))

df = pd.DataFrame(linhas)
saida = os.path.join(AQUI, 'resumo_validacao.csv')
df.to_csv(saida, sep=';', index=False)
print(f'{len(df)} realizações resumidas em {saida}\n')
if len(df):
    tab = df.groupby(['b', 'L'])['rho_medio'].agg(['count', 'mean', 'std'])
    print(tab.to_string())
