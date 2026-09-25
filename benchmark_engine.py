"""
benchmark_engine.py — mede o custo computacional do engine_v2 para a Seção 3.2.

Coloque este arquivo NA MESMA PASTA do engine_v2.py e rode:

    python benchmark_engine.py

Ele imprime um bloco de texto pronto para ser colado na conversa/dissertação.
Não gera quadros nem arquivos de dados permanentes (usa uma pasta temporária).
"""
import os
import platform
import shutil
import sys
import time
import contextlib
import io
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_v2 as M


def silencioso(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def limpar():
    for d in glob.glob(os.path.join(M.BASE_DIR, 'v2_*')):
        shutil.rmtree(d, ignore_errors=True)


def configurar(L, passos, invasao=True, mu=1e-4):
    M.TAMANHOS_REDE = [L]
    M.STEPS = passos
    M.CONFIG_SEMENTES = [12345]
    M.GERAR_FRAMES = False
    M.CONTINUAR_SIMULACAO = False
    M.CHECKPOINT_SKIP = 10 ** 9          # sem gravar checkpoints no meio
    M.PARADA_ANTECIPADA = False
    M.TAXA_MUTACAO_C = M.TAXA_MUTACAO_D = mu
    M.K_FERMI = 0.0
    M.PROBABILIDADE_ATUALIZACAO = 1.0
    M.PROPORCAO_RANDOM_BURACOS = 0.407254
    M.b = 1.3
    M.EPSILON_P = 0.001
    if invasao:
        M.PROPORCAO_RANDOM_COOPERADORES = 0.0
        M.ESTRATEGIA_MAR, M.ESTRATEGIA_INVASOR = 1, 0
        M.CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": 9}]
    else:
        M.PROPORCAO_RANDOM_COOPERADORES = 0.5
        M.CONFIG_INVASORES = []


def cronometrar(L, passos, invasao=True):
    configurar(L, passos, invasao)
    limpar()
    t0 = time.perf_counter()
    silencioso(M.executar_simulacao)
    dt = time.perf_counter() - t0
    limpar()
    return dt


def main():
    print('=' * 68)
    print('AMBIENTE')
    print('=' * 68)
    print(f'processador      : {platform.processor() or platform.machine()}')
    print(f'sistema          : {platform.platform()}')
    print(f'python           : {platform.python_version()}')
    print(f'numpy            : {np.__version__}')
    try:
        import numba
        print(f'numba            : {numba.__version__}')
        print(f'threads do numba : {numba.get_num_threads()}')
    except Exception as e:                                    # noqa: BLE001
        print(f'numba            : indisponível ({e})')
    print(f'núcleos lógicos  : {os.cpu_count()}')

    print('\n' + '=' * 68)
    print('COMPILAÇÃO (primeira chamada)')
    print('=' * 68)
    frio = cronometrar(51, 5, invasao=True)
    quente = cronometrar(51, 5, invasao=True)
    print(f'5 passos a frio  : {frio:.2f} s')
    print(f'5 passos a quente: {quente:.2f} s')
    print(f'compilação       : {frio - quente:.2f} s')

    print('\n' + '=' * 68)
    print('CUSTO POR PASSO')
    print('=' * 68)
    print(f'{"protocolo":<22}{"L":>6}{"passos":>8}{"tempo":>10}{"ms/passo":>11}{"ns/sítio/passo":>16}')
    for invasao, nome in ((True, 'invasão (colônia)'), (False, 'cond. inicial aleat.')):
        for L, passos in ((101, 2000), (201, 1000), (401, 300)):
            dt = cronometrar(L, passos, invasao)
            ms = 1e3 * dt / passos
            ns = 1e9 * dt / (passos * L * L)
            print(f'{nome:<22}{L:>6}{passos:>8}{dt:>9.1f}s{ms:>11.2f}{ns:>16.1f}')

    print('\n' + '=' * 68)
    print('EXTRAPOLAÇÃO')
    print('=' * 68)
    dt = cronometrar(201, 1000, invasao=True)
    print(f'rodada de 20000 passos em L=201: ~{20000 * dt / 1000 / 60:.1f} min por semente')

    print('\n' + '=' * 68)
    print('VARREDURA DETERMINÍSTICA (mu = 0, com parada antecipada)')
    print('=' * 68)
    M.PARADA_ANTECIPADA = True
    M.PERIODO_MAX = getattr(M, 'PERIODO_MAX', 16)
    passos_ate_travar = []
    tempos = []
    for L in (101, 201):
        configurar(L, 3000, invasao=True, mu=0.0)
        M.PARADA_ANTECIPADA = True
        tot = 0.0
        n = 5
        for s in range(n):
            M.CONFIG_SEMENTES = [2000 + s]
            limpar()
            t0 = time.perf_counter()
            saida = io.StringIO()
            with contextlib.redirect_stdout(saida):
                M.executar_simulacao()
            tot += time.perf_counter() - t0
            for linha in saida.getvalue().splitlines():
                if 'ciclo de período' in linha:
                    passos_ate_travar.append(int(linha.split('em t=')[1].split(':')[0]))
            limpar()
        tempos.append((L, tot / n))
        print(f'L={L}: {tot / n:.2f} s por realização (média de {n})')
    if passos_ate_travar:
        print(f'passos até travar: mediana {int(np.median(passos_ate_travar))}, '
              f'máximo {max(passos_ate_travar)} (em {len(passos_ate_travar)} realizações que travaram)')
    else:
        print('nenhuma realização travou dentro do limite: aumente PERIODO_MAX ou STEPS')
    print('\nCopie tudo acima.')


if __name__ == '__main__':
    main()
