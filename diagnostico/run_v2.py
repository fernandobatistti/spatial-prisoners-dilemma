"""
Etapa 3 do diagnóstico (Apêndice D): roda o engine_v2.py na mesma rodada do gráfico original
(semente 44, L = 201, b = 1,2, epsilon = 0,01, p = 0,407254), com as quatro taxas de mutação
mu = 0, 1e-5, 1e-4, 1e-3, e 20000 passos.

Uso (de dentro de scripts_figuras/):   python diagnostico/run_v2.py

As pastas v2_b1.2000_eps0.01_mutC..._pbur0.4073_... são criadas ao lado do engine_v2.py
(em scripts_figuras/). Depois disso, rode:   python fig_D_01_diagnostico.py

A colônia inicial precisa ser a MESMA do engine.py original, para que "mesma rodada" seja
verdade. Se o engine.py original estiver em diagnostico/, a configuração dos invasores é
copiada dele; senão, usa-se a do engine_v2.py e um aviso é impresso.
"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))
import engine_v2 as motor                                   # noqa: E402

MUS = [0.0, 1e-5, 1e-4, 1e-3]

motor.CONFIG_SEMENTES = [44]
motor.TAMANHOS_REDE = [201]
motor.STEPS = 20000
motor.b = 1.2
motor.EPSILON_P = 0.01
motor.PROPORCAO_RANDOM_BURACOS = 0.407254
motor.PROPORCAO_RANDOM_COOPERADORES = 0.0
motor.CONFIG_BURACOS = []
motor.TIPO_VIZINHANCA = 'moore'
motor.PROFUNDIDADE_VIZINHANCA = 1
motor.TOPOLOGIA_TABULEIRO = 'toroide'
motor.PROBABILIDADE_ATUALIZACAO = 1.0
motor.K_FERMI = 0.0
motor.GERAR_FRAMES = False
motor.CONTINUAR_SIMULACAO = False

try:
    sys.path.insert(0, AQUI)
    import engine as original                               # o engine.py ORIGINAL
    if hasattr(original, 'CONFIG_INVASORES'):
        motor.CONFIG_INVASORES = original.CONFIG_INVASORES
        print(f"colônia inicial copiada do engine.py original: {motor.CONFIG_INVASORES}")
except ImportError:
    print("[AVISO] engine.py original não encontrado em diagnostico/; usando a colônia inicial "
          f"do engine_v2.py: {motor.CONFIG_INVASORES}. Confira se é a mesma do original.")

for mu in MUS:
    motor.TAXA_MUTACAO_C = mu
    motor.TAXA_MUTACAO_D = mu
    print(f"\n===== mu = {mu:g} =====")
    motor.executar_simulacao()
