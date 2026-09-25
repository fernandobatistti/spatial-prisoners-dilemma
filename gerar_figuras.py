"""
Refaz todas as figuras da dissertação:  python gerar_figuras.py  [trecho ...]

Roda, em ordem, todos os fig_*.py desta pasta. Com argumentos, roda só os scripts cujo nome
contém algum deles (ex.: `python gerar_figuras.py 2_1 5_0` ou `python gerar_figuras.py evento`).
Nenhum script simula os experimentos do Capítulo 5: eles leem os dados gravados. Os do
Capítulo 2 que simulam (percolação, distância química, força, queima, observáveis) guardam o
resultado em cache (dados_*.json/.npz) e só recalculam se os parâmetros mudarem.
fig_D_01_diagnostico.py precisa de dados externos e é pulado se eles não existirem.
"""
import glob
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(glob.glob(os.path.join(AQUI, 'fig_*.py')))
if len(sys.argv) > 1:
    scripts = [s for s in scripts if any(a in os.path.basename(s) for a in sys.argv[1:])]

falhas = []
for s in scripts:
    nome = os.path.basename(s)
    print(f"\n=== {nome}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, s], cwd=AQUI)
    print(f"    ({time.time() - t0:.0f} s)")
    if r.returncode != 0:
        falhas.append(nome)
print("\n" + ("tudo certo." if not falhas else "FALHARAM: " + ", ".join(falhas)))
