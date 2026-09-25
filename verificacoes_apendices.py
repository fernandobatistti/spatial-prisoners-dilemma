"""
verificacoes_apendices.py — reproduz os números citados nos apêndices e na Fundamentação.

Uso:  python verificacoes_apendices.py            (todas as verificações, ~2 min)
      python verificacoes_apendices.py A4 B1      (apenas as seções pedidas)

Seções:
  A4  Dilema do Prisioneiro no campo médio (A.2, A.4): Lyapunov, frações parciais,
      solução implícita e correção logarítmica
  A5  Limiares k/m e epsilon máximo (A.5, Tabela A.1)
  A6  Replicador com mutação (A.6): polinômio e pontos fixos
  A7  Dimensão fractal do maior aglomerado de Moore no limiar (A.7.2)
  A9  Frente reta sem buracos (A.9.1)
  B1  Desertor único em cada intervalo de b (Fig. 2.10, Seção 2.2.5, B.1)
  B2  Pequenos aglomerados de cooperadores (B.2)
  W   Caminhante (walker) em função de b (Seção 2.2.5)
Os números da medida de d_min (A.8.5) e da sobrevivência da queima (A.8.3) são
produzidos por fig_quimica.py e sobrevivencia_queima.py.
"""
import os
import sys
from fractions import Fraction

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from nucleo_numpy import passo_racional  # noqa: E402


def titulo(t):
    print('\n' + '=' * 70 + '\n' + t + '\n' + '=' * 70)


# ---------------------------------------------------------------------------
def secao_A4():
    import sympy as sp
    from scipy.integrate import solve_ivp
    titulo('A.2 / A.4 — campo médio')
    xs = sp.symbols('x1:4')
    A = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'a{min(i, j)}{max(i, j)}'))  # simétrica
    X = sp.Matrix(xs)
    f = A * X
    phi = (X.T * A * X)[0]
    xd = [xs[i] * (f[i] - phi) for i in range(3)]
    dphi = sum(sp.diff(phi, xs[i]) * xd[i] for i in range(3))
    var2 = 2 * sum(xs[i] * (f[i] - phi) ** 2 for i in range(3))
    print('dphi/dt = 2 Var(f) (A simétrica):',
          sp.simplify((dphi - var2).subs(xs[2], 1 - xs[0] - xs[1])) == 0)

    x, b, e = sp.symbols('x b epsilon', positive=True)
    print('frações parciais (eps=0):', sp.apart(1 / (x ** 2 * (1 - x)), x))
    be = b - 1 - e
    G = sp.log(x) / e - sp.log(1 - x) / (b - 1) - be / (e * (b - 1)) * sp.log(be * x + e)
    alvo = -1 / (x * (1 - x) * ((b - 1) * x + e * (1 - x)))
    print('solução implícita (eps>0) confere:', sp.simplify(sp.diff(G, x) + alvo) == 0)

    bb, x0 = 1.65, 0.99
    K = np.log(x0 / (1 - x0)) - 1 / x0
    sol = solve_ivp(lambda t, y: -(bb - 1) * y ** 2 * (1 - y), (0, 1e5), [x0],
                    rtol=1e-12, atol=1e-16, dense_output=True)
    print(f'{"t":>8} {"numérico":>12} {"1/[(b-1)t]":>12} {"com correção":>13}')
    for T in (1e2, 1e3, 1e4, 1e5):
        u = (bb - 1) * T
        print(f'{T:8.0e} {sol.sol(T)[0]:12.5e} {1 / u:12.5e} {1 / (u - np.log(u) - K):13.5e}')


# ---------------------------------------------------------------------------
def eps_max(b, z=8):
    return min(((k - b * m) / (z - m), k, m) for m in range(0, z)
               for k in range(z + 1) if k - b * m > 1e-12)


def secao_A5():
    titulo('A.5 — limiares k/m e epsilon máximo (vizinhança de Moore, z = 8)')
    z = 8
    lim = sorted({Fraction(k, m) for m in range(1, z + 1) for k in range(z + 1)
                  if 1 < Fraction(k, m) <= 2})
    print('limiares em (1, 2]:', ', '.join(str(v) for v in lim))
    print(f'{"b":>6} {"eps_max":>9}  par crítico (k, m)')
    for b in (1.15, 1.30, 1.45, 1.55, 1.65, 1.70, 1.90):
        v, k, m = eps_max(b)
        print(f'{b:6.2f} {v:9.4f}  ({k}, {m})')
    for rr in (1, 2):
        print(f'r = {rr}: z_vN = {2 * rr * (rr + 1)}, z_M = {(2 * rr + 1) ** 2 - 1}')


# ---------------------------------------------------------------------------
def secao_A6():
    import sympy as sp
    from scipy.optimize import brentq
    titulo('A.6 — replicador com mutação')
    x, b, e, m = sp.symbols('x b epsilon mu')
    fC = x
    fD = b * x + e * (1 - x)
    phi = x * fC + (1 - x) * fD
    ex = x * fC * (1 - m) + (1 - x) * fD * m - x * phi
    poli = (m * e + (m * (b - 2 * e) - e) * x
            - ((b - 1 - 2 * e) + m * (b + 1 - e)) * x ** 2 + (b - 1 - e) * x ** 3)
    print('polinômio do Apêndice A.6 confere:', sp.expand(ex - poli) == 0)

    def xdot(xx, bb, ee, mu):
        fc = xx
        fd = bb * xx + ee * (1 - xx)
        ph = xx * fc + (1 - xx) * fd
        return xx * fc * (1 - mu) + (1 - xx) * fd * mu - xx * ph
    for bb, ee, mu in [(1.65, 0.01, 1e-4), (1.65, 0.0, 1e-4)]:
        raiz = brentq(lambda v: xdot(v, bb, ee, mu), 1e-12, 0.5)
        prev = mu if ee > 0 else mu * bb / (bb - 1)
        print(f'b={bb}, eps={ee}, mu={mu}: x* = {raiz:.4e}   previsão = {prev:.4e}')


# ---------------------------------------------------------------------------
def secao_A7(amostras=(200, 200, 200, 60, 60)):
    from scipy.ndimage import label
    titulo('A.7.2 — dimensão fractal do maior aglomerado (Moore, p = p_c)')
    rng = np.random.default_rng(1)
    S8 = np.ones((3, 3), int)
    Ls = [32, 64, 128, 256, 512]
    Ms = []
    for L, n in zip(Ls, amostras):
        mm = []
        for _ in range(n):
            lab, _ = label(rng.random((L, L)) >= 0.592746, structure=S8)
            t = np.bincount(lab.ravel())
            t[0] = 0
            mm.append(t.max())
        Ms.append(np.mean(mm))
        print(f'L = {L:4d}: M = {Ms[-1]:.1f}')
    df = np.polyfit(np.log(Ls), np.log(Ms), 1)[0]
    print(f'd_f ajustado = {df:.3f}   (teoria 91/48 = {91 / 48:.3f})')


# ---------------------------------------------------------------------------
def secao_A9():
    titulo('A.9.1 — frente reta numa rede sem buracos')
    L = 60
    c = L // 2
    for b in (1.2, 1.5, 1.62, 1.7, 2.5, 2.7):
        g = np.ones((L, L), np.int32)
        g[:, :c] = 0
        g[:, :2] = 1                       # segunda frente, longe, para fechar o toro
        g2, *_ = passo_racional(g, 1.0, 0.0, b, 1e-4)
        avan = (g2[:, c] == 0).mean()
        rec = (g2[:, c - 1] == 1).mean()
        estado = 'avança' if avan == 1 else ('recua' if rec == 1 else 'parada')
        print(f'b = {b:4.2f}: {estado}  (fração que avança {avan:.2f}, que recua {rec:.2f})')


# ---------------------------------------------------------------------------
def secao_B1():
    titulo('B.1 / Fig. 2.10 — desertor único em cada intervalo de b')
    z = 8
    lim = sorted({Fraction(k, m) for m in range(1, z + 1) for k in range(z + 1)
                  if 1 < Fraction(k, m) <= 2})
    bordas = [1.0] + [float(v) for v in lim]
    L = 81
    c = L // 2
    for a, bsup in zip(bordas[:-1], bordas[1:]):
        b = (a + bsup) / 2
        g = np.zeros((L, L), np.int32)
        g[c, c] = 1
        hist = [g.copy()]
        seq = []
        per = None
        for t in range(1, 61):
            g, *_ = passo_racional(g, 1.0, 0.0, b, 1e-4)
            seq.append(int(g.sum()))
            for k, h in enumerate(reversed(hist), 1):
                if np.array_equal(h, g):
                    per = k
                    break
            if per:
                break
            hist.append(g.copy())
        desc = f'período {per} (detectado em t={t})' if per else 'sem ciclo até t=60'
        print(f'{a:.3f} < b < {bsup:.3f}: desertores {seq[:6]}...  {desc}')
    print('\nFração de desertores após 400 passos, b > 5/3 (efeito de tamanho):')
    for L in (81, 101, 121, 161):
        c = L // 2
        linha = []
        for b in (1.72, 1.85):
            g = np.zeros((L, L), np.int32)
            g[c, c] = 1
            for _ in range(400):
                g, *_ = passo_racional(g, 1.0, 0.0, b, 1e-4)
            linha.append(f'b={b}: {g.mean():.3f}')
        print(f'L = {L}: ' + ' | '.join(linha))


# ---------------------------------------------------------------------------
def secao_B2():
    titulo('B.2 — pequenos aglomerados de cooperadores num mar de desertores')
    L = 31
    c = 15
    casos = [('isolado', [(0, 0)]), ('par', [(0, 0), (0, 1)]),
             ('trio em L', [(0, 0), (0, 1), (1, 0)]),
             ('bloco 2x2', [(0, 0), (0, 1), (1, 0), (1, 1)])]
    for nome, cels in casos:
        out = []
        for b in (1.2, 1.45, 1.55):
            g = np.ones((L, L), np.int32)
            for r, cc in cels:
                g[c + r, c + cc] = 0
            for _ in range(6):
                g, *_ = passo_racional(g, 1.0, 0.0, b, 1e-4)
            out.append(f'b={b}: {int((g == 0).sum())} C')
        print(f'{nome:10s} após 6 passos -> ' + ' | '.join(out))


# ---------------------------------------------------------------------------
def secao_W():
    titulo('Seção 2.2.5 — caminhante (walker) de 10 cooperadores')
    L = 61

    def walker():
        g = np.ones((L, L), np.int32)
        g[44:46, 24:28] = 0
        g[46, 27] = 0
        g[47, 27] = 0
        return g
    for b in (1.45, 1.52, 1.58, 1.62, 1.65, 1.66):
        g = walker()
        y0 = np.where(g == 0)[0].mean()
        for _ in range(20):
            g, *_ = passo_racional(g, 1.0, 0.0, b, 1e-4)
        ys, _ = np.where(g == 0)
        print(f'b = {b}: {len(ys)} C após 20 passos, deslocamento vertical = {y0 - ys.mean():.1f}')


SECOES = {'A4': secao_A4, 'A5': secao_A5, 'A6': secao_A6, 'A7': secao_A7,
          'A9': secao_A9, 'B1': secao_B1, 'B2': secao_B2, 'W': secao_W}

if __name__ == '__main__':
    pedidas = sys.argv[1:] or list(SECOES)
    for s in pedidas:
        SECOES[s.upper()]()
