"""Variação de run_diag: python run_mu.py MU PASSOS [P_BURACOS]
Ex.: python run_mu.py 0 300 0.0   -> diag_mu0_p0.npz (controle balístico usado na figura)"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)                          # engine.py original deve estar nesta pasta
sys.path.insert(1, os.path.join(AQUI, '..'))      # nucleo_numpy.py
import numpy as np
from run_diag import rodar
mu=float(sys.argv[1]); steps=int(sys.argv[2]); p=float(sys.argv[3]) if len(sys.argv)>3 else None
out,_=rodar(mu=mu, STEPS=steps, p_bur=p, verbose=False)
tag = f"mu{mu:g}" + (f"_p{p:g}" if p is not None else "")
np.savez_compressed(os.path.join(AQUI, f'diag_{tag}.npz'), **{k:v for k,v in out.items() if k not in ('sub_lab','ux','uy')})
print('ok', tag)
