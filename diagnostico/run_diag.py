"""Roda a semente 44 (L=201) com o engine.py ORIGINAL e grava os diagnósticos
(Rg2 antigo, massa do cluster rastreado, etc.) em diag_s44.npz. Uso: python run_diag.py"""
import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)                          # engine.py original deve estar nesta pasta
sys.path.insert(1, os.path.join(AQUI, '..'))      # nucleo_numpy.py
import time, numpy as np
import engine as E
from nucleo_numpy import passo_racional, TorusLabeler, cinematica_original, rg2_desembrulhado, bfs_quimica
from scipy.ndimage import label as nd_label

def rodar(L=201, semente=44, STEPS=20000, p_bur=None, b=None, mu=None, snaps=(), tag=None, verbose=True):
    if p_bur is not None: E.PROPORCAO_RANDOM_BURACOS = p_bur
    bb = E.b if b is None else b
    if mu is not None: E.TAXA_MUTACAO_C = E.TAXA_MUTACAO_D = mu
    muC, muD = E.TAXA_MUTACAO_C, E.TAXA_MUTACAO_D
    np.random.seed(semente); np.random.seed(semente)
    grid, _ = E.inicializar_universo(L)
    active = grid != 2
    c0 = L//2
    # componente do substrato que contém a semente + distância química
    sub_lab, _ = nd_label(active, structure=np.ones((3,3)))
    ell, ux, uy = bfs_quimica(active, c0, c0)
    no_semente = ell >= 0
    # deslocamento de imagem mínima a partir da semente (estático)
    rr, cc = np.indices((L,L))
    dxs = cc - c0; dxs = dxs - L*np.round(dxs/L)
    dys = rr - c0; dys = dys - L*np.round(dys/L)
    r2_img = dxs**2 + dys**2
    lab = TorusLabeler(L)
    tl,n,info = lab(grid==0)
    cx,cy,rg2,_,m,px,py,rot = cinematica_original(tl,n,info,L)
    cols = {k:[] for k in ['t','frac_c','rg2','cx','cy','m_trk','m_max','n_C','n_val','trk_is_max',
                           'wrap','rg2_unw','n_mut','n_rac','R2_orig','R2_orig_filt','ell_max','ell_mean','N_seed']}
    snapshots = {}
    t0=time.time()
    for t in range(1, STEPS+1):
        prop, nc, nd, pay = passo_racional(grid, 1.0, 0.0, bb, 0.01)
        mut=(prop+1)%2; rm=np.random.rand(L,L)
        mm = ((grid==0)&(rm<muC))|((grid==1)&(rm<muD))
        n_rac = int(np.sum((prop!=grid)&active))
        pg=np.where(mm,mut,prop); pg[~active]=2
        n_mut = int(np.sum(mm & active))
        grid=pg
        isC = grid==0
        tl,n,info = lab(isC)
        cx,cy,rg2,_,m,px,py,rot = cinematica_original(tl,n,info,L,cx,cy)
        tam = np.bincount(tl.ravel(), minlength=n+1); tam[0]=0
        val = tam >= 4
        mmax = tam.max() if n else 0
        cols['t'].append(t); cols['frac_c'].append(isC.sum()/L**2); cols['rg2'].append(rg2)
        cols['cx'].append(cx); cols['cy'].append(cy); cols['m_trk'].append(m); cols['m_max'].append(mmax)
        cols['n_C'].append(int(isC.sum())); cols['n_val'].append(int(val.sum()))
        cols['trk_is_max'].append(bool(rot>0 and tam[rot]==mmax))
        cols['wrap'].append(bool(px or py))
        cols['rg2_unw'].append(rg2_desembrulhado(tl,info,rot,L) if rot>0 and not (px or py) else np.nan)
        cols['n_mut'].append(n_mut); cols['n_rac'].append(n_rac)
        # observável tipo Grassberger: R^2 medido da ORIGEM fixa, sobre C no componente da semente
        sel = isC & no_semente
        cols['N_seed'].append(int(sel.sum()))
        cols['R2_orig'].append(r2_img[sel].mean() if sel.any() else np.nan)
        # filtro: remove C isolados (sem vizinho C) -> mutantes efêmeros
        self_ok = sel & (nc >= 1)  # nc é do passo anterior; recalcula:
        ncC = sum(np.roll(np.roll(isC,-dr,0),-dc,1) for dr,dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)])
        selF = sel & (ncC>=1)
        cols['R2_orig_filt'].append(r2_img[selF].mean() if selF.any() else np.nan)
        # distância química dos C que pertencem ao maior cluster
        if n:
            big = tl == np.argmax(tam)
            e = ell[big & no_semente]
            cols['ell_max'].append(e.max() if e.size else np.nan)
        else:
            cols['ell_max'].append(np.nan)
        cols['ell_mean'].append(ell[selF].mean() if selF.any() else np.nan)
        if t in snaps: snapshots[t] = (grid.copy(), tl.copy(), rot, cx, cy)
        if verbose and t % 2000 == 0:
            print(f"t={t} C={isC.sum()} Rg2={rg2:.0f} m_trk={m} m_max={mmax} nval={val.sum()} ({time.time()-t0:.0f}s)", flush=True)
    out = {k: np.array(v) for k,v in cols.items()}
    out['ell'] = ell; out['active'] = active; out['sub_lab']=sub_lab
    out['ux']=ux; out['uy']=uy
    return out, snapshots

if __name__ == "__main__":
    snaps = (1,10,30,60,150,399,420,450,500,700,900,1200,2000,3000,3100,3300,5000,10000,16000,20000)
    out, sn = rodar(snaps=snaps)
    np.savez_compressed(os.path.join(AQUI, 'diag_s44.npz'), **out)
    np.savez_compressed(os.path.join(AQUI, 'snaps_s44.npz'), **{f"g{t}": v[0] for t,v in sn.items()},
                        **{f"l{t}": v[1] for t,v in sn.items()},
                        **{f"r{t}": np.array([v[2],v[3],v[4]]) for t,v in sn.items()})
