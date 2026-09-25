import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
FIGURAS = os.path.join(AQUI, '..', 'figuras')
os.makedirs(FIGURAS, exist_ok=True)
sys.path.insert(0, AQUI)
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.ndimage import label
rng=np.random.default_rng(11)
MOORE=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
L=801; c=L//2; S8=np.ones((3,3)); T=[1,2,4,8,16,32,64,128,256]
vivos=np.zeros(len(T)); n=0; t0=time.time()
while time.time()-t0<200:
    occ=rng.random((L,L))>=0.592746; occ[c,c]=True
    lab,_=label(occ,structure=S8); cl=lab==lab[c,c]
    ys,xs=np.where(cl)
    if len(ys)==1: lmax=0
    else:
        idx=-np.ones((L,L),int); idx[ys,xs]=np.arange(len(ys)); rows=[];cols=[]
        for dr,dc in MOORE:
            r2=ys+dr;c2=xs+dc; ok=(r2>=0)&(r2<L)&(c2>=0)&(c2<L); ok[ok]=cl[r2[ok],c2[ok]]
            rows.append(idx[ys[ok],xs[ok]]); cols.append(idx[r2[ok],c2[ok]])
        A=coo_matrix((np.ones(sum(map(len,rows))),(np.concatenate(rows),np.concatenate(cols))),shape=(len(ys),)*2).tocsr()
        d=shortest_path(A,unweighted=True,indices=idx[c,c],directed=False); lmax=d.max()
        if len(ys)>0 and (ys.min()==0 or xs.min()==0 or ys.max()==L-1 or xs.max()==L-1): lmax=np.inf  # atingiu a borda
    vivos+=np.array([lmax>=t for t in T]); n+=1
P=vivos/n
print('realizações:',n)
for t,p in zip(T,P): print(t, round(p,4))
m=np.array(T)>=8
print('delta ajustado (t>=8):', round(-np.polyfit(np.log(np.array(T)[m]),np.log(P[m]),1)[0],4), '| previsão beta/(nu d_min) =', round((5/48)/1.1307,4))
