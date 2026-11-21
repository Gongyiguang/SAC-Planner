"""Standalone illustration; does not run SAC-Planner.
Whole-block baseline is illustrative, not manuscript non-split method.
"""
import argparse
from pathlib import Path
import numpy as np
OFFSETS=tuple((dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc)
def build(mask,R):
 h,w=mask.shape; labels=np.full((h,w),-1,np.int32); centers=[]
 for r0 in range(0,h,R):
  for c0 in range(0,w,R):
   r1,c1=min(r0+R,h),min(c0+R,w)
   for r in range(r0,r1):
    for c in range(c0,c1):
     if not mask[r,c] or labels[r,c]>=0: continue
     gid=len(centers); labels[r,c]=gid; stack=[(r,c)]; cells=[]
     while stack:
      rr,cc=stack.pop(); cells.append((rr,cc))
      for dr,dc in OFFSETS:
       nr,nc=rr+dr,cc+dc
       if r0<=nr<r1 and c0<=nc<c1 and mask[nr,nc] and labels[nr,nc]<0:
        labels[nr,nc]=gid; stack.append((nr,nc))
     centers.append(np.mean(cells,axis=0))
 adj=[set() for _ in centers]
 for dr,dc in ((0,1),(1,-1),(1,0),(1,1)):
  ra,rb=max(0,-dr),h-max(0,dr); ca,cb=max(0,-dc),w-max(0,dc)
  aa=labels[ra:rb,ca:cb]; bb=labels[ra+dr:rb+dr,ca+dc:cb+dc]
  valid=(aa>=0)&(bb>=0)&(aa!=bb)
  for u,v in np.unique(np.stack((aa[valid],bb[valid]),axis=1),axis=0):
   adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
 return labels,np.asarray(centers).reshape(-1,2),adj
def groups(adj):
 unseen=set(range(len(adj))); sizes=[]
 while unseen:
  stack=[unseen.pop()]; size=0
  while stack:
   u=stack.pop(); size+=1
   for v in adj[u]:
    if v in unseen: unseen.remove(v); stack.append(v)
  sizes.append(size)
 return sorted(sizes,reverse=True)
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--R',type=int,default=8)
 p.add_argument('--map',type=Path,default=Path(__file__).with_name('sample_map.npz'))
 p.add_argument('--output',type=Path,default=Path('demo_component_graph.png')); a=p.parse_args()
 if a.R<1: p.error('--R must be positive')
 with np.load(a.map,allow_pickle=False) as d:
  obs=d['obstaclemask']
  if obs.ndim!=2 or obs.dtype!=np.bool_: p.error('obstaclemask must be 2D boolean')
  free=~obs
 labels,centers,adj=build(free,a.R); gs=groups(adj)
 coarse=np.zeros(((free.shape[0]+a.R-1)//a.R,(free.shape[1]+a.R-1)//a.R),bool)
 for r,c in np.ndindex(coarse.shape): coarse[r,c]=free[r*a.R:min((r+1)*a.R,free.shape[0]),c*a.R:min((c+1)*a.R,free.shape[1])].all()
 _,_,naive=build(coarse,a.R*0+1); ng=groups(naive)
 print(f'block size R = {a.R}\npassable cells       : {free.sum()}\ncomponents (vertices): {len(adj)}\ngraph edges          : {sum(map(len,adj))//2}\nconnected groups     : {len(gs)} (largest: {max(gs,default=0)} vertices)\n')
 print('naive block rule (a block is kept only if fully passable)')
 print(f'blocks kept          : {len(naive)}\nconnected groups     : {len(ng)} (largest: {max(ng,default=0)} blocks)')
 import matplotlib; matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 rgb=np.full((*labels.shape,3),.12); colors=np.random.default_rng(0).random((len(centers),3))*.65+.3
 valid=labels>=0; rgb[valid]=colors[labels[valid]]
 fig,ax=plt.subplots(1,2,figsize=(13,6),constrained_layout=True); ax[0].imshow(rgb)
 ax[0].set_title('Block-local eight-connected components')
 ax[0].set_xticks(np.arange(-.5,free.shape[1],a.R),minor=True); ax[0].set_yticks(np.arange(-.5,free.shape[0],a.R),minor=True); ax[0].grid(which='minor',alpha=.25)
 ax[1].imshow(labels<0,cmap='gray_r',vmin=0,vmax=1)
 seg=[[centers[u][::-1],centers[v][::-1]] for u,ns in enumerate(adj) for v in ns if u<v]
 ax[1].add_collection(LineCollection(seg,colors='#6b7280',linewidths=.45,alpha=.65))
 if len(centers): ax[1].scatter(centers[:,1],centers[:,0],s=7,c='#2878b5')
 ax[1].set_title('Component graph (schematic edges)')
 for q in ax:
  q.set(xlabel='Column',ylabel='Row',xlim=(-.5,free.shape[1]-.5),ylim=(free.shape[0]-.5,-.5)); q.set_aspect('equal')
 fig.savefig(a.output,dpi=160); plt.close(fig); print(f'Figure saved to {a.output.resolve()}')
if __name__=='__main__': main()
