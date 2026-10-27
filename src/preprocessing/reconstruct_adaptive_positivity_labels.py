"""Common10m synchronized peak bounds and common5m local depth bounds; no solver calls."""
import os
os.environ['NUMBA_NUM_THREADS']='2'
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,argparse,time
import numpy as np
import netCDF4,rasterio
from rasterio.windows import Window
from numba import njit,prange
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_adaptive_positivity import OUT,REPORT,CASES,PRIOR
from reconstruct_three_ward_common_depth import ground_tables,bounds,volume
from reference_label_policy import classify_labels
from prepare_sfincs_cpu_build import dump
from audit_three_ward_negative_volumes import readnp,arr
NR,NC=1012,770

@njit(parallel=True)
def quarters(z,pre,count,owners,rp,rm,ap,am,hyd):
 out=np.full((4,count.shape[0],4),np.nan,np.float64);bar=np.full((count.shape[0],4),np.inf,np.float32)
 nc=hyd.shape[1]//20
 for c in prange(count.shape[0]):
  r=c//nc;col=c%nc
  for q in range(4):
   rr=r*20+(q//2)*10;cc=col*20+(q%2)*10
   for dr in range(10):
    for dc in range(10):
     h=hyd[rr+dr,cc+dc]
     if np.isfinite(h):bar[c,q]=min(bar[c,q],h)
   n=count[c,q]
   if n==0:continue
   ar=n*.25;j=owners[c,q]
   if np.isfinite(rp[c]):
    v=volume(z[c,q],pre[c,q],n,rp[c])/ar;out[0,c,q]=v;out[1,c,q]=v
   else:out[0,c,q]=0.;out[1,c,q]=volume(z[c,q],pre[c,q],n,rm[c]+.01)/ar
   if np.isfinite(ap[j]):
    v=volume(z[c,q],pre[c,q],n,ap[j])/ar;out[2,c,q]=v;out[3,c,q]=v
   else:out[2,c,q]=0.;out[3,c,q]=volume(z[c,q],pre[c,q],n,am[j]+.01)/ar
 return out,bar

def reshape5(x):return x.reshape(NR,NC,2,2).transpose(0,2,1,3).reshape(NR*2,NC*2)
def main(only=None):
 out=ROOT/OUT;report=ROOT/REPORT;start=time.monotonic();geo=ROOT/'data/processed/three_ward_adaptive_connectivity';g=readnp(geo/'adaptive_cell_geometry.npz');support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area']
 owner=np.full((NR*2,NC*2),-1,np.int32)
 for size in [5,10]:
  ids=np.flatnonzero(g['size_m']==size);rr=g['row_fine'][ids]//10;cc=g['col_fine'][ids]//10
  for dr in range(size//5):
   for dc in range(size//5):assert np.all(owner[rr+dr,cc+dc]==-1);owner[rr+dr,cc+dc]=ids
 owner4=owner.reshape(NR,2,NC,2).transpose(0,2,1,3).reshape(NR,NC,4)
 runs=json.loads((out/'run_manifest.json').read_text())['runs'];cases=[r for r in runs if only is None or r['case_id']==only];data={}
 for run in cases:
  c=run['case_id'];p=ROOT/run['folder'];assert json.loads((p/'execution.json').read_text())['completion_marker']
  with netCDF4.Dataset(ROOT/run['regular_reference_folder']/'sfincs_map.nc')as d:rp=np.flipud(arr(d['zsmax'][0]));rm=np.flipud(arr(d['zb'][:]))
  with netCDF4.Dataset(p/'sfincs_map.nc')as d:ap=arr(d['zsmax'][0]);am=arr(d['zb'][:]);states=arr(d['zs'][:]).astype('f4')
  native_minimum=np.loadtxt(p/'udt_adaptive_cell_summary.csv',delimiter=',',skiprows=1,usecols=2);assert len(native_minimum)==len(ap)
  data[c]={'native_negative':native_minimum<-.001,'rp':rp,'rm':rm,'ap':ap,'am':am,'states':states,'labels':np.full((8,NR,NC),np.nan),'quarters':np.full((4,NR,NC,4),np.nan),'native_barrier':np.full(len(ap),np.inf)}
 qa=np.zeros((NR,NC,4));commonbarrier=np.full((NR,NC),np.inf)
 base=ROOT/'data/processed/three_ward_sfincs_scenarios';con=ROOT/'data/processed/three_ward_urban_constraints'
 with rasterio.open(base/'terrain_sampling_0_50m.tif')as dem,rasterio.open(con/'constraints_0_50m.tif')as cl,rasterio.open(con/'hydraulic_obstacle_0_50m.tif')as hyd:
  for r0 in range(0,NR,32):
   nr=min(32,NR-r0);w=Window(0,r0*20,NC*20,nr*20);z=dem.read(1,window=w);cls=cl.read(1,window=w);hz=hyd.read(1,window=w);building=np.where(cls==1,hz,np.inf);zz,pre,counts,roads,unknown=ground_tables(z,cls,valid[r0:r0+nr]);own=np.ascontiguousarray(owner4[r0:r0+nr].reshape(-1,4));assert not np.any((counts>0)&(own<0));assert np.array_equal(counts.sum(1).reshape(nr,NC)*.25,area[r0:r0+nr]);qa[r0:r0+nr]=counts.reshape(nr,NC,4)*.25
   for c,d in data.items():
    result,nowners=bounds(zz,pre,counts,own,d['rp'][r0:r0+nr].ravel(),d['rm'][r0:r0+nr].ravel(),d['ap'],d['am'],d['states']);d['labels'][:,r0:r0+nr]=result.reshape(8,nr,NC)
    q,bar=quarters(zz,pre,counts,own,d['rp'][r0:r0+nr].ravel(),d['rm'][r0:r0+nr].ravel(),d['ap'],d['am'],building);d['quarters'][:,r0:r0+nr]=q.reshape(4,nr,NC,4)
    commonbarrier[r0:r0+nr]=bar.min(1).reshape(nr,NC)
    sel=own>=0;np.minimum.at(d['native_barrier'],own[sel],bar[sel])
   if r0%128==0:print('GROUND',r0+nr,'/',NR,'elapsed',round(time.monotonic()-start),flush=True)
 area5=reshape5(qa);np.savez_compressed(out/'common_local_support_5m.npz',**{'Ground Area':area5,'Ward Code':np.repeat(np.repeat(support['Ward Code'],2,0),2,1),'Eligible Ground Mask':area5>0})
 checks=[]
 for c,d in data.items():
  rl,ru,lo,up,sl,su,s600,s1200=d['labels'];up=np.maximum(up,lo);up=np.maximum(up,su);assert np.all(lo[valid]<=up[valid]+1e-7)
  nativecontact=np.isfinite(d['ap'])&np.isfinite(d['native_barrier'])&(d['ap']>=d['native_barrier']-.01)
  contacts=np.zeros((NR,NC),bool);negative=np.zeros((NR,NC),bool)
  for q in range(4):
   owners=owner4[:,:,q];sel=owners>=0;contacts[sel]|=nativecontact[owners[sel]];negative[sel]|=d['native_negative'][owners[sel]]
  masks=classify_labels(lo,up,valid,negative,contacts,interval_source_known=True)
  np.savez_compressed(out/(c+'_adaptive_candidate_reference.npz'),**{'Peak Mean Depth Lower':lo,'Peak Mean Depth Upper':up,'Peak Interval Width':up-lo,'Sample Peak Lower 300s':sl,'Sample Peak Upper 300s':su,'Sample Peak Lower 600s':s600,'Sample Peak Lower 1200s':s1200,'Assumed Barrier Contact':contacts,'Online Ever Negative':negative,**masks})
  np.savez_compressed(out/(c+'_local_depth_bounds_5m.npz'),**{k:reshape5(d['quarters'][i])for i,k in enumerate(['Regular Lower','Regular Upper','Adaptive Lower','Adaptive Upper'])})
  np.savez_compressed(out/(c+'_native_peak_bounds.npz'),**{'Peak Water Level':d['ap'],'Minimum Ground Level':d['am'],'Assumed Barrier Contact':nativecontact,'Native Assumed Barrier':d['native_barrier']})
  old=readnp(ROOT/PRIOR/(c+'_candidate_reference.npz'));rlerr=float(np.nanmax(abs(rl-old['Peak Mean Depth Lower'])));ruerr=float(np.nanmax(abs(ru-old['Peak Mean Depth Upper'])));assert max(rlerr,ruerr)<1e-12
  checks.append({'case':c,'regular_depth_reconstruction_max_difference_m':max(rlerr,ruerr),'common_ground_area_m2':float(area.sum()),'local_ground_area_m2':float(area5.sum()),'all_bounds_finite':bool(np.all(np.isfinite(lo[valid]))and np.all(np.isfinite(up[valid]))),'intervals_ordered':bool(np.all(up[valid]>=lo[valid])),'training':False})
 dump(report/(((only+'_')if only else'')+'label_reconstruction.json'),{'cases':checks,'quarter_ground_area_reaggregation_exact':bool(np.array_equal(qa.sum(2),area)),'method':'common10m peak of synchronized mean depth is bounded by online native maxima and synchronized300s states; common5m local bounds use each single native state; no peak-of-means substitution or midpoint filling','new_hydrodynamic_runs':0});print(json.dumps(checks),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',choices=CASES);main(a.parse_args().only)
