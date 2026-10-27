"""Controlled road portal on/off trial. Same nodes, storage, rain, loss; banks closed.
All outputs synthetic and uncalibrated. No real event replay or new water outlet.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,csv,time
import numpy as np,rasterio
from rasterio.transform import Affine
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b');PRO=ROOT/'data/processed/three_ward_road_connectivity';OUT=ROOT/'data/exp/data-briefing/three_ward_road_connectivity';OLD=ROOT/'data/processed/three_ward_surface_trial'
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cs(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def step(z,h,area,src,dst,width,crest,length,rough,theta,rain,dt):
 v=h*area;inputv=rain*dt;v+=inputv;loss=np.minimum(v,area*.015/3600*dt);v-=loss;h=v/area;eta=z+h;delta=eta[src]-eta[dst];hf=np.maximum(np.maximum(eta[src],eta[dst])-crest,0);q=np.sign(delta)*width*hf**(5/3)/rough*np.sqrt(np.abs(delta)/length);eq=np.abs(delta)/(1/area[src]+1/area[dst]);q=np.sign(q)*np.minimum(abs(q),theta*eq/dt);out=np.bincount(src,weights=np.maximum(q,0),minlength=len(z))+np.bincount(dst,weights=np.maximum(-q,0),minlength=len(z));fac=np.minimum(1,np.divide(.95*v,out*dt,out=np.ones_like(v),where=out>0));q*=np.where(q>=0,fac[src],fac[dst]);v+=np.bincount(dst,weights=q*dt,minlength=len(z))-np.bincount(src,weights=q*dt,minlength=len(z));assert v.min()>-1e-7;return np.maximum(v,0)/area,float(inputv.sum()),float(loss.sum())

def tests():
 # Six faces at a tiny central storage: degree-scaled cap prevents overshoot.
 s=np.zeros(6,dtype=int);d=np.arange(1,7);ar=np.array([25]+[100]*6,dtype=float);z=np.zeros(7);h=np.array([.1]+[2]*6);vol0=float((h*ar).sum());h,ri,lo=step(z,h,ar,s,d,np.ones(6)*5,np.zeros(6),np.ones(6)*5,np.ones(6)*.03,np.ones(6)*(.8/6),np.zeros(7),2.5);assert h.max()<=2+1e-10 and h.min()>=0;assert abs((h*ar).sum()+lo-vol0)<1e-8
 # Lake at rest with varying beds and crest heights; no redistribution.
 z=np.array([0.,1.]);h=np.array([2.,1.]);ar=np.ones(2)*25;hn,*_=step(z,h,ar,np.array([0]),np.array([1]),np.array([5.]),np.array([1.]),np.array([5.]),np.array([.03]),np.array([.2]),ar*.015/3600,2.5);assert np.allclose(h,hn)
 return {'high_degree_surface_maximum_principle':'pass','high_degree_mass_balance_and_positivity':'pass','lake_at_rest_with_loss_balanced_by_source':'pass'}

def main():
 g=dict(np.load(PRO/'road_graph.npz'));audit=json.loads((OUT/'connectivity_audit.json').read_text());a=dict(np.load(OLD/'surface_inputs.npz'));shape=tuple(audit['grid_shape']);size=int(np.prod(shape));N=len(g['z']);src=g['edges'][:,0];dst=g['edges'][:,1];deg=np.bincount(np.r_[src,dst],minlength=N);theta=np.minimum(.2,.8/np.maximum(deg[src],deg[dst]));results={};validation={'unit_checks':tests()};summary=[];volumes=[];connection=[]
 for key,keep in [('R0',g['edge_kind']!=2),('R1',np.ones(len(src),dtype=bool))]:
  e=g['edges'][keep];sp=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(N,N));ncomp,labels=connected_components(sp,directed=False);counts=np.bincount(labels);connection.append({'scenario':key,'total_graph_components':ncomp,'largest_component_nodes':int(counts.max()),'isolated_nodes':int(np.sum(counts==1)),'road_portals_enabled':key=='R1'})
 cs(OUT/'graph_connectivity_comparison.csv',connection)
 def run(key,opened,dt,save):
  keep=np.ones(len(src),dtype=bool) if opened else g['edge_kind']!=2;ss=src[keep];dd=dst[keep];args=[g[n][keep] for n in ['width','crest','length','roughness']];th=theta[keep];h=np.zeros(N);peak=h.copy();rain=loss=0.;maxres=0.;start=time.monotonic()
  for it in range(int(5400/dt)):
   minute=it*dt/60;factor=.5 if minute<15 or 45<=minute<60 else 1.5 if 15<=minute<45 else 0.;h,ri,lo=step(g['z'],h,g['area'],ss,dd,*args,th,g['rain_m3_s']*factor,dt);rain+=ri;loss+=lo;np.maximum(peak,h,out=peak);storage=float((h*g['area']).sum());res=storage-rain+loss;maxres=max(maxres,abs(res))
   if (it+1)*dt%300==0 and save:volumes.append({'scenario':key,'time_min':(it+1)*dt/60,'rain_m3':rain,'loss_m3':loss,'storage_m3':storage,'mass_residual_m3':res})
   if (it+1)*dt%1800==0:print(key,'minutes',int((it+1)*dt/60),'maxdepth',round(float(h.max()),3),flush=True)
  assert maxres/max(rain,1)<1e-9
  if save:
   weighted=np.bincount(g['parent'],weights=peak*g['area'],minlength=size);den=np.bincount(g['parent'],weights=g['area'],minlength=size);depth=np.divide(weighted,den,out=np.full(size,np.nan),where=den>0).reshape(shape);maxp=np.full(size,np.nan);np.maximum.at(maxp,g['parent'],peak) # np.maximum cannot initializeNaN; use explicit finite initial.
   maxp=np.full(size,-np.inf);np.maximum.at(maxp,g['parent'],peak);maxp[maxp<0]=np.nan;maxp=maxp.reshape(shape)
   np.savez_compressed(PRO/f'{key}_result.npz',node_peak=peak,node_final=h,area_weighted_peak=depth,cell_max_peak=maxp)
   tf=Affine(*audit['transform'])
   with rasterio.open(PRO/f'{key}_area_weighted_peak.tif','w',driver='GTiff',height=shape[0],width=shape[1],count=1,dtype='float32',crs=6677,transform=tf,nodata=np.nan,compress='deflate') as ds:ds.write(depth.astype('float32'),1)
   for wi,name in [(1,'千代田区'),(2,'中央区'),(3,'港区')]:
    for t,ty in [(None,'all_open_surface'),(1,'road_components')]:
     k=(g['ward']==wi)&(True if t is None else g['node_type']==t);summary.append({'scenario':key,'ward':name,'surface_group':ty,'open_area_m2':float(g['area'][k].sum()),'peak_ge0_3m_open_area_m2':float(g['area'][k&(peak>=.3)].sum()),'peak_gt2m_open_area_m2':float(g['area'][k&(peak>2)].sum()),'area_weighted_mean_peak_m':float(np.average(peak[k],weights=g['area'][k])),'max_node_peak_m':float(peak[k].max())})
   validation[key]={'dt_s':dt,'rain_m3':rain,'loss_m3':loss,'final_storage_m3':storage,'mass_residual_max_m3':maxres,'relative_mass_residual':maxres/rain,'negative_nodes':int(np.sum(h<0)),'runtime_s':time.monotonic()-start};results[key]=peak
  return peak
 run('R0',False,2.5,True);run('R1',True,2.5,True);fine=run('R1_dt1_25',True,1.25,False);diff=np.abs(results['R1']-fine);validation['timestep_check']={'dt_s':[2.5,1.25],'area_weighted_mean_abs_peak_difference_m':float(np.average(diff,weights=g['area'])),'p95_node_abs_difference_m':float(np.quantile(diff,.95)),'max_node_abs_difference_m':float(diff.max()),'node_area_over0_1m_difference_m2':float(g['area'][diff>.1].sum()),'over0_3m_area_relative_difference':float(abs(g['area'][results['R1']>=.3].sum()-g['area'][fine>=.3].sum())/g['area'][fine>=.3].sum()),'local_depth_validation':False}
 np.savez_compressed(PRO/'R1_timestep_check.npz',node_peak=fine,node_abs_difference=diff);js(OUT/'numerical_validation.json',validation);cs(OUT/'ward_model_comparison.csv',summary);cs(OUT/'water_balance.csv',volumes)
 # Qualify reduction/increase rather than labelling every deep cell a false pool.
 reduction=results['R0']-results['R1'];diagnostics=[]
 for wi,name in [(1,'千代田区'),(2,'中央区'),(3,'港区')]:
  k=g['ward']==wi;diagnostics.append({'ward':name,'area_with_peak_reduced_gt0_1m_m2':float(g['area'][k&(reduction>.1)].sum()),'area_with_peak_increased_gt0_1m_m2':float(g['area'][k&(reduction<-.1)].sum()),'road_area_with_peak_reduced_gt0_1m_m2':float(g['area'][k&(g['node_type']==1)&(reduction>.1)].sum()),'note':'Controlled synthetic topology test. Redistribution may increase elsewhere; no observed flooding claim.'})
 cs(OUT/'redistribution_diagnostics.csv',diagnostics);print('DONE',json.dumps(validation['timestep_check']),flush=True)
if __name__=='__main__':main()
