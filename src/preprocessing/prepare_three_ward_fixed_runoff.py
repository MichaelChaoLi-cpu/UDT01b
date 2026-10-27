"""Resolution-independent roof runoff scenario on the existing 0.50 m geometry.
No measured roof drainage is inferred. Raw terrain and prior experiments are preserved.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,hashlib
import numpy as np
import rasterio
from scipy.ndimage import label,distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_urban_constraints'
OLD=ROOT/'data/processed/three_ward_sfincs_scenarios'
OUT=ROOT/'data/processed/three_ward_fixed_runoff'

def main():
 OUT.mkdir(exist_ok=True)
 inputs=[P/'constraints_0_50m.tif',OLD/'active_mask_20m.tif']
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
 if (OUT/'contract.json').exists():
  c=json.loads((OUT/'contract.json').read_text());assert c['source_sha256']==hashes;print('Reuse completed allocation',flush=True);return
 with rasterio.open(inputs[0]) as d:cl=d.read(1);tf=d.transform
 with rasterio.open(inputs[1]) as d:m=d.read(1)
 inside=np.repeat(np.repeat(m==1,40,0),40,1)
 open_=inside&(cl!=1)
 print('Label fixed-fine open ground',cl.shape,flush=True)
 labels=np.empty(cl.shape,dtype='int32');n=label(open_,output=labels)
 road_ids=np.unique(labels[(cl==2)&open_]);road_ids=road_ids[road_ids>0]
 allowed=np.zeros(n+1,dtype=bool);allowed[road_ids]=True
 eligible=allowed[labels]&open_
 counts=np.bincount(labels[open_],minlength=n+1);excluded=float(counts[~allowed].sum()*.25)
 del labels,counts,allowed
 areas={dx:open_.reshape(m.shape[0]*(20//dx),int(dx/.5),m.shape[1]*(20//dx),int(dx/.5)).sum((1,3),dtype=np.int64)*.25 for dx in [20,10]}
 roofs={dx:np.zeros_like(a) for dx,a in areas.items()}
 hist=np.zeros(4096,dtype=np.int64);roof_pixels=0;max_distance=0.;tiles=0
 # Core tiles have a halo. A selected nearest target closer than the halo is
 # globally nearest; unresolved cases abort, never silently truncate the search.
 block=1024;halo=512;nr,nc=cl.shape
 for r0 in range(0,nr,block):
  for c0 in range(0,nc,block):
   r1=min(nr,r0+block);c1=min(nc,c0+block)
   donor=inside[r0:r1,c0:c1]&(cl[r0:r1,c0:c1]==1)
   if not donor.any():continue
   rs=max(0,r0-halo);re=min(nr,r1+halo);cs=max(0,c0-halo);ce=min(nc,c1+halo)
   e=eligible[rs:re,cs:ce];assert e.any(),'No road-connected receiving ground in halo'
   dist,idx=distance_transform_edt(~e,return_indices=True)
   dr,dc=np.where(donor);rr=dr+r0-rs;cc=dc+c0-cs;ds=dist[rr,cc]
   assert np.all(ds<halo),'Increase halo: nearest recipient not globally guaranteed'
   target_r=idx[0,rr,cc]+rs;target_c=idx[1,rr,cc]+cs
   assert np.all(eligible[target_r,target_c])
   for dx in [20,10]:
    f=int(dx/.5);a=roofs[dx];ix=(target_r//f)*a.shape[1]+target_c//f
    a+=np.bincount(ix,minlength=a.size).reshape(a.shape)*.25
   roof_pixels+=len(ds);max_distance=max(max_distance,float(ds.max()*.5));hist+=np.bincount(np.minimum((ds*.5).astype(int),len(hist)-1),minlength=len(hist));tiles+=1
  print('Allocation fine rows',r0,'/',nr,flush=True)
 roof_area=roof_pixels*.25;expected=float((m==1).sum()*400)
 assert np.isclose(roof_area+float(open_.sum()*.25),expected,rtol=0,atol=1e-8)
 for dx in [20,10]:
  a=areas[dx]+roofs[dx]
  with rasterio.open(P/f'active_mask_{dx}m.tif') as d:mm=d.read(1)
  assert np.all(a[mm!=1]==0);assert float(a.sum())==expected
  profile=dict(driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float64',crs=6677,transform=rasterio.transform.from_origin(tf.c,tf.f,dx,dx),compress='deflate',tiled=True)
  for name,v in [('contributing_area',a),('roof_contributing_area',roofs[dx])]:
   with rasterio.open(OUT/f'{name}_{dx}m.tif','w',**profile) as d:d.write(v,1)
 assert np.array_equal((areas[10]+roofs[10]).reshape(*[m.shape[0],2,m.shape[1],2]).sum((1,3)),areas[20]+roofs[20])
 c={'source_sha256':hashes,'fine_geometry_resolution_m':.5,'original_interior_area_m2':expected,'open_ground_area_m2':float(open_.sum()*.25),'all_roof_area_m2':roof_area,'road_connected_open_components':len(road_ids),'all_open_components':int(n),'non_road_connected_open_area_m2':excluded,'roof_nearest_distance_max_m':max_distance,'roof_nearest_distance_quantiles_m_1m_histogram':{str(p):int(np.searchsorted(np.cumsum(hist),p*roof_pixels))+1 for p in [.5,.9,.95,.99]},'halo_m':halo*.5,'tile_count':tiles,'20m_10m_contributing_area_exactly_nested':True,'mass_conserved':True,'method':'Every fixed 0.50 m roof pixel sends its rain to nearest open-ground pixel belonging to a 4-connected component containing classified ground roads; open-ground rain remains in place; aggregate these identical recipients conservatively to either hydraulic grid. Euclidean transfer is a roof-drainage scenario, not a surface-flow route or measured downpipe network.','roof_transfer_loss_or_delay':0,'pipes_infiltration_pumps':False,'building_hydraulics_unchanged':True,'measured_downpipes_available':False,'uncertainty':'Excluded isolated courtyards retain their own direct rain but receive no roof runoff. Roof runoff could instead enter underground pipes in reality; this scenario sends all of it to the surface.'}
 (OUT/'contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');print(json.dumps(c,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
