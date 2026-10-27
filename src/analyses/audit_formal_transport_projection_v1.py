"""Direct fine-raster/raw-NetCDF checks at distributed cells and synchronized states."""
from pathlib import Path
import os,sys,json
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_formal_transport_access_v1 import P,R,OLD,STRICT,F,H,CASES,read,load,dump

def main():
 s=load(P/'projection_support.npz');support=load(F/'factor_comparison_v1/common_support.npz');cache=np.load(STRICT/'fine_elevation_support.npz');rindex={int(c):i for i,c in enumerate(s['road_cells'])};gindex={int(c):i for i,c in enumerate(s['ground_cells'])};checks=[]
 rc=s['road_cells'][s['road_known']];gc=s['ground_cells'][s['ground_known']];chosen=np.unique(np.r_[rc[np.linspace(0,len(rc)-1,48,dtype=int)],gc[np.linspace(0,len(gc)-1,24,dtype=int)]])
 fine={}
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif')as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif')as cl:
  for cell in chosen:
   r,c=divmod(int(cell),770);w=Window(c*20,r*20,20,20);z=dem.read(1,window=w);classes=cl.read(1,window=w);fine[int(cell)]={'road':z[np.isfinite(z)&(classes==2)].astype('f8'),'ground':z[np.isfinite(z)&(classes!=1)].astype('f8')}
 runs={r['case_id']:r for r in json.loads((F/'run_manifest.json').read_text())['runs']};policy=json.loads((P/'analysis_contract.json').read_text());frames=policy['frame_ids'];state_frames=['0','14400','32400'];comparisons=0
 for case in CASES:
  run=runs[case];e=load(P/'exposures'/(case+'.npz'));old=run.get('original_case_id',case);base=H if run['reused']else F;clock=load(base/'execution'/(old+'_output_clock.npz'))['Actual Map Time'];passed=True;maxerr=0.
  with Dataset(ROOT/run['folder']/'sfincs_map.nc')as ds:
   bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for frame in state_frames:
    ti=frames.index(frame);k=int(np.argmin(abs(clock-int(frame))));passed &=abs(clock[k]-int(frame))<=.001 and abs(e['actual_times'][ti]-clock[k])<1e-12;eta=np.flipud(np.ma.asarray(ds['zs'][k],dtype='f8').filled(np.nan)).ravel()
    for cell in chosen:
     for name,index in [('road',rindex),('ground',gindex)]:
      if int(cell)not in index or not len(fine[int(cell)][name]):continue
      z=fine[int(cell)][name];lo=float(np.maximum(eta[cell]-z,0).mean())if np.isfinite(eta[cell])else 0.;up=lo if np.isfinite(eta[cell])else float(np.maximum(bed[cell]+.01-z,0).mean());j=index[int(cell)];x=e[name+'_lower'][ti,j];y=e[name+'_upper'][ti,j];maxerr=max(maxerr,abs(x-lo),abs(y-up));passed &=abs(x-lo)<1e-7 and abs(y-up)<1e-7;comparisons+=1
  checks.append(dict(check=case+'_distributed_fine_integrals_and_actual_clock',passed=bool(passed),max_depth_error_m=maxerr));assert passed,case
 dump(R/'projection_direct_validation.json',dict(all_passed=True,checks=checks,distributed_cells=len(chosen),synchronous_frame_ids=state_frames,individual_depth_comparisons=comparisons,depth_tolerance_m=1e-7,new_hydraulics=False));print('DIRECT PROJECTION CHECKS',comparisons,'PASSED')
if __name__=='__main__':main()
