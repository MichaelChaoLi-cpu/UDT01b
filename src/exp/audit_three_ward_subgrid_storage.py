"""Batch audit of shallow-water volume tables against the fixed fine terrain."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,argparse
import numpy as np
import rasterio,netCDF4
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';RUN=ROOT/'data/processed/three_ward_sfincs_framework/urban_grid_comparison';OUT=ROOT/'data/exp/data-briefing/three_ward_fixed_runoff'

def main(levels=64):
 OUT.mkdir(exist_ok=True);tables={};summaries=[]
 for dx in [20,10]:
  with rasterio.open(P/f'active_mask_{dx}m.tif') as d:m=d.read(1)
  ii=np.where(np.flipud(m).ravel(order='F')>0)[0];rs,cs=np.unravel_index(ii,m.shape,order='F');rn=m.shape[0]-1-rs
  with netCDF4.Dataset((RUN/f'urban_{dx}m_template/sfincs_subgrid.nc' if levels==64 else RUN.parent/f'fixed_runoff_comparison/urban_storage{levels}_{dx}m_template/sfincs_subgrid.nc')) as d:
   z=np.full((levels,*m.shape),np.nan,dtype='float32');z[:,rn,cs]=np.asarray(d['z_level'][:],dtype='float32').T;vol=np.full(m.shape,np.nan,dtype='float32');vol[rn,cs]=np.asarray(d['z_volmax'][:])
  tables[dx]=(m,z,vol)
 with rasterio.open(P/'hydraulic_obstacle_0_50m.tif') as dem,rasterio.open(P/'constraints_0_50m.tif') as con:
  metrics={(dx,h):{'errors':[],'relative_errors':[],'exact_sum':0.,'table_sum':0.,'building_exact_sum':0.,'building_table_sum':0.} for dx in [20,10] for h in [.05,.1,.3]}
  for r0 in range(0,dem.height,800):
   win=Window(0,r0,dem.width,min(800,dem.height-r0));fine=dem.read(1,window=win);cl=con.read(1,window=win)
   for dx,(m,z,vol) in tables.items():
    f=int(dx/.5);start=r0//f;rows=int(win.height)//f;lev=z[:,start:start+rows];vm=vol[start:start+rows];active=m[start:start+rows]==1;building=(cl==1).reshape(rows,f,m.shape[1],f).any((1,3))&active
    for height in [.05,.1,.3]:
     water=lev[0]+height;count=np.maximum(0,np.minimum(levels-2,(lev<=water[None]).sum(0)-1));lo=np.take_along_axis(lev,count[None],0)[0];hi=np.take_along_axis(lev,(count+1)[None],0)[0];vv=(count+(water-lo)/np.maximum(hi-lo,1e-10))*vm/(levels-1);vv=np.where(water>lev[-1],vm+(water-lev[-1])*dx**2,vv)
     w=np.repeat(np.repeat(water,f,0),f,1);exact=np.maximum(w-fine,0).reshape(rows,f,m.shape[1],f).sum((1,3),dtype=np.float64)*.25;err=(vv-exact)/dx**2;rel=(vv-exact)/np.maximum(exact,1e-6);q=metrics[(dx,height)];q['errors'].append(err[active]);q['relative_errors'].append(rel[active]);q['exact_sum']+=float(exact[active].sum());q['table_sum']+=float(vv[active].sum());q['building_exact_sum']+=float(exact[building].sum());q['building_table_sum']+=float(vv[building].sum())
   print('Storage audit fine rows',r0,flush=True)
 for (dx,height),q in metrics.items():
  errors=np.concatenate(q.pop('errors'));rels=np.concatenate(q.pop('relative_errors'));row={'dx_m':dx,'water_level_above_cell_lowest_ground_m':height,'active_cells':len(errors),'equivalent_full_cell_depth_error_mean_m':float(errors.mean()),'abs_equivalent_depth_error_p95_m':float(np.percentile(abs(errors),95)),'table_vs_exact_volume_relative_bias':q['table_sum']/q['exact_sum']-1,'building_cells_table_vs_exact_volume_relative_bias':q['building_table_sum']/q['building_exact_sum']-1,'fraction_cells_relative_volume_error_above10percent':float((abs(rels)>.1).mean()),**q};summaries.append(row)
 record={'method':'Compare official equal-volume hypsometry with exact 0.50m raster storage at each cell minimum + 0.05/0.10/0.30m. This is a batch table-approximation audit, not an event recalibration or point survey. Values are aggregated over the same original model domain; active-cell supports differ only for fully covered buildings. Fine elevations are pre-builder raster values; builder tie-breaking increments of 1e-6m introduce an additional small residual.','nlevels':levels,'results':summaries,'ground_modified':False}
 (OUT/('shallow_storage_audit.json' if levels==64 else f'shallow_storage_audit_{levels}.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--levels',type=int,choices=[64,253],default=64);a=ap.parse_args();main(a.levels)
