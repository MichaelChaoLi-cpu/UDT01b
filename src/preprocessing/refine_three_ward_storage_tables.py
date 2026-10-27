"""Refine official equal-volume storage tables; preserve existing edge conveyance.
253 = 4*(64-1)+1, so every old face-table knot is retained exactly.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import rasterio,netCDF4
from rasterio.windows import Window
from numba import njit,prange
from hydromt_sfincs.subgrid import subgrid_v_table
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';BASE=ROOT/'data/processed/three_ward_sfincs_framework';PRIOR=BASE/'urban_grid_comparison';OUT=BASE/'fixed_runoff_comparison'
@njit(parallel=True)
def refine_cells(blocks,active):
 out=np.zeros((len(active),253),dtype=np.float32)
 for j in prange(len(active)):
  z,v,zmin,zmax=subgrid_v_table(blocks[active[j]],.5,.5,253,-20.,99999.)
  out[j]=z
 return out

def main(dx):
 folder=OUT/f'urban_storage253_{dx}m_template';folder.mkdir(exist_ok=True)
 if (folder/'storage_refinement_complete.json').exists():print('Reuse',folder,flush=True);return
 source=PRIOR/f'urban_{dx}m_template'
 for p in source.iterdir():
  if p.suffix in ['.inp','.ind','.msk','.obs','.crs','.dep'] or p.name=='flux_sections.json':shutil.copy2(p,folder/p.name)
 with rasterio.open(P/f'active_mask_{dx}m.tif') as d:m=d.read(1)
 ii=np.where(np.flipud(m).ravel(order='F')>0)[0];rs,cs=np.unravel_index(ii,m.shape,order='F');rn=m.shape[0]-1-rs;lookup=np.full(m.shape,-1,dtype='int32');lookup[rn,cs]=np.arange(len(ii));zs=np.empty((len(ii),253),dtype='float32');f=int(dx/.5)
 with rasterio.open(P/'hydraulic_obstacle_0_50m.tif') as d:
  for r0 in range(0,d.height,800):
   win=Window(0,r0,d.width,min(800,d.height-r0));a=d.read(1,window=win);r=r0//f;rows=int(win.height)//f;sel=m[r:r+rows].ravel()>0;ids=np.where(sel)[0];blocks=np.ascontiguousarray(a.reshape(rows,f,m.shape[1],f).transpose(0,2,1,3).reshape(rows*m.shape[1],f*f));assert np.isfinite(blocks[ids]).all();z=refine_cells(blocks,ids);zs[lookup[r:r+rows].ravel()[ids]]=z;print('Refine storage',dx,r0,'/',d.height,flush=True)
 source_nc=source/'sfincs_subgrid.nc';dest=folder/'sfincs_subgrid.nc';error=0.;uv_error=0.
 with netCDF4.Dataset(source_nc) as old,netCDF4.Dataset(dest,'w') as new:
  for name,dim in old.dimensions.items():new.createDimension(name,253 if name=='levels' else len(dim))
  for key in old.ncattrs():new.setncattr(key,old.getncattr(key))
  new.setncattr('storage_refinement','253 equal-volume knots via unmodified HydroMT-SFINCS1.2.2 subgrid_v_table; face curves resampled retaining every old knot exactly; no DEM changes')
  for name,v in old.variables.items():
   nv=new.createVariable(name,v.dtype,v.dimensions,fill_value=v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None)
   for key in v.ncattrs():
    if key!='_FillValue':nv.setncattr(key,v.getncattr(key))
   if name=='z_level':
    for i in range(0,len(ii),5000):
     original=np.asarray(v[i:i+5000]);assert np.isfinite(original).all() and np.isfinite(zs[i:i+5000]).all();error=max(error,float(abs(zs[i:i+5000,::4]-original).max()));nv[i:i+5000]=zs[i:i+5000]
   elif 'levels' in v.dimensions:
    assert v.dimensions[-1]=='levels'
    for i in range(0,v.shape[0],5000):
     a=np.asarray(v[i:i+5000]);assert np.isfinite(a).all();b=np.empty((len(a),253),dtype=a.dtype)
     for j in range(63):
      for k in range(4):b[:,j*4+k]=a[:,j]+(a[:,j+1]-a[:,j])*(k/4)
     b[:,-1]=a[:,-1];uv_error=max(uv_error,float(abs(b[:,::4]-a).max()));nv[i:i+5000]=b
   else:nv[:]=v[:]
 assert error<.002,f'Fine sample/table mismatch: {error}';assert uv_error==0.
 record={'dx_m':dx,'old_nlevels':64,'new_nlevels':253,'source_subgrid_sha256':hashlib.sha256(source_nc.read_bytes()).hexdigest(),'fine_terrain_sha256':hashlib.sha256((P/'hydraulic_obstacle_0_50m.tif').read_bytes()).hexdigest(),'new_subgrid_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'max_storage_old_knot_elevation_difference_m':error,'max_face_old_knot_difference':uv_error,'face_piecewise_linear_functions_preserved':True,'building_height_and_all_ground_elevations_unchanged':True,'method':'Only the equal-volume storage curve is recomputed from original fine terrain using the pinned official builder routine. Old face tables are linearly subdivided 4-fold with all old nodes retained, preserving existing piecewise-linear conveyance rather than conflating two changes.'}
 (folder/'storage_refinement_complete.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dx',type=int,choices=[20,10],required=True);args=ap.parse_args();main(args.dx)
