"""Independent official-solver input/geometry audit for the paired model."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,hashlib
import numpy as np,xarray as xr,rasterio,netCDF4
from hydromt_sfincs.utils import find_uv_indices
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_adaptive_connectivity';RUN=ROOT/'data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison';OUT=ROOT/'data/exp/data-briefing/three_ward_adaptive_connectivity'
def main():
 OUT.mkdir(exist_ok=True)
 with rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/active_mask_10m.tif') as d:m=d.read(1)
 with np.load(P/'regular_face_geometry.npz') as q:rr=q['row_fine'];cc=q['col_fine'];di=q['direction']
 nm,mu,nu=find_uv_indices(xr.DataArray(np.flipud(m),dims=['y','x']));rn=np.where(di==0,rr//20,(rr+1)//20);col=cc//20-np.where(di==0,1,0);flat=col*m.shape[0]+m.shape[0]-1-rn;ids=np.where(di==0,mu[flat],nu[flat]);uv_order=bool(np.array_equal(ids,np.arange(len(rr))))
 with np.load(P/'adaptive_cell_geometry.npz') as g:
  coarse=np.flatnonzero(g['size_m']==10);rows=g['row_fine'][coarse]//20;cols=g['col_fine'][coarse]//20;lookup=np.full(m.shape,-1,np.int32);ii=np.flatnonzero(np.flipud(m).ravel(order='F')>0);rs,cs=np.unravel_index(ii,m.shape,order='F');lookup[m.shape[0]-1-rs,cs]=np.arange(len(ii));original=lookup[rows,cols];assert np.all(original>=0);errors={k:0. for k in ['z_zmin','z_zmax','z_volmax','z_level']}
  with netCDF4.Dataset(RUN/'corrected_10m_template/sfincs_subgrid.nc') as a,netCDF4.Dataset(RUN/'adaptive_10_5m_template/sfincs_subgrid.nc') as b:
   for j in range(0,len(coarse),20000):
    for key in errors:errors[key]=max(errors[key],float(np.max(abs(np.asarray(a[key][original[j:j+20000]],dtype='float32')-np.asarray(b[key][coarse[j:j+20000]])))))
  expected_rain=float(g['contributing_area_m2'][g['mask']==1].sum());hashes={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in json.loads((P/'fine_cache.json').read_text())['source_sha256'].items()}
 closure=[]
 for case,kind in [('corrected_10m','regular'),('adaptive_10_5m','adaptive')]:
  with np.load(P/(kind+'_face_geometry.npz')) as q:
   w=q['native_open_width_m'];di=q['direction'];rr=q['row_fine'];cc=q['col_fine'];ll=q['length_pixels']
   if kind=='regular':
    x=-11360+cc*.5;y=-32320-rr*.5;y=np.where(di==1,y-.5,y-ll*.25);x=np.where(di==1,x+ll*.25,x)
   else:
    with np.load(P/'adaptive_cell_geometry.npz') as g:x=(g['x'][q['cell_a']]+g['x'][q['cell_b']])/2;y=(g['y'][q['cell_a']]+g['y'][q['cell_b']])/2
  with netCDF4.Dataset(RUN/('pilot_'+case)/'sfincs_his.nc') as d:points=np.c_[d['thindam_x'][:],d['thindam_y'][:]]
  dist,ids=cKDTree(np.c_[x[w==0],y[w==0]]).query(points);expected=json.loads((RUN/(case+'_template')/'builder_complete.json').read_text())['fully_building_closed_faces'];closure.append({'case':case,'expected_closed_faces':expected,'solver_closed_faces':len(points),'unique_matches':len(np.unique(ids)),'maximum_solver_midpoint_match_error_m':float(dist.max()),'passed':len(points)==expected and len(np.unique(ids))==expected and float(dist.max())<.01})
 pilots={case:json.loads((RUN/('pilot_'+case)/'forcing_validation.json').read_text()) for case in ['corrected_10m','adaptive_10_5m']};boundary=json.loads((P/'boundary_geometry_equivalence.json').read_text());result={'official_regular_face_index_order_exact':uv_order,'coarse_storage_float32_comparison_max_errors':errors,'raw_input_hashes_unchanged':hashes,'nested_rain_contributing_area_m2':expected_rain,'pilot_precipitation':pilots,'building_closure_snap_audit':closure,'boundary_geometry_equivalence':boundary};result['passed']=uv_order and max(errors.values())==0 and all(hashes.values()) and all(q['passed'] for q in closure) and all(q['passed'] for q in pilots.values()) and boundary['removed_interfaces_already_building_closed_in_baseline']
 (OUT/'geometry_and_forcing_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
