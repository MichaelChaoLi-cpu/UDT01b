"""Geometry-only batch screen of disconnected open spaces merged in one cell.
5m here is a geometry audit, never a hydraulic simulation or observed connection.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json
import numpy as np
import rasterio
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';OUT=ROOT/'data/exp/data-briefing/three_ward_fixed_runoff'

def main():
 OUT.mkdir(exist_ok=True)
 with rasterio.open(P/'constraints_0_50m.tif') as d:cl=d.read(1)
 with rasterio.open(OLD/'active_mask_20m.tif') as d:original=d.read(1)
 results=[]
 for dx in [20,10,5]:
  f=int(dx/.5);m=np.repeat(np.repeat(original,20//dx,0),20//dx,1);flags=np.zeros(m.shape,dtype='uint8');open_count=0;flag_open_count=0;road_flag_open=0;active_cells=0
  for r in range(m.shape[0]):
   for c in np.where(m[r]==1)[0]:
    a=cl[r*f:(r+1)*f,c*f:(c+1)*f];op=a!=1;cnt=int(op.sum())
    if not cnt:continue
    active_cells+=1;open_count+=cnt
    if not np.any(a==1):continue
    lab,n=label(op)
    if n<2:continue
    edge_ids=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]));edge_ids=edge_ids[edge_ids>0]
    if len(edge_ids)>=2:
     flags[r,c]|=1;flag_open_count+=cnt
    road_ids=np.unique(lab[a==2]);road_ids=road_ids[road_ids>0]
    if len(np.intersect1d(edge_ids,road_ids))>=2:
     flags[r,c]|=2;road_flag_open+=cnt
   if r%100==0:print('Connectivity audit',dx,'row',r,'/',m.shape[0],flush=True)
  q={'geometry_grid_m':dx,'active_open_cells':active_cells,'common_open_area_km2':open_count*.25/1e6,'cells_multiple_edge_touching_open_components':int(((flags&1)>0).sum()),'open_ground_inside_geometry_flagged_cells_km2':flag_open_count*.25/1e6,'flagged_open_area_fraction':flag_open_count/max(open_count,1),'cells_multiple_road_components_touching_edges':int(((flags&2)>0).sum()),'open_ground_inside_road_flagged_cells_km2':road_flag_open*.25/1e6,'hydraulic_simulation_at_this_grid_available':dx!=5}
  results.append(q)
  with rasterio.open(P/f'subcell_connectivity_flags_{dx}m.tif','w',driver='GTiff',height=flags.shape[0],width=flags.shape[1],count=1,dtype='uint8',crs=6677,transform=rasterio.transform.from_origin(-11360,-32320,dx,dx),compress='deflate',tiled=True) as d:d.write(flags,1)
 record={'results':results,'method':'Each aligned grid cell is split into 4-connected open-ground components on the fixed 0.50m building mask. Flag if multiple components touch cell faces; separately flag if multiple such components contain classified ground roads. This identifies potential single-water-level mixing inside a coarse cell. No ground elevation, flow direction, water depth, curb, pipe or observed hydraulic connection is inferred.','flag_is_proof_of_false_actual_flow':False,'scope':'whole original interior domain including cross-ward contributing area; fully covered building cells excluded','original_inputs_modified':False,'5m_geometry_only_no_hydraulic_convergence_claim':True}
 (OUT/'subcell_connectivity_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
