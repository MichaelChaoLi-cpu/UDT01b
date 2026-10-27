#!/usr/bin/env python3
"""Separate three-ward and buffer storage without rerunning connectivity or hydraulics."""
from pathlib import Path
import csv,json,time
import numpy as np,pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_protection_requirements_register_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG;BASE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
def main():
 start=time.monotonic();controls=pd.read_csv(R/'control_storage_native.csv');cg=controls['Grid Index'].to_numpy();target=controls['Target Storage'].to_numpy();possible=controls['Target Above Native Minimum'].to_numpy()
 with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz') as z:ward=z['Ward Code'].ravel()
 inventory=pd.read_csv(BASE/'input_inventory.csv');fields=['Case','Zone','Frame','Three Ward Stored Volume','Buffer Stored Volume','Three Ward Control Excess Storage','Buffer Control Excess Storage'];checks=[]
 with (R/'zone_spatial_storage_native.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in inventory.itertuples(index=False):
   case=row.case_id
   with np.load(P/'zones'/f'{case}.npz') as z:cells=z['node_grid_index'];label=z['zone_index'];ids=z['zone_ids']
   lookup=np.full(1012*770,-1,dtype='int32');lookup[cells]=np.arange(len(cells));ix=lookup[cg];assert (ix>=0).all();cl=label[ix];inside=ward[cells]>0;controlinside=ward[cg]>0;n=len(ids)
   with Dataset(ROOT/row.map_path) as d:
    for frame in range(109):
     v=np.flipud(np.ma.asarray(d['subgrid_volume'][frame],dtype='float64').filled(np.nan)).ravel()[cells];assigned=label>0;assert np.isfinite(v[assigned]).all()
     # Unassigned cells can be outside the common mask;never enter sums.
     a=np.bincount(label[assigned&inside],weights=v[assigned&inside],minlength=n+1);b=np.bincount(label[assigned&~inside],weights=v[assigned&~inside],minlength=n+1)
     e=np.maximum(v[ix]-target,0.);e[~possible]=0.;c=np.bincount(cl[controlinside],weights=e[controlinside],minlength=n+1);dd=np.bincount(cl[~controlinside],weights=e[~controlinside],minlength=n+1)
     for zi,zid in enumerate(ids,1):writer.writerow(dict(zip(fields,[case,str(zid),frame,float(a[zi]),float(b[zi]),float(c[zi]),float(dd[zi])])))
   checks.append({'case':case,'frames':109,'zones':n});print('SCOPE_SPLIT',case,flush=True)
 (R/'scope_split_validation.json').write_text(json.dumps({'passed':True,'cases':checks,'three_ward_control_cells':int(controlinside.sum()),'buffer_control_cells':int((~controlinside).sum()),'new_hydraulics':0,'seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
