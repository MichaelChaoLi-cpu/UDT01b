#!/usr/bin/env python3
"""Stream existing native maps into provisional control-zone storage registers.
Run in the existing physical-measures Python runtime (netCDF4 present).
No hydraulic executable is invoked. Zones are observed-wet connectivity envelopes,
not contributing catchments;screening storage is not required pumping capacity.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_protection_requirements_register_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
BASE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def arr(x):return np.ma.asarray(x,dtype='float64').filled(np.nan)

def main():
 start=time.monotonic();(P/'zones').mkdir(exist_ok=True)
 with np.load(P/'road_control_bridge.npz') as z:control_grid=z['grid_index'];control_target=z['target_level'];link_grid=z['link_grid'];link_group=z['link_group'];group_ids=z['group_ids']
 with np.load(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz') as z:native_grid=z['native_grid_north_index'];native_kcs=z['native_kcs']
 with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz') as z:
  nodegrid=z['Node Grid Index'];edges=z['Surface Edge Index'];ef=z['Surface Edge Features'];face_ids=z['Surface Edge Native Face Id'];receiver_link=z['Receiver Link Index'];receiver_native_face=z['Receiver Link Native Face Id']
 with np.load(ROOT/'data/processed/three_ward_five_event_population_validation_v1/motor_common_support.npz') as z:common=z['common_admitted'].ravel()
 with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz') as z:ward=z['Ward Code'].ravel()
 n=len(nodegrid);keepnode=common[nodegrid];gridtonode=np.full(1012*770,-1,dtype='int32');gridtonode[nodegrid]=np.arange(n)
 gridtonative=np.full(1012*770,-1,dtype='int32');gridtonative[native_grid]=np.arange(len(native_grid));native_ix=gridtonative[nodegrid]
 assert (native_ix>=0).all() and (native_kcs[native_ix]==1).all()
 # Deduplicate reciprocal surface edges before undirected component analysis.
 sel=edges[0]<edges[1];all_a,all_b=edges[:,sel];all_fids=face_ids[sel];crest=ef[sel,2];supported=keepnode[all_a]&keepnode[all_b];a=all_a[supported];b=all_b[supported];crest=crest[supported];fids=all_fids[supported]
 ci=gridtonode[control_grid];control_ok=ci>=0;control_ok[control_ok]&=keepnode[ci[control_ok]]
 ci=ci[control_ok];cg=control_grid[control_ok];target=control_target[control_ok];ng=native_ix[ci]
 inventory=pd.read_csv(BASE/'input_inventory.csv');assert len(inventory)==10
 ref=ROOT/str(inventory.map_path.iloc[0]);curvepath=ref.parent/'sfincs_subgrid.nc'
 with Dataset(curvepath) as d:
  levels=np.asarray(d['z_level'][:],dtype='float32');vmax=np.asarray(d['z_volmax'][:],dtype='float64');zmin=np.asarray(d['z_zmin'][:],dtype='float64');zmax=np.asarray(d['z_zmax'][:],dtype='float64')
 input_paths=[ROOT/str(v) for v in inventory.map_path]+[curvepath,BASE/'input_inventory.csv',ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz',ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz',ROOT/'data/processed/three_ward_five_event_population_validation_v1/motor_common_support.npz',ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz',P/'road_control_bridge.npz']
 initial_stats={str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in input_paths}
 assert levels.shape==(len(native_grid),253) and np.isfinite(levels).all()
 def eta_of_volume(v,ix):
  vm=vmax[ix];z=np.maximum(zmin[ix],-20.).copy();full=v>=vm*.999
  z[full]=np.maximum(zmax[ix[full]],-20.)+(v[full]-vm[full])/100.
  m=~full&(v>1e-6);loc=np.flatnonzero(m);ids=ix[loc];frac=v[loc]/(vmax[ids]/252.);j=np.minimum(np.floor(frac).astype('int32'),251);frac-=j
  z[loc]=levels[ids,j]+frac*(levels[ids,j+1]-levels[ids,j]);return z
 # Invert the exact source operator,including its0.999full-cell branch.
 target_possible=target>=np.maximum(zmin[ng],-20.)
 lower=np.zeros(len(ng));upper=vmax[ng]+100*np.maximum(0,target-zmax[ng])+1.
 for _ in range(45):
  mid=(lower+upper)/2;below=eta_of_volume(mid,ng)<target;lower[below]=mid[below];upper[~below]=mid[~below]
 target_storage=upper;target_storage[~target_possible]=np.nan
 inv_error=np.max(abs(eta_of_volume(upper[target_possible],ng[target_possible])-target[target_possible]),initial=0.)
 assert inv_error<1e-4,'Storage inversion cannot support a point-valued diagnostic'
 pd.DataFrame({'Grid Index':cg,'Target Water Level':target,'Target Storage':target_storage,'Target Above Native Minimum':target_possible}).to_csv(R/'control_storage_native.csv',index=False)
 zone_rows=[];links_rows=[];case_checks=[]
 trajectory_path=R/'zone_storage_native.csv';fields=['Case','Zone','Frame','Actual Time','Analysis Time','Stored Volume','Road Control Excess Storage','Road Control Stored Volume']
 with trajectory_path.open('w',newline='') as tf:
  writer=csv.DictWriter(tf,fieldnames=fields);writer.writeheader()
  for case_i,row in inventory.iterrows():
   if (P/'STOP').exists():raise RuntimeError('User stop')
   assert time.monotonic()-start<2400,'Native existing-result audit time budget'
   case=str(row.case_id);path=ROOT/str(row.map_path);assert path.exists()
   assert os.path.samefile(path.parent/'sfincs_subgrid.nc',curvepath),'Different storage curve must be independently audited'
   with np.load(ROOT/str(row.clock_path)) as z:actual=z['Actual Map Time'].astype(float)
   times=actual.copy();times[0]=0.;times[-1]=32400.
   assert len(actual)==109 and np.max(abs(actual-np.arange(109)*300))<=.001
   volume=np.lib.format.open_memmap(R/'working_volumes.npy',mode='w+',dtype='float32',shape=(109,n))
   excess=np.zeros((109,len(ci)),dtype='float64');everwet=np.zeros(n,bool);activeface=np.zeros(len(a),bool);max_recon=0.;missing=0
   with Dataset(path) as d:
    assert d['subgrid_volume'].shape==(109,1012,770) and d['subgrid_volume'].dtype==np.dtype('float32')
    for frame in range(109):
     v=arr(d['subgrid_volume'][frame]);v=np.flipud(v).ravel()[nodegrid]
     assert np.isfinite(v[keepnode]).all() and v[keepnode].min(initial=0)>=0
     v[~keepnode]=0.;volume[frame]=v.astype('float32')
     eta=eta_of_volume(v,native_ix);stored_eta=np.flipud(arr(d['zs'][frame])).ravel()[nodegrid];finite=keepnode&np.isfinite(stored_eta)
     error=np.max(abs(eta[finite]-stored_eta[finite]),initial=0.);max_recon=max(max_recon,float(error))
     assert error<1e-4,'Native storage-to-level mismatch'
     wet=keepnode&(v>1e-6);everwet|=wet
     activeface|=wet[a]&wet[b]&(np.maximum(eta[a],eta[b])>=crest)
     ec=np.maximum(v[ci]-target_storage,0.);ec[~target_possible]=0.;excess[frame]=ec
     if frame%27==0:
      dump(R/'progress_native.json',{'stage':'native_storage','case':case,'case_number':case_i+1,'cases':10,'frame':frame,'frames':109,'seconds':time.monotonic()-start});print('NATIVE',case,frame,'/108',flush=True)
   volume.flush()
   graph=coo_matrix((np.ones(int(activeface.sum()),dtype='uint8'),(a[activeface],b[activeface])),shape=(n,n)).tocsr()
   _,component=connected_components(graph,directed=False)
   requested=np.unique(component[ci[(excess.max(axis=0)>1e-6)&target_possible]])
   # Zones are fixed observed-wet connectivity envelopes over this case,not catchments.
   label=np.zeros(n,dtype='int32');ids=[]
   for zi,c in enumerate(requested,1):label[(component==c)&everwet]=zi;ids.append(f'{case}__Z{zi:05d}')
   nz=len(ids);control_label=label[ci];zonecell=np.bincount(label,minlength=nz+1)
   zoneward={code:np.bincount(label[ward[nodegrid]==code],minlength=nz+1) for code in [0,1,2,3]}
   zonecontrols=np.bincount(control_label[target_possible],minlength=nz+1)
   lg_nodes=gridtonode[link_grid];supported_lg=lg_nodes>=0;lg_zone=np.zeros(len(lg_nodes),int);lg_zone[supported_lg]=label[lg_nodes[supported_lg]]
   link_pairs=np.unique(np.stack([lg_zone[lg_zone>0],link_group[lg_zone>0]],axis=1),axis=0) if (lg_zone>0).any() else np.empty((0,2),int)
   for zoneid,group in link_pairs:links_rows.append({'Case':case,'Zone':ids[int(zoneid)-1],'Road Group':str(group_ids[int(group)]),'Join Rule':'many-to-many membership;do not add repeated zone volumes across groups'})
   counts_case=[];maxvol=np.zeros(nz+1);maxex=np.zeros(nz+1);peakex=np.zeros(nz+1,dtype=int)
   for frame in range(109):
    values=np.asarray(volume[frame],dtype='float64');vv=np.bincount(label,weights=values,minlength=nz+1);ee=np.bincount(control_label,weights=excess[frame],minlength=nz+1);rc=np.bincount(control_label,weights=values[ci]*target_possible,minlength=nz+1)
    assert abs(vv[1:].sum()-values[label>0].sum())<1e-5
    assert abs(ee[1:].sum()-excess[frame,control_label>0].sum())<1e-5
    higher=ee>maxex;peakex[higher]=frame;maxvol=np.maximum(maxvol,vv);maxex=np.maximum(maxex,ee)
    for zi,zid in enumerate(ids,1):writer.writerow({'Case':case,'Zone':zid,'Frame':frame,'Actual Time':float(actual[frame]),'Analysis Time':float(times[frame]),'Stored Volume':float(vv[zi]),'Road Control Excess Storage':float(ee[zi]),'Road Control Stored Volume':float(rc[zi])})
   for zi,zid in enumerate(ids,1):
    zone_rows.append({'Case':case,'Zone':zid,'Native Cells':int(zonecell[zi]),'Chiyoda Cells':int(zoneward[1][zi]),'Chuo Cells':int(zoneward[2][zi]),'Minato Cells':int(zoneward[3][zi]),'Outside Three Wards Cells':int(zoneward[0][zi]),'Road Control Cells':int(zonecontrols[zi]),'Maximum Simultaneous Stored Volume':float(maxvol[zi]),'Maximum Road Control Excess Storage':float(maxex[zi]),'Control Excess Peak Frame':int(peakex[zi]),'Volume Meaning':'common-supported observed-wet control-zone storage;not a contributing catchment','Excess Meaning':'subset of road-controlled cells only;not full-zone drainage requirement','Required Pump Capacity':None,'Required Capacity Status':'unresolved ongoing load,target stages outside road cells and local boundary-flux coverage','Target Freeze Status':'candidate linked groups;not a final protection target'})
   np.savez_compressed(P/'zones'/f'{case}.npz',node_grid_index=nodegrid,zone_index=label,zone_ids=np.array(ids),activated_native_faces=fids[activeface],actual_times=actual,analysis_times=times)
   # Record boundary face IDs needed for a future local budget;do not infer flux from storage differences.
   all_labels=label;different=all_labels[all_a]!=all_labels[all_b];boundarypairs=[]
   for zz in np.flatnonzero(different):
    for zi,other in [(int(all_labels[all_a[zz]]),int(all_labels[all_b[zz]])),(int(all_labels[all_b[zz]]),int(all_labels[all_a[zz]]))]:
     if zi>0:boundarypairs.append({'Case':case,'Zone':ids[zi-1],'Native Face':int(all_fids[zz]),'Other Zone Index':other,'Boundary Type':'ground_exchange','Flux Evidence':'not_assigned'})
   for j in range(receiver_link.shape[1]):
    zi=int(label[receiver_link[0,j]])
    if zi>0:boundarypairs.append({'Case':case,'Zone':ids[zi-1],'Native Face':int(receiver_native_face[j]),'Other Zone Index':-1,'Boundary Type':'prescribed_water_exchange','Flux Evidence':'existing_named_crosssection_mapping_not_yet_verified'})
   pd.DataFrame(boundarypairs,columns=['Case','Zone','Native Face','Other Zone Index','Boundary Type','Flux Evidence']).to_csv(R/f'{case}_zone_boundary_faces.csv',index=False)
   case_checks.append({'case':case,'frames':109,'zones':nz,'maximum_storage_to_level_error_m':max_recon,'volume_and_excess_partition_exact':True,'supported_controls':int(target_possible.sum()),'held_controls':int(len(control_grid)-target_possible.sum()),'boundary_face_records':len(boundarypairs),'actual_clock_max_difference_s':float(np.max(abs(actual-times)))})
   del volume
   (R/'working_volumes.npy').unlink()
   print('NATIVE_CASE_COMPLETE',case,'zones',nz,flush=True)
 pd.DataFrame(zone_rows).to_csv(R/'zone_register_native.csv',index=False);pd.DataFrame(links_rows).to_csv(R/'group_zone_links_native.csv',index=False)
 print('BIND_NATIVE_SOURCES',flush=True)
 source_sha={str(p.relative_to(ROOT)):sha(p) for p in input_paths}
 assert all((ROOT/k).stat().st_size==v[0] and (ROOT/k).stat().st_mtime_ns==v[1] for k,v in initial_stats.items())
 dump(R/'native_validation.json',{'passed':True,'cases':case_checks,'clock_and_original_native_storage_audit_passed':True,'storage_target_inversion_error_m':float(inv_error),'source_control_cells':len(control_grid),'ground_supported_controls':len(ci),'target_supported_controls':int(target_possible.sum()),'unsupported_or_below_min_controls':int(len(control_grid)-target_possible.sum()),'control_target_rule':'minimum sampled road elevation+0.30m;screening at closure threshold','zones':'fixed per-case observed-wet connectivity envelopes;not catchments','native_flow_budgets_reconstructed':False,'required_discharge_computed':False,'source_sha256':source_sha,'source_stats_unchanged':True,'new_hydraulics':0,'new_people_counts':False,'seconds':time.monotonic()-start})
 print('NATIVE_REGISTERS_COMPLETE',len(zone_rows),len(links_rows),flush=True)
if __name__=='__main__':main()
