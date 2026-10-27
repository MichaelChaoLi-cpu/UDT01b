#!/usr/bin/env python3
"""Audit saved receiving-face flow on existing provisional zones; no solver launch.
Uses actual per-step dt and the solver's float32 applied-volume multiplication.
Ground exchange is unobserved, never filled with zero or inferred from storage.
"""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from netCDF4 import Dataset,chartostring
ROOT=Path(__file__).resolve().parents[2]
OLD='three_ward_protection_requirements_register_v1';TAG='three_ward_target_boundary_budget_audit_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
OP=ROOT/'data/processed'/OLD;OR=ROOT/'data/exp/data-preprocessing'/OLD

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sections(p):
 lines=[s.strip() for s in p.read_text().splitlines() if s.strip()];out=[];j=0
 while j<len(lines):
  name=lines[j];n=int(lines[j+1].split()[0]);xy=np.array([[float(v) for v in s.split()] for s in lines[j+2:j+2+n]]);assert xy.shape==(2,2);out.append((name,xy));j+=n+2
 return out

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 source={};checks=[];allfaces=[];alltraj=[];allsums=[]
 def bind(p):
  key=str(p.relative_to(ROOT));st=p.stat();source[key]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(p)}
 topo=ROOT/'data/processed/three_ward_negative_trace_interface_v1/whole_domain_metadata/whole_native_topology.bin'
 with topo.open('rb') as f:
  n,nf=map(int,np.fromfile(f,'<i4',2));sr=np.fromfile(f,'<i4',n)-1;cc=np.fromfile(f,'<i4',n)-1;kcs=np.fromfile(f,'<i4',n);f.seek(n*4*4,1);a=np.fromfile(f,'<i4',nf)-1;b=np.fromfile(f,'<i4',nf)-1
 pos=np.c_[-11360+(cc+.5)*10,-42440+(sr+.5)*10];nativegrid=(1011-sr)*770+cc
 with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz') as z:blocked=np.isin(np.arange(nf),z['Blocked Native Face Id']);open_ground_ids=np.unique(z['Surface Edge Native Face Id'])
 with np.load(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz') as z:assert np.array_equal(nativegrid,z['native_grid_north_index'])
 receiver=np.flatnonzero(((kcs[a]==1)&(kcs[b]==2))|((kcs[a]==2)&(kcs[b]==1)));mid=(pos[a[receiver]]+pos[b[receiver]])/2
 inventory=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv';inv=pd.read_csv(inventory)
 bind(topo);bind(inventory);bind(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz')
 for _,row in inv.iterrows():
  assert time.monotonic()-start<2400,'Read-only audit time budget exceeded'
  if (P/'STOP').exists():raise RuntimeError('User stop requested')
  case=row.case_id;folder=(ROOT/row.map_path).parent
  audit=json.loads((ROOT/row.audit_path).read_text());assert audit['all_checks_passed']
  sec=sections(folder/'sfincs.crs');names=[s[0] for s in sec];xy=np.array([s[1].mean(0) for s in sec]);dist,ix=cKDTree(mid).query(xy);faces=receiver[ix]
  normal=np.array([[-(s[1][1,1]-s[1][0,1]),s[1][1,0]-s[1][0,0]] for s in sec]);land=np.where(kcs[a[faces]]==1,a[faces],b[faces]);water=np.where(kcs[a[faces]]==2,a[faces],b[faces])
  assert len(faces)==len(receiver)==len(np.unique(faces)) and dist.max()<1e-5
  assert (np.sum(normal*(pos[water]-pos[land]),axis=1)>0).all()
  with np.load(OP/'zones'/f'{case}.npz') as z:
   zid=z['zone_ids'];ng=z['node_grid_index'];zz=z['zone_index'];actual=z['actual_times']
  gridzone=np.zeros(1012*770,dtype='int32');gridzone[ng]=zz;nativezone=gridzone[nativegrid];nz=len(zid)
  # Full topology perimeter includes configured closed faces omitted from the open graph.
  different=nativezone[a]!=nativezone[b];perim=np.flatnonzero(different&((nativezone[a]>0)|(nativezone[b]>0)))
  sx=np.full(nf,-1,dtype='int32');sx[faces]=np.arange(len(faces));records=[]
  for face in perim:
   for cell,other in [(a[face],b[face]),(b[face],a[face])]:
    zi=int(nativezone[cell])
    if zi==0:continue
    typ='prescribed_water_exchange' if kcs[other]==2 else 'ground_exchange'
    si=int(sx[face]);status=('saved_receiver_flow_verified' if si>=0 else ('configured_closed_ground_no_saved_flow' if blocked[face] else 'ground_flow_not_saved'))
    records.append({'Case':case,'Zone':str(zid[zi-1]),'Native Face':int(face),'Other Zone Index':int(nativezone[other]),'Boundary Type':typ,'Configured Closed':bool(blocked[face]),'Section Index':si if si>=0 else None,'Section Name':names[si] if si>=0 else None,'Flow Evidence':status})
  boundary=pd.DataFrame(records);assert not boundary.duplicated(['Zone','Native Face']).any()
  old=pd.read_csv(OR/f'{case}_zone_boundary_faces.csv');newkeys=set(zip(boundary.Zone,boundary['Native Face']));assert set(zip(old.Zone,old['Native Face'])).issubset(newkeys)
  ground_open=boundary[(boundary['Boundary Type']=='ground_exchange')&~boundary['Configured Closed']]
  assert np.isin(ground_open['Native Face'],open_ground_ids).all()
  landzone=nativezone[land];sel=np.flatnonzero(landzone>0)
  S=coo_matrix((np.ones(len(sel)),(sel,landzone[sel]-1)),shape=(len(faces),nz)).tocsr()
  clock=pd.read_csv(folder/'udt_output_clock.csv',float_precision='round_trip');led=pd.read_csv(folder/'udt_step_budget.csv',float_precision='round_trip');hist=clock[clock.write_history=='T'];maps=clock[clock.write_map=='T'];hb=hist.step_index.to_numpy(dtype=int);dt=led.actual_step_dt.to_numpy();nt=len(hist)
  assert np.array_equal(hb,np.arange(1,len(led))) and np.array_equal(led.step_index,np.arange(len(led)))
  assert np.array_equal(hist.actual_state_time.to_numpy(),led.actual_state_time.to_numpy()[hb])
  assert np.max(abs(np.diff(led.actual_state_time)-dt[1:]))<1e-8
  assert np.array_equal(actual,maps.actual_state_time.to_numpy())
  out=np.zeros((nt,nz));inside=np.zeros_like(out);go=np.zeros(nt);gi=np.zeros(nt);maxclosed=0.;max_q=np.zeros(nz);min_q=np.zeros(nz)
  with Dataset(folder/'sfincs_his.nc') as d:
   d.set_auto_mask(False)
   assert str(d['crosssection_discharge'].units)=='m3 s-1'
   assert [str(v).strip() for v in chartostring(d['crosssection_name'][:])]==names
   assert np.array_equal(np.asarray(d['time'][:]),hist.nominal_output_time.to_numpy().astype('f4'))
   for j in range(0,nt,512):
    q=np.asarray(d['crosssection_discharge'][j:j+512],dtype='float32');assert np.isfinite(q).all() and (q>-99990).all()
    maxclosed=max(maxclosed,float(np.max(abs(q[:,blocked[faces]]),initial=0.)))
    # Same-step volume,not trapezoids on nominal history labels.
    v=(q*dt[hb[j:j+len(q)],None].astype('float32')).astype('float64')
    po=np.maximum(v,0);pi=np.maximum(-v,0)
    go[j:j+len(q)]=po.sum(axis=1);gi[j:j+len(q)]=pi.sum(axis=1)
    out[j:j+len(q)]=S.T.dot(po.T).T;inside[j:j+len(q)]=S.T.dot(pi.T).T
    net=S.T.dot(q.astype('float64').T).T;max_q=np.maximum(max_q,net.max(axis=0));min_q=np.minimum(min_q,net.min(axis=0))
  co=np.vstack([np.zeros(nz),np.cumsum(out,axis=0)]);ci=np.vstack([np.zeros(nz),np.cumsum(inside,axis=0)])
  gco=np.r_[0,np.cumsum(go)];gci=np.r_[0,np.cumsum(gi)]
  error=max(np.max(abs(gco-led.boundary_out_cumulative)),np.max(abs(gci-led.boundary_in_cumulative)))
  assert maxclosed==0 and error<1e-5,(case,maxclosed,error)
  ms=maps.step_index.to_numpy(dtype=int);cmo=co[ms];cmi=ci[ms]
  for zi,zone in enumerate(zid):
   bnd=boundary[boundary.Zone==zone];gr=bnd['Boundary Type'].eq('ground_exchange');recv=~gr
   allsums.append({'Case':case,'Zone':str(zone),'Boundary Memberships':len(bnd),'Ground Face Memberships':int(gr.sum()),'Configured Closed Ground Faces':int((gr&bnd['Configured Closed']).sum()),'Unobserved Open Ground Faces':int((gr&~bnd['Configured Closed']).sum()),'Receiving Face Memberships':int(recv.sum()),'Saved Receiving Outflow':float(cmo[-1,zi]-cmo[0,zi]),'Saved Receiving Inflow':float(cmi[-1,zi]-cmi[0,zi]),'Maximum Net Receiving Rate':float(max_q[zi]),'Minimum Net Receiving Rate':float(min_q[zi]),'Budget Complete':False,'Required Pump Capacity':None})
   for frame in range(len(ms)):
    alltraj.append({'Case':case,'Zone':str(zone),'Frame':frame,'Actual Time':float(actual[frame]),'Actual Step':int(ms[frame]),'Receiving Outflow Since First Map':float(cmo[frame,zi]-cmo[0,zi]),'Receiving Inflow Since First Map':float(cmi[frame,zi]-cmi[0,zi]),'Unknown Ground Exchange':None,'Unknown Applied Rainfall':None})
  allfaces.append(boundary)
  c={'case':case,'sections':len(faces),'unique_face_mapping_and_positive_land_to_water':True,'complete_actual_step_history':True,'max_closed_face_flow_m3_s':maxclosed,'max_global_prefix_budget_difference_m3':float(error),'zone_receiving_outflow_since_first_map_m3':float((cmo[-1]-cmo[0]).sum()),'zone_receiving_inflow_since_first_map_m3':float((cmi[-1]-cmi[0]).sum()),'new_boundary_memberships':len(boundary)-len(old),'prior_boundary_memberships_preserved':True}
  checks.append(c)
  for path in [folder/'sfincs.crs',folder/'sfincs_his.nc',folder/'udt_output_clock.csv',folder/'udt_step_budget.csv',OP/'zones'/f'{case}.npz',ROOT/row.audit_path]:bind(path)
  dump(R/'progress.json',{'cases_completed':len(checks),'cases_total':len(inv),'last_case':case,'elapsed_seconds':time.monotonic()-start})
  print('CASE_COMPLETE',case,'prefix_error_m3',error,'new_boundary_memberships',c['new_boundary_memberships'],flush=True)
 pd.concat(allfaces,ignore_index=True).to_csv(R/'boundary_flow_evidence.csv',index=False)
 pd.DataFrame(alltraj).to_csv(R/'receiving_exchange_trajectories.csv',index=False)
 pd.DataFrame(allsums).to_csv(R/'zone_flow_coverage.csv',index=False)
 assert all((ROOT/k).stat().st_size==v['bytes'] and (ROOT/k).stat().st_mtime_ns==v['mtime_ns'] for k,v in source.items())
 dump(R/'native_validation.json',{'passed':True,'cases':checks,'source_bindings':source,'source_stats_unchanged':True,'integration':'positive land-to-water;float32(q*actual_step_dt) applied-volume convention;actual map-step indexing;subtract first-map cumulative','local_budget_complete':False,'unknown_terms_not_filled':True,'new_hydraulic_runs':0,'required_capacity_computed':False,'elapsed_seconds':time.monotonic()-start})
 print('ALL_CASES_COMPLETE',len(checks),flush=True)
if __name__=='__main__':main()
