#!/usr/bin/env python3
"""Reference-only road groups and exact normal-network bypass diagnostics.
Uses existing results only;does not execute hydraulics,fit traffic or count people.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,time,collections
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from temporal_transport_common_v1 import Reachability
TAG='three_ward_protection_requirements_register_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
BASE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
M=ROOT/'data/processed/three_ward_motor_flood_access_v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 inputs=[M/'road_inventory.csv',BASE/'road_closure_episodes.csv',BASE/'clipped_road_lengths.npz',M/'road_projection.npz',ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/turn_router.npz',ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/multi_network_endpoints.csv',ROOT/'data/processed/three_ward_night_population_transport_v1/normal_node_access.npz']
 bindings={str(p.relative_to(ROOT)):sha(p) for p in inputs}
 inv=pd.read_csv(inputs[0]);ep=pd.read_csv(inputs[1]);ep=ep[(ep.bound=='lower')&(ep.arm=='flood')].copy();ep['Parent']=ep['case'].str.split('__').str[0]
 with np.load(inputs[2]) as z:q=z['qualified'];length=z['union_length_m'];ward=z['ward_length_m']
 assert inv.road_index.is_unique and np.array_equal(inv.road_index,np.arange(len(inv)))
 candidates=np.unique(ep.road_index);assert q[candidates].all() and len(candidates)==6064
 active=np.flatnonzero(q);parent=np.arange(len(inv));
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def join(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 # Group by genuine source-node adjacency AND name/way plus grade.
 # An unnamed road uses original way identity to avoid merging every local street.
 seen={}
 for row in inv.loc[active].itertuples(index=False):
  name='' if pd.isna(row.name) else str(row.name).strip()
  key=('name',name,str(row.grade_key)) if name else ('way',int(row.osm_way_id),str(row.grade_key))
  for node in [row.from_node,row.to_node]:
   k=(key,str(node))
   if k in seen:join(row.road_index,seen[k])
   else:seen[k]=row.road_index
 roots={find(int(i)) for i in candidates};members=collections.defaultdict(list)
 for i in active:
  root=find(int(i))
  if root in roots:members[root].append(int(i))
 groups=sorted(members.values(),key=lambda xs:min(xs));group_ids=[f'RG{i+1:05d}' for i in range(len(groups))]
 road_group=np.full(len(inv),-1,dtype='int32')
 for i,roads in enumerate(groups):road_group[roads]=i
 assert (road_group[candidates]>=0).all()
 engine=Reachability();normal=engine.direct(np.zeros(len(inv),bool))
 with np.load(inputs[-1]) as z:assert np.array_equal(normal,z['bidirectional'])
 pr=engine.pr;core_anchors=np.unique(np.r_[pr['from_index'][q[pr['arc_to_road']]],pr['to_index'][q[pr['arc_to_road']]]]);core_anchors=core_anchors[normal[core_anchors]]
 closed_union=np.zeros(len(inv),bool);closed_union[candidates]=True
 arterial={'motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link'}
 rows=[];memberrows=[];is_candidate=closed_union;case_parent_count=ep.groupby('road_index').Parent.nunique();group_case=[]
 for gi,roads in enumerate(groups):
  if (P/'STOP').exists():raise RuntimeError('User stop')
  assert time.monotonic()-start<1200,'Read-only road audit time budget'
  a=inv.loc[roads];c=np.asarray(roads)[is_candidate[roads]];closed=np.zeros(len(inv),bool);closed[c]=True
  ok=engine.direct(closed);lost=core_anchors[~ok[core_anchors]]
  incident=np.unique(np.r_[pr['from_index'][np.isin(pr['arc_to_road'],c)],pr['to_index'][np.isin(pr['arc_to_road'],c)]])
  outside=np.setdiff1d(lost,incident,assume_unique=True)
  back=bool(a.highway.isin(arterial).any())
  tier=('backbone_and_nonincident_dependency' if back and len(outside) else 'backbone_exposure_with_bypass' if back else 'nonincident_access_dependency' if len(outside) else 'incident_endpoint_dependency' if len(lost) else 'exposure_with_portal_bypass')
  name=next(iter(a['name'].dropna().astype(str)),f'Unnamed way {int(a.osm_way_id.iloc[0])}')
  gid=group_ids[gi]
  rows.append({'Road Group':gid,'Road Name':name,'Grade Key':str(a.grade_key.iloc[0]),'Source Roads':len(roads),'Exposure Candidate Roads':len(c),'Qualified Group Length':float(length[roads].sum()/1000),'Exposure Candidate Length':float(length[c].sum()/1000),'Chiyoda Length':float(ward[0,roads].sum()/1000),'Chuo Length':float(ward[1,roads].sum()/1000),'Minato Length':float(ward[2,roads].sum()/1000),'Backbone Class Candidate':back,'Normal Anchor Loss Under Group Probe':len(lost),'Nonincident Anchor Loss Under Group Probe':len(outside),'Priority Evidence':tier,'Service Closure Threshold':.30,'Stricter Slow Threshold':.10,'Protection Status':'proposed_group_not_frozen','Alternative Evidence':'exact portal reachability under hypothetical group-union closure;not a historical simultaneous state','Population Interpretation':'node counts only;no affected population computed'})
  for r in roads:memberrows.append({'Road Group':gid,'Road Index':r,'Source Edge':str(inv.source_edge_id.iloc[r]),'Road Name':None if pd.isna(inv['name'].iloc[r]) else str(inv['name'].iloc[r]),'Road Class':str(inv.highway.iloc[r]),'Grade Key':str(inv.grade_key.iloc[r]),'Candidate Closure':bool(is_candidate[r]),'Unique Clipped Length':float(length[r]/1000),'Parent Closure Count':int(case_parent_count.get(r,0))})
  if gi%25==0 or gi==len(groups)-1:
   dump(R/'progress.json',{'stage':'road_groups','completed':gi+1,'total':len(groups),'seconds':time.monotonic()-start});print('ROAD_GROUP',gi+1,'/',len(groups),flush=True)
 tab=pd.DataFrame(rows);mem=pd.DataFrame(memberrows)
 ep['Road Group']=ep.road_index.map(lambda r:group_ids[road_group[r]])
 for (gid,case),g in ep.groupby(['Road Group','case']):
  group_case.append({'Road Group':gid,'Case':case,'Candidate Roads With Closure':g.road_index.nunique(),'Observed Closure Episode Span':float((g.observed_span_min*g.three_ward_clipped_length_m/1000/60).sum()),'Right Censored Episodes':int(g.right_censored.sum()),'Interpretation':'sum of observed episode spans,not trapezoidal closure exposure or exact recovery'})
 tab.to_parquet(P/'road_target_groups_preprocessed.parquet',index=False);mem.to_parquet(P/'road_group_members_preprocessed.parquet',index=False);pd.DataFrame(group_case).to_parquet(P/'road_group_case_evidence_preprocessed.parquet',index=False)
 np.savez_compressed(P/'road_group_index.npz',road_group_index=road_group,group_ids=np.array(group_ids),candidate_road_indices=candidates,normal_core_anchor_nodes=core_anchors)
 # Road-to-cell controls:one minimum allowable level per shared cell,not one volume per road.
 nodecells={};cellgroups=collections.defaultdict(set)
 for r in candidates:
  refs=pr['point_refs'][pr['offsets'][r]:pr['offsets'][r+1]]
  for cell,z in zip(pr['point_coarse'][refs],pr['point_z'][refs]):
   cell=int(cell);target=float(z)+.30
   nodecells[cell]=min(nodecells.get(cell,float('inf')),target);cellgroups[cell].add(int(road_group[r]))
 controls=pd.DataFrame([{'Grid Index':k,'Target Water Level':v,'Linked Road Groups':len(cellgroups[k]),'Target Rule':'minimum sampled road level+0.30m;screening at closure threshold,not safety certification'} for k,v in sorted(nodecells.items())]);controls.to_parquet(P/'road_control_cells_preprocessed.parquet',index=False)
 links=[{'Grid Index':k,'Road Group':group_ids[g]} for k,gs in sorted(cellgroups.items()) for g in sorted(gs)];pd.DataFrame(links).to_parquet(P/'road_control_cell_group_links_preprocessed.parquet',index=False)
 # Small plain-array bridge permits NetCDF reads in the existing runtime without pyarrow installation.
 np.savez_compressed(P/'road_control_bridge.npz',grid_index=controls['Grid Index'].to_numpy(),target_level=controls['Target Water Level'].to_numpy(),link_grid=np.array([x['Grid Index'] for x in links]),link_group=np.array([group_ids.index(x['Road Group']) for x in links],dtype='int32'),group_ids=np.array(group_ids))
 assert mem['Candidate Closure'].sum()==6064 and np.isclose(mem.loc[mem['Candidate Closure'],'Unique Clipped Length'].sum(),216.08385000638467)
 assert mem['Road Index'].is_unique and controls['Grid Index'].is_unique
 assert all(sha(ROOT/k)==h for k,h in bindings.items())
 dump(R/'road_validation.json',{'passed':True,'groups':len(tab),'member_roads':len(mem),'candidate_roads':6064,'candidate_length_km':216.08385000638467,'priority_counts':tab['Priority Evidence'].value_counts().to_dict(),'normal_core_anchors':len(core_anchors),'normal_network_replay_exact':True,'probe_semantics':'Hypothetical normal-network removal of each group candidate union;not a new hydraulic state or actual historical closure','road_control_cells':len(controls),'source_sha256':bindings,'source_unchanged':True,'people_counted':False,'new_hydraulics':0,'seconds':time.monotonic()-start})
 print('ROAD_REGISTERS_COMPLETE',len(tab),len(mem),len(controls),flush=True)
if __name__=='__main__':main()
