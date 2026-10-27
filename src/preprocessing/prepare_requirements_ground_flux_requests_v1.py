#!/usr/bin/env python3
"""Prepare inactive native-ground crosssection requests using existing solver API.
Does not edit a run folder,launch Docker,build an image,or change hydraulic inputs.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_zone_rain_flux_readiness_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
B=ROOT/'data/processed/three_ward_target_boundary_budget_audit_v1'
SRC=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src'
CTX=ROOT/'data/processed/three_ward_historical_radar_experiment_v2/context'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def location(p,needle):
 s=p.read_text().splitlines();hits=[i+1 for i,v in enumerate(s) if needle in v];assert hits;return {'file':str(p.relative_to(ROOT)),'lines':hits,'sha256':sha(p),'expression':needle}

def main():
 q=P/'requests';q.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 topo=ROOT/'data/processed/three_ward_negative_trace_interface_v1/whole_domain_metadata/whole_native_topology.bin'
 with topo.open('rb') as f:
  n,nf=map(int,np.fromfile(f,'<i4',2));sr=np.fromfile(f,'<i4',n)-1;cc=np.fromfile(f,'<i4',n)-1;kcs=np.fromfile(f,'<i4',n);f.seek(n*4*4,1);a=np.fromfile(f,'<i4',nf)-1;b=np.fromfile(f,'<i4',nf)-1
 pos=np.c_[-11360+(cc+.5)*10,-42440+(sr+.5)*10]
 boundary=pd.read_parquet(B/'zone_boundary_flow_evidence_preprocessed.parquet');ground=boundary[boundary['Boundary Type']=='ground_exchange'].copy()
 inventory=pd.read_csv(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv');requestrows=[];sums=[];incidence=[]
 for row in inventory.itertuples():
  case=row.case_id;bb=ground[ground.Case==case];faces=np.sort(bb['Native Face'].unique());assert (kcs[a[faces]]==1).all() and (kcs[b[faces]]==1).all()
  disp=pos[b[faces]]-pos[a[faces]];assert (np.linalg.norm(disp,axis=1)==10).all()
  lines=[];namebyface={};mid=(pos[a[faces]]+pos[b[faces]])/2
  for i,face in enumerate(faces):
   normal=disp[i]/10.;tangent=np.array([normal[1],-normal[0]]);p0=mid[i]-4.9*tangent;p1=mid[i]+4.9*tangent;name=f'G{int(face)+1:07d}';namebyface[int(face)]=name
   assert np.dot(np.array([-(p1-p0)[1],(p1-p0)[0]]),normal)>0
   lines.extend([name,'2 2',f'{p0[0]:.4f} {p0[1]:.4f}',f'{p1[0]:.4f} {p1[1]:.4f}'])
   closed=bool(bb.loc[bb['Native Face']==face,'Configured Closed'].iloc[0]);requestrows.append({'Case':case,'Section Name':name,'Native Face':int(face),'From Native Cell':int(a[face]),'To Native Cell':int(b[face]),'Face Width':10.,'Configured Closed':closed,'Runtime Face Mapping Verified':False,'Sign Convention':'positive from native endpoint A to B;zone sign in incidence table'})
  path=q/f'{case}_ground_faces.crs';path.write_text('\n'.join(lines)+'\n')
  with np.load(ROOT/'data/processed/three_ward_protection_requirements_register_v1/zones'/f'{case}.npz') as z:
   gz=np.zeros(1012*770,dtype=int);gz[z['node_grid_index']]=z['zone_index'];zids=z['zone_ids'];nz=gz[(1011-sr)*770+cc]
  for _,rec in bb.iterrows():
   face=int(rec['Native Face']);za=int(nz[a[face]]);zb=int(nz[b[face]]);target=str(rec.Zone)
   sign=1 if za>0 and str(zids[za-1])==target else -1
   assert (sign==1 and za>0 and str(zids[za-1])==target) or (sign==-1 and zb>0 and str(zids[zb-1])==target)
   incidence.append({'Case':case,'Zone':target,'Native Face':face,'Section Name':namebyface[face],'Positive Outward Sign':sign,'Configured Closed':bool(rec['Configured Closed'])})
  folder=(ROOT/row.map_path).parent;execution=json.loads((folder/'execution.json').read_text());runtime=float(execution['runtime_seconds']);steps=len(pd.read_csv(folder/'udt_step_budget.csv',usecols=['step_index']))-1
  closed=bb.drop_duplicates('Native Face')['Configured Closed'].sum()
  sums.append({'Case':case,'New Ground Sections':len(faces),'Open Ground Sections':int(len(faces)-closed),'Configured Closed Ground Sections':int(closed),'Original Receiving Sections':6763,'History Steps':steps,'Additional Raw Discharge Bytes':int(steps*len(faces)*4),'Original Runtime Seconds':runtime,'Proposed Wallclock Stop Seconds':2*runtime+600,'Request Path':str(path.relative_to(ROOT)),'Request SHA256':sha(path),'Execution Enabled':False})
 pd.DataFrame(requestrows).to_parquet(P/'ground_face_requests_preprocessed.parquet',index=False)
 inc=pd.DataFrame(incidence);inc.to_parquet(P/'ground_zone_incidence_preprocessed.parquet',index=False);s=pd.DataFrame(sums);s.to_parquet(P/'ground_request_budget_preprocessed.parquet',index=False)
 assert not inc.duplicated(['Case','Zone','Native Face']).any()
 shared=inc.groupby(['Case','Native Face'])['Positive Outward Sign'].agg(['count','sum']);assert shared.loc[shared['count']==2,'sum'].eq(0).all() and shared['count'].le(2).all()
 # Mandatory source evidence;existing standard API,not a newly implemented logger.
 evidence=[location(SRC/'sfincs_crosssections.f90','allocate(crs_uv_index(1000, ncrs))'),location(SRC/'sfincs_crosssections.f90','find_uv_points_intersected_by_polyline'),location(SRC/'sfincs_crosssections.f90','qq(icrs) = qq(icrs) + q(indx)*crs_idir(ip, icrs)*dxycrs'),location(SRC/'sfincs_ncoutput.f90','call get_discharges_through_crosssections(qq)'),location(SRC/'sfincs_ncoutput.f90','uz = 0.5*(uv(nmd1) + uv(nmu1))'),location(CTX/'sfincs_continuity.f90','call udt_limit_fluxes(dt,t)'),location(CTX/'sfincs_continuity.f90','call udt_budget_begin()'),location(CTX/'udt_clock_budget.f90','rain_term(nm)=real(dv,real64)'),location(CTX/'udt_clock_budget.f90','rain_increment(nm)=real(actual_dvol,real64)-horizontal_term(nm)'),location(CTX/'udt_clock_budget.f90','call negative_trace_finish(actual_time,dt)')]
 chosen=s.sort_values('Open Ground Sections',ascending=False).iloc[0]['Case'];paired=chosen.replace('__H__','__L__') if '__H__' in chosen else chosen.replace('__L__','__H__');batch=s[s.Case.isin([chosen,paired])]
 plan={'status':'prepared_not_executed','execution_enabled':False,'current_work_new_simulations':0,'new_solver_patch_needed_for_face_flow':False,'route':'Append one native-face segment per requested ground face to a copy of existing sfincs.crs;retain original B sections first;existing image and actual every-step history. Do not write into original folders.','source_evidence':evidence,'saved_output_limit':'Five-minute cell-centered u/v cannot recover positivity-limited same-step face q. Negative-event traces do not provide all faces at all steps. No ground discharge reconstructed from storage differences.','proposed_minimum_replay':{'cases':[str(chosen),str(paired)],'selection_rule':'Largest open-ground boundary coverage workload,then same-parent paired stage;diagnostic qualification rather than protection priority','execute_first_only':str(chosen),'then':'Only if original numerical gates and exact no-physics-change checks pass,replay paired stage. Stop after these two;no automatic ten-case expansion.','combined_wallclock_stop_seconds':float(batch['Proposed Wallclock Stop Seconds'].sum()),'additional_raw_q_bytes':int(batch['Additional Raw Discharge Bytes'].sum()),'budget_meaning':'Operational stop ceiling 2x each recorded baseline runtime+600s;not a predicted runtime. Raw q payload excludes compression,metadata,names and original outputs;confirm free-space before launch.'},'qualification_gates':['New run must use original physical inputs,forcing,roof assignment,stage,geometry,masks,image and numerical admission rules','Native mesh identity and actual step dt/clocks must match original;original109 map states must be bitwise equal including censor masks','Retain original B section names/order and verify their q records unchanged;new G names must map one-to-one to requested native face and sign','Negative volumes remain zero;original global rain and same-step budget gates unchanged;all configured closed requested faces have q=0 every step','Save applied same-step volumes at actual step times;preserve signed orientation and gross directions;shared internal faces cancel once','Quantify float32 face-product vs grouped-continuity rounding and map storage quantization separately;do not call the partial closure difference numerical error','If independent local applied-source/roundoff attribution is required,existing rain_term,horizontal_term,rain_increment are in-memory only:qualify a passive zone-reduction logger separately before claiming exact local residual closure'],'stop_on':['Any source/identity/clock/map difference','Any missing or multiply matched face,incorrect sign,nonfinite q or closed-face flux','Original numerical gate failure','Operational wallclock or available-disk ceiling'],'capacity_status':'Not ready;target stages,service window and effective unit/receiving allowance still unresolved;no pump count set'}
 dump(R/'ground_flux_interface_plan.json',plan)
 dump(R/'interface_validation.json',{'passed':True,'request_cases':len(s),'request_unique_case_faces':len(requestrows),'zone_face_memberships':len(inc),'shared_ground_faces_with_opposite_sign':int(shared['count'].eq(2).sum()),'one_segment_per_physical_face':True,'geometry_statically_checked':True,'runtime_mapping_verified':False,'source_evidence_checks':len(evidence),'original_run_folders_changed':False,'new_hydraulic_runs':0,'execution_enabled':False,'source_sha256':{str(topo.relative_to(ROOT)):sha(topo),str((B/'zone_boundary_flow_evidence_preprocessed.parquet').relative_to(ROOT)):sha(B/'zone_boundary_flow_evidence_preprocessed.parquet')}})
 print('GROUND_REQUESTS_COMPLETE',len(requestrows),'memberships',len(inc),'first',chosen,flush=True)
if __name__=='__main__':main()
