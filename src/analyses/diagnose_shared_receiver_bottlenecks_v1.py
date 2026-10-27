#!/usr/bin/env python3
"""Read-only diagnosis of six completed treatments. No solver execution or retuning."""
from pathlib import Path
import io, json, hashlib, html, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
TAG='three_ward_measure_bottleneck_diagnosis_v1'
P=ROOT/'data/processed'/TAG; R=ROOT/'data/exp/estimation'/TAG
D=ROOT/'data/processed'
C=D/'three_ward_intake_receiver_candidates_v1'
S=D/'three_ward_protection_service_demand_plan_v1'
G=D/'three_ward_protection_requirements_register_v1'
A=D/'three_ward_shared_receiver_allocation_v1'
WARD={1:'Chiyoda',2:'Chuo',3:'Minato'}
BIND={}; VALID={}; OUTPUTS=[]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p); BIND[str(p.relative_to(ROOT))]=sha(p); return p
def csv(p):return pd.read_csv(bind(p))
def parq(p):return pd.read_parquet(bind(p))
def js(p):return json.loads(bind(p).read_text())
def npz(p):return np.load(bind(p))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(name,df):
 f=P/(name+'.csv');df.to_csv(f,index=False,float_format='%.12g');OUTPUTS.append(f)
 return df
def integ(y,t):return float(np.trapezoid(np.asarray(y,dtype=float),np.asarray(t,dtype=float)/3600))
def tab(df):return '<div class="scroll">'+df.to_html(index=False,float_format=lambda x:f'{x:,.3f}',escape=True)+'</div>'

class HashReader(io.RawIOBase):
 def __init__(self,p):self.f=Path(p).open('rb');self.h=hashlib.sha256()
 def readable(self):return True
 def readinto(self,b):
  n=self.f.readinto(b)
  if n:self.h.update(memoryview(b)[:n])
  return n
 def close(self):self.f.close();super().close()

def trace(run,hp,hr,sites,target):
 case=run['case_id'];fp=ROOT/run['folder']/'udt_pump_steps.csv';before=fp.stat();n=len(sites)
 names=['requested_m3','actual_m3','trigger_only_m3','head_only_m3','both_blocked_m3','eligible_storage_shortfall_m3','site_seconds','active_site_seconds','head_blocked_site_seconds','trigger_blocked_site_seconds','local_excess_site_seconds','transfer_above_local_target_m3','transfer_below_local_target_m3']
 agg={k:np.zeros(n) for k in names}; pool_upper={k:0. for k in sites['Receiving Pool'].unique()}
 poolmask={k:sites['Receiving Pool'].eq(k).to_numpy() for k in pool_upper}
 pump_file=bind(ROOT/run['folder']/'udt_pumps.dat')
 spec=np.loadtxt(pump_file,skiprows=1,ndmin=2)
 assert np.array_equal(spec[:,0],sites['Source Native Cell One Based']) and np.array_equal(spec[:,1],sites['Receiver Native Cell One Based'])
 assert np.allclose(spec[:,2],sites['Reserved Rate'],rtol=0,atol=1e-15) and np.all(spec[:,3]==.05) and np.all(spec[:,4]==5.)
 records=0;steps=0;max_qerr=0.;lasttime=None;firsttime=None
 reader=HashReader(fp)
 cols=['actual_end_time_s','dt_s','pump_id','source_native_cell','receiver_native_cell','receiver_kcs','requested_m3','transferred_m3','source_prior_volume_m3','head_m','depth_m','limited_by_storage','blocked_by_trigger_or_head']
 with io.BufferedReader(reader,buffer_size=1024*1024) as stream:
  for chunk in pd.read_csv(stream,usecols=cols,chunksize=n*5000):
   assert len(chunk)%n==0
   def arr(k):return chunk[k].to_numpy().reshape(-1,n)
   ids=arr('pump_id');assert (ids==np.arange(1,n+1)).all()
   assert (arr('source_native_cell')==spec[None,:,0]).all() and (arr('receiver_native_cell')==spec[None,:,1]).all() and (arr('receiver_kcs')==2).all()
   dt=arr('dt_s');tm=arr('actual_end_time_s');assert (dt>0).all() and np.max(np.ptp(dt,axis=1))==0 and np.max(np.ptp(tm,axis=1))==0
   times=tm[:,0]
   if lasttime is not None:assert abs(times[0]-lasttime-dt[0,0])<1e-7
   assert np.max(np.abs(np.diff(times)-dt[1:,0]),initial=0)<1e-7
   firsttime=times[0] if firsttime is None else firsttime;lasttime=times[-1]
   req=arr('requested_m3');v=arr('transferred_m3');prior=arr('source_prior_volume_m3');head=arr('head_m');depth=arr('depth_m')
   t=depth<.05;h=head>5.;eligible=~(t|h)
   assert (arr('blocked_by_trigger_or_head')=='T').__eq__(t|h).all()
   assert ((arr('limited_by_storage')=='T')==((v<req)&eligible)).all()
   assert (v>=0).all() and (v<=req+1e-12).all() and (v<=prior+1e-10).all()
   max_qerr=max(max_qerr,float(np.max(np.abs(req-sites['Reserved Rate'].to_numpy()[None,:]*dt))))
   shortage=np.where(eligible,req-v,0.)
   assert np.max(np.abs(req-v-np.where(t&~h,req,0)-np.where(h&~t,req,0)-np.where(h&t,req,0)-shortage))<1e-10
   excess=np.maximum(prior-target[None,:],0);above=np.minimum(v,excess)
   vals=[req,v,np.where(t&~h,req,0),np.where(h&~t,req,0),np.where(h&t,req,0),shortage,dt,np.where(v>0,dt,0),np.where(h,dt,0),np.where(t,dt,0),np.where(excess>1e-6,dt,0),above,v-above]
   for k,vv in zip(names,vals):agg[k]+=vv.sum(axis=0)
   # Conditional instantaneous envelope on already-computed treatment states.
   # Its integral is NOT a replay prediction or achievable extra extraction.
   feasible=np.where(eligible,np.minimum(.5*dt,np.maximum(prior,0)/(1+64*np.finfo(np.float32).eps)),0)
   for pool,m in poolmask.items():
    possible=np.minimum(.5*dt[:,0],feasible[:,m].sum(axis=1));actual=v[:,m].sum(axis=1)
    assert (actual<=possible+1e-7).all();pool_upper[pool]+=float(possible.sum())
   records+=len(chunk);steps+=len(times)
  while stream.read(1024*1024):pass
  digest=reader.h.hexdigest()
 after=fp.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
 BIND[str(fp.relative_to(ROOT))]=digest
 audit=js(hr/'execution'/case/'audit.json');bud=js(hr/'execution'/case/'pump_budget_summary.json')
 assert audit['all_checks_passed'] and all(audit['shared_receiver_checks'].values()) and bud['passed']
 assert records==bud['pump_records'] and steps==bud['actual_steps'] and abs(sum(agg['actual_m3'])-bud['actual_transferred_m3'])<1e-5
 assert max_qerr<1e-10
 out=sites[['Pump ID','Grid Index','Receiving Pool','Ward Code','Reserved Rate']].copy()
 for k,v in agg.items():out[k]=v
 out.insert(0,'case',case);out.insert(0,'stage',run['stage']);out.insert(0,'event',run['event_id'])
 assert np.max(abs(out.requested_m3-out[['actual_m3','trigger_only_m3','head_only_m3','both_blocked_m3','eligible_storage_shortfall_m3']].sum(axis=1)))<1e-5
 out['utilization_percent']=100*out.actual_m3/out.requested_m3
 pools=out.groupby(['event','stage','case','Receiving Pool'],as_index=False)[['Reserved Rate']+names].sum()
 pools['conditional_fixed_state_envelope_m3']=pools['Receiving Pool'].map(pool_upper)
 pools['utilization_percent']=100*pools.actual_m3/pools.requested_m3
 pools['active_site_time_percent']=100*pools.active_site_seconds/pools.site_seconds
 wards=out.groupby(['event','stage','case','Ward Code'],as_index=False)[['Reserved Rate']+names].sum();wards['Ward']=wards['Ward Code'].map(WARD)
 wards['utilization_percent']=100*wards.actual_m3/wards.requested_m3
 VALID[case]={'records':records,'steps':steps,'actual_transferred_m3':float(out.actual_m3.sum()),'max_rate_serialization_error_m3':max_qerr,'cause_ledger_closes':True,'audit_passed':True,'trace_sha256':digest,'end_s':float(lasttime),'first_end_s':float(firsttime)}
 print('TRACE_DONE',case,round(out.actual_m3.sum(),2),flush=True)
 return out,pools,wards

def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 sites=csv(D/'three_ward_shared_receiver_pilot_v1/pump_sites.csv').sort_values('Pump ID')
 assert len(sites)==62 and sites['Source Native Cell One Based'].nunique()==62
 assert np.allclose(sites.groupby('Receiving Pool')['Reserved Rate'].sum(),.5,rtol=0,atol=1e-12)
 z=npz(C/'candidate_index.npz');grid=z['source_grid'];target=z['target_storage'];z.close()
 si=np.searchsorted(grid,sites['Grid Index'].to_numpy());assert np.array_equal(grid[si],sites['Grid Index'])
 memberships=parq(S/'road_policy_membership_preprocessed.parquet');members=parq(G/'road_group_members_preprocessed.parquet');links=parq(G/'road_control_cell_group_links_preprocessed.parquet')
 sources=parq(C/'source_candidates_preprocessed.parquet');assert members['Road Index'].is_unique
 lengths=npz(D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');weight=lengths['union_length_m']/1000;lengths.close()
 group_to_priority=memberships.set_index('Road Group')['Priority Evidence'].to_dict()
 selected_groups=set(links.loc[links['Grid Index'].isin(sites['Grid Index']),'Road Group'])
 membership=memberships.copy();membership['Selected Intake Cell Linked']=membership['Road Group'].isin(selected_groups)
 cov=membership.groupby('Priority Evidence',as_index=False).agg(groups=('Road Group','size'),groups_linked_to_selected_cell=('Selected Intake Cell Linked','sum'),candidate_length_km=('Exposure Candidate Length','sum'))
 save('road_priority_spatial_coverage',cov)
 # Geometric co-location is not an influence radius or causal coverage measure.
 save('ward_static_reservations',sites.groupby('Ward Code',as_index=False).agg(native_intakes=('Pump ID','size'),reserved_m3_s=('Reserved Rate','sum')).assign(Ward=lambda x:x['Ward Code'].map(WARD)))
 allsites=[];allpools=[];allwards=[];roads=[];pops=[];off=[];priority=[];groups=[];anchors=[];runtimes=[];support0=None;qi0=None
 for event in ['E20240731','E20250911','E20240830']:
  if event=='E20240731':
   hp=D/'three_ward_shared_receiver_pilot_v1';hr=ROOT/'data/exp/estimation/three_ward_shared_receiver_pilot_v1';ap=D/'three_ward_shared_receiver_access_v1'
  else:
   hp=D/'three_ward_shared_receiver_transfer_v1/batches'/event;hr=ROOT/'data/exp/estimation/three_ward_shared_receiver_transfer_v1/batches'/event;ap=hp/'access'
  mask=npz(hp/'common_road_support.npz');support=mask['qualified_roads'];admitted=mask['common_admitted'];mask.close()
  if support0 is None:support0=support.copy();admit0=admitted.copy()
  else:assert np.array_equal(support0,support) and np.array_equal(admit0,admitted)
  rr=csv(hp/'paired_road_effects.csv');rr.insert(0,'event',event);roads.append(rr)
  pp=csv(ap/'paired_population_summary.csv');pp.insert(0,'event',event);pops.append(pp)
  oo=csv(hp/'off_target_summary.csv');oo.insert(0,'event',event);off.append(oo)
  aa=csv(ap/'anchor_access_summary.csv');aa.insert(0,'event',event);anchors.append(aa)
  # Independently re-integrate original trajectories; retain paired bounds exactly.
  rt=csv(hp/'road_trajectories.csv');pt=csv(ap/'paired_population_trajectories.csv')
  for row in rr.to_dict('records'):
   f=rt[(rt.Case==row['Case'])&(rt['Depth Branch']==row['Depth Branch'])&(rt.Scope==row['Scope'])];b=f[f.Arm=='baseline'].sort_values('Frame');t=f[f.Arm=='drainage'].sort_values('Frame');tt=b['Analysis Time']
   assert np.array_equal(b.Frame,t.Frame)
   v=integ(b['Closed Road Length'].to_numpy()-t['Closed Road Length'].to_numpy(),tt)
   assert abs(v-row['Closure Reduction km h'])<1e-8
  for row in pp.to_dict('records'):
   f=pt[(pt.case==row['case'])&(pt.bound==row['bound'])&(pt.buffer_m==row['buffer_m'])&(pt.scope==row['scope'])].sort_values('frame')
   assert (f.net_avoided_loss_lower<=f.net_avoided_loss_upper).all()
   for side in ['lower','upper']:assert abs(integ(f['net_avoided_loss_'+side],f.elapsed_s)-row['net_avoided_loss_'+side+'_resident_hours'])<1e-6
  runs=js(hp/'run_manifest.json')['runs']
  for run in runs:
   case=run['case_id'];s,p,w=trace(run,hp,hr,sites,target[si]);allsites.append(s);allpools.append(p);allwards.append(w)
   execution=js(ROOT/run['folder']/'execution.json');runtimes.append({'case':case,'runtime_seconds':execution['runtime_seconds']})
   with npz(hp/'exposure'/(case+'__baseline.npz')) as z:b=z['road_state_codes'];qi=z['road_indices'];times=z['analysis_times']
   with npz(hp/'exposure'/(case+'__drainage.npz')) as z:t=z['road_state_codes'];assert np.array_equal(qi,z['road_indices']) and np.array_equal(times,z['analysis_times'])
   assert b.shape==t.shape==(109,2,len(qi)) and len(np.unique(qi))==len(qi)
   if qi0 is None:qi0=qi.copy()
   else:assert np.array_equal(qi0,qi)
   assert support[qi].all() and support.sum()==len(qi)
   idx=np.full(len(support),-1,int);idx[qi]=np.arange(len(qi))
   m=members[members['Road Index'].isin(qi)].copy();m['ix']=idx[m['Road Index']];m['Priority Evidence']=m['Road Group'].map(group_to_priority)
   # Each road belongs to one group; include every other qualified road explicitly.
   bins=np.full(len(qi),'outside_existing_group_inventory',object)
   bins[m['ix'].to_numpy()]=m['Priority Evidence'].to_numpy()
   for branch_i,branch in enumerate(['lower','upper']):
    closedb=b[:,branch_i]==2;closedt=t[:,branch_i]==2
    restored=closedb&~closedt;new=~closedb&closedt
    total=0.
    for name in np.unique(bins):
     ii=np.flatnonzero(bins==name);ww=weight[qi[ii]];bb=closedb[:,ii]@ww;dd=closedt[:,ii]@ww;red=integ(bb-dd,times);total+=red
     priority.append({'event':event,'stage':run['stage'],'branch':branch,'priority_evidence':name,'qualified_km':float(ww.sum()),'baseline_km_h':integ(bb,times),'net_reduction_km_h':red,'restored_km_h':integ(restored[:,ii]@ww,times),'newly_closed_km_h':integ(new[:,ii]@ww,times)})
    ref=rr[(rr.Case==case)&(rr['Depth Branch']==branch)&(rr.Scope=='Three wards')].iloc[0];assert abs(total-ref['Closure Reduction km h'])<1e-8
    for group,sub in m.groupby('Road Group'):
     ii=sub['ix'].to_numpy();ww=weight[qi[ii]];cb=closedb[:,ii].any(axis=1);ct=closedt[:,ii].any(axis=1)
     full_members=members[members['Road Group'].eq(group)]
     groups.append({'event':event,'stage':run['stage'],'branch':branch,'road_group':group,'priority_evidence':group_to_priority[group],
      'selected_cell_linked':group in selected_groups,'qualified_member_km':float(ww.sum()),'all_registered_members_qualified':len(sub)==len(full_members),
      'baseline_known_member_closed_hours':integ(cb,times),'drainage_known_member_closed_hours':integ(ct,times),
      'known_group_closure_reduction_hours':integ(cb.astype(int)-ct.astype(int),times),
      'restored_edge_km_h':integ(restored[:,ii]@ww,times),'baseline_peak_known_closed_km':float(np.max(closedb[:,ii]@ww))})
  print('EVENT_ANALYZED',event,flush=True)
 rd=save('road_effects_whole_window',pd.concat(roads,ignore_index=True));pop=save('population_effects_whole_window',pd.concat(pops,ignore_index=True));of=save('off_target_summary',pd.concat(off,ignore_index=True));anchor=save('anchor_access_summary',pd.concat(anchors,ignore_index=True))
 site=save('pump_site_operating_budget',pd.concat(allsites,ignore_index=True));pool=save('pump_pool_operating_budget',pd.concat(allpools,ignore_index=True));ward=save('pump_ward_operating_budget',pd.concat(allwards,ignore_index=True));pr=save('road_priority_effects',pd.DataFrame(priority));gg=save('road_group_continuity_proxy',pd.DataFrame(groups))
 ref=parq(A/'case_allocation_summary_preprocessed.parquet');ref=ref[ref.Portfolio.eq('R1000_T30_Finite_0p5')];save('july_reference_request_summary',ref)
 tr=parq(A/'snapshot_allocation_trajectories_preprocessed.parquet');tr=tr[tr.Portfolio.eq('R1000_T30_Finite_0p5')];save('july_reference_request_trajectories',tr)
 # A transparent whole-window operating ledger, not a causal decomposition of benefits.
 budget=site.groupby(['event','stage','case'],as_index=False)[['Reserved Rate','requested_m3','actual_m3','trigger_only_m3','head_only_m3','both_blocked_m3','eligible_storage_shortfall_m3','transfer_above_local_target_m3','transfer_below_local_target_m3']].sum()
 budget['utilization_percent']=100*budget.actual_m3/budget.requested_m3
 save('whole_window_operating_budget',budget);save('observed_solver_runtimes',pd.DataFrame(runtimes))
 dump(P/'input_bindings.json',{'sha256':BIND,'script_sha256':sha(Path(__file__)),'trace_hashing':'one streaming pass;size and mtime stable across read','new_simulations':0})
 dump(R/'validation.json',{'passed':True,'events':3,'treated_cases':6,'same_qualified_road_mask_and_admitted_grid':True,'same_road_order':True,'common_road_km':float(weight[support0].sum()),'independent_full_window_integrals_match':True,'paired_population_bounds_retained':True,'all_road_priority_bins_sum_to_regional_effect':True,'all_original_audits_passed':True,'trace_operating_ledgers_close':True,'run_checks':VALID,'new_hydraulic_runs':0,'training':0,'sealed_test_access':False,'AnaSOP_changed':False})
 dump(P/'outputs.json',{'files':[{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in OUTPUTS]})
 print('ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
