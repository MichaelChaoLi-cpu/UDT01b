#!/usr/bin/env python3
"""Aggregate qualified frozen-control adverse changes; no solver, tuning, or new routing."""
from pathlib import Path
import sys,os,json,hashlib,time,functools,html
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'.frame-control/runtimes/physical-measures-py312/lib/python3.12/site-packages'))
sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
import numpy as np,pandas as pd
from netCDF4 import Dataset
from shapely.geometry import shape,Point
from shapely.strtree import STRtree
from pyproj import Transformer
from temporal_transport_common_v1 import population_inputs,counts
from run_temporal_road_population_v1 import envelope,signed_envelope
from shared_receiver_pilot_common_v1 import dump,sha
from summarize_three_event_controls_v1 import contexts
TAG='three_ward_local_adverse_diagnosis_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
D=ROOT/'data/processed';TEMP=D/'three_ward_five_event_temporal_analysis_v1'
BIND={};STATS={};COMPARE=[]

def bind(path):
 path=Path(path);key=str(path.relative_to(ROOT))
 if key not in BIND:
  BIND[key]=sha(path);st=path.stat();STATS[key]=(st.st_size,st.st_mtime_ns)
 return path

def integrate(values,ts):return np.trapezoid(values.astype(float),ts,axis=0)/3600

def make_context(event,base,br,static,suffix):
 assert json.loads(bind(br/'final_summary_validation.json').read_text())['passed']
 batches={a:base/'batches'/a for a in ['O1','S1']};batches['Static']=static
 access={a:batches[a]/'access' for a in ['O1','S1']};access['Static']=(D/'three_ward_shared_receiver_access_v1')if event=='E20240731'else static/'access';access['No action']=access['O1']
 entries={}
 for stage in ['H','L']:
  for scheme in ['O1','S1','Static']:
   case=f'{event}__{stage}__U'+(suffix if scheme=='Static'else '__'+scheme.lower());bp=batches[scheme]
   with np.load(bind(bp/'exposure'/(case+'__drainage.npz')))as z:depths=np.stack([z['lower'],z['upper']],axis=1);codes=z['road_state_codes'];qi=z['road_indices'];ts=z['analysis_times']
   with np.load(bind(access[scheme]/'population'/(case+'.npz')))as z:keys=z['node_cache_keys'];ids=z['block_ids'];assert np.array_equal(ts,z['analysis_times'])
   entries[stage,scheme]=dict(case=case,bp=bp,access=access[scheme],depths=depths,codes=codes,qi=qi,ts=ts,keys=keys[:,:,1],ids=ids)
   if scheme=='O1':
    with np.load(bind(bp/'exposure'/(case+'__baseline.npz')))as z:dd=np.stack([z['lower'],z['upper']],axis=1);cc=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])and np.array_equal(ts,z['analysis_times'])
    entries[stage,'No action']=dict(case=f'{event}__{stage}__U',bp=bp,access=access['O1'],depths=dd,codes=cc,qi=qi,ts=ts,keys=keys[:,:,0],ids=ids)
 return entries

def population_pair(event,stage,reference,treatment,a,b,blocks,stock,normal,infos):
 assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['ids'],blocks.block_id.to_numpy(dtype=str))
 ts=a['ts'];assert np.array_equal(ts,b['ts']);dt=np.diff(ts)/3600;w=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
 @functools.lru_cache(maxsize=120)
 def node(folder,key):
  f=Path(folder)/'node_cache'/(key+'.npz')
  if not f.exists():f=TEMP/'reachability_cache'/(key+'.npz')
  with np.load(bind(f))as z:
   assert hashlib.sha256(z['closed_packed'].tobytes()).hexdigest()==key
   return np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool)
 rows=[];frame_rows=[]
 for bi,bound in enumerate(['lower','upper']):
  accum={buf:np.zeros((6,len(stock)),dtype='f8')for buf in infos};loss_time={buf:np.zeros((2,len(stock)),dtype='f8')for buf in infos};first={buf:np.full(len(stock),np.nan)for buf in infos};last={buf:np.full(len(stock),np.nan)for buf in infos}
  for ti in range(109):
   x=node(str(a['access']),str(a['keys'][ti,bi]));y=node(str(b['access']),str(b['keys'][ti,bi]));delta=x.astype('i1')-y.astype('i1');delta[~normal]=0
   for buf,info in infos.items():
    lo,hi=signed_envelope(delta,info,stock);ra,rb=envelope(normal&~x&y,info,stock);na,nb=envelope(normal&x&~y,info,stock)
    values=np.stack([-hi,-lo,stock*ra,stock*rb,stock*na,stock*nb]);accum[buf]+=values*w[ti];loss_time[buf]+=np.stack([na,nb])*w[ti]
    use=nb&(stock>0);first[buf][use&~np.isfinite(first[buf])]=ts[ti];last[buf][use]=ts[ti]
    frame_rows.append(dict(event=event,stage=stage,reference=reference,treatment=treatment,bound=bound,buffer_m=buf,frame=ti,elapsed_s=float(ts[ti]),**{k:float(v.sum())for k,v in zip(['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper'],values)}))
  for buf,v in accum.items():
   assert (v[0]<=v[1]+1e-7).all()
   for j in np.flatnonzero(np.max(np.abs(v),axis=0)>1e-10):
    row=dict(event=event,stage=stage,reference=reference,treatment=treatment,bound=bound,buffer_m=buf,block_id=str(blocks.block_id.iloc[j]),town_id=str(blocks.town_id.iloc[j]),ward=str(blocks.ward.iloc[j]),name=str(blocks['name'].iloc[j]),official_residents=int(stock[j]),normal_support_complete=bool(infos[buf]['complete'][j]),first_sampled_possible_loss_s=None if not np.isfinite(first[buf][j])else float(first[buf][j]),last_sampled_possible_loss_s=None if not np.isfinite(last[buf][j])else float(last[buf][j]),sampled_all_candidate_new_loss_h=float(loss_time[buf][0,j]),sampled_any_candidate_new_loss_h=float(loss_time[buf][1,j]))
    row.update({k+'_resident_hours':float(v[i,j])for i,k in enumerate(['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper'])});rows.append(row)
 return rows,frame_rows

def road_pair(event,stage,reference,treatment,a,b,clip,inventory,membership,policy):
 qi=a['qi'];ts=a['ts'];assert np.array_equal(qi,b['qi'])and np.array_equal(ts,b['ts']);ww=clip['union_length_m'][qi]/1000
 rows=[];groups=[];summary=[]
 for bi,bound in enumerate(['lower','upper']):
  x=a['depths'][:,bi];y=b['depths'][:,bi];added=(x<.3)&(y>=.3);restored=(x>=.3)&(y<.3);ad=integrate(added,ts);re=integrate(restored,ts);delta=y-x
  peak=float((added@ww).max());up=integrate(added&(delta>=.1),ts);tiny=integrate(added&(delta<=.01),ts)
  summary.append(dict(event=event,stage=stage,reference=reference,treatment=treatment,bound=bound,new_closed_km_h=float(ad@ww),restored_km_h=float(re@ww),net_avoided_km_h=float((re-ad)@ww),peak_simultaneous_new_closed_km=peak,new_road_closure_with_depth_increase_ge_0p1_km_h=float(up@ww),new_road_closure_with_depth_increase_le_0p01_km_h=float(tiny@ww),ever_new_closed_unique_length_km=float(ww[added.any(0)].sum())))
  for j in np.flatnonzero(added.any(0)):
   rid=int(qi[j]);frames=np.flatnonzero(added[:,j]);other=(a['depths'][:,1-bi,j]<.3)&(b['depths'][:,1-bi,j]>=.3)
   rows.append(dict(event=event,stage=stage,reference=reference,treatment=treatment,bound=bound,road_index=rid,source_edge_id=inventory.source_edge_id.iloc[rid],road_name=inventory['name'].iloc[rid],road_group=membership.get(rid,'unregistered'),length_km=float(ww[j]),new_closed_km_h=float(ad[j]*ww[j]),new_closed_duration_h=float(ad[j]),first_sampled_new_closed_s=float(ts[frames[0]]),last_sampled_new_closed_s=float(ts[frames[-1]]),baseline_depth_min_at_new_closure_m=float(x[frames,j].min()),baseline_depth_max_at_new_closure_m=float(x[frames,j].max()),treatment_depth_max_at_new_closure_m=float(y[frames,j].max()),max_depth_increase_at_new_closure_m=float(delta[frames,j].max()),min_depth_increase_at_new_closure_m=float(delta[frames,j].min()),both_depth_branches_new_closure_frames=int(np.count_nonzero(other&added[:,j])),small_shift_new_closure_km_h=float(tiny[j]*ww[j]),large_shift_new_closure_km_h=float(up[j]*ww[j])))
  index={int(r):j for j,r in enumerate(qi)}
  for group,g in policy.groupby('Road Group'):
   ids=members_global[members_global['Road Group'].eq(group)]['Road Index'].to_numpy(dtype=int);ii=np.array([index[k]for k in ids if k in index],dtype=int)
   if not len(ii):continue
   gain=float(re[ii]@ww[ii]);loss=float(ad[ii]@ww[ii])
   if gain+loss<1e-12:continue
   groups.append(dict(event=event,stage=stage,reference=reference,treatment=treatment,bound=bound,road_group=group,road_name=str(g['Road Name'].iloc[0]),priority_evidence=str(g['Priority Evidence'].iloc[0]),all_registered_members_qualified=len(ii)==len(ids),restored_km_h=gain,new_closed_km_h=loss,net_avoided_km_h=gain-loss,any_new_closure_sampled_h=float(integrate(added[:,ii].any(axis=1),ts))))
  assert abs(sum(v['new_closed_km_h']for v in groups if v['bound']==bound)-float(ad@ww))<1e-8,'Road partition mismatch'
 return rows,groups,summary

def native_adverse(cases,ward,affine):
 js=json.loads(bind(D/'three_ward_population_block_stability_v1/town_population_stability_6677.geojson').read_text());polys=[shape(f['geometry'])for f in js['features']];tree=STRtree(polys);props=[f['properties']for f in js['features']];trans=Transformer.from_crs(6677,4326,always_xy=True);rows=[]
 for event,base,br,static,suffix in cases:
  for arm in ['O1','S1']:
   bp=base/'batches'/arm;df=pd.read_csv(bind(bp/'off_target_head_changes.csv'));df=df[df.Scope.eq('Three wards')&df['All Supported Certain Head Rise Area'].gt(0)]
   if df.empty:continue
   with np.load(bind(bp/'common_road_support.npz'))as z:common=z['common_admitted'].ravel()&(ward>0)
   manifest=json.loads(bind(bp/'run_manifest.json').read_text())['runs']
   for run in manifest:
    sel=df[df.Case.eq(run['case_id'])]
    if sel.empty:continue
    before=bind(ROOT/run['original_folder']/'sfincs_map.nc');after=bind(ROOT/run['folder']/'sfincs_map.nc')
    with Dataset(before)as a,Dataset(after)as b:
     bed0=np.flipud(np.ma.asarray(a['zb'][:],dtype=float).filled(np.nan)).ravel();bed1=np.flipud(np.ma.asarray(b['zb'][:],dtype=float).filled(np.nan)).ravel()
     for q in sel.to_dict('records'):
      ti=int(q['Frame']);x=np.flipud(np.ma.asarray(a['zs'][ti],dtype=float).filled(np.nan)).ravel();y=np.flipud(np.ma.asarray(b['zs'][ti],dtype=float).filled(np.nan)).ravel();rise=np.where(np.isfinite(y),y,bed1)-np.where(np.isfinite(x),x,bed0+.01);ii=np.flatnonzero(common&(rise>=.1));assert abs(len(ii)*.0001-q['All Supported Certain Head Rise Area'])<1e-9
      for ix in ii:
       row,col=divmod(int(ix),770);px=affine[2]+(col+.5)*affine[0];py=affine[5]+(row+.5)*affine[4];hits=tree.query(Point(px,py),predicate='within');hit=props[int(hits[0])]if len(hits)==1 else {};lon,lat=trans.transform(px,py)
       rows.append(dict(event=event,stage=run['stage'],treatment=arm,reference='No action',frame=ti,elapsed_s=float(q['Analysis Time']),grid_index=int(ix),ward_code=int(ward[ix]),town_id=hit.get('town_id','unresolved'),town_name=hit.get('name','boundary_or_unresolved'),town_match_count=len(hits),longitude=lon,latitude=lat,certain_head_rise_m=float(rise[ix]),cell_area_m2=100))
    print('NATIVE_SCREEN',event,arm,run['stage'],len(sel),flush=True)
 return rows

def main():
 global members_global
 started=time.monotonic();assert not (R/'validation.json').exists(),'Preserve completed diagnosis'
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 cases=contexts(D/'three_ward_shared_receiver_controls_august_v1')
 blocks,stock,normal,infos=population_inputs();assert len(blocks)==4746 and stock.sum()==496345
 source=D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz';clip=dict(np.load(bind(source)))
 inventory=pd.read_csv(bind(D/'three_ward_motor_flood_access_v1/road_inventory.csv'));assert np.array_equal(inventory.road_index.to_numpy(),np.arange(len(inventory)))
 members_global=pd.read_parquet(bind(D/'three_ward_protection_requirements_register_v1/road_group_members_preprocessed.parquet'));assert not members_global['Road Index'].duplicated().any();membership=members_global.set_index('Road Index')['Road Group'].to_dict()
 policy=pd.read_parquet(bind(D/'three_ward_protection_service_demand_plan_v1/road_policy_membership_preprocessed.parquet'))
 for f in [D/'three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv',D/'three_ward_night_population_transport_v1/block_access_support.npz',D/'three_ward_night_population_transport_v1/normal_node_access.npz']:bind(f)
 roadrows=[];grouprows=[];summaries=[];poprows=[];popframes=[]
 for event,base,br,static,suffix in cases:
  entries=make_context(event,base,br,static,suffix)
  for stage in ['H','L']:
   for reference,treatment in [('No action','O1'),('No action','S1'),('Static','O1'),('Static','S1'),('O1','S1')]:
    a=entries[stage,reference];b=entries[stage,treatment];rr,gg,ss=road_pair(event,stage,reference,treatment,a,b,clip,inventory,membership,policy);roadrows+=rr;grouprows+=gg;summaries+=ss
    pp,ff=population_pair(event,stage,reference,treatment,a,b,blocks,stock,normal,infos);poprows+=pp;popframes+=ff
    # Exact regional primary and sensitivity reconstruction against previously qualified trajectories.
    actual=pd.DataFrame(ff)
    if reference=='No action':
     expected=pd.read_csv(bind(b['bp']/'access/paired_population_trajectories.csv'))
    elif reference=='Static':expected=pd.read_csv(bind(base/'paired_population_vs_static_trajectories.csv'))
    else:expected=pd.read_csv(bind(base/'s1_vs_o1_population_trajectories.csv'))
    if 'contrast'in expected:expected=expected[expected.contrast.eq(treatment)]
    expected=expected[expected.stage.eq(stage)&expected.scope.eq('Three wards')]
    actual=actual.sort_values(['bound','buffer_m','frame']).reset_index(drop=True);expected=expected.sort_values(['bound','buffer_m','frame']).reset_index(drop=True)
    names=['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']
    if reference=='No action':expected=expected.rename(columns={'net_avoided_loss_lower':'net_lower','net_avoided_loss_upper':'net_upper'})
    assert len(actual)==len(expected)and np.array_equal(actual[names].to_numpy(),expected[names].to_numpy()),'Population source mismatch'
    print('PAIR_COMPLETE',event,stage,reference,treatment,len(rr),len(pp),flush=True)
  pd.DataFrame(summaries).to_csv(P/'road_pair_summary.csv',index=False)
 road=pd.DataFrame(roadrows);road.to_csv(P/'new_closure_roads.csv',index=False);pd.DataFrame(grouprows).to_csv(P/'road_group_effects.csv',index=False)
 pop=pd.DataFrame(poprows);pop.to_csv(P/'block_population_effects.csv',index=False);pd.DataFrame(popframes).to_csv(P/'population_pair_trajectories.csv',index=False)
 keys=['event','stage','reference','treatment','bound','buffer_m','ward','town_id','name'];values=[x for x in pop if x.endswith('_resident_hours')]
 towns=pop.groupby(keys,as_index=False)[values].sum();towns['net_direction']=np.where(towns.net_lower_resident_hours>1e-8,'positive_across_candidate_bounds',np.where(towns.net_upper_resident_hours< -1e-8,'negative_across_candidate_bounds',np.where(towns.new_loss_upper_resident_hours>1e-8,'new_loss_with_uncertain_or_mixed_net','nonnegative_or_zero')));towns.to_csv(P/'town_population_effects.csv',index=False)
 with np.load(bind(D/'three_ward_common_depth_labels/common_support_10m.npz'))as z:affine=z['Affine Transform']
 with np.load(bind(D/'three_ward_surrogate_inputs/static_tensor_10m.npz'))as z:ward=z['Ward Code'].ravel()
 native=pd.DataFrame(native_adverse(cases,ward,affine));native.to_csv(P/'native_head_rise_samples.csv',index=False)
 changes=[k for k,(size,mtime)in STATS.items()if (ROOT/k).stat().st_size!=size or (ROOT/k).stat().st_mtime_ns!=mtime];assert not changes
 dump(P/'source_bindings.json',BIND)
 dump(R/'validation.json',dict(passed=True,comparisons=30,events=3,all109frames=True,both_depth_branches=True,all_three_candidate_radii=True,population_paired_trajectories_reproduced=True,road_groups_partition_reconciled=True,native_head_area_reproduced=True,source_files_unchanged=True,new_hydraulics=0,new_routing=0,training=False,sealed_test=False,official_population_stock=int(stock.sum()),seconds=time.monotonic()-started,script_sha256=sha(Path(__file__))))
 print('DIAGNOSIS_COMPLETE',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
