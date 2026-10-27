"""Read-only, within-event synthesis of qualified frozen O1/S1 controls."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
ARMS=['O1','S1']

def contexts(august):
 return [
  ('E20240731',ROOT/'data/processed/three_ward_shared_receiver_controls_v1',ROOT/'data/exp/estimation/three_ward_shared_receiver_controls_v1',ROOT/'data/processed/three_ward_shared_receiver_pilot_v1','__shared_r1000_q0p5'),
  ('E20250911',ROOT/'data/processed/three_ward_shared_receiver_controls_transfer_v1',ROOT/'data/exp/estimation/three_ward_shared_receiver_controls_transfer_v1',ROOT/'data/processed/three_ward_shared_receiver_transfer_v1/batches/E20250911','__shared_r1000_q0p5_transfer'),
  ('E20240830',august,ROOT/'data/exp/estimation'/august.name,ROOT/'data/processed/three_ward_shared_receiver_transfer_v1/batches/E20240830','__shared_r1000_q0p5_transfer')]

def write_json(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def reopening(codes,times):
 """Sampled sustained reopening; final closure is right-censored, not 9h recovery."""
 closed=np.asarray(codes,dtype=bool)
 if not closed.any():return 'never_closed',0.
 if closed[-1]:return 'right_censored',None
 return 'reopened',float(times[np.flatnonzero(closed)[-1]+1])

def build(cases,out,report,require_qualified=True):
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 road=[];paired=[];direct=[];head=[];groups=[];recovery=[];coverage=[];bindings={}
 members=pd.read_parquet(ROOT/'data/processed/three_ward_protection_requirements_register_v1/road_group_members_preprocessed.parquet')
 clip=np.load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');weights=clip['union_length_m']/1000
 support_vectors=[]
 def bind(f):
  f=Path(f);h=hashlib.sha256()
  with f.open('rb') as stream:
   for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
  bindings[str(f.relative_to(ROOT))]=h.hexdigest()
 for event,base,br,static,suffix in cases:
  if require_qualified:
   a=br/'final_summary_validation.json';assert json.loads(a.read_text())['passed'];bind(a)
  for name,target in [('paired_population_vs_static_summary.csv',paired),('s1_vs_o1_population_summary.csv',direct),('group_continuity_vs_static.csv',groups)]:
   f=base/name;d=pd.read_csv(f);d.insert(0,'event',event);target.append(d);bind(f)
  supports=[]
  for scheme,path in [('Static',static)]+[(arm,base/'batches'/arm)for arm in ARMS]:
   f=path/'common_road_support.npz'
   with np.load(f)as z:q=z['qualified_roads'].astype(bool)
   supports.append(q);bind(f)
   coverage.append(dict(event=event,scheme=scheme,qualified_roads=int(q.sum()),qualified_length_km=float(weights[q].sum()),support_sha256=hashlib.sha256(np.packbits(q).tobytes()).hexdigest()))
   f=path/'paired_road_effects.csv';d=pd.read_csv(f);d=d[d.Scope.eq('Three wards')];d.insert(0,'scheme',scheme);d.insert(0,'event',event);road.append(d);bind(f)
   if scheme!='Static':
    f=path/'expanded_head_transfer_summary.csv';d=pd.read_csv(f);d.insert(0,'scheme',scheme);d.insert(0,'event',event);head.append(d);bind(f)
  assert all(np.array_equal(supports[0],q)for q in supports[1:]),'Within-event supports differ'
  support_vectors.append(supports[0])
  # Opening means none of the registered group members is closed; incomplete groups remain explicit.
  for stage in ['H','L']:
   for scheme,path in [('Static',static)]+[(arm,base/'batches'/arm)for arm in ARMS]:
    case=f'{event}__{stage}__U'+(suffix if scheme=='Static'else '__'+scheme.lower())
    f=path/'exposure'/(case+'__drainage.npz');bind(f)
    with np.load(f)as z:
     codes=z['road_state_codes'];indices=z['road_indices'];times=z['analysis_times']
    assert codes.shape==(109,2,len(indices)) and len(times)==109 and abs(times[-1]-32400)<.01
    lookup={int(k):i for i,k in enumerate(indices)}
    for group,g in members.groupby('Road Group'):
     ii=[lookup[int(k)]for k in g['Road Index']if int(k)in lookup]
     complete=len(ii)==len(g)
     for bi,bound in enumerate(['lower','upper']):
      status,opening=('unsupported',None)if not ii else reopening((codes[:,bi,ii]==2).any(axis=1),times)
      recovery.append(dict(event=event,scheme=scheme,stage=stage,bound=bound,road_group=group,all_registered_members_qualified=complete,registered_members=len(g),qualified_members=len(ii),status=status,sustained_reopening_elapsed_s=opening))
 frames={'road':pd.concat(road,ignore_index=True),'paired':pd.concat(paired,ignore_index=True),'direct':pd.concat(direct,ignore_index=True),'head':pd.concat(head,ignore_index=True),'groups':pd.concat(groups,ignore_index=True),'recovery':pd.DataFrame(recovery)}
 filenames={'road':'three_event_road_effects.csv','paired':'three_event_population_vs_static.csv','direct':'three_event_s1_vs_o1_population.csv','head':'three_event_head_transfer.csv','groups':'three_event_group_open_time_gains.csv','recovery':'three_event_group_reopening.csv'}
 for k,d in frames.items():d.to_csv(out/filenames[k],index=False)
 pd.DataFrame(coverage).to_csv(out/'three_event_road_coverage.csv',index=False)
 rows=[]
 for event,base,br,static,suffix in cases:
  for stage in ['H','L']:
   rd=frames['road'];rd=rd[rd.event.eq(event)&rd.Stage.eq(stage)&rd['Depth Branch'].eq('lower')].set_index('scheme')
   dd=frames['direct'];dd=dd[dd.event.eq(event)&dd.stage.eq(stage)&dd.bound.eq('lower')&dd.buffer_m.eq(10)&dd.scope.eq('Three wards')];assert len(dd)==1;d=dd.iloc[0]
   rows.append(dict(event=event,stage=stage,O1_closure_reduction_km_h=float(rd.loc['O1','Closure Reduction km h']),S1_closure_reduction_km_h=float(rd.loc['S1','Closure Reduction km h']),S1_minus_O1_road_benefit_km_h=float(rd.loc['S1','Closure Reduction km h']-rd.loc['O1','Closure Reduction km h']),S1_vs_O1_population_net_lower_resident_hours=float(d.net_lower_resident_hours),S1_vs_O1_population_net_upper_resident_hours=float(d.net_upper_resident_hours)))
 # Keep interval branches, scope, and receiver limitations instead of ranking by interval midpoint.
 result=dict(passed=True,events=[x[0]for x in cases],rows=rows,within_event_supports_exact=True,cross_event_supports_identical=all(np.array_equal(support_vectors[0],q)for q in support_vectors[1:]),coverage=coverage,road_unit='km h;paired closure-length integral,not unique road length',population_unit='resident hours;2020 at-residence stock and candidate-location intervals,not unique people or confidence intervals',population_ranking='Do not rank by interval midpoint; intervals spanning zero retain unresolved net direction.',recovery='Five-minute sampled sustained reopening of fully qualified groups; never-closed and still-closed-at9h remain separate. Unknown members are never treated as open.',scope='Within-event contrasts only; no pooled weather effects without support harmonization.',receiver='Prescribed stage; added receiving-river storage/downstream effects are not simulated.',history='Frozen July selection transferred to already-inspected parents;not untouched test or observed reconstruction. August excludes prior-window rainfall and starts dry.',cost='Same2.5m3/s allowance is not equal equipment or financial cost.',automatic_new_runs=False)
 write_json(report/'three_event_synthesis.json',result);write_json(out/'three_event_input_bindings.json',bindings)
 return result

def synthesize(out,report):
 result=build(contexts(out),out,report,require_qualified=False)
 # The caller independently verifies August28+8gates before this invocation.
 assert result['events']==['E20240731','E20250911','E20240830'] and len(result['rows'])==6
 for _,_,br,_,_ in contexts(out)[:2]:assert json.loads((br/'final_summary_validation.json').read_text())['passed']
 return result

def selfcheck(out,report):
 assert reopening([False,True,False,False],[0,300,600,900])==('reopened',600.)
 assert reopening([False,True],[0,300])==('right_censored',None)
 assert reopening([False,False],[0,300])==('never_closed',0.)
 result=build(contexts(out)[:2],out,report)
 assert len(result['rows'])==4
 for row in result['rows']:
  assert row['S1_minus_O1_road_benefit_km_h']>0
 # The merger preserves each original paired value and every uncertainty branch.
 merged=pd.read_csv(out/'three_event_population_vs_static.csv')
 for event,base,_,_,_ in contexts(out)[:2]:
  a=merged[merged.event.eq(event)].drop(columns='event').reset_index(drop=True);b=pd.read_csv(base/'paired_population_vs_static_summary.csv');pd.testing.assert_frame_equal(a,b,rtol=1e-12,atol=1e-9)
 write_json(report/'validation.json',dict(passed=True,two_completed_events_reproduced=True,all_population_branches_preserved=True,reopening_censor_tests=True,no_new_hydraulics=True))
