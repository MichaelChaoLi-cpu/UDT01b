"""Fixed, equal-length assumed road protection: no new flood simulation or optimization claim."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']: os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,time,collections,gzip,argparse
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_motor_flood_access_v1 import P as SOURCE, read,write,dump,load,sha, N
from run_motor_flood_access_v1 import graph,route
P=ROOT/'data/processed/three_ward_motor_road_protection_v1'
R=ROOT/'data/exp/estimation/three_ward_motor_road_protection_v1'
ARMS=['none','corridor','distributed']

def setup():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 router=load(SOURCE/'turn_router.npz');projection=load(SOURCE/'road_projection.npz');roads=read(SOURCE/'road_inventory.csv');probes=read(SOURCE/'probe_inventory.csv');old=load(SOURCE/'accessibility.npz');support=load(SOURCE/'evidence_support.npz');scenarios=read(SOURCE/'scenario_index.csv')
 return router,projection,roads,probes,old,support,scenarios

def common_pairs(old,support,direction=0):
 return np.isfinite(old['normal_road_distance_m'][:,direction]).all(axis=0)&support['eligible_outward'if direction==0 else'eligible_two_direction'][None,:]

def selection():
 router,p,roads,probes,old,support,scenarios=setup()
 assert not (P/'protocol.json').exists(),'Selection frozen; use run/report, never silently overwrite.'
 pi=next(i for i,s in enumerate(scenarios) if s['case_id']=='R1__L__U'and s['frame_id']=='14400'and float(s['threshold_m'])==.3 and s['unknown_branch']=='unknown_available')
 closed=np.unpackbits(old['closed_roads_packed'][pi])[:len(roads)].astype(bool)
 core=np.array([r['inside_core']=='True'for r in roads]);length=np.array([float(r['length_m'])for r in roads]);candidate=closed&p['qualified']&core
 # Distances/predecessors in the original open graph; protection never introduces a new road.
 g=graph(router,np.zeros(len(roads),bool));normal,pred=dijkstra(g.T.tocsr(),indices=router['target_destination'],return_predecessors=True,directed=True)
 common=common_pairs(old,support);packages=[];route_records=[];freq=np.zeros(len(roads),int)
 for target,j in zip(*np.where(common&~np.isfinite(old['road_distance_m'][pi,0]))):
  node=int(router['probe_source'][j]);stop=int(router['target_destination'][target]);used=[];steps=0
  while node!=stop:
   nxt=int(pred[target,node]);assert nxt>=0
   a,b=router['indptr'][node:node+2];at=int(a+np.searchsorted(router['indices'][a:b],nxt));assert router['indices'][at]==nxt
   r=int(router['road_index'][at])
   if r>=0:used.append(r)
   node=nxt;steps+=1;assert steps<len(router['indices'])
  used=np.unique(used);wet=used[closed[used]];unprotectable=wet[~candidate[wet]]
  route_records.append(dict(probe_id=probes[j]['source_display_cell_id'],target=['North','East','South','West'][target],wet_roads=wet.tolist(),unprotectable_roads=unprotectable.tolist(),native_route_length_m=float(normal[target,router['probe_source'][j]])))
  if len(wet) and not len(unprotectable):packages.append(set(map(int,wet)));freq[wet]+=1
 assert packages,'No repairable planning paths'
 # A transparent normal-route package heuristic, not an optimal all-network solver.
 selected=set();steps=[];budget=5000.
 while True:
  best=None
  for package in packages:
   add=package-selected;cost=float(length[list(add)].sum())if add else 0.
   if cost<=0 or length[list(selected)].sum()+cost>budget:continue
   before=sum(q<=selected for q in packages);after=sum(q<=(selected|add) for q in packages);gain=after-before
   key=(gain/cost,gain,-cost,-min(add))
   if best is None or key>best[0]:best=(key,add,cost,gain)
  if best is None:break
  _,add,cost,gain=best;selected|=add;steps.append(dict(step=len(steps)+1,added_roads=sorted(add),added_length_m=cost,new_completed_normal_route_packages=gain,total_length_m=float(length[list(selected)].sum())))
 assert selected
 target_length=float(length[list(selected)].sum())
 # No flood or accessibility outcome ranking in the distributed arm. Geographic500m strata,
 # longest-first within each stratum; visit farthest-first spatial strata, one road per stratum/round.
 coords={};candidateids={roads[i]['source_edge_id']:i for i in np.flatnonzero(candidate)}
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line);j=candidateids.get(e['edge_id'])
   if j is not None:
    xy=np.asarray(e['geometry']['coordinates']);coords[j]=tuple((xy[0]+xy[-1])/2)
 bins=collections.defaultdict(list)
 for j in np.flatnonzero(candidate):
  x,y=coords[int(j)];bins[(int(np.floor(x/500)),int(np.floor(y/500)))].append(int(j))
 for b in bins:bins[b].sort(key=lambda j:(-length[j],j))
 # Geographic order is fixed without route outcomes: first near domain center,
 # then maximize distance to previously visited strata, avoiding a west-to-east budget bias.
 pending=set(bins);first=min(pending,key=lambda b:((b[0]*500+250+7510)**2+(b[1]*500+250+37380)**2,b));bin_order=[first];pending.remove(first)
 while pending:
  nxt=min(pending,key=lambda b:(-min((b[0]-a[0])**2+(b[1]-a[1])**2 for a in bin_order),b));bin_order.append(nxt);pending.remove(nxt)
 distributed=set();spent=0.;rounds=0
 while True:
  added=False
  for b in bin_order:
   choices=[j for j in bins[b]if j not in distributed and spent+length[j]<=target_length]
   if choices:
    j=choices[0];distributed.add(j);spent+=length[j];added=True
  rounds+=1
  if not added:break
 assert abs(spent-target_length)<=max(25.,.005*target_length),(spent,target_length)
 masks=np.zeros((3,len(roads)),bool);masks[1,list(selected)]=True;masks[2,list(distributed)]=True
 assert np.all(~masks[1:]|candidate)
 np.savez_compressed(P/'protected_roads.npz',protected=masks,candidate=candidate)
 contract=dict(human_decision='MILI-D-20261017-010',selection_case=scenarios[pi],planning_index=pi,nominal_cap_m=budget,actual_length_m={ARMS[i]:float(length[masks[i]].sum())for i in range(3)},length_definition='Sum of entire source physical road-segment centerline lengths intersecting core, counted once; not lanes, not clipped ward-length, not engineering cost.',length_match_tolerance_m=max(25.,.005*target_length),arms=ARMS,corridor_selection='Greedy complete normal-shortest-route repair packages for lost common-supported outward source-to-sector pairs at planning condition. Score newly completed packages per added source length; no all-network optimality or observational benefit claim.',distributed_selection='Same eligible flooded source-ground-road pool;500m endpoint-midpoint strata, longest-first within stratum, farthest-first spatially dispersed strata, one road per stratum per round until matched budget. No outcome-based re-ranking; one deterministic allocation, not an ensemble of random allocations.',selection_candidate_roads=int(candidate.sum()),planning_pairs=int(common.sum()),planning_lost_pairs=int((common&~np.isfinite(old['road_distance_m'][pi,0])).sum()),repairable_route_packages=len(packages),evaluation='Both plans frozen for all624 original network states;624 no-intervention outcomes reused,1248 intervention evaluations. Fixed actual times primary; non-simultaneous peak is separate stress envelope.',main_transfer_cases=['R1__H__U','R125__L__U','R125__H__U'],selection_bias='Planning-case improvement is in-sample and intended by heuristic. Stronger rainfall and stage are conditional variants of the same historical parent, not independent-event validation.',protection_assumption='Remove flood-closure gates only on selected qualified native ground-road segments. Assume passability regardless of modeled depth; do not change elevations, water levels, flow, unknown roads, permissions, turn rules or native directions. Does not prove protection feasible or account for flooding redirected to other roads.',metrics='Fixed common93outward/89bidirectional probes; common source-to-sector loss; reopened pairs; paired path length savings only among routes reachable without intervention. Newly restored pairs separately counted; no travel time, congestion, capacity, people or affected areas.',stopping='Only two fixed plans across624existingstates; stop on input mutation, road-mask violation, routing monotonicity failure or budget mismatch. No search/tuning after outcome inspection.',new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False)
 dump(P/'protocol.json',contract);dump(P/'planning_route_packages.json',route_records);dump(P/'selection_steps.json',steps)
 inventory=[]
 for arm in [1,2]:
  for j in np.flatnonzero(masks[arm]):inventory.append(dict(arm=ARMS[arm],**roads[j],midpoint_x_m=coords[int(j)][0],midpoint_y_m=coords[int(j)][1],selection_common_normal_path_count=int(freq[j])))
 write(P/'protected_road_inventory.csv',inventory)
 sources=[SOURCE/n for n in ['turn_router.npz','road_projection.npz','road_inventory.csv','probe_inventory.csv','accessibility.npz','evidence_support.npz','scenario_index.csv','analysis_contract.json']]+[SOURCE/'exposures'/(c+'.npz')for c in sorted({s['case_id']for s in scenarios})]+[N/'surface_physical_edges.jsonl.gz',ROOT/'src/analyses/run_motor_flood_access_v1.py',ROOT/'src/preprocessing/prepare_motor_flood_access_v1.py']
 dump(P/'source_bindings.json',{str(f.relative_to(ROOT)):sha(f)for f in sources});dump(P/'frozen_selection_binding.json',{str(f.relative_to(ROOT)):sha(f)for f in [P/'protocol.json',P/'protected_roads.npz',P/'protected_road_inventory.csv',P/'planning_route_packages.json',P/'selection_steps.json']})
 print('FROZEN SELECTION',contract['actual_length_m'],'roads',[int(m.sum())for m in masks], 'packages',len(packages),flush=True)

def run():
 start=time.monotonic();router,p,roads,probes,old,support,scenarios=setup();masks=load(P/'protected_roads.npz')['protected'];contract=json.loads((P/'protocol.json').read_text());assert not(P/'accessibility.npz').exists(),'Do not overwrite completed intervention results'
 distances=[old['road_distance_m']];cache={};n=len(roads)
 for ai in [1,2]:
  result=[]
  for i,s in enumerate(scenarios):
   closed=np.unpackbits(old['closed_roads_packed'][i])[:n].astype(bool);closed&=~masks[ai];sig=np.packbits(closed).tobytes()
   if sig not in cache:cache[sig]=route(router,graph(router,closed))
   result.append(cache[sig])
   if (i+1)%52==0:
    dump(R/'execution_state.json',dict(status='running',arm=ARMS[ai],intervention_evaluations_done=(ai-1)*624+i+1,intervention_evaluations_total=1248,seconds=time.monotonic()-start,new_hydraulics=False));print(ARMS[ai],i+1,'/624',round(time.monotonic()-start),flush=True)
  distances.append(np.stack(result))
 np.savez_compressed(P/'accessibility.npz',road_distance_m=np.stack(distances))
 dump(R/'execution_state.json',dict(status='network_complete',intervention_evaluations_done=1248,reused_no_intervention=624,unique_intervention_graphs=len(cache),seconds=time.monotonic()-start,new_hydraulics=False))


def summarize():
 router,p,roads,probes,old,support,scenarios=setup();a=load(P/'accessibility.npz')['road_distance_m'];masks=load(P/'protected_roads.npz')['protected'];classes=[];groups=collections.defaultdict(list);burden=[];pairout=common_pairs(old,support);pairback=common_pairs(old,support,1)
 for i,s in enumerate(scenarios):groups[s['case_id'],s['frame_id']].append(i)
 for ai,arm in enumerate(ARMS):
  for(case,frame),ids in groups.items():
   finite=np.isfinite(a[ai,ids]);out=finite[:,0].any(axis=1);two=finite.any(axis=2).all(axis=1)
   for direction,values,key in [('outward',out,'eligible_outward'),('two_direction',two,'eligible_two_direction')]:
    ok=support[key];counts=dict(robust_modeled_loss=int((ok&~values.any(axis=0)).sum()),conditional_sensitivity=int((ok&values.any(axis=0)&~values.all(axis=0)).sum()),retained_access=int((ok&values.all(axis=0)).sum()),insufficient_evidence=int((~ok).sum()))
    classes.append(dict(arm=arm,case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',direction=direction,**counts))
  for i,s in enumerate(scenarios):
   bi=int(s['unknown_branch']=='unknown_withheld')
   for direction,common in [(0,pairout),(1,pairback)]:
    before=old['road_distance_m'][i,direction];after=a[ai,i,direction];norm=old['normal_road_distance_m'][bi,direction];restored=common&~np.isfinite(before)&np.isfinite(after);survive=common&np.isfinite(before);savings=before[survive]-after[survive];remaining=common&np.isfinite(after);extra=after[remaining]-norm[remaining]
    burden.append(dict(arm=arm,**s,direction=['outward','return'][direction],common_direction_pairs=int(common.sum()),lost_direction_pairs=int((common&~np.isfinite(after)).sum()),restored_direction_pairs=int(restored.sum()),baseline_reachable_pairs=int(survive.sum()),paired_distance_saving_sum_m=float(savings.sum()),paired_distance_saving_median_m=float(np.median(savings))if len(savings)else None,paired_distance_saving_p90_m=float(np.quantile(savings,.9))if len(savings)else None,remaining_pairs=int(remaining.sum()),remaining_extra_distance_median_m=float(np.median(extra))if len(extra)else None))
 write(R/'classification_summary.csv',classes);write(R/'directional_benefits.csv',burden)
 # Actual support and branch sensitivity per point retained for downstream regional maps.
 pc=[]
 for ai,arm in enumerate(ARMS):
  for(case,frame),ids in groups.items():
   out=np.isfinite(a[ai,ids,0]).any(axis=1)
   for j,pr in enumerate(probes):
    label='insufficient_evidence'if not support['eligible_outward'][j]else 'robust_modeled_loss'if not out[:,j].any()else'retained_access'if out[:,j].all()else'conditional_sensitivity'
    pc.append(dict(arm=arm,case_id=case,frame_id=frame,probe_id=pr['source_display_cell_id'],outward_classification=label,x_m=pr['x_m'],y_m=pr['y_m']))
 write(R/'probe_classification.csv',pc)
 # Validation checks every intervention mask, finite distances, unchanged support, controls.
 checks=[]
 def check(name,passed):
  checks.append(dict(check=name,passed=bool(passed)));assert passed,name
 def equal(x,y):return np.array_equal(np.isfinite(x),np.isfinite(y))and np.allclose(x[np.isfinite(x)],y[np.isfinite(y)],atol=1e-7,rtol=0)
 check('no_intervention_exactly_reproduced',equal(a[0],old['road_distance_m']))
 qual=p['qualified'];length=np.array([float(r['length_m'])for r in roads]);protocol=json.loads((P/'protocol.json').read_text())
 check('only_qualified_core_ground_selected',np.all(~masks[1:]|(qual&np.array([r['inside_core']=='True'for r in roads]))))
 check('budgets_match',abs(length[masks[1]].sum()-length[masks[2]].sum())<=protocol['length_match_tolerance_m'])
 check('no_intervention_mask_empty',not masks[0].any())
 for ai in [1,2]:
  for i,s in enumerate(scenarios):
   closed=np.unpackbits(old['closed_roads_packed'][i])[:len(roads)].astype(bool);newclosed=closed&~masks[ai];check(f'mask_{ai}_{i}',np.array_equal(newclosed&~qual,closed&~qual)and not newclosed[masks[ai]].any())
   before=a[0,i];after=a[ai,i];base=old['normal_road_distance_m'][int(s['unknown_branch']=='unknown_withheld')];check(f'monotonic_{ai}_{i}',np.all(np.isfinite(after[np.isfinite(before)]))and np.all(after[np.isfinite(before)]<=before[np.isfinite(before)]+1e-6)and np.all(after[np.isfinite(base)]>=base[np.isfinite(base)]-1e-6)and not (np.isfinite(after)&~np.isfinite(base)).any())
   if not (closed&masks[ai]).any():check(f'unchanged_without_activated_protection_{ai}_{i}',equal(before,after))
 # Independently reroute all four planning and high-stress branches, using frozen masks.
 sampleids=[i for i,s in enumerate(scenarios)if s['case_id']in ['R1__L__U','R125__H__U']and s['frame_id']=='14400']
 for ai in [1,2]:
  for i in sampleids:
   closed=np.unpackbits(old['closed_roads_packed'][i])[:len(roads)].astype(bool)&~masks[ai];check(f'reroute_{ai}_{i}',equal(route(router,graph(router,closed)),a[ai,i]))
 for name in ['source_bindings.json','frozen_selection_binding.json']:
  bindings=json.loads((P/name).read_text());check(name,all(sha(ROOT/f)==h for f,h in bindings.items()))
 check('624states_3arms_same93outward89bidirectional',a.shape==(3,624,2,4,209)and support['eligible_outward'].sum()==93 and support['eligible_two_direction'].sum()==89)
 check('all_classifications_partition209',all(sum(int(r[k])for k in ['robust_modeled_loss','conditional_sensitivity','retained_access','insufficient_evidence'])==209 for r in classes))
 dump(R/'validation.json',dict(all_passed=True,checks=checks,count=len(checks),independent_reroutes=len(sampleids)*2,no_new_hydraulics=True,no_people=True));print('VALIDATED',len(checks),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','run','summarize']);args=parser.parse_args();{'select':selection,'run':run,'summarize':summarize}[args.phase]()
