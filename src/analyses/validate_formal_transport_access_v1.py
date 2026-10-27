"""Independent closure semantics, native-arc and full shortest-path qualification."""
from pathlib import Path
import os,sys,json,collections,itertools,time,hashlib
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_formal_transport_access_v1 import P,R,OLD,STRICT,T,F,D,CASES,PROJECTIONS,UNCERTAINTIES,KINDS,IDENTITIES,THRESHOLDS,read,write,dump,load,sha

def main():
 start=time.monotonic();checks=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)))
  if not v:raise AssertionError(name)
 sources=json.loads((P/'source_bindings.json').read_text());check('all_projected_sources_unchanged',all(sha(ROOT/k)==v for k,v in sources.items()))
 check('no_sealed_test_contents',not any('sealed_test-'in p for p in sources))
 s=load(P/'projection_support.npz');w=load(OLD/'local_walking_graph.npz');a=load(P/'alternative_distances.npz');meta=read(OLD/'road_segment_inventory.csv');probes=read(OLD/'surface_probe_inventory.csv');contexts=read(P/'road_context_index.csv');scenarios=read(P/'scenario_index.csv');selections=read(P/'station_closure_selection.csv');edges=json.loads((T/'hypothetical_network_edges.json').read_text());old=load(T/'hypothetical_network.npz')
 check('complete_registered_network_matrix',len(contexts)==1248 and len(scenarios)==9984 and len({(r['case_id'],r['frame_id'],r['projection'],r['road_threshold_m'],r['unknown_branch'],r['identity_policy'],r['shutdown_type'])for r in scenarios})==9984)
 sourcegraph=load(ROOT/'data/processed/three_ward_normal_transport_network_v1/walk_conservative_graph.npz');si=w['source_arc_index'];gi=w['source_global_index']
 check('native_permissions_directions_lengths_unchanged',np.array_equal(gi[w['from_index']],sourcegraph['from_index'][si])and np.array_equal(gi[w['to_index']],sourcegraph['to_index'][si])and np.array_equal(w['length_m'],sourcegraph['length_m'][si]))
 check('no_probe_resnapping',all(str(w['node_ids'][w['probe_node_index'][j]])==p['source_walk_node']for j,p in enumerate(probes)))
 check('all_surface_targets_native_outside_nodes',not w['inside_core'][w['exit_node_index']].any()and len(w['exit_node_index'])==461)
 nw=len(w['node_ids']);ne=len(w['from_index']);ni={str(k):i for i,k in enumerate(w['node_ids'])};keys=list(map(str,old['node_ids']));oi={k:i for i,k in enumerate(keys)};ri=np.array([i for i,e in enumerate(edges)if e['edge_kind']not in ['initial_walk','final_walk']]);cu=[];cv=[]
 for p in read(T/'portal_inventory.csv'):
  for k in json.loads(p['walk_node_ids']):
   if k in ni:cu.extend([ni[k],nw+oi['P:'+p['portal_id']]]);cv.extend([nw+oi['P:'+p['portal_id']],ni[k]])
 uu=np.r_[w['from_index'],old['from_index'][ri]+nw,cu];vv=np.r_[w['to_index'],old['to_index'][ri]+nw,cv];ll=np.r_[w['length_m'],np.zeros(len(ri)+len(cu))];nn=nw+len(keys)
 def reducer(u,v,weights,n):
  order=np.lexsort((v,u));us=u[order];vs=v[order];groups=np.r_[0,np.flatnonzero((us[1:]!=us[:-1])|(vs[1:]!=vs[:-1]))+1]
  def matrix(keep):
   val=np.minimum.reduceat(np.where(keep[order],weights[order],np.inf),groups);ok=np.isfinite(val);return csr_matrix((val[ok],(us[groups][ok],vs[groups][ok])),shape=(n,n))
  return matrix
 walk=reducer(w['from_index'],w['to_index'],w['length_m'],nw);mixed=reducer(uu,vv,ll,nn);pn=w['probe_node_index'];exits=w['exit_node_index'];anchors=load(T/'paired_accessibility_matrices.npz')['anchor_station_ids'];ad=[nw+oi['D:station:'+str(x)]for x in anchors];ao=[nw+oi['O:station:'+str(x)]for x in anchors]
 def wd(g):return np.stack([dijkstra(g.T.tocsr(),indices=exits,directed=True,min_only=True),dijkstra(g,indices=exits,directed=True,min_only=True)])[:,pn]
 def rd(g):return np.stack([dijkstra(g.T.tocsr(),indices=ad,directed=True,min_only=True),dijkstra(g,indices=ao,directed=True,min_only=True)])[:,pn]
 def same(x,y):return np.array_equal(np.isfinite(x),np.isfinite(y))and np.allclose(x[np.isfinite(x)],y[np.isfinite(y)],atol=1e-7,rtol=1e-10)
 offsets=w['edge_sample_offsets'];indices=w['edge_sample_road_index'];lens=np.diff(offsets);grade=np.array([p['known_surface_grade']=='True'and p['ambiguous_parallel_geometry']=='False'for p in meta]);outside=np.array([p['has_out_of_grid_sample']=='True'for p in meta]);reduced=np.logical_and.reduceat(np.r_[s['road_known'][indices],True],offsets[:-1]);expected_unknown=~grade|outside|(lens==0)|~reduced
 check('strict_unknown_reconstructed',np.array_equal(expected_unknown,s['unknown'][0]))
 common=load(F/'factor_comparison_v1/common_support.npz')['common_admitted'].ravel();sp=load(OLD/'ground_path_supplement.npz');pointok=common[sp['coarse_flat_index']];allpoints=np.logical_and.reduceat(np.r_[pointok[sp['sample_point_index']],True],sp['sample_offsets'][:-1]);expected_qualified=sp['qualified']&allpoints
 check('supplement_source_scope_and_common_support',np.array_equal(expected_qualified,s['supplement_qualified'])and not np.any(s['supplement_qualified']&~expected_unknown))
 check('supplement_unknown_reconstructed',np.array_equal(expected_unknown&~expected_qualified,s['unknown'][1]))
 check('unresolved_vertical_segments_stay_unknown',all(s['unknown'][:,i].all()for i,p in enumerate(meta)if p['known_surface_grade']!='True'))
 for pi,projection in enumerate(PROJECTIONS):
  for bi,branch in enumerate(UNCERTAINTIES):
   keep=~s['unknown'][pi,w['arc_basis']]if bi else np.ones(ne,bool)
   check(projection+branch+'_normal_walking_paths',same(wd(walk(keep)),a['baseline_walking_exit_m'][pi,bi]))
   check(projection+branch+'_normal_mixed_paths',same(rd(mixed(np.r_[keep,np.ones(len(ri)+len(cu),bool)])),a['baseline_mixed_min_walking_m'][pi,bi]))
 contexts_by={};wcache={};ecache={};road_ok=True;walk_ok=True
 for ci,row in enumerate(contexts):
  case=row['case_id'];e=ecache.get(case)
  if e is None:ecache.clear();e=load(P/'exposures'/(case+'.npz'));ecache[case]=e
  frame=row['frame_id'];ti=json.loads((P/'analysis_contract.json').read_text())['frame_ids'].index(frame);pi=PROJECTIONS.index(row['projection']);bi=UNCERTAINTIES.index(row['unknown_branch']);bound='lower'if bi==0 else'upper';values=np.where(s['road_known'][indices],e['road_'+bound][ti,indices],-np.inf);mx=np.maximum.reduceat(np.r_[values,-np.inf],offsets[:-1]);mx[lens==0]=-np.inf;wet=grade&(mx>=float(row['road_threshold_m']))
  if pi:wet |=expected_qualified&(e['supplement_'+bound][ti]>=float(row['road_threshold_m']))
  closed=wet|(s['unknown'][pi]if bi else False);road_ok &=np.array_equal(closed,a['road_closed'][ci]);keep=~closed[w['arc_basis']];sig=hashlib.sha256(keep.tobytes()).digest()
  if sig not in wcache:wcache[sig]=wd(walk(keep))
  walk_ok &=same(wcache[sig],a['walking_exit_m'][ci]);contexts_by[case,frame,row['projection'],float(row['road_threshold_m']),row['unknown_branch']]=ci
 check('all1248_road_masks_reproduced',road_ok);check('all1248_outward_return_walking_distances_recomputed',walk_ok)
 links={p:collections.defaultdict(set)for p in IDENTITIES}
 for row in read(ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv'):
  for p in IDENTITIES:links[p][row['osm_node_id']].add(row['station_id'])
 for row in read(T/'identity_attribution_sensitivity_links.csv'):links[IDENTITIES[1]][row['osm_node_id']].add(row['station_id'])
 portalids={x['portal_id']for x in read(T/'portal_inventory.csv')};selection_by={};selectok=True;ecache={}
 for row in selections:
  case=row['case_id'];e=ecache.get(case)
  if e is None:ecache.clear();e=load(P/'exposures'/(case+'.npz'));ecache[case]=e
  ti=json.loads((P/'analysis_contract.json').read_text())['frame_ids'].index(row['frame_id']);bound='lower'if row['unknown_branch']=='optimistic'else'upper';ports=set();targets=set()
  for j,osm in enumerate(map(str,s['entrance_ids'])):
   cells=s['entrance_cell_index'][s['entrance_offsets'][j]:s['entrance_offsets'][j+1]];total=s['ground_area'][cells].sum();admitted=s['ground_area'][cells][s['ground_known'][cells]].sum();wet=s['ground_area'][cells][s['ground_known'][cells]&(e['ground_'+bound][ti,cells]>=.3)].sum()
   if total<=0 or admitted<100 or admitted/total<.8 or wet<100:continue
   if links[row['identity_policy']][osm]and'osm:'+osm in portalids:ports.add('osm:'+osm);targets.update(links[row['identity_policy']][osm])
  selectok &=ports==set(filter(None,row['selected_portal_ids'].split(';')))and targets==set(filter(None,row['target_station_ids'].split(';')))
  selection_by[case,row['frame_id'],row['unknown_branch'],row['identity_policy']]=(ports,targets)
 check('all624_station_selections_reproduced',selectok)
 # Independent edge-kind implementation, including an explicit no-rail-closure arm.
 rcache={};rail_ok=True;distance_ok=True;longer=True;nest={};before=a['baseline_mixed_min_walking_m']
 for i,row in enumerate(scenarios):
  ci=int(row['road_context_index']);pi=PROJECTIONS.index(row['projection']);bi=UNCERTAINTIES.index(row['unknown_branch']);ports,targets=selection_by[row['case_id'],row['frame_id'],row['unknown_branch'],row['identity_policy']];kind=row['shutdown_type'];cut=[]
  for k in ri:
   e=edges[k];c=False
   if kind!=KINDS[0]:c=e['edge_kind']in['station_entry','station_exit']and e['portal_id']in ports
   if kind in ['station','section']:c |=e['edge_kind']in['station_entry','station_exit','board','alight']and e['station_id']in targets;c |=e['edge_kind']=='transfer'and bool(targets.intersection(e.get('transfer_units',[])))
   if kind=='section':c |=e['edge_kind']=='rail_service'and bool(targets.intersection(e['section_exposure_station_ids']))
   cut.append(c)
  cut=np.array(cut,bool);saved=np.unpackbits(a['rail_closed_packed'][i])[:len(ri)].astype(bool);rail_ok &=np.array_equal(cut,saved)
  keep=np.r_[~a['road_closed'][ci,w['arc_basis']],~cut,np.ones(len(cu),bool)];sig=hashlib.sha256(keep.tobytes()).digest()
  if sig not in rcache:rcache[sig]=rd(mixed(keep))
  distance_ok &=same(rcache[sig],a['mixed_min_walking_m'][i]);after=a['mixed_min_walking_m'][i];finite=np.isfinite(after);longer &=np.all(after[finite]>=before[pi,bi][finite]-1e-6)
  nest[row['case_id'],row['frame_id'],row['projection'],float(row['road_threshold_m']),row['unknown_branch'],row['identity_policy'],kind]=i
  if i%1600==0:print('VALIDATE NETWORK',i,'of9984',flush=True)
 check('all9984_rail_closures_independently_reproduced',rail_ok);check('all9984_outward_return_mixed_distances_recomputed',distance_ok);check('closures_do_not_improve_shortest_walking_distance',longer)
 nested=True
 policy=json.loads((P/'analysis_contract.json').read_text())
 for case,frame,projection,th,br,identity in itertools.product(CASES,policy['frame_ids'],PROJECTIONS,THRESHOLDS,UNCERTAINTIES,IDENTITIES):
  inds=[nest[case,frame,projection,th,br,identity,k]for k in KINDS];cuts=[np.unpackbits(a['rail_closed_packed'][i])[:len(ri)].astype(bool)for i in inds];nested &=not cuts[0].any()and all(not np.any(cuts[j]&~cuts[j+1])for j in range(3))
 check('four_shutdown_arms_nested_and_no_rail_control_explicit',nested)
 # Preserve the complete earlier historical U network outputs wherever support is identical.
 goldenok=True
 for pi,base in enumerate([STRICT,OLD]):
  prior=load(base/'alternative_distances.npz');oldrows=read(base/'scenario_index.csv')
  for row in oldrows:
   case='R1__'+('L'if row['case_id']=='radar_low'else'H')+'__U';si=nest[case,row['frame_id'],PROJECTIONS[pi],float(row['road_threshold_m']),row['unknown_branch'],row['identity_policy'],row['shutdown_type']]
   goldenok &=same(a['mixed_min_walking_m'][si],prior['mixed_min_walking_m'][int(row['scenario_index'])])
 check('all1248_prior_historical_U_scenarios_reproduced',goldenok)
 # Classifications independently read saved distances and original matched baselines.
 basepossible=np.isfinite(a['baseline_walking_exit_m'])|np.isfinite(a['baseline_mixed_min_walking_m']);check('matched_directional_normal_baselines',np.array_equal(basepossible,a['baseline_possible']))
 for pi in range(2):
  ik=np.zeros(nw,bool);keep=~s['unknown'][pi,w['arc_basis']];ik[w['from_index'][keep]]=True;ik[w['to_index'][keep]]=True
  check(PROJECTIONS[pi]+'_evidence_cohort',np.array_equal(basepossible[pi,:,0].all(axis=0)&ik[pn],a['evidence_outward'][pi])and np.array_equal(basepossible[pi].all(axis=(0,1))&ik[pn],a['evidence_two_direction'][pi]))
 classes=read(R/'probe_classification.csv');grouped=collections.defaultdict(list)
 for i,row in enumerate(scenarios):grouped[row['case_id'],row['frame_id'],row['projection'],row['shutdown_type']].append(i)
 classok=True;pid={p['probe_id']:i for i,p in enumerate(probes)}
 for row in classes:
  pi=PROJECTIONS.index(row['projection']);j=pid[row['probe_id']];ids=grouped[row['case_id'],row['frame_id'],row['projection'],row['shutdown_type']];ci=np.array([int(scenarios[i]['road_context_index'])for i in ids]);reach=np.isfinite(a['walking_exit_m'][ci,:,j])|np.isfinite(a['mixed_min_walking_m'][ids,:,j]);two=reach.all(axis=1)
  for direction,values,eligible in [('outward',reach[:,0],a['evidence_outward'][pi,j]),('two_direction',two,a['evidence_two_direction'][pi,j])]:
   label='insufficient_evidence'if not eligible else'robust_modeled_disconnection'if not values.any()else'stable_modeled_alternative'if values.all()else'conditional_sensitivity';classok &=label==row[direction+'_classification']
 check('all_probe_classifications_recomputed_from_directional_paths',classok)
 check('no_population_training_or_new_hydraulics',all(policy[k]is False for k in ['new_hydraulics','people_counted','training','sealed_tests_opened']))
 check('sources_still_unchanged_at_end',all(sha(ROOT/k)==v for k,v in sources.items()))
 dump(R/'independent_validation.json',dict(all_passed=True,checks=checks,network_cases=9984,seconds=time.monotonic()-start,people_counted=False,new_hydraulics=False));print('INDEPENDENT VALIDATION',len(checks),'PASSED',flush=True)
if __name__=='__main__':main()
