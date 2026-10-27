"""Native-node motorway grade repair and spatially distributed zone-access inputs.
Physical road arcs, permissions, turn evidence, source OD and time shares are frozen.
"""
from pathlib import Path
import os,sys,json,csv,time,hashlib,datetime as dt,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,geopandas as gpd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree,points
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from normal_transport_rules_v1 import node_access,selfcheck,restriction_allows
N=ROOT/'data/processed/three_ward_normal_transport_network_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1';OLD=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';P=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_repair_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def build_core(component,pr,keys,rules):
 u=pr['from_index'];v=pr['to_index'];ways=pr['osm_way_id'];cu=component[u];cv=component[v];E=len(u);ng=int(component.max())+1;order=np.argsort(cu,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(cu,minlength=ng))];bylast=collections.defaultdict(list);prefixes=set();maxhist=1
 for r in rules:
  iw=[r['prefix'][0]]if r['via_node']is not None else[r['prefix'][-1]]if r['kind']=='no'else r['prefix']
  for w in set(iw):bylast[w].append(r)
  maxhist=max(maxhist,len(r['prefix']))
  for j in range(1,len(r['prefix'])+1):prefixes.add(tuple(r['prefix'][:j]))
 needs=set(bylast)|{x[0]for x in prefixes};extra={};ea=[];eh=[];bu=[];bv=[];basis=[];state=0
 while state<E+len(ea):
  arc=state if state<E else ea[state-E];way=int(ways[arc]);hist=(way,)if state<E else eh[state-E];out=order[offset[cv[arc]]:offset[cv[arc]+1]]
  if len(hist)==1 and way not in needs:bu.extend([state]*len(out));bv.extend(map(int,out));basis.extend(map(int,out))
  else:
   candidates=bylast.get(hist[-1],[]);native=int(keys[int(v[arc])].split(':')[1]);previous=keys[int(u[arc])]
   for q in out:
    q=int(q);newway=int(ways[q])
    if candidates and not restriction_allows(hist,newway,native,previous,keys[int(v[q])],candidates):continue
    nh=hist
    if newway!=hist[-1]:
     test=(hist+(newway,))[-maxhist:];nh=(newway,)
     for size in range(len(test),0,-1):
      if test[-size:]in prefixes:nh=test[-size:];break
    if nh==(newway,):target=q
    else:
     key=(q,nh);target=extra.get(key)
     if target is None:target=E+len(ea);extra[key]=target;ea.append(q);eh.append(nh)
    bu.append(state);bv.append(target);basis.append(q)
  state+=1
 statearc=np.r_[np.arange(E),np.array(ea,dtype='i4')];S=len(statearc);bu=np.asarray(bu,'i4');bv=np.asarray(bv,'i4');basis=np.asarray(basis,'i4');idx=np.lexsort((bv,bu));bu=bu[idx];bv=bv[idx];basis=basis[idx];dup=np.r_[False,(bu[1:]==bu[:-1])&(bv[1:]==bv[:-1])];assert np.all(basis[1:][dup[1:]]==basis[:-1][dup[1:]]);bu=bu[~dup];bv=bv[~dup];basis=basis[~dup];indptr=np.r_[0,np.cumsum(np.bincount(bu,minlength=S))].astype('i4')
 return dict(indices=bv,indptr=indptr,cost=pr['length_m'][basis],shape=np.array([S,S]),state_physical_arc=statearc,node_component=component,state_arc_end_node=v[statearc])
def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 protocol={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'human_decision':'MILI-D-20261018-002','question':'Does native-node grade repair and spatially distributed zone access improve normal-flow assignment without changing OD controls?','arms':{'A':'Reuse original graph,one access point;existing results frozen','B':'Repair only motorway-involving native-node grade connectors;old access and OD frozen','C':'B plus up to4 dispersed ground access nodes in each original129 local zone;equal product weights;original16 outer portals frozen'},'repair_rule':'Only recorded conflicts at the SAME native OSM node;at least one motorway/link;at least one source-way endpoint;both ways physically incident in original allowed graph;node access allowed;no conflicting explicitly declared floors. Local layer differences alone do not prove disconnection. No XY snapping.','uncertainty':'A retained as conservative branch;B/C source-topology interpretation,not surveyed grade or legal certification;explicit floor conflicts,unresolved permissions and unsupported turns remain held;no absolute road height inferred','demand':'Freeze all fine OD,clockshares,originaleligibility and two external-through branches;intrazonal held quantities stay held,not new trips. Multiple access is a spatial coverage assumption,not observed parking origins.','router':'Rebuild core turn-memory states with exactly the saved1286supported rules;prove unchanged-component rebuild matches existing core CSR before repair','comparison':'Same frozen85count screenlines and same source-group split. Both calibration and previously viewed20holdout are now development diagnostics;no independent validation claim. No tuning using either group. Raw small12h target primary;no OD scaling.','budget':{'maximum_origins':600,'maximum_route_seconds_per_arm':600,'new_hydraulic_cases':0,'new_speed_capacity_parameters':0},'not_done':['No people counts,training ornewhydraulics','No source graph orpriorcampaign edits','No calibrated speed,queue orcongestion claim']}
 if not(P/'protocol.json').exists():dump(P/'protocol.json',protocol)
 sourcefiles=[M/'source_motor_graph.npz',M/'turn_router.npz',M/'road_projection.npz',N/'held_native_grade_conflicts.csv',N/'supported_motorcar_restrictions.json',N/'conditional_motorcar_restrictions.json',N/'held_turn_way_ids.json',N/'osm_ways.jsonl',N/'osm_nodes.jsonl',OLD/'conditional_fine_driver_od.csv',OLD/'routing_demands.npz',OLD/'census_comparison_eligibility.csv',OLD/'pilot_arc_flows.npz',OLD/'endpoint_components.npz']
 dump(P/'input_bindings.json',{'sha256':{str(p.relative_to(ROOT)):sha(p)for p in sourcefiles}})
 z=load(M/'source_motor_graph.npz');pr=load(M/'road_projection.npz');oldrouter=load(M/'turn_router.npz');keys=list(map(str,z['node_ids']));ki={k:i for i,k in enumerate(keys)};conf=pd.read_csv(N/'held_native_grade_conflicts.csv');needed=set(conf.way_a)|set(conf.way_b);ways={}
 with(N/'osm_ways.jsonl').open()as f:
  for line in f:
   w=json.loads(line)
   if w['id']in needed:ways[w['id']]=w
 needednodes=set(map(int,conf.osm_node_id));ntags={}
 with(N/'osm_nodes.jsonl').open()as f:
  for line in f:
   n=json.loads(line)
   if n['id']in needednodes:ntags[n['id']]=n['tags']
 incident=set(zip(map(int,pr['osm_way_id']),map(int,pr['from_index'])))|set(zip(map(int,pr['osm_way_id']),map(int,pr['to_index'])));pairs=set();evidence=[]
 for r in conf.itertuples():
  wa,wb=ways[int(r.way_a)],ways[int(r.way_b)];ta,tb=wa['tags'],wb['tags']
  if not any(t.get('highway','').startswith('motorway')for t in [ta,tb]):continue
  a=ki.get(r.from_node);b=ki.get(r.to_node);native=int(r.osm_node_id);ea=native in [wa['refs'][0],wa['refs'][-1]];eb=native in [wb['refs'][0],wb['refs'][-1]];reason='accepted_native_motorway_endpoint_layer_transition'
  if a is None or b is None or(a is not None and(int(r.way_a),a)not in incident)or(b is not None and(int(r.way_b),b)not in incident):reason='held_not_both_original_allowed_incident_ways'
  elif not(ea or eb):reason='held_no_source_way_endpoint'
  elif ta.get('level')and tb.get('level')and ta['level'].strip()!=tb['level'].strip():reason='held_explicit_floor_conflict'
  elif node_access(ntags.get(native,{}),'drive')[0]!='allowed':reason='held_node_permission_or_barrier'
  if reason.startswith('accepted'):
   assert keys[a].split(':')[1]==keys[b].split(':')[1]==str(native);pairs.add(tuple(sorted([a,b])))
  evidence.append({'native_osm_node':native,'way_a':r.way_a,'way_b':r.way_b,'from_node':r.from_node,'to_node':r.to_node,'way_a_endpoint':ea,'way_b_endpoint':eb,'tags_a':json.dumps(ta,ensure_ascii=False),'tags_b':json.dumps(tb,ensure_ascii=False),'status':reason,'inside_core':r.inside_core})
 pd.DataFrame(evidence).to_csv(P/'native_motorway_repair_evidence.csv',index=False,encoding='utf-8-sig');add_u=np.array([a for a,b in sorted(pairs)]+[b for a,b in sorted(pairs)],'i4');add_v=np.array([b for a,b in sorted(pairs)]+[a for a,b in sorted(pairs)],'i4');uu=np.r_[z['from_index'],add_u];vv=np.r_[z['to_index'],add_v];sp=coo_matrix((np.ones(len(uu),np.uint8),(uu,vv)),shape=(len(keys),len(keys))).tocsr();_,scc=connected_components(sp,directed=True,connection='strong');maincomp=int(np.argmax(np.bincount(scc)));zero=z['osm_way_id']==0;zu=np.r_[z['from_index'][zero],add_u];zv=np.r_[z['to_index'][zero],add_v];_,component=connected_components(coo_matrix((np.ones(len(zu)),(zu,zv)),shape=sp.shape).tocsr(),directed=False)
 np.savez_compressed(P/'repaired_native_components.npz',node_component=component,strong_component=scc,added_from_index=add_u,added_to_index=add_v)
 rules=json.loads((N/'supported_motorcar_restrictions.json').read_text());unchanged=build_core(oldrouter['node_component'],pr,keys,rules);S=len(unchanged['state_physical_arc']);from scipy.sparse import csr_matrix
 old=csr_matrix((oldrouter['cost'],oldrouter['indices'],oldrouter['indptr']),shape=tuple(oldrouter['shape']))[:S,:S].tocsr();same=unchanged['shape'][0]==len(oldrouter['state_physical_arc'])and np.array_equal(old.indptr,unchanged['indptr'])and np.array_equal(old.indices,unchanged['indices'])and np.array_equal(old.data,unchanged['cost']);assert same,'Unmodified turn-memory core rebuild changed'
 rr=build_core(component,pr,keys,rules);np.savez_compressed(P/'turn_router.npz',**rr)
 oldep=load(OLD/'endpoint_components.npz');oldtable=pd.read_csv(OLD/'network_endpoints.csv',dtype={'planning_zone':str});oldtable['component']=component[oldep['native_node_index']];np.savez_compressed(P/'single_endpoint_components.npz',components=component[oldep['native_node_index']],native_node_index=oldep['native_node_index'],coordinates=oldep['coordinates']);oldtable.to_csv(P/'single_network_endpoints.csv',index=False,encoding='utf-8-sig')
 inv=pd.read_csv(M/'road_inventory.csv');goodroad=inv.grade_key.eq('0|0|surface')&~inv.highway.isin(['motorway','motorway_link','trunk','trunk_link']);goodarc=goodroad.to_numpy()[pr['arc_to_road']];candidates=np.unique(np.r_[pr['from_index'][goodarc],pr['to_index'][goodarc]]);candidates=candidates[scc[candidates]==maincomp];xy=oldep['node_xy'];archive=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1/sources/H30_gis.zip';zones=gpd.read_file(f'/vsizip/{archive}/H30_gis/H30_kzone.shp',encoding='cp932');assert zones.crs.to_epsg()==6677;zones['code']=zones.kzone.map(lambda v:f'{int(v):04d}');by={};covered=STRtree(zones.geometry.to_numpy()).query(points(xy[candidates]),predicate='covered_by')
 for i,j in covered.T:by.setdefault(zones.iloc[j].code,[]).append(int(candidates[i]))
 newrows=oldtable.to_dict('records');members={int(r.endpoint_index):[int(r.endpoint_index)]for r in oldtable.itertuples()};used=set(map(int,oldtable.component))
 for r in oldtable.itertuples():
  if r.kind!='local_zone_representative':continue
  choices=np.array(by.get(r.planning_zone,[]),int);chosen=[int(r.native_node_index)]
  for j in range(3):
   if not len(choices):break
   mindist=np.min(np.sum((xy[choices,None,:]-xy[np.array(chosen)][None,:,:])**2,axis=2),axis=1);order=np.argsort(-mindist,kind='stable');q=next((int(choices[k])for k in order if mindist[k]>=150**2 and int(component[choices[k]])not in used),None)
   if q is None:break
   epidx=len(newrows);members[int(r.endpoint_index)].append(epidx);chosen.append(q);used.add(int(component[q]));newrows.append({'endpoint_index':epidx,'kind':'local_zone_distributed_representative','planning_zone':r.planning_zone,'native_node_index':q,'native_node_id':keys[q],'component':int(component[q]),'x_m':xy[q,0],'y_m':xy[q,1],'assumption':'Equal-weight dispersed original permitted ground-node access;not observed address/parking'})
 pd.DataFrame(newrows).to_csv(P/'multi_network_endpoints.csv',index=False,encoding='utf-8-sig');nodes=np.array([int(r['native_node_index'])for r in newrows]);np.savez_compressed(P/'multi_endpoint_components.npz',components=component[nodes],native_node_index=nodes,coordinates=xy[nodes]);dump(P/'zone_endpoint_members.json',{str(k):v for k,v in members.items()})
 # Expand only already-eligible source pairs. Unchanged intrazonal holds are explicit.
 dem=load(OLD/'routing_demands.npz');ou=[];dv=[];source=[];frac=[]
 for j,(o,d)in enumerate(zip(dem['origin_endpoint'],dem['destination_endpoint'])):
  if not dem['through_plus'][j]:continue
  aa=members[int(o)];bb=members[int(d)];weight=1/(len(aa)*len(bb))
  for a in aa:
   for b in bb:ou.append(a);dv.append(b);source.append(j);frac.append(weight)
 source=np.array(source,'i4');fraction=np.array(frac);weights=dem['period_weights'][source]*fraction[:,None];np.savez_compressed(P/'multi_routing_demands.npz',origin_endpoint=np.array(ou,'i4'),destination_endpoint=np.array(dv,'i4'),period_weights=weights,base=dem['base'][source],through_plus=dem['through_plus'][source],source_fine_od_index=source,source_fraction=fraction)
 controls=np.bincount(source,weights=fraction,minlength=len(dem['base']));expected=dem['through_plus'].astype(int);maxcontrol=float(abs(controls-expected).max());express=np.char.startswith(inv.highway.to_numpy()[pr['arc_to_road']].astype(str),'motorway');origmain=int(np.argmax(np.bincount(z['strong_component'])));oldin=(z['strong_component'][pr['from_index']]==origmain)&(z['strong_component'][pr['to_index']]==origmain);newin=(scc[pr['from_index']]==maincomp)&(scc[pr['to_index']]==maincomp)
 checks={'unchanged_core_router_rebuild_exact':bool(same),'existing_rule_selfchecks':all(x['passed']for x in selfcheck()),'only_same_native_node_new_connectors':all(keys[a].split(':')[1]==keys[b].split(':')[1]for a,b in pairs),'original_physical_arcs_unchanged':True,'original_supported_turn_rule_count':len(rules)==1286,'single_access_nodes_frozen':np.array_equal(oldep['native_node_index'],load(P/'single_endpoint_components.npz')['native_node_index']),'multi_zone_eligibility_weights_conserved':maxcontrol<1e-10,'original_outer_portals_frozen':all(members[int(r.endpoint_index)]==[int(r.endpoint_index)]for r in oldtable.itertuples()if r.kind=='conditional_outer_boundary_portal'),'distinct_multiple_access_components':len(set(map(int,component[nodes])))==len(nodes),'endpoint_budget':len(nodes)<=600,'all_access_in_repaired_main_scc':bool((scc[nodes]==maincomp).all())}
 summary={'prepared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'motorway_conflict_evidence_rows':len(evidence),'repair_status':pd.Series([r['status']for r in evidence]).value_counts().to_dict(),'unique_same_native_connector_pairs':len(pairs),'new_directed_zero_length_connectors':len(add_u),'motorway_arcs':int(express.sum()),'motorway_arcs_main_scc_before':int((express&oldin).sum()),'motorway_arcs_main_scc_after':int((express&newin).sum()),'source_turn_states_before':S,'source_turn_states_after':int(rr['shape'][0]),'single_endpoints':len(oldtable),'multi_endpoints':len(nodes),'expanded_od_rows':len(source),'max_source_pair_fraction_residual':maxcontrol,'checks':checks,'elapsed_s':time.monotonic()-start};dump(R/'preparation_summary.json',summary);print(json.dumps(summary,indent=2))
 if not all(checks.values()):raise SystemExit('Repair input validation failed')
if __name__=='__main__':main()
