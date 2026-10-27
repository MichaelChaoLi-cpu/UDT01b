"""Independent source-arc, fine-depth, paired-baseline and path qualification."""
from pathlib import Path
import os,sys,json,csv,collections,itertools,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_historical_surface_coverage_v2 import P,R,N,T,H,CASES,TIMES,load,read,write,dump,sha

def main():
 started=time.monotonic();checks=[]
 def check(name,ok):
  checks.append(dict(check=name,passed=bool(ok)));assert ok,name
 sources=json.loads((P/'source_bindings.json').read_text());check('all_frozen_sources_unchanged',all(sha(ROOT/p)==h for p,h in sources.items()));check('no_sealed_content_bound',not any('sealed_test-' in p for p in sources))
 contract=json.loads((P/'analysis_contract.json').read_text());check('excluded_tasks_not_executed',all(contract[k] is False for k in ['new_hydraulics','training','sealed_tests_opened','people_counted']))
 e=load(P/'surface_exposure.npz');w=load(P/'local_walking_graph.npz');a=load(P/'alternative_distances.npz');original=load(N/'walk_conservative_graph.npz');meta=read(P/'road_segment_inventory.csv');probes=read(P/'surface_probe_inventory.csv');contexts=read(P/'road_context_index.csv');supplement=load(P/'ground_path_supplement.npz');scenarios=read(P/'scenario_index.csv');old=load(T/'hypothetical_network.npz');edges=json.loads((T/'hypothetical_network_edges.json').read_text());sel=read(P/'station_closure_selection.csv')
 check('twelve_predeclared_synchronous_times',np.array_equal(e['times_s'],TIMES) and len(TIMES)==12 and np.max(abs(e['actual_times']-TIMES))<=.001)
 check('peak_frame_is_separate',str(e['frame_ids'][-1])=='peak_envelope' and all((r['simultaneous']=='True')==(r['frame_id']!='peak_envelope') for r in contexts+scenarios))
 si=w['source_arc_index'];gidx=w['source_global_index'];check('native_arc_permissions_directions_weights_unchanged',np.array_equal(gidx[w['from_index']],original['from_index'][si]) and np.array_equal(gidx[w['to_index']],original['to_index'][si]) and np.array_equal(w['length_m'],original['length_m'][si]) and np.array_equal(w['node_ids'],original['node_ids'][gidx]))
 check('all_probes_remain_same_native_nodes',all(str(w['node_ids'][w['probe_node_index'][j]])==p['source_walk_node'] for j,p in enumerate(probes)))
 check('exit_targets_are_source_native_outside_core_nodes',len(w['exit_node_index'])>0 and not w['inside_core'][w['exit_node_index']].any())
 nw=len(w['node_ids']);ni={str(k):i for i,k in enumerate(w['node_ids'])};oldkeys=list(map(str,old['node_ids']));oi={k:i for i,k in enumerate(oldkeys)};ri=np.array([i for i,x in enumerate(edges) if x['edge_kind'] not in ['initial_walk','final_walk']]);cu=[];cv=[]
 for r in read(T/'portal_inventory.csv'):
  for k in json.loads(r['walk_node_ids']):
   if k in ni:cu.extend([ni[k],nw+oi['P:'+r['portal_id']]]);cv.extend([nw+oi['P:'+r['portal_id']],ni[k]])
 uu=np.r_[w['from_index'],old['from_index'][ri]+nw,cu];vv=np.r_[w['to_index'],old['to_index'][ri]+nw,cv];ll=np.r_[w['length_m'],np.zeros(len(ri)+len(cu))];ne=len(w['from_index']);nn=nw+len(oldkeys)
 # Independent minimum-weight parallel-edge reducer; explicit zero links retained.
 order=np.lexsort((vv,uu));us=uu[order];vs=vv[order];groups=np.r_[0,np.flatnonzero((us[1:]!=us[:-1])|(vs[1:]!=vs[:-1]))+1];gu=us[groups];gv=vs[groups]
 def mixed(keep):
  d=np.minimum.reduceat(np.where(keep[order],ll[order],np.inf),groups);ok=np.isfinite(d);return csr_matrix((d[ok],(gu[ok],gv[ok])),shape=(nn,nn))
 wo=np.lexsort((w['to_index'],w['from_index']));wu=w['from_index'][wo];wv=w['to_index'][wo];wg=np.r_[0,np.flatnonzero((wu[1:]!=wu[:-1])|(wv[1:]!=wv[:-1]))+1]
 def walk(keep):
  d=np.minimum.reduceat(np.where(keep[wo],w['length_m'][wo],np.inf),wg);ok=np.isfinite(d);return csr_matrix((d[ok],(wu[wg][ok],wv[wg][ok])),shape=(nw,nw))
 anchors=load(T/'paired_accessibility_matrices.npz')['anchor_station_ids'];ad=[nw+oi['D:station:'+str(s)] for s in anchors];ao=[nw+oi['O:station:'+str(s)] for s in anchors];pn=w['probe_node_index']
 def exitdist(g):return np.stack([dijkstra(g.T.tocsr(),indices=w['exit_node_index'],directed=True,min_only=True),dijkstra(g,indices=w['exit_node_index'],directed=True,min_only=True)])[:,pn]
 def raildist(g):return np.stack([dijkstra(g.T.tocsr(),indices=ad,directed=True,min_only=True),dijkstra(g,indices=ao,directed=True,min_only=True)])[:,pn]
 def same(x,y):return np.array_equal(np.isfinite(x),np.isfinite(y)) and np.allclose(x[np.isfinite(x)],y[np.isfinite(y)],atol=1e-7,rtol=1e-10)
 # Fine-depth check at deterministic distributed cells, independent of cumulative integration.
 chosen=np.unique(np.r_[e['road_cells'][np.linspace(0,len(e['road_cells'])-1,160,dtype=int)],e['ground_cells'][np.linspace(0,len(e['ground_cells'])-1,64,dtype=int)]])
 allindex={int(c):j for j,c in enumerate(e['sample_cells'])};roadindex={int(c):j for j,c in enumerate(e['road_cells'])};groundindex={int(c):j for j,c in enumerate(e['ground_cells'])};depthok=True
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cls:
  for cell in chosen:
   r,c=divmod(int(cell),770);zz=dem.read(1,window=Window(c*20,r*20,20,20));cl=cls.read(1,window=Window(c*20,r*20,20,20));k=allindex[int(cell)]
   for name,dic,mask in [('road',roadindex,np.isfinite(zz)&(cl==2)),('ground',groundindex,np.isfinite(zz)&(cl!=1))]:
    if int(cell) not in dic:continue
    j=dic[int(cell)];z=zz[mask]
    if not len(z):continue
    for case in range(2):
     for t in range(12):
      eta=e['source_eta'][case,t,k];bed=e['source_bed_minimum'][case,k];lo=float(np.maximum(eta-z.astype('f8'),0).mean()) if np.isfinite(eta) else 0.;up=lo if np.isfinite(eta) else float(np.maximum(bed+.01-z.astype('f8'),0).mean())
      x=e[name+'_lower'][case,t,j];y=e[name+'_upper'][case,t,j]
      if np.isfinite(x):depthok &= abs(x-lo)<1e-7 and abs(y-up)<1e-7
 check('independent_fine_elevation_snapshot_depth_integrals',depthok)
 # Verify sampled source eta directly against raw NetCDF at a fixed state.
 etacheck=True
 for ci,case in enumerate(CASES):
  clock=load(H/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];ix=int(np.argmin(abs(clock-14400)))
  with Dataset(H/'runs'/case/'sfincs_map.nc') as d:z=np.flipud(np.ma.asarray(d['zs'][ix],dtype='f8').filled(np.nan)).ravel()[e['sample_cells']]
  etacheck &= np.allclose(z,e['source_eta'][ci,TIMES.index(14400)],equal_nan=True)
 check('raw_19jst_netcdf_matches_sampled_actual_state',etacheck)
 offsets=w['edge_sample_offsets'];ind=w['edge_sample_road_index'];lens=np.diff(offsets);grade=np.array([x['known_surface_grade']=='True' and x['ambiguous_parallel_geometry']=='False' for x in meta]);known=e['road_known'];reduced=np.logical_and.reduceat(np.r_[known[ind],True],offsets[:-1]);unknown=~grade|np.array([x['has_out_of_grid_sample']=='True' for x in meta])|(lens==0)|~reduced
 check('original_unknown_mask_preserved_before_supplement',np.array_equal(unknown,supplement['old_unknown']))
 check('only_unknown_exact_ground_segments_supplemented',not np.any(supplement['qualified']&~unknown) and all(meta[i]['grade_key']=='0|0|surface' and meta[i]['ambiguous_parallel_geometry']=='False' and meta[i]['has_out_of_grid_sample']=='False' for i in np.flatnonzero(supplement['qualified'])))
 unknown &= ~supplement['qualified']
 check('unknown_hazard_mask_independently_reconstructed',np.array_equal(unknown,a['road_unknown']))
 roadok=True;walkok=True;walkcache={};contextkeep=[]
 for i,r in enumerate(contexts):
  ci=CASES.index(r['case_id']);t=list(map(str,e['frame_ids'])).index(r['frame_id']);bound='lower' if r['unknown_branch']=='optimistic' else 'upper';values=e['road_'+bound][ci,t];samples=np.where(known[ind],values[ind],-np.inf);mx=np.maximum.reduceat(np.r_[samples,-np.inf],offsets[:-1]);mx[lens==0]=-np.inf;wet=(grade&(mx>=float(r['road_threshold_m'])))|(supplement['qualified']&(supplement[bound][ci,t]>=float(r['road_threshold_m'])));closed=wet|(unknown if r['unknown_branch']=='pessimistic' else False);roadok &= np.array_equal(closed,a['road_closed'][i]);keep=~closed[w['arc_basis']];contextkeep.append(keep);sig=keep.tobytes()
  if sig not in walkcache:walkcache[sig]=exitdist(walk(keep))
  walkok &= same(walkcache[sig],a['walking_exit_m'][i])
 check('all104_road_masks_and_grade_withholding_reproduced',roadok);check('all104_surface_exit_distances_recomputed',walkok)
 for b,branch in enumerate(['optimistic','pessimistic']):
  keep=~unknown[w['arc_basis']] if branch=='pessimistic' else np.ones(ne,bool);check('paired_'+branch+'_normal_exit_paths',same(exitdist(walk(keep)),a['baseline_walking_exit_m'][b]));check('paired_'+branch+'_normal_rail_paths',same(raildist(mixed(np.r_[keep,np.ones(len(ri)+len(cu),bool)])),a['baseline_mixed_min_walking_m'][b]))
 baseline=np.isfinite(a['baseline_walking_exit_m'][:,0])|np.isfinite(a['baseline_mixed_min_walking_m'][:,0]);check('normal_unknown_gaps_not_new_storm_losses',np.array_equal(baseline,a['baseline_possible']))
 selections={(x['case_id'],x['frame_id'],x['unknown_branch'],x['identity_policy']):x for x in sel};railcache={};railok=True;longer=True;nested={}
 for i,r in enumerate(scenarios):
  s=selections[r['case_id'],r['frame_id'],r['unknown_branch'],r['identity_policy']];ports=set(filter(None,s['selected_portal_ids'].split(';')));targets=set(filter(None,s['target_station_ids'].split(';')));kind=r['shutdown_type'];remove=[]
  for k in ri:
   x=edges[k];cut=x['edge_kind'] in ['station_entry','station_exit'] and x['portal_id'] in ports
   if kind!='entrance':cut |= (x['edge_kind'] in ['station_entry','station_exit','board','alight'] and x['station_id'] in targets) or (x['edge_kind']=='transfer' and bool(targets.intersection(x.get('transfer_units',[]))))
   if kind=='section':cut |= x['edge_kind']=='rail_service' and bool(targets.intersection(x['section_exposure_station_ids']))
   remove.append(cut)
  ci=int(r['road_context_index']);keep=np.r_[contextkeep[ci],~np.array(remove,bool),np.ones(len(cu),bool)];sig=keep.tobytes()
  if sig not in railcache:railcache[sig]=raildist(mixed(keep))
  railok &= same(railcache[sig],a['mixed_min_walking_m'][i]);b=0 if r['unknown_branch']=='optimistic' else 1;before=a['baseline_mixed_min_walking_m'][b];after=a['mixed_min_walking_m'][i];finite=np.isfinite(after);longer &= np.all(after[finite]>=before[finite]-1e-6)
  nested[(r['case_id'],r['frame_id'],float(r['road_threshold_m']),r['unknown_branch'],r['identity_policy'],kind)]=keep
  if i%120==0:print('VALIDATE PATHS',i,'of',len(scenarios),flush=True)
 check('all624_mixed_graph_shortest_walking_distances_recomputed',railok);check('paired_closures_never_improve_shortest_walking_distance',longer)
 nestedok=True
 for c,f,th,br,pol in itertools.product(CASES,map(str,e['frame_ids']),[.1,.3],['optimistic','pessimistic'],['original_identity','source_normalized_identity']):
  k=[nested[c,f,th,br,pol,kind] for kind in ['entrance','station','section']];nestedok &= not np.any(k[1]&~k[0]) and not np.any(k[2]&~k[1])
 check('three_shutdown_types_retain_nested_arc_sets',nestedok)
 # Reproduce source-supported closure selections for each actual frame.
 links={p:collections.defaultdict(set) for p in ['original_identity','source_normalized_identity']}
 for x in read(ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv'):
  for p in links:links[p][x['osm_node_id']].add(x['station_id'])
 for x in read(T/'identity_attribution_sensitivity_links.csv'):links['source_normalized_identity'][x['osm_node_id']].add(x['station_id'])
 portalids={x['portal_id'] for x in read(T/'portal_inventory.csv')};so=e['entrance_offsets'];gi=e['entrance_cell_index'];area=e['ground_area'];gk=e['ground_known'];selectok=True
 for s in sel:
  c=CASES.index(s['case_id']);t=list(map(str,e['frame_ids'])).index(s['frame_id']);bound='lower' if s['unknown_branch']=='optimistic' else 'upper';gd=e['ground_'+bound][c,t];ports=set();targets=set()
  for j,osm in enumerate(map(str,e['entrance_ids'])):
   ii=gi[so[j]:so[j+1]];total=area[ii].sum();admitted_area=area[ii][gk[ii]].sum();wet=area[ii][gk[ii]&(gd[ii]>=.3)].sum()
   if total<=0 or admitted_area<100 or admitted_area/total<.8 or wet<100:continue
   if links[s['identity_policy']][osm] and 'osm:'+osm in portalids:ports.add('osm:'+osm);targets.update(links[s['identity_policy']][osm])
  selectok &= ports==set(filter(None,s['selected_portal_ids'].split(';'))) and targets==set(filter(None,s['target_station_ids'].split(';')))
 check('all_frame_entrance_closures_reproduce_frozen_quality_and_identity',selectok)
 oldsel=read(ROOT/'data/processed/three_ward_historical_transport_islands_v1/closure_selection.csv');peakok=True
 for s in sel:
  if s['frame_id']!='peak_envelope':continue
  bound='lower' if s['unknown_branch']=='optimistic' else 'upper';prior=next(x for x in oldsel if x['case_id']==s['case_id'] and x['identity_policy']==s['identity_policy'] and x['depth_bound']==bound);peakok &= s['selected_portal_ids']==prior['selected_portal_ids'] and s['target_station_ids']==prior['target_station_ids']
 check('prior_peak_rail_exposure_and_station_targets_unchanged',peakok)
 incident=np.zeros(nw,bool)
 for x,y,b in zip(w['from_index'],w['to_index'],w['arc_basis']):
  if not unknown[b]:incident[x]=True;incident[y]=True
 eligible=baseline.all(axis=0)&incident[pn];check('strict_classification_baseline_and_projection_mask',np.array_equal(eligible,a['evidence_eligible']))
 lookup=collections.defaultdict(list)
 for i,r in enumerate(scenarios):lookup[r['case_id'],r['frame_id'],r['shutdown_type']].append(i)
 classok=True;classrows=read(R/'probe_classification.csv');probeindex={p['probe_id']:j for j,p in enumerate(probes)}
 for r in classrows:
  ii=lookup[r['case_id'],r['frame_id'],r['shutdown_type']];ci=[int(scenarios[i]['road_context_index']) for i in ii];j=probeindex[r['probe_id']];available=np.isfinite(a['walking_exit_m'][ci,0,j])|np.isfinite(a['mixed_min_walking_m'][ii,0,j]);label='insufficient_evidence' if not eligible[j] else 'robust_modeled_disconnection' if not available.any() else 'stable_modeled_alternative' if available.all() else 'conditional_sensitivity'
  classok &= r['classification']==label and int(r['branches_with_outward_alternative'])==int(available.sum())
 check('all_probe_robust_sensitive_unknown_classes_recomputed',classok)
 aggok=True
 for r in read(R/'classification_summary.csv'):
  rr=[x for x in classrows if x['case_id']==r['case_id'] and x['frame_id']==r['frame_id'] and x['shutdown_type']==r['shutdown_type'] and (r['ward']=='all' or x['ward']==r['ward'])];counts=collections.Counter(x['classification'] for x in rr);aggok &= len(rr)==int(r['origin_probes']) and all(counts[k]==int(r[k]) for k in ['robust_modeled_disconnection','conditional_sensitivity','insufficient_evidence','stable_modeled_alternative'])
 check('all_ward_classification_counts_recomputed',aggok)
 statok=True
 for r in read(R/'scenario_accessibility_summary.csv'):
  i=int(r['scenario_index']);ci=int(r['road_context_index']);b=0 if r['unknown_branch']=='optimistic' else 1;alt=np.isfinite(a['walking_exit_m'][ci,0])|np.isfinite(a['mixed_min_walking_m'][i,0]);statok &= int(r['new_no_outward_alternative'])==int((baseline[b]&~alt).sum()) and int(r['baseline_connected'])==int(baseline[b].sum())
 check('all_scenario_storm_losses_use_paired_baseline',statok)
 # Independently reconstruct the supplementary depth maxima from original water-level output.
 qualified=supplement['qualified'];sp=supplement['sample_point_index'];off=supplement['sample_offsets'];coarse=supplement['coarse_flat_index'];fine=supplement['fine_terrain_z'];finite_points=np.isfinite(fine)&np.isin(supplement['fine_class'],[0,2])&supplement['point_admitted'];qpoints=np.zeros(len(fine),bool)
 for j in np.flatnonzero(qualified):qpoints[sp[off[j]:off[j+1]]]=True
 check('all_qualified_samples_have_open_finite_admitted_ground',finite_points[qpoints].all())
 check('fine_to_coarse_mapping_and_common_admission_unchanged',np.array_equal(coarse,(supplement['fine_pixel_ids']//15400//20)*770+(supplement['fine_pixel_ids']%15400//20)) and np.array_equal(supplement['point_admitted'],load(H/'paired_exposure_bounds.npz')['common_admitted'].ravel()[coarse]))
 from shapely import STRtree
 from shapely.geometry import shape,LineString
 from shapely.ops import transform
 from pyproj import Transformer
 import gzip
 water=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features'];project=Transformer.from_crs(4326,6677,always_xy=True);tree=STRtree([transform(project.transform,shape(f['geometry'])) for f in water]);needed={meta[j]['source_edge_id'] for j in np.flatnonzero(qualified)};seen=set();sourceok=True
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt') as f:
  for line in f:
   source=json.loads(line)
   if source['edge_id'] not in needed:continue
   seen.add(source['edge_id']);tags=source['tags'];sourceok &= source['grade_key']=='0|0|surface' and tags.get('highway') not in ['steps','elevator','corridor'] and tags.get('indoor')!='yes' and tags.get('covered') in [None,'no','false','0'] and tags.get('location') not in ['underground','underwater','rooftop'] and tags.get('bridge') in [None,'no','false','0'] and tags.get('tunnel') in [None,'no','false','0'] and len(tree.query(LineString(source['geometry']['coordinates']),predicate='intersects'))==0
 check('all_supplemented_source_geometries_are_ground_uncovered_nonwater',sourceok and seen==needed)

 depthsok=True
 for c,case in enumerate(CASES):
  clock=load(H/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];ii=[int(np.argmin(abs(clock-t))) for t in TIMES]
  with Dataset(H/'runs'/case/'sfincs_map.nc') as d:
   bed=np.flipud(np.ma.asarray(d['zb'][:],dtype='f8').filled(np.nan)).ravel()[coarse]
   for t,k in enumerate(ii+[None]):
    water=np.flipud(np.ma.asarray(d['zs'][k] if k is not None else d['zsmax'][0],dtype='f8').filled(np.nan)).ravel()[coarse];lo=np.where(np.isfinite(water),np.maximum(water-fine,0),0);up=np.where(np.isfinite(water),lo,np.maximum(bed+.01-fine,0))
    for bound,value in [('lower',lo),('upper',up)]:
     maximum=np.maximum.reduceat(np.r_[value[sp],-np.inf],off[:-1]);depthsok &= np.allclose(maximum[qualified],supplement[bound][c,t,qualified],atol=1e-7)
 check('all_supplementary_path_depth_maxima_reconstructed_from_raw_outputs',depthsok)
 # Check fine data independently at distributed used pixels.
 fineok=True;pick=np.flatnonzero(qpoints);pick=pick[np.linspace(0,len(pick)-1,min(len(pick),128),dtype=int)]
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cls:
  for j in pick:
   rr,cc=divmod(int(supplement['fine_pixel_ids'][j]),15400);z=dem.read(1,window=Window(cc,rr,1,1))[0,0];cl=cls.read(1,window=Window(cc,rr,1,1))[0,0];fineok &= z==fine[j] and cl==supplement['fine_class'][j]
 check('supplement_fine_samples_match_original_dem_and_masks',fineok)
 check('sources_still_unchanged_after_validation',all(sha(ROOT/p)==h for p,h in sources.items()))
 dump(R/'independent_validation.json',dict(all_passed=True,checks=checks,seconds=time.monotonic()-started,independently_recomputed_mixed_graphs=len(railcache),independently_recomputed_walk_graphs=len(walkcache),new_hydraulics=False,people_counted=False,sealed_tests_opened=False))
 print('VALIDATION DONE',len(checks),'checks',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
