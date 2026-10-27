"""Freeze native walking geometry and synchronized flood exposure; no solver calls."""
from pathlib import Path
import os,sys,json,csv,gzip,hashlib,math,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
N=ROOT/'data/processed/three_ward_normal_transport_network_v1';T=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1';H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
P=ROOT/'data/processed/three_ward_surface_alternatives_v1';R=ROOT/'data/exp/estimation/three_ward_surface_alternatives_v1'
TIMES=sorted(set(range(0,32401,3600))|{12600,16200});CASES=['radar_low','radar_raised']
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p) as f:return {k:f[k] for k in f.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def nodekey(way,a,b):return (int(way),min(int(a),int(b)),max(int(a),int(b)))
def sample(coords):
 q=np.asarray(coords,float);d=np.r_[0,np.cumsum(np.hypot(*np.diff(q,axis=0).T))];s=np.unique(np.r_[d,np.arange(0,d[-1],2.5)])
 x=np.interp(s,d,q[:,0]);y=np.interp(s,d,q[:,1]);rr=np.floor((-32320-y)/10).astype(int);cc=np.floor((x+11360)/10).astype(int);ok=(rr>=0)&(rr<1012)&(cc>=0)&(cc<770)
 return np.unique(rr[ok]*770+cc[ok]),bool((~ok).any())
@njit
def depth_at(sorted_z,cumulative,n,eta,zmin):
 nt,nc=eta.shape;lo=np.zeros((nt,nc));up=np.zeros((nt,nc))
 for t in range(nt):
  for c in range(nc):
   count=n[c]
   if not count:lo[t,c]=np.nan;up[t,c]=np.nan;continue
   zz=eta[t,c];dry=not np.isfinite(zz)
   if dry:zz=zmin[c]+.01
   a=0;b=count
   while a<b:
    mid=(a+b)//2
    if sorted_z[c,mid]<zz:a=mid+1
    else:b=mid
   v=max(0.,(a*zz-(cumulative[c,a-1] if a else 0.))/count)
   up[t,c]=v
   if not dry:lo[t,c]=v
 return lo,up

def main():
 started=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 policy={'human_decision':'MILI-D-20261017-002','scope':'Whole three wards; frozen native pedestrian graph clipped by retained source segments intersecting the hydraulic rectangle. Regional rail graph unchanged. No planar noding or nearest-station joins.',
 'times_s':TIMES,'time_basis':'Actual output clocks; exact selected output within0.001s. Twelve predeclared simultaneous states plus a separately labeled non-simultaneous event peak envelope. No duration or time-dependent journey inference.',
 'road_rule':'Sample original physical polylines every2.5m plus source vertices; map samples into10m cells; use mean depth over classified0.5m road ground inside that cell. Any sampled cell reaching declared depth closes the entire source segment as a hypothesis. Not observed road water depth or certified pedestrian safety.',
 'road_depth_thresholds_m':[.1,.3],'grade_rule':'Only exact0|0|surface geometry is linked to surface flood depth. All other levels, bridges, tunnels, connectors or ambiguous geometry retain unknown hazard; surface water is never assigned to their deck/tunnel height.',
 'uncertainty_branches':{'optimistic':'Close at lower-depth threshold; retain unknown exposure links. Original and normalized station identities both evaluated.','pessimistic':'Close at upper-depth threshold and withhold unknown-exposure links, including unsupported/out-of-domain source segments and relative-height crossings. Identity branches remain separate.'},
 'exit_definition':'Native walking endpoints outside the frozen three-ward union adjacent through a source arc to a node inside it; reaching an exit is crossing the analysis boundary, not arrival at a verified safe refuge. Flooding beyond this boundary is unassessed.',
 'rail_alternative':'Explicit native walking node to existing source-mapped portal links only; remove old bounded synthetic walking legs. All remaining station circulation, provider transfer and operating arcs retain declared assumptions. Destination set remains the six frozen external anchors.',
 'distance':'Unrestricted shortest modeled walking distance to any cross-core exit; shortest total walking distance within mixed walk+rail paths to any declared external anchor. Rail edges carry zero walking distance; neither output is travel time.2km is a display/analysis budget only, not a safety or feasibility standard.',
 'classification':'Within each case/time and shutdown type, robust modeled disconnection requires a supported normal baseline and failure in every declared road-threshold, identity and unknown-link branch of BOTH surface-exit and rail alternatives. Branch disagreements are conditional sensitivity; missing baseline/projection evidence is insufficient. Peak envelopes never prove simultaneous isolation.',
 'population':'Official block baseline and building uncertainty untouched. Only geometry names used for context. No population weighting or stranded-person counts.',
 'paired_baseline':'Each unknown-exposure branch has its own normal baseline with the same unknown-link availability, no flood closures and normal rail. Unknown-only baseline gaps never count as storm disconnection. Robust classification requires outward baseline support in both branches.',
 'batch_gates':['correct space/time linkage and no grade confusion','paired normal baseline and explicit gaps','surface and rail alternatives with distance and unknown support','synchronized states separated from peak envelopes','frozen rules, reproducibility and assumption-sensitive interpretation'],
 'new_hydraulics':False,'training':False,'sealed_tests_opened':False,'people_counted':False}
 dump(P/'analysis_contract.json',policy)
 paths=[N/n for n in ['walk_conservative_graph.npz','surface_nodes.csv','surface_physical_edges.jsonl.gz']]+[T/n for n in ['hypothetical_network.npz','hypothetical_network_edges.json','portal_inventory.csv','access_probes.csv','scenario_policy.json','identity_attribution_sensitivity_links.csv','source_bindings.json']]+[H/'paired_exposure_bounds.npz',H/'run_manifest.json',ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz',ROOT/'data/processed/three_ward_station_surface_risk_v1/entrance_screen_summary.csv',ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv',ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson',ROOT/'data/processed/three_ward_minimum_scenarios_v1/partition_manifest.json',ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/selection.json']
 paths += [ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv',ROOT/'data/processed/three_ward_historical_transport_islands_v1/closure_selection.csv']
 paths += [ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif',ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif']
 for case in CASES:paths += [H/'runs'/case/'sfincs_map.nc',H/'execution'/(case+'_output_clock.npz'),H/'execution'/(case+'_candidate_reference.npz')]
 sources={str(p.relative_to(ROOT)):sha(p) for p in paths};dump(P/'source_bindings.json',sources)
 w=load(N/'walk_conservative_graph.npz');ids=list(map(str,w['node_ids']));wi={k:i for i,k in enumerate(ids)};xy=np.full((len(ids),2),np.nan)
 with (N/'surface_nodes.csv').open() as f:
  for n in csv.DictReader(f):
   if n['node_id'] in wi:xy[wi[n['node_id']]]=[float(n['x_m']),float(n['y_m'])]
 assert np.isfinite(xy).all()
 # Retain source segments, including their genuine endpoints outside the rectangle.
 bases=[];keymap={};coords_by_edge=[]
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt') as f:
  for ln in f:
   e=json.loads(ln);aa=wi.get(e['from_node']);bb=wi.get(e['to_node'])
   if aa is None or bb is None:continue
   q=np.asarray(e['geometry']['coordinates']);mn=q.min(axis=0);mx=q.max(axis=0)
   if mx[0]<-11360 or mn[0]>-3660 or mx[1]<-42440 or mn[1]>-32320:continue
   k=nodekey(e['osm_way_id'],aa,bb)
   if k in keymap:
    # Ambiguous parallel segment support is withheld rather than silently merged.
    bases[keymap[k]]['ambiguous_parallel_geometry']=True;continue
   keymap[k]=len(bases);bases.append(dict(edge_index=len(bases),source_edge_id=e['edge_id'],osm_way_id=e['osm_way_id'],from_global=aa,to_global=bb,grade_key=e['grade_key'],known_surface_grade=e['grade_key']=='0|0|surface',ambiguous_parallel_geometry=False,source_length_m=float(e['length_m'])));coords_by_edge.append(e['geometry']['coordinates'])
 insidebox=(xy[:,0]>=-11360)&(xy[:,0]<=-3660)&(xy[:,1]>=-42440)&(xy[:,1]<=-32320)
 u=w['from_index'];v=w['to_index'];way=w['osm_way_id'];arc_basis=np.full(len(u),-1,np.int32)
 for j in range(len(u)):
  k=nodekey(way[j],u[j],v[j])
  if k not in keymap and way[j]==0 and (insidebox[u[j]] or insidebox[v[j]]):
   keymap[k]=len(bases);bases.append(dict(edge_index=len(bases),source_edge_id='native-level:'+str(len(bases)),osm_way_id=0,from_global=int(u[j]),to_global=int(v[j]),grade_key='native_level_transition',known_surface_grade=False,ambiguous_parallel_geometry=False,source_length_m=float(w['length_m'][j])));coords_by_edge.append(xy[[u[j],v[j]]].tolist())
  if k in keymap:arc_basis[j]=keymap[k]
 keep=arc_basis>=0;used=np.unique(arc_basis[keep]);remap=np.full(len(bases),-1,int);remap[used]=np.arange(len(used));bases=[bases[i] for i in used];coords_by_edge=[coords_by_edge[i] for i in used];arc_basis=remap[arc_basis[keep]].astype('i4')
 probes=[p for p in read(T/'access_probes.csv') if p['cohort']=='surface_grid'];global_probes=[wi[p['source_walk_node']] for p in probes]
 active_nodes=np.unique(np.r_[u[keep],v[keep],global_probes]);local=np.full(len(ids),-1,np.int32);local[active_nodes]=np.arange(len(active_nodes));uu=local[u[keep]];vv=local[v[keep]]
 # Source-native crossing endpoints; no new link is created by proximity.
 inc=w['inside_core'][active_nodes];exitnodes=np.unique(np.r_[vv[inc[uu]&~inc[vv]],uu[~inc[uu]&inc[vv]]]);assert len(exitnodes)
 lists=[];offsets=[0];metadata=[]
 for i,(e,q) in enumerate(zip(bases,coords_by_edge)):
  cells,out=sample(q);lists.append(cells);offsets.append(offsets[-1]+len(cells));metadata.append({k:v for k,v in e.items() if k not in ['from_global','to_global']}|dict(edge_index=i,from_node=ids[e['from_global']],to_node=ids[e['to_global']],sampled_cells=len(cells),has_out_of_grid_sample=out))
 allcells=np.concatenate(lists);roadcells=np.unique(allcells);rindex=np.searchsorted(roadcells,allcells).astype('i4')
 entrance=read(ROOT/'data/processed/three_ward_station_surface_risk_v1/entrance_screen_summary.csv');s=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=s['Eligible Ground Mask'];gs=[];goff=[0]
 for e in entrance:
  x,y=float(e['x_m']),float(e['y_m']);r=math.floor((-32320-y)/10);c=math.floor((x+11360)/10);r0,r1=max(0,r-3),min(1012,r+4);c0,c1=max(0,c-3),min(770,c+4)
  if r0<r1 and c0<c1:
   rr,cc=np.mgrid[r0:r1,c0:c1];ok=((-11360+(cc+.5)*10-x)**2+(-32320-(rr+.5)*10-y)**2<=400)&valid[rr,cc];cells=(rr[ok]*770+cc[ok]).astype(int)
  else:cells=np.array([],int)
  gs.append(cells);goff.append(goff[-1]+len(cells))
 gflat=np.concatenate(gs);groundcells=np.unique(gflat);gindex=np.searchsorted(groundcells,gflat).astype('i4');both=np.union1d(roadcells,groundcells)
 print('GEOMETRY',len(active_nodes),'nodes',len(uu),'arcs',len(bases),'segments',len(roadcells),'road cells',len(groundcells),'entrance cells',flush=True)
 # Sorted fine elevations allow exact positive-depth integration at each selected state.
 rz=np.full((len(roadcells),400),np.inf,'f4');gz=np.full((len(groundcells),400),np.inf,'f4')
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cls:
  for row in range(0,1012,16):
   nr=min(16,1012-row);win=Window(0,row*20,15400,nr*20);z=dem.read(1,window=win).reshape(nr,20,770,20).transpose(0,2,1,3).reshape(-1,400);cl=cls.read(1,window=win).reshape(nr,20,770,20).transpose(0,2,1,3).reshape(-1,400)
   for cells,dest,road in [(roadcells,rz,True),(groundcells,gz,False)]:
    jj=np.flatnonzero((cells>=row*770)&(cells<(row+nr)*770));kk=cells[jj]-row*770;zz=z[kk];cc=cl[kk];ok=np.isfinite(zz)&((cc==2) if road else (cc!=1));ok &= valid.ravel()[cells[jj],None];dest[jj]=np.sort(np.where(ok,zz,np.inf),axis=1)
 rn=np.isfinite(rz).sum(axis=1).astype('i4');gn=np.isfinite(gz).sum(axis=1).astype('i4');rc=np.cumsum(np.where(np.isfinite(rz),rz,0),axis=1,dtype='f8');gc=np.cumsum(np.where(np.isfinite(gz),gz,0),axis=1,dtype='f8')
 assert np.array_equal(rn*.25,s['Road Ground Area'].ravel()[roadcells]);assert np.array_equal(gn*.25,s['Ground Area'].ravel()[groundcells])
 np.savez_compressed(P/'fine_elevation_support.npz',road_cells=roadcells,ground_cells=groundcells,road_sorted_z=rz,road_cumulative_z=rc,road_count=rn,ground_sorted_z=gz,ground_cumulative_z=gc,ground_count=gn)
 h=load(H/'paired_exposure_bounds.npz');good=h['common_admitted'].ravel();rlo=[];rup=[];glo=[];gup=[];etas=[];mins=[];actual=[]
 for case in CASES:
  clock=load(H/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];ii=np.array([int(np.argmin(abs(clock-t))) for t in TIMES]);assert np.max(abs(clock[ii]-TIMES))<=.001;actual.append(clock[ii])
  with Dataset(H/'runs'/case/'sfincs_map.nc') as d:
   eta=np.stack([np.flipud(np.ma.asarray(d['zs'][int(i)],dtype='f8').filled(np.nan)).ravel()[both] for i in ii]);zmin=np.flipud(np.ma.asarray(d['zb'][:],dtype='f8').filled(np.nan)).ravel()[both]
  etas.append(eta);mins.append(zmin);ri=np.searchsorted(both,roadcells);gi=np.searchsorted(both,groundcells)
  a,b=depth_at(rz,rc,rn,eta[:,ri],zmin[ri]);rlo.append(a);rup.append(b);a,b=depth_at(gz,gc,gn,eta[:,gi],zmin[gi]);glo.append(a);gup.append(b)
 rlo=np.stack(rlo);rup=np.stack(rup);glo=np.stack(glo);gup=np.stack(gup)
 # Append event peaks as a separate frame, never a simultaneous state.
 rlo=np.concatenate([rlo,h['road_lower'].reshape(2,-1)[:,roadcells,None].transpose(0,2,1)],axis=1);rup=np.concatenate([rup,h['road_upper'].reshape(2,-1)[:,roadcells,None].transpose(0,2,1)],axis=1)
 glo=np.concatenate([glo,h['ground_lower'].reshape(2,-1)[:,groundcells,None].transpose(0,2,1)],axis=1);gup=np.concatenate([gup,h['ground_upper'].reshape(2,-1)[:,groundcells,None].transpose(0,2,1)],axis=1)
 assert np.all(rlo[:,:,rn>0]<=rup[:,:,rn>0]+1e-8);assert np.all(glo<=gup+1e-8)
 # The prior19:00 fine-ground calculation is an independent implementation.
 prior=h['snapshot_19jst_lower'].reshape(2,-1)[:,groundcells];ix19=TIMES.index(14400);assert np.allclose(glo[:,ix19],prior,equal_nan=True,atol=1e-6)
 np.savez_compressed(P/'surface_exposure.npz',road_cells=roadcells,ground_cells=groundcells,road_lower=rlo,road_upper=rup,ground_lower=glo,ground_upper=gup,road_known=good[roadcells]&(rn>0),ground_known=good[groundcells],ground_area=gn*.25,entrance_offsets=np.array(goff),entrance_cell_index=gindex,entrance_ids=np.array([e['osm_node_id'] for e in entrance]),sample_cells=both,source_eta=np.stack(etas),source_bed_minimum=np.stack(mins),actual_times=np.stack(actual),times_s=np.array(TIMES),frame_ids=np.array([str(t) for t in TIMES]+['peak_envelope']))
 np.savez_compressed(P/'local_walking_graph.npz',node_ids=np.array(ids)[active_nodes],source_global_index=active_nodes,from_index=uu,to_index=vv,length_m=w['length_m'][keep],source_arc_index=np.flatnonzero(keep),arc_basis=arc_basis,inside_core=inc,x_m=xy[active_nodes,0],y_m=xy[active_nodes,1],exit_node_index=exitnodes,probe_node_index=local[global_probes],probe_ids=np.array([p['probe_id'] for p in probes]),edge_sample_offsets=np.array(offsets),edge_sample_road_index=rindex)
 write(P/'road_segment_inventory.csv',metadata);write(P/'surface_probe_inventory.csv',probes)
 dump(R/'preparation_validation.json',dict(all_passed=True,nodes=len(active_nodes),arcs=len(uu),road_segments=len(bases),native_boundary_exit_nodes=len(exitnodes),simultaneous_states=12,peak_frame_separate=True,fine_support_exact_match=True,prior_19jst_ground_match=True,max_clock_error_s=float(np.max(abs(np.stack(actual)-TIMES))),seconds=time.monotonic()-started))
 print('PREPARATION DONE',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
