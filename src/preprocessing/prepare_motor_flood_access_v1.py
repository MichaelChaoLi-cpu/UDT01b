"""Reuse existing car graph; map native source roads to existing terrain/flood states."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','2')
from pathlib import Path
import sys,json,gzip,collections,time
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import shape,LineString
from shapely.ops import transform
from shapely import STRtree
from pyproj import Transformer
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_historical_surface_access_v1 import read,write,dump,load,sha
from prepare_historical_surface_coverage_v2 import edge_maximum,samples
N=ROOT/'data/processed/three_ward_normal_transport_network_v1'
P=ROOT/'data/processed/three_ward_motor_flood_access_v1'
R=ROOT/'data/exp/estimation/three_ward_motor_flood_access_v1'
F=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
CASES=[f'{rain}__{stage}__{arm}'for rain in ['R0','R1','R125']for stage in ['L','H']for arm in ['T','U']]

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'exposures').mkdir(exist_ok=True)
 z=load(N/'drive_conservative_graph.npz');keys=list(map(str,z['node_ids']));ix={k:i for i,k in enumerate(keys)};physical=np.flatnonzero(z['osm_way_id']!=0);u=z['from_index'][physical];v=z['to_index'][physical];ways=z['osm_way_id'][physical];length=z['length_m'][physical]
 lookup=collections.defaultdict(list)
 for i in range(len(physical)):lookup[int(ways[i]),int(u[i]),int(v[i])].append(i)
 road_of_arc=np.full(len(physical),-1,'i4');roads=[];pixlists=[];counts=collections.Counter()
 common=load(F/'factor_comparison_v1/common_support.npz')['common_admitted'].ravel()
 waterpath=ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson';water=json.loads(waterpath.read_text())['features'];proj=Transformer.from_crs(4326,6677,always_xy=True);tree=STRtree([transform(proj.transform,shape(f['geometry']))for f in water])
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as inp:
  for line in inp:
   e=json.loads(line)
   if e['from_node']not in ix or e['to_node']not in ix:continue
   a=ix[e['from_node']];b=ix[e['to_node']];matched=lookup.get((e['osm_way_id'],a,b),[])+lookup.get((e['osm_way_id'],b,a),[])
   matched=[i for i in matched if abs(length[i]-e['length_m'])<1e-7]
   if not matched:continue
   if any(road_of_arc[i]>=0 for i in matched):raise ValueError('Ambiguous source-road correspondence')
   j=len(roads);road_of_arc[matched]=j;tag=e['tags'];xy=e['geometry']['coordinates'];coords=np.asarray(xy);inbox=np.max(coords[:,0])>=-11360 and np.min(coords[:,0])<=-3660 and np.max(coords[:,1])>=-42440 and np.min(coords[:,1])<=-32320
   reason='outside_model_domain';pixels=np.array([],dtype='i8')
   if inbox:
    reason='unknown_road_deck_or_tunnel_height'
    if e['grade_key']=='0|0|surface':
     reason='covered_or_indoor'if tag.get('covered')not in[None,'no','false','0']or tag.get('indoor')=='yes'else'candidate_ground'
     if reason=='candidate_ground':
      if len(tree.query(LineString(xy),predicate='intersects')):reason='intersects_mapped_water'
      else:
       pixels,out=samples(xy)
       if out or not len(pixels):reason='partly_outside_model_domain'
   roads.append(dict(road_index=j,source_edge_id=e['edge_id'],osm_way_id=e['osm_way_id'],name=tag.get('name',''),highway=e['highway'],grade_key=e['grade_key'],inside_core=e['inside_core'],length_m=e['length_m'],reason=reason,from_node=e['from_node'],to_node=e['to_node']))
   pixlists.append(pixels if reason=='candidate_ground'else np.array([],dtype='i8'))
 assert np.all(road_of_arc>=0)
 offsets=np.r_[0,np.cumsum([len(x)for x in pixlists])];flat=np.concatenate(pixlists);pixels=np.unique(flat);refs=np.searchsorted(pixels,flat).astype('i4');rr=pixels//15400;cc=pixels%15400;coarse=(rr//20)*770+cc//20;fine_z=np.full(len(pixels),np.nan,'f4');fine_cl=np.full(len(pixels),255,'u1')
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif')as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif')as cls:
  for row in range(0,20240,512):
   which=np.flatnonzero((rr>=row)&(rr<row+512))
   if not len(which):continue
   win=Window(0,row,15400,min(512,20240-row));fine_z[which]=dem.read(1,window=win)[rr[which]-row,cc[which]];fine_cl[which]=cls.read(1,window=win)[rr[which]-row,cc[which]]
 qualified=np.zeros(len(roads),bool)
 for j,road in enumerate(roads):
  if road['reason']=='candidate_ground':
   p=refs[offsets[j]:offsets[j+1]]
   if not np.isfinite(fine_z[p]).all():road['reason']='missing_fine_terrain'
   elif not np.isin(fine_cl[p],[0,2]).all():road['reason']='building_or_structure_mask_conflict'
   elif not common[coarse[p]].all():road['reason']='outside_common_admitted_support'
   else:qualified[j]=True;road['reason']='qualified_source_ground_centerline'
  counts[road['reason']]+=1
 np.savez_compressed(P/'road_projection.npz',physical_source_arc_index=physical,from_index=u,to_index=v,length_m=length,osm_way_id=ways,arc_to_road=road_of_arc,qualified=qualified,point_z=fine_z,point_coarse=coarse,point_refs=refs,offsets=offsets)
 write(P/'road_inventory.csv',roads);np.savez_compressed(P/'source_motor_graph.npz',**z)
 # Keep existing geographic cells and source-native car probes; do not invent connectors.
 probes=read(ROOT/'data/processed/three_ward_motor_vehicle_inputs_v1/motor_probe_inventory.csv');write(P/'probe_inventory.csv',probes)
 runby={x['case_id']:x for x in json.loads((F/'run_manifest.json').read_text())['runs']};times=np.array([0,3600,7200,10800,12600,14400,16200,18000,21600,25200,28800,32400]);checks=[]
 for case in CASES:
  run=runby[case];base=H if run['reused']else F;old=run.get('original_case_id',case);clock=load(base/'execution'/(old+'_output_clock.npz'))['Actual Map Time'];indices=[int(np.argmin(abs(clock-t)))for t in times];assert max(abs(clock[indices]-times))<=.001
  auditpath=ROOT/run['original_audit']if run['reused']else ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1/execution'/case/'audit.json';assert json.loads(auditpath.read_text())['all_checks_passed']
  lower=[];upper=[]
  with Dataset(ROOT/run['folder']/'sfincs_map.nc')as ds:
   bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for k in indices+[None]:
    eta=np.flipud(np.ma.asarray(ds['zs'][k]if k is not None else ds['zsmax'][0],dtype='f8').filled(np.nan)).ravel();lo,up=edge_maximum(fine_z,coarse,refs,offsets,qualified,eta,bed);lower.append(lo);upper.append(up)
  np.savez_compressed(P/'exposures'/(case+'.npz'),lower=np.array(lower),upper=np.array(upper),actual_times=clock[indices]);checks.append(dict(case_id=case,original_gate=True,actual_clock_error_s=float(max(abs(clock[indices]-times)))));print('MOTOR EXPOSURE',case,round(time.monotonic()-start),flush=True)
 sources=[N/n for n in ['drive_conservative_graph.npz','supported_motorcar_restrictions.json','conditional_motorcar_restrictions.json','held_turn_way_ids.json','surface_physical_edges.jsonl.gz']]+[waterpath,F/'factor_comparison_v1/common_support.npz',ROOT/'data/processed/three_ward_motor_vehicle_inputs_v1/motor_probe_inventory.csv',ROOT/'src/preprocessing/normal_transport_rules_v1.py']
 dump(P/'source_bindings.json',{str(p.relative_to(ROOT)):sha(p)for p in sources})
 dump(P/'analysis_contract.json',dict(human_decision='MILI-D-20261017-009',source_motor_graph='Existing conservative normal motorcar graph; preserve supported turn history and native one-way/level links. No new pointwise rule investigation.',cases=CASES,times_s=times.tolist(),frame_ids=[str(t)for t in times]+['peak_envelope'],depth_thresholds_m=[.1,.3],threshold_role='Analyst-declared vehicle road interruption sensitivities, not certified passenger-car wading limits or operational closure thresholds.',flood_uncertainty=['lower_depth_unknown_available','upper_depth_unknown_withheld'],road_measure='Max depth along source motorcar centerline samples every0.5m plus original vertices, using original0.5m ground under10m modeled level; source permission is motorcar-specific. Not surveyed lane water depth.',unknown='Unmodeled roads, unknown bridge/tunnel/deck heights, covered roads and unsupported ground samples remain unknown, not dry. Each branch has the same matched normal baseline.',destinations='Native cross-core road boundary exits grouped in four geographic directions; crossing the study boundary is not reaching a safe destination or completing an urban trip.',metrics='Directional access loss and extra shortest road distance in paired normal/closed graph; no traffic volumes, capacity, queues, travel times or affected people.',new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False))
 write(R/'projection_coverage.csv',[dict(reason=k,source_segments=v,core_source_segments=sum(x['inside_core']and x['reason']==k for x in roads),whole_core_intersecting_length_km=sum(x['length_m']for x in roads if x['inside_core']and x['reason']==k)/1000)for k,v in counts.items()])
 dump(R/'preparation_validation.json',dict(all_passed=True,original_arcs_mapped=len(physical),source_roads=len(roads),qualified_source_ground_segments=int(qualified.sum()),cases=checks,elapsed_s=time.monotonic()-start,new_hydraulics=False));print('PREPARED MOTOR FLOOD INPUTS',len(roads),int(qualified.sum()),flush=True)
if __name__=='__main__':main()
