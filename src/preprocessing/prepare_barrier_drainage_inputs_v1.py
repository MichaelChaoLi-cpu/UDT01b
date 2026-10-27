"""Freeze hypothetical physical measures on the existing118-road corridor."""
import os,sys,json,gzip,shutil,datetime as dt
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np,pandas as pd,rasterio
from scipy.ndimage import binary_dilation,uniform_filter
from scipy.spatial import cKDTree
from shapely.geometry import shape,LineString,mapping
from shapely.strtree import STRtree
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import load,dump,sha
D=ROOT/'data/processed';TAG='three_ward_barrier_drainage_v1';P=D/TAG;R=ROOT/'data/exp/estimation'/TAG;TEMP=D/'three_ward_five_event_temporal_analysis_v1';MEAS=D/'three_ward_demand_protection_feedback_v1';OLD=D/'three_ward_cross_event_protection_v1';NEW=D/'three_ward_five_event_population_validation_v1';BASEIMAGE='sha256:5415c73fc6ac335bd225336d0b2055c0ea0345c2eced42ba9cef8e1c44ae1ace'
def main():
 assert not(P/'protocol.json').exists();(P/'measures').mkdir(exist_ok=True);(P/'execution').mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 clip=load(TEMP/'clipped_road_lengths.npz');pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');prot=clip['protected'];q=clip['qualified'];assert prot.sum()==118 and not(prot&~q).any()
 meta=NEW/'runs/E20240618__H__U/udt_negative_native_map.csv';native=pd.read_csv(meta);assert np.array_equal(native.native_cell_index,np.arange(len(native))+1);sr=native.row_south_one_based.to_numpy()-1;cc=native.col_east_one_based.to_numpy()-1;mask=native.kcs.to_numpy();idx=(1011-sr)*770+cc;xy=np.c_[-11360+(cc+.5)*10,-42440+(sr+.5)*10];flat=np.zeros(1012*770,bool)
 for i in np.flatnonzero(prot):flat[pr['point_coarse'][pr['point_refs'][pr['offsets'][i]:pr['offsets'][i+1]]]]=True
 tube=binary_dilation(flat.reshape(1012,770),iterations=1).ravel()[idx]&(mask==1);top=D/'three_ward_negative_trace_interface_v1/whole_domain_metadata/whole_native_topology.bin'
 with top.open('rb')as f:
  n,nf=map(int,np.fromfile(f,'<i4',2));r0=np.fromfile(f,'<i4',n)-1;c0=np.fromfile(f,'<i4',n)-1;mk=np.fromfile(f,'<i4',n);f.seek(n*4*4,1);a=np.fromfile(f,'<i4',nf)-1;b=np.fromfile(f,'<i4',nf)-1
 assert np.array_equal(sr,r0)and np.array_equal(cc,c0)and np.array_equal(mask,mk)
 faces=np.flatnonzero(tube[a]^tube[b]);faces=faces[(mask[a[faces]]==1)&(mask[b[faces]]==1)];blocked=load(D/'three_ward_surrogate_inputs/surface_connectivity_graph.npz')['Blocked Native Face Id'];faces=np.setdiff1d(faces,blocked);features=json.loads((TEMP/'clipped_roads_6677.geojson').read_text())['features'];roads=[shape(f['geometry'])for f in features];tree=STRtree(roads);segments=[];rejected=[]
 demfile=D/'three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif'
 with rasterio.open(demfile)as dem:
  for face in faces:
   mid=(xy[a[face]]+xy[b[face]])*.5;normal=(xy[b[face]]-xy[a[face]])/10.;tangent=np.array([-normal[1],normal[0]]);ends=np.array([mid-tangent*5,mid+tangent*5]);line=LineString(ends)
   # Leave every mapped motor-road crossing open;do not draw a traffic-blocking ring.
   if len(tree.query(line.buffer(1.),predicate='intersects')):rejected.append(dict(face=int(face),reason='mapped_road_opening_retained'));continue
   pts=mid[None,:]+np.linspace(-4.75,4.75,20)[:,None]*tangent;z=np.array([x[0]for x in dem.sample(pts)])
   if not np.isfinite(z).all()or np.any(z<=-9000):rejected.append(dict(face=int(face),reason='unsupported_crest_elevation'));continue
   segments.append(dict(face=int(face),a=int(a[face]),b=int(b[face]),x1=float(ends[0,0]),y1=float(ends[0,1]),x2=float(ends[1,0]),y2=float(ends[1,1]),crest_m=float(z.max()+.5),ground_max_m=float(z.max()),height_above_face_ground_m=.5,Cd=.6))
 assert segments,'No eligible barrier faces';weir=P/'measures/barriers.weir'
 with weir.open('w')as f:
  for i,s in enumerate(segments):f.write(f'barrier_{i+1:04d}\n2 4\n{s["x1"]:.3f} {s["y1"]:.3f} {s["crest_m"]:.6f} 0.6\n{s["x2"]:.3f} {s["y2"]:.3f} {s["crest_m"]:.6f} 0.6\n')
 pd.DataFrame(segments).to_csv(P/'measures/barrier_faces.csv',index=False);pd.DataFrame(rejected).to_csv(P/'measures/barrier_exclusions.csv',index=False)
 with Dataset(NEW/'runs/E20240618__H__U/sfincs_map.nc')as ds:bed=np.flipud(np.asarray(ds['zb'][:])).ravel()
 candidates=np.flatnonzero(tube&np.isfinite(bed[idx]));water=np.flatnonzero((mask==2)&native.water_group_code.isin([17,51,71,79,106,110,111]).to_numpy());dist,ix=cKDTree(xy[water]).query(xy[candidates]);receiver=water[ix];localmean=uniform_filter(np.where(np.isfinite(bed),bed,0.).reshape(1012,770),size=5).ravel();score=localmean[idx[candidates]]-bed[idx[candidates]];order=np.lexsort((candidates,bed[idx[candidates]],-score));selected=[];pumps=[];excluded=[]
 for j in order:
  cell=candidates[j]
  if dist[j]>500:excluded.append(dict(native_cell=int(cell+1),reason='receiver_farther_than500m'));continue
  if selected and np.min(np.linalg.norm(xy[selected]-xy[cell],axis=1))<100:continue
  selected.append(cell);w=int(receiver[j]);pumps.append(dict(pump_id=len(pumps)+1,source_native_cell=int(cell+1),receiver_native_cell=w+1,source_x=float(xy[cell,0]),source_y=float(xy[cell,1]),receiver_x=float(xy[w,0]),receiver_y=float(xy[w,1]),receiver_group=int(native.water_group_code.iloc[w]),straight_link_length_m=float(dist[j]),capacity_m3_s=.5,trigger_depth_above_native_min_m=.05,maximum_lift_m=5.,static_depression_score_m=float(score[j]),source_ground_m=float(bed[idx[cell]]),receiver_kind='prescribed-stage reservoir;not modeled channel storage'))
  if len(pumps)==20:break
 assert pumps
 with(P/'measures/udt_pumps.dat').open('w')as f:
  f.write(str(len(pumps))+'\n')
  for s in pumps:f.write(f'{s["source_native_cell"]} {s["receiver_native_cell"]} 0.5 0.05 5.0\n')
 pd.DataFrame(pumps).to_csv(P/'measures/pump_sites.csv',index=False);pd.DataFrame(excluded).to_csv(P/'measures/pump_exclusions.csv',index=False);np.savez_compressed(P/'measures/spatial_contract.npz',corridor_native=tube,protected_source_roads=prot,barrier_face_indices=np.array([s['face']for s in segments]),native_grid_north_index=idx,native_kcs=mask,native_xy=xy)
 geo=[]
 for s in segments:geo.append(dict(type='Feature',properties=dict(kind='barrier',**s),geometry=mapping(LineString([(s['x1'],s['y1']),(s['x2'],s['y2'])]))))
 for s in pumps:geo.append(dict(type='Feature',properties=dict(kind='pump',**s),geometry=mapping(LineString([(s['source_x'],s['source_y']),(s['receiver_x'],s['receiver_y'])]))))
 dump(P/'measures/deployment_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=geo))
 budget=dict(concurrent_hydraulic_runs=1,cpus=6,memory_gib=7,maximum_new_runs=12,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=86400,maximum_new_output_gib=200,minimum_free_gib=100,automatic_retries=0,postprocessing_active_wall_s=14400)
 protocol=dict(decision='MILI-D-20261018-016',status='prepared_before_outcomes',question='Do bounded overtoppable barriers and explicit finite-capacity pumping reduce road interruption and resident vehicle-access loss without unacceptable neighboring transfer?',events=['E20250911','E20240618'],stages=['H','L'],arms=['none','barrier','drainage','combined'],four_baselines_reused=True,new_runs=12,barrier_height_above_static_face_ground_m=.5,barrier_coefficient=.6,barrier_faces=len(segments),barrier_length_m=10*len(segments),pumps=len(pumps),total_capacity_m3_s=.5*len(pumps),pump_trigger_depth_m=.05,max_lift_m=5.,maximum_straight_transfer_link_m=500,selection='Frozen118roads from prior2025event;static terrain minima within one-cell-expanded corridor;at most20pumps spaced100m;closest eligible prescribed-water receiver. No new event-outcome optimization;2025is selection-event context,2024is transfer.',barrier_openings='All mapped road-centerline crossings buffered1m remain open;no impermeable enclosure or automatic vehicle closures from barrier placement. Hypothetical parallel protection only;exact opening widths unresolved at10m.',receiver='Water transferred explicitly to identified fixed-stage boundary reservoirs;stage-driven head cutoff included,but river storage,pipe/hose alignment,pump curve,energy and receiver capacity not resolved. No sewer augmentation or engineering effectiveness claim.',pump_operator='Opt-in bounded volume exchange;reserve prior donor storage for all pumps before face limiter;no rainfall or simultaneous incoming transfer pre-spending. Internal transfer sources and sinks paired;prescribed-stage discharge recorded as other_sink. No storage clipping.',clock='Same actual seconds,9h,109frames,original rain and roof assignment',quality='Original numerical thresholds,closed-face zeros and common masks retained;extend ONLY explicit source/sink accounting. Stop on any failed gate.',comparisons='Same event,stage,time and source geometry. Report protected corridor and all other roads separately;adjacent water-depth increases and stock-access loss increments retained,never netted away.',population='2020resident stock496345;frozen0/10/25m candidate intervals;no actual stranded-person/daytime claim',traffic='Actual physical-treatment road states feed unchanged17h proxy routes and costs;no forced-dry road overlay in treatment arms',budget=budget,training=False,sealed_tests=False,engineering_parameters_observed=False)
 dump(P/'protocol.json',protocol)
 bound=[meta,top,demfile,TEMP/'clipped_road_lengths.npz',TEMP/'clipped_roads_6677.geojson',D/'three_ward_motor_flood_access_v1/road_projection.npz',D/'three_ward_surrogate_inputs/surface_connectivity_graph.npz',OLD/'run_manifest.json',NEW/'run_manifest.json']
 binding={str(f.relative_to(ROOT)):sha(f)for f in bound};dump(P/'protected_sources.json',binding);dump(R/'preparation_validation.json',dict(passed=True,barrier_faces=len(segments),road_openings_retained=sum(r['reason']=='mapped_road_opening_retained'for r in rejected),pump_count=len(pumps),selection_uses_new_treatment_outcomes=False,inputs_are_hypotheses=True));print('MEASURE GEOMETRY FROZEN',len(segments),'faces',len(pumps),'pumps')
if __name__=='__main__':main()
