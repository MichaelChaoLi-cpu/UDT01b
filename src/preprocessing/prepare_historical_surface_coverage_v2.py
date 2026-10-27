"""At-grade path correspondence supplement; immutable v1/network/hydraulics reused."""
from pathlib import Path
import os,sys,json,gzip,shutil,time,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
from shapely import STRtree
from shapely.geometry import shape,LineString
from shapely.ops import transform
from pyproj import Transformer
from numba import njit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_historical_surface_access_v1 import N,T,H,CASES,TIMES,read,write,dump,load,sha
OLD=ROOT/'data/processed/three_ward_surface_alternatives_v1';OLDR=ROOT/'data/exp/estimation/three_ward_surface_alternatives_v1'
P=ROOT/'data/processed/three_ward_surface_coverage_v2';R=ROOT/'data/exp/estimation/three_ward_surface_coverage_v2'
def samples(coords):
 q=np.asarray(coords,float);d=np.r_[0,np.cumsum(np.hypot(*np.diff(q,axis=0).T))];s=np.unique(np.r_[d,np.arange(0,d[-1],.5)]);x=np.interp(s,d,q[:,0]);y=np.interp(s,d,q[:,1]);rr=np.floor((-32320-y)/.5).astype(int);cc=np.floor((x+11360)/.5).astype(int);ok=(rr>=0)&(rr<20240)&(cc>=0)&(cc<15400)
 return np.unique(rr[ok]*15400+cc[ok]),bool((~ok).any())
@njit
def edge_maximum(point_z,point_coarse,point_refs,offsets,qualified,eta,bed):
 lo=np.full(len(qualified),np.nan);up=lo.copy()
 for j in range(len(qualified)):
  if not qualified[j]:continue
  a=0.;b=0.
  for k in range(offsets[j],offsets[j+1]):
   p=point_refs[k];c=point_coarse[p];zz=point_z[p];water=eta[c]
   if np.isfinite(water):d=max(0.,water-zz);a=max(a,d);b=max(b,d)
   else:b=max(b,max(0.,bed[c]+.01-zz))
  lo[j]=a;up[j]=b
 return lo,up

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 original=json.loads((OLD/'analysis_contract.json').read_text());policy=dict(original);policy.update(human_decision='MILI-D-20261017-003',parent_analysis='three_ward_surface_alternatives_v1',
 supplement_rule='Only originally unknown, exact0|0|surface, non-ambiguous in-domain source segments. Sample unchanged native polylines every0.5m and all source vertices. Every sample must have finite original0.5m terrain, fine constraint class0or2, common admitted10m support and no intersection with mapped PLATEAU water. Explicit covered/indoor/underground/bridge/tunnel or stair/elevator/corridor paths do not qualify. No snapping or new graph edges.',
 supplement_measure='For qualified segments compute max positive depth at the sampled fine path elevations under each original10m state water level; retain dry-state lower/upper censor bounds. Keep old classified-road-cell depth triggers as well. A supplementation assumption, not surveyed road/deck heights or submeter hydraulic accuracy.',
 uncertainty='Tier A original classified-road projection unchanged; Tier B source-supported ground-path supplement separately flagged; remaining unknown bridges/tunnels/structures stay unknown. Eight original threshold/identity/availability branches retained; results with Tier B are conditional on the supplement. Original strict run retained as a support-model comparison.',
 readiness='Judge conditional regional scenario-experiment readiness from coverage, unaltered physical outputs, explicit remaining masks and fully reproduced paths. No requirement of complete bridge/tunnel elevations or real-event validation. Absolute city-isolation or affected-population claims remain outside scope.')
 dump(P/'analysis_contract.json',policy)
 # Inputs are separately copied, never hard-linked to a writable predecessor.
 for name in ['local_walking_graph.npz','surface_exposure.npz','surface_probe_inventory.csv']:
  shutil.copy2(OLD/name,P/name)
 meta=read(OLD/'road_segment_inventory.csv');a=load(OLD/'alternative_distances.npz');oldunknown=a['road_unknown'];e=load(P/'surface_exposure.npz');w=load(P/'local_walking_graph.npz')
 sources=json.loads((OLD/'source_bindings.json').read_text())
 for p in [OLD/n for n in ['analysis_contract.json','local_walking_graph.npz','surface_exposure.npz','surface_probe_inventory.csv','road_segment_inventory.csv','alternative_distances.npz']]+[OLDR/n for n in ['normal_baseline_and_support.csv','classification_summary.csv','probe_classification.csv','analysis_summary.json','batch_readiness.json','independent_validation.json','delivery_manifest.json']]+[ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson']:
  sources[str(p.relative_to(ROOT))]=sha(p)
 dump(P/'source_bindings.json',sources)
 water=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features'];xy=Transformer.from_crs(4326,6677,always_xy=True);waters=[transform(xy.transform,shape(f['geometry'])) for f in water];wtree=STRtree(waters)
 wanted={r['source_edge_id']:i for i,r in enumerate(meta) if oldunknown[i] and r['known_surface_grade']=='True' and r['ambiguous_parallel_geometry']=='False' and r['has_out_of_grid_sample']=='False'}
 reasons=['original_classified_road' if not oldunknown[i] else 'unresolved_grade_or_level' if r['known_surface_grade']!='True' else 'ambiguous_parallel_geometry' if r['ambiguous_parallel_geometry']=='True' else 'partly_outside_hydraulic_grid' if r['has_out_of_grid_sample']=='True' else 'source_geometry_unresolved' for i,r in enumerate(meta)]
 lists=[np.array([],np.int64) for _ in meta];highways=['']*len(meta);found=0
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt') as f:
  for ln in f:
   x=json.loads(ln);i=wanted.get(x['edge_id'])
   if i is None:continue
   found+=1;t=x['tags'];highways[i]=x['highway'];bad=t.get('highway') in ['steps','elevator','corridor'] or t.get('indoor')=='yes' or t.get('covered') not in [None,'no','false','0'] or t.get('location') in ['underground','underwater','rooftop'] or t.get('bridge') not in [None,'no','false','0'] or t.get('tunnel') not in [None,'no','false','0']
   if bad:reasons[i]='explicit_vertical_covered_or_structure';continue
   if len(wtree.query(LineString(x['geometry']['coordinates']),predicate='intersects')):reasons[i]='intersects_mapped_water';continue
   pix,out=samples(x['geometry']['coordinates'])
   if out or not len(pix):reasons[i]='invalid_or_outside_samples';continue
   lists[i]=pix;reasons[i]='candidate_ground_path'
 offsets=np.r_[0,np.cumsum([len(x) for x in lists])];flat=np.concatenate(lists);pixels=np.unique(flat);refs=np.searchsorted(pixels,flat).astype('i4');rr=pixels//15400;cc=pixels%15400;coarse=(rr//20)*770+cc//20
 z=np.full(len(pixels),np.nan,'f4');cl=np.full(len(pixels),255,'u1')
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cls:
  for row in range(0,20240,512):
   ind=np.flatnonzero((rr>=row)&(rr<row+512))
   if not len(ind):continue
   nr=min(512,20240-row);win=Window(0,row,15400,nr);z[ind]=dem.read(1,window=win)[rr[ind]-row,cc[ind]];cl[ind]=cls.read(1,window=win)[rr[ind]-row,cc[ind]]
 good=load(H/'paired_exposure_bounds.npz')['common_admitted'].ravel();pointok=np.isfinite(z)&np.isin(cl,[0,2])&good[coarse];qualified=np.zeros(len(meta),bool)
 for i in range(len(meta)):
  if reasons[i]!='candidate_ground_path':continue
  pp=refs[offsets[i]:offsets[i+1]]
  if not np.isfinite(z[pp]).all():reasons[i]='missing_fine_terrain'
  elif (cl[pp]==1).any():reasons[i]='source_path_intersects_building_mask'
  elif (cl[pp]==3).any():reasons[i]='source_path_intersects_uncertain_structure_mask'
  elif not np.isin(cl[pp],[0,2]).all():reasons[i]='unsupported_fine_surface_class'
  elif not good[coarse[pp]].all():reasons[i]='nonadmitted_or_unmodelled_coarse_support'
  else:qualified[i]=True;reasons[i]='supplemented_ground_path'
 print('COVERAGE CANDIDATES',len(wanted),'qualified',int(qualified.sum()),'sample_points',len(pixels),flush=True)
 lower=np.full((2,13,len(meta)),np.nan);upper=lower.copy()
 for ci,case in enumerate(CASES):
  times=load(H/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];ii=[int(np.argmin(abs(times-t))) for t in TIMES]
  with Dataset(H/'runs'/case/'sfincs_map.nc') as d:
   bed=np.flipud(np.ma.asarray(d['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for j,k in enumerate(ii+[None]):
    eta=np.flipud(np.ma.asarray(d['zs'][k] if k is not None else d['zsmax'][0],dtype='f8').filled(np.nan)).ravel();lower[ci,j],upper[ci,j]=edge_maximum(z,coarse,refs,offsets,qualified,eta,bed)
 assert not np.any(qualified&~oldunknown);assert all(meta[i]['grade_key']=='0|0|surface' for i in np.flatnonzero(qualified));assert np.all(lower[:,:,qualified]<=upper[:,:,qualified]+1e-8)
 np.savez_compressed(P/'ground_path_supplement.npz',qualified=qualified,lower=lower,upper=upper,fine_pixel_ids=pixels,fine_terrain_z=z,fine_class=cl,coarse_flat_index=coarse,point_admitted=good[coarse],sample_offsets=offsets,sample_point_index=refs,old_unknown=oldunknown)
 for i,r in enumerate(meta):r.update(support_tier='classified_road' if not oldunknown[i] else 'ground_path_supplement' if qualified[i] else 'unknown',supplement_qualified=bool(qualified[i]),supplement_reason=reasons[i],source_highway=highways[i],fine_path_samples=int(offsets[i+1]-offsets[i]))
 write(P/'road_segment_inventory.csv',meta);write(R/'coverage_reason_summary.csv',[dict(reason=k,segments=v,length_km=sum(float(meta[i]['source_length_m']) for i in range(len(meta)) if reasons[i]==k)/1000) for k,v in collections.Counter(reasons).items()])
 prep=json.loads((OLDR/'preparation_validation.json').read_text());prep.update(parent='three_ward_surface_alternatives_v1',supplemented_ground_segments=int(qualified.sum()),unknown_segments_before=int(oldunknown.sum()),unknown_segments_after=int((oldunknown&~qualified).sum()),all_passed=True,source_network_and_exposure_bytes_unchanged=all(sha(OLD/n)==sha(P/n) for n in ['local_walking_graph.npz','surface_exposure.npz','surface_probe_inventory.csv']),seconds=time.monotonic()-start);dump(R/'preparation_validation.json',prep);print(json.dumps(prep,indent=2),flush=True)
if __name__=='__main__':main()
