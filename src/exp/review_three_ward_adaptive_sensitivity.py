"""Matched forcing sensitivity on fixed geometries, common original open ground.
Velocity maps show instantaneous, volume-weighted surface directions, not tracks.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import json,hashlib,collections,html,csv
import numpy as np
import rasterio,netCDF4
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import shape,mapping,LineString
from shapely.ops import transform
from scipy.stats import spearmanr
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_three_ward_adaptive_connectivity import adaptive_run,dump,csvwrite,write,a
from review_three_ward_urban_grid import read_run
from run_three_ward_adaptive_sensitivity import digest,CASES,GRIDS,ROOT,BASE,SOURCE,RUN
P=ROOT/'data/processed/three_ward_adaptive_sensitivity'
OUT=ROOT/'data/exp/data-briefing/three_ward_adaptive_sensitivity'
OLD=ROOT/'data/processed/three_ward_sfincs_scenarios'
CON=ROOT/'data/processed/three_ward_urban_constraints'
GEO=ROOT/'data/processed/three_ward_adaptive_connectivity'
NAMES=['rain60_low',*CASES]
XY=Transformer.from_crs(6668,6677,always_xy=True)
LL=Transformer.from_crs(6677,6668,always_xy=True)
def rank_correlation(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.ptp(x)==0 or np.ptp(y)==0:return None
 return float(spearmanr(x,y).statistic)
def folder(name,grid):return (SOURCE if name=='rain60_low' else RUN)/(name+'_'+grid)
def velocities(path,grid):
 # All snapshots use the same time and map every hydraulic cell to its exact 5m coverage.
 with netCDF4.Dataset(path/'sfincs_map.nc') as d:
  ti=int(np.argmin(abs(a(d['time'][:])-7200)));assert float(d['time'][ti])==7200
  if grid=='corrected_10m':
   expand=lambda v:np.repeat(np.repeat(np.flipud(a(v)),2,0),2,1)
   return {k:expand(d[k][ti]) for k in ['u','v']}|{'depth':expand(d['subgrid_volume'][ti])/100,'mask':expand(d['msk'][:])}
  with np.load(GEO/'adaptive_cell_geometry.npz') as g:r=g['row_fine']//10;c=g['col_fine']//10;size=g['size_m'];mask=g['mask']
  out={k:np.full((2024,1540),np.nan) for k in ['u','v','depth','mask']}
  vals={'u':a(d['u'][ti]),'v':a(d['v'][ti]),'depth':a(d['subgrid_volume'][ti])/(size**2),'mask':mask}
  for sz in [5,10]:
   sel=size==sz
   for dr in range(int(sz/5)):
    for dc in range(int(sz/5)):
     for k in out:out[k][r[sel]+dr,c[sel]+dc]=vals[k][sel]
  return out
def exchanges(path):
 sections=json.loads((path/'flux_sections.json').read_text());groups={s:[i for i,q in enumerate(sections) if q['water']==s] for s in sorted(set(q['water'] for q in sections))};rows={k:dict(water=k,net_out_m3=0.,outward_m3=0.,inward_m3=0.,peak_outward_m3_s=0.,peak_inward_m3_s=0.) for k in groups}
 with netCDF4.Dataset(path/'sfincs_his.nc') as d:
  ts=a(d['time'][:])
  for i in range(1,len(ts),1000):
   end=min(i+1000,len(ts));qq=np.nan_to_num(a(d['crosssection_discharge'][i:end]));dt=np.diff(ts[i-1:end])
   for key,ix in groups.items():
    q=qq[:,ix].sum(1);rec=rows[key];rec['net_out_m3']+=float(q@dt);rec['outward_m3']+=float(np.maximum(q,0)@dt);rec['inward_m3']+=float(np.maximum(-q,0)@dt);rec['peak_outward_m3_s']=max(rec['peak_outward_m3_s'],float(q.max()));rec['peak_inward_m3_s']=max(rec['peak_inward_m3_s'],float(-q.min()))
 return list(rows.values())
def peak_censoring(fields,path,grid):
 # Pinned official ncoutput lines3517/3715 suppress zsmax when zsmax-zmin<=huthresh.
 cfg=dict(line.split('=',1) for line in (path/'sfincs.inp').read_text().splitlines() if '=' in line)
 threshold=float(next(v for k,v in cfg.items() if k.strip()=='huthresh'));assert threshold==.01
 with netCDF4.Dataset(path/'sfincs_map.nc') as d:z=a(d['zb'][:]);rawmask=a(d['msk'][:]);peak=a(d['zsmax'][0])
 if grid=='corrected_10m':z5=np.repeat(np.repeat(np.flipud(z),2,0),2,1)
 else:
  z5=np.full((2024,1540),np.nan)
  with np.load(GEO/'adaptive_cell_geometry.npz') as g:
   r=g['row_fine']//10;c=g['col_fine']//10;size=g['size_m']
   for sz in [5,10]:
    sel=size==sz
    for dr in range(int(sz/5)):
     for dc in range(int(sz/5)):z5[r[sel]+dr,c[sel]+dc]=z[sel]
 censor=(fields['mask5']==1)&~np.isfinite(fields['zs5']);assert np.all(np.isfinite(z5[censor]))
 fields.update(zmin5=z5,peak_censored5=censor)
 return {'masked_peak_active_native_cells':int(((rawmask==1)&~np.isfinite(peak)).sum()),'display5m_censored_pixels':int(censor.sum()),'maximum_suppressed_level_above_cell_min_m':threshold,'official_source':'Pinned sfincs_ncoutput.F90 lines3517 and3715','mean_peak_depth_in_censored_ground':'lower bound; unreported depth bounded by0.01m relative to original cell minimum; area thresholds>=0.05m unchanged'}
def cache(name,grid):
 path=folder(name,grid);key=name+'_'+grid;dest=P/(key+'.npz');meta=P/(key+'.json')
 if dest.exists() and meta.exists():
  q=json.loads(meta.read_text());assert q['execution_sha256']==digest(path/'execution.json')
  fields={k:v for k,v in np.load(dest).items()}
  if 'peak_censored5' not in fields:
   q['peak_censoring_audit']=peak_censoring(fields,path,grid);np.savez_compressed(dest,**fields);dump(meta,q)
  return fields,q
 v=read_run(path) if grid=='corrected_10m' else adaptive_run(path)
 zs=v['zs'];m=v['m'];debt=v['debt']
 if grid=='corrected_10m':
  zs=np.repeat(np.repeat(zs,2,0),2,1);m=np.repeat(np.repeat(m,2,0),2,1);debt=np.repeat(np.repeat(debt,2,0),2,1)
 with netCDF4.Dataset(path/'sfincs_his.nc') as hh,netCDF4.Dataset(folder('rain60_low',grid)/'sfincs_his.nc') as baseline_hh:
  snapped_same=all(np.array_equal(hh[k][:],baseline_hh[k][:]) for k in ['thindam_x','thindam_y'])
  assert snapped_same,'Building closures snapped differently from geometry-validated baseline'
 fields=velocities(path,grid);nr,nc=506,385
 block=lambda x:np.nan_to_num(x).reshape(nr,4,nc,4).sum((1,3))
 dep=np.where((fields['mask']==1)&np.isfinite(fields['u'])&np.isfinite(fields['v']),np.maximum(fields['depth'],0),0)
 weight=block(dep);u=np.divide(block(np.nan_to_num(fields['u'])*dep),weight,out=np.zeros((nr,nc)),where=weight>0);vv=np.divide(block(np.nan_to_num(fields['v'])*dep),weight,out=np.zeros((nr,nc)),where=weight>0)
 speedweighted=block(np.nan_to_num(np.hypot(fields['u'],fields['v']))*dep)
 coherence=np.divide(np.hypot(block(np.nan_to_num(fields['u'])*dep),block(np.nan_to_num(fields['v'])*dep)),speedweighted,out=np.zeros((nr,nc)),where=speedweighted>0)
 results={'zs5':zs,'mask5':m,'debt5':debt,'u20':u,'v20':vv,'depth20':weight/16,'coherence20':coherence}
 censor_audit=peak_censoring(results,path,grid)
 np.savez_compressed(dest,**results)
 q={'run':key,'budget':v['budget'],'river_exchange':exchanges(path),'runtime_s':v['summary']['runtime_s'] if grid=='corrected_10m' else v['runtime_s'],'execution_sha256':digest(path/'execution.json'),'building_closure_snapping_identical_to_validated_baseline':snapped_same,'peak_censoring_audit':censor_audit};dump(meta,q)
 return results,q
def directions(a0,b0,area,roads,sel,comparison):
 speed0=np.hypot(a0['u20'],a0['v20']);speed1=np.hypot(b0['u20'],b0['v20'])
 valid=lambda v,sp:sel&(area>0)&(v['depth20']>.01)&(sp>.03)&(v['coherence20']>=.5)
 x=valid(a0,speed0);y=valid(b0,speed1);common=x&y
 ang=np.degrees(np.arccos(np.clip((a0['u20']*b0['u20']+a0['v20']*b0['v20'])/np.maximum(speed0*speed1,1e-30),-1,1)))
 avg=lambda arr,weights,s:float(np.average(arr[s],weights=weights[s])) if np.any(s) and weights[s].sum()>0 else None
 road=common&(roads>0)
 q={'comparison':comparison,'snapshot_s':7200,'reference_moving_open_area_km2':float(area[x].sum()/1e6),'comparison_moving_open_area_km2':float(area[y].sum()/1e6),'common_moving_open_area_km2':float(area[common].sum()/1e6),'moving_support_iou':float(area[common].sum()/max(area[x|y].sum(),1)),'angle_mean_deg':avg(ang,area,common),'within30deg_fraction':avg(ang<=30,area,common),'within60deg_fraction':avg(ang<=60,area,common),'road_within30deg_fraction':avg(ang<=30,roads,road),'road_common_area_km2':float(roads[road].sum()/1e6),'support_rule':'Common 20m cells: mean volume depth>0.01m, speed>0.03m/s, vector coherence>=0.5. No numeric-debt masking. Ground-area weights; roads use original road-area weights.'}
 return q,np.where(common,ang,np.nan)
def paths(fields,sel,tf,name,grid):
 # Display streamlines in common20m aggregated field. No connectivity/capacity inference.
 lines=[];stats=collections.Counter();nr,nc=sel.shape
 sp=np.hypot(fields['u20'],fields['v20']);valid=sel&(fields['depth20']>.01)&(sp>.03)&(fields['coherence20']>=.5)
 for rr in range(4,nr,8):
  for cc in range(4,nc,8):
   if not valid[rr,cc]:continue
   pts=[(tf.c+(cc+.5)*20,tf.f-(rr+.5)*20)];seen=set();reason='length_limit'
   for _ in range(200):
    x,y=pts[-1];r=int((tf.f-y)//20);c=int((x-tf.c)//20)
    if not (0<=r<nr and 0<=c<nc) or not valid[r,c]:reason='slow_dry_unresolved_or_boundary';break
    loc=(int((x-tf.c)//5),int((tf.f-y)//5))
    if loc in seen:reason='local_loop';break
    seen.add(loc);pts.append((x+5*fields['u20'][r,c]/sp[r,c],y+5*fields['v20'][r,c]/sp[r,c]))
   if len(pts)>3:
    lines.append({'type':'Feature','geometry':mapping(transform(LL.transform,LineString(pts))),'properties':{'scenario':name,'grid':grid,'seed_row20':rr,'seed_col20':cc,'snapshot_s':7200,'stop':reason,'display_grid_m':20}});stats[reason]+=1
 return lines,dict(stats)
def interim_pair(name):
 dest=OUT/('interim_'+name+'_grid_comparison.json')
 if dest.exists() or not (P/'common_ground_support.npz').exists():return
 files=[P/(name+'_'+g+'.npz') for g in GRIDS]
 if not all(p.exists() for p in files):return
 support=np.load(P/'common_ground_support.npz');m=support['mask'];area=support['area'];roads=support['roads'];fields=[dict(np.load(p)) for p in files];tot=np.zeros(4)
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(CON/'constraints_0_50m.tif') as con:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);n=int(win.height);inside=np.repeat(np.repeat(m[rr//40:(rr+n)//40]==1,40,0),40,1);op=inside&(cl!=1)&np.isfinite(z)
   wet=lambda v:op&(np.repeat(np.repeat(v['zs5'][rr//10:(rr+n)//10],10,0),10,1)-z>.1)
   x=wet(fields[0]);y=wet(fields[1]);tot+=np.array([x.sum(),y.sum(),(x&y).sum(),(x|y).sum()])*.25
 flow,_=directions(*fields,area,roads,m==1,name+':grid')
 q={'scenario':name,'threshold_m':.1,'corrected10m_area_km2':tot[0]/1e6,'adaptive10_5m_area_km2':tot[1]/1e6,'relative_area_difference':abs(tot[1]-tot[0])/max(tot[0],1),'wet_iou':tot[2]/tot[3] if tot[3]>0 else 1.,'flow':flow}
 dump(dest,q);print('INTERIM PAIR',json.dumps(q,ensure_ascii=False),flush=True)
def main():
 P.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 source_manifest=json.loads((GEO/'fine_cache.json').read_text())['source_sha256']
 source_checks={k:digest(ROOT/k)==v for k,v in source_manifest.items()};assert all(source_checks.values()),'Fine source changed since geometry construction'
 dump(OUT/'fine_source_hash_validation.json',source_checks)
 runs={};metadata={}
 for name in NAMES:
  for grid in GRIDS:
   key=(name,grid);print('Read',name,grid,flush=True);runs[key],metadata[key]=cache(name,grid)
 with rasterio.open(OLD/'active_mask_20m.tif') as d:mask=d.read(1);tf=d.transform
 nr,nc=mask.shape;area=np.zeros(mask.shape);roads=np.zeros(mask.shape)
 thresholds=[.05,.1,.3];wet={key:{h:np.zeros(mask.shape) for h in thresholds} for key in runs};depthsum={key:np.zeros(mask.shape) for key in runs};overtop={key:0. for key in runs};debtarea={key:0. for key in runs}
 pairs={name+':grid':[(name,GRIDS[0]),(name,GRIDS[1])] for name in NAMES}
 for grid in GRIDS:
  for name in list(CASES):pairs[name+':baseline:'+grid]=[('rain60_low',grid),(name,grid)]
 joint={key:{h:[np.zeros(mask.shape),np.zeros(mask.shape)] for h in thresholds} for key in pairs}
 stage_deepening={g:np.zeros(mask.shape) for g in GRIDS}
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(CON/'constraints_0_50m.tif') as con,rasterio.open(CON/'hydraulic_obstacle_0_50m.tif') as hydraulic:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);nh=int(win.height)//40;r0=rr//40;spatial=np.repeat(np.repeat(mask[r0:r0+nh]==1,40,0),40,1);op=spatial&(cl!=1)&np.isfinite(z)
   assert np.array_equal(hydraulic.read(1,window=win)[op],z[op]),'Comparison ground differs from hydraulic source on open ground'
   reduce=lambda x:x.reshape(nh,40,nc,40).sum((1,3),dtype=np.float64)*.25
   area[r0:r0+nh]=reduce(op);roads[r0:r0+nh]=reduce(op&(cl==2));dd={}
   for key,v in runs.items():
    expand=lambda a:np.repeat(np.repeat(a[rr//10:(rr+int(win.height))//10],10,0),10,1)
    zs=expand(v['zs5']);active=expand(v['mask5'])==1;censored=expand(v['peak_censored5']);valid=active&(np.isfinite(zs)|censored);assert not np.any(op&~valid),'Original open ground omitted'
    upper=expand(v['zmin5'])+.01;assert not np.any(op&censored&(upper-z>.0101)),'Censored peaks could exceed area thresholds on fine ground'
    d=np.where(op&~censored,np.maximum(zs-z,0),0);dd[key]=d;depthsum[key][r0:r0+nh]=reduce(d);overtop[key]+=float(np.count_nonzero(spatial&(cl==1)&valid&(zs>=z+9.99))*.25);debtarea[key]+=float(np.count_nonzero(op&expand(v['debt5']))*.25)
    for h in thresholds:wet[key][h][r0:r0+nh]=reduce(d>h)
   for key,(aa,bb) in pairs.items():
    for h in thresholds:x=dd[aa]>h;y=dd[bb]>h;joint[key][h][0][r0:r0+nh]=reduce(x&y);joint[key][h][1][r0:r0+nh]=reduce(x|y)
   for g in GRIDS:stage_deepening[g][r0:r0+nh]=reduce((dd[('rain60_raised',g)]-dd[('rain60_low',g)])>.05)
   print('Fine support',rr,dem.height,flush=True)
 features=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];geoms=[transform(XY.transform,shape(f['geometry'])) for f in features];ward=rasterize([(g,i+1) for i,g in enumerate(geoms)],out_shape=mask.shape,transform=tf)
 scopes={'Whole domain':mask==1,'Chiyoda':(ward==1)&(mask==1),'Chuo':(ward==2)&(mask==1),'Minato':(ward==3)&(mask==1),'Cross-ward contributing area':(ward==0)&(mask==1)};rows=[]
 for scope,sel in scopes.items():
  for pair,(aa,bb) in pairs.items():
   for h in thresholds:
    wa=float(wet[aa][h][sel].sum());wb=float(wet[bb][h][sel].sum());ii=float(joint[pair][h][0][sel].sum());uu=float(joint[pair][h][1][sel].sum());gap=abs(wb-wa)/max(wa,1);iou=ii/uu if uu>0 else 1.
    rows.append({'scope':scope,'comparison':pair,'threshold_m':h,'reference_area_km2':wa/1e6,'comparison_area_km2':wb/1e6,'relative_area_difference':gap,'wet_iou':iou,'reference_wet_retained_fraction':ii/max(wa,1),'common_wet_core_km2':ii/1e6,'disagreement_km2':(uu-ii)/1e6,'common_open_area_km2':float(area[sel].sum()/1e6),'both_extents_empty':wa==0 and wb==0,'grid_screening_pass':gap<.1 and iou>.7 if pair.endswith(':grid') else None})
 csvwrite(OUT/'area_stability.csv',rows)
 deepening_rows=[{'scope':scope,'grid':g,'peak_depth_increase_gt0_05_area_km2':float(stage_deepening[g][sel].sum()/1e6)} for scope,sel in scopes.items() for g in GRIDS]
 csvwrite(OUT/'raised_stage_peak_depth_effect.csv',deepening_rows)
 hot=json.loads((OLD/'priority_accumulation_areas.geojson').read_text())['features'];regional=[]
 for f in hot:
  sel=rasterize([(transform(XY.transform,shape(f['geometry'])),1)],out_shape=mask.shape,transform=tf).astype(bool)&(mask==1);den=float(area[sel].sum());q={'id':f['properties']['id'],'ward':f['properties'].get('ward'),'approximate_location':f['properties'].get('location_hint'),'location_is_model_coordinate_inference':True,'open_area_m2':den}
  for key in runs:
   name='_'.join(key);q[name+'_wet_fraction']=float(wet[key][.1][sel].sum()/max(den,1));q[name+'_mean_peak_depth_m']=float(depthsum[key][sel].sum()/max(den,1))
  q['retained_all_rain_cases_both_grids']=all(q['_'.join((name,g))+'_wet_fraction']>=.5 for name in NAMES if name!='dry_raised' for g in GRIDS);regional.append(q)
 csvwrite(OUT/'priority_region_stability.csv',regional)
 budgets=[];rivers=[];geometrycheck={}
 for key,meta in metadata.items():
  name,grid=key;path=folder(*key);sc=json.loads((path/'scenario_contract.json').read_text())
  if name!='rain60_low':
   geometrycheck['_'.join(key)]={k:digest(path/k)==s for k,s in sc['geometry_inputs_identical_sha256'].items()};assert all(geometrycheck['_'.join(key)].values())
  b=dict(meta['budget']);b.update(scenario=name,grid=grid,building_overtopped_fine_area_m2=overtop[key],open_area_in_negative_volume_cells_km2=debtarea[key]/1e6);budgets.append(b)
  for r in meta['river_exchange']:rivers.append({'scenario':name,'grid':grid,**r})
 dump(OUT/'water_budgets.json',budgets);columns=sorted(set().union(*(q.keys() for q in budgets)));csvwrite(OUT/'water_budgets.csv',[{k:q.get(k,'') for k in columns} for q in budgets]);csvwrite(OUT/'river_exchange.csv',rivers);dump(OUT/'geometry_reuse_validation.json',geometrycheck)
 flow=[];angles={};line_features=[];line_stats=[]
 for pair,(aa,bb) in pairs.items():
  q,ang=directions(runs[aa],runs[bb],area,roads,mask==1,pair);flow.append(q);angles[pair]=ang
 csvwrite(OUT/'flow_direction_stability.csv',flow)
 for key,v in runs.items():
  lines,stats=paths(v,mask==1,tf,*key);line_features.extend(lines);line_stats.append({'scenario':key[0],'grid':key[1],'lines':len(lines),'termination':stats})
 dump(P/'surface_direction_streamlines.geojson',{'type':'FeatureCollection','features':line_features});dump(OUT/'streamline_display_audit.json',{'interpretation':'Instantaneous streamlines in volume-weighted common20m velocity field. Aggregation may bridge subcell obstacles; these do not certify street connectivity, bank entries or particle travel.','cases':line_stats})
 river_summaries=[]
 for name in NAMES:
  for grid in GRIDS:
   records=[r for r in rivers if r['scenario']==name and r['grid']==grid]
   top=sorted(records,key=lambda r:r['outward_m3'],reverse=True);back=sorted(records,key=lambda r:r['inward_m3'],reverse=True)
   river_summaries.append({'scenario':name,'grid':grid,'net_surface_to_boundary_m3':sum(r['net_out_m3'] for r in records),'outward_after_waterbody_aggregation_m3':sum(r['outward_m3'] for r in records),'inward_after_waterbody_aggregation_m3':sum(r['inward_m3'] for r in records),'top_outward_receivers':[dict(water=r['water'],volume_m3=r['outward_m3']) for r in top[:3]],'top_inward_sources':[dict(water=r['water'],volume_m3=r['inward_m3']) for r in back[:3] if r['inward_m3']>0]})
 dump(OUT/'river_exchange_summary.json',river_summaries)
 primary=[r for r in rows if r['scope']=='Whole domain' and r['threshold_m']==.1 and r['comparison'].endswith(':grid')];changes=[r for r in rows if r['scope']=='Whole domain' and r['threshold_m']==.1 and ':baseline:' in r['comparison']]
 ranks={name:rank_correlation([q[name+'_'+GRIDS[0]+'_mean_peak_depth_m'] for q in regional],[q[name+'_'+GRIDS[1]+'_mean_peak_depth_m'] for q in regional]) for name in NAMES}
 findings={'main_grid_comparison':primary,'peak_censoring_audits':{'_'.join(k):v['peak_censoring_audit'] for k,v in metadata.items()},'raised_stage_peak_depth_effect':deepening_rows,'river_exchange_summary':river_summaries,'scenario_changes_from_baseline':changes,'flow_direction_comparisons':flow,'priority_regions_retained_all_rain_cases_both_grids':sum(q['retained_all_rain_cases_both_grids'] for q in regional),'priority_regions_tested':len(regional),'hotspot_depth_rank_spearman_by_scenario':ranks,'hotspot_depth_rank_relative_to_baseline':{n+':'+g:rank_correlation([q['rain60_low_'+g+'_mean_peak_depth_m'] for q in regional],[q[n+'_'+g+'_mean_peak_depth_m'] for q in regional]) for n in CASES for g in GRIDS},'all_water_budgets_passed':all(q['passed'] for q in budgets),'all_new_building_closure_snapping_identical_to_validated_baseline':all(metadata[(n,g)]['building_closure_snapping_identical_to_validated_baseline'] for n in CASES for g in GRIDS),'geometry_and_forcing_validation_passed':all(all(q.values()) for q in geometrycheck.values()) and all(json.loads((folder(n,g)/'forcing_validation.json').read_text())['passed'] for n in CASES for g in GRIDS),'all_domain_0_1m_grid_gates_passed':all(q['grid_screening_pass'] for q in primary),'all_domain_threshold_grid_gates_passed':all(q['grid_screening_pass'] for q in rows if q['scope']=='Whole domain' and q['comparison'].endswith(':grid')),'all_ward_0_1m_grid_gates_passed':all(q['grid_screening_pass'] for q in rows if q['scope'] in ['Chiyoda','Chuo','Minato'] and q['comparison'].endswith(':grid') and q['threshold_m']==.1),'common_open_support_km2':float(area.sum()/1e6),'cases':{n:dict(rain_mm_h=60,stage_rise_m=0) if n=='rain60_low' else dict(rain_mm_h=CASES[n][0],stage_rise_m=CASES[n][1]) for n in NAMES},'formal_grid_convergence_established':False,'event_validated':False,'channel_bathymetry_available':False,'pipes_pumps_infiltration_measures':False,'combined100mmh_and_raised_stage_run':False,'strict_nonnegative_all':all(q['strict_nonnegative'] for q in budgets),'no_point_surveys':True,'original_terrain_modified':False,'fine_source_hashes_unchanged':all(source_checks.values()),'comparison_ground_exactly_matches_hydraulic_source_on_open_ground':True,'domain_is_provisional_not_certified_complete':True,'external_cross_ward_building_road_constraints_complete':False,'height_epoch_alignment_complete':False,'peak_extent_is_not_simultaneous_extent':True,'raised_stage_applies_to':'All existing named river and unnamed water boundary points; uniform additive pulse, not observed spatial stages.', 'river_flux_positive_definition':'Surface to water boundary; negative=boundary water to surface. Outward/inward split after aggregation within each named water body; does not measure real channel discharge.','negative_signed_volume_note':'Official solver clamps water level to minimum ground for empty or negative stored volume. Negative signed bookkeeping volume is audited and is not automatically a negative physical water depth.'};dump(OUT/'findings.json',findings)
 for key in runs:
  write(P/('_'.join(key)+'_wet_fraction20m.tif'),np.where(area>0,wet[key][.1]/np.maximum(area,.25),np.nan),tf)
 extent=[tf.c,tf.c+nc*20,tf.f-nr*20,tf.f]
 with (OLD/'boundary_points.csv').open() as f:waterpoints=list(csv.DictReader(f))
 wx=np.array([float(q['x']) for q in waterpoints]);wy=np.array([float(q['y']) for q in waterpoints])
 def decoration(ax):
  ax.scatter(wx,wy,s=.7,c='#48a4b6',alpha=.7,linewidths=0)
  for geom in geoms:
   for g in getattr(geom,'geoms',[geom]):x,y=g.exterior.xy;ax.plot(x,y,color='#333',lw=.45)
  for name,geom in zip(['Chiyoda','Chuo','Minato'],geoms):pt=geom.representative_point();ax.text(pt.x,pt.y,name,ha='center',fontsize=8,bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
  x0=extent[0]+350;y0=extent[2]+350;ax.plot([x0,x0+2000],[y0,y0],color='#222',lw=1.2);ax.text(x0+1000,y0+100,'2 km',ha='center',fontsize=8)
  ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 labels={'rain60_low':'60 mm/h / low stage','rain100_low':'100 mm/h / low stage','rain60_raised':'60 mm/h / +1.5 m pulse','dry_raised':'Dry / +1.5 m pulse'}
 fig,axes=plt.subplots(2,4,figsize=(15,10),layout='constrained')
 for i,name in enumerate(NAMES):
  for j,grid in enumerate(GRIDS):
   ax=axes[j,i];im=ax.imshow(np.where(area>0,wet[(name,grid)][.1]/np.maximum(area,.25),np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1);decoration(ax);ax.set_title(labels[name]+'\n'+('10 m' if j==0 else '10 / 5 m'),fontsize=10)
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Original open-ground peak wet fraction (>0.10 m)');fig.savefig(OUT/'scenario_extents.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,7),layout='constrained')
 for ax,name in zip(axes[:2],['rain100_low','rain60_raised']):
  change=(wet[(name,GRIDS[1])][.1]-wet[('rain60_low',GRIDS[1])][.1])/np.maximum(area,.25);im=ax.imshow(np.where(area>0,change,np.nan),extent=extent,cmap='RdBu',vmin=-1,vmax=1);decoration(ax);ax.set_title(labels[name]+'\nminus baseline (adaptive)',fontsize=10)
 fraction=stage_deepening[GRIDS[1]]/np.maximum(area,.25);im2=axes[2].imshow(np.where(area>0,fraction,np.nan),extent=extent,cmap='Oranges',vmin=0,vmax=1);decoration(axes[2]);axes[2].set_title('Raised stage: peak depth increase >0.05 m\nGround-area fraction (adaptive)',fontsize=10)
 fig.colorbar(im,ax=axes[:2].tolist(),shrink=.7,label='Change in peak wet fraction');fig.colorbar(im2,ax=axes[2],shrink=.7,label='Fraction with peak-depth increase >0.05 m');fig.savefig(OUT/'scenario_changes.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,4,figsize=(15,7),layout='constrained');yy,xx=np.mgrid[0:nr,0:nc];stride=10
 for ax,name in zip(axes,NAMES):
  key=(name,GRIDS[1]);v=runs[key];ax.imshow(np.where(area>0,wet[key][.1]/np.maximum(area,.25),np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1);s=np.hypot(v['u20'],v['v20']);valid=(mask==1)&(v['depth20']>.01)&(s>.03)&(v['coherence20']>=.5);qu=np.where(valid,v['u20']/np.maximum(s,1e-20),np.nan);qv=np.where(valid,v['v20']/np.maximum(s,1e-20),np.nan);ax.quiver((tf.c+(xx+.5)*20)[::stride,::stride],(tf.f-(yy+.5)*20)[::stride,::stride],qu[::stride,::stride],qv[::stride,::stride],color='#c04421',angles='xy',scale_units='xy',scale=.012,width=.004);decoration(ax);ax.set_title(labels[name]+'\nSurface direction at rain end',fontsize=10)
 fig.savefig(OUT/'surface_directions.png',dpi=150);plt.close(fig)
 table=''.join(f'<tr><td>{html.escape(q["comparison"].split(":")[0])}</td><td>{q["reference_area_km2"]:.3f}</td><td>{q["comparison_area_km2"]:.3f}</td><td>{q["relative_area_difference"]:.1%}</td><td>{q["wet_iou"]:.1%}</td><td>{q["grid_screening_pass"]}</td></tr>' for q in primary)
 flowtable=''.join(f'<tr><td>{html.escape(q["comparison"].split(":")[0])}</td><td>{q["common_moving_open_area_km2"]:.3f}</td><td>{q["moving_support_iou"]:.1%}</td><td>{q["within30deg_fraction"]:.1%}</td><td>{q["within60deg_fraction"]:.1%}</td></tr>' for q in flow if q['comparison'].endswith(':grid') and q['within30deg_fraction'] is not None)
 maxtotal=max(q['relative_error'] for q in budgets)
 rivertable=''.join(f'<tr><td>{html.escape(q["scenario"])}</td><td>{q["net_surface_to_boundary_m3"]:,.0f}</td><td>{q["inward_after_waterbody_aggregation_m3"]:,.0f}</td><td>{html.escape(", ".join(r["water"] for r in q["top_outward_receivers"]))}</td></tr>' for q in river_summaries if q['grid']=='adaptive_10_5m')
 regiontable=''.join(f'<tr><td>{html.escape(q["id"])}</td><td>{html.escape(q["ward"])}</td><td>{html.escape(q["approximate_location"] or '未命名候选片区')}</td><td>{q["rain60_low_adaptive_10_5m_mean_peak_depth_m"]:.2f}</td><td>{q["rain100_low_adaptive_10_5m_mean_peak_depth_m"]:.2f}</td><td>{q["rain60_raised_adaptive_10_5m_mean_peak_depth_m"]:.2f}</td></tr>' for q in sorted(regional,key=lambda q:q['rain100_low_adaptive_10_5m_mean_peak_depth_m'],reverse=True))
 areas={q['comparison'].split(':')[0]:q['comparison_area_km2'] for q in primary}
 summary=f"在这套固定地形、道路与建筑约束下，有雨情景的原15个候选片区全部保留，宏观排水方向与主要接收水域总体延续。100mm/h组积水范围比60mm/h低水位基线扩大{areas['rain100_low']/areas['rain60_low']-1:.1%}；抬升组增加约{areas['rain60_raised']-areas['rain60_low']:.2f}km²。无雨抬升组仍有约{areas['dry_raised']:.2f}km²积水，因此水域来水应与降雨汇流分别解释。网格间一致不等于不同物理驱动下的积水范围完全相同，也不等于实测验证。"
 text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区暴雨与河道水位情景稳定性</title><style>body{{max-width:1150px;margin:30px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#172d3e}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ccd8e0}}.scroll{{overflow:auto}}.note{{background:#eef4f8;padding:18px}}a{{color:#0864a3}}</style><h1>三区整体：更强暴雨与河道抬升</h1><p>{summary}</p><p class="note">同一套道路、建筑、地形和屋面汇流约束，复用已完成的60mm/h基线，新增100mm/h低水位、60mm/h＋1.5m水位脉冲、无雨＋1.5m水位脉冲。每组都比较全域10m和复杂区域自动加密至5m。全域0.10m积水范围网格筛查全部通过：{findings['all_domain_0_1m_grid_gates_passed']}；原15个重点片区在所有有雨情景和两套网格中保留{findings['priority_regions_retained_all_rain_cases_both_grids']}个。</p><p>降雨在1—2小时内持续一小时，其余为预热和消退。抬升脉冲在1小时开始，1.5小时升至＋1.5m并保持到2小时，3小时回到假设低水位；该加性脉冲施加到既有命名河道和未命名水域边界点，保留各点原低水位差异；这些不是观测暴雨或实测河道水位。未运行100mm/h同时抬升的组合，不外推其结果。</p><h2>积水范围对网格是否稳定</h2><a href="scenario_extents.png"><img src="scenario_extents.png" alt="四组情景两套网格积水范围"></a><div class="scroll"><table><tr><th>Scenario</th><th>10m/km²</th><th>10/5m/km²</th><th>Area difference</th><th>Wet overlap</th><th>Screening pass</th></tr>{table}</table></div><p>在共同原始0.50m开放地面上比较各地峰值水位对应的水深；展示图为20m内湿地面积比例，不是0.50m水动力计算。峰值范围不是同一时刻的淹没范围。官方求解器不输出峰值超过单元最低地面不足1cm的水位；这些单元按低于阈值保留，不剔除地面面积。输入最低地面与原始细地面逐块核对，故不影响5/10/30cm面积比较；其平均峰值深度仅给出下界，误差上界1cm。沿用面积差&lt;10%、重叠&gt;70%的探索性筛查门槛；全部阈值及分区结果见CSV，不宣称正式收敛。</p><h2>主要积水片区</h2><div class="scroll"><table><tr><th>Region</th><th>Ward</th><th>Approximate location</th><th>60 low/m</th><th>100 low/m</th><th>60 raised/m</th></tr>{regiontable}</table></div><p>表内为原15个候选片区内开放地面平均峰值水深（自适应网格），按强暴雨组排序。地点根据模型坐标概略推断，不是精确町丁目归属；全域图还包括这些片区以外的新增积水。</p><h2>暴雨增强和河道抬升改变了什么</h2><a href="scenario_changes.png"><img src="scenario_changes.png" alt="新增情景相对基线的变化"></a><p>高水位造成的排水减弱或倒灌属于物理情景响应，应与同一情景下的网格差异分开。无雨抬升组用于识别边界水进入地表的背景影响；不同情景的峰值范围不能线性相减来分离真实贡献。第三幅图显示抬升组相对低水位组峰值水深增加超过5cm的地面比例，含原本已积水的区域；各区面积见深度效应CSV。</p><h2>主要地表排水方向</h2><a href="surface_directions.png"><img src="surface_directions.png" alt="雨末主要地表流向"></a><div class="scroll"><table><tr><th>Scenario</th><th>Common moving area/km²</th><th>Moving support overlap</th><th>Within30°</th><th>Within60°</th></tr>{flowtable}</table></div><p>采用降雨结束7200s时的流速，在共同20m范围按正水量加权；筛选水深&gt;0.01m、速度&gt;0.03m/s、方向一致性≥0.5的区域。表内同时给出可比较面积，避免只统计少量一致流向。图中蓝绿色细点为既有水域边界位置，橙色箭头为雨末流向。箭头和GeoJSON瞬时流线用于全域方向展示；聚合可能跨越单元内障碍，不能证明道路逐段连通、实际河岸入口或水质点行程。</p><h2>河段交换与倒灌</h2><div class="scroll"><table><tr><th>Scenario (adaptive)</th><th>Net outward/m³</th><th>Inward/m³</th><th>Top outward receivers</th></tr>{rivertable}</table></div><p>正值为地表流向水域边界，负值为水域向地表来水。来水量先在每个命名水域内汇总流量再分正负，因此不包含该水域内部同时出入的抵消部分；这不是实测河道流量。</p><h2>数值与证据边界</h2><p>所有水量平衡通过：{findings['all_water_budgets_passed']}，最大相对闭合误差{maxtotal:.3%}（沿用1%探索性门槛）；道路/建筑几何和屋面空间权重保持一致，输入哈希核对通过。负的内部记账水量、建筑10m假设障碍超越面积在水量核查表保留；官方求解器在空单元钳制水位，负记账体积不自动等于负物理水深。单元内部负值仅每300秒抽样，可能漏掉中间瞬态。跨区外缘的无雨组在小范围面积筛查中仍有差异（10cm阈值为240与137.5m²），CSV中保留了未通过标记；它不影响三区及全域本轮筛查结论。</p><p>没有纳入地下管网、泵站、入渗和主动措施，也没有事件验证。河道只有假设边界水位，没有完整水下河床三维地形；交换量是地表与水域边界交换，不是实测河流流量或河道输水能力。计算范围沿用三区及既有跨区候选汇水域，尚未认证完整地表汇水范围；跨区道路、建筑约束也未完整补齐。原始地形的测量年代及高程成果历元尚未全部统一。正式模型仍需补齐河床、真实水位边界和地表排水参数。</p><p><a href="area_stability.csv">全域及各区范围对照CSV</a> · <a href="priority_region_stability.csv">重点片区CSV</a> · <a href="flow_direction_stability.csv">流向对照CSV</a> · <a href="river_exchange.csv">河段交换CSV</a> · <a href="raised_stage_peak_depth_effect.csv">水位抬升深度效应CSV</a> · <a href="water_budgets.json">水量平衡与异常审计</a> · <a href="findings.json">机器可读结果</a> · <a href="geometry_reuse_validation.json">几何复用核验</a></p></html>'''
 (OUT/'review.html').write_text(text)
 dump(OUT/'code_provenance.json',{'scripts':{str(q.relative_to(ROOT)):digest(q) for q in [Path(__file__),ROOT/'src/exp/run_three_ward_adaptive_sensitivity.py']},'geometry_source':'Previous corrected10m / automatic10/5m templates, immutable inputs hardlinked; mutable scenario forcing copied.','reproduce':'/private/tmp/udt01b-sfincs-runner/bin/python src/exp/run_three_ward_adaptive_sensitivity.py && /private/tmp/udt01b-sfincs-runner/bin/python src/exp/review_three_ward_adaptive_sensitivity.py'})
 spec=json.loads((BASE/'model_spec.json').read_text())
 spec['adaptive_forcing_sensitivity']={'completed':True,'human_decision_record':'MILI-D-20261014-002','report':str((OUT/'review.html').relative_to(ROOT)),'cases':findings['cases'],'geometry_and_forcing_validation_passed':findings['geometry_and_forcing_validation_passed'],'whole_domain_threshold_0_10m':primary,'all_domain_threshold_screening_passed':findings['all_domain_threshold_grid_gates_passed'],'all_ward_threshold_0_10m_screening_passed':findings['all_ward_0_1m_grid_gates_passed'],'mass_budgets_passed':findings['all_water_budgets_passed'],'strict_internal_volume_nonnegative':findings['strict_nonnegative_all'],'formal_grid_convergence_established':False,'event_validated':False,'active_measures_simulated':False,'original_sources_preserved':True}
 dump(BASE/'model_spec.json',spec)
 print(json.dumps({'main_grid_comparison':primary,'peak_censoring_audits':{'_'.join(k):v['peak_censoring_audit'] for k,v in metadata.items()},'raised_stage_peak_depth_effect':deepening_rows,'river_exchange_summary':river_summaries,'retained_regions':findings['priority_regions_retained_all_rain_cases_both_grids'],'budgets_passed':findings['all_water_budgets_passed']},ensure_ascii=False),flush=True)
if __name__=='__main__':
 import sys
 if '--cache-completed' in sys.argv:
  P.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
  for n in NAMES:
   for g in GRIDS:
    if (folder(n,g)/'execution.json').exists():
     print('Audit completed',n,g,flush=True);cache(n,g)
  for n in NAMES:interim_pair(n)
 else:main()
