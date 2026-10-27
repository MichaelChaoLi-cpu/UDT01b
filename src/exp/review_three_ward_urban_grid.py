"""Area/region/direction stability on common non-building ground, with raw-volume audits."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import json,csv,html,re,hashlib,collections
import numpy as np
import rasterio,netCDF4
from rasterio.features import rasterize,shapes
from rasterio.windows import Window
from shapely.geometry import shape,mapping,LineString
from shapely.ops import transform
from scipy.ndimage import distance_transform_edt,label,binary_dilation
from scipy.stats import spearmanr
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';RUN=ROOT/'data/processed/three_ward_sfincs_framework/urban_grid_comparison';OUT=ROOT/'data/exp/data-briefing/three_ward_urban_grid'
NAMES=['rain60_low','rain100_low','dry_raised','rain60_raised'];XY=Transformer.from_crs(6668,6677,always_xy=True);LL=Transformer.from_crs(6677,6668,always_xy=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rows):
 if rows:
  with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def arr(v):return np.asarray(np.ma.asarray(v[:],dtype=np.float64).filled(np.nan))
def north(v,i=None):return np.flip(np.asarray(np.ma.asarray(v[:] if i is None else v[i],dtype=np.float64).filled(np.nan)),axis=-2)
def write(p,a,tf):
 with rasterio.open(p,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float32',crs=6677,transform=tf,nodata=np.nan,compress='deflate',tiled=True) as d:d.write(a.astype('float32'),1)
def read_run(folder):
 sc=json.loads((folder/'scenario_contract.json').read_text());dx=sc['dx'];ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'];filecheck={n:hashlib.sha256((folder/n).read_bytes()).hexdigest()==s for n,s in ex['input_sha256'].items()};assert all(filecheck.values()),'Input mutated after run'
 with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
  m=north(d['msk']).astype('uint8');
  expected_file=P/f'active_mask_{dx}m.tif' if sc['urban'] else OLD/'active_mask_20m.tif'
  with rasterio.open(expected_file) as expected_ds:assert np.array_equal(m,expected_ds.read(1)),'Engine mask differs from fixed comparison domain'
  interior=m==1;h=north(d['hmax'])[0];z=north(d['zsmax'])[0];ts=arr(d['time']);vol=north(d['subgrid_volume']);stored=np.nansum(vol[:,interior],axis=1,dtype=np.float64);debt=np.any(vol<-.001,axis=0)&interior;neg=float(np.max(np.nansum(-np.minimum(vol[:,interior],0),axis=1)));minvol=float(np.nanmin(vol[:,interior]));i=int(np.argmin(abs(ts-7200)));u=north(d['u'],i);v=north(d['v'],i);depth=vol[i]/dx**2;rain=float(np.nansum(north(d['cumprcp'])[-1][interior],dtype=np.float64)*dx**2)
 sections=json.loads((folder/'flux_sections.json').read_text());groups=sorted(set(q['water'] for q in sections));groupids={q:[i for i,a in enumerate(sections) if a['water']==q] for q in groups};river=collections.Counter();net=out=inc=0.
 with netCDF4.Dataset(folder/'sfincs_his.nc') as d:
  qt=arr(d['time']);status=float(arr(d['status'])[0]);names=[str(x).strip(' \x00') for x in netCDF4.chartostring(d['crosssection_name'][:])];assert names==[q['name'] for q in sections]
  for i in range(1,len(qt),1000):
   end=min(i+1000,len(qt));q=np.asarray(np.ma.asarray(d['crosssection_discharge'][i:end],dtype=np.float64).filled(0));dt=np.diff(qt[i-1:end]);qnet=q.sum(axis=1,dtype=np.float64);net+=float(qnet@dt);out+=float(np.maximum(qnet,0)@dt);inc+=float(np.maximum(-qnet,0)@dt)
   for g,ids in groupids.items():river[g]+=float(q[:,ids].sum(axis=1,dtype=np.float64)@dt)
  if 'point_discharge' in d.variables:pass
 # Roof .dis values are rates, use the exact written file as mass input.
 roof=0.
 if (folder/'sfincs.dis').exists():
  ds=np.loadtxt(folder/'sfincs.dis');src=np.loadtxt(folder/'sfincs.src',usecols=[0,1]);ll,bot,ri,tt=sc['bounds'];rr=((tt-src[:,1])/dx).astype(int);cc=((src[:,0]-ll)/dx).astype(int);assert len(src)==ds.shape[1]-1 and np.all((rr>=0)&(rr<m.shape[0])&(cc>=0)&(cc<m.shape[1])) and np.all(m[rr,cc]==1),'Roof sources outside active interior';roof=float(np.trapezoid(ds[:,1:].sum(axis=1),ds[:,0]))
 reported_rain=rain
 if (folder/'sfincs.precip').exists():
  precip_input=np.loadtxt(folder/'sfincs.precip');rain_depth_mm=float(np.trapezoid(precip_input[:,1],precip_input[:,0]))/3600.;rain=float(interior.sum()*dx**2*rain_depth_mm/1000.)
 else:
  # Effective surface rain includes fixed-fine roof runoff. Galibier-specific
  # corner coordinates and ascending rows are verified in the input pilot.
  with netCDF4.Dataset(folder/'roof_surface_precip.nc') as d:
   sec=arr(d['time'])*60.;rate=arr(d['Precipitation']);rate=np.nan_to_num(rate);rate=np.flip(rate[:,1:-1,1:-1],axis=1);assert rate.shape[1:]==m.shape and np.all(rate[:,~interior]==0)
   volume_rate=rate[:,interior].sum(axis=1,dtype=np.float64)*dx**2/3.6e6;rain=float(np.trapezoid(volume_rate,sec))
 expected=sc['original_rain_interior_volume_m3'];actual=rain+roof;residual=float(stored[-1]-stored[0]+net-actual);scale=max(actual,out,inc,1.);budget={'run':folder.name,'dx_m':dx,'rain_ground_m3':rain,'reported_float32_cumprcp_volume_m3':reported_rain,'cumprcp_reporting_roundoff_m3':reported_rain-rain,'roof_source_m3':roof,'total_input_m3':actual,'expected_original_rain_m3':expected,'rain_conservation_relative_error':abs(actual-expected)/max(expected,1.),'boundary_net_out_m3':net,'stored_initial_m3':float(stored[0]),'stored_final_m3':float(stored[-1]),'residual_m3':residual,'relative_error':abs(residual)/scale,'passed':abs(residual)/scale<.01 and status==0,'min_cell_volume_m3':minvol,'max_negative_volume_total_m3':neg,'negative_volume_cells':int(debt.sum()),'strict_nonnegative':minvol>=-.001,'internal_volume_sampling_interval_seconds':300,'negative_volumes_may_be_missed_between_snapshots':True,'history_interval_s':float(np.diff(qt).mean()),'flux_integral':'right_rectangle_at_0.4s_history','final_map_timestamp_s':float(ts[-1]),'file_hashes_unchanged':all(filecheck.values())}
 summary={'run':folder.name,'dx_m':dx,'scenario':sc['name'],'interior_area_km2':float(interior.sum()*dx**2/1e6),'native_cell_mean_depth_gt0_10_km2':float(((h>.1)&interior).sum()*dx**2/1e6),'native_cell_mean_depth_gt0_30_km2':float(((h>.3)&interior).sum()*dx**2/1e6),'peak_mean_depth_max_m':float(np.nanmax(h[interior])),'runtime_s':ex['runtime_seconds']}
 return {'sc':sc,'m':m,'h':h,'zs':z,'debt':debt,'u':u,'v':v,'depth':depth,'budget':budget,'summary':summary,'river':{('海域（未命名wtr）' if k=='(unnamed water body)' else k):v for k,v in river.items()}}

def main():
 OUT.mkdir(exist_ok=True)
 with rasterio.open(OLD/'active_mask_20m.tif') as d:mask=d.read(1);tf=d.transform
 nr,nc=mask.shape;l,b,r,t=rasterio.transform.array_bounds(nr,nc,tf);safe=(mask==1)&(distance_transform_edt(mask!=2)*20>40)&(distance_transform_edt(mask>0)*20>40)
 runs={f'{dx}_{s}':read_run(RUN/f'urban_{s}_{dx}m') for dx in [20,10] for s in NAMES};runs['control']=read_run(RUN/'terrain64_rain60_low_20m');budgets=[v['budget'] for v in runs.values()];csvwrite(OUT/'water_budgets.csv',budgets);dump(OUT/'water_budgets.json',budgets)
 debt20=np.zeros(mask.shape,dtype=bool)
 for k,a in runs.items():
  if k=='control':continue
  d=a['debt'];debt20|=d if d.shape==mask.shape else d.reshape(nr,2,nc,2).any(axis=(1,3))
 # Audit finite building barriers over the whole domain before choosing comparison support.
 barrier20=np.zeros(mask.shape,dtype=bool);barrier_by_case={k:np.zeros_like(a['m'],dtype=bool) for k,a in runs.items() if k!='control'}
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(P/'constraints_0_50m.tif') as con:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);base=np.where((cl==1)&np.isfinite(z),z,np.inf)
   for dx in [20,10]:
    n=int(dx/.5);r0=rr//n;nh=int(win.height)//n;mw=base.shape[1]//n;minimum=base.reshape(nh,n,mw,n).min(axis=(1,3))
    for name in NAMES:
     a=runs[f'{dx}_{name}'];barrier_by_case[f'{dx}_{name}'][r0:r0+nh]=(a['m'][r0:r0+nh]==1)&(a['zs'][r0:r0+nh]>=minimum+9.99)
 for k,bb in barrier_by_case.items():barrier20|=bb if bb.shape==mask.shape else bb.reshape(nr,2,nc,2).any(axis=(1,3))
 barrier_uncertain=binary_dilation(barrier20,iterations=2)&(mask==1)
 trusted=safe&~debt20&~barrier_uncertain;area=np.zeros(mask.shape,dtype='float64');known=np.zeros_like(area);roads=np.zeros_like(area);data={k:{'wet':np.zeros_like(area),'depthsum':np.zeros_like(area),'excess':np.zeros_like(area)} for k in runs};metrics={s:collections.Counter() for s in NAMES};threshold_metrics={(s,h):collections.Counter() for s in NAMES for h in [.05,.3]};jointareas={s:{'intersection':np.zeros_like(area),'union':np.zeros_like(area)} for s in NAMES};stagechange={dx:0. for dx in [20,10]};overtop={k:0 for k in runs};stablewet=np.zeros_like(area)
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(P/'constraints_0_50m.tif') as con:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);nrows=int(win.height)//40;r0=rr//40;spatial=np.repeat(np.repeat(mask[r0:r0+nrows]==1,40,axis=0),40,axis=1);open_=spatial&(cl!=1)&np.isfinite(z);trust=np.repeat(np.repeat(trusted[r0:r0+nrows],40,axis=0),40,axis=1)&open_;reduce=lambda x:x.reshape(nrows,40,nc,40).sum(axis=(1,3),dtype=np.float64)*.25
   area[r0:r0+nrows]=reduce(open_);known[r0:r0+nrows]=reduce(trust);roads[r0:r0+nrows]=reduce(trust&(cl==2));depths={}
   for k,a in runs.items():
    dx=a['sc']['dx'];n=int(dx/.5);start=rr//n;count=int(win.height)//n;zs=np.repeat(np.repeat(a['zs'][start:start+count],n,axis=0),n,axis=1);active=np.repeat(np.repeat(a['m'][start:start+count]==1,n,axis=0),n,axis=1);depth=np.maximum(zs-z,0);depth=np.where(open_&active&np.isfinite(depth),depth,0);depths[k]=depth;data[k]['wet'][r0:r0+nrows]=reduce(depth>.1);data[k]['depthsum'][r0:r0+nrows]=reduce(depth);data[k]['excess'][r0:r0+nrows]=reduce(np.maximum(depth-.1,0));overtop[k]+=int(np.count_nonzero(spatial&(cl==1)&active&np.isfinite(zs)&(zs>=z+9.99)))
   for s in NAMES:
    d20=depths[f'20_{s}'];d10=depths[f'10_{s}'];a20=d20>.1;a10=d10>.1;met=metrics[s];met['area20_m2']+=float(np.count_nonzero(a20&trust)*.25);met['area10_m2']+=float(np.count_nonzero(a10&trust)*.25);met['intersection_m2']+=float(np.count_nonzero(a20&a10&trust)*.25);met['union_m2']+=float(np.count_nonzero((a20|a10)&trust)*.25);met['common_open_area_m2']+=float(trust.sum()*.25);met['depth_abs_difference_sum_m3']+=float(np.abs(d20-d10)[trust].sum()*.25);met['raw_area20_m2']+=float(np.count_nonzero(a20&open_)*.25);met['raw_area10_m2']+=float(np.count_nonzero(a10&open_)*.25);met['raw_intersection_m2']+=float(np.count_nonzero(a20&a10&open_)*.25);met['raw_union_m2']+=float(np.count_nonzero((a20|a10)&open_)*.25);jointareas[s]['intersection'][r0:r0+nrows]=reduce(a20&a10&open_);jointareas[s]['union'][r0:r0+nrows]=reduce((a20|a10)&open_)
    for threshold in [.05,.3]:
     aa=d20>threshold;bb=d10>threshold;qq=threshold_metrics[(s,threshold)]
     for support,suffix in [(open_,'raw'),(trust,'qualified')]:
      qq[f'{suffix}_area20_m2']+=float((aa&support).sum()*.25);qq[f'{suffix}_area10_m2']+=float((bb&support).sum()*.25);qq[f'{suffix}_intersection_m2']+=float((aa&bb&support).sum()*.25);qq[f'{suffix}_union_m2']+=float(((aa|bb)&support).sum()*.25)

   allwet=np.logical_and.reduce([depths[f'{dx}_{s}']>.1 for dx in [20,10] for s in ['rain60_low','rain100_low','rain60_raised']]);stablewet[r0:r0+nrows]=reduce(allwet&trust)
   for dx in [20,10]:stagechange[dx]+=float(np.count_nonzero((depths[f'{dx}_rain60_raised']-depths[f'{dx}_rain60_low']>.05)&trust)*.25)
   print('Downscale comparison rows',rr,'/',dem.height,flush=True)
 comparisons=[]
 for s,q in metrics.items():
  a=dict(q);a.update({'scenario':s,'flood_threshold_m':.1,'raw_wet_area_iou':a['raw_intersection_m2']/max(a['raw_union_m2'],1),'raw_relative_area_difference':abs(a['raw_area10_m2']-a['raw_area20_m2'])/max(a['raw_area20_m2'],1),'fine_wet_area_iou':a['intersection_m2']/max(a['union_m2'],1),'relative_area_difference':abs(a['area10_m2']-a['area20_m2'])/max(a['area20_m2'],1),'mean_abs_open_ground_peak_difference_m':a['depth_abs_difference_sum_m3']/max(a['common_open_area_m2'],1),'screening_area_stability_pass':abs(a['area10_m2']-a['area20_m2'])/max(a['area20_m2'],1)<.1 and a['intersection_m2']/max(a['union_m2'],1)>.7});comparisons.append(a)
 csvwrite(OUT/'grid_comparison.csv',comparisons);dump(OUT/'grid_comparison.json',comparisons)
 threshold_rows=[]
 for (scenario,threshold),q in threshold_metrics.items():
  threshold_rows.append({'scenario':scenario,'threshold_m':threshold,'raw_area20_km2':q['raw_area20_m2']/1e6,'raw_area10_km2':q['raw_area10_m2']/1e6,'raw_relative_area_difference':abs(q['raw_area10_m2']-q['raw_area20_m2'])/max(q['raw_area20_m2'],1),'raw_wet_iou':q['raw_intersection_m2']/max(q['raw_union_m2'],1),'qualified_area20_km2':q['qualified_area20_m2']/1e6,'qualified_area10_km2':q['qualified_area10_m2']/1e6,'qualified_relative_area_difference':abs(q['qualified_area10_m2']-q['qualified_area20_m2'])/max(q['qualified_area20_m2'],1),'qualified_wet_iou':q['qualified_intersection_m2']/max(q['qualified_union_m2'],1)})
 for q in comparisons:threshold_rows.append({'scenario':q['scenario'],'threshold_m':.1,'raw_area20_km2':q['raw_area20_m2']/1e6,'raw_area10_km2':q['raw_area10_m2']/1e6,'raw_relative_area_difference':q['raw_relative_area_difference'],'raw_wet_iou':q['raw_wet_area_iou'],'qualified_area20_km2':q['area20_m2']/1e6,'qualified_area10_km2':q['area10_m2']/1e6,'qualified_relative_area_difference':q['relative_area_difference'],'qualified_wet_iou':q['fine_wet_area_iou']})
 csvwrite(OUT/'depth_threshold_comparison.csv',threshold_rows)

 wards=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];wg=[transform(XY.transform,shape(f['geometry'])) for f in wards];wid=rasterize([(g,i+1) for i,g in enumerate(wg)],out_shape=mask.shape,transform=tf);wr=[]
 for k,a in runs.items():
  for i,name in enumerate(['千代田区','中央区','港区','跨区汇水部分'],1):
   sel=(wid==i) if i<4 else (wid==0)&(mask==1);row={'run':a['summary']['run'],'ward':name,'open_ground_interior_km2':float(area[sel].sum()/1e6),'open_ground_peak_depth_gt0_10_km2':float(data[k]['wet'][sel].sum()/1e6),'trusted_open_ground_peak_depth_gt0_10_km2':float(data[k]['wet'][sel&trusted].sum()/1e6)};wr.append(row)
  a['summary'].update({'open_ground_peak_depth_gt0_10_km2':float(data[k]['wet'].sum()/1e6),'trusted_open_ground_peak_depth_gt0_10_km2':float(data[k]['wet'][trusted].sum()/1e6),'building_barrier_overtopped_fine_pixels':overtop[k]});write(P/f'{a["summary"]["run"]}_open_mean_peak_depth_20m.tif',np.where(area>0,data[k]['depthsum']/np.maximum(area,1),np.nan),tf)
 
 for a in runs.values():
  dx=a['sc']['dx'];ntf=rasterio.transform.from_origin(l,t,dx,dx);write(P/f'{a["summary"]["run"]}_native_mean_peak_depth_{dx}m.tif',np.where(a['m']==1,a['h'],np.nan),ntf)
 csvwrite(OUT/'scenario_summary.csv',[a['summary'] for a in runs.values()]);csvwrite(OUT/'ward_summary.csv',wr)
 lookup={(q['run'],q['ward']):q for q in wr};wardcomp=[]
 for scenario in NAMES:
  for name in ['千代田区','中央区','港区','跨区汇水部分']:
   a20=lookup[(f'urban_{scenario}_20m',name)];a10=lookup[(f'urban_{scenario}_10m',name)];i=['千代田区','中央区','港区','跨区汇水部分'].index(name)+1;region=(wid==i) if i<4 else wid==0;sel=region&trusted;aa=a20['trusted_open_ground_peak_depth_gt0_10_km2'];bb=a10['trusted_open_ground_peak_depth_gt0_10_km2'];iou=float(jointareas[scenario]['intersection'][sel].sum()/max(jointareas[scenario]['union'][sel].sum(),1));wardcomp.append({'scenario':scenario,'ward':name,'raw_area20_km2':a20['open_ground_peak_depth_gt0_10_km2'],'raw_area10_km2':a10['open_ground_peak_depth_gt0_10_km2'],'trusted_area20_km2':aa,'trusted_area10_km2':bb,'relative_area_difference':abs(bb-aa)/max(aa,1e-6),'trusted_wet_iou':iou,'screening_area_stability_pass':abs(bb-aa)/max(aa,1e-6)<.1 and iou>.7})
 csvwrite(OUT/'ward_grid_comparison.csv',wardcomp)
 
 riverrows=[{'run':a['summary']['run'],'receiving_water':name,'net_land_out_m3':value,'interpretation':'hypothetical boundary exchange, not measured river flow'} for a in runs.values() for name,value in a['river'].items()];csvwrite(OUT/'receiver_exchange.csv',riverrows)
 oldhot=json.loads((OLD/'priority_accumulation_areas.geojson').read_text())['features'];hotrows=[]
 hints={q['id']:q for q in csv.DictReader((ROOT/'data/exp/data-briefing/three_ward_sfincs_scenarios/priority_accumulation_areas.csv').open(encoding='utf-8-sig'))}
 for f in oldhot:
  ident=f['properties']['id'];g=transform(XY.transform,shape(f['geometry']));sel=rasterize([(g,1)],out_shape=mask.shape,transform=tf).astype(bool)&trusted;aa=area[sel].sum();q={'id':ident,'location_hint':hints[ident]['location_hint'],'common_trusted_open_area_km2':float(aa/1e6)}
  for dx in [20,10]:
   q[f'mean_peak_depth_{dx}m']=float(data[f'{dx}_rain60_low']['depthsum'][sel].sum()/max(aa,1));q[f'peak_excess_volume_proxy_{dx}m_m3']=float(data[f'{dx}_rain60_low']['excess'][sel].sum());q[f'min_wet_fraction_3rain_cases_{dx}m']=float(min(data[f'{dx}_{s}']['wet'][sel].sum()/max(aa,1) for s in ['rain60_low','rain100_low','rain60_raised']))
  q['retains_broad_priority_both_grids']=q['min_wet_fraction_3rain_cases_20m']>=.5 and q['min_wet_fraction_3rain_cases_10m']>=.5;hotrows.append(q)
 for dx in [20,10]:
  order=sorted(range(len(hotrows)),key=lambda j:hotrows[j][f'peak_excess_volume_proxy_{dx}m_m3'],reverse=True)
  for rank,j in enumerate(order,1):hotrows[j][f'priority_rank_{dx}m']=rank
 csvwrite(OUT/'prior_area_stability.csv',hotrows);rankcorr=float(spearmanr([q['priority_rank_20m'] for q in hotrows],[q['priority_rank_10m'] for q in hotrows]).statistic)
 # Compare coarse velocity snapshots on the same ground support, not streamline counts.
 a=runs['20_rain60_low'];bb=runs['10_rain60_low'];average=lambda ar:np.divide(np.nansum(ar.reshape(nr,2,nc,2),axis=(1,3)),np.isfinite(ar).reshape(nr,2,nc,2).sum(axis=(1,3)),out=np.full((nr,nc),np.nan),where=np.isfinite(ar).reshape(nr,2,nc,2).sum(axis=(1,3))>0);u=average(bb['u']);v=average(bb['v']);dep=average(bb['depth']);speed0=np.hypot(a['u'],a['v']);speed1=np.hypot(u,v);sel=trusted&(area>0)&(a['depth']>.01)&(dep>.01)&(speed0>.03)&(speed1>.03);cos=np.clip((a['u']*u+a['v']*v)/np.maximum(speed0*speed1,1e-20),-1,1);angle=np.degrees(np.arccos(cos));weights=area[sel];roadsel=sel&(roads>20)
 direction={'snapshot_time_s':7200,'common_valid_20m_cells':int(sel.sum()),'mean_angle_difference_deg_area_weighted':float(np.average(angle[sel],weights=weights)) if sel.any() else None,'direction_within30deg_fraction':float(np.average(angle[sel]<=30,weights=weights)) if sel.any() else None,'direction_within60deg_fraction':float(np.average(angle[sel]<=60,weights=weights)) if sel.any() else None,'road_valid_cells':int(roadsel.sum()),'road_direction_within30deg_fraction':float(np.average(angle[roadsel]<=30,weights=area[roadsel])) if roadsel.any() else None,'interpretation':'rain-end instantaneous velocity directions; not particle tracks, bank openings or measured drainage'};dump(OUT/'flow_direction_comparison.json',direction)
 paths=[];pathstats=[]
 with rasterio.open(P/'constraints_0_50m.tif') as con:
  for dx in [20,10]:
   aa=runs[f'{dx}_rain60_low'];count=collections.Counter()
   for rr in range(4,nr,8):
    for cc in range(4,nc,8):
     x0=l+(cc+.5)*20;y0=t-(rr+.5)*20
     if not trusted[rr,cc] or next(con.sample([(x0,y0)]))[0]==1:continue
     pts=[(x0,y0)];seen=set();reason='length_limit'
     for n in range(250):
      x,y=pts[-1];row=int((t-y)//dx);col=int((x-l)//dx)
      if not (0<=row<aa['m'].shape[0] and 0<=col<aa['m'].shape[1]):reason='domain_edge';break
      if aa['m'][row,col]!=1:reason='boundary_or_inactive';break
      pr=int((t-y)//20);pc=int((x-l)//20)
      if not (0<=pr<nr and 0<=pc<nc) or not trusted[pr,pc]:reason='identified_uncertainty_or_edge';break
      if next(con.sample([(x,y)]))[0]==1:reason='building_or_unresolved_subcell_turn';break
      ux,vy=aa['u'][row,col],aa['v'][row,col];sp=np.hypot(ux,vy)
      if not np.isfinite(sp) or sp<.03 or aa['depth'][row,col]<.01:reason='slow_or_dry';break
      bin_=(int(x//5),int(y//5))
      if bin_ in seen:reason='local_loop';break
      seen.add(bin_);pts.append((x+5*ux/sp,y+5*vy/sp))
     if len(pts)>3:
      paths.append({'type':'Feature','geometry':mapping(transform(LL.transform,LineString(pts))),'properties':{'dx_m':dx,'seed_20m_row':rr,'seed_20m_col':cc,'stop':reason,'snapshot_s':7200}});count[reason]+=1
   pathstats.append({'dx_m':dx,'total':sum(count.values()),'termination':dict(count),'shared_seed_spacing_m':160})
 dump(P/'surface_flow_paths_20_10m.geojson',{'type':'FeatureCollection','features':paths});dump(OUT/'flow_path_summary.json',pathstats)
 # Broad areas persistent across all six wet scenarios; exclude edge/numeric debt cells.
 persistent=trusted&(stablewet>=area*.25)&(stablewet>=20);labs,num=label(persistent);sizes=np.bincount(labs.ravel());order=[i for i in np.argsort(sizes)[::-1] if i and sizes[i]>=25][:20];regions=[]
 for j,i in enumerate(order,1):
  ss=labs==i;polys=[shape(g) for g,val in shapes(ss.astype('uint8'),mask=ss,transform=tf) if val==1]
  from shapely.ops import unary_union
  geom=unary_union(polys);row={'id':f'U{j:02d}','grid_envelope_km2':float(ss.sum()*400/1e6),'common_fine_flood_area_km2':float(stablewet[ss].sum()/1e6),'criterion':'>=25% open ground wet>0.1m in six urban rain scenarios; at least1ha connected20m envelope','validated_event':False};regions.append({'type':'Feature','properties':row,'geometry':mapping(transform(LL.transform,geom))})
 dump(P/'stable_broad_accumulation_areas.geojson',{'type':'FeatureCollection','features':regions})
 geometryeffect={'common_open_mean_peak_60mm_terrain64_m':float(data['control']['depthsum'][trusted].sum()/max(area[trusted].sum(),1)),'common_open_mean_peak_60mm_urban20_m':float(data['20_rain60_low']['depthsum'][trusted].sum()/max(area[trusted].sum(),1)),'common_open_wet_area_terrain64_km2':float(data['control']['wet'][trusted].sum()/1e6),'common_open_wet_area_urban20_km2':float(data['20_rain60_low']['wet'][trusted].sum()/1e6),'building_road_effect_separated_from_subgrid_nlevels':True}
 low_receivers={str(dx):sorted(runs[f'{dx}_rain60_low']['river'].items(),key=lambda q:q[1],reverse=True) for dx in [20,10]}
 result={'low_stage_receiver_ranking':low_receivers,'scope':'three wards plus same original connected cross-ward candidate domain','budgets_all_passed':all(q['passed'] for q in budgets),'rain_volume_conservation_passed':all(q['rain_conservation_relative_error']<.001 for q in budgets),'strict_internal_nonnegative_all':all(q['strict_nonnegative'] for q in budgets),'negative_volume_union_20m_blocks_excluded':int(debt20.sum()),'building_barrier_overtopped_union_20m_blocks':int(barrier20.sum()),'building_barrier_uncertain_plus40m_area_km2':float(barrier_uncertain.sum()*400/1e6),'comparison_trusted_open_area_km2':float(known.sum()/1e6),'grid_area_stability_all_wet_cases':all(q['screening_area_stability_pass'] for q in comparisons if q['scenario']!='dry_raised'),'ward_area_stability_all_wet_cases':{name:all(q['screening_area_stability_pass'] for q in wardcomp if q['ward']==name and q['scenario']!='dry_raised') for name in ['千代田区','中央区','港区']},'prior_area_priority_rank_spearman':rankcorr,'prior_areas_retained':int(sum(q['retains_broad_priority_both_grids'] for q in hotrows)),'prior_areas_total':len(hotrows),'stable_broad_areas_count':len(regions),'stable_broad_areas_max_output':20,'stable_regions_are_20m_envelopes_not_verified_fine_connected_water':True,'river_stage_raise_depth_increase_gt0_05_area_km2':{str(dx):v/1e6 for dx,v in stagechange.items()},'geometry_effect':geometryeffect,'direction':direction,'no_pointwise_survey':True,'event_validation':False,'building_obstacle_height_m_assumed':10,'barrier_overtopping_checks':overtop,'limitations':['Subgrid cells do not resolve all disconnected pockets or narrow street turns.','Finite assumed building obstacles were overtopped in flagged cells. Those cells plus40m neighbourhood are excluded from trusted comparisons; downstream effects have not been proven absent. Raw full-domain comparison remains available.','Ground road constraint is classified plan geometry and n=.03; original DEM not carved and no kerb walls invented.','Roof rain is immediate within partially covered cells; fully covered cells use nearest open-cell sources, not measured downpipes.','Original20m receiver strip retained at10m for controlled comparison; river bank geometry not refined.','Cross-ward exterior has terrain and some PLATEAU objects; building/road completeness has not been established.','Synthetic rainfall/stages; pipes, pumps, measured river sections and event replay absent.']};dump(OUT/'findings.json',result);write(P/'constraint_sensitivity_flags_20m.tif',(mask==1).astype('uint8')*(~safe).astype('uint8')+debt20.astype('uint8')*2+barrier_uncertain.astype('uint8')*4,tf)
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False;extent=[l/1000,r/1000,b/1000,t/1000]
 fig,axes=plt.subplots(1,3,figsize=(13,7),layout='constrained')
 for ax,key,title in zip(axes,['control','20_rain60_low','10_rain60_low'],['纯地形 20 m（同参数）','建筑＋地面道路 20 m','建筑＋地面道路 10 m']):
  a=np.where(area>0,data[key]['depthsum']/np.maximum(area,1),np.nan);im=ax.imshow(a,extent=extent,vmin=0,vmax=1,cmap='Blues');ax.set_title(title);ax.set_xlabel('JGD2011 IX 东向 / km');ax.set_aspect('equal');ax.imshow(np.where((mask==1)&(~safe|debt20|barrier_uncertain),1,np.nan),extent=extent,cmap=matplotlib.colors.ListedColormap(['#888888']),alpha=.35,vmin=0,vmax=1)
  for g in wg:
   gs=list(g.geoms) if hasattr(g,'geoms') else [g]
   for q in gs:ax.plot(np.asarray(q.exterior.coords)[:,0]/1000,np.asarray(q.exterior.coords)[:,1]/1000,color='#555',lw=.7)
  for name,g in zip(['千代田区','中央区','港区'],wg):
   point=g.representative_point();ax.text(point.x/1000,point.y/1000,name,fontsize=9,ha='center',bbox={'facecolor':'white','edgecolor':'none','alpha':.7})
 axes[0].set_ylabel('北向 / km');fig.colorbar(im,ax=axes,label='开放地面平均峰值水深 / m（共同20 m单元汇总）',shrink=.6,extend='max');fig.suptitle('三区 60 mm/h × 1 h 情景：建筑、道路与网格对照\n灰色：已识别的边界、数值或建筑障碍敏感区');fig.savefig(OUT/'urban_grid_peak_comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,10),layout='constrained');dep20=np.where(area>0,data['20_rain60_low']['depthsum']/np.maximum(area,1),np.nan);dep10=np.where(area>0,data['10_rain60_low']['depthsum']/np.maximum(area,1),np.nan);im=axes[0].imshow(dep10-dep20,extent=extent,vmin=-.3,vmax=.3,cmap='RdBu');axes[0].set_title('10 m − 20 m：开放地面平均峰值差');fig.colorbar(im,ax=axes[0],label='m',shrink=.6,extend='both');im=axes[1].imshow(np.where(sel,angle,np.nan),extent=extent,vmin=0,vmax=90,cmap='viridis');axes[1].set_title('降雨结束时汇流方向差');fig.colorbar(im,ax=axes[1],label='角度 / 度',shrink=.6)
 for ax in axes:ax.set_xlabel('东向 / km');ax.set_aspect('equal')
 fig.savefig(OUT/'urban_grid_stability.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,10),layout='constrained')
 for ax,dx in zip(axes,[20,10]):
  ax.imshow(np.where(area>0,data[f'{dx}_rain60_low']['depthsum']/np.maximum(area,1),np.nan),extent=extent,vmin=0,vmax=.7,cmap='Blues')
  for feature in paths:
   if feature['properties']['dx_m']!=dx:continue
   coords=np.array(feature['geometry']['coordinates']);xx,yy=XY.transform(coords[:,0],coords[:,1]);ax.plot(xx/1000,yy/1000,color='#e67722',lw=.55,alpha=.65)
  ax.set_title(f'{dx} m：共同种子、降雨结束流速路径');ax.set_xlabel('东向 / km');ax.set_aspect('equal')
 fig.savefig(OUT/'urban_surface_routes.png',dpi=160);plt.close(fig)
 def table(rows):
  return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(k))+'</th>' for k in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{v:.4g}' if isinstance(v,float) else str(v))+'</td>' for v in q.values())+'</tr>' for q in rows)+'</tbody></table></div>'
 labels={'rain60_low':'60 mm/h，低水位','rain100_low':'100 mm/h，低水位','dry_raised':'无雨，水位+1.5 m','rain60_raised':'60 mm/h，水位+1.5 m'}
 comparison_display=[{'情景':labels[q['scenario']],'20 m 积水 km²':q['area20_m2']/1e6,'10 m 积水 km²':q['area10_m2']/1e6,'面积差 %':q['relative_area_difference']*100,'湿区 IoU':q['fine_wet_area_iou'],'全域未筛选 IoU':q['raw_wet_area_iou'],'本轮面积稳定':q['screening_area_stability_pass']} for q in comparisons]
 hot_display=[{'原编号':q['id'],'区域提示':q['location_hint'],'20 m 排序':q['priority_rank_20m'],'10 m 排序':q['priority_rank_10m'],'两网格均保留':q['retains_broad_priority_both_grids']} for q in hotrows]
 ward_display=[{'区':q['ward'],'20 m 可比较积水 km²':q['trusted_area20_km2'],'10 m 可比较积水 km²':q['trusted_area10_km2'],'面积差 %':q['relative_area_difference']*100,'湿区 IoU':q['trusted_wet_iou']} for q in wardcomp if q['scenario']=='rain60_low' and q['ward']!='跨区汇水部分']
 budget_display=[{'计算':q['run'],'水量残差 %':q['relative_error']*100,'雨量误差 %':q['rain_conservation_relative_error']*100,'5分钟输出中负体积网格':q['negative_volume_cells'],'收支检查':q['passed']} for q in budgets]
 text=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区建筑、道路与10m对照</title><style>body{{font:16px/1.65 sans-serif;color:#18313d;max-width:1200px;margin:30px auto;padding:0 18px}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ccd5da;white-space:nowrap}}a{{color:#087688}}</style><h1>三区建筑、地面道路与10 m对照</h1><p>保持三区整体视角，沿用原汇水域和假设水位。采用同一0.50 m采样地形、原始建筑底面与道路结构分类；10 m水动力网格并不等于0.50 m流场精度。</p><p>雨量守恒检查：{result['rain_volume_conservation_passed']}；水量收支检查：{result['budgets_all_passed']}；严格内部体积非负：{result['strict_internal_nonnegative_all']}。比较剔除了边界40 m和出现负体积的{result['negative_volume_union_20m_blocks_excluded']}个20 m块，另有{result['building_barrier_overtopped_union_20m_blocks']}个20 m块达到假设建筑障碍顶高，周围40 m一并标为约束敏感区。共同可比较开放地面（已剔除识别的不确定区）{result['comparison_trusted_open_area_km2']:.2f} km²；全域未筛选统计同时保留在CSV中。</p><p>三区共同口径的网格面积稳定性判据：面积差&lt;10%，湿区IoU&gt;0.70。三个雨情景均通过：{result['grid_area_stability_all_wet_cases']}。原15个候选区中{result['prior_areas_retained']}个在两种网格及三种雨情景下至少一半开放地面仍超过0.10 m；区域优先级Spearman={rankcorr:.3f}。</p><img src="urban_grid_peak_comparison.png" alt="三区峰值对照"><h2>共同开放地面的网格稳定性</h2>{table(comparison_display)}<img src="urban_grid_stability.png" alt="水深与方向差"><img src="urban_surface_routes.png" alt="共同出发点的地表汇流路径"><p>橙色线为同一时刻的流速场路径示意；遇建筑或未解析的子网格转向停止，并非真实排水口连接或水粒子轨迹。</p><h2>分区对照：60 mm/h、低水位</h2>{table(ward_display)}<h2>原候选区的区域对照</h2>{table(hot_display)}<h2>数值与雨量检查</h2>{table(budget_display)}<h2>输入和解释边界</h2><p>建筑占地采用地面相对+10 m的有限高度障碍情景，保留原始地形；达到顶高的区域另行标记，不能解释为有效实体阻水。完全覆盖网格的屋面雨量按就近开放网格恢复；部分覆盖网格假设屋面径流即时进入本格开放空间。道路仅用结构分类限制地面道路范围与糙率n=0.03，其他地面n=0.04；LOD1道路Z=0不作高程。桥梁、高架、地下段不当作地表道路。子网格64级表在对照中保持一致。</p><p>河道水位仍为情景，水边20 m条带保持原模型布局。管网、泵站、真实屋面排水口尚缺，跨区建筑覆盖完整性尚未认证；实际岸顶、事件验证尚未解决。DEM独立精度和成果时期仍未核实，网格对照只检验同一输入下的敏感性。本次不逐点核查，也不据此宣布街道实测风险或数值收敛。</p><p>官方子网格方法说明：<a href="https://sfincs.readthedocs.io/en/latest/developments.html">SFINCS subgrid</a>；运行固定于2.4.0 Galibier镜像摘要，输入哈希和执行记录保留。</p><p><a href="findings.json">完整判断与限制</a> · <a href="grid_comparison.csv">网格对照CSV</a> · <a href="depth_threshold_comparison.csv">不同积水深度阈值CSV</a> · <a href="ward_summary.csv">分区面积CSV</a> · <a href="ward_grid_comparison.csv">分区网格对照CSV</a> · <a href="prior_area_stability.csv">候选区稳定性CSV</a> · <a href="water_budgets.csv">水量检查CSV</a> · <a href="{os.path.relpath(P/'stable_broad_accumulation_areas.geojson',OUT)}">稳定积水区GeoJSON</a> · <a href="{os.path.relpath(P/'surface_flow_paths_20_10m.geojson',OUT)}">两种网格汇流路径GeoJSON</a></p>'''
 (OUT/'review.html').write_text(text);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
def replot_overview():
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 with rasterio.open(OLD/'active_mask_20m.tif') as d:mask=d.read(1);bounds=d.bounds
 with rasterio.open(P/'constraint_sensitivity_flags_20m.tif') as d:flags=d.read(1)
 extent=[bounds.left/1000,bounds.right/1000,bounds.bottom/1000,bounds.top/1000];ward_features=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];wg=[transform(XY.transform,shape(f['geometry'])) for f in ward_features]
 fig,axes=plt.subplots(1,3,figsize=(13,7),layout='constrained')
 for ax,key,title in zip(axes,['terrain64_rain60_low_20m','urban_rain60_low_20m','urban_rain60_low_10m'],['纯地形20 m（同参数）','建筑＋地面道路20 m','建筑＋地面道路10 m']):
  with rasterio.open(P/f'{key}_open_mean_peak_depth_20m.tif') as d:a=d.read(1)
  im=ax.imshow(a,extent=extent,vmin=0,vmax=1,cmap='Blues');ax.imshow(np.where((mask==1)&(flags>0),1,np.nan),extent=extent,cmap=matplotlib.colors.ListedColormap(['#888888']),alpha=.35,vmin=0,vmax=1);ax.set_title(title);ax.set_xlabel('JGD2011 IX 东向 / km');ax.set_aspect('equal')
  for name,g in zip(['千代田区','中央区','港区'],wg):
   gs=list(g.geoms) if hasattr(g,'geoms') else [g]
   for q in gs:ax.plot(np.asarray(q.exterior.coords)[:,0]/1000,np.asarray(q.exterior.coords)[:,1]/1000,color='#555',lw=.7)
   point=g.representative_point();ax.text(point.x/1000,point.y/1000,name,fontsize=9,ha='center',bbox={'facecolor':'white','edgecolor':'none','alpha':.7})
 axes[0].set_ylabel('北向 / km');fig.colorbar(im,ax=axes,label='开放地面平均峰值水深 / m（共同20 m单元汇总）',shrink=.75,extend='max');fig.suptitle('三区60 mm/h × 1 h情景：建筑、道路与网格对照\n灰色：已识别的边界、数值或建筑障碍敏感区');fig.savefig(OUT/'urban_grid_peak_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':
 import sys
 replot_overview() if '--plot-only' in sys.argv else main()
