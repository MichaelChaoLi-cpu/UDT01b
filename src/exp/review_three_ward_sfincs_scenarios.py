"""Review official SFINCS terrain-led scenarios, budgets, hotspot areas and snapshot flow lines."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import json,csv,html,hashlib,re,collections
import numpy as np
import netCDF4
import rasterio
from rasterio.features import rasterize,shapes
from rasterio.warp import reproject,Resampling
from shapely.geometry import shape,mapping,LineString,Point,Polygon
from shapely.ops import transform,unary_union
from scipy.ndimage import label,generate_binary_structure,distance_transform_edt
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];PRO=ROOT/'data/processed/three_ward_sfincs_scenarios';BASE=ROOT/'data/processed/three_ward_sfincs_framework';OUT=ROOT/'data/exp/data-briefing/three_ward_sfincs_scenarios'
XY=Transformer.from_crs(6668,6677,always_xy=True);LL=Transformer.from_crs(6677,6668,always_xy=True)
CASES=['dry_low','rain60_low','rain100_low','dry_raised','rain60_raised','rain60_closed','rain60_low_dt1']
NAMES={'dry_low':'无雨／低水位','rain60_low':'60mm/h／低水位','rain100_low':'100mm/h／低水位','dry_raised':'无雨／水位抬升','rain60_raised':'60mm/h／水位抬升','rain60_closed':'60mm/h／封闭域检验','rain60_low_dt1':'60mm/h／时间步减半'}
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rows):
 if rows:
  with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def raster(p,a,tf):
 with rasterio.open(p,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float32',crs=6677,transform=tf,nodata=np.nan,compress='deflate',tiled=True) as d:d.write(a.astype('float32'),1)
def array(v):return np.asarray(np.ma.asarray(v[:],dtype=np.float64).filled(np.nan))
def north(v,index=None):
 a=np.asarray(np.ma.asarray(v[:] if index is None else v[index],dtype=np.float64).filled(np.nan));return np.flip(a,axis=-2)
def draw(g,ax,**kw):
 if g.geom_type=='Polygon':ax.fill(*np.asarray(g.exterior.coords).T,**kw)
 elif hasattr(g,'geoms'):
  for q in g.geoms:draw(q,ax,**kw)
def main():
 OUT.mkdir(exist_ok=True);c=json.loads((PRO/'terrain_contract.json').read_text());l,b,r,t=c['bounds']
 with rasterio.open(PRO/'active_mask_20m.tif') as d:mask=d.read(1);tf=d.transform
 with rasterio.open(PRO/'fallback_fraction_20m.tif') as d:fallback=d.read(1)
 interior=mask==1;active=mask>0;edge_uncertain=(distance_transform_edt(mask!=2)*20<=40)&interior
 outer_uncertain=(distance_transform_edt(mask>0)*20<=40)&interior
 safe=interior&~edge_uncertain&~outer_uncertain
 wb=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];ward_names=['千代田区','中央区','港区'];ward_geoms=[transform(XY.transform,shape(f['geometry'])) for f in wb];ward_id=rasterize([(mapping(g),i+1) for i,g in enumerate(ward_geoms)],out_shape=mask.shape,transform=tf)
 water=json.loads((ROOT/'data/processed/central_three_water_network/water_inventory.geojson').read_text())['features'];waters=[(transform(XY.transform,shape(f['geometry'])),f['properties']['name']) for f in water];named=[(g,n) for g,n in waters if n!='(unnamed water body)']
 summary=[];ward_rows=[];peak={};budget=[];vectors=None;execution=[];river_rows=[];numeric_debt=np.zeros_like(interior)
 sections=json.loads((PRO/'flux_section_index.json').read_text())
 with (PRO/'boundary_points.csv').open(encoding='utf-8-sig') as f:bp=list(csv.DictReader(f))
 row_water={(int(x['row_north']),int(x['col_east'])):x['water_name'] for x in bp}
 section_water=[row_water[(x['boundary_row'],x['boundary_col'])] for x in sections]
 for name in CASES:
  actual_run=name
  if name in ['dry_raised','rain60_raised'] and (BASE/('terrain_'+name+'_hisstep_20m')/'execution.json').is_file():actual_run=name+'_hisstep'
  folder=BASE/('terrain_'+actual_run+'_20m');ex=json.loads((folder/'execution.json').read_text());execution.append(ex);sc=json.loads((folder/'scenario_contract.json').read_text())
  with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
   assert np.array_equal(north(d.variables['msk']),np.where(mask==2,1,mask) if name=='rain60_closed' else mask),'Engine mask orientation mismatch'
   hp=np.nan_to_num(north(d.variables['hmax'])[0],nan=0.);hp=np.maximum(hp,0.);hp[~interior]=np.nan;peak[name]=hp;raster(PRO/(name+'_peak_mean_depth_20m.tif'),hp,tf)
   vols=north(d.variables['subgrid_volume']);ts=array(d.variables['time']);solve_mask=active if name=='rain60_closed' else interior
   stored=np.nansum(vols[:,solve_mask],axis=1,dtype=np.float64)
   rain=north(d.variables['cumprcp']);rain=rain[-1] if rain.ndim==3 else rain;actual_rain=float(np.nansum(rain[solve_mask],dtype=np.float64)*400)
   minvol=float(np.nanmin(vols[:,solve_mask]));status=None
   debt_cells=np.any(vols<-.001,axis=0)&interior;numeric_debt|=debt_cells
   negative_total=np.nansum(-np.minimum(vols[:,solve_mask],0),axis=1);negative_max=float(np.max(negative_total))
   row_debt={'negative_internal_volume_cells_sampled_5min':int(debt_cells.sum()),'max_total_negative_internal_volume_m3_sampled_5min':negative_max,'internal_volume_nonnegative_at_sampled_times':minvol>=-.001}
   if name=='rain60_low':
    i=int(np.argmin(abs(ts-7200)));u=north(d.variables['u'],i);v=north(d.variables['v'],i);hm=vols[i]/400;vectors=(u,v,hm,float(ts[i]))
   sample_peak=np.nanmax(vols[:,interior],axis=0)/400
   # hmax is official mean-depth conversion of peak water level, not zsmax-zmin.
   row={'scenario':name,'actual_run':actual_run,'label':NAMES[name],'rain_mm_h':sc['rain_rate_mm_h'],'stage_rise_m':sc['river_stage_rise_m'],'interior_area_mean_depth_gt0_10m_km2':float(np.count_nonzero(hp[interior]>.1)*400/1e6),'interior_area_mean_depth_gt0_30m_km2':float(np.count_nonzero(hp[interior]>.3)*400/1e6),'interior_area_mean_depth_gt1_00m_km2':float(np.count_nonzero(hp[interior]>1.)*400/1e6),'max_mean_depth_m':float(np.nanmax(hp)),'max_interior_storage_m3_sampled_5min':float(np.max(np.nansum(vols[:,interior],axis=1,dtype=np.float64))),'final_interior_storage_m3':float(np.nansum(vols[-1,interior],dtype=np.float64)),'solver_runtime_s':ex['runtime_seconds'],'minimum_cell_volume_m3':minvol}
   row.update(row_debt);summary.append(row)
   for j,n in enumerate(ward_names,1):
    wm=interior&(ward_id==j);ward_rows.append({'scenario':name,'ward':n,'included_interior_km2':float(wm.sum()*400/1e6),'area_mean_depth_gt0_10m_km2':float(((hp>.1)&wm).sum()*400/1e6),'area_mean_depth_gt0_30m_km2':float(((hp>.3)&wm).sum()*400/1e6),'area_mean_depth_gt1_00m_km2':float(((hp>1)&wm).sum()*400/1e6)})
  with netCDF4.Dataset(folder/'sfincs_his.nc') as his:
   names=[str(x).strip(' \x00') for x in netCDF4.chartostring(his.variables['crosssection_name'][:])]
   assert names==[x['name'] for x in sections],'Flux section ordering mismatch'
   status=float(array(his.variables['status'])[0]);qt=array(his.variables['time']);q=array(his.variables['crosssection_discharge']);qnet=np.nansum(q,axis=1,dtype=np.float64)
   cadence=float(np.diff(qt).mean());per_step=actual_run.endswith('_hisstep')
   def integral(series):return float(np.sum(series[1:]*np.diff(qt))) if per_step else float(np.trapezoid(series,qt))
   if name!='rain60_closed':
    for water_name in sorted(set(section_water)):
     indices=[j for j,n in enumerate(section_water) if n==water_name];series=np.nansum(q[:,indices],axis=1,dtype=np.float64);river_rows.append({'scenario':name,'receiving_water':water_name,'net_volume_from_land_m3':integral(series),'peak_net_outflow_m3_s':float(series.max()),'peak_net_inflow_m3_s':float(-series.min()),'interpretation':'hypothetical stage boundary exchange,not observed river discharge or capacity'})
   if name=='rain60_closed':outflow=inflow=netflow=0.
   else:
    outflow=integral(np.maximum(qnet,0));inflow=integral(np.maximum(-qnet,0));netflow=integral(qnet)
   residual=float(stored[-1]-stored[0]+netflow-actual_rain);scale=max(actual_rain,outflow,inflow,1.);budget.append({'scenario':name,'actual_run':actual_run,'stored_initial_m3':float(stored[0]),'stored_final_m3':float(stored[-1]),'actual_model_rain_volume_m3':actual_rain,'net_land_to_boundary_volume_m3':netflow,'net_outflow_positive_portion_m3':outflow,'net_inflow_positive_portion_m3':inflow,'closure_residual_m3':residual,'closure_relative':abs(residual)/scale,'closure_sampling_seconds':cadence,'flux_integral_method':'per_step_right_rectangle' if per_step else 'sampled_trapezoid','status':status,'passed':abs(residual)/scale<.01 and status==0,'internal_volume_nonnegative':minvol>=-.001,'negative_volume_m3_max_sampled':negative_max,'negative_volume_to_rain_ratio':negative_max/actual_rain if actual_rain>0 else None,'negative_volume_to_largest_budget_term_ratio':negative_max/scale})
  csvwrite(OUT/(name+'_volume_timeseries.csv'),[{'time_s':float(x),'land_storage_m3':float(y)} for x,y in zip(ts,stored)])
 csvwrite(OUT/'river_boundary_exchange.csv',river_rows);csvwrite(OUT/'scenario_summary.csv',summary);csvwrite(OUT/'ward_summary.csv',ward_rows);csvwrite(OUT/'water_budgets.csv',budget);dump(OUT/'water_budgets.json',budget)
 delta=peak['rain60_raised']-peak['rain60_low'];raster(PRO/'raised_stage_peak_depth_change_20m.tif',delta,tf)
 changed=safe&((peak['rain60_low']>.1)|(peak['rain60_low_dt1']>.1));diff=abs(peak['rain60_low']-peak['rain60_low_dt1']);a0=np.count_nonzero(peak['rain60_low'][interior]>.1);a1=np.count_nonzero(peak['rain60_low_dt1'][interior]>.1)
 dtcheck={'dtmax_original_s':2.,'dtmax_repeat_s':1.,'actual_step_in_this_run_original_s':.8,'actual_step_in_this_run_repeat_s':.4,'p95_abs_mean_peak_difference_in_wet_interior_m':float(np.nanquantile(diff[changed],.95)),'max_abs_mean_peak_difference_m':float(np.nanmax(diff[interior])),'area_gt0_10m_relative_change':abs(a1-a0)/max(a0,1),'initial_screening_passed':float(np.nanquantile(diff[changed],.95))<.05 and abs(a1-a0)/max(a0,1)<.10,'grid_10_vs20m_convergence_tested':False,'20m_outputs_are_cell_average_not_street_point_depth':True}
 dump(OUT/'numerical_validation.json',{'water_budgets_all_passed':all(x['passed'] for x in budget),'time_step':dtcheck,'active_subgrid_input_gap_filling_required':False,'event_calibration':False,'strict_internal_volume_nonnegative_all_cases':all(x['internal_volume_nonnegative'] for x in budget),'dry_wet_numerical_debt_cells_excluded_from_hotspots':int(numeric_debt.sum())})
 # Priorities use contiguous areas in land away from uncertain model/bank edges.
 stable=safe&~numeric_debt&(peak['rain60_low']>.1)&(peak['rain100_low']>.1)&(peak['rain60_raised']>.1)
 lab,num=label(stable,generate_binary_structure(2,1));sizes=np.bincount(lab.ravel());hot=[];features=[]
 for k in np.argsort(sizes[1:])[::-1]+1:
  if sizes[k]<25:continue
  yy,xx=np.where(lab==k);xm=float(l+(xx.mean()+.5)*20);ym=float(t-(yy.mean()+.5)*20);point=Point(xm,ym);ng,nn=min(named,key=lambda g:g[0].distance(point));dist=float(ng.distance(point));wj=np.bincount(ward_id[yy,xx],minlength=4).argmax();lon,lat=LL.transform(xm,ym)
  hint='';regions=[('新橋・芝大門付近',(139.749,35.654,139.764,35.670)),('赤坂・溜池山王付近',(139.735,35.666,139.744,35.678)),('神田岩本町付近',(139.770,35.688,139.783,35.698)),('日比谷付近',(139.752,35.671,139.764,35.679)),('高輪・品川駅付近',(139.730,35.620,139.743,35.635)),('外神田付近',(139.765,35.698,139.779,35.708)),('西麻布付近',(139.716,35.651,139.730,35.662))]
  for label_name,bb in regions:
   if bb[0]<=lon<=bb[2] and bb[1]<=lat<=bb[3]:hint=label_name;break
  score=peak['rain60_low'][yy,xx];row={'id':f'H{len(hot)+1:02d}','ward':ward_names[wj-1] if wj else '跨区','location_hint':hint,'location_hint_basis':'approximate geographic inference from model coordinates,not official town-boundary attribution','area_km2':float(sizes[k]*400/1e6),'mean_peak_depth_m':float(np.mean(score)),'p95_peak_cell_mean_depth_m':float(np.quantile(score,.95)),'nearby_named_water':nn,'distance_to_named_water_m':round(dist),'longitude':float(lon),'latitude':float(lat),'plateau_fallback_area_fraction':float(np.nanmean(fallback[yy,xx])),'interpretation':'persistent in3synthetic scenarios;no claim of observed flood or actual river entry'}
  hot.append(row);gs=[shape(g) for g,v in shapes((lab==k).astype('uint8'),transform=tf) if v==1];g=unary_union(gs);features.append({'type':'Feature','properties':row,'geometry':mapping(transform(LL.transform,g))})
  if len(hot)>=15:break
 csvwrite(OUT/'priority_accumulation_areas.csv',hot);dump(PRO/'priority_accumulation_areas.geojson',{'type':'FeatureCollection','features':features})
 # Snapshot trajectories from official model velocities, not verified street or inlet routes.
 u,v,hm,snapshot=vectors;speed=np.hypot(u,v);flow=[];paths=[];occupied=set()
 for rr in range(0,mask.shape[0],8):
  for cc in range(0,mask.shape[1],8):
   if not interior[rr,cc] or hm[rr,cc]<.01 or speed[rr,cc]<.03:continue
   pts=[];reason='length_limit';x=l+(cc+.5)*20;y=t-(rr+.5)*20;visited=set()
   for step in range(250):
    ri=int((t-y)/20);ci=int((x-l)/20)
    if ri<0 or ci<0 or ri>=mask.shape[0] or ci>=mask.shape[1]:reason='model_edge';break
    if mask[ri,ci]==2:reason='hypothetical_receiving_boundary';break
    if mask[ri,ci]==0:reason='inactive_or_water_edge';break
    if hm[ri,ci]<.01 or not np.isfinite(speed[ri,ci]) or speed[ri,ci]<.02:reason='slow_or_dry';break
    key=(ri,ci)
    if step%5==0:
     if key in visited:reason='local_loop';break
     visited.add(key)
    pts.append((x,y));length=5.;x+=length*u[ri,ci]/speed[ri,ci];y+=length*v[ri,ci]/speed[ri,ci]
   if len(pts)<12:continue
   seed=(rr//8,cc//8)
   if seed in occupied:continue
   occupied.add(seed);line=LineString(pts);p=Point(pts[-1]);ng,nn=min(named,key=lambda g:g[0].distance(p));props={'id':f'F{len(paths)+1:03d}','snapshot_time_s':snapshot,'snapshot_since_rain_start_min':(snapshot-3600)/60,'length_m':round(line.length,1),'termination':reason,'nearest_named_water_to_end':nn,'distance_end_to_water_m':round(ng.distance(p)),'not_verified_inlet':True,'method':'stationary snapshot streamline of official20m velocity;not a time-dependent parcel track'};paths.append({'type':'Feature','properties':props,'geometry':mapping(transform(LL.transform,line))});flow.append(line)
 dump(PRO/'surface_flow_paths_snapshot.geojson',{'type':'FeatureCollection','features':paths})
 low_exchange=sorted([r for r in river_rows if r['scenario']=='rain60_low'],key=lambda r:-r['net_volume_from_land_m3']);exchange_total=sum(r['net_volume_from_land_m3'] for r in low_exchange);furukawa=next(r for r in low_exchange if r['receiving_water']=='古川')
 dump(OUT/'surface_flow_summary.json',{'snapshot_paths':len(paths),'termination_counts':dict(collections.Counter(p['properties']['termination'] for p in paths)),'receiving_boundary_exchange_for_rain60_low':low_exchange,'interpretation':'conditional snapshot directions and imposed-stage exchange,not verified outfalls or river capacity'})
 # Standard scientific figures, common scale across rain/stage cases.
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 bounds=[0,.05,.1,.2,.5,1.,2.,5.];cmap=ListedColormap(['#eff2ef','#d9efed','#91d8d5','#4eafc2','#377ab5','#5051a0','#853f83']);norm=BoundaryNorm(bounds,cmap.N)
 def base(ax):
  ax.imshow(np.where(active,1.,np.nan),extent=[l,r,b,t],cmap=ListedColormap(['#eeeeea']),origin='upper')
  for g,n in waters:draw(g,ax,color='#73a9c1',alpha=.6,lw=0)
  for g in ward_geoms:
   for p in ([g] if g.geom_type=='Polygon' else g.geoms):ax.plot(*np.asarray(p.exterior.coords).T,color='#555',lw=.6)
  ax.set_xlim(l,r);ax.set_ylim(b,t);ax.set_aspect('equal');ax.set_xlabel('Easting (m, EPSG:6677)');ax.set_ylabel('Northing (m)')
 fig,axs=plt.subplots(1,3,figsize=(15,7),constrained_layout=True);im=None
 for ax,name in zip(axs,['rain60_low','rain100_low','rain60_raised']):
  base(ax);im=ax.imshow(peak[name],extent=[l,r,b,t],origin='upper',cmap=cmap,norm=norm);ax.set_title(NAMES[name]+'\n一小时暴雨＋两小时退水')
  for h in hot[:8]:
   x,y=XY.transform(h['longitude'],h['latitude']);ax.text(x,y,h['id'],fontsize=8,color='#161616')
 fig.colorbar(im,ax=axs,extend='max',label='全时段峰值格点平均水深（m），非街道点水深',shrink=.7);fig.suptitle('三区地形主导积水情景：未计管网抽排／建筑实体障碍');fig.savefig(OUT/'scenario_peak_mean_depth.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,10),constrained_layout=True);base(ax);im=ax.imshow(peak['rain60_low'],extent=[l,r,b,t],origin='upper',cmap=cmap,norm=norm)
 for line in flow:
  xy=np.asarray(line.coords);ax.plot(xy[:,0],xy[:,1],color='#965225',lw=.8,alpha=.7)
  if len(xy)>10:ax.annotate('',xy=xy[-1],xytext=xy[-8],arrowprops=dict(arrowstyle='->',lw=.8,color='#965225'))
 for name in ['古川','神田川','日本橋川','亀島川','隅田川','日比谷濠']:
  parts=[g for g,n in waters if n==name]
  if parts:
   p=max(parts,key=lambda g:g.area).representative_point();ax.text(p.x,p.y,name,fontsize=10,color='#13394e',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
 fig.colorbar(im,ax=ax,extend='max',label='峰值格点平均水深（m）',shrink=.65);ax.set_title('60mm/h情景：雨停时的地表流线\n流线终点不等于已核实的排水入口');fig.savefig(OUT/'surface_flow_paths.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,7),constrained_layout=True);base(axs[0]);base(axs[1]);im=axs[0].imshow(delta,extent=[l,r,b,t],origin='upper',cmap='RdBu_r',vmin=-.5,vmax=.5);axs[0].set_title('水位抬升情景－低水位情景');fig.colorbar(im,ax=axs[0],label='峰值格点平均水深差（m）',shrink=.7);im=axs[1].imshow(np.where(edge_uncertain|outer_uncertain,1,np.where((fallback>.01)&interior,2,np.nan)),extent=[l,r,b,t],origin='upper',cmap=ListedColormap(['#db9446','#ac8cbb']),vmin=1,vmax=2);axs[1].set_title('橙：岸边／域边不确定带\n紫：采用部分PLATEAU补缺');fig.savefig(OUT/'stage_effect_and_uncertainty.png',dpi=160);plt.close(fig)
 findings={'scenario_summary':summary,'priority_hotspots_count':len(hot),'snapshot_paths_count':len(paths),'water_budgets_all_passed':all(x['passed'] for x in budget),'time_step_check':dtcheck,'strict_internal_volume_nonnegative_all_cases':all(x['internal_volume_nonnegative'] for x in budget),'dry_wet_numerical_debt_cells_excluded_from_hotspots':int(numeric_debt.sum()),'known_river_3d':json.loads((OUT/'river_3d_availability.json').read_text()),'stage_newly_above_0_10m_area_km2':float(np.count_nonzero((peak['rain60_raised']>.1)&(peak['rain60_low']<=.1)&interior)*400/1e6),'stage_depth_increase_gt0_05m_area_km2':float(np.count_nonzero((delta>.05)&interior)*400/1e6),'additional_rain_area_gt0_10m_km2':float((np.count_nonzero(peak['rain100_low'][interior]>.1)-np.count_nonzero(peak['rain60_low'][interior]>.1))*400/1e6),'priority_hotspots':hot,'validity':'terrain-led synthetic screening;no true riverbed,pipe drainage,building barriers or event calibration;10m grid convergence pending'}
 dump(OUT/'findings.json',findings)
 lines=['三区官方SFINCS整体暴雨情景试算','使用原始0.50m DEM与PLATEAU5m地形备用层；20m水动力网格、0.50m地形采样，备用源不具有0.50m精度。','模拟1小时干地表边界预热、1小时均匀60/100mm/h暴雨、2小时退水。河位低值为DEM相对包络，抬升情景叠加1.5m脉冲；不是实测T.P.河段时序或2024事件重演。','未知池塘和濠出入口封闭；无管网抽排、无入渗，未把RoofEdge变成建筑实体。河床/断面没有取得，水位边界行及40m邻近区域不作优先热点。']
 lines += [f'{x["label"]}：格点平均峰值深度>0.10m面积{x["interior_area_mean_depth_gt0_10m_km2"]:.3f}km²，>0.30m面积{x["interior_area_mean_depth_gt0_30m_km2"]:.3f}km²。' for x in summary[:5]]
 lines += [f'水量闭合通过：{all(x["passed"] for x in budget)}；严格内部库容非负：{all(x["internal_volume_nonnegative"] for x in budget)}；时间步减半初筛：{dtcheck["initial_screening_passed"]}；10/20m网格收敛尚未验证。',f'稳定热点{len(hot)}处，输出瞬时流线{len(paths)}条。流线来自雨停时模拟速度场，不代表实际雨水口或完整进河路径。','现有126个水体只有LOD0/LOD1几何，39个LOD1高程全部零，无河床、三维河道断面。晴天小流量是用户观察，未作为实测流量输入；暴雨和潮汐可能改变水位及受纳能力。']
 (OUT/'assessment.txt').write_text('\n'.join(lines)+'\n')
 esc=html.escape
 rows=''.join('<tr>'+''.join(f'<td>{esc(str(v))}</td>' for v in [x['label'],f"{x['interior_area_mean_depth_gt0_10m_km2']:.3f}",f"{x['interior_area_mean_depth_gt0_30m_km2']:.3f}"] )+'</tr>' for x in summary)
 hs=''.join('<tr>'+''.join(f'<td>{esc(str(v))}</td>' for v in [x['id'],x['ward'],x['location_hint'],f"{x['area_km2']:.3f}",x['nearby_named_water'],f"{x['distance_to_named_water_m']}m",f"{x['longitude']:.5f}, {x['latitude']:.5f}"])+'</tr>' for x in hot)
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区整体暴雨情景模拟</title><style>body{{margin:0;background:#f5f7f8;color:#243941;font:16px/1.7 -apple-system,sans-serif}}main{{max-width:1200px;margin:auto;padding:26px}}section{{background:white;padding:20px;margin:15px 0;border-radius:8px}}.note{{padding:20px;background:#fff0d5;border-left:5px solid #c58c39}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;min-width:650px}}td,th{{border:1px solid #d8e0e4;padding:9px;text-align:left}}.scroll{{overflow:auto}}a{{color:#176d82}}pre{{white-space:pre-wrap;word-break:break-word}}</style><main><h1>三区整体暴雨情景模拟</h1><div class="note">官方SFINCS2.4引擎／HydroMT-SFINCS1.2.2子网格。结果是地形主导的合成情景初筛，尚未计管网、建筑实体、真实河床和事件校准；10/20m网格收敛尚未验证。地图颜色为20m格点平均水深，不能读作街道精确水深。蓝灰面是现有水域，不参与陆地水深统计。</div><section><h2>情景设置与结果</h2><p>活跃网格陆地约{c['active_land_cell_area_km2']:.2f}km²，其中约{c['boundary_row_area_km2']:.2f}km²为给定水位边界行，内部陆地统计约{c['interior_land_area_km2']:.2f}km²。均匀暴雨持续1小时，前后各有1小时边界预热和2小时退水。低河位是假设的DEM相对包络；抬升情景增加1.5m脉冲。无雨对照区分河水与降雨作用，封闭域及减半时间步用于检验。</p><div class="scroll"><table><tr><th>情景</th><th>平均深度&gt;0.10m面积（km²）</th><th>平均深度&gt;0.30m面积（km²）</th></tr>{rows}</table></div></section><img src="scenario_peak_mean_depth.png" alt="三区暴雨和水位组合的峰值平均水深"><section><h2>优先积水区</h2><p>以下区域在三种有雨情景中均超过0.10m格点平均深度，连片面积至少1ha，避开岸边和域边40m不确定带。位置提示按模型坐标概略定位，不是正式町界归属或灾情记录。邻近水体仅作定位，不表示该处已能排入河流。</p><div class="scroll"><table><tr><th>ID</th><th>区</th><th>位置提示</th><th>面积km²</th><th>邻近水体</th><th>距离</th><th>坐标</th></tr>{hs}</table></div></section><img src="surface_flow_paths.png" alt="官方速度场雨停时的地表流线"><section><h2>地表排水路径</h2><p>在60mm/h低水位情景中，古川边界接收约{furukawa['net_volume_from_land_m3']/10000:.1f}万m³，占模拟边界净出流约{furukawa['net_volume_from_land_m3']/exchange_total*100:.0f}%；其次为海侧与日本橋川边界。大量流线也在陆地低洼处减速汇聚。这些是给定受纳水位下的模拟交换量，不能当作实际河道容量。</p><p>图中流线取自60mm/h低水位情景的雨停时速度场，输出{len(paths)}条候选流线。局部汇聚、进入水位边界、变慢或遇未计算区均记录终止原因；它们不是已证实的雨水口、岸边开口或进河通道。</p></section><img src="stage_effect_and_uncertainty.png" alt="假设水位影响和地形来源不确定区域"><section><h2>河道三维资料是否具备</h2><p>当前126个水体对象仅有LOD0线和LOD1面；39个LOD1高程全零，没有河床面、完整断面或LOD2/3实体。DEM提供岸边地面线索。另有桥梁模型和三维道路附属物；模型包范围内实际发现157个集水桝、99个侧沟、21个排水沟功能记录，尚未按三区精确裁剪，也不代表完整排水设施。柵・壁代码同时含围栏和墙，不能自动当作挡水墙。采用给定河位边界可以先模拟地表，但不能由此计算真实河道容积和输水能力。</p><p>晴天流量小可帮助设定较低起始水位的假设；流量小不等于水位低，潮汐、闸门及暴雨上游来水仍可能影响地表排水。</p></section><section><h2>数值核查与文件</h2><p>水量闭合通过：{all(x['passed'] for x in budget)}；严格内部库容非负：{all(x['internal_volume_nonnegative'] for x in budget)}。已将出现干湿数值透支的格点排除出优先热点；这项数值问题仍保留，不能将初筛当作完整验证。时间步减半初筛通过：{dtcheck['initial_screening_passed']}。水量由官方子网格库容、累积降雨和断面净流量独立核对；抬升情景已按0.8秒计算步输出积分，原先10秒和2秒流量抽样存在混叠偏差。步长控制参数2→1，对应此次实际步长约0.8→0.4秒。尚未进行10/20m网格收敛或实测验证。</p><ul><li><a href="scenario_summary.csv">情景汇总</a>／<a href="ward_summary.csv">按区统计</a>／<a href="river_boundary_exchange.csv">假设河道边界交换量</a></li><li><a href="priority_accumulation_areas.csv">优先积水区清单</a></li><li><a href="water_budgets.json">水量平衡</a>／<a href="numerical_validation.json">数值检验</a></li><li><a href="../../../processed/three_ward_sfincs_scenarios/priority_accumulation_areas.geojson">积水区GeoJSON</a>／<a href="../../../processed/three_ward_sfincs_scenarios/surface_flow_paths_snapshot.geojson">地表流线GeoJSON</a></li></ul></section></main></html>''')
 page=OUT/'review.html';page.write_text(re.sub(r'(<img src="([^"]+)"[^>]*>)',r'<a href="\2" target="_blank">\1</a>',page.read_text()))
 print(json.dumps({'summary':summary,'budgets':budget,'dtcheck':dtcheck,'hotspots':hot[:5],'paths':len(paths)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
