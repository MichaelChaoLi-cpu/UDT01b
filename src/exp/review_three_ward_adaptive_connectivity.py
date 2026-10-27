"""Whole-domain matched10m/adaptive5m surface-flood comparison and audits.
Peak levels are mapped onto the same original0.50m open-ground support for area
comparison; this mapping is not a0.50m hydraulic simulation or event validation.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import json,csv,html,hashlib,collections
import numpy as np
import netCDF4,rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_three_ward_urban_grid import read_run
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_adaptive_connectivity';CON=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';BASE=ROOT/'data/processed/three_ward_sfincs_framework';RUN=BASE/'adaptive_connectivity_comparison';OUT=ROOT/'data/exp/data-briefing/three_ward_adaptive_connectivity';XY=Transformer.from_crs(6668,6677,always_xy=True)
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rows):
 if rows:
  with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def a(v):return np.asarray(np.ma.asarray(v).filled(np.nan),dtype=np.float64)
def write(p,x,tf):
 with rasterio.open(p,'w',driver='GTiff',height=x.shape[0],width=x.shape[1],count=1,dtype='float32',crs=6677,transform=tf,nodata=np.nan,compress='deflate',tiled=True) as d:d.write(x.astype('float32'),1)

def adaptive_run(folder=None):
 folder=folder or RUN/'rain60_low_adaptive_10_5m';ex=json.loads((folder/'execution.json').read_text());sc=json.loads((folder/'scenario_contract.json').read_text());assert ex['returncode']==0 and ex['completion_marker']
 hashes={k:hashlib.sha256((folder/k).read_bytes()).hexdigest()==v for k,v in ex['input_sha256'].items()};assert all(hashes.values())
 with np.load(P/'adaptive_cell_geometry.npz') as d:g={k:d[k] for k in d.files}
 sel=g['mask']==1;size=g['size_m'];cellarea=size**2;stored=[];minvol=np.inf;negmax=0.;debt=np.zeros(len(size),bool)
 with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
  assert np.array_equal(a(d['msk'][:]).astype('uint8'),g['mask']);assert np.max(abs(a(d['mesh2d_face_x'][:])-g['x']))<.001 and np.max(abs(a(d['mesh2d_face_y'][:])-g['y']))<.001
  times=a(d['time'][:]);zs=a(d['zsmax'][0]);zs5=np.full((2024,1540),np.nan);m5=np.zeros((2024,1540),'uint8');debt5=np.zeros((2024,1540),bool)
  for i in range(len(times)):
   v=a(d['subgrid_volume'][i]);stored.append(float(v[sel].sum()));minvol=min(minvol,float(v[sel].min()));negmax=max(negmax,float(-np.minimum(v[sel],0).sum()));debt|=(v<-.001)&sel
  reported_rain=float(np.dot(a(d['cumprcp'][-1])[sel],cellarea[sel]));stored=np.array(stored)
  for k in [10.,5.]:
   ii=np.flatnonzero(size==k);r=g['row_fine'][ii]//10;c=g['col_fine'][ii]//10
   for dr in range(int(k/5)):
    for dc in range(int(k/5)):zs5[r+dr,c+dc]=zs[ii];m5[r+dr,c+dc]=g['mask'][ii];debt5[r+dr,c+dc]=debt[ii]
 sections=json.loads((folder/'flux_sections.json').read_text());net=out=inc=0.;river=collections.Counter()
 with netCDF4.Dataset(folder/'sfincs_his.nc') as d:
  ts=a(d['time'][:]);status=float(a(d['status'][:]).ravel()[0]);names=[str(x).strip(' \x00') for x in netCDF4.chartostring(d['crosssection_name'][:])];assert names==[q['name'] for q in sections]
  groups={name:[i for i,s in enumerate(sections) if s['water']==name] for name in sorted(set(q['water'] for q in sections))}
  for i in range(1,len(ts),1000):
   end=min(i+1000,len(ts));q=np.nan_to_num(a(d['crosssection_discharge'][i:end]));dt=np.diff(ts[i-1:end]);qn=q.sum(1);net+=float(qn@dt);out+=float(np.maximum(qn,0)@dt);inc+=float(np.maximum(-qn,0)@dt)
   for key,inds in groups.items():river[key]+=float(q[:,inds].sum(1)@dt)
  thdxy=np.c_[a(d['thindam_x'][:]),a(d['thindam_y'][:])]
 # Exact intended rain integral;0.4s float32 cumulative-rain reporting is audited separately.
 rain=sc['original_rain_interior_volume_m3'];residual=float(stored[-1]-stored[0]+net-rain);relative=float(abs(residual)/max(rain,out,inc,1))
 budget={'run':folder.name,'total_input_m3':rain,'reported_float32_cumprcp_m3':reported_rain,'cumprcp_reporting_roundoff_m3':reported_rain-rain,'stored_initial_m3':stored[0],'stored_final_m3':stored[-1],'boundary_net_out_m3':net,'residual_m3':residual,'relative_error':relative,'passed':relative<.01 and status==0,'min_cell_volume_m3':minvol,'max_negative_volume_total_m3':negmax,'negative_volume_cells':int(debt.sum()),'strict_nonnegative':minvol>=-.001,'final_map_timestamp_s':float(times[-1]),'engine_status':status,'history_interval_s':float(np.diff(ts).mean()),'flux_integral':'right rectangle at written history times','input_hashes_unchanged':all(hashes.values()),'negative_volume_sampling_s':float(np.max(np.diff(times))) if len(times)>1 else None,'unsampled_negative_volume_possible':True}
 return {'zs':zs5,'m':m5,'debt':debt5,'budget':budget,'river':dict(river),'runtime_s':ex['runtime_seconds'],'thdxy':thdxy}

def boundary_audit(case,run):
 with np.load(P/('regular_face_geometry.npz' if case=='corrected_10m' else 'adaptive_face_geometry.npz')) as q:rr=q['row_fine'];cc=q['col_fine'];di=q['direction'];ll=q['length_pixels'];w=q['native_open_width_m']
 x=-11360+cc*.5;y=-32320-rr*.5;y=np.where(di==1,y-.5,y-ll*.25);x=np.where(di==1,x+ll*.25,x)
 if case!='corrected_10m':
  with np.load(P/'adaptive_cell_geometry.npz') as g,np.load(P/'adaptive_face_geometry.npz') as q:
   ia=q['cell_a'];ib=q['cell_b'];x=(g['x'][ia]+g['x'][ib])/2;y=(g['y'][ia]+g['y'][ib])/2
 # Official give_thindam_information reports neighboring cell-center midpoints,
 # which differ from the physical face midpoint at a coarse/fine interface.
 tree=cKDTree(np.c_[x[w==0],y[w==0]])
 if case=='corrected_10m':
  with netCDF4.Dataset(RUN/('rain60_low_'+case)/'sfincs_his.nc') as d:actual=np.c_[a(d['thindam_x'][:]),a(d['thindam_y'][:])]
 else:actual=run['thdxy']
 dist,ix=tree.query(actual);expected=json.loads((RUN/(case+'_template')/'builder_complete.json').read_text())['fully_building_closed_faces'];rec={'case':case,'expected_closed_faces':expected,'solver_snapped_closed_faces':len(actual),'maximum_distance_to_intended_closed_face_m':float(dist.max()),'unique_snapped_faces':len(np.unique(ix)),'coordinate_definition':'Official solver reports neighboring cell-center midpoints; compared using that same convention','passed':len(actual)==expected and len(np.unique(ix))==expected and float(dist.max())<.01};return rec

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 prior=read_run(BASE/'fixed_runoff_comparison/rain60_low_storage253_10m');regular=read_run(RUN/'rain60_low_corrected_10m');adapt=adaptive_run();runs={'previous10m':prior,'corrected10m':regular,'adaptive10_5m':adapt}
 with rasterio.open(OLD/'active_mask_20m.tif') as d:m=d.read(1);tf=d.transform
 nr,nc=m.shape;area=np.zeros(m.shape);wet={k:{h:np.zeros(m.shape) for h in [.05,.1,.3]} for k in runs};depthsum={k:np.zeros(m.shape) for k in runs};pairs={'face_correction':['previous10m','corrected10m'],'adaptive_grid':['corrected10m','adaptive10_5m']};joint={k:{h:[np.zeros(m.shape),np.zeros(m.shape)] for h in [.05,.1,.3]} for k in pairs};overtop={k:0 for k in runs};debt_open_area={k:0. for k in runs};unrepresented={k:0 for k in runs}
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(CON/'constraints_0_50m.tif') as con:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);h=int(win.height)//40;r0=rr//40;spatial=np.repeat(np.repeat(m[r0:r0+h]==1,40,0),40,1);op=spatial&(cl!=1)&np.isfinite(z);reduce=lambda x:x.reshape(h,40,nc,40).sum((1,3),dtype=np.float64)*.25;area[r0:r0+h]=reduce(op);dd={}
   for key,v in runs.items():
    f=10 if key=='adaptive10_5m' else 20;start=rr//f;count=int(win.height)//f;zs=np.repeat(np.repeat(v['zs'][start:start+count],f,0),f,1);active=np.repeat(np.repeat(v['m'][start:start+count]==1,f,0),f,1);valid=active&np.isfinite(zs);unrepresented[key]+=int((op&~valid).sum());assert not np.any(op&~valid),'Open ground dropped from matched comparison';depth=np.where(op&valid,np.maximum(zs-z,0),0);dd[key]=depth;depthsum[key][r0:r0+h]=reduce(depth)
    overtop[key]+=int(np.count_nonzero(spatial&(cl==1)&valid&(zs>=z+9.99)))
    debt_fine=np.repeat(np.repeat(v['debt'][start:start+count],f,0),f,1);debt_open_area[key]+=float(np.count_nonzero(op&debt_fine)*.25)
    for threshold in wet[key]:wet[key][threshold][r0:r0+h]=reduce(depth>threshold)
   for key,(aa,bb) in pairs.items():
    for threshold in joint[key]:u=dd[aa]>threshold;v=dd[bb]>threshold;joint[key][threshold][0][r0:r0+h]=reduce(u&v);joint[key][threshold][1][r0:r0+h]=reduce(u|v)
   print('Compare common fine terrain',rr,'/',dem.height,flush=True)
 features=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];geoms=[transform(XY.transform,shape(f['geometry'])) for f in features];ward=rasterize([(g,i+1) for i,g in enumerate(geoms)],out_shape=m.shape,transform=tf);scopes={'Whole domain':m==1,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'Cross-ward contributing area':(ward==0)&(m==1)};rows=[]
 for scope,sel in scopes.items():
  for pair,(ka,kb) in pairs.items():
   for threshold in [.05,.1,.3]:
    aa=float(wet[ka][threshold][sel].sum());bb=float(wet[kb][threshold][sel].sum());inter=float(joint[pair][threshold][0][sel].sum());union=float(joint[pair][threshold][1][sel].sum());gap=abs(bb-aa)/max(aa,1);iou=inter/max(union,1)
    rows.append({'scope':scope,'comparison':pair,'reference_model':ka,'comparison_model':kb,'threshold_m':threshold,'reference_area_km2':aa/1e6,'comparison_area_km2':bb/1e6,'relative_area_difference':gap,'wet_iou':iou,'stable_wet_core_km2':inter/1e6,'grid_disagreement_km2':(union-inter)/1e6,'wet_union_km2':union/1e6,'common_open_area_km2':float(area[sel].sum()/1e6),'screening_stability_pass':gap<.1 and iou>.7})
 csvwrite(OUT/'whole_domain_and_ward_comparison.csv',rows)
 regional=[];hot=json.loads((OLD/'priority_accumulation_areas.geojson').read_text())['features']
 for f in hot:
  sel=rasterize([(transform(XY.transform,shape(f['geometry'])),1)],out_shape=m.shape,transform=tf).astype(bool)&(m==1);den=float(area[sel].sum());q={'id':f['properties']['id'],'common_open_area_m2':den}
  for k in runs:q[k+'_wet_fraction']=float(wet[k][.1][sel].sum()/max(den,1));q[k+'_mean_peak_depth_m']=float(depthsum[k][sel].sum()/max(den,1))
  q['retained_both_current_models']=min(q['corrected10m_wet_fraction'],q['adaptive10_5m_wet_fraction'])>=.5;regional.append(q)
 csvwrite(OUT/'regional_stability.csv',regional)
 budgets=[]
 for k,v in runs.items():b=dict(v['budget']);b['experiment']=k;b['building_overtopped_fine_area_m2']=overtop[k]*.25;b['open_area_in_negative_volume_cells_km2']=debt_open_area[k]/1e6;budgets.append(b)
 dump(OUT/'water_budgets.json',budgets);cols=sorted(set().union(*(q.keys() for q in budgets)));csvwrite(OUT/'water_budgets.csv',[{k:q.get(k,'') for k in cols} for q in budgets])
 geometry={k:json.loads((RUN/(k+'_template')/'builder_complete.json').read_text()) for k in ['corrected_10m','adaptive_10_5m']};audits=[boundary_audit('corrected_10m',regular),boundary_audit('adaptive_10_5m',adapt)];dump(OUT/'building_closure_snap_audit.json',audits);assert all(x['passed'] for x in audits)
 input_validation=json.loads((OUT/'geometry_and_forcing_validation.json').read_text());assert input_validation['passed']
 primary=next(q for q in rows if q['scope']=='Whole domain' and q['comparison']=='adaptive_grid' and q['threshold_m']==.1);isolation=next(q for q in rows if q['scope']=='Whole domain' and q['comparison']=='face_correction' and q['threshold_m']==.1)
 findings={'scenario':'Synthetic60mm/h for1h after1h dry warmup,2h recession; matched hypothetical low river stages.','main_grid_comparison':primary,'isolated_face_correction_effect_at10m':isolation,'geometry':geometry,'prior20_10m_area_difference':.1337690850988627,'comparison_to_prior_caveat':'Prior20/10m and current10/adaptive5m are different grid pairs; improved agreement cannot be attributed solely to face correction.','all_current_water_budgets_passed':all(q['passed'] for q in budgets if q['experiment']!='previous10m'),'open_area_in_negative_volume_cells_km2':{k:v/1e6 for k,v in debt_open_area.items()},'strict_nonnegative_current':all(q['strict_nonnegative'] for q in budgets if q['experiment']!='previous10m'),'previous15_regions_retained':sum(q['retained_both_current_models'] for q in regional),'depth_rank_spearman':float(spearmanr([q['corrected10m_mean_peak_depth_m'] for q in regional],[q['adaptive10_5m_mean_peak_depth_m'] for q in regional]).statistic),'all_three_wards_area_gate_passed':all(q['screening_stability_pass'] for q in rows if q['comparison']=='adaptive_grid' and q['threshold_m']==.1 and q['scope'] in ['Chiyoda','Chuo','Minato']),'all_three_thresholds_domain_gate_passed':all(q['screening_stability_pass'] for q in rows if q['comparison']=='adaptive_grid' and q['scope']=='Whole domain'),'formal_grid_convergence_established':False,'event_validated':False,'active_measures_simulated':False,'no_point_surveys':True,'raw_dem_or_building_elevations_modified':False,'surface_pipe_infiltration_pump_removal_included':False,'common_open_support_km2':float(area.sum()/1e6),'geometry_and_forcing_validation_passed':input_validation['passed'],'boundary_geometry_equivalence':json.loads((P/'boundary_geometry_equivalence.json').read_text()),'matched_forcing_nested5_to10_exactly':json.loads((P/'allocation5_complete.json').read_text())['exactly_nested_to_previous10m'],'building_closure_snap_audits':audits,'river_exchange_volume_m3':{k:v['river'] for k,v in runs.items()},'runtime_seconds':{'corrected10m':regular['summary']['runtime_s'],'adaptive10_5m':adapt['runtime_s']}}
 dump(OUT/'findings.json',findings)
 # Raster outputs retain common20m display support, not a fictitious fine hydraulic solution.
 for key in runs:write(P/f'{key}_wet_fraction_20m.tif',np.where(area>0,wet[key][.1]/np.maximum(area,.25),np.nan),tf)
 write(P/'current_grid_disagreement_fraction_20m.tif',np.where(area>0,(joint['adaptive_grid'][.1][1]-joint['adaptive_grid'][.1][0])/np.maximum(area,.25),np.nan),tf)
 write(P/'adaptive_minus_corrected_wet_fraction_20m.tif',np.where(area>0,(wet['adaptive10_5m'][.1]-wet['corrected10m'][.1])/np.maximum(area,.25),np.nan),tf)
 extent=[tf.c,tf.c+nc*20,tf.f-nr*20,tf.f];fig,ax=plt.subplots(1,3,figsize=(15,7),layout='constrained')
 for i,key in enumerate(['corrected10m','adaptive10_5m']):im=ax[i].imshow(np.where(area>0,wet[key][.1]/np.maximum(area,.25),np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1);ax[i].set_title('Corrected whole-domain 10 m' if i==0 else 'Whole-domain 10 m / refined 5 m')
 change=np.where(area>0,(wet['adaptive10_5m'][.1]-wet['corrected10m'][.1])/np.maximum(area,.25),np.nan);jm=ax[2].imshow(change,extent=extent,cmap='RdBu',vmin=-1,vmax=1);ax[2].set_title('Change in wet fraction (adaptive − 10 m)')
 for aa in ax:
  for geom in geoms:
   for g in getattr(geom,'geoms',[geom]):x,y=g.exterior.xy;aa.plot(x,y,color='black',lw=.5)
  for name,geom in zip(['Chiyoda','Chuo','Minato'],geoms):pt=geom.representative_point();aa.text(pt.x,pt.y,name,ha='center',fontsize=9,bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
  aa.set_xlim(extent[0],extent[1]);aa.set_ylim(extent[2],extent[3]);aa.set_aspect('equal');aa.set_xticks([]);aa.set_yticks([]);x0=extent[0]+350;y0=extent[2]+350;aa.plot([x0,x0+2000],[y0,y0],color='black');aa.text(x0+1000,y0+100,'2 km',ha='center',fontsize=9)
 fig.colorbar(im,ax=ax[:2],shrink=.65,label='Open-ground wet fraction (>0.10 m)');fig.colorbar(jm,ax=ax[2],shrink=.65,label='Wet fraction change');fig.savefig(OUT/'whole_domain_comparison.png',dpi=160);plt.close(fig)
 with rasterio.open(P/'automatic_refinement_10m.tif') as d:ref=d.read(1);rt=d.transform
 fig,aa=plt.subplots(figsize=(7,8),layout='constrained');rex=[rt.c,rt.c+ref.shape[1]*10,rt.f-ref.shape[0]*10,rt.f];aa.imshow(np.where(np.repeat(np.repeat(m==1,2,0),2,1),ref,np.nan),extent=rex,cmap=matplotlib.colors.ListedColormap(['#ecf1f4','#d8883a']),vmin=0,vmax=1)
 for geom in geoms:
  for gg in getattr(geom,'geoms',[geom]):xx,yy=gg.exterior.xy;aa.plot(xx,yy,color='#253745',lw=.8)
 for name,geom in zip(['Chiyoda','Chuo','Minato'],geoms):pt=geom.representative_point();aa.text(pt.x,pt.y,name,ha='center',fontsize=11,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
 aa.set_title('Geometry-selected refinement zones (orange: 5 m)');aa.set_aspect('equal');aa.set_xticks([]);aa.set_yticks([]);aa.set_xlim(rex[0],rex[1]);aa.set_ylim(rex[2],rex[3]);fig.savefig(OUT/'automatic_refinement_map.png',dpi=160);plt.close(fig)
 current=geometry['adaptive_10_5m'];table=''.join(f'<tr><td>{html.escape(q["scope"])}</td><td>{q["reference_area_km2"]:.3f}</td><td>{q["comparison_area_km2"]:.3f}</td><td>{q["relative_area_difference"]:.1%}</td><td>{q["wet_iou"]:.1%}</td></tr>' for q in rows if q['comparison']=='adaptive_grid' and q['threshold_m']==.1)
 text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区整体道路连通性与自动5m加密对照</title><style>body{{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.65 sans-serif;color:#182b3b}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #cbd5df}}.scroll{{overflow:auto}}.note{{background:#edf3f8;padding:18px}}a{{color:#075ba7}}</style><h1>三区整体：道路截面修正与自动5m加密</h1><p class="note">修正后全域10m与自动10/5m的积水面积分别为{primary['reference_area_km2']:.3f}和{primary['comparison_area_km2']:.3f}km²；面积差异{primary['relative_area_difference']:.1%}，范围重叠{primary['wet_iou']:.1%}。本情景面积稳定性筛查是否通过：{primary['screening_stability_pass']}。仍不能宣称正式网格收敛或真实暴雨预测精度。</p><p>同一三区及跨区汇水范围、原始0.50m地形和建筑障碍、统一屋面分配、60mm/h一小时暴雨及假设低河道水位。0.10m为各地峰值水深阈值，各地峰值不一定同时出现。</p><img src="whole_domain_comparison.png" alt="三区整体10m与自动5m加密积水范围对照"><div class="scroll"><table><tr><th>Scope</th><th>Corrected10m/km²</th><th>Adaptive10/5m/km²</th><th>Area difference</th><th>Wet overlap</th></tr>{table}</table></div><h2>批量处理了什么</h2><p>道路过流按原始0.50m相邻像元的截面保守重建：界面高程取两侧原有水力高程的较高者，糙率取两侧较高值，再调用未修改的官方253段过流函数。建筑完全封闭的界面使用官方薄壁阻断；这些是既有建筑的表达修正，未新建防洪工程。所有DEM及既有建筑障碍高程均保持原值。</p><p>全域10m有{geometry['corrected_10m']['fully_building_closed_faces']:,}个建筑封闭界面；自动加密后有{current['fully_building_closed_faces']:,}个。求解器实际吸附位置与预定建筑封闭界面核对通过。{current['automatic_refinement_parent_cells']:,}个10m复杂单元按几何规则加密，共{current['cells']:,}个计算单元，其中{current['cells_by_resolution_m']['5']:,}个为5m；选择未使用积水结果或逐点调查。</p><p>原10m多开放空间混合候选面积1.349km²；加密后剩余{current['open_area_inside_residual_flagged_cells_km2']:.3f}km²。此指标是单元内几何表达能力，不是实测错误积水面积；仍保留分离空间共用一个单元水位的局限。</p><img src="automatic_refinement_map.png" alt="三区自动5m加密范围"><h2>区分道路修正与网格变化</h2><p>同为10m时，旧模型与修正模型积水面积分别为{isolation['reference_area_km2']:.3f}和{isolation['comparison_area_km2']:.3f}km²，变化{isolation['relative_area_difference']:.1%}。旧20/10m面积差异13.4%，当前比较为10m/自适应5m；两种网格组合不同，不能把一致性改善全部归因于道路修正。稳定积水核心{primary['stable_wet_core_km2']:.3f}km²，网格差异带{primary['grid_disagreement_km2']:.3f}km²，均为情景统计。</p><h2>输入与数值检查</h2><p>屋面雨水按同一原始0.50m几何分配，5m汇总再严格嵌套到10m，与旧规则逐单元完全一致。加密模型粗单元中心的双线性插值恰为四个5m权重的面积平均；两种模型的官方求解器短试算均须通过逐单元降雨验证。粗单元蓄水表原样复制，细单元使用同一未修改官方蓄水函数重建。</p><p>当前两模型水量收支检查通过：{findings['all_current_water_budgets_passed']}；严格非负水量通过：{findings['strict_nonnegative_current']}。约300s地图采样可能漏掉中间时刻的内部负体积。固定官方求解器在内部体积低于零时仍把水位限制在最低地面；负体积记录不能直接解释为物理负水深，仍需检查其对湿干转换的影响。建筑障碍漫越、负水量及河段交换详见数值检查文件，未通过变化掩膜删除不一致结果。保留既有15片区中的{findings['previous15_regions_retained']}个，深度排序相关{findings['depth_rank_spearman']:.3f}。</p><h2>解释范围</h2><p>本轮是同一合成暴雨情景的整体敏感性检验；没有真实事件校准，也没有管网排水、入渗、泵站或主动措施。建筑几何不能证明现场不存在门洞或地下通道；0.50m降尺度仅用于统一面积统计，不是0.50m水动力求解。一个自适应加密对照不足以证明所有情景正式收敛。</p><p><a href="findings.json">完整结果</a> · <a href="geometry_and_forcing_validation.json">独立输入与几何核验</a> · <a href="whole_domain_and_ward_comparison.csv">三个水深阈值与各区对照</a> · <a href="water_budgets.json">水量和数值检查</a> · <a href="building_closure_snap_audit.json">阻断边界吸附核验</a> · <a href="regional_stability.csv">主要片区稳定性</a></p><p>官方依据：<a href="https://github.com/Deltares-research/cht_sfincs/blob/8d29dcc7d4138e75001ecce1708137791cc3eb6f/cht_sfincs/quadtree_builder.py">固定Deltares网格构建代码</a> · <a href="https://github.com/Deltares/SFINCS/tree/1f1d8286520a8709b2e513854dff18a3616583db">固定官方SFINCS2.4求解器源码</a>。</p></html>'''
 (OUT/'review.html').write_text(text)
 sources=[ROOT/'src/exp/validate_three_ward_adaptive_geometry.py',ROOT/'src/preprocessing/prepare_three_ward_adaptive_connectivity.py',ROOT/'src/preprocessing/prepare_three_ward_adaptive_forcing.py',ROOT/'src/exp/run_three_ward_adaptive_connectivity.py',Path(__file__)];dump(OUT/'code_provenance.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
