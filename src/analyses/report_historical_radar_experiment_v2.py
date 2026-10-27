"""Whole-three-ward historical-rainfall / hypothetical-stage comparison.
Peak exposure is not observed flooded depth, a shutdown decision or a probability.
"""
from pathlib import Path
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2');os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import sys,csv,json,html,math,hashlib,re
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from compare_urban_geometry_pair_controls_v1 import road_depth_bounds
from audit_three_ward_negative_volumes import arr,structured,fine_peaks
P=ROOT/'data/processed/three_ward_historical_radar_experiment_v2';R=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v2'
CASES=['radar_low','radar_raised'];SCOPES={'Whole domain':None,'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3,'Cross-ward buffer':0}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rr(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 if rows:
  with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
 manifest=json.loads((P/'run_manifest.json').read_text());runs=manifest['runs'];assert [r['case_id'] for r in runs]==CASES
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];road=support['Road Ground Area'];ward=support['Ward Code']
 audits=[json.loads((R/'execution'/case/'audit.json').read_text()) for case in CASES];assert all(a['all_checks_passed'] for a in audits)
 runtime_qualification=dict(all_passed=True,cases=CASES,horizon_s=32400,unit_reader_qualification_sha256=sha(R/'solver_qualification.json'),audit_sha256={case:sha(R/'execution'/case/'audit.json') for case in CASES},scope='Actual block-rainfall CPU solver, original numerical tolerances,9h experiment only; no formal reference-library release')
 dump(R/'runtime_qualification.json',runtime_qualification)
 sb=json.loads((P/'solver_binding.json').read_text());sb.update(hydraulic_end_to_end_qualified=True,qualified_cases=CASES,runtime_qualification_sha256=sha(R/'runtime_qualification.json'));dump(P/'solver_binding.json',sb)
 labels=[np.load(P/'execution'/(case+'_candidate_reference.npz')) for case in CASES]
 lower=np.stack([l['Peak Mean Depth Lower'] for l in labels]);upper=np.stack([l['Peak Mean Depth Upper'] for l in labels])
 good=valid.copy()
 for lab in labels:good &=lab['Admitted Mask']&~lab['Quarantine Mask']
 peaks=[];minimum=[];snapshots=[];velocities=[];budgets=[]
 for i,(run,audit) in enumerate(zip(runs,audits)):
  folder=ROOT/run['folder']
  with Dataset(folder/'sfincs_map.nc') as d:
   peaks.append(np.flipud(arr(d['zsmax'][0])));minimum.append(np.flipud(arr(d['zb'][:])))
   # Fixed overview time19:00JST (t=14400s), chosen independently of outcomes.
   actual_clock=np.load(P/'execution'/(run['case_id']+'_output_clock.npz'))['Actual Map Time'];ix=int(np.argmin(abs(actual_clock-14400)));assert abs(actual_clock[ix]-14400)<=.001;snapshots.append(np.flipud(arr(d['zs'][ix])))
   velocities.append((np.flipud(arr(d['u'][ix])),np.flipud(arr(d['v'][ix]))))
  led=structured(folder/'udt_step_budget.csv');last=led[-1];budgets.append(dict(case_id=run['case_id'],rain_m3=float(last['rain_applied_cumulative']),final_storage_m3=float(last['signed_storage']),outflow_m3=float(last['boundary_out_cumulative']),inflow_m3=float(last['boundary_in_cumulative']),final_budget_residual_m3=float(last['budget_residual']),maximum_abs_prefix_residual_m3=audit['summary']['max_budget_residual_m3'],minimum_native_volume_m3=audit['summary']['minimum_volume_m3'],negative_cells=audit['summary']['negative_cells'],actual_steps=audit['summary']['actual_steps'],actual_final_time_s=float(last['actual_state_time']),runtime_s=audit['summary']['runtime_s']))
 peaks=np.stack(peaks);minimum=np.stack(minimum);snapshots=np.stack(snapshots)
 rlo=np.full_like(lower,np.nan);rup=rlo.copy();ra=np.zeros_like(area);snapshot_lo=np.full_like(lower,np.nan)
 with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cl,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/hydraulic_obstacle_0_50m.tif') as hyd:
  tf=rasterio.transform.from_origin(dem.transform.c,dem.transform.f,10,10)
  for row in range(0,1012,32):
   n=min(32,1012-row);win=Window(0,row*20,15400,n*20);z=dem.read(1,window=win);classes=cl.read(1,window=win)
   a,b,c=road_depth_bounds(z,classes,valid[row:row+n],peaks[:,row:row+n],minimum[:,row:row+n]);rlo[:,row:row+n]=a;rup[:,row:row+n]=b;ra[row:row+n]=c
   sl,su,ga,_,_=fine_peaks(z,classes,hyd.read(1,window=win),valid[row:row+n],snapshots[:,row:row+n],minimum[:,row:row+n]);assert np.array_equal(ga,area[row:row+n]);snapshot_lo[:,row:row+n]=sl
 assert np.array_equal(ra,road)
 np.savez_compressed(P/'paired_exposure_bounds.npz',ground_lower=lower,ground_upper=upper,road_lower=rlo,road_upper=rup,common_admitted=good,ground_area=area,road_area=ra,snapshot_19jst_lower=snapshot_lo)
 for k,case in enumerate(CASES):
  profile=dict(driver='GTiff',height=1012,width=770,count=1,dtype='float32',crs=6677,transform=tf,compress='deflate',nodata=np.nan)
  for name,a in [('ground_peak_lower',lower[k]),('ground_peak_upper',upper[k]),('road_peak_lower',rlo[k])]:
   with rasterio.open(P/(case+'_'+name+'.tif'),'w',**profile) as f:f.write(np.where(good,a,np.nan).astype('f4'),1)
 metrics=[];coverage=[];differences=[]
 for scope,code in SCOPES.items():
  region=valid if code is None else valid&(ward>0) if code==-1 else valid&(ward==code);use=region&good
  coverage.append(dict(scope=scope,eligible_open_ground_km2=float(area[region].sum()/1e6),common_admitted_km2=float(area[use].sum()/1e6),excluded_km2=float(area[region&~good].sum()/1e6),low_quarantine_km2=float(area[region&labels[0]['Quarantine Mask']].sum()/1e6),raised_quarantine_km2=float(area[region&labels[1]['Quarantine Mask']].sum()/1e6)))
  for name,ll,uu,weights in [('Open ground',lower,upper,area),('Classified ground roads',rlo,rup,ra)]:
   selected=use&(weights>0);w=weights[selected];lo=ll[:,selected];up=uu[:,selected];mass=float(w.sum())
   if not mass:continue
   dl=lo[1]-up[0];du=up[1]-lo[0];differences.append(dict(scope=scope,support=name,area_km2=mass/1e6,mean_raised_minus_low_lower_m=float(w@dl/mass),mean_raised_minus_low_upper_m=float(w@du/mass),certain_increase_ge0p1m_km2=float(w[dl>=.1].sum()/1e6),certain_decrease_ge0p1m_km2=float(w[du<=-.1].sum()/1e6)))
   for th in [.1,.3,.5]:
    tl=lo[0]>th;tu=up[0]>th;rl=lo[1]>th;ru=up[1]>th
    interlo=float(w[tl&rl].sum());interup=float(w[tu&ru].sum());unionlo=float(w[tl|rl].sum());unionup=float(w[tu|ru].sum())
    metrics.append(dict(scope=scope,support=name,threshold_m=th,admitted_km2=mass/1e6,low_certain_km2=float(w[tl].sum()/1e6),low_possible_km2=float(w[tu].sum()/1e6),raised_certain_km2=float(w[rl].sum()/1e6),raised_possible_km2=float(w[ru].sum()/1e6),certainly_added_by_raised_km2=float(w[rl&~tu].sum()/1e6),certainly_removed_by_raised_km2=float(w[tl&~ru].sum()/1e6),jaccard_lower=interlo/unionup if unionup else 1.,jaccard_upper=min(1.,interup/unionlo) if unionlo else 1.))
 entrances=[]
 for ent in rr(ROOT/'data/processed/three_ward_station_surface_risk_v1/entrance_screen_summary.csv'):
  x=float(ent['x_m']);y=float(ent['y_m']);row=int(math.floor((-32320-y)/10));col=int(math.floor((x+11360)/10));r0=max(0,row-3);r1=min(1012,row+4);c0=max(0,col-3);c1=min(770,col+4)
  if r1<=r0 or c1<=c0:ri=ci=np.array([],int)
  else:
   ri,ci=np.mgrid[r0:r1,c0:c1];sel=((-11360+(ci+.5)*10-x)**2+(-32320-(ri+.5)*10-y)**2<=400)&valid[ri,ci];ri,ci=ri[sel],ci[sel]
  total=float(area[ri,ci].sum());ok=good[ri,ci];ri,ci=ri[ok],ci[ok];admitted=float(area[ri,ci].sum());quality=total>0 and admitted>=100 and admitted/total>=.8
  rec=dict(osm_node_id=ent['osm_node_id'],station_ids_original=ent['station_ids'],radius_m=20,eligible_ground_m2=total,common_admitted_m2=admitted,neighborhood_quality=quality,not_ingress_or_shutdown=True)
  for k,name in enumerate(['low','raised']):
   certain=float(area[ri,ci][lower[k,ri,ci]>=.3].sum());possible=float(area[ri,ci][upper[k,ri,ci]>=.3].sum());rec[name+'_certain_0p3_footprint_m2']=certain;rec[name+'_possible_0p3_footprint_m2']=possible;rec[name+'_screen_trigger']=quality and certain>=100
  entrances.append(rec)
 # Keep the original7 water groups distinct, including duplicate river names.
 from load_three_ward_minimum_scenarios import load_geometry
 geo=load_geometry();exchange=[np.load(P/'execution'/(case+'_group_exchange.npz')) for case in CASES]
 group_rows=[]
 for j,code in enumerate(geo['Boundary Group Codes']):
  row=dict(water_group_code=int(code),source_water_name=str(geo['Boundary Group Names'][j]))
  for k,case in enumerate(CASES):
   assert int(exchange[k]['Group Codes'][j])==int(code)
   final=exchange[k]['Cumulative Directional Volumes'][-1]
   row[case+'_out_m3']=float(final[2*j]);row[case+'_in_m3']=float(final[2*j+1])
  group_rows.append(row)
 write(R/'water_group_exchange.csv',group_rows)
 entrance_summary=dict(inventory_points=len(entrances),quality_passed=sum(e['neighborhood_quality'] for e in entrances),low_trigger_points=sum(e['low_screen_trigger'] for e in entrances),raised_trigger_points=sum(e['raised_screen_trigger'] for e in entrances),added_trigger_points=sum(e['raised_screen_trigger'] and not e['low_screen_trigger'] for e in entrances),removed_trigger_points=sum(e['low_screen_trigger'] and not e['raised_screen_trigger'] for e in entrances),threshold_is_hypothetical_exposure_screen=True,station_closures_inferred=False)
 for name,rows in [('wet_extent.csv',metrics),('quality_coverage.csv',coverage),('depth_change.csv',differences),('water_budgets.csv',budgets),('entrance_exposure.csv',entrances)]:write(R/name,rows)
 # Whole-domain matched maps, fixed ranges and no outcome-selected zoom.
 extent=[-11.36,-3.66,-42.44,-32.32]
 def decorate(ax):
  for code,label in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   ax.contour((ward==code).astype('f4'),levels=[.5],extent=extent,origin='upper',colors='#586b7b',linewidths=.45)
   rows,cols=np.where(ward==code);x=(-11360+(float(cols.mean())+.5)*10)/1000;y=(-32320-(float(rows.mean())+.5)*10)/1000
   ax.text(x,y,label,ha='center',va='center',fontsize=8,bbox=dict(fc='white',alpha=.8,ec='none',pad=1))
  ax.set(xlabel='Easting (km), JGD2011 / IX',ylabel='Northing (km)');ax.set_aspect('equal')
 fig,axes=plt.subplots(1,3,figsize=(14,8),sharex=True,sharey=True,constrained_layout=True)
 for k,ax in enumerate(axes[:2]):
  im=ax.imshow(np.where(good,lower[k],np.nan),extent=extent,origin='upper',vmin=0,vmax=.8,cmap='Blues');ax.set_title(['Historical radar / low stage','Same radar / raised stage'][k]);decorate(ax);fig.colorbar(im,ax=ax,shrink=.7,extend='max',label='Peak ground mean-depth lower bound (m)')
 dl=lower[1]-upper[0];du=upper[1]-lower[0];confirmed=np.where(dl>0,dl,np.where(du<0,du,0));im=axes[2].imshow(np.where(good,confirmed,np.nan),extent=extent,origin='upper',vmin=-.3,vmax=.3,cmap='RdBu_r');axes[2].set_title('Raised minus low: separated bounds');decorate(axes[2]);fig.colorbar(im,ax=axes[2],shrink=.7,extend='both',label='Confirmed signed change (m)')
 fig.suptitle('2024-08-21 radar-driven surface-flood scenarios | 15:00-24:00 JST\nEvent peak envelope; river stages hypothetical; common admitted support only')
 fig.savefig(R/'peak_comparison.png',dpi=165);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,8),sharex=True,sharey=True,constrained_layout=True)
 for k,ax in enumerate(axes):
  im=ax.imshow(np.where(good,snapshot_lo[k],np.nan),extent=extent,origin='upper',vmin=0,vmax=.8,cmap='Blues');ax.set_title(CASES[k]+' | 19:00 JST snapshot');decorate(ax);fig.colorbar(im,ax=ax,shrink=.7,extend='max',label='Snapshot ground mean-depth lower bound (m)')
  ri,ci=np.mgrid[8:1012:16,8:770:16];uu,vv=velocities[k];keep=good[ri,ci]&(snapshot_lo[k,ri,ci]>.1)&np.isfinite(uu[ri,ci])&np.isfinite(vv[ri,ci])&(np.hypot(uu[ri,ci],vv[ri,ci])>.03)
  speed=np.hypot(uu[ri[keep],ci[keep]],vv[ri[keep],ci[keep]])
  ax.quiver((-11360+(ci[keep]+.5)*10)/1000,(-32320-(ri[keep]+.5)*10)/1000,uu[ri[keep],ci[keep]]/speed,vv[ri[keep],ci[keep]]/speed,color='#b74615',scale=35,width=.002)
 fig.suptitle('Surface-flow direction diagnostic at one fixed snapshot\nNormalized instantaneous directions; not trajectories or cumulative drainage paths')
 fig.savefig(R/'flow_snapshot_19jst.png',dpi=165);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
 for run,case in zip(runs,CASES):
  folder=ROOT/run['folder'];led=structured(folder/'udt_step_budget.csv');ix=np.r_[np.arange(0,len(led),100),len(led)-1];hours=15+led['actual_state_time'][ix]/3600
  axes[0].plot(hours,led['signed_storage'][ix]/1e6,label=case+' storage')
  axes[1].plot(hours,led['boundary_out_cumulative'][ix]/1e6,label=case+' gross out');axes[1].plot(hours,led['boundary_in_cumulative'][ix]/1e6,ls='--',label=case+' gross in')
 axes[0].set_ylabel('Whole-domain storage (million m3)');axes[1].set_ylabel('Gross boundary water (million m3)');axes[1].set(xlim=(15,24),xticks=np.arange(15,25),xlabel='Local hour (JST), 24 = next midnight')
 for ax in axes:ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=9)
 fig.savefig(R/'water_evolution.png',dpi=165);plt.close(fig)
 checks=[]
 def check(n,ok):checks.append(dict(check=n,passed=bool(ok)));assert ok,n
 for rel,h in json.loads((P/'protected_sources.json').read_text()).items():check('protected_'+rel,sha(ROOT/rel)==h)
 check('original_numerical_gates_pass_both',all(a['all_checks_passed'] for a in audits))
 check('no_negative_native_volume',all(b['minimum_native_volume_m3']>=0 and b['negative_cells']==0 for b in budgets))
 check('rain_applied_same_between_stage_cases',abs(budgets[0]['rain_m3']-budgets[1]['rain_m3'])<=max(b['rain_m3'] for b in budgets)*1e-6)
 check('ordered_extent_and_jaccard_bounds',all(0<=x['low_certain_km2']<=x['low_possible_km2'] and 0<=x['raised_certain_km2']<=x['raised_possible_km2'] and 0<=x['jaccard_lower']<=x['jaccard_upper']<=1 for x in metrics))
 summary=dict(completed_cases=CASES,all_numerical_gates_passed=True,window_jst=['2024-08-21T15:00:00+09:00','2024-08-22T00:00:00+09:00'],
  whole_three_ward_metrics=[x for x in metrics if x['scope']=='Three wards'],ward_open_ground_gt0p1=[x for x in metrics if x['scope'] in ['Chiyoda','Chuo','Minato'] and x['support']=='Open ground' and x['threshold_m']==.1],quality_coverage=coverage,water_budgets=budgets,water_group_exchange=group_rows,entrance_screen=entrance_summary,
  historical_rain_observed=True,river_stage_observed=False,sewer_pumps_infiltration_included=False,real_world_flood_accuracy_established=False,training=False,sealed_test=False,people_counted=False,formal_reference_release=False,
  numerical_policy='Original tolerances and masks; explicit9h output-horizon and block-forcing extension. Not original4h formal library membership.',
  preserved_aborted_attempt='v1 low-stage run stopped by legacy linear-only diagnostic guard at first step; retained unchanged, no valid outcome. v2 extends the qualified input profile and adds actual-budget/rejection fixtures; no numerical thresholds relaxed.')
 dump(R/'summary.json',summary);dump(R/'comparison_validation.json',dict(all_passed=True,checks=checks))
 body='<h1>历史雨场驱动的三区积水实验</h1><p class="lead">2024年8月21日15:00—24:00 JST，东京都心三区及必要跨区汇水范围，全域10 m城市约束模型。两组沿用同一份约1 km／5分钟雷达雨场，只改变明确标注的水位情景。</p><div class="notice"><b>这是历史降雨驱动的地表积水情景，不是经校准的事件复现。</b><p>管网、泵站、入渗均未计入；全部屋面雨水即时分配到地表，15:00初始地表采用既有干状态假设，并非实测初始积水。河道水位没有本次实测时序，低水位与＋1.5 m过程均为假设。地表暴露不能直接等同于实际停运或受困人数。</p></div>'
 body+='<h2>数值与输入核验</h2><p>5分钟强度按各原始时段保持常值，并对实际计算步精确积分；没有线性平滑、提前起雨或重复加入屋面水量。原完成、实际时钟、正性、雨量、水量平衡、双向交换和标签隔离规则保留；本次单独把输出合同延长到9小时，不修改原4小时训练库合同。</p>'
 body+=table(['情景','施加雨量 m³','末态储水 m³','累计出流 m³','累计入流 m³','末态残差 m³','最小原生体积 m³'],[[b['case_id'],f"{b['rain_m3']:,.2f}",f"{b['final_storage_m3']:,.2f}",f"{b['outflow_m3']:,.2f}",f"{b['inflow_m3']:,.2f}",f"{b['final_budget_residual_m3']:.5f}",b['minimum_native_volume_m3']] for b in budgets])
 body+='<h2>三区主要积水与道路暴露</h2><p>对照只使用两组共同准入的原开放地表。每格深度由在线峰值水位重新积分到0.50 m细地表，保留浅水删失上下界；道路另外按细道路地表积分。表中“确定／可能”仅表示浅水删失等已证明数值区间下的超阈值面积；不包含DEM误差、10 m空间离散误差、降雨和水位假设的全部不确定性，也不是概率或置信区间。</p>'
 body+=table(['范围','支持','阈值 m','低水位面积 km²','抬升水位面积 km²','抬升明确新增 km²','抬升明确减少 km²'],[[x['scope'],x['support'],x['threshold_m'],f"{x['low_certain_km2']:.3f}–{x['low_possible_km2']:.3f}",f"{x['raised_certain_km2']:.3f}–{x['raised_possible_km2']:.3f}",f"{x['certainly_added_by_raised_km2']:.3f}",f"{x['certainly_removed_by_raised_km2']:.3f}"] for x in metrics if x['scope']=='Three wards' or (x['scope'] in ['Chiyoda','Chuo','Minato'] and ((x['support']=='Open ground' and x['threshold_m']==.1) or (x['support']=='Classified ground roads' and x['threshold_m']==.3)))])
 body+=table(['范围','原开放地表 km²','共同准入 km²','剔除 km²'],[[x['scope'],f"{x['eligible_open_ground_km2']:.3f}",f"{x['common_admitted_km2']:.3f}",f"{x['excluded_km2']:.3f}"] for x in coverage])
 body+='<img src="peak_comparison.png" alt="Whole-domain peak comparison"><p>事件峰值包络可出现在不同时刻，不能解释为同一时刻全域同时积水。白色包括域外、建筑及未准入范围，不作零水深处理。</p><h2>地表流向与排水受阻</h2><img src="flow_snapshot_19jst.png" alt="Surface flow direction diagnostic"><p>统一展示19:00 JST快照。箭头按该帧模型瞬时速度归一化，仅表示方向、不表示速度或实际位移，只用于区域流向诊断；尚未核验内部每格面的累计输运，不能据此宣称完整排水轨迹或道路阻断持续时间。</p><img src="water_evolution.png" alt="Storage and boundary exchange">'
 body+=table(['水域组','源水体名称','低水位出／入 m³','抬升水位出／入 m³'],[[x['water_group_code'],x['source_water_name'],f"{x['radar_low_out_m3']:,.0f} / {x['radar_low_in_m3']:,.0f}",f"{x['radar_raised_out_m3']:,.0f} / {x['radar_raised_in_m3']:,.0f}"] for x in group_rows])
 body+='<p>按每个边界面先分出流／入流再累积，避免双向交换抵消。重复的古川名称保留为独立水域组，未命名水体不作猜测匹配。</p>'
 body+='<h2>站口接近区域暴露初筛</h2><p>沿用既有站口清单与20 m格心圆邻域，要求共同有效开放地表≥100 m²、覆盖≥80%；邻域内≥0.3 m足迹达到100 m²时记为暴露筛选触发。该阈值沿用前期压力分析参数，不是运营关闭或人员安全标准，站口归属未知继续保留。</p>'
 body+=table(['站口点清单','共同质量达标','低水位触发点','抬升水位触发点','新增／减少'],[[entrance_summary['inventory_points'],entrance_summary['quality_passed'],entrance_summary['low_trigger_points'],entrance_summary['raised_trigger_points'],str(entrance_summary['added_trigger_points'])+' / '+str(entrance_summary['removed_trigger_points'])]])
 body+='<p>尚未推断地下进水、JR／地铁实际停运、道路关闭长度或受困人数。本次也未计算网络可达性；下一步可把通过数值检查的暴露范围接入既有站口、整站、区段关闭假设，继续区分积水证据与运营假设。</p><h2>版本、局限与复现</h2><p>首版在首步被旧诊断中的线性雨型保护条件中止，文件原样保留。第二版接通已核验的时段积分，并增加真实水量诊断及错误配置拒绝测试；没有放宽水量或负体积门槛。首轮九小时审计还遗漏了旧4小时名义时刻比较，已保留其记录并改用实际状态时钟复核；原0.001秒容差及300.001秒最大输出间隔均保留，原始结果未改写、首组未重跑。城市几何沿用既有数据，未重建成完整2024年现状；站口清单采用2025年官方站线与2026年OSM编辑快照，不能视为2024年完整站口调查。跨区地表范围沿用既有汇水范围，未补入完整上游流域或实测河流入流，其影响仅由假设水位边界代表。原12＋4划分、质量掩膜和人口不确定性均保留；本事件尚未加入正式训练库，没有训练或访问封存测试。3D城市信息已进入几何约束，但本次两组均使用相同城市模型，因此不能从本次比较单独量化3D信息的优势。</p><ul>'
 files=[R/'summary.json',R/'wet_extent.csv',R/'quality_coverage.csv',R/'depth_change.csv',R/'water_budgets.csv',R/'water_group_exchange.csv',R/'entrance_exposure.csv',R/'solver_qualification.json',R/'runtime_qualification.json',R/'clock_adapter_correction.json',R/'comparison_validation.json',R/'execution/radar_low/audit.json',R/'execution/radar_raised/audit.json',P/'experiment_contract.json',P/'run_manifest.json',P/'solver_binding.json',ROOT/'data/exp/data-preprocessing/tokyo_historical_radar_20240821_v1/review.html']
 for f in files:body+='<li><a href="'+os.path.relpath(f,R)+'">'+html.escape(str(f.relative_to(ROOT)))+'</a></li>'
 body+='</ul><p>未独立校准粗糙度、地下排水与河况；雷达与地面站存在偏差，详见雨场核验。输出是当前假设下的潜在暴露，不能宣称实测淹没范围或三维模型优于二维的准确性证据。</p>'
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区历史雷达暴雨实验</title><style>body{font:16px/1.75 system-ui;color:#253b48;background:#f4f7fa;max-width:1200px;margin:auto;padding:32px}h1,h2{line-height:1.35}.lead{font-size:19px}.notice{background:#fff4df;border-left:5px solid #d49c3d;padding:18px}img{max-width:100%;margin:22px 0}h2{margin-top:34px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d7e1ea;text-align:left}th{background:#e6eef5}a{color:#17658c}</style>'+body+'</html>'
 (R/'review.html').write_text(page)
 broken=[x for x in re.findall(r'(?:src|href)="([^"]+)"',page) if not (R/x).resolve().is_file()];assert not broken
 dump(R/'report_link_validation.json',dict(all_passed=True,broken_links=broken))
 files=[p for p in P.rglob('*') if p.is_file()]+[p for p in R.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','independent_validation.json','report.log']]
 scripts=[ROOT/'src/preprocessing'/n for n in ['prepare_historical_radar_solver_v2.py','qualify_prepare_historical_radar_v2.py','radar_budget_profile_fixture_v2.f90']]+[ROOT/'src/analyses'/n for n in ['audit_historical_radar_run_v2.py','run_historical_radar_experiment_v2.py','report_historical_radar_experiment_v2.py','validate_historical_radar_experiment_v2.py','inspect_historical_radar_progress_v2.py','audit_historical_radar_run_v2_actual_clock.py','resume_historical_radar_experiment_v2.py','compare_urban_geometry_pair_controls_v1.py','audit_three_ward_negative_volumes.py']]+[ROOT/'src/preprocessing'/n for n in ['reference_label_policy.py','radar_block_rain_fixture_v1.f90','radar_reader_fixture_v1.f90']]
 cache={};items={}
 for p in files+scripts:
  st=p.stat();key=(st.st_dev,st.st_ino)
  if key not in cache:cache[key]=sha(p)
  items[str(p.relative_to(ROOT))]=cache[key]
 dump(R/'artifact_manifest.json',dict(files_sha256=items,new_completed_runs=2,retained_aborted_v1_attempt=True,training=False,sealed_test=False))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
