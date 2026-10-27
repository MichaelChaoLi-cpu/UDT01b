"""Common-support, whole-domain grid-sensitivity attribution; no point validation."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import json,csv,html,hashlib
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt,binary_dilation
from scipy.stats import spearmanr
from shapely.geometry import shape
from shapely.ops import transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from review_three_ward_urban_grid import read_run,write,csvwrite,XY
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';P=ROOT/'data/processed/three_ward_urban_constraints';F=ROOT/'data/processed/three_ward_fixed_runoff';BASE=ROOT/'data/processed/three_ward_sfincs_framework';OUT=ROOT/'data/exp/data-briefing/three_ward_fixed_runoff'

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def allocation_diagnostic():
 c=json.loads((F/'contract.json').read_text());original=json.loads((P/'model_contracts.json').read_text());legacy={}
 for dx in [20,10]:
  q=next(x for x in original if x['dx']==dx)
  with rasterio.open(P/f'active_mask_{dx}m.tif') as d:m=d.read(1)
  a=(m==1).astype(float)*dx**2
  for donor in q['roof_donors']:a[donor['row'],donor['col']]+=donor['area_m2']
  legacy[dx]=a
 parent=legacy[10].reshape(legacy[20].shape[0],2,legacy[20].shape[1],2).sum((1,3));c['old_allocation_changed_contributing_area_m2_half_L1']=float(abs(parent-legacy[20]).sum()/2);c['old_grid_change_relocated_rain_volume_60mm_m3']=float(abs(parent-legacy[20]).sum()/2*.06);c['new_grid_change_relocated_rain_volume_60mm_m3']=0.
 dump(OUT/'allocation_diagnostic.json',c);return c

def main():
 OUT.mkdir(exist_ok=True)
 allocation=allocation_diagnostic()
 runs={f'{method}_{dx}':read_run(BASE/('urban_grid_comparison' if method=='old' else 'fixed_runoff_comparison')/(f'urban_rain60_low_{dx}m' if method=='old' else f'rain60_low_storage253_{dx}m' if method=='refined' else f'rain60_low_{dx}m')) for method in ['old','fixed','refined'] for dx in [20,10]}
 with rasterio.open(OLD/'active_mask_20m.tif') as d:m=d.read(1);tf=d.transform
 nr,nc=m.shape;area=np.zeros(m.shape);wet={k:{h:np.zeros(m.shape) for h in [.05,.1,.3]} for k in runs};depthsum={k:np.zeros(m.shape) for k in runs};intersection={q:{h:np.zeros(m.shape) for h in [.05,.1,.3]} for q in ['old','fixed','refined']};union={q:{h:np.zeros(m.shape) for h in [.05,.1,.3]} for q in ['old','fixed','refined']};barrier={k:np.zeros(m.shape,dtype=bool) for k in runs};debt={k:(a['debt'] if a['sc']['dx']==20 else a['debt'].reshape(nr,2,nc,2).any((1,3))) for k,a in runs.items()};overtop={k:0 for k in runs}
 with rasterio.open(OLD/'terrain_sampling_0_50m.tif') as dem,rasterio.open(P/'constraints_0_50m.tif') as con:
  for rr in range(0,dem.height,800):
   win=Window(0,rr,dem.width,min(800,dem.height-rr));z=dem.read(1,window=win);cl=con.read(1,window=win);h=int(win.height)//40;r0=rr//40;spatial=np.repeat(np.repeat(m[r0:r0+h]==1,40,0),40,1);open_=spatial&(cl!=1)&np.isfinite(z);reduce=lambda x:x.reshape(h,40,nc,40).sum((1,3),dtype=np.float64)*.25;area[r0:r0+h]=reduce(open_);dd={}
   for k,a in runs.items():
    f=int(a['sc']['dx']/.5);start=rr//f;count=int(win.height)//f;zs=np.repeat(np.repeat(a['zs'][start:start+count],f,0),f,1);active=np.repeat(np.repeat(a['m'][start:start+count]==1,f,0),f,1);d=np.where(open_&active&np.isfinite(zs),np.maximum(zs-z,0),0);dd[k]=d;depthsum[k][r0:r0+h]=reduce(d)
    b=spatial&(cl==1)&active&np.isfinite(zs)&(zs>=z+9.99);overtop[k]+=int(b.sum());barrier[k][r0:r0+h]=b.reshape(h,40,nc,40).any((1,3))
    for th in wet[k]:wet[k][th][r0:r0+h]=reduce(d>th)
   for method in ['old','fixed','refined']:
    for th in intersection[method]:
     aa=dd[f'{method}_20']>th;bb=dd[f'{method}_10']>th;intersection[method][th][r0:r0+h]=reduce(aa&bb);union[method][th][r0:r0+h]=reduce(aa|bb)
   print('Compare fine rows',rr,'/',dem.height,flush=True)
 # Hold a SINGLE qualified support across before/after; raw full-open results lead.
 edge=(m==1)&((distance_transform_edt(m!=2)*20<=40)|(distance_transform_edt(m>0)*20<=40));all_debt=np.logical_or.reduce(list(debt.values()));all_barrier=binary_dilation(np.logical_or.reduce(list(barrier.values())),iterations=2)&(m==1);qualified=(m==1)&~edge&~all_debt&~all_barrier
 wardfeatures=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];geoms=[transform(XY.transform,shape(f['geometry'])) for f in wardfeatures];wid=rasterize([(g,i+1) for i,g in enumerate(geoms)],out_shape=m.shape,transform=tf)
 scopes={'整体（含跨区汇水范围）':m==1,'千代田区':wid==1,'中央区':wid==2,'港区':wid==3,'跨区汇水部分':(wid==0)&(m==1)};rows=[]
 for scope,sel in scopes.items():
  for support,ss in [('all_open_ground',sel),('shared_qualified_ground',sel&qualified)]:
   for th in [.05,.1,.3]:
    for method in ['old','fixed','refined']:
     aa=float(wet[f'{method}_20'][th][ss].sum());bb=float(wet[f'{method}_10'][th][ss].sum());inter=float(intersection[method][th][ss].sum());un=float(union[method][th][ss].sum());rows.append({'scope':scope,'support':support,'threshold_m':th,'roof_allocation':method,'area20_km2':aa/1e6,'area10_km2':bb/1e6,'relative_area_difference':abs(bb-aa)/max(aa,1),'signed_area_change':(bb-aa)/max(aa,1),'wet_iou':inter/max(un,1),'stable_wet_core_km2':inter/1e6,'grid_disagreement_band_km2':(un-inter)/1e6,'envelope_wet_union_km2':un/1e6,'common_open_area_km2':float(area[ss].sum()/1e6),'exploratory_area_stability_pass':abs(bb-aa)/max(aa,1)<.1 and inter/max(un,1)>.7})
 csvwrite(OUT/'whole_domain_and_ward_comparison.csv',rows)
 budgets=[]
 for k,a in runs.items():
  q=dict(a['budget']);q['experiment']=k;q['rain_ground_m3_includes_roof_runoff']=(not k.startswith('old'));q['building_barrier_overtopped_fine_area_m2']=overtop[k]*.25;q['barrier_sensitive_20m_blocks']=int(barrier[k].sum());budgets.append(q)
 csvwrite(OUT/'water_budgets.csv',budgets);dump(OUT/'water_budgets.json',budgets)
 hot=json.loads((OLD/'priority_accumulation_areas.geojson').read_text())['features'];hotrows=[]
 for f in hot:
  ident=f['properties']['id'];sel=rasterize([(transform(XY.transform,shape(f['geometry'])),1)],out_shape=m.shape,transform=tf).astype(bool)&(m==1);q={'id':ident,'common_open_area_m2':float(area[sel].sum())}
  for k in runs:q[k+'_wet_fraction']=float(wet[k][.1][sel].sum()/max(area[sel].sum(),1));q[k+'_mean_peak_depth_m']=float(depthsum[k][sel].sum()/max(area[sel].sum(),1))
  q['refined_grid_disagreement_km2']=float((union['refined'][.1][sel]-intersection['refined'][.1][sel]).sum()/1e6);q['refined_grid_disagreement_open_fraction']=float((union['refined'][.1][sel]-intersection['refined'][.1][sel]).sum()/max(area[sel].sum(),1))
  q['fixed_retained_both_grids']=min(q['refined_20_wet_fraction'],q['refined_10_wet_fraction'])>=.5;hotrows.append(q)
 csvwrite(OUT/'regional_stability.csv',hotrows)
 primary=[q for q in rows if q['scope']=='整体（含跨区汇水范围）' and q['support']=='all_open_ground' and q['threshold_m']==.1];old=next(q for q in primary if q['roof_allocation']=='old');roof_only=next(q for q in primary if q['roof_allocation']=='fixed');new=next(q for q in primary if q['roof_allocation']=='refined');difference=old['relative_area_difference']-new['relative_area_difference']
 findings={'storm':'synthetic 60 mm/h for one hour; one-hour dry warmup, two-hour recession; unchanged hypothetical low river stages','raw_domain_comparison_before':old,'raw_domain_comparison_after_fixed_roof_only':roof_only,'raw_domain_comparison_after':new,'area_sensitivity_reduction_percentage_points':difference*100,'storage_table_levels_before_after':[64,253],'face_conveyance_functions_unchanged':True,'stable_core_and_uncertainty_band_reported':True,'sensitivity_reduced':difference>0,'exploratory_area_stability_pass':new['exploratory_area_stability_pass'],'three_wards_0_10m_area_gate_passed':all(q['exploratory_area_stability_pass'] for q in rows if q['roof_allocation']=='refined' and q['support']=='all_open_ground' and q['threshold_m']==.1 and q['scope'] in ['千代田区','中央区','港区']),'all_three_thresholds_whole_domain_gate_passed':all(q['exploratory_area_stability_pass'] for q in rows if q['roof_allocation']=='refined' and q['support']=='all_open_ground' and q['scope']=='整体（含跨区汇水范围）'),'formal_grid_convergence_established':False,'reason_formal_convergence_not_established':'Only two hydraulic resolutions and one synthetic storm; no third-grid test or event validation. Roof drainage is a plausible scenario, not observed.','all_mass_budgets_passed':all(q['passed'] for q in budgets),'max_budget_relative_error':max(q['relative_error'] for q in budgets),'max_fixed_budget_relative_error':max(q['relative_error'] for q in budgets if not q['experiment'].startswith('old')),'all_fixed_rain_conserved':all(q['rain_conservation_relative_error']<1e-6 for q in budgets if not q['experiment'].startswith('old')),'strict_nonnegative_refined':all(q['strict_nonnegative'] for q in budgets if q['experiment'].startswith('refined')),'strict_nonnegative_fixed':all(q['strict_nonnegative'] for q in budgets if not q['experiment'].startswith('old')),'fixed_building_barrier_overtopped_area_m2':{k:overtop[k]*.25 for k in runs if not k.startswith('old')},'old_15_regions_retained_at_60mm_h':sum(q['fixed_retained_both_grids'] for q in hotrows),'region_depth_rank_spearman_fixed20_fixed10':float(spearmanr([q['refined_20_mean_peak_depth_m'] for q in hotrows],[q['refined_10_mean_peak_depth_m'] for q in hotrows]).statistic),'whole_open_area_km2':float(area.sum()/1e6),'shared_qualified_area_km2':float(area[qualified].sum()/1e6),'no_point_checks':True,'original_dem_or_hydraulic_obstacles_changed':False,'storage_curve_changed':True,'event_validated':False,'allocation':allocation,'next_gate':'Use a whole-domain10m mesh with automatic5m refinement of geometry-ambiguous building/road cells; retain the same fine geometry, roof distribution and external scenarios. Verify shallow storage and face conveyance accuracy and conservative mesh/forcing transfer. Compare the refined whole-domain model against10m before claiming area convergence. This is an agent recommendation; the adaptive builder/resource route is not yet implemented. Do not tune ground heights to force matching extents.'}
 storage_before=json.loads((OUT/'shallow_storage_audit.json').read_text());storage_after=json.loads((OUT/'shallow_storage_audit_253.json').read_text());connectivity=json.loads((OUT/'subcell_connectivity_audit.json').read_text());findings['storage_precision_audit']={'before':storage_before,'after':storage_after};findings['geometry_only_connectivity_audit']=connectivity;dump(OUT/'findings.json',findings)
 write(F/'refined_stable_wet_core_fraction_20m.tif',np.where(area>0,intersection['refined'][.1]/np.maximum(area,.25),np.nan),tf)
 write(F/'refined_grid_disagreement_fraction_20m.tif',np.where(area>0,(union['refined'][.1]-intersection['refined'][.1])/np.maximum(area,.25),np.nan),tf)
 for k in runs:write(F/f'{k}_open_mean_peak_depth_20m.tif',np.where(area>0,depthsum[k]/np.maximum(area,.25),np.nan),tf)
 write(F/'shared_uncertainty_flags_20m.tif',(edge.astype('uint8')+all_debt*2+all_barrier*4).astype('float32'),tf)
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');ext=[tf.c,tf.c+nc*20,tf.f-nr*20,tf.f];fig,axs=plt.subplots(2,3,figsize=(18,12),layout='constrained')
 for col,method in enumerate(['old','fixed','refined']):
  v=np.where(area>0,wet[f'{method}_10'][.1]/np.maximum(area,.25),np.nan);im=axs[0,col].imshow(v,extent=ext,cmap='Blues',vmin=0,vmax=1);axs[0,col].set_title(('原规则＋64段' if method=='old' else '统一屋面＋64段' if method=='fixed' else '统一屋面＋253段')+'：10 m 网格积水面积占比',fontproperties=font)
  d=np.where(area>0,(wet[f'{method}_10'][.1]-wet[f'{method}_20'][.1])/np.maximum(area,.25),np.nan);jm=axs[1,col].imshow(d,extent=ext,cmap='RdBu',vmin=-1,vmax=1);axs[1,col].set_title(('原规则＋64段' if method=='old' else '统一屋面＋64段' if method=='fixed' else '统一屋面＋253段')+'：10 m − 20 m 积水面积占比',fontproperties=font)
  for ax in axs[:,col]:
   for g in geoms:
    for p in getattr(g,'geoms',[g]):x,y=p.exterior.xy;ax.plot(x,y,color='black',lw=.5)
   ax.set_aspect('equal');ax.set_xlim(ext[0],ext[1]);ax.set_ylim(ext[2],ext[3]);ax.set_xticks([]);ax.set_yticks([])
   for name,g in zip(['千代田区','中央区','港区'],geoms):
    pt=g.representative_point()
    if ext[0]<pt.x<ext[1] and ext[2]<pt.y<ext[3]:ax.text(pt.x,pt.y,name,fontproperties=font,fontsize=11,ha='center',color='#151515',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=2))
   sx=ext[0]+350;sy=ext[2]+350;ax.plot([sx,sx+2000],[sy,sy],color='black',lw=2);ax.text(sx+1000,sy+100,'2 km',ha='center',fontsize=10)

 fig.colorbar(im,ax=axs[0,:],shrink=.7,label='Wet fraction of common open ground (>0.10 m)');fig.colorbar(jm,ax=axs[1,:],shrink=.7,label='Change in wet fraction (10m minus20m)');fig.savefig(OUT/'grid_sensitivity_before_after.png',dpi=150);plt.close(fig)
 table=''.join(f'<tr><td>{html.escape(q["scope"])}</td><td>{"原规则＋64段" if q["roof_allocation"]=="old" else "统一屋面＋64段" if q["roof_allocation"]=="fixed" else "统一屋面＋253段"}</td><td>{q["area20_km2"]:.3f}</td><td>{q["area10_km2"]:.3f}</td><td>{q["relative_area_difference"]:.1%}</td><td>{q["wet_iou"]:.1%}</td></tr>' for q in rows if q['support']=='all_open_ground' and q['threshold_m']==.1)
 text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>三区积水范围的网格敏感性</title><style>body{{max-width:1000px;margin:32px auto;padding:0 18px;font:17px/1.65 sans-serif;color:#152638}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ccd3da}}.scroll{{overflow-x:auto}}.note{{background:#eef3f7;padding:18px}}a{{color:#075ba7}}</style><h1>三区积水范围：先统一输入，再判断网格影响</h1><p>已完成同一范围、同一地形、同一建筑障碍与糙率、同一 60 mm/h 暴雨和假设低河道水位的 20 m／10 m 对照。分别采用原屋面分配、统一屋面分配以及统一屋面分配＋蓄水曲线加密三组设置，修正空间降雨坐标约定。第三组只加密蓄水曲线，原有道路过流函数完整保留；建筑障碍未抬高或降低。</p><p class="note">积水面积差异从 {old['relative_area_difference']:.1%} 变为 {new['relative_area_difference']:.1%}；湿区重叠度从 {old['wet_iou']:.1%} 变为 {new['wet_iou']:.1%}。{"敏感性有所降低。" if difference>0 else "本轮修正没有降低面积敏感性。"} 最终两网格共同预测的积水核心为 {new['stable_wet_core_km2']:.3f} km²，只被其中一种网格预测为积水的差异带为 {new['grid_disagreement_band_km2']:.3f} km²；并集包络为 {new['envelope_wet_union_km2']:.3f} km²。这些都是本情景的网格敏感性统计，不能作为观测范围或所有误差的上下界。两种网格、一个情景不足以证明正式收敛。</p><img src="grid_sensitivity_before_after.png" alt="全域屋面雨水分配修正前后的网格敏感性地图"><p>上图为 10 m 计算的潜在峰值积水在共同开放地面的面积占比；下图为相对 20 m 的占比变化。统一在 20 m 空间单元展示，属于面积汇总，未把 0.50 m 高程重建解释为 0.50 m 水动力计算。各地峰值可能发生在不同时间。</p><div class="scroll"><table><tr><th>范围</th><th>屋面规则</th><th>20 m/km²</th><th>10 m/km²</th><th>面积差异</th><th>湿区重叠</th></tr>{table}</table></div><h2>本轮修正与数值检查</h2><p>原规则在粗细网格中移置了约 {allocation['old_grid_change_relocated_rain_volume_60mm_m3']:,.0f} m³ 的雨水输入位置。仅统一屋面分配时，20 m／10 m 面积差异为 {roof_only['relative_area_difference']:.1%}，再加密蓄水曲线后为 {new['relative_area_difference']:.1%}。新规则先在固定 0.50 m 几何上分配所有 {allocation['all_roof_area_m2']/1e6:.2f} km² 屋面雨水，再严格守恒地汇总到两种网格。开放地面的直接降雨保留；屋面雨水送到含地面道路的连通开放地面。该转移描述假设的屋面排水，不是雨水穿越建筑的地表路径。</p><p>官方 Galibier 降雨读取器把 x/y 当作角点并加半个单元；直接写入像元中心会出现半格平移。错误接口的两次短试算被保留且拒绝使用，修正后的两次逐单元输入验证通过，最大总量误差小于 0.0001%。完整试算最大水量收支误差为 {findings['max_fixed_budget_relative_error']:.3%}。输入文件 SHA256 在模拟前后核对。</p><p>既有 15 个主要积水候选片区，本次在两种网格 60 mm/h 情景中保留 {findings['old_15_regions_retained_at_60mm_h']} 个；深度排序相关系数 {findings['region_depth_rank_spearman_fixed20_fixed10']:.3f}。这只说明整体位置及排序的稳定性。</p><h2>浅水蓄水曲线的批量修正</h2><p>使用固定官方 HydroMT-SFINCS 的原始函数，把同一细尺度地形的等体积分段从 64 加密至 253。253 = 4 × (64 − 1) + 1，旧过流曲线的所有节点原样保留，线性区间细分不会改变原过流函数。旧蓄水节点复算差异为 0，新表只增加中间的实际地形蓄水信息，未修改官方求解器。</p><p>水位高于各单元最低地面 0.10 m 的批量检查中，95% 单元的等效全单元水深误差上限，20 m 从约 3.6 cm 降至 1.4 cm，10 m 从约 3.7 cm 降至 1.3 cm。该检查的浅水位置因单元最低点而变化，不能解释为每条道路的实测误差，也不能代替模拟网格收敛检查。</p><p><a href="shallow_storage_audit.json">64 段蓄水表审计</a> · <a href="shallow_storage_audit_253.json">253 段蓄水表审计</a></p><h2>仍需解决的部分</h2><p>建筑障碍仍是地面上加 10 m 的简化表达，不是真实建筑高度。障碍漫越面积：最终 20 m 为 {overtop['refined_20']*.25/1e6:.6f} km²，10 m 为 {overtop['refined_10']*.25/1e6:.6f} km²；负体积检查是否完全通过：{findings['strict_nonnegative_refined']}。完整共同开放地面上的结果作为主结论，另附所有试算共用的数值与边缘不确定区剔除结果，不能用变动掩膜制造一致。</p><p>三区整体几何检查同时比较了 20 m、10 m 和 5 m 单元内的开放空间分离程度。5 m 在这里仅为几何检查，尚未执行 5 m 水动力模拟，不能用于宣称积水范围收敛。下一步优先处理道路过流截面与单元内假连通。下一轮建议以三区统一 10 m 网格为基础，按相同的建筑／道路几何规则自动加密复杂区域至 5 m，再与 10 m 全域结果比较。它仍是覆盖三区的整体模型，不需要逐点实地核查。加密网格的官方构建工具兼容性、计算资源和质量守恒需先验证，尚未执行。原生 DEM 不下切、不校正至目标积水面积。</p><p>实际落水管、管网、入渗、泵站及上游流量未纳入；全部屋面雨水即时留在地表属于明确的情景假设。最大假设屋面转移距离约 {allocation['roof_nearest_distance_max_m']:.0f} m，99% 在约 {allocation['roof_nearest_distance_quantiles_m_1m_histogram']['0.99']} m 内。长距离转移不能当作已核实的排水连接。未进行真实暴雨校准或主动措施效果评估。</p><p><a href="whole_domain_and_ward_comparison.csv">全部区域和阈值对照</a> · <a href="water_budgets.csv">水量与障碍检查</a> · <a href="regional_stability.csv">15 个片区稳定性</a> · <a href="subcell_connectivity_audit.json">全域单元内部连通性检查</a> · <a href="allocation_diagnostic.json">屋面分配审计</a> · <a href="findings.json">机器可读结果</a></p><p>接口依据：<a href="https://github.com/Deltares/SFINCS/blob/1f1d8286520a8709b2e513854dff18a3616583db/source/src/sfincs_ncinput.F90">固定版本官方 NetCDF 输入源码</a>和<a href="https://sfincs.readthedocs.io/en/stable/parameters.html">SFINCS 2.4 官方参数文档</a>。</p></html>'''
 (OUT/'review.html').write_text(text)
 spec_path=BASE/'model_spec.json';spec=json.loads(spec_path.read_text());spec['latest_screening_report']=str((OUT/'review.html').relative_to(ROOT));spec['grid_sensitivity_attribution']={'completed':True,'report':str((OUT/'review.html').relative_to(ROOT)),'synthetic_storm_mm_h':60,'river_stage':'unchanged hypothetical low stage','grids_m':[20,10],'roof_allocation_resolution_independent':True,'nlevels_before_after':[64,253],'whole_domain_0_10m_area_gate_passed':findings['exploratory_area_stability_pass'],'three_wards_0_10m_area_gate_passed':findings['three_wards_0_10m_area_gate_passed'],'formal_grid_convergence_established':False,'event_validated':False,'active_measures_simulated':False,'raw_dem_unchanged':True,'prior_runs_preserved':True,'next_step_is_agent_recommendation_not_human_research_decision':'Batch refine shallow face conveyance and build a whole-domain10m mesh with geometry-based5m refinement, after verifying official builder compatibility, resource feasibility and conservative forcing. Hold roof runoff, domain and external scenarios fixed. Other rain/high-stage scenarios still need reruns.'};dump(spec_path,spec)
 dump(OUT/'code_provenance.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/preprocessing/prepare_three_ward_fixed_runoff.py',ROOT/'src/exp/run_three_ward_fixed_runoff.py',Path(__file__),ROOT/'src/exp/review_three_ward_urban_grid.py',ROOT/'src/exp/audit_three_ward_subgrid_storage.py',ROOT/'src/preprocessing/refine_three_ward_storage_tables.py',ROOT/'src/exp/audit_three_ward_subcell_connectivity.py']})
 print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
