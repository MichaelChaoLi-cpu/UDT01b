"""Common-ground and transport-exposure comparison after unchanged run gates pass."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2');os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,csv,math,html,hashlib
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
from numba import njit,prange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_sfincs_cpu_build import sha,dump
from audit_three_ward_negative_volumes import arr,structured
P=ROOT/'data/processed/three_ward_urban_geometry_pair_controls_v1';R=ROOT/'data/exp/estimation/three_ward_urban_geometry_pair_controls_v1';PREP=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1';CASES=['dry_raised','rain60_raised']
SCOPES={'Whole domain':None,'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3,'Adjoining contributing area':0}
def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(h,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
@njit(parallel=True)
def road_depth_bounds(z,cl,valid,peaks,minimum):
 nr,nc=valid.shape;lo=np.full((2,nr,nc),np.nan);up=lo.copy();area=np.zeros((nr,nc))
 for k in prange(nr*nc):
  r=k//nc;c=k%nc
  if not valid[r,c]:continue
  count=0;lower=np.zeros(2);upper=np.zeros(2)
  for i in range(20):
   for j in range(20):
    if cl[r*20+i,c*20+j]!=2 or not np.isfinite(z[r*20+i,c*20+j]):continue
    count+=1
    for a in range(2):
     if np.isfinite(peaks[a,r,c]):lower[a]+=max(0.,peaks[a,r,c]-z[r*20+i,c*20+j]);upper[a]=lower[a]
     else:upper[a]+=max(0.,minimum[a,r,c]+.01-z[r*20+i,c*20+j])
  area[r,c]=count*.25
  if count:
   for a in range(2):lo[a,r,c]=lower[a]/count;up[a,r,c]=upper[a]/count
 return lo,up,area

def compare():
 manifest=json.loads((P/'execution_manifest.json').read_text());lookup={x['case_id']:x for x in manifest['runs']};reuse={x['case_id']:x for x in manifest['reused_U']};support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];ward=support['Ward Code'];road=support['Road Ground Area'];metrics=[];depthstats=[];cover=[];budgets=[];entrances=[];case_arrays={};checks=[]
 def check(name,v):checks.append({'check':name,'pass':bool(v)})
 for case in CASES:
  at=json.loads((R/case/'T/audit.json').read_text());au=json.loads((ROOT/reuse[case]['audit_file']).read_text());assert at['all_checks_passed'] and au['all_checks_passed']
  labels=[np.load(P/'execution'/(case+'_candidate_reference.npz')),np.load(ROOT/reuse[case]['labels_file'])];folders=[ROOT/lookup[case]['folder'],ROOT/reuse[case]['folder']];good=valid.copy()
  for lab in labels:good&=lab['Admitted Mask']&~lab['Quarantine Mask']&np.isfinite(lab['Peak Mean Depth Lower'])&np.isfinite(lab['Peak Mean Depth Upper'])
  lower=np.stack([x['Peak Mean Depth Lower'] for x in labels]);upper=np.stack([x['Peak Mean Depth Upper'] for x in labels]);case_arrays[case]=(lower,upper,good)
  check(case+'_no_midpoint_and_both_admitted',np.all(upper[:,good]>=lower[:,good]) and not np.any(good&(labels[0]['Quarantine Mask']|labels[1]['Quarantine Mask'])))
  peaks=[];minimum=[]
  for a,(folder,audit) in enumerate(zip(folders,[at,au])):
   with Dataset(folder/'sfincs_map.nc') as d:peaks.append(np.flipud(arr(d['zsmax'][0])));minimum.append(np.flipud(arr(d['zb'][:])))
   led=structured(folder/'udt_step_budget.csv');last=led[-1];budgets.append({'case_id':case,'arm':['T','U'][a],'rain_applied_m3':float(last['rain_applied_cumulative']),'signed_final_storage_m3':float(last['signed_storage']),'gross_out_m3':float(last['boundary_out_cumulative']),'gross_in_m3':float(last['boundary_in_cumulative']),'endpoint_budget_residual_m3':float(last['budget_residual']),'maximum_abs_prefix_residual_m3':audit['summary']['max_budget_residual_m3'],'minimum_native_volume_m3':audit['summary']['minimum_volume_m3'],'actual_steps':audit['summary']['actual_steps'],'actual_final_time_s':audit['summary']['actual_final_time_s'],'new_run':a==0,'all_original_checks_passed':True})
  # Road-specific fine-ground integration: do not weight generic ground means
  # and silently call them road water depths.
  rlo=np.full_like(lower,np.nan);rup=rlo.copy();ra=np.zeros_like(area)
  with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif') as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as cl:
   for row in range(0,1012,32):
    count=min(32,1012-row);win=Window(0,row*20,15400,count*20);a,b,c=road_depth_bounds(dem.read(1,window=win),cl.read(1,window=win),valid[row:row+count],np.stack(peaks)[:,row:row+count],np.stack(minimum)[:,row:row+count]);rlo[:,row:row+count]=a;rup[:,row:row+count]=b;ra[row:row+count]=c
  check(case+'_road_support_matches_original',np.array_equal(ra,road));np.savez_compressed(P/(case+'_road_bounds.npz'),lower=rlo,upper=rup,area=ra,paired_mask=good&(ra>0))
  for scope,code in SCOPES.items():
   region=valid if code is None else valid&(ward>0) if code==-1 else valid&(ward==code);sel=region&good;support_area=float(area[sel].sum());cover.append({'case_id':case,'scope':scope,'eligible_ground_km2':float(area[region].sum()/1e6),'paired_admitted_ground_km2':support_area/1e6,'excluded_ground_km2':float(area[region&~good].sum()/1e6),'both_exact_ground_km2':float(area[sel&labels[0]['Exact Label Mask']&labels[1]['Exact Label Mask']].sum()/1e6)})
   for domain,ll,uu,aa in [('Open ground',lower,upper,area),('Classified ground roads',rlo,rup,ra)]:
    use=sel&(aa>0);weights=aa[use];mass=float(weights.sum())
    if not mass:continue
    dl=ll[1,use]-uu[0,use];du=uu[1,use]-ll[0,use];abslo=np.maximum.reduce([dl,-du,np.zeros_like(dl)]);absup=np.maximum(abs(dl),abs(du));depthstats.append({'case_id':case,'scope':scope,'support':domain,'area_km2':mass/1e6,'mean_U_minus_T_lower_m':float(np.sum(weights*dl)/mass),'mean_U_minus_T_upper_m':float(np.sum(weights*du)/mass),'MAE_difference_lower_m':float(np.sum(weights*abslo)/mass),'MAE_difference_upper_m':float(np.sum(weights*absup)/mass),'RMSE_difference_lower_m':float(np.sqrt(np.sum(weights*abslo**2)/mass)),'RMSE_difference_upper_m':float(np.sqrt(np.sum(weights*absup**2)/mass)),'certain_increase_ge0p1m_km2':float(weights[dl>=.1].sum()/1e6),'certain_decrease_ge0p1m_km2':float(weights[du<=-.1].sum()/1e6)})
    for th in [.1,.3,.5]:
     tl=ll[0,use]>th;tu=uu[0,use]>th;ul=ll[1,use]>th;uu_=uu[1,use]>th;interlo=float(weights[tl&ul].sum());interup=float(weights[tu&uu_].sum());unionlo=float(weights[tl|ul].sum());unionup=float(weights[tu|uu_].sum());metrics.append({'case_id':case,'scope':scope,'support':domain,'threshold_m':th,'admitted_area_km2':mass/1e6,'T_certain_wet_km2':float(weights[tl].sum()/1e6),'T_possible_wet_km2':float(weights[tu].sum()/1e6),'U_certain_wet_km2':float(weights[ul].sum()/1e6),'U_possible_wet_km2':float(weights[uu_].sum()/1e6),'certainly_added_by_U_km2':float(weights[ul&~tu].sum()/1e6),'certainly_removed_by_U_km2':float(weights[tl&~uu_].sum()/1e6),'wet_Jaccard_lower':interlo/unionup if unionup else 1.,'wet_Jaccard_upper':min(1.,interup/unionlo) if unionlo else 1.,'agreement_not_accuracy':True})
  # Keep the already-used 20m cell-center-disk entrance exposure definition.
  for ent in rr(ROOT/'data/processed/three_ward_station_surface_risk_v1/entrance_screen_summary.csv'):
   x=float(ent['x_m']);y=float(ent['y_m']);r=int(math.floor((-32320-y)/10));c=int(math.floor((x+11360)/10));r0=max(0,r-3);r1=min(1012,r+4);c0=max(0,c-3);c1=min(770,c+4)
   if r1<=r0 or c1<=c0:ri=ci=np.array([],int)
   else:
    ri,ci=np.mgrid[r0:r1,c0:c1];keep=((-11360+(ci+.5)*10-x)**2+(-32320-(ri+.5)*10-y)**2<=400)&valid[ri,ci];ri,ci=ri[keep],ci[keep]
   total=float(area[ri,ci].sum());ok=good[ri,ci];ri=ri[ok];ci=ci[ok];admitted=float(area[ri,ci].sum());row={'case_id':case,'osm_node_id':ent['osm_node_id'],'station_ids_original':ent['station_ids'],'radius_m':20,'eligible_ground_m2':total,'paired_admitted_m2':admitted,'paired_coverage_fraction':admitted/total if total else None,'neighborhood_quality':total>0 and admitted>=100 and admitted/total>=.8,'not_ingress_or_shutdown':True}
   for k,name in enumerate(['T','U']):
    row[name+'_nearby_lower_peak_max_m']=float(lower[k,ri,ci].max()) if len(ri) else None;row[name+'_certain_0p3m_footprint_m2']=float(area[ri,ci][lower[k,ri,ci]>=.3].sum())
   row['T_hypothetical_exposure_trigger']=row['neighborhood_quality'] and row['T_certain_0p3m_footprint_m2']>=100;row['U_hypothetical_exposure_trigger']=row['neighborhood_quality'] and row['U_certain_0p3m_footprint_m2']>=100;entrances.append(row)
 csvwrite(R/'paired_wet_extent.csv',metrics);csvwrite(R/'paired_depth_differences.csv',depthstats);csvwrite(R/'paired_quality_coverage.csv',cover);csvwrite(R/'paired_water_budgets.csv',budgets);csvwrite(R/'paired_entrance_exposure.csv',entrances)
 # Whole-region matched maps; use same range, no response-selected patches.
 for case,(lo,up,good) in case_arrays.items():
  fig,axes=plt.subplots(1,3,figsize=(14,7.8),sharex=True,sharey=True);extent=[-11.36,-3.66,-42.44,-32.32]
  for k,ax in enumerate(axes[:2]):
   im=ax.imshow(np.where(good,lo[k],np.nan),extent=extent,origin='upper',vmin=0,vmax=.75,cmap='Blues');ax.set_title(['T  Terrain only','U  Urban geometry'][k]);fig.colorbar(im,ax=ax,shrink=.7,label='Peak open-ground mean depth lower bound (m)')
  dl=lo[1]-up[0];du=up[1]-lo[0];confirmed=np.where(dl>0,dl,np.where(du<0,du,0.));im=axes[2].imshow(np.where(good,confirmed,np.nan),extent=extent,origin='upper',vmin=-.3,vmax=.3,cmap='RdBu_r');axes[2].set_title('U minus T: confirmed signed change');fig.colorbar(im,ax=axes[2],shrink=.7,label='Separated-bound difference (m)')
  for ax in axes:ax.set_aspect('equal');ax.set_xlabel('JGD2011 / IX easting (km)')
  axes[0].set_ylabel('Northing (km)');fig.suptitle(case+' | common admitted ground only',fontsize=15);fig.text(.5,.02,'Event peak envelope, not simultaneous depth. White = outside/withheld. Gray-zero difference can include overlapping intervals.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(R/(case+'_comparison.png'),dpi=160);plt.close(fig)
 for rel,h in json.loads((P/'protected_sources.json').read_text()).items():check('unchanged_'+rel,sha(ROOT/rel)==h)
 check('two_new_T_no_new_U_no_added_parents',len(manifest['runs'])==2 and len(manifest['reused_U'])==2 and manifest['additional_parent_events']==0)
 dump(R/'comparison_validation.json',{'all_passed':all(x['pass'] for x in checks),'checks':checks,'training':False,'sealed_tests':False,'people_counted':False});assert all(x['pass'] for x in checks)
 summary={'completed_cases':CASES,'new_T_runs':2,'reused_U_runs':2,'all_runtime_gates_passed':True,'new_parent_events':0,'training':False,'sealed_tests':False,'stranded_people_computed':False,'accuracy_improvement_established':False,'whole_three_ward_metrics':[x for x in metrics if x['scope']=='Three wards'],'whole_three_ward_depth_differences':[x for x in depthstats if x['scope']=='Three wards'],'water_budgets':budgets,'entrance_comparison':[]}
 for case in CASES:
  q=[x for x in entrances if x['case_id']==case and x['neighborhood_quality']];summary['entrance_comparison'].append({'case_id':case,'total_points':675,'quality_passed_paired_points':len(q),'T_exposure_trigger_points':sum(x['T_hypothetical_exposure_trigger'] for x in q),'U_exposure_trigger_points':sum(x['U_hypothetical_exposure_trigger'] for x in q),'trigger_added_by_U':sum(x['U_hypothetical_exposure_trigger'] and not x['T_hypothetical_exposure_trigger'] for x in q),'trigger_removed_by_U':sum(x['T_hypothetical_exposure_trigger'] and not x['U_hypothetical_exposure_trigger'] for x in q)})
 dump(R/'summary.json',summary);render(summary,metrics,depthstats,cover,budgets);print('PAIRED COMPARISON COMPLETE',flush=True)

def render(summary,metrics,depthstats,cover,budgets):
 topline=[]
 for x in metrics:
  if x['scope']=='Three wards':topline.append([x['case_id'],x['support'],x['threshold_m'],f"{x['T_certain_wet_km2']:.3f}–{x['T_possible_wet_km2']:.3f}",f"{x['U_certain_wet_km2']:.3f}–{x['U_possible_wet_km2']:.3f}",f"{x['certainly_added_by_U_km2']:.3f}",f"{x['certainly_removed_by_U_km2']:.3f}"])
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区地形与城市几何 · 配对控制结果</title><style>body{margin:0;background:#f3f5f7;color:#253b48;font:16px/1.8 system-ui}main{max-width:1180px;margin:auto;padding:32px 24px 70px}section,.notice{background:white;border:1px solid #dbe4e9;padding:22px 26px;margin:22px 0;border-radius:10px}.notice{background:#fff7e8;border-left:5px solid #cf8b40}h1{font-size:32px;line-height:1.4}h2{font-size:23px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce4e8}th{background:#eef3f5}img{max-width:100%}a{color:#19698f}.small{font-size:13px;color:#5b7180}</style><main><p class="small">UDT01b · WHOLE THREE WARDS · 2026-10-16</p><h1>同一暴雨与河道边界：<br>城市几何怎样改变积水与交通暴露？</h1><p>完成干控制与降雨控制配对，新增两次全域10 m仅地形运行（T），复用两份经版本、几何和物理降雨函数核验的城市几何结果（U）。保持原训练／验证划分、质量政策与人口不确定性；不训练、不打开封存测试、不计算受困人数。</p><div class="notice"><b>比较的是城市信息组合带来的差异，尚不证明哪组更接近真实积水。</b><p>U包含建筑障碍、道路糙率、格面连通和屋面雨量分配；其中建筑+10 m仍是障碍假设，不是真实逐楼高度。本次采用合成60 mm/h降雨与1.5 m河道水位抬升控制，不是历史事件复现。</p></div><section><h2>1. 配对条件与数值核验</h2><p>两组共用约38.98 km²原始降雨汇水范围、基础地形、10 m格网和同一固定版本求解器。T恢复建筑占地储水，U保留原几何约束；T的原生索引、格面和诊断元数据独立生成。两组实际执行水位文件完全相同。</p><p>旧U控制与新准备包的冗余时间节点数量不同。新的T执行副本采用旧U的原始降雨时间节点与相同水位文件，保留原物理雨型和汇水总量；新准备包及历史输出未改写。准备包水位曲线与复用版本的最大表示差约3.02×10⁻⁸ m，配对运行中通过共用旧版本消除了这项差异。详见复用与输入适配记录。</p>'''
 body+=table(['事件','组','施加雨量 m³','末态储水 m³','累计出流 m³','累计入流 m³','末态残差 m³','最小原生体积 m³'],[[x['case_id'],x['arm'],f"{x['rain_applied_m3']:.2f}",f"{x['signed_final_storage_m3']:.2f}",f"{x['gross_out_m3']:.2f}",f"{x['gross_in_m3']:.2f}",f"{x['endpoint_budget_residual_m3']:.6f}",x['minimum_native_volume_m3']] for x in budgets])
 body+='''<p>全部配对运行通过原完成时长、秒级降雨、实际时钟、每步正性、同一步水量平衡及完整双向边界交换检查。储水预算包含每组全部地面单元，未用曝光掩膜裁剪水量。两组自适应步时钟独立核验，不要求逐步相同。</p></section><section><h2>2. 三区共同地表与道路暴露</h2><p>水深由各组原生在线最高水位重新积分到同一原始开放细地表。道路水深单独在相同道路细地表积分，不以通用开放地面水深直接替代。数值区间保留、不取中点；共同支持取两组准入交集，保留隔离区并报告剔除面积。</p>'''
 body+=table(['事件','支持','阈值 m','T湿区范围 km²','U湿区范围 km²','U明确新增 km²','U明确减少 km²'],topline)
 body+=table(['事件','范围','原开放地表 km²','共同准入 km²','剔除 km²'],[[x['case_id'],x['scope'],f"{x['eligible_ground_km2']:.3f}",f"{x['paired_admitted_ground_km2']:.3f}",f"{x['excluded_ground_km2']:.3f}"] for x in cover if x['scope'] in ['Three wards','Chiyoda','Chuo','Minato']])
 body+='''<p>湿区采用严格“水深&gt;阈值”的原标签规则。站口初筛沿用其原“≥阈值”的规则，二者在表中分开。范围上下界是数值区间支持下的确定／可能超阈值面积，不是统计置信区间。</p></section><section><h2>3. 全域空间对照</h2>'''
 for case in CASES:body+='<h3>'+case+'</h3><img src="'+case+'_comparison.png" alt="whole region paired comparison"><p class="small">每格事件峰值可发生于不同时刻。第三幅只显示上下界分离后能确定的变化，0包含区间重叠；白色为域外或未准入，不是零水深。保持全域视角，未选择响应最极端的小片区。</p>'
 body+='</section><section><h2>4. 对车站暴露判断的影响</h2><p>沿用675个既有站口点及20 m格心圆邻域，在共同准入地面上比较≥0.3 m足迹。只有有效面积≥100 m²且共同覆盖≥80%的邻域进入触发对照，足迹≥100 m²仅为前轮压力情景参数，不是运营关闭阈值。</p>'
 body+=table(['事件','共同质量达标站口点','T触发点','U触发点','U新增触发','U减少触发'],[[x['case_id'],x['quality_passed_paired_points'],x['T_exposure_trigger_points'],x['U_exposure_trigger_points'],x['trigger_added_by_U'],x['trigger_removed_by_U']] for x in summary['entrance_comparison']])
 body+='''<p>这一步仅比较站口接近区域的地表暴露，保留原站口归属未知；没有计算地下进水、实际停运、交通孤岛或受困人数。道路结果是地表面积，不能直接换算关闭道路长度。峰值图不能证明水流方向，内部累计格面路径指标仍未核验。</p></section><section><h2>5. 复现与交付</h2><p>原始地形、旧模拟结果、12＋4事件与正式质量掩膜保持不变。历史雷达／格点降雨仍是待补输入，本次不声称历史事件准确性或三维模型必然优越。后续应把稳定的暴露差异接入同一交通网络，并在具备独立历史观测时检验准确性。</p><ul>'''
 files=[R/'summary.json',R/'paired_wet_extent.csv',R/'paired_depth_differences.csv',R/'paired_quality_coverage.csv',R/'paired_water_budgets.csv',R/'paired_entrance_exposure.csv',R/'urban_reuse_audit.json',R/'dry_raised/T/audit.json',R/'rain60_raised/T/audit.json',R/'comparison_validation.json',P/'execution_manifest.json']
 for f in files:body+='<li><a href="'+os.path.relpath(f,R)+'">'+html.escape(f.name if f.name!='audit.json' else str(f.relative_to(R)))+'</a></li>'
 body+='</ul></section></main></html>';(R/'review.html').write_text(body)
 artifactfiles=[p for p in P.rglob('*') if p.is_file()]+[p for p in R.rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json','delivery_validation.json']]+[ROOT/'src/analyses'/n for n in ['run_urban_geometry_pair_controls_v1.py','audit_urban_geometry_terrain_control_v1.py','compare_urban_geometry_pair_controls_v1.py']];hashes={};items=[]
 for p in artifactfiles:
  st=p.stat();key=(st.st_dev,st.st_ino)
  if key not in hashes:hashes[key]=sha(p)
  items.append({'file':str(p.relative_to(ROOT)),'sha256':hashes[key]})
 dump(R/'delivery_manifest.json',{'artifacts':items,'new_T_runs':2,'reused_U_runs':2,'new_parents':0,'training':False})
 import re
 broken=[x for x in re.findall(r'(?:src|href)="([^"]+)"',body) if not (R/x).resolve().is_file()];dump(R/'delivery_validation.json',{'all_passed':not broken,'broken_links':broken,'runtime_and_comparison_checks_passed':True,'new_T_runs':2,'reused_U_runs':2});assert not broken
if __name__=='__main__':compare()
