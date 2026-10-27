"""Read-only B1 hydraulic geometry comparison; never starts a solver or rail scenario.
Reuses qualified reference masks, fine-road integration and the existing station
neighborhood definition. Numerical bounds are retained; no midpoint labels.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']: os.environ.pop(key,None)
from pathlib import Path
import sys,json,csv,html
import numpy as np
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from prepare_sfincs_cpu_build import sha,dump
from audit_three_ward_negative_volumes import arr,structured
from compare_urban_geometry_pair_controls_v1 import road_depth_bounds,table,csvwrite
from prepare_historical_surface_access_v1 import depth_at
P=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
R=ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
O=R/'B1_geometry_comparison'
D=ROOT/'data/exp/research-design/three_ward_formal_transport_matrix_v1'
SCOPES={'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3}

def main():
 O.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((D/'protocol.json').read_text())
 manifest=json.loads((P/'run_manifest.json').read_text())
 state=json.loads((R/'campaign_state.json').read_text())
 runs={x['case_id']:x for x in manifest['runs']}
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')
 valid=support['Eligible Ground Mask'];area=support['Ground Area'];road=support['Road Ground Area'];ward=support['Ward Code']
 cache=np.load(ROOT/'data/processed/three_ward_surface_alternatives_v1/fine_elevation_support.npz')
 # Only the small station cache is loaded; the large road cache is not materialized.
 gcells=cache['ground_cells'];gz=cache['ground_sorted_z'];gcz=cache['ground_cumulative_z'];gn=cache['ground_count']
 exposure=np.load(ROOT/'data/processed/three_ward_surface_coverage_v2/surface_exposure.npz')
 assert np.array_equal(gcells,exposure['ground_cells'])
 off=exposure['entrance_offsets'];inds=exposure['entrance_cell_index'];entrance_ids=exposure['entrance_ids']
 times=np.asarray(protocol['grid']['synchronized_times_s'])
 checks=[];sources={};metrics=[];stats=[];coverage=[];budgets=[];entrances=[];station_summaries=[];group_rows=[];arrays={}
 def bind(path):
  sources[str(path.relative_to(ROOT))]=sha(path)
 def check(name,ok):
  checks.append({'check':name,'pass':bool(ok)})
  assert ok,name
 for path in [D/'protocol.json',P/'run_manifest.json',R/'campaign_state.json',ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz']:
  bind(path)
 def mask_for(code):return valid&(ward>0) if code==-1 else valid&(ward==code)
 for stage in ['L','H']:
  ids=[f'R1__{stage}__{a}' for a in ['T','U']]
  labs=[];folders=[];audits=[];peaks=[];mins=[];stations=[]
  for ci,case in enumerate(ids):
   run=runs[case];folder=ROOT/run['folder'];old=run.get('original_case_id',case)
   auditfile=ROOT/run['original_audit'] if run['reused'] else R/'execution'/case/'audit.json'
   labelsfile=(H if run['reused'] else P)/'execution'/(old+'_candidate_reference.npz')
   clockfile=(H if run['reused'] else P)/'execution'/(old+'_output_clock.npz')
   audit=json.loads(auditfile.read_text());lab=np.load(labelsfile);clock=np.load(clockfile)['Actual Map Time']
   check(case+'_original_numerical_gates',audit['all_checks_passed'] and all(audit['required_gates'].values()))
   ii=[int(np.argmin(abs(clock-t))) for t in times]
   check(case+'_actual_clock_twelve_states',np.max(abs(clock[ii]-times))<=.001)
   with Dataset(folder/'sfincs_map.nc')as ds:
    peak=np.flipud(arr(ds['zsmax'][0]));bed=np.flipud(arr(ds['zb'][:]))
    eta=np.stack([np.flipud(arr(ds['zs'][i])).ravel()[gcells] for i in ii]+[peak.ravel()[gcells]])
   gl,gu=depth_at(gz,gcz,gn,eta,bed.ravel()[gcells]);stations.append((gl,gu))
   check(case+'_station_peak_independent_reconstruction',np.allclose(gl[-1],lab['Peak Mean Depth Lower'].ravel()[gcells],atol=1e-6,equal_nan=True) and np.allclose(gu[-1],lab['Peak Mean Depth Upper'].ravel()[gcells],atol=1e-6,equal_nan=True))
   if run['reused']:
    source_i=['radar_low','radar_raised'].index(old)
    check(case+'_unchanged_prior_station_states',np.allclose(gl,exposure['ground_lower'][source_i],atol=1e-6,equal_nan=True) and np.allclose(gu,exposure['ground_upper'][source_i],atol=1e-6,equal_nan=True))
   led=structured(folder/'udt_step_budget.csv');last=led[-1]
   budgets.append(dict(case_id=case,stage=stage,arm=run['arm'],rain_applied_m3=float(last['rain_applied_cumulative']),final_storage_m3=float(last['signed_storage']),gross_out_m3=float(last['boundary_out_cumulative']),gross_in_m3=float(last['boundary_in_cumulative']),endpoint_residual_m3=float(last['budget_residual']),max_prefix_residual_m3=audit['summary']['max_budget_residual_m3'],negative_cells=audit['summary']['negative_cells'],runtime_s=audit['summary']['runtime_s']))
   gf=(H if run['reused'] else P)/'execution'/(old+'_group_exchange.npz')
   exchange=np.load(gf);values=exchange['Cumulative Directional Volumes'][-1]
   for j,code in enumerate(exchange['Group Codes']):group_rows.append(dict(case_id=case,water_group=int(code),gross_out_m3=float(values[2*j]),gross_in_m3=float(values[2*j+1])))
   for path in [auditfile,labelsfile,clockfile,gf,folder/'sfincs.bzs',folder/'udt_run_identity.txt']:bind(path)
   labs.append(lab);folders.append(folder);audits.append(audit);peaks.append(peak);mins.append(bed)
  check(stage+'_same_stage_bytes',(folders[0]/'sfincs.bzs').read_bytes()==(folders[1]/'sfincs.bzs').read_bytes())
  br=budgets[-2:]
  check(stage+'_same_source_rain_budget',abs(br[0]['rain_applied_m3']-br[1]['rain_applied_m3'])<1e-6*max(x['rain_applied_m3']for x in br))
  good=valid.copy()
  for lab in labs:good &= lab['Admitted Mask']&~lab['Quarantine Mask']&np.isfinite(lab['Peak Mean Depth Lower'])&np.isfinite(lab['Peak Mean Depth Upper'])
  lower=np.stack([lab['Peak Mean Depth Lower']for lab in labs]);upper=np.stack([lab['Peak Mean Depth Upper']for lab in labs])
  check(stage+'_nonnegative_ordered_bounds',np.all(lower[:,good]>=0) and np.all(upper[:,good]>=lower[:,good]))
  rlo=np.full_like(lower,np.nan);rup=rlo.copy();ra=np.zeros_like(area)
  with rasterio.open(ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif')as dem,rasterio.open(ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif')as cls:
   for row in range(0,1012,32):
    n=min(32,1012-row);win=Window(0,row*20,15400,n*20)
    a,b,c=road_depth_bounds(dem.read(1,window=win),cls.read(1,window=win),valid[row:row+n],np.stack(peaks)[:,row:row+n],np.stack(mins)[:,row:row+n])
    rlo[:,row:row+n]=a;rup[:,row:row+n]=b;ra[row:row+n]=c
  check(stage+'_identical_fine_road_area',np.array_equal(ra,road))
  prior=np.load(H/'paired_exposure_bounds.npz');old_i=0 if stage=='L' else 1
  check(stage+'_unchanged_prior_U_road_peaks',np.allclose(rlo[1],prior['road_lower'][old_i],atol=1e-6,equal_nan=True) and np.allclose(rup[1],prior['road_upper'][old_i],atol=1e-6,equal_nan=True))
  for scope,code in SCOPES.items():
   region=mask_for(code);sel=good&region
   coverage.append(dict(stage=stage,scope=scope,eligible_km2=float(area[region].sum()/1e6),paired_admitted_km2=float(area[sel].sum()/1e6),excluded_km2=float(area[region&~good].sum()/1e6)))
   for name,ll,uu,aa in [('Open ground',lower,upper,area),('Classified ground roads',rlo,rup,ra)]:
    use=sel&(aa>0);w=aa[use];mass=float(w.sum());dl=ll[1,use]-uu[0,use];du=uu[1,use]-ll[0,use]
    al=np.maximum.reduce([dl,-du,np.zeros_like(dl)]);au=np.maximum(abs(dl),abs(du))
    stats.append(dict(stage=stage,scope=scope,support=name,area_km2=mass/1e6,mean_U_minus_T_lower_m=float(np.dot(w,dl)/mass),mean_U_minus_T_upper_m=float(np.dot(w,du)/mass),mean_absolute_difference_lower_m=float(np.dot(w,al)/mass),mean_absolute_difference_upper_m=float(np.dot(w,au)/mass),certain_increase_ge0p1m_km2=float(w[dl>=.1].sum()/1e6),certain_decrease_ge0p1m_km2=float(w[du<=-.1].sum()/1e6)))
    for th in [.1,.3,.5]:
     tl=ll[0,use]>th;tu=uu[0,use]>th;ul=ll[1,use]>th;up=uu[1,use]>th
     inter=float(w[tl&ul].sum());union=float(w[tu|up].sum())
     t=float(w[tl].sum()/1e6);u=float(w[ul].sum()/1e6);add=float(w[ul&~tu].sum()/1e6);remove=float(w[tl&~up].sum()/1e6)
     check(f'{stage}_{scope}_{name}_{th}_wet_accounting',abs((u-t)-(add-remove))<=float(w[(tu!=tl)|(up!=ul)].sum()/1e6)+1e-10)
     metrics.append(dict(stage=stage,scope=scope,support=name,threshold_m=th,T_certain_km2=t,T_possible_km2=float(w[tu].sum()/1e6),U_certain_km2=u,U_possible_km2=float(w[up].sum()/1e6),certain_added_by_U_km2=add,certain_removed_by_U_km2=remove,wet_jaccard_lower=inter/union if union else 1.,not_accuracy=True))
  # Same 675 points, same20m cell-center disk, same100m2 and80% support rule.
  gknown=good.ravel()[gcells];ga=gn*.25
  for frame,frame_id in enumerate([str(t)for t in times]+['peak_envelope']):
   records=[]
   for j,osm in enumerate(entrance_ids):
    cells=inds[off[j]:off[j+1]];total=float(ga[cells].sum());cells=cells[gknown[cells]];admitted=float(ga[cells].sum());quality=total>0 and admitted>=100 and admitted/total>=.8
    rec=dict(stage=stage,frame_id=frame_id,simultaneous=frame_id!='peak_envelope',osm_node_id=str(osm),paired_admitted_m2=admitted,eligible_m2=total,quality=quality)
    for arm,(gl,gu) in zip(['T','U'],stations):
     for bound,vals in [('lower',gl),('upper',gu)]:
      footprint=float(ga[cells][vals[frame,cells]>=.3].sum());rec[f'{arm}_{bound}_trigger']=bool(quality and footprint>=100)
    entrances.append(rec);records.append(rec)
   q=[x for x in records if x['quality']]
   station_summaries.append(dict(stage=stage,frame_id=frame_id,simultaneous=frame_id!='peak_envelope',total_entrance_points=len(records),paired_quality_points=len(q),**{f'{arm}_{bound}_trigger_points':sum(x[f'{arm}_{bound}_trigger']for x in q)for arm in ['T','U']for bound in ['lower','upper']},certain_added_by_U=sum(x['U_lower_trigger']and not x['T_upper_trigger']for x in q),certain_removed_by_U=sum(x['T_lower_trigger']and not x['U_upper_trigger']for x in q)))
  np.savez_compressed(O/f'{stage}_paired_peak_bounds.npz',ground_lower=lower,ground_upper=upper,road_lower=rlo,road_upper=rup,paired_admitted=good,ground_area=area,road_area=ra)
  arrays[stage]=(lower,upper,good)
  print('COMPARED',stage,flush=True)
 # Snapshot existing later-batch audits; no later-batch result interpretation here.
 audit_rows=[]
 for path in sorted((R/'execution').glob('*/audit.json')):
  a=json.loads(path.read_text());audit_rows.append(dict(case_id=a['case_id'],all_checks_passed=a['all_checks_passed'],checks=len(a['checks']),negative_cells=a['summary']['negative_cells'],max_prefix_residual_m3=a['summary']['max_budget_residual_m3']))
 csvwrite(O/'campaign_numerical_status.csv',audit_rows)
 for name,rows in [('paired_wet_extent',metrics),('paired_depth_difference',stats),('paired_quality_coverage',coverage),('paired_water_budget',budgets),('paired_group_exchange',group_rows),('entrance_exposure_detail',entrances),('entrance_exposure_summary',station_summaries)]:csvwrite(O/(name+'.csv'),rows)
 for stage,(lo,up,good)in arrays.items():
  region=good&(ward>0);dl=lo[1]-up[0];du=up[1]-lo[0];signed=np.where(dl>0,dl,np.where(du<0,du,0.))
  fig,axes=plt.subplots(1,3,figsize=(13,7.6));extent=[-11.36,-3.66,-42.44,-32.32]
  for i,ax in enumerate(axes[:2]):
   im=ax.imshow(np.where(region,lo[i],np.nan),extent=extent,origin='upper',vmin=0,vmax=.75,cmap='Blues');ax.set_title(['T: Terrain only','U: Urban geometry'][i]);fig.colorbar(im,ax=ax,shrink=.65,label='Peak ground mean depth lower bound (m)')
  im=axes[2].imshow(np.where(region,signed,np.nan),extent=extent,origin='upper',vmin=-.3,vmax=.3,cmap='RdBu_r');axes[2].set_title('U minus T');fig.colorbar(im,ax=axes[2],shrink=.65,label='Depth difference (m)')
  for ax in axes:ax.set_aspect('equal');ax.set_xlabel('JGD2011 / IX easting (km)')
  axes[0].set_ylabel('Northing (km)');fig.suptitle('Historical radar rainfall | '+('Low-stage assumption'if stage=='L'else'Raised-stage assumption'))
  fig.text(.5,.025,'Non-simultaneous peak envelope. White = outside common admitted three-ward support. Zero change includes overlapping bounds.',ha='center',fontsize=8)
  fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(O/f'{stage}_geometry_comparison.png',dpi=155);plt.close(fig)
 check('sources_unchanged_during_comparison',all(sha(ROOT/k)==v for k,v in sources.items()))
 summary=dict(campaign_status_at_review=state['status'],completed_new_cases=len(state['completed_new_cases']),campaign_active_hours=state['active_wall_seconds']/3600,first_batch_original_gates_passed=True,accuracy_improvement_established=False,transport_accessibility_computed=False,new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False,whole_three_ward_metrics=[x for x in metrics if x['scope']=='Three wards'],whole_three_ward_depth_differences=[x for x in stats if x['scope']=='Three wards'],peak_station_exposure=[x for x in station_summaries if x['frame_id']=='peak_envelope'],recommendation='Proceed with B2 no-rain interpretation and then B3 intensity interpretation using already completed outputs; no additional hydraulic runs. Preserve unknown transport branches. Do not conclude improved real-world accuracy or transport isolation from exposure alone.')
 dump(O/'summary.json',summary);dump(O/'validation.json',dict(all_passed=all(x['pass']for x in checks),checks=checks));dump(O/'source_bindings.json',sources)
 dump(R/'review_checkpoint.json',dict(human_decision='MILI-D-20261017-005',requested_review='First B1 geometry comparison before scientific progression',actual_controller_state=state['status'],later_batches_already_completed_before_review=True,no_new_hydraulic_run_started=True,next_scientific_step=summary['recommendation'],first_batch_comparison=str((O/'review.html').relative_to(ROOT))))
 render(summary,metrics,stats,coverage,budgets,station_summaries,audit_rows,len(checks))
 # Link from the user's existing live matrix page, without changing its frozen protocol or code.
 page=D/'review.html';text=page.read_text();marker='<!-- B1_REVIEW_LINK -->'
 if marker not in text:text=text.replace('<h2>首批与完整水动力矩阵</h2>',marker+'<p><a href="'+os.path.relpath(O/'review.html',D)+'">首批地形／城市几何配对结果与后续推进判断</a></p><h2>首批与完整水动力矩阵</h2>');page.write_text(text)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def render(s,metrics,stats,coverage,budgets,station_summaries,audit_rows,nchecks):
 stage_name={'L':'低水位假设','H':'水位抬升假设'}
 rows=[]
 for x in metrics:
  if x['scope']=='Three wards' and ((x['support']=='Open ground'and x['threshold_m']==.1)or(x['support']=='Classified ground roads'and x['threshold_m']==.3)):
   rows.append([stage_name[x['stage']],'开放地表 >0.1m'if x['support']=='Open ground'else'分类道路地表 >0.3m',f"{x['T_certain_km2']:.3f}–{x['T_possible_km2']:.3f}",f"{x['U_certain_km2']:.3f}–{x['U_possible_km2']:.3f}",f"{x['certain_added_by_U_km2']:.3f}",f"{x['certain_removed_by_U_km2']:.3f}",f"{x['wet_jaccard_lower']:.3f}"])
 body='<h1>首批结果：城市几何改变了什么？</h1><p>三区整体视角；原历史雷达雨场、相同10 m网格、相同固定求解器、相同配对水位。新增两份T结果与已通过的两份U结果比较。</p>'
 body+='<aside><b>实际状态：原控制器已完成全部10份新增水动力计算，约8小时54分，均通过原数值审计。</b><p>此前规则是每批数值通过即自动推进，因此B2与B3已计算；本页只解释首批B1。当前没有追加水动力计算。数值通过与研究解释复核分开记录，后续先评阅已有结果。</p></aside>'
 body+='<h2>三区共同地表的积水差异</h2>'+table(['水位','指标','T范围 km²','U范围 km²','U明确新增 km²','U明确减少 km²','湿区重叠下界'],rows)
 body+='<p>使用每组事件在线峰值在同一原始细地表上积分，不用原生h直接充当地面平均水深。共同准入交集排除任一组隔离区域。表中的范围是数值上下界，不是统计置信区间；“新增”和“减少”是不同位置的空间变化，不能仅看净面积。Jaccard仅衡量两组一致程度，不是真实精度。</p>'
 body+=table(['水位','区域','可用地表 km²','共同准入 km²','排除 km²'],[[stage_name[x['stage']],x['scope'],f"{x['eligible_km2']:.5f}",f"{x['paired_admitted_km2']:.5f}",f"{x['excluded_km2']:.5f}"]for x in coverage])
 body+='<h2>空间变化与局部水深</h2>'
 body+=table(['水位','支持','面积加权U−T下界 m','上界 m','平均绝对差下界 m','上界 m','明确增加≥0.1m km²','明确减少≥0.1m km²'],[[stage_name[x['stage']],x['support'],f"{x['mean_U_minus_T_lower_m']:.4f}",f"{x['mean_U_minus_T_upper_m']:.4f}",f"{x['mean_absolute_difference_lower_m']:.4f}",f"{x['mean_absolute_difference_upper_m']:.4f}",f"{x['certain_increase_ge0p1m_km2']:.3f}",f"{x['certain_decrease_ge0p1m_km2']:.3f}"]for x in stats if x['scope']=='Three wards'])
 for stage in ['L','H']:body+='<h3>'+stage_name[stage]+'</h3><img src="'+stage+'_geometry_comparison.png" alt="Whole three-ward paired peak depth maps">'
 body+='<p>图为非同时峰值包络，不能据此判断同一时刻交通断连、水流方向或断连持续时间。差值0也包括区间重叠，白色为共同准入外。</p><h2>站口周边暴露</h2>'
 body+=table(['水位','时刻','共同质量达标点','T触发下界—上界','U触发下界—上界','U明确新增','U明确减少'],[[stage_name[x['stage']],('峰值包络'if x['frame_id']=='peak_envelope'else'19:00 JST'),x['paired_quality_points'],str(x['T_lower_trigger_points'])+'—'+str(x['T_upper_trigger_points']),str(x['U_lower_trigger_points'])+'—'+str(x['U_upper_trigger_points']),x['certain_added_by_U'],x['certain_removed_by_U']]for x in station_summaries if x['frame_id']in['14400','peak_envelope']])
 body+='<p>保留675个站口点、20 m格心圆邻域、共同有效面积≥100 m²且覆盖≥80%的原规则；≥0.3 m足迹≥100 m²仅是既有假设关闭情景的触发参数。完整CSV含12个统一时刻，峰值另列。暴露点数不等于车站数量、地下进水、真实停运或交通孤岛。</p>'
 body+='<h2>数值与全域水量</h2>'+table(['算例','降雨 m³','末态储水 m³','累计出流 m³','累计入流 m³','最大过程残差 m³','负体积单元'],[[x['case_id'],f"{x['rain_applied_m3']:.2f}",f"{x['final_storage_m3']:.2f}",f"{x['gross_out_m3']:.2f}",f"{x['gross_in_m3']:.2f}",f"{x['max_prefix_residual_m3']:.6f}",x['negative_cells']]for x in budgets])
 body+='<p>预算使用每组完整计算域，未裁成共同曝光掩膜；T恢复建筑占地储水，不能把组间总储水差直接叫作三区积水增量。全部新算例共'+str(sum(x['checks']for x in audit_rows))+'项已存数值检查通过；本次对照'+str(nchecks)+'项检查通过。未重跑求解器、未修改原准入门槛。</p>'
 body+='<h2>推进判断</h2><p><b>可继续分析下一批已有结果；不需要再启动水动力计算。</b>先用B2无雨控制分离假设水位带来的影响，再用B3检查1.25倍雨量下几何差异是否保持。交通后处理需继续使用同一张分层图，分别保留严格道路层、步道补缺层与桥隧高程未知分支，之后才能判断这些几何变化是否改变交通孤岛。</p><p>本轮比较表明城市信息会改变模型输出，尚不能称为准确性改善。U同时改变建筑阻水、道路输水和保守屋面分配；T与U仍是二维求解器的不同城市表达。需要独立积水观测才能判断哪组更接近真实。河道水位是假设，无管网、泵站、入渗与实测完整河床。人口和楼宇分配不确定性保留，不计算受困人数，不训练，不接触封存测试。</p>'
 body+='<h2>复现文件</h2><ul>'
 for f in sorted(O.iterdir()):
  if f.suffix in ['.csv','.json','.npz']:body+='<li><a href="'+f.name+'">'+html.escape(f.name)+'</a></li>'
 body+='</ul><p>Framelog: MILI-D-20261017-005。原计算合同和源码未改写，新增科学评阅检查点单独记录。</p>'
 (O/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>首批三区城市几何对照结果</title><style>body{font:16px/1.8 system-ui;max-width:1220px;margin:auto;padding:32px;color:#253d4b;background:#f4f7f9}h1,h2{line-height:1.4}h2{margin-top:34px}aside{padding:20px;background:#fff2da;border-left:5px solid #d59137}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #d6e0e6;padding:9px;text-align:left}th{background:#e7eef3}.scroll{overflow:auto}img{max-width:100%}a{color:#176992}</style>'+body+'</html>')

if __name__=='__main__':main()
