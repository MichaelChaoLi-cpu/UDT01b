"""Whole-area negative-volume review and skill-routed quality evidence notes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,csv,re,html,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from audit_three_ward_negative_volumes import OUT,REPORT,CASES,readnp,csvwrite
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump

def rows(p):return list(csv.DictReader(p.open()))
def main(update_sop=False):
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen audit')
 s=json.loads((report/'summary.json').read_text());coverage=rows(report/'scope_coverage.csv');wet=rows(report/'wet_area_exposure.csv');cohorts=rows(report/'static_cohort_comparison.csv');patch=rows(report/'principal_peak_regions.csv')
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');ward=support['Ward Code'];valid=support['Eligible Ground Mask']
 def cv(c,scope):return next(r for r in coverage if r['case']==c and r['scope']==scope)
 def wr(c,scope,th=.1):return next(r for r in wet if r['case']==c and r['scope']==scope and float(r['threshold_m'])==th)
 def co(c,n):return next(r for r in cohorts if r['case']==c and r['cohort']==n and r['population']=='Online negative')
 # Exploratory figure only; no publication status or new safety threshold.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(2,2,figsize=(12,12),layout='constrained',height_ratios=[1.8,1])
 extent=[-11.36,-3.66,-42.44,-32.32]
 masks={}
 for i,case in enumerate(CASES):
  d=readnp(out/(case+'_audit_masks.npz'));masks[case]=d;ax=axs[0,i]
  ax.imshow(np.where(valid,1,np.nan),extent=extent,origin='upper',cmap=ListedColormap(['#e9edef']),vmin=0,vmax=1,interpolation='none')
  wetmask=valid&(d['Peak Mean Depth Lower']>.1)
  ax.imshow(np.where(wetmask,1,np.nan),extent=extent,origin='upper',cmap=ListedColormap(['#79aab9']),vmin=0,vmax=1,interpolation='none')
  # Contours show ward allocation on the analysis mesh, not surveyed boundaries.
  for code,label in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   field=(ward==code).astype(float);ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,field,levels=[.5],colors=['#62757c'],linewidths=.5)
   rr,cc=np.where((ward==code)&valid);ax.text(-11.36+(np.median(cc)+.5)*.01,-32.32-(np.median(rr)+.5)*.01,label,fontsize=10,color='#173541',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':2})
  for flag,color,marker,size in [(d['Online Ever Negative']&~d['Online Only Negative'],'#bd453a','.',10),(d['Online Only Negative'],'#e19c27','x',9)]:
   rr,cc=np.where(flag);ax.scatter(-11.36+(cc+.5)*.01,-32.32-(rr+.5)*.01,c=color,s=size,marker=marker,lw=.65,zorder=5)
  ax.set(title=('Dry + raised stage'if i==0 else'Rain + raised stage')+f" | {s['cases'][case]['online_ever_cells']} flagged cells",xlabel='EPSG:6677 easting (km)',ylabel='EPSG:6677 northing (km)');ax.set_aspect('equal')
  ax.legend(handles=[Line2D([0],[0],color='#79aab9',lw=5,label='Peak mean depth > 0.10 m'),Line2D([0],[0],color='#bd453a',marker='.',ls='',label='Negative seen in saved frames'),Line2D([0],[0],color='#e19c27',marker='x',ls='',label='Online flag only')],fontsize=8,loc='lower left')
 timeline=rows(report/'step_negative_timeline.csv')
 for case,color in zip(CASES,['#347e93','#bc6848']):
  records=[r for r in timeline if r['case']==case];t=np.array([float(r['actual_time_s'])for r in records])/3600
  axs[1,0].plot(t,[float(r['negative_cells'])for r in records],color=color,label=case,lw=.9)
  axs[1,1].plot(t,[float(r['negative_debt_m3'])for r in records],color=color,label=case,lw=.9)
 for ax,title,y in [(axs[1,0],'Actual native-step negative counts','Simultaneous flagged cells'),(axs[1,1],'Negative storage retained in the ledger','Total negative storage deficit (m3)')]:
  ax.axvline(1,color='#85949c',ls='--',lw=.8);ax.set(title=title,xlabel='Actual event time (h)',ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.15)
 fig.savefig(report/'negative_distribution_and_timing.png',dpi=160);plt.close(fig)
 # Short English analytical notes keep all prior rule and variable status intact.
 if update_sop:
  p=ROOT/'docs/AnaSOP.md';backup=report/'previous-AnaSOP.md'
  if not backup.exists():backup.write_bytes(p.read_bytes())
  before=p.read_text();parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M)
  additions={4:'''The two subsequent regular-grid open-control replays now have a read-only spatial negative-volume audit. Every-step flags identify 64 dry-control and 977 rain-control native cells, compared with 23 and 830 cells visible in 300-second state snapshots. Reconstruct each replay's Peak Mean Depth bounds from its own online maximum levels over the same original eligible fine ground, and screen those bounds with its own online ever-negative mask and current assumed-barrier contact. Keep these candidate diagnostic masks separate from historical labels and the closed reference registry. No cell-specific first-crossing time or episode duration is inferred from an event-wide flag or sparse states. No final variable, spatial buffer, numerical threshold or training authorization is added.''',6:'''The new whole-area audit finds strong concentration next to prescribed-stage boundary cells. Source inspection supports a possible wetting-and-drying overdraw mechanism: continuity preserves signed storage, maps sufficiently depleted storage to the local minimum water level, and momentum restricts outward flux only after the prior storage is already negative. Sparse states and global step summaries cannot establish cell-specific crossing fluxes or bound propagated errors. Treat static open-face connected peak-exceedance regions as descriptive spatial support, not simultaneous water surfaces or realized drainage paths. Small direct overlap with the rainy principal peak regions supports continued whole-area diagnostic use, while the dry control retains a larger relative gap in the smaller Minato wet support. Global budget closure does not establish positivity or unaffected neighbors.''',7:'''The negative-volume distribution, global occurrence clock, sparse local observations, original-ground peak reconstruction and direct label-support losses have now been audited for the two completed open controls. Continue to withhold directly flagged cells under the existing threshold, retain all signed volumes for balance, and report per-ward wet-support coverage. The two replays are candidate partial numerical-reference diagnostics; no automatic full-domain reference admission, historical relabeling, training or prospective-library execution follows. Before another separately authorized reference run, add compact per-cell crossing/minimum/negative-step summaries and crossing-step source and face contributions, prioritizing aggregate boundary groups rather than individual sites. Do not infer a spatial error bound or implement clipping from this audit.'''}
  for i,part in enumerate(parts):
   if not part.startswith('## '):continue
   n=int(re.match(r'## (\d)',part).group(1))
   if n in additions and additions[n]not in part:parts[i]=part.rstrip()+'\n\n'+additions[n]+'\n\n'
  p.write_text(''.join(parts));old=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M)
  changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(old,parts)if a!=b}
  assert changed=={4,6,7}
  dump(report/'anasop-update.json',{'changed_sections':[4,6,7],'skills':['data-preprocessing','estimation-framework-planning'],'human_record':'MILI-D-20261015-010','prior_sha256':sha(backup),'current_sha256':sha(p),'final_variables_changed':False,'formulas_changed':False,'other_sections_unchanged':True})
 dry=s['cases'][CASES[0]];rain=s['cases'][CASES[1]]
 retained={c:min(float(r['largest_retained_component_ground_pct_of_original'])for r in patch if r['case']==c and float(r['threshold_m'])==.1)for c in CASES}
 fragmentation_note=f"按原质量掩膜剔除后，前10个>0.10 m峰值连片中，最大剩余连通部分至少保留原地面面积的：干组{retained['dry_raised']:.3f}%，雨组{retained['rain60_raised']:.3f}%。这仅说明评价支持未明显碎裂，不是重新计算后的地表排水连通性或水深稳定性。"
 table=[]
 for c in CASES:
  x=s['cases'][c];w=wr(c,'Three wards');top=[r for r in patch if r['case']==c and float(r['threshold_m'])==.1]
  table.append({'case':c,'online_negative_cells':x['online_ever_cells'],'snapshot_negative_cells':x['sampled_ever_cells'],'first_global_negative_s':x['first_global_negative_time_s'],'negative_three_ward_ground_pct':float(cv(c,'Three wards')['online_negative_ground_pct']),'negative_three_ward_wet10cm_pct':float(w['negative_certain_wet_pct']),'max_top10_peak_region_direct_negative_pct':max(float(r['negative_ground_pct'])for r in top),'adjacent_stage_boundary_cells_pct':float(co(c,'Direct boundary neighbor')['matching_cells_pct']),'open_receiver_face_connected_cells_pct':float(co(c,'Open receiver face connected')['matching_cells_pct'])})
 csvwrite(report/'review_summary.csv',table)
 text=f'''# 三区负体积：分布、时刻、机制与标签支持\n\n本次只读分析两个已完成控制组；没有运行求解器、训练或批量运行58个情景。所有分析沿用 -0.001 m³ 标记阈值、原标签区间政策及原始地面支持。Framelog MILI-D-20261015-010。\n\n## 可以得出的判断\n\n负体积单元在全域占比小，集中在规定水位格的水陆交界；降雨情景主要峰值积水斑块的直接覆盖损失较小，可以继续用于带标记的全域探索。不能据此把全部网格作为无条件参考标签，也不能证明邻域水深与地表流路不受影响。\n\n|控制组|在线曾负体积|300 s抽样可见|全域首次出现，事件秒|三区开敞地面占比|三区>0.10 m积水地面直接重叠|\n|---|---:|---:|---:|---:|---:|\n'''
 for r in table:text+=f"|{r['case']}|{r['online_negative_cells']}|{r['snapshot_negative_cells']}|{r['first_global_negative_s']:.3f}|{r['negative_three_ward_ground_pct']:.4f}%|{r['negative_three_ward_wet10cm_pct']:.4f}%|\n"
 text+='''\n分母是共同10 m网格的有效开敞地面面积，非三区行政总面积；积水指事件峰值平均水深严格大于阈值的单元支持，非同时积水或细地面真实边界。\n\n## 空间分布与成因证据\n\n'''
 for r in table:text+=f"- {r['case']}：{r['adjacent_stage_boundary_cells_pct']:.2f}% 的标记单元相邻规定水位格；{r['open_receiver_face_connected_cells_pct']:.2f}% 有已记录的开放接收格面。前10个>0.10 m峰值连片中，单个连片直接受负体积隔离的比例最高 {r['max_top10_peak_region_direct_negative_pct']:.3f}%。\n"
 text+=f'''\n干组不含降雨仍出现负体积，因此问题不以雨量读取为必要条件。两组静态几何、参数、水位文件逐字节相同，仅降雨不同；有 {s['comparison']['shared_cells']} 个共同标记单元、{s['comparison']['rain_only_cells']} 个仅在降雨重放中被标记。规定水位空间邻近与开放连接是关联证据，不能直接归因于真实河流、某个岸顶或DEM误差。建筑混合格与封闭格面邻近在标记单元中没有相对未标记总体富集；该比较未控制湿润程度和边界距离，不构成因果检验。\n\n固定源码的直接机制：连续方程把每步格面净交换和非负降雨加到有符号储量，不在该处裁零；储量极低／为负时，水位退回子网格最低高程；动量方程只在旧储量已负时限制继续出流，并非对本步可用体积的严格正性保证。空间和时间证据与水陆湿干转换的单步过抽相符。缺少触发步骤的逐格面贡献，尚不能逐单元证实触发面或做修正后的反事实判断。源码行号与哈希见 source_mechanism_evidence.json。\n\n## 时刻：可确认与不可恢复\n\n- 干组全域第一次跨过阈值：{dry['first_global_negative_time_s']:.6f} s，负体积同时单元最多 {dry['max_simultaneous_negative_cells']} 个；结束时仍有 {dry['final_negative_cells']} 个。\n- 降雨组：{rain['first_global_negative_time_s']:.6f} s，即起雨约1.6秒后；同时单元最多 {rain['max_simultaneous_negative_cells']} 个；结束时仍有 {rain['final_negative_cells']} 个。\n- 最大负储量总额分别约 {dry['max_debt_m3']:.3f}、{rain['max_debt_m3']:.3f} m³，单格在线最小值约 {dry['minimum_online_volume_m3']:.3f}、{rain['minimum_online_volume_m3']:.3f} m³。不能一概解释为浮点舍入或已消失的瞬时现象，也不能直接解释为负的真实水深。\n- 全域每步数量／总量／历史最小值有实际时间。每个单元仅有事件级曾负标记及300秒状态；首次观测负值不等于首次跨阈值，帧数不等于持续时长。干组 {dry['online_only_cells']} 个、雨组 {rain['online_only_cells']} 个标记单元在所有输出帧均未被观察为负；其局地发生时刻明确保留未知。\n\n## 对主要积水区和参考标签的影响\n\n降雨组三个区>0.10 m峰值积水地面约 {float(wr('rain60_raised','Three wards')['certain_wet_km2']):.6f} km²；负体积单元直接重叠 {float(wr('rain60_raised','Three wards')['negative_certain_wet_km2']):.6f} km²（{float(wr('rain60_raised','Three wards')['negative_certain_wet_pct']):.4f}%）。连同已有假设建筑障碍接触，共隔离该积水支持的 {float(wr('rain60_raised','Three wards')['all_quarantine_certain_wet_pct']):.4f}%。因此主要雨淹片区在直接掩膜覆盖层面保留较完整；这是覆盖判断，不是无误差或修正后范围稳定的证据。\n\n干组全三区的对应比例约 {float(wr('dry_raised','Three wards')['negative_certain_wet_pct']):.3f}%，但港区的小面积>0.10 m支持中达到 {float(wr('dry_raised','Minato')['negative_certain_wet_pct']):.2f}%。不能用全域干地面比例掩盖这一局地覆盖不足。所有阈值0.05/0.10/0.30/0.50 m与各区结果均保留，不选便利阈值。\n\n当前候选标签由本次重放的在线峰值，在原0.50 m裸地像元上重新积分，排除建筑、保留浅水上下界；没有把原生h直接当平均水深。在线标记替换同次运行的抽样标记后，新增加隔离地面：干组 {dry['extra_quarantine_over_same_run_snapshots_km2']:.6f} km²、降雨组 {rain['extra_quarantine_over_same_run_snapshots_km2']:.6f} km²。当前全部隔离地面分别 {dry['all_current_quarantine_km2']:.6f}、{rain['all_current_quarantine_km2']:.6f} km²，含当前假设障碍接触。\n\n原政策允许通过全事件硬门槛后形成“部分数值参考”，不要求因少量负体积把全域一律丢弃。本次只建立新的诊断候选支持；旧注册表、默认加载器及旧标签不改，也没有完成正式准入登记。完整准入需把新峰值、在线掩膜、浅水证明、字段／边界绑定及已通过的实际时钟和预算检查组成同一版本，逐项复核后才可登记；训练仍须另行授权。质量标记只用于监督和评价，不能成为预测输入。\n\n## 下一步最小工作\n\n先补充紧凑诊断接口：逐单元首次／末次跨阈值、最小体积、负值步数和触发步前后储量；仅在跨阈值时记录四向格面净交换、实际入雨、开放宽度、面高程及边界组。按7个水域组聚合即可，不展开逐点现地核查，也不导出全时空格面立方体。实施后若需证明修正有效，应另行授权相同两组的有限对照；当前没有补跑。没有依据本次结果裁零、扩大隔离缓冲、修改政策或启动58个库情景。\n\n## 评价支持的连通性\n\n{fragmentation_note}\n\n## 文件\n\n- negative_distribution_and_timing.png：全域分布与真实步时钟。\n- scope_coverage.csv、wet_area_exposure.csv：三区及全域的支持损失。\n- principal_peak_regions.csv：开放格面连接的前10峰值超阈连片，非同时水面／实际流路。\n- static_cohort_comparison.csv、receiver_group_exposure.csv：边界、道路、建筑等总体对比；水域组采用代码，不合并同名古川。\n- negative_cell_evidence.csv、sampled_first_detection.csv：逐格证据和不确定时刻；不作为逐点核查任务清单。\n- step_negative_timeline.csv、phase_summary.csv：每步和分阶段全域负储量。\n- 新NPZ／GeoTIFF在 {OUT}，仅供诊断。\n'''
 (report/'README.md').write_text(text)
 def table_html(records,columns):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>'for key,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[key]))+'</td>'for key,label in columns)+'</tr>'for r in records)+'</tbody></table></div>'
 small=[{'case':'干组'if r['case']==CASES[0]else'降雨组','cells':r['online_negative_cells'],'time':f"{r['first_global_negative_s']:.3f}",'ground':f"{r['negative_three_ward_ground_pct']:.4f}%",'wet':f"{r['negative_three_ward_wet10cm_pct']:.4f}%"}for r in table]
 wardrows=[{'ward':n,'dry':cv('dry_raised',n)['online_negative_cells'],'rain':cv('rain60_raised',n)['online_negative_cells'],'rainwet':f"{float(wr('rain60_raised',n)['negative_certain_wet_pct']):.4f}%"}for n in ['Chiyoda','Chuo','Minato','Contributing area']]
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区负体积空间与标签影响核查</title><style>body{font:16px/1.8 system-ui,sans-serif;background:#eef3f4;color:#203945;margin:0}main{max-width:1100px;margin:auto;padding:28px}section{background:white;border:1px solid #dce5e8;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:29px;line-height:1.45}h2{font-size:21px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dce5e8}.scroll{overflow:auto}a{color:#146d83}.note{border-left:4px solid #c58d40;padding-left:16px}.muted{color:#607780}@media(max-width:650px){main{padding:12px}section{padding:17px}}</style><main><p>Mili · 两个既有控制组 · 全域10 m</p><h1>负体积主要集中在水陆边界<br>直接积水覆盖损失小，参考标签继续受限</h1><p>只读分析现有输出，重新积分本次峰值标签；没有训练、补跑或执行58个情景。</p><section><h2>三区整体判断</h2>__TABLE__<p class="muted">地面占比以有效开敞地面为分母；积水重叠以峰值平均水深>0.10 m的有效地面为分母。该阈值是分析阈值，不是安全标准。</p><p>干组约83%、雨组约89%的标记单元紧邻规定水位格。雨组前10个主要峰值连片，直接负体积隔离比例均低于0.18%；干组港区的小面积积水支持受影响约7.27%，不能被全域平均掩盖。</p></section><section><h2>分布与发生时刻</h2><img src="negative_distribution_and_timing.png" alt="两个控制组的全域负体积分布和每步负储量时序">__WARDS__<p>全域首次出现时刻来自实际每步记录；单格只有事件级标记与300秒状态，局地首次跨阈值和持续时间无法恢复。干组41个、雨组147个标记单元在所有输出帧均未被观察为负，不能给它们伪造发生时刻。</p></section><section><h2>成因：源码支持的机制与证据边界</h2><p>连续方程保留有符号储量；负储量的水位映射回子网格最低高程；动量仅在旧储量已负时限制继续出流。它不能保证本步从正体积不被过量抽空。干组无雨也出现问题，空间又集中于规定水位边界，与湿干转换过抽机制相符。</p><p>这尚未逐格验证触发面的贡献；没有证据把它直接归因为DEM高程错误、建筑约束或真实河道地形。单格最低负储量约−38 m³，结束时干组12格、雨组517格仍为负，不能一概视为已消失的舍入误差。</p></section><section><h2>标签准入与下一步</h2><p class="note">可继续做带标记的三区整体探索。直接标记单元隔离，其余是候选部分参考；目前不正式放行。整体水量平衡通过，并不保证局部正性、邻域无误差或地表流路正确。</p><p>__FRAGMENTATION__</p><p>本次新峰值、新在线掩膜和当前障碍接触在同一支持上重新构建；原政策、注册表、旧标签和训练限制保留。没有增加空间缓冲、裁零或把质量标记加入输入。</p><p>优先补紧凑的逐单元跨阈值和触发步骤格面诊断，按7个水域组汇总；先解释边界湿干数值过程。若要证明修正后的积水范围稳定，仍需另行授权有限对照。</p></section><section><h2>可复核证据</h2><p><a href="README.md">完整说明与限制</a> · <a href="scope_coverage.csv">分区覆盖</a> · <a href="wet_area_exposure.csv">积水支持损失</a> · <a href="principal_peak_regions.csv">主要峰值连片</a> · <a href="receiver_group_exposure.csv">水域组</a> · <a href="static_cohort_comparison.csv">静态分布对比</a> · <a href="sampled_first_detection.csv">抽样首次观测</a> · <a href="negative_cell_evidence.csv">单元证据</a> · <a href="source_mechanism_evidence.json">源码机制</a> · <a href="summary.json">检查明细</a></p></section></main></html>'''
 page=page.replace('__TABLE__',table_html(small,[('case','控制组'),('cells','在线曾负单元'),('time','全域首次出现(s)'),('ground','三区地面占比'),('wet','三区积水直接重叠')])).replace('__WARDS__',table_html(wardrows,[('ward','区/范围'),('dry','干组单元'),('rain','雨组单元'),('rainwet','雨组积水直接重叠')]))
 (report/'review.html').write_text(page.replace('__FRAGMENTATION__',html.escape(fragmentation_note)))
 print(json.dumps({'report':str(report/'review.html'),'AnaSOP_4_6_7_updated':update_sop,'training':False}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--update-sop',action='store_true');main(a.parse_args().update_sop)
