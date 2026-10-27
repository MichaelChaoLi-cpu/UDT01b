"""Whole-three-ward review of actual negative crossings, without new model changes."""
from pathlib import Path
import os,sys,json,csv,re,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_negative_trace_controls import OUT,REPORT,CASES,INTERFACE
from prepare_sfincs_cpu_build import sha,dump
from reference_label_policy import event_admission
from audit_three_ward_negative_volumes import readnp

def rows(p):
 with p.open()as f:return list(csv.DictReader(f))
def main():
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen controls')
 result=json.loads((report/'summary.json').read_text());budget=json.loads((report/'budget/summary.json').read_text());assert result['all_checks_passed']and budget['all_checks_passed']
 groups=rows(report/'water_group_cell_summary.csv');first=rows(report/'first_crossing_evidence.csv');wet=rows(report/'wet_support_exposure.csv');patch=rows(report/'principal_peak_regions.csv');coverage=rows(report/'scope_coverage.csv')
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];ward=support['Ward Code'];area=support['Ground Area'];names=json.loads((ROOT/INTERFACE/'output_schema.json').read_text())['groups']
 admission={};manifest=json.loads((out/'run_manifest.json').read_text())
 for run in manifest['runs']:
  case=run['case_id'];folder=ROOT/run['folder'];prior=ROOT/run['previous_e2e_folder'];b=budget['case_summaries'][case];x=result['cases'][case];masks=readnp(out/(case+'_candidate_reference.npz'))
  from netCDF4 import Dataset,chartostring
  with Dataset(folder/'sfincs_his.nc')as d,Dataset(prior/'sfincs_his.nc')as old:
   status=float(d['status'][0])==0;section_names=np.array_equal(chartostring(d['crosssection_name'][:]),chartostring(old['crosssection_name'][:]))
  gates={'completion':True,'identity':True,'geometry':x['all_written_states_and_peaks_identical_to_previous'] and x['boundary_closure_audit']['geometry_admission_gate'],'fields':x['all_written_states_and_peaks_identical_to_previous'],'time':True,'clock_alignment':True,'boundary':status and section_names,'rain':abs(b['actual_applied_rain_m3']-b['continuous_encoded_input_rain_m3'])<=1e-6*max(b['continuous_encoded_input_rain_m3'],1),'budget':b['budget_endpoint_pass']and b['budget_prefix_pass'],'censoring':x['all_written_states_and_peaks_identical_to_previous']}
  preview=event_admission(gates,masks)
  admission[case]={'required_gates':gates,'frozen_policy_function_preview':preview,'evidence_level':'online every-native-step negative screening; frozen function state name retains snapshot wording','candidate_partial_numerical_reference':preview['technical_reference'],'formal_registry_changed':False,'training_release':False,'physical_event_accuracy_verified':False,'mask_extent':'direct affected cells only; propagated errors not bounded'}
 dump(report/'reference_admission_assessment.json',admission)
 binding=json.loads((ROOT/INTERFACE/'solver_binding.json').read_text())
 dump(out/'qualification.json',{'solver_image_id':binding['image_id'],'patch_sha256':binding['patch_sha256'],'actual_trace_end_to_end_qualified':True,'qualification_scope':'exactly two existing matched regular10m fixed-geometry controls','actual_run_ids':[r['run_id']for r in manifest['runs']],'actual_clock_budget_output_pass':budget['all_checks_passed'],'crossing_and_label_support_pass':result['all_checks_passed'],'declared_boundary_closures_satisfied':result['all_declared_boundary_closures_satisfied'],'candidate_partial_references':{k:a['candidate_partial_numerical_reference']for k,a in admission.items()},'formal_registry_changed':False,'formal_reference_release':False,'training':0,'batch_58':0,'physical_accuracy_verified':False,'older_build_stage_receipts_preserved':True})
 # Pinned physical-source evidence: diagnostic additions do not alter these equations.
 source=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src';evidence={}
 patterns={'sfincs_boundaries.f90':['if (bndtype == 1)','q(ip) = ub * hnmb','if (z_volume(nmi) <= 0.0)','kcuv(ip) == 3'],
 'sfincs_domain.f90':['call initialize_boundaries','index_kcuv2'],
 'sfincs_structures.f90':['kcuv(indx) = 0','istruc(indx) = 1'],
 'sfincs_momentum.f90':['kcuv(ip) == 1','if (z_volume(nm) < 0.0)','if (z_volume(nmu) < 0.0)'],
 'sfincs_continuity.f90':['z_volume(nm) = z_volume(nm) + dvol','elseif (z_volume(nm) <= 1.0e-6)'],
 'sfincs_lib.f90':['call initialize_domain','call read_structures','call update_boundaries(t, dt','call compute_fluxes(dt','call compute_water_levels(t, dt'],
 'sfincs_input.f90':["'bndtype',bndtype,1","'alpha',alfa","'dtmax',dtmax"]}
 for name,needles in patterns.items():
  text=(source/name).read_text();evidence[name]={'sha256':sha(source/name),'lines':[{'line':i,'text':line.strip()}for i,line in enumerate(text.splitlines(),1)if any(n in line for n in needles)]}
 evidence['effective_configs']={}
 for run in manifest['runs']:
  cfg={k.strip():v.strip()for line in(ROOT/run['folder']/'sfincs.inp').read_text().splitlines()if'='in line for k,v in[line.split('=',1)]}
  evidence['effective_configs'][run['case_id']]={k:cfg.get(k,default)for k,default in [('bndtype','1'),('btfilter','60.0'),('btrelax','3600.0'),('alpha','0.50'),('dtmax','60.0'),('huthresh','0.05')]}
 dump(report/'source_mechanism_evidence.json',evidence)
 # Scientific exploratory figure: first observed negative endpoint on full domain plus mechanisms.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 from matplotlib.lines import Line2D
 fig,axs=plt.subplots(2,2,figsize=(12,11),layout='constrained',height_ratios=[1.8,1])
 extent=[-11.36,-3.66,-42.44,-32.32]
 for i,case in enumerate(CASES):
  d=readnp(out/(case+'_actual_negative_times.npz'));m=readnp(out/(case+'_candidate_reference.npz'));t=d['First Below Actual Endpoint'];r,c=np.where(np.isfinite(t));ax=axs[0,i]
  ax.imshow(np.where(valid,1,np.nan),origin='upper',extent=extent,cmap=ListedColormap(['#e9edef']),vmin=0,vmax=1,interpolation='none')
  ax.imshow(np.where(valid&(m['Peak Mean Depth Lower']>.1),1,np.nan),origin='upper',extent=extent,cmap=ListedColormap(['#a3c3ce']),vmin=0,vmax=1,interpolation='none')
  for code,label in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,(ward==code).astype('f4'),levels=[.5],colors='#6f828a',linewidths=.5)
   rr,cc=np.where(valid&(ward==code));ax.text(-11.36+(np.median(cc)+.5)*.01,-32.32-(np.median(rr)+.5)*.01,label,fontsize=9,color='#203c49',bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':2})
  dots=ax.scatter(-11.36+(c+.5)*.01,-32.32-(r+.5)*.01,c=t[r,c]/60,s=7,cmap='plasma',vmin=60,vmax=240,zorder=5)
  ax.set(title=('Dry + raised stage'if i==0 else'Rain + raised stage')+f" | {len(r)} cells",xlabel='EPSG:6677 easting (km)',ylabel='EPSG:6677 northing (km)');ax.set_aspect('equal')
 fig.colorbar(dots,ax=list(axs[0]),label='First below-threshold actual endpoint (min)',shrink=.75,pad=.02)
 labels=['Boundary outflow dominant','Interior outflow dominant','Tied outflow'];colors=['#bc6949','#427f96','#a8a28a']
 for i,case in enumerate(CASES):
  stat=result['cases'][case]['first_down_mechanisms'];total=result['cases'][case]['ever_negative_cells'];left=0
  for label,color in zip(labels,colors):
   value=stat.get(label,0);axs[1,0].barh(i,value/total*100,left=left,color=color,label=label if i==0 else None);left+=value/total*100
 axs[1,0].set(yticks=[0,1],yticklabels=['Dry','Rain'],xlim=[0,100],xlabel='Share of first-crossing cells (%)',title='Outgoing-face contribution at first crossing');axs[1,0].legend(fontsize=8,loc='lower left',bbox_to_anchor=(0,-.43))
 for case,color in zip(CASES,['#427f96','#bc6949']):
  t=np.array([float(r['first_below_endpoint_s'])/3600 for r in first if r['case']==case]);axs[1,1].hist(t,bins=np.arange(0,4.01,.5),histtype='step',lw=1.6,color=color,label='Dry'if case==CASES[0]else'Rain')
 axs[1,1].set(xlabel='First below actual endpoint (h)',ylabel='Unique cells',title='Occurrence distribution, one first crossing per cell');axs[1,1].legend(fontsize=8)
 fig.savefig(report/'actual_crossing_overview.png',dpi=160);plt.close(fig)
 def wetrow(case,scope='Three wards',th=.1):return next(r for r in wet if r['case']==case and r['scope']==scope and float(r['threshold_m'])==th)
 summaries=[]
 for case in CASES:
  x=result['cases'][case];s=x['first_down_mechanisms'];n=x['ever_negative_cells'];w=wetrow(case)
  summaries.append({'case':case,'unique_negative_cells':n,'first_endpoint_s':x['first_negative_endpoint_s'],'down_events':x['down_crossing_events'],'up_events':x['up_crossing_events'],'repeated_cells':x['repeated_crossing_cells'],'boundary_dominant_first_pct':s.get('Boundary outflow dominant',0)/n*100,'interior_dominant_first_pct':s.get('Interior outflow dominant',0)/n*100,'negative_wet_support_pct':float(w['negative_overlap_pct']),'candidate_partial_reference':admission[case]['candidate_partial_numerical_reference']})
 lines=[]
 for r in summaries:
  case=r['case'];x=result['cases'][case];b=budget['case_summaries'][case];top=[p for p in patch if p['case']==case and float(p['threshold_m'])==.1]
  lines.append(f"|{case}|{r['unique_negative_cells']}|{r['first_endpoint_s']:.6f}|{r['down_events']}|{r['up_events']}|{r['repeated_cells']}|{r['boundary_dominant_first_pct']:.3f}%|{r['negative_wet_support_pct']:.4f}%|")
 text='''# 两个控制组实际重放：负体积触发机制与参考支持\n\n只重放既有合成强迫的干组和降雨组，完整全域10 m、4小时、同一抬升水位；原科学曲线、几何、数值参数、谱系和划分逐字节保留。使用冻结的负体积诊断镜像，未训练、未执行58个情景。Framelog MILI-D-20261015-012。\n\n## 实际事件结果\n\n|组别|曾负单元|首次实际端点(s)|向下穿越次数|恢复次数|重复穿越单元|首次边界出流主导|三区>0.10m积水支持直接重叠|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''+ '\n'.join(lines)+'\n\n'
 for case in CASES:
  x=result['cases'][case];b=budget['case_summaries'][case];n=x['down_crossing_events'];me=x['all_down_mechanisms']
  text+=f"- {case}：{me.get('horizontal_depletion',0)}/{n} 次向下穿越存在负的实际水平增量，{me.get('nonnegative_rain',0)}/{n} 次雨量项非负；{me.get('outflow_exceeds_previous_plus_inflow_and_source',0)}/{n} 次单面流出合计超过前态体积加流入与实际入雨。全过程单格最小体积 {x['minimum_volume_m3']:.6f} m³，事件结束仍有 {x['final_negative_cells']} 格低于阈值。全域最大累计预算残差 {b['max_abs_step_budget_residual_m3']:.9g} m³。\n"
 text+='''\n以上是实际计算步的有符号记账证据。不是观测积水，也不是从300秒帧推测或把旧记录补写成真值。首次／最小值及穿越记录在同一步实际时钟上核验，逐步负值计数可由向下／向上穿越序列独立重建。\n\n## 成因与水域组解释\n\n跨阈值时的四向交换已经直接记录，因此可以区分边界出流和内部出流；按唯一单元的首次穿越统计空间影响，另留全部重复穿越，避免用振荡次数代替影响范围。水域组的单元表按真实开放接收面连接，可多重归属，不能直接相加。格面表则按实际相邻边界组编码，0类表示内部／缺失面，不等于某条河流。最大出流面相同量时按固定方向顺序选择代表，不把它当唯一物理成因。\n\n固定源码显示，当前使用默认弱反射给定水位边界。该分支在旧体积非正时阻止继续出流，内部动量也有相应限制；但旧体积仍为正时，通量没有按本步可用体积统一收缩。实际水平交换可以在单步耗尽存量和入雨后继续扣减，产生负的有符号储量。低储量随后被映射回最低水位，有符号债务保留。这解释了为什么全域水量收支闭合仍可伴随局部负体积。\n\n这是对已观察计算步骤的数值机制判断，不证明边界水位或DEM实测有错，也不能只凭相关性认定某个参数是唯一原因。没有运行缩短步长、改变边界或加入正性限制的反事实对照，实际传播误差仍无界定。原源码及当前配置的证据另列。\n\n## 关闭边界面违规交换\n\n'''
 for case in CASES:
  z=result['cases'][case]['boundary_closure_audit']
  text+=f"- {case}：全部 {z['closed_boundary_faces']} 个已声明关闭的接收水域边界面中，{z['faces_with_nonzero_applied_flux']} 个面在实际步历史中出现非零流量；累计向外 {z['gross_closed_outward_m3']:.6f} m³、向内 {z['gross_closed_inward_m3']:.6f} m³（双向累计，非净量）。穿越触发记录仅捕获其中 {z['closed_crossing_face_observations']} 条关闭面交换，不代表整个事件只有这些异常。\n"
  for code,g in z['by_water_group'].items():
   if g['faces_with_nonzero_flux']:text+=f"  - 水域 {code} {g['name']}：{g['faces_with_nonzero_flux']} 面；向外 {g['gross_outward_m3']:.6f} m³、向内 {g['gross_inward_m3']:.6f} m³。\n"
 text+='''\n官方求解器固定源码先建立边界面列表，再读取薄坝并将对应面置为关闭；弱反射边界更新仍遍历先前列表，只检查结构标记3，未排除薄坝关闭标记0，而内部动量更新跳过标记0。这与关闭面被边界更新重新赋流的实际诊断一致。该缺陷与正体积单步透支是两个需区分的数值问题。本轮只确认，不修改源码物理行为。\n\n## 主要积水区和标签准入\n\n'''
 for case in CASES:
  wr=wetrow(case);top=[p for p in patch if p['case']==case and float(p['threshold_m'])==.1]
  text+=f"- {case}：三区峰值平均水深>0.10 m的开敞支持 {float(wr['certain_wet_ground_km2']):.6f} km²，负体积单元直接重叠 {float(wr['negative_overlap_km2']):.6f} km²（{float(wr['negative_overlap_pct']):.4f}%）；含假设建筑障碍接触的全部隔离比例 {float(wr['quarantine_overlap_pct']):.4f}%。前10峰值连片在直接隔离后，最大剩余连通部分至少保留 {min(float(p['largest_retained_support_pct'])for p in top):.3f}% 原开敞支持。\n"
 text+=f'''\n干组港区的小面积>0.10 m支持中，负体积重叠仍为 {float(wetrow('dry_raised','Minato')['negative_overlap_pct']):.2f}%，不能被全域平均掩盖。0.05/0.10/0.30/0.50 m及三区各区、全域和跨区贡献范围均单独保留。阈值是分析规则，不是安全标准。\n\n新旧两组的全部写出水位、体积、速度状态和在线峰值逐元素对照；结果见核验明细。新候选峰值标签在原0.50 m开敞地面上重新积分，负体积在线掩膜、浅水区间及假设障碍接触重算，未套用旧原生h。已核验的同结果仅说明新诊断没有改变模型输出，不证明消除负体积后的范围或排水路径稳定。峰值连片不是同时水面或完整动态流路。\n\n参考准入使用原冻结政策逐项评估：完成、身份／几何、字段、时间和实际时钟、边界、雨量、水量平衡与浅水证明；通过后才审查隔离后的剩余支持。结果见 reference_admission_assessment.json。两组均因关闭边界约束失败而未通过全事件几何门槛，暂缓参考标签准入；局部隔离不能挽救全事件失败。当前生成的标签文件仅供核查，默认注册表和加载器未改，训练未放行。原政策函数名称仍含snapshot，本次证据实际增强为每原生步在线筛查，不重命名或暗改旧政策。直接隔离不构成误差传播边界。\n\n## 后续优先项\n\n继续保持三区整体视角，首先修复关闭薄坝边界仍被更新赋流的问题，再评审统一的正性保持格面通量限制：边界面与内部面共同检查供水格的可用体积，使用同一面共享交换量，保留守恒。不要通过裁零修正负储量。本轮只诊断，尚未修改通量／边界，也未证明修正后的结果。任何下一组有限修正对照须另行明确范围，58个库情景继续未运行。\n\n## 证据文件\n\n- first_crossing_evidence.csv：每个实际曾负单元的首次穿越、格面贡献与组归属。\n- water_group_cell_summary.csv：7个水域组和无直接连接类，单元数非互斥。\n- crossing_face_mechanisms.csv：实际触发面流入／流出及最大出流面归属；不是全事件预算。\n- wet_support_exposure.csv、scope_coverage.csv、principal_peak_regions.csv：主要积水支持与隔离影响。\n- boundary_closure_audit.json、all_closed_boundary_faces.csv：全部关闭边界面的每步历史核查及水域组汇总。\n- reduction-order-note.json：审计求和顺序修正，保留原失败记录；按原生顺序的最终有符号储量与预算精确相等。\n- reference_admission_assessment.json：原政策的候选准入预览，未修改默认访问。\n- budget/：实际时钟、预算与输出对接复核。\n- 原始每步、穿越及单元汇总在 {OUT}/runs；所有原运行保留。\n'''
 (report/'README.md').write_text(text)
 def tab(data,columns):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>'for key,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[key]))+'</td>'for key,label in columns)+'</tr>'for r in data)+'</tbody></table></div>'
 compact=[{'case':'干组'if r['case']==CASES[0]else'降雨组','cells':r['unique_negative_cells'],'down':r['down_events'],'repeat':r['repeated_cells'],'boundary':f"{r['boundary_dominant_first_pct']:.2f}%",'wet':f"{r['negative_wet_support_pct']:.4f}%"}for r in summaries]
 groupdisplay=[{'case':'干组'if r['case']==CASES[0]else'降雨组','group':r['water_group_code']+' '+r['water_group_name'],'cells':r['negative_unique_cells'],'down':r['down_crossings'],'boundary':r['boundary_dominant_down'],'internal':r['interior_dominant_down']}for r in groups]
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两个控制组真实负体积触发核查</title><style>body{font:16px/1.8 system-ui,sans-serif;background:#eef3f4;color:#203b47;margin:0}main{max-width:1120px;margin:auto;padding:28px}section{background:white;border:1px solid #dce5e8;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:29px;line-height:1.4}h2{font-size:21px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dce5e8}.scroll{overflow:auto}a{color:#146d83}.note{border-left:4px solid #bd8640;padding-left:16px}@media(max-width:650px){main{padding:12px}section{padding:17px}}</style><main><p>Mili · 全域10 m · 两个既有控制重放</p><h1>实际跨阈值与触发格面证据</h1><p>完整4小时干组与降雨组，使用冻结诊断镜像；科学曲线、几何和数值参数保留。未训练、未运行58个情景。</p><section><h2>全域结果与主要积水支持</h2>__SUMMARY__<p>“边界主导”按每格首次穿越时，所有边界出流与内部出流的比较计算。积水重叠以三区峰值平均水深>0.10 m的开敞支持为分母。</p><img src="actual_crossing_overview.png" alt="真实首次跨阈值分布及首次出流贡献"></section><section><h2>按水域组核查</h2>__GROUPS__<p>单元按开放接收面可多重归属，组间数量不能直接相加。重复穿越另留，避免把振荡次数当影响范围。组0不强制归给最近河流。</p></section><section><h2>机制与结论边界</h2><p>实际水平交换、入雨与储量在同一步对接，四向格面项可独立重构。当前边界／内部出流规则在旧体积非正时截停出流，但对仍为正的微小存量没有统一的本步可用体积约束。实际步骤可净流出过量而出现负储量。</p><p class="note">全域水量闭合不保证局部体积非负。该机制解释不等于DEM有错，也没有通过反事实运行证明某个参数是唯一成因。直接掩膜不能界定传播误差。</p></section><section><h2>关闭边界约束失败</h2><p class="note">578个已声明关闭的接收水域边界面中，干组2个、降雨组18个存在实际非零交换。固定源码的边界列表先于薄坝关闭建立，边界更新未排除关闭标记，与实测诊断一致。两组均未通过全事件几何门槛，暂缓参考标签准入。</p><p>关闭面全部逐步历史与按水域组累计交换见<a href="boundary_closure_audit.json">关闭边界核查</a>和<a href="all_closed_boundary_faces.csv">逐面明细</a>。小面积直接重叠与全域预算闭合不能挽救几何约束失败。</p></section><section><h2>主要积水区与参考准入</h2><p>新旧写出状态、在线峰值及重新积分的候选标签逐项对照；新诊断是否改变结果见核验明细。这不等于消除负体积后结果已稳定，峰值连片也不是同时水面或动态排水流路。</p><p>沿用原政策评估全事件硬门槛与单元隔离。两组均暂缓准入，标签文件只保留供审查；旧注册表与默认加载器未改，训练仍未放行。干组港区的小面积积水支持限制继续单列。</p><p>下一步先修复关闭边界赋流，再评审覆盖内部面与边界面的守恒正性限制，保持三区整体对照。本轮未修改通量或裁零。</p></section><section><h2>可复核证据</h2><p><a href="README.md">完整说明</a> · <a href="summary.json">实际触发检查</a> · <a href="first_crossing_evidence.csv">唯一单元首次穿越</a> · <a href="water_group_cell_summary.csv">水域组单元汇总</a> · <a href="crossing_face_mechanisms.csv">实际格面机制汇总</a> · <a href="wet_support_exposure.csv">积水支持损失</a> · <a href="principal_peak_regions.csv">主要峰值连片</a> · <a href="reference_admission_assessment.json">候选准入评估</a> · <a href="budget/summary.json">时钟与预算复核</a></p></section></main></html>'''
 (report/'review.html').write_text(page.replace('__SUMMARY__',tab(compact,[('case','组别'),('cells','曾负单元'),('down','向下穿越'),('repeat','重复穿越单元'),('boundary','首次边界出流主导'),('wet','三区积水直接重叠')])).replace('__GROUPS__',tab(groupdisplay,[('case','组别'),('group','水域代码／名称'),('cells','曾负单元'),('down','向下穿越'),('boundary','边界主导次数'),('internal','内部主导次数')])))
 # Research-skill routed evidence update only, with historical artifacts retained.
 p=ROOT/'docs/AnaSOP.md';backup=report/'previous-AnaSOP.md'
 if not backup.exists():backup.write_bytes(p.read_bytes())
 before=p.read_text();parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M)
 counts=', '.join(f"{result['cases'][c]['ever_negative_cells']} for the {c.split('_')[0]} control"for c in CASES)
 add={4:f'''Subsequent separately authorized control versions have now collected actual native-step crossing and minimum-volume histories, together with triggering face exchanges. Every-step negative masks identify {counts}. These new records are bound to the actual clocks and original forcing and geometry; they do not backfill missing observations into historical artifacts. Native signed volumes remain unmodified for budgets, current peak labels are reintegrated over original eligible ground, and the existing direct quarantine and shallow-interval rules remain unchanged. No response-quality flag is added as a predictor. The complete applied-step history also records unintended exchange through declared closed receiving-water faces, affecting two faces in the dry control and eighteen in the rain control. These diagnostic artifacts remain quarantined from reference use.''',6:'''Actual downward-crossing steps now permit numerical attribution through the signed continuity update and four native-face contributions. Separate first-crossing cells from recurrent transitions, and distinguish the group of an actual outgoing boundary face from a cell's potentially multiple direct-open receiver memberships. The current weakly reflective boundary and interior momentum rules restrict outward transport after the prior storage is nonpositive or negative; they do not guarantee positivity after exhausting a small positive donor volume in one step. Interpret the observed depletion evidence as a numerical mechanism under prescribed stages, without inferring erroneous measured terrain, unique parameter causation or a bounded propagation distance. Agreement with previous written outputs tests diagnostic noninterference, not stability after a future positivity correction. A separate closure defect is supported by the runtime face history and pinned source order: boundary-face lists are constructed before thin-dam closures, and subsequent boundary updates omit the closed-face exclusion. Address closure compliance before evaluating conservative positivity corrections.''',7:'''The two matched four-hour regular-grid replays with compact crossing diagnostics have now completed. Their actual clocks, same-step budgets, output correspondence, transition reconstruction and current ground-label support are audited. The unchanged frozen policy holds both new control versions because their declared closed boundary constraints fail the whole-event geometry gate. Direct online-negative and assumed-barrier exclusions cannot rescue that failure. Candidate label files are retained for audit only. The historical registry and default reference access remain closed; no training or prospective batch is executed. Real-world flood accuracy and full dynamic-route labels remain unverified. Any subsequent conservative positivity change and finite comparison must be separately scoped; do not repair signed storage by clipping or expand spatial exclusions without evidence.'''}
 for i,part in enumerate(parts):
  if not part.startswith('## '):continue
  n=int(re.match(r'## (\d)',part).group(1))
  if n in add and add[n]not in part:parts[i]=part.rstrip()+'\n\n'+add[n]+'\n\n'
 p.write_text(''.join(parts));original=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M)
 changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(original,parts)if a!=b};assert changed=={4,6,7}
 dump(report/'anasop-update.json',{'sections_changed':[4,6,7],'skills':['data-preprocessing','estimation-framework-planning'],'human_record':'MILI-D-20261015-012','prior_sha256':sha(backup),'current_sha256':sha(p),'new_final_variables':0,'new_formulas':0,'other_sections_unchanged':True})
 print(json.dumps({'report':str(report/'review.html'),'candidate_technical_references':{k:v['candidate_partial_numerical_reference']for k,v in admission.items()}}))
if __name__=='__main__':main()
