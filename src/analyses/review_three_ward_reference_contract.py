# Deliver the numerical contract, diagnostic registry and whole-domain QC review.
import os
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,csv,re,argparse,sys,html
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from reference_label_policy import POLICY,CONTRACT
from audit_three_ward_reference_contract import OUT,REPORT,BASE,LABEL,INPUT,read_npz,sha,dump


def prepare(root):
 root=Path(root);out=root/OUT;report=root/REPORT
 diag=json.loads((report/'run_diagnostics.json').read_text());controls={}
 for case in ['rain60_low','rain100_low','rain60_raised','dry_raised']:
  state=diag[case+'_corrected_10m']['label_admission']
  controls[case]={'family_id':'CONTROL-LEGACY','split':'open_control','technical_reference':state['technical_reference'],'state':state['state'],
   'hold_reasons':state['reasons'],'diagnostic_masks':case+'_reference_masks.npz','training_release':False}
 registry={'version':POLICY['version'],'policy_sha256':sha(out/'label_policy.json'),'output_contract_sha256':sha(out/'output_contract.json'),'controls':controls,'default_response_access':'closed','diagnostic_access':'explicit opt-in for these already-open controls only'}
 if (out/'freeze.json').exists():
  assert json.loads((out/'reference_registry.json').read_text())==registry
  return
 dump(out/'reference_registry.json',registry)
 lib=root/'data/processed/three_ward_minimum_scenarios_v1';runs=json.loads((lib/'run_manifest.json').read_text())['pending_runs']
 dump(out/'future_run_contract_bindings.json',{'forcing_library_freeze_sha256':sha(lib/'freeze.json'),'policy_sha256':registry['policy_sha256'],'output_contract_sha256':registry['output_contract_sha256'],
  'execution_authorized':False,'training_authorized':False,'all_runs_unrun':True,
  'requires_before_execution':['actual integration-clock or equivalent matched-step budget export qualification','same frozen forcing and geometry','output and source/sink ledger validation','separate execution instruction'],
  'runs':[r|{'policy_sha256':registry['policy_sha256'],'output_contract_sha256':registry['output_contract_sha256'],'response_available':False} for r in runs]})
 print('Prepared registry and58 unrun contract bindings',flush=True)


def documents(root):
 root=Path(root);report=root/REPORT;path=root/'docs/AnaSOP.md';before=path.read_text();backup=report/'previous-AnaSOP.md'
 if not backup.exists():backup.write_text(before)
 parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);old=list(parts)
 for i,s in enumerate(parts):
  if s.startswith('## 4. '):
   s=s.replace('The working tolerances define this version\'s technical package and remain subject to research-policy freeze before new held-out responses.',
    'These technical gates are now versioned and frozen before new held-out responses. They are engineering screening settings, not physical accuracy standards or authorization to train.')
   s=s.replace('The forcing catalog is frozen for audit; the label-quality policy and formal reference release remain pending.',
    'The forcing catalog and numerical label-quality rules are frozen for audit; strict whole-event reference admission and formal training release remain on hold pending actual-clock qualification.')
   if '### Numerical reference admission and output requirements' not in s:
    s=s.rstrip()+"""

### Numerical reference admission and output requirements

Cell-level rules and whole-event qualification are distinct gates. A complete event must match its frozen parent family, forcing, geometry, solver and output versions; finish its four-hour horizon with successful status; preserve the native map and boundary associations; provide finite required storage and complete signed boundary flux; prove any shallow censoring; and pass water-budget and integration-clock checks. Missing diagnostics or a failed global gate hold the entire event. A small spatial exclusion cannot rescue a failed event, and none of these gates releases training automatically.

Cell quarantine retains invalid or missing bounds, any sampled native signed bookkeeping volume below minus 0.001 cubic metres, and contact with the assumed building barrier within 0.01 m. The current flag maps remain unchanged. These direct affected cells do not bound numerical error propagation. Report exact, interval, wide, quarantined and unknown support, their open-ground areas, and quarantined shares of certainly wet ground for each ward, the three wards together and the full modeled domain. A development-only union is retained for fixed-support diagnostics; it is not a predictor or a permanent exclusion for future events. The four existing controls' quarantine union covers 0.07694875 square kilometres.

Primary labels use the regular 10 m reference only. Refined-grid intervals remain diagnostic. Exact-width tolerance is 0.0000001 m; admitted interval width is at most 0.01 m plus 0.000001 m storage tolerance. Wider intervals remain withheld, preserving both endpoints. Unknown values, shallow masks and intervals are never replaced by dry zeros or midpoints. The same event masks and core parent pool apply to every compared learning method. Matching-grid comparisons additionally require a complete union of their numerical exclusions; an incomplete refined-grid barrier mask cannot establish clean paired reference support.

The existing eight runs pass the new nominal endpoint-budget gate. Six rainy runs fail prefix closure around rainfall onset when evaluated at nominal written times. Pinned solver-source inspection shows that a state is advanced before it is written with a scheduled output time; actual integration times are not stored in these files. This supports a clock-mismatch mechanism, not a demonstrated correction or proof of artificial water creation. All eight runs remain on strict event hold because actual state/flux clock alignment is unverified. Existing peak bounds and masks remain available as explicitly requested open development diagnostics.

The peak-output profile retains native online maximum levels over the complete target window, synchronized full fields at no more than 300-second spacing, and complete boundary-face flux histories with nominal gaps no more than 0.401 seconds. Actual integration times or an equivalent verified checkpoint ledger are required for strict process closure; scheduled timestamps alone do not meet that requirement. Signed storage is retained for bookkeeping balance. The reconstructed eligible-ground target is kept separate from solver whole-cell mean depth and from depth above a native minimum. Absence of negative volume at written frames does not prove every-step nonnegativity. No new or sealed responses are read, generated or released.

"""
   parts[i]=s
  if s.startswith('## 6. '):
   s=s.replace('A proposed pilot interval-width limit is 0.01 m, reviewed using training/validation evidence before test opening; wider intervals remain flagged and excluded from primary regression. Do not choose the limit after viewing test performance.',
    'The versioned numerical interval-width limit is 0.01 m with a 0.000001 m storage tolerance; wider intervals remain flagged and excluded from primary regression. This screen is frozen before new test responses. A later policy revision requires a new version and a fresh untouched evaluation set; do not retune it after viewing held-out performance.')
   if '### Whole-domain water-budget and temporal-output gate' not in s:
    marker='### Connection to active measures'
    addition=r"""### Whole-domain water-budget and temporal-output gate

For every event, retain a signed hydraulic-domain ledger at the actual state checkpoints:

\[
E_e(t)=S_e(t)-S_e(0)+B^{\mathrm{out}}_e(t)-B^{\mathrm{in}}_e(t)-P_e(t)-J_e(t)+K_e(t).
\]

Here \(E_e(t)\) is water-budget residual, \(S_e(t)\) is signed native bookkeeping storage, \(B^{\mathrm{out}}_e(t)\) and \(B^{\mathrm{in}}_e(t)\) are cumulative gross boundary outflow and inflow, \(P_e(t)\) is cumulative rainfall input including the fixed roof allocation, and \(J_e(t)\) and \(K_e(t)\) are other cumulative sources and sinks. Every volume is in cubic metres. Split boundary flux signs at each native interface before grouping, so simultaneous inflow and outflow cannot cancel the denominator. Do not clip negative storage or remove quarantined cells from this global ledger. Roof runoff is already included in rainfall input; adding it a second time is an accounting error. Other sources and sinks are zero in the baseline and require an explicit extension if enabled.

Use a positive scale and a hybrid absolute-relative engineering gate:

\[
M_e(t)=\max\{P_e(t)+J_e(t)+B^{\mathrm{in}}_e(t),B^{\mathrm{out}}_e(t)+K_e(t),|S_e(0)|,1\,\mathrm{m}^3\},\qquad
|E_e(t)|\leq\max\{1\,\mathrm{m}^3,\epsilon(t)M_e(t)\}.
\]

Here \(M_e(t)\) is the comparison volume and \(\epsilon(t)\) is 0.001 at the event endpoint and 0.01 at each written-state prefix. These are numerical screening gates, not universal hydraulic standards or real-event accuracy tolerances. The denominator remains meaningful with zero rainfall and incoming boundary water. Rainfall is integrated over exact forcing knots; final reported cumulative precipitation is a separate float32-roundoff diagnostic. Integrate instantaneous boundary flux over actual written times by the trapezoidal rule and retain right-rectangle differences; require their final difference to fit the endpoint gate. Their agreement does not bound unobserved flux. Unknown fluxes are missing evidence, never zero. Per-ward water closure is not claimed without cross-ward internal-face exchanges.

Actual state and flux integration clocks, or verified equivalent step-integrated checkpoint ledgers, are required before interpreting process closure. Existing nominal clocks are insufficient for that certification. Retain failed checks and distinguish time-alignment holds from established physical mass errors; do not shift clocks, remove onset frames or relax tolerances to force a pass.

The current peak-only profile does not require new denser-output simulations. Existing 300-second states support coarse evolution and aggregation bounds. Future timing and drainage-path targets would start with at most 60-second synchronized fields and a 30-second development comparison, with task-specific tolerance qualification before use. Cadence alone does not establish accuracy. Dynamic conservation or active operation additionally requires interval-integrated native-face flux, rain and source/sink increments, matched endpoint storage and verified connectivity. Single-station history, cell-centred velocity and boundary-only flux cannot replace these fields. Four-hour targets do not imply complete recession.

"""
    s=s.replace(marker,addition+marker)
   parts[i]=s
  if s.startswith('## 7. '):
   s=s.replace('Label arithmetic: constructed and checked; physical/event accuracy: unestablished | Freeze point/interval quality policy; preserve grid uncertainty; no broad denser-output requirement for peak-only labels.',
    'Label arithmetic and versioned rules: checked; strict reference admission: held; physical/event accuracy: unestablished | Preserve quarantine and intervals; verify actual state/flux integration clocks and process closure before reference release; no broad denser-output requirement for peak-only labels.')
   if 'Numerical cell-admission and output contracts are implemented;' not in s:
    s=s.replace('Reference responses for the new forcings remain absent.',
    'Reference responses for the new forcings remain absent. Numerical cell-admission and output contracts are implemented; actual-clock and process-budget qualification still hold the inspected controls from strict reference admission.')
   s=s.replace('reference admission, target-output qualification, training-only scaling and response-release rules must be resolved before learning.',
    'reference admission, actual-clock output qualification, training-only scaling and response release must be resolved before learning.')
   parts[i]=s
 changed=[parts[i].splitlines()[0] for i in range(len(parts)) if parts[i]!=old[i]]
 assert all(x.startswith(('## 4. ','## 6. ','## 7. ')) for x in changed)
 path.write_text(''.join(parts))
 dump(report/'anasop-update.json',{'responsible_skills':{'4':'data-preprocessing','6':'estimation-framework-planning','7':'estimation-framework-planning'},'sections_changed':changed,'other_sections_unchanged':True,'preimage_sha256':sha(backup),'after_sha256':sha(path),'language':'English','decision':'MILI-D-20261015-006','candidate_variables_remain_no':True,'no_training':True})
 return changed


def deliver(root):
 root=Path(root);out=root/OUT;report=root/REPORT;v=json.loads((report/'validation.json').read_text());summary=json.loads((report/'summary.json').read_text())
 if not v['all_checks_passed']:raise ValueError('Policy/product tests must pass before delivery')
 prepare(root);changed=documents(root)
 rows=list(csv.DictReader((report/'run_admission.csv').open()));scope=list(csv.DictReader((report/'label_admission_by_scope.csv').open()));wet=list(csv.DictReader((report/'wet_quarantine_coverage.csv').open()))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 s=read_npz(root/LABEL/'common_support_10m.npz');valid=s['Eligible Ground Mask'];extent=[-11360,-3660,-42440,-32320]
 fig,axs=plt.subplots(1,4,figsize=(14,5.5),layout='constrained')
 for ax,case in zip(axs,['rain60_low','rain100_low','rain60_raised','dry_raised']):
  m=read_npz(out/(case+'_reference_masks.npz'));base=np.where(valid,0.,np.nan);base[m['Quarantine Mask']]=1
  from matplotlib.colors import ListedColormap
  ax.imshow(base,extent=extent,cmap=ListedColormap(['#e8eef1','#ba482f']),vmin=0,vmax=1,interpolation='nearest');ax.set_title(case);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 fig.legend(handles=[Patch(color='#e8eef1',label='Eligible ground'),Patch(color='#ba482f',label='Direct numerical quarantine')],loc='lower center',ncol=2)
 fig.suptitle('Existing controls: unchanged direct quarantine (strict event admission on hold)');fig.savefig(report/'quarantine_overview.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for case,color in [('rain60_low','#277a9d'),('rain100_low','#bb6034')]:
  path=report/(case+'_corrected_10m_budget_checkpoints.csv');a=list(csv.DictReader(path.open()));t=np.array([float(r['time_s']) for r in a])/3600;res=np.array([abs(float(r['residual_m3'])) for r in a]);gate=np.array([float(r['prefix_gate_m3']) for r in a]);axs[0].plot(t,res,label=case,color=color);axs[1].plot(t,res,color=color,label=case);axs[1].plot(t,gate,ls='--',color=color,alpha=.6)
 axs[0].set_xlim(.9,1.2);axs[0].set_title('Rain onset: nominal clock residual');axs[1].set_yscale('symlog',linthresh=1);axs[1].set_title('Residual versus prefix gates (dashed)')
 for ax in axs:ax.set_xlabel('Nominal output time (h)');ax.set_ylabel('Absolute budget residual (m3)');ax.grid(alpha=.25);ax.legend()
 fig.savefig(report/'nominal_budget_clock_diagnostic.png',dpi=150);plt.close(fig)
 table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['run'],'pass' if r['nominal_endpoint_budget_pass']=='True' else 'hold','pass' if r['nominal_prefix_budget_pass']=='True' else 'hold','unverified',f"{float(r['end_relative_residual'])*100:.5f}%",'hold'])+'</tr>' for r in rows)
 cells=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['case'],f"{float(r['quarantine_km2']):.6f}",f"{float(r['quarantine_percent']):.3f}%",f"{float(r['interval_km2']):.6f}"])+'</tr>' for r in scope if r['scope']=='Whole domain')
 text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区参考准入与输出合同</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:30px;color:#203745;background:#f5f7f9}section{background:#fff;border:1px solid #d9e3e9;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:22px}.note{background:#fff0df;border-left:4px solid #bd7528;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e0e7ec}img{width:100%;height:auto}a{color:#176989}.scroll{overflow:auto}code{overflow-wrap:anywhere}</style>
<h1>三区参考标签：规则已落实，严格准入暂缓</h1><p>保持三区及现有相邻汇水范围。只读核查8次已有运行；新水动力模拟0，训练0，封存响应读取0。</p>
<p class="note"><strong>规则测试通过 ≠ 水动力结果全部准入。</strong> 数值政策与输出合同已实现并冻结；已有峰值上下界和掩膜原样保留为开放开发诊断。由于实际计算时钟未记录，8次已有运行均未达到本版严格过程准入条件。</p>
<section><h2>发现的全域问题：末端闭合掩盖名义时刻偏差</h2><p>8次运行的名义末端平衡均通过；6次有雨运行在3600秒雨刚开始的过程检查未通过，储水已经增加约468 m³（60 mm/h）或780 m³（100 mm/h）。此时按名义时刻积分的累计雨量为0。</p><p>固定源码先推进时间、更新状态，再用计划输出时刻tout写当前状态，并未保存实际积分时刻。这支持时间标注不同步的机制，但不证明全部差额的精确归因。未平移时钟、删掉起雨帧或放宽门槛，未把该差额直接解释为凭空造水。</p><img src="nominal_budget_clock_diagnostic.png" alt="名义时刻水量残差与过程门槛"><div class="scroll"><table><tr><th>Existing run</th><th>Endpoint gate</th><th>Nominal prefix</th><th>Actual clock</th><th>Endpoint residual / gross scale</th><th>Strict admission</th></tr>__RUN_TABLE__</table></div><p>比例使用逐边界面分正负后的总交换量，避免把同时进出相消；因此与旧表按汇总净流量分正负的分母不同。接近零输入时的比例不适合当精度指标，起雨问题按绝对m³和门槛展示。</p></section>
<section><h2>隔离区与区间标签</h2><img src="quarantine_overview.png" alt="四个已有控制组直接数值隔离区全域概览"><table><tr><th>Control</th><th>Quarantine / km²</th><th>Share of eligible ground</th><th>Interval / km²</th></tr>__CELL_TABLE__</table><p>开发控制组隔离并集0.07694875 km²，掩膜未改变。GeoTIFF逐格标记：0范围外、1单元级精确、2区间、3隔离、4宽区间；单元级分类不代表整次事件已获准。直接隔离不是误差传播范围的证明。</p><ul><li>隔离：无效／缺失上下界、抽样原生记账体积低于−0.001m³、达到假设建筑障碍顶以下0.01m的单元。保留原值和原因。</li><li>精确：上下界差≤10⁻⁷m；区间：宽度≤0.01m加10⁻⁶m存储容差。宽区间保留但不进入主回归。不补零，不取中点。</li><li>跨区、PLATEAU补缺和内部连通疑义是信息标志，不一律排除。掩膜只留在监督侧，不输入预测模型。</li><li>按全域、三区合计、各区和跨区汇水区报告面积及“确定湿区中被隔离的比例”。自适应结果不替代10m参考，完整共同单元障碍接触掩膜尚未打包。</li></ul></section>
<section><h2>水量平衡合同</h2><p>残差 = 储水变化 + 边界出水 − 边界入水 − 雨量 − 其他源 + 其他汇。使用全水动力域的有符号记账储水；不得裁掉负值或排除隔离区来制造闭合。屋面雨量已计入有效输入，不再次相加。</p><p>每个边界面先分正负再汇总。尺度为“雨量＋其他源＋总入水、总出水＋其他汇、初始储水绝对值、1m³”的最大值。末端门槛为max(1m³,0.1%×尺度)，每个状态前缀为max(1m³,1%×尺度)。降雨文件体积误差≤10⁻⁶。</p><p>瞬时边界流量按实际写出时刻梯形积分，保留右矩形对照；最终差异须落在末端门槛内，二者一致仍不能约束未写出通量。未知通量不填零。累计降雨报告的float32误差单独核对。无雨组也必须计入水域来水。没有跨区内部面通量时不宣称各区水量闭合。</p></section>
<section><h2>时序输出合同</h2><table><tr><th>用途</th><th>必须输出／条件</th><th>当前状态</th></tr><tr><td>峰值代理标签</td><td>4小时原生在线峰值；≤300s完整原生状态；≤0.401s边界交换；原生网格、单位、浅水截断与源哈希；实际积分时钟或同检查点步积分账本</td><td>峰值和状态已有；实际时钟仍缺，不自动补跑</td></tr><tr><td>到达时间、积水历时、动态路径</td><td>以后从≤60s全域同步场和30s开发对照开始，再按具体目标核验；频率本身不是精度证明</td><td>未合格，不把0.4s单站历史当全域输出</td></tr><tr><td>动态守恒与主动调控</td><td>原生每个面的区间积分通量、雨量／源汇增量、对应时刻储水和连接关系；必要时重启状态</td><td>内部面通量不全；继续留待后续阶段</td></tr></table><p>负体积仅每300s抽样，不能证明每一步均非负。四小时是本轮目标窗口，不是完全退水事件。缺项或失败时整事件hold，不通过改掩膜救回。规则应用于未来58个待运行变体的合同绑定，但不授予执行权限。</p></section>
<section><h2>交付与下一步</h2><p>__CHECK_COUNT__项规则／产品检查通过，包括“末端通过但过程失败”“无雨倒灌”“裁负体积破坏收支”和“缺少时钟则整事件暂缓”等故障样例。没有重写旧标签、原始数据或冻结的强迫库。</p><p>下一步优先实现并核验求解器实际时钟／匹配检查点预算的导出接口，再另行授权参考运行。此处未改求解器、未执行任何模拟。不要据此宣称已有参考库可训练或实测洪水精度已获验证。</p><p><a href="README.md">复现与合同说明</a> · <a href="validation.json">检查明细</a> · <a href="run_admission.csv">整事件准入</a> · <a href="label_admission_by_scope.csv">分区单元覆盖</a> · <a href="wet_quarantine_coverage.csv">湿区隔离比例</a> · <a href="run_diagnostics.json">完整诊断</a> · <a href="remaining_prerequisites.csv">剩余事项</a></p></section></html>'''
 text=text.replace('__RUN_TABLE__',table).replace('__CELL_TABLE__',cells).replace('__CHECK_COUNT__',str(len(v['checks'])))
 (report/'review.html').write_text(text)
 prerequisites=[{'priority':'P0','item':'Actual integration clock or matched-step budget','status':'missing in existing files; all strict event admissions held','next':'Qualify a file/export adapter with state times and boundary/source integration intervals; no inferred time shift or threshold relaxation.'},
 {'priority':'P0','item':'Process closure','status':'six rainy nominal-prefix checks fail; eight endpoints pass','next':'Recheck at verified actual timestamps or exact step-integrated checkpoints before formal reference admission; preserve present failures.'},
 {'priority':'P1','item':'Cell masks and interval rules','status':'implemented; unchanged existing masks','next':'Apply same version to every future parent; report global and wet support; no training release.'},
 {'priority':'P1','item':'Adaptive paired-clean support','status':'diagnostic only; barrier contact not packaged at common-cell resolution','next':'Complete mask if clean matched-grid regression claims are later required; do not mix as primary reference.'},
 {'priority':'P1','item':'Every-step negativity','status':'not proved;300s snapshots only','next':'Prefer online minimum/signed deficit counters for stronger numerical certification; never infer all-step cleanliness from snapshots.'},
 {'priority':'P1','item':'Timing and dynamic paths','status':'not qualified','next':'Future dedicated cadence/flux qualification only when that target is authorized; no blanket peak-label rerun.'},
 {'priority':'P2','item':'Physical event accuracy and measures','status':'unchanged gaps','next':'Upstream coverage, epoch, riverbed and observed events remain needed for physical claims; active measures remain milestone5.'}]
 from audit_three_ward_reference_contract import csvwrite
 csvwrite(report/'remaining_prerequisites.csv',prerequisites)
 readme="""# 三区参考标签数值准入与水动力输出合同

政策已实现并冻结；严格整事件准入仍暂缓。已只读核查8次已有运行，无新模拟、无训练、无封存响应读取。

## 结果

- 8次名义末端预算通过新门槛；6次有雨运行在起雨处的名义过程检查失败。
- 固定源码证明先推进计算状态再用计划输出时刻写出，没有实际积分时钟；这支持时钟不同步机制，但不是完整差额归因。没有平移时间、删掉起雨检查点或放宽门槛。
- 严格检查需要实际状态与通量时钟或经核验的同检查点步积分账本。8次运行因此全为hold_event，旧峰值与掩膜仅供显式诊断读取。
- 4个常规控制组的标签上下界、精确／区间／隔离／宽区间掩膜保持原样，隔离并集0.07694875km²。直接隔离不界定错误传播范围。
- 分类GeoTIFF：0范围外、1单元级精确、2区间、3隔离、4宽区间。先通过整事件门，单元分类才可能用于正式参考；本版没有训练释放。
- 每组49个全域检查点，分区覆盖与湿区隔离比例，以及全部边界段的7组编码绑定均落盘。同名古川组未合并。

## 规则

详见处理目录中的label_policy.json与output_contract.json。数值设置为版本化工程筛查门槛，不是公认物理精度标准或用户提供的具体容差。

- 精确宽度≤1e-7m；区间宽度≤0.01m加1e-6m存储容差。保留上下界，宽区间不进入主回归。未知不补零、区间不取中点。
- 抽样原生记账体积<−0.001m³、假设建筑障碍顶以下0.01m接触和无效界限直接隔离。体积是记账状态，不等于负物理水深。只检查300s快照不能证明每一步都非负。
- 全域水量残差使用有符号储水、逐面总入／出水及实际雨量；不得剔除隔离区、裁负值、遗漏未知通量或重复加屋面源。
- 末端max(1m³,0.001×体积尺度)，每个状态前缀max(1m³,0.01×尺度)。尺度纳入无雨倒灌，先按每个边界面分正负再汇总。对瞬时通量使用梯形积分并保留右矩形对照；最终差异也不得超过末端门槛。
- 峰值任务保留原生在线峰值、≤300s全域状态和≤0.401s完整边界历史。实际积分时钟／步积分检查点仍是严格过程准入必要条件。当前不触发补跑。
- 动态时长／路径需另外核验≤60s场和30s开发对照；动态守恒与主动控制还需原生每面区间通量、源汇增量和同步储水。时序频率不自动证明合格。

## 本地复现与文件

使用已有numpy、netCDF4、rasterio、scipy和matplotlib运行环境，不安装依赖。项目根目录：

```sh
python src/analyses/validate_three_ward_reference_contract.py
python src/analyses/review_three_ward_reference_contract.py
```

审计入口为src/analyses/audit_three_ward_reference_contract.py。当前交付目录已冻结，入口拒绝覆盖；修订政策必须另立版本，不得覆盖原记录。规则核心为src/preprocessing/reference_label_policy.py；默认关闭的读取接口为src/preprocessing/load_three_ward_reference_labels.py，diagnostic_only=True仅开放已核查控制组的诊断数组。不存在任何求解器执行接口。

处理目录data/processed/three_ward_reference_contract_v1保存JSON合同、4套NPZ／GeoTIFF分类、开发隔离并集、边界段编码、参考准入登记、58个未运行变体的合同绑定及冻结哈希。旧强迫库未修改。

报告目录保存逐运行49个水量检查点、事件准入、分区覆盖、湿区隔离比例、源码与输入输出哈希、故障测试和审查页。原始数据和旧标签未改动。

AnaSOP第4、6、7节通过data-preprocessing与estimation-framework-planning职责同步；其他节保持原样，候选变量仍为no。Framelog: MILI-D-20261015-006。
"""
 (report/'README.md').write_text(readme)
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='freeze.json'}
 code={str(p.relative_to(root)):sha(p) for p in [root/'src/preprocessing/reference_label_policy.py',root/'src/preprocessing/load_three_ward_reference_labels.py',root/'src/analyses/audit_three_ward_reference_contract.py',root/'src/analyses/validate_three_ward_reference_contract.py',Path(__file__)]}
 freeze={'status':'policies_and_diagnostic_masks_frozen_strict_reference_release_on_hold','files_sha256':files,'code_sha256':code,'validation_sha256':sha(report/'validation.json'),'training_release':False,'new_simulations':0}
 if (out/'freeze.json').exists():assert json.loads((out/'freeze.json').read_text())==freeze
 else:dump(out/'freeze.json',freeze)
 dump(report/'product_manifest.json',{'processed_products':files,'freeze_sha256':sha(out/'freeze.json'),'reference_release':False,'new_runs':0,'training':0})
 print(json.dumps({'review':str(report/'review.html'),'tests':len(v['checks']),'strict_event_holds':len(summary['event_holds']),'ana_sop_sections_changed':changed},ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();prepare(a.root) if a.prepare_only else deliver(a.root)
