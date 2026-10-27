"""Interface-QC report and skill-routed AnaSOP Sections 4, 6 and 7 only."""
from pathlib import Path
import json,csv,sys,os,re,html,hashlib,argparse
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_second_clock_budget import OUT,REPORT,sha,dump

def update_sop(root):
 path=root/'docs/AnaSOP.md';report=root/REPORT;before=path.read_text();backup=report/'previous-AnaSOP.md'
 if not backup.exists():backup.write_text(before)
 parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);old=list(parts)
 for i,s in enumerate(parts):
  if s.startswith('## 4. '):
   marker='An undeployed integer-second conversion candidate' if 'An undeployed integer-second conversion candidate' in s else 'A versioned, scoped rainfall-reader correction'
   a=s.index(marker);b=s.index('The minimum future numerical export',a)
   s=s[:a]+'''A versioned, scoped rainfall-reader correction now preserves the integer-second knots used by the current forcing library. It reads time values at double precision, validates the UTC reference and rejects unsupported fractional seconds, duplicate or reversed times, nonfinite values and downstream precision loss. Compiled tests exercised the patched precipitation-reader body against actual NetCDF input for all 50 existing new forcings and eight preselected adaptive companions. Decoded time errors and the corresponding piecewise-linear rainfall-integral errors are zero. The existing spatial rain values, valid zeros, coordinate convention, roof allocation, parent families and partitions are preserved. These are forcing-interface checks on small spatial fixtures carrying the original temporal inputs; they are not new whole-domain hydraulic results. The complete operational solver image has not been rebuilt or qualified.

'''+s[b:]
   s=s.replace('The export schema and fail-closed arithmetic validator are implemented, but no running solver exports these new fields yet.', 'Additive diagnostic hooks are now implemented in the versioned source: a true initial state, every-step clock and signed-storage ledger, applied net rainfall, gross native boundary exchange, observed meteo brackets, online negative-volume summaries and an ever-negative mask. Output records and final peak windows are bound to the accounted state. Source-addition rounding and internal-flux residuals remain visible. The current accounting profile is restricted to the existing projected CPU subgrid baseline; unsupported source, sink, storage and wave processes fail closed. Seven-group aggregation remains an optional extension. Compiled bookkeeping fixtures test these exports without advancing hydraulic states; operational validation remains pending.')
   s=s.replace('Exact checkpoint instrumentation and operational source-term accumulation remain untested.', 'Source integration and isolated bookkeeping are tested; full-solver operational clock and volume qualification remains outstanding.')
   parts[i]=s
  elif s.startswith('## 6. '):
   marker='### Connection to active measures'
   add='''A second-preserving transport and matched-step diagnostic source version has passed compiled interface tests on the existing forcing inputs and isolated bookkeeping fixtures. This verifies the input and accounting components only. Complete operational build and dependency qualification, then separately authorized hydraulic end-to-end checks, are still required before constructing admissible responses. Preserve the original forcing curves, parent-family partitions and numerical policy. Reference uncertainty and Scenario coverage and split audit must distinguish component verification from image qualification, response availability and reference release.

'''
   if add not in s:
    assert marker in s;s=s.replace(marker,add+marker)
   parts[i]=s
  elif s.startswith('## 7. '):
   s=s.replace('Scientific curves and family partitions: constructed; legacy transport: held; response library: absent', 'Scientific curves and partitions: constructed; corrected transport and budget components: tested; operational image: held; response library: absent')
   s=s.replace('Keep frozen curves and family inheritance; qualify a versioned second-preserving reader and same-step diagnostic export before separately authorized reference runs; no sealed-response inspection during setup.', 'Keep frozen curves and family inheritance; qualify the complete operational transport and diagnostic build before separately authorized reference runs; component tests do not release sealed responses or reference labels.')
   s=s.replace('The verified minute-truncation defect explains the onset discrepancy conditionally and holds all pending legacy rainfall exports; actual-clock and applied-step ledger qualification still hold the inspected controls from strict reference admission.', 'The verified minute-truncation defect explains the onset discrepancy conditionally. A scoped correction now preserves all 50 existing forcing curves in compiled reader tests, and actual-clock and applied-step diagnostic hooks pass isolated bookkeeping tests. All 58 descendants remain unrun and held pending complete operational qualification; inspected old controls retain strict reference holds. No previous hydraulic result has been regenerated or relabeled.')
   parts[i]=s
 after=''.join(parts);path.write_text(after)
 base=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M)
 changed=[int(re.match(r'## (\d)',a).group(1))for a,b in zip(base,parts)if a!=b and a.startswith('## ')]
 assert set(changed)=={4,6,7},changed
 assert all(parts[i]==base[i]for i in range(len(parts))if not parts[i].startswith(('## 4. ','## 6. ','## 7. ')))
 dump(report/'anasop-update.json',{'sections_changed':changed,'skills':{'4':'data-preprocessing','6':'estimation-framework-planning','7':'estimation-framework-planning'},'human_record':'MILI-D-20261015-008','candidate_variable_status_unchanged':True,'english_readable_names_only':True,'prior_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'current_sha256':sha(path),'other_sections_unchanged':True,'new_final_variables':0,'new_formulas':0,'section8_outputs_still_planned':True,'section3_observational_and_coverage_gaps_unchanged':True})

def report(root):
 folder=root/REPORT;summary=json.loads((folder/'summary.json').read_text());validation=json.loads((folder/'validation.json').read_text())
 assert summary['all_checks_passed']and validation['all_checks_passed']
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sys.path.insert(0,str(root/'src/preprocessing'))
 from sfincs_clock_budget_interface import decode_legacy_fews_minutes,scheduler_curve
 cid='rain100_raised_current_geometry'
 with np.load(root/'data/processed/three_ward_minimum_scenarios_v1'/f'{cid}.npz',allow_pickle=False)as z:t=z['Event Time'];rain=z['Physical Rainfall Rate']
 q=np.linspace(3598.8,3601.8,1000);legacy_t,legacy_r=scheduler_curve(decode_legacy_fews_minutes(t/60),rain)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(8.7,3.9),layout='constrained')
 ax.plot(q,np.interp(q,legacy_t,legacy_r),ls='--',lw=2,color='#b36239',label='Legacy decoded / meteo curve')
 ax.plot(q,np.interp(q,t,rain),lw=2.4,color='#216d83',label='Original curve = corrected decoded curve')
 ax.axvline(3600,color='#80909a',lw=.8);ax.set(xlabel='Event time (s)',ylabel='Rainfall rate (mm/h)',title='Integer-second control ramp retained by the reader correction')
 ax.legend(loc='upper left',fontsize=8.5);fig.savefig(folder/'rain_reader_transport.png',dpi=160);plt.close(fig)
 rows=list(csv.DictReader((folder/'50_case_transport_receipt.csv').open()));splits={k:sum(x['split']==k for x in rows)for k in sorted({x['split']for x in rows})}
 ledger=list(csv.DictReader((folder/'fixture_normal/udt_step_budget.csv').open()));residual=max(abs(float(x['budget_residual']))for x in ledger)
 readme=f'''# 秒级降雨读取与同一步诊断：接口核验\n\n当前 root: `{root}`。版本化实现与接口检查已完成；完整水动力镜像尚未重建或端到端验证。没有新模拟、训练或测试响应读取。\n\n## 检查结果\n\n- {summary['checks']} 项检查通过；50 份原有情景时序逐份经过实际补丁雨量读取子程序。最大节点偏差 0 秒，时间转换引起的分段线性累计雨量偏差 0 mm。\n- 8 个原有加密网格副本也通过时间、5 m 坐标和谱系检查。50 份输入按原划分统计：{splits}。48 个合成父事件与 2 个新控制组的计数含义保持原样。\n- 原始输入曲线、父事件、family、划分及旧标签/数值政策的 SHA 校验不变；58 个待运行副本依旧未运行、不放行。\n- 屋面仅按原有规则分配一次；两种网格的地面与屋面贡献之和保持一致。NetCDF 浮点雨量保留既有 real32 量化，0 mm 指时间读取引入的误差，不能解释为物理精度或消除了浮点量化。\n- 账本隔离测试记录真实测试调用时钟、步长和步号；同时流入与流出分别累计；10 m/5 m 原生边界宽度不混用；负体积不裁零。正常小型账本 fixture 最大残差 {residual:.9g} m³，这不是东京模拟的残差。\n\n## 实现范围\n\n降雨补丁局限于 NetCDF precipitation 读入：先直接读入 real64 分钟，再保留整数秒；验证参考日期、UTC 和严格递增。下游 real32 仅在整数秒可精确表示的范围内使用。亚秒、非分钟单位、非 UTC、超范围、非有限值和重复/倒序时刻均拒收。不改冻结 exporter、不改水位曲线、不补写旧结果。\n\n新源码包含实际连续方程前后的诊断 hook、每步累计账本、原始名义输出时间与实际状态的记录号对接、实际峰值窗口、最终真实时钟、在线最低体积与 ever-negative 掩膜。实际气象插值 bracket 逐步写入；初始化行明确标为未观察气象 bracket。地面体积和边界项保留符号，源项加法舍入与内部通量余差另列，不调预算容差。\n\n当前支持 projected CPU / subgrid 基线，原有 regular10m 和 adaptive10/5m 均采用该 subgrid 路径。新入渗、外部源、点流量、额外储量、spinup 和波浪/GPU 等未核验配置拒收。七组河段聚合仍可后续增加，不影响当前全域总账；动态内部流量立方体尚未添加。\n\n## 测试边界\n\n真实 patched rain reader 的完整子程序从补丁源码逐字提取、编译，并通过真实 NetCDF C 库读取 50 个小空间 fixture，每个 fixture 携带对应情景的完整原时序。这个测试保留了雨量读取、reshape、翻转与时刻处理逻辑；它不等同于重新输出 50 份全域降雨文件。系统没有安装生产 Fortran NetCDF 模块，因此使用明确标为 test-only 的签名桥接器。实际连续方程、主循环和最终输出 hook 已编译；连续方程从未执行，主循环和输出模块使用依赖 shell 做编译检查，未完整链接 SFINCS 镜像。账本 fixture 只做预设项的累计，不解水动力方程。\n\n## 交付与复现\n\n- 源码补丁、固定 base、patched 源码及 SHA 清单：`{OUT}`。\n- 50 份时间读取表：`50_case_transport_receipt.csv`；8 个加密副本：`adaptive_transport_receipt.csv`。\n- `validation.json`：检查明细；`compile_receipt.json`：编译与执行边界；`provenance.json`：保全记录。\n- `pending_run_bindings.json`：58 个原副本与补丁绑定，未产生执行许可。身份 staging helper 要求另一个镜像 digest，并拒绝把旧镜像登记为新实现。\n- AnaSOP 仅第 4、6、7 节按对应 skill 更新；变量仍为候选，参考准入保持 hold。Framelog: MILI-D-20261015-008。\n\n在未冻结的新版本中，可从项目 root 运行本地 Python 脚本 `src/preprocessing/prepare_sfincs_second_clock_budget.py` 和 `src/analyses/test_three_ward_second_clock_budget.py` 重建补丁与接口测试；先准备 NumPy、netCDF4、gfortran 和 NetCDF C 开发库。脚本不会调用求解器；冻结版本拒绝覆盖。交付完整性复核使用 `src/analyses/validate_three_ward_second_clock_budget_delivery.py`，只读不改冻结产物。\n\n下一道缺口是完整 CPU 求解器/Fortran NetCDF 镜像的构建和链接核验；再经单独授权进行水动力端到端检查。当前没有可据此新增的洪水结果，也没有训练准入。\n'''
 (folder/'README.md').write_text(readme)
 esc=html.escape;table=''.join('<tr><td>'+esc(x['case_id'])+'</td><td>'+esc(x['split'])+'</td><td>'+x['knots']+'</td><td>'+x['time_error_s']+'</td><td>'+x['time_integral_error_mm']+'</td></tr>'for x in rows)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>秒级降雨与同一步诊断</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#eff3f5;color:#1e3442;margin:0}main{max-width:1060px;margin:auto;padding:28px}section{background:white;border:1px solid #dbe3e8;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:28px;line-height:1.4}h2{font-size:21px}p{margin:10px 0}.tag{color:#0b6473}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{padding:16px;background:#eaf4f5;border-radius:9px}.card strong{font-size:26px;display:block}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e0e7eb;padding:9px}a{color:#126882}.scroll{overflow:auto}code{overflow-wrap:anywhere}.limit{border-left:4px solid #bf8038;padding-left:14px}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:12px}section{padding:17px}}</style><main><p class="tag">Mili · 三个区整体模型 · 接口核验</p><h1>秒级降雨读取已修正，实际时钟与同一步预算已接入源码</h1><p>原有曲线、父事件谱系和划分全部保留。没有新水动力模拟、机器学习训练或封存响应读取。</p><section><div class="cards"><div class="card"><strong>50 / 50</strong>原情景时序读取通过</div><div class="card"><strong>0 秒 / 0 mm</strong>节点与时间转换积分偏差</div><div class="card"><strong>__N__ 项</strong>接口与保全检查通过</div></div><p>另有 8 个预选加密副本完成接口核验。屋面保持原分配规则；地面与屋面贡献没有重复计入。</p></section><section><h2>改变的是读取时钟，原曲线不改</h2><p>只修正降雨读入链路：直接以双精度读取分钟，再严格转换为整数秒。非有限、重复、倒序、非 UTC、亚秒或下游精度超限时刻拒收。原有空间雨量、有效零、坐标、翻转与水位输入保留。</p><img src="rain_reader_transport.png" alt="原控制组雨型、旧读入曲线与修正后曲线"><p>图为输入接口的雨强曲线比较，不是新积水模拟。0 mm 指读取时刻造成的积分偏差，不消除原有浮点雨量量化。</p></section><section><h2>预算与状态来自同一步</h2><p>每步输出真实时刻、步长、步号、有符号储量、实际净降雨累计量、分别累计的原生边界进水与出水。源项加法舍入和内部通量余差另列，保留负体积与在线 ever-negative 掩膜。输出记录号与实际状态对接，包含初始化、峰值窗口和最终真实时刻。</p><p>隔离账本覆盖双向边界交换、10 m／5 m 边界宽度、干控制组、负体积与遗漏单元拒收。当前 CPU 地表基线之外的源汇、额外储量、spinup、波浪/GPU 等未核验配置拒收。</p></section><section><h2>核验到哪里</h2><div class="scroll"><table><tr><th>层次</th><th>状态</th></tr><tr><td>实际 patched 降雨读取子程序</td><td>已编译；真实 NetCDF C I/O 桥接，小空间 fixture 携带原时序</td></tr><tr><td>连续方程、主循环、最终输出接线</td><td>源码编译通过；主循环和输出依赖以 shell 做编译检查</td></tr><tr><td>同一步预算组件</td><td>隔离预设账本通过；不推进水动力状态</td></tr><tr><td>完整 CPU／Fortran NetCDF 镜像</td><td>尚未完整构建、链接或端到端核验</td></tr><tr><td>参考标签与训练</td><td>原准入保持；58 个副本未运行、不放行</td></tr></table></div><p class="limit">组件测试通过不等于正式模型验证。下一步先补完整生产构建和链接证据；水动力端到端运行仍需单独授权。七组河段累计可后续增加，当前为全域总账。</p></section><section><h2>50 份输入逐份核验</h2><div class="scroll"><table><tr><th>Case</th><th>原划分</th><th>节点数</th><th>时间偏差 s</th><th>积分偏差 mm</th></tr>__ROWS__</table></div></section><section><h2>可复核交付</h2><p><a href="README.md">复现说明与限制</a> · <a href="validation.json">检查明细</a> · <a href="compile_receipt.json">编译证据</a> · <a href="50_case_transport_receipt.csv">50 份读取记录</a> · <a href="adaptive_transport_receipt.csv">8 个加密副本</a> · <a href="../../../processed/three_ward_second_clock_budget_v1/sfincs_second_clock_budget.patch">源码补丁</a> · <a href="../../../processed/three_ward_second_clock_budget_v1/BUILD.md">接入说明</a></p><p>AnaSOP 第 4、6、7 节已同步；变量候选状态不变。人类决策记录 MILI-D-20261015-008。</p></section></main></html>'''
 (folder/'review.html').write_text(page.replace('__N__',str(summary['checks'])).replace('__ROWS__',table))
 # Versioned output schema is a description of implemented sidecars, not an admission override.
 dump(root/OUT/'diagnostic_export_schema.json',{'version':'second-clock-budget-interface-v1','status':'source_hooks_compiled_components_tested_production_image_unqualified','accounting_profile':'projected CPU subgrid; existing no-action source/sink baseline only','step_ledger_fields':list(ledger[0]),'output_map_fields':list(csv.DictReader((folder/'fixture_normal/udt_output_clock.csv').open()).fieldnames),'initial_row':'actual pre-step state, no observed meteo bracket','negative_mask_threshold_m3':-.001,'cumulative_other_source_sink':'explicit zero only in guarded baseline','reduction':'float64; native source term remains real32; actual post-addition increment exported','boundary':'gross split per native face before sum, real32 physical q*width*dt, cancellation residual retained','run_identity':'schema, run, parent, family, split, forcing semantic hash, geometry file hash, base commit, patch hash, rebuilt image digest','reference_admission_release':False,'new_simulations':0,'training':0})

def freeze(root):
 out=root/OUT;folder=root/REPORT
 assert json.loads((folder/'validation.json').read_text())['all_checks_passed']
 files=[p for base in [out,folder]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
 files += [root/'src/preprocessing/prepare_sfincs_second_clock_budget.py',root/'src/preprocessing/stage_second_clock_budget_identity.py',root/'src/analyses/test_three_ward_second_clock_budget.py',root/'src/analyses/review_three_ward_second_clock_budget.py',root/'src/analyses/validate_three_ward_second_clock_budget_delivery.py']
 files+=list((root/'src/preprocessing/sfincs_second_clock_budget_v1').glob('*.f90'))+list((root/'src/analyses/second_clock_budget_tests').glob('*'))
 dump(out/'freeze.json',{'version':'second-clock-budget-interface-v1','state':'components_verified_operational_execution_held','files_sha256':{str(p.relative_to(root)):sha(p)for p in sorted(files)if p.is_file()},'new_hydrodynamic_runs':0,'training':0,'no_reference_release':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--update-sop',action='store_true');p.add_argument('--freeze',action='store_true');a=p.parse_args()
 if(ROOT/OUT/'freeze.json').exists():raise FileExistsError('Frozen report; use delivery validator')
 if a.update_sop:update_sop(ROOT)
 report(ROOT)
 if a.freeze:freeze(ROOT)
