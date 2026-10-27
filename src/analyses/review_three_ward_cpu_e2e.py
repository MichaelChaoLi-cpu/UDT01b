"""Reproducible control qualification report and skill-routed workflow updates."""
from pathlib import Path
import sys,json,re,os,csv,argparse,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import sha,dump

def update_sop(root):
 p=root/'docs/AnaSOP.md';folder=root/REPORT;backup=folder/'previous-AnaSOP.md'
 if not backup.exists():backup.write_bytes(p.read_bytes())
 before=p.read_text();parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M)
 # Workflow authorization metadata follows the explicit user decision; research questions are unchanged.
 parts[0]=parts[0].replace('Status: planning only; no training, new hydraulic simulations, acquisition or literature searches are authorized.', 'Status: surrogate methods remain planned; two existing open-control hydraulic replays are separately authorized for numerical qualification. No training, batch scenario execution, acquisition or literature searches are authorized.')
 summary=json.loads((folder/'summary.json').read_text());good=summary['all_checks_passed']
 status='pass the existing endpoint and prefix budget gates'if good else'retain unresolved checks under the existing gates'
 add4=f'''Two existing open controls, one without rain and one with prescribed rain under the same raised receiving-water trajectory, have now been replayed on the complete regular 10 m domain. They remain descendants of the existing open-control family and add no independent event or sealed response. Their decoded rain knots, actual step clocks, applied net-rainfall and gross boundary ledgers, and map/history/peak record correspondence are audited together. They {status}. Continuous input-curve volume and actually applied discrete rainfall are reported separately; native float32 nominal-time and map-volume write precision is accounted for separately from the float64 step budget. Online negative-volume flags and actual peak-window endpoints are retained. No automatic label or training release follows from this control qualification.

The first end-to-end attempts exposed two input-interface gaps: the legacy date-only rainfall reference needed explicit midnight compatibility, and the intended linear-rainfall option was not recognized under its old spelling. The corrected, versioned transport preserves the original rainfall and water-level files, parent family and split; only the staged runtime configuration activates the recognized interpolation option. Rejected attempts remain part of the audit. The astronomical-tide table warning is unreachable for the current no-tidal-file configuration; astronomical-tide execution itself remains unqualified. The prospective 58-case library is still unrun.

'''
 add6='''Use the targeted open-control replays to assess numerical transport and accounting only. Passing actual-clock and same-step budget checks is distinct from hydraulic grid convergence, observational flood accuracy, complete dynamic path labels, or reference-label admission. Keep source and sink accounting on the captured actual step, preserve signed storage and explicit negative-volume flags, and retain the existing budget gates. Input-curve and applied-rainfall differences arise from the declared time integration and must not be silently removed by shifting clocks. Select a native peak record by its actual accumulation interval, and report any empty final window. Freeze the compatible solver version and recognized runtime options before separately authorized scenario-library execution; do not infer a batch or training authorization from the two control replays.

'''
 for i,s in enumerate(parts):
  if s.startswith('## 4. '):
   s=s.replace('No solver initialization, hydraulic update, new event response or training has occurred; hydraulic end-to-end qualification and reference release remain pending.', 'That build and isolated-interface check did not initialize or advance the hydraulic solver. The subsequent targeted control replays are reported below; new event-library responses, training and reference release remain pending.')
   s=s.replace('hydraulic end-to-end qualification remains pending.', 'targeted regular-grid open-control end-to-end evidence is now reported below; broader reference qualification remains pending.')
   if 'Two existing open controls, one without rain'not in s:s=s.rstrip()+'\n\n'+add4
   parts[i]=s
  elif s.startswith('## 6. '):
   s=s.replace('Keep this build-level qualification distinct from numerical event validation: no hydraulic update or new event response has been generated.', 'That build-level qualification generated no hydraulic update or event response. The subsequent two existing-control replays provide targeted numerical event evidence as described below; no prospective library response has been generated.')
   s=s.replace('Separately authorized end-to-end hydraulic checks must still establish actual-grid clock, source, boundary and storage diagnostics before admitting labels.', 'The separately authorized control checks now establish targeted actual-grid clock, source, boundary and storage diagnostics; broader scenario qualification and label admission remain pending.')
   marker='### Connection to active measures'
   if add6 not in s:s=s.replace(marker,add6+marker)
   parts[i]=s
  elif s.startswith('## 7. '):
   s=s.replace('This update does not execute training or additional hydraulic cases.', 'The current update reports two separately authorized existing-control replays; training and the prospective batch remain unexecuted.')
   s=s.replace('hydraulic end-to-end: held; response library: absent','two regular-grid open-control end-to-end replays: audited; response library: absent')
   s=s.replace('All 58 descendants remain unrun and held pending hydraulic end-to-end qualification;', 'Two existing regular-grid open controls now provide targeted actual-clock and same-step budget evidence. All 58 prospective descendants remain unrun and held pending scenario-level qualification;')
   marker='The first four milestones establish input meaning, reference labels, scenario diversity and surrogate reliability.'
   if 'Current authorization is limited to the two open-control replays'not in s:
    s=s.replace(marker,marker+' Current authorization is limited to the two open-control replays and their interface repairs; it does not extend to training or the prospective batch.')
   parts[i]=s
 after=''.join(parts);p.write_text(after)
 original=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M)
 changed=[0 if not a.startswith('## ')else int(re.match(r'## (\d)',a).group(1))for a,b in zip(original,parts)if a!=b]
 assert set(changed)=={0,4,6,7}
 dump(folder/'anasop-update.json',{'sections_changed':[4,6,7],'workflow_authorization_preamble_updated':True,'skills':{'4':'data-preprocessing','6':'estimation-framework-planning','7':'estimation-framework-planning','authorization_metadata':'estimation-framework-planning analytical workflow'},'human_record':'MILI-D-20261015-009','current_sha256':sha(p),'prior_sha256':sha(backup),'other_sections_unchanged':True,'new_final_variables':0,'new_formulas':0,'questions_and_methods_unchanged':True})

def report(root):
 folder=root/REPORT;out=root/OUT;s=json.loads((folder/'summary.json').read_text());results=s['case_summaries']
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
 for cid,color in [('dry_raised','#2b7184'),('rain60_raised','#bc7247')]:
  run=next(r for r in json.loads((out/'run_manifest.json').read_text())['runs']if r['case_id']==cid)
  d=np.genfromtxt(root/run['folder']/'udt_step_budget.csv',delimiter=',',names=True)
  axs[0].plot(d['actual_state_time']/3600,d['budget_residual'],label=cid,color=color,lw=1)
  if cid=='rain60_raised':
   near=np.genfromtxt(folder/(cid+'_onset_actual_steps.csv'),delimiter=',',names=True)
   axs[1].plot(near['actual_time_s'],near['applied_rain_m3'],'o-',label='Applied same-step rain',color=color,ms=3)
   axs[1].plot(near['actual_time_s'],near['continuous_file_rain_m3'],'--',label='Exact input-curve integral',color='#2b7184')
 axs[0].axhline(0,color='#82919b',lw=.5);axs[0].set(xlabel='Actual event time (h)',ylabel='Signed budget residual (m3)',title='Whole-domain applied-step accounting');axs[0].legend(fontsize=8)
 axs[1].axvline(3600,color='#82919b',lw=.6);axs[1].set(xlabel='Actual event time (s)',ylabel='Cumulative rainfall volume (m3)',title='Rainfall onset: input and applied integration');axs[1].legend(fontsize=8)
 fig.savefig(folder/'actual_clock_budget_controls.png',dpi=160);plt.close(fig)
 rows=[]
 for cid,x in results.items():
  rows.append({'case':cid,'actual_steps':x['actual_steps'],'actual_end_s':x['actual_final_time_s'],'rain_applied_m3':x['actual_applied_rain_m3'],'boundary_in_m3':x['boundary_in_m3'],'boundary_out_m3':x['boundary_out_m3'],'end_residual_m3':x['end_budget_residual_m3'],'max_prefix_abs_m3':x['max_abs_step_budget_residual_m3'],'online_negative_cells':x['online_ever_negative_native_cells'],'map_records':x['map_records'],'history_records':x['history_records'],'peak_records':x['peak_records'],'empty_peak_records':x['zero_length_final_peak_records']})
 with(folder/'control_summary.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 status='通过'if s['all_checks_passed']else'存在未通过项'
 ncases=len(results);image=next(iter(results.values()))['image_id'];counts=len(s['checks'])
 rain=results['rain60_raised']
 integration_note=f"降雨组实际累计入雨 {rain['actual_applied_rain_m3']:.6f} m³，原曲线连续积分 {rain['continuous_encoded_input_rain_m3']:.6f} m³，末端相差 {rain['final_applied_minus_continuous_rain_m3']:.6f} m³；过程累计量最大差约 {rain['max_applied_minus_continuous_rain_m3']:.3f} m³。两者分别代表实际时间步施加量和输入曲线积分，不能互相替代。"
 notes=[]
 for cid,x in results.items():
  notes.append(f"- {cid}: {x['actual_steps']} 步；实际结束 {x['actual_final_time_s']:.9f} s；预算末端残差 {x['end_budget_residual_m3']:.9g} m³，最大过程绝对残差 {x['max_abs_step_budget_residual_m3']:.9g} m³；在线曾低于 -0.001 m³ 的原生单元 {x['online_ever_negative_native_cells']} 个；末尾零长度峰值窗口 {x['zero_length_final_peak_records']} 个。")
 text=f'''# 两个全域 10 m 控制组端到端核验\n\n状态：{status}。仅重放已有 dry_raised 与 rain60_raised 两个 open_control，使用相同的抬升水位时序。完整 4 h、原有全域 10 m 网格和道路／建筑约束保留。不是两个新独立父事件；未训练，未批量运行 58 个情景。\n\n## 结果\n\n{counts} 项检查，{status}。\n\n'''+ '\n'.join(notes)+f'''\n\n- 节点读取：原始六个秒级时刻均完整保留，原 NetCDF 与水位文件的 SHA 未变。\n- 时钟：保存真实初始状态和每一步的实际时间、dt、步号，并逐条对接 map、history 和 peak 记录。不能把原有名义时间当作真实积分时间。\n- 预算：保留原有前缀 1%、末端 0.1%、绝对下限 1 m³ 的门槛；使用同一步实际入雨量、分别累计的边界流入与流出、有符号原生储量。没有移动时钟或放宽政策。\n- 原 NetCDF 的名义时间与单元体积为 float32；时间按该 dtype 的精确转换核验，并以记录号对接实际双精度时钟。体积与 float64 账本的求和差按逐单元写出舍入界单独核验，不把地图写出舍入变成预算误差。\n- 曲线精确积分与实际时间步的降雨累计分别保留，后者受时间离散影响。读取节点无偏差不等于每一步雨量等于连续解析积分。\n- 在线负体积／峰值实际窗口均保留，不裁零，也不把空的末尾峰值窗口当成完整事件峰值。\n\n## 两个接口问题及修复记录\n\n{integration_note}\n\n1. v1 在旧控制组的日期短写参考单位上拒收，初始化未完成、无水动力时间步。v2 在读取内部把确切的日期短写解释为午夜，保留原始属性和全部输入数据；生产库 50 个原情景输入加 7 个参考格式／拒收测试全部通过。\n2. 旧配置使用 ampr_block，但官方读取键为 amprblock。v2 首次尝试在第一步诊断检查时拒收。只在新运行副本增加 amprblock=0；源配置、雨量和水位文件不改。其余未识别的 stopdepth 旧字段保留，求解器原生稳定性条件未改。\n\n两次失败目录均在 attempts 下保留，不当作完成结果；第二次进入过一次连续方程，但未形成已核算的完整时间步记录。随后完成的两组才用于本报告。兼容源码／补丁、镜像 metadata 和实际 identity 均在 compatibility_v2 下绑定。当前实际镜像：`{image}`。\n\n## 天文潮表警告\n\n固定源码先将潮汐分量数设为 0，仅在非空潮汐文件、存在边界点且允许读取该文件时才进入天文潮表路径。当前两组均未提供潮汐文件，默认 none；运行日志也没有潮汐分量加载记录。上游数组越界警告所在路径对当前给定水位时序配置不可达。未修改天文潮源码，也未核验启用天文潮文件的情况。\n\n## 证据与范围\n\n- `astronomical_tide_audit.json`：固定源码与当前输入的调用条件。\n- `runtime_configuration_audit.json`：识别键与运行副本修复；原源文件不改。\n- `compatibility_v2_validation.json`：57 项生产接口回归检查。\n- `*_map_checkpoints.csv`、`*_onset_actual_steps.csv`、`*_peak_windows.csv`：实际时钟、预算及输出对接。\n- `control_summary.csv`、`summary.json`、`validation.json`：全域结果和每项判断。\n- 原始输出和步级账本在 `{OUT}/runs`；失败尝试也保留。\n- AnaSOP 第 4、6、7 节及开头授权状态同步；其他研究问题、方法和候选变量状态不改。Framelog MILI-D-20261015-009。\n\n本次提供两个既有控制组的数值接口证据；没有观测洪水验证、完整河道动力学、全域网格收敛或训练标签自动放行。在线负体积仍按已有隔离规则处理。58 个 prospective 副本继续未运行，默认参考 loader 和训练准入不变。后续启动库之前必须把日期兼容版本与 recognized amprblock=0 固定到新运行合同。\n'''
 (folder/'README.md').write_text(text)
 esc=html.escape;tab=''.join('<tr>'+''.join('<td>'+esc(str(x[k]))+'</td>'for k in ['case','actual_steps','rain_applied_m3','end_residual_m3','online_negative_cells','empty_peak_records'])+'</tr>'for x in rows)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两个全域10m控制组端到端核验</title><style>body{font:16px/1.75 system-ui,sans-serif;background:#eff3f5;color:#203747;margin:0}main{max-width:1100px;margin:auto;padding:28px}section{background:white;border:1px solid #dbe3e7;border-radius:12px;padding:23px;margin:18px 0}h1{font-size:28px;line-height:1.4}h2{font-size:21px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dfe7eb}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:#eaf3f4;border-radius:9px;padding:16px}.card strong{font-size:25px;display:block}.scroll{overflow:auto}a{color:#176c80}code{overflow-wrap:anywhere;font-size:13px}.note{border-left:4px solid #b78242;padding-left:14px}@media(max-width:650px){main{padding:12px}.cards{grid-template-columns:1fr}section{padding:17px}}</style><main><p>Mili · 全域10m · 两个既有 open controls</p><h1>真实时钟与同一步水量预算：端到端核验__STATUS__</h1><p>干控制组和60 mm/h降雨控制组采用同一抬升水位；保持原曲线、谱系和划分。未训练，未批量运行58个情景。</p><section><div class="cards"><div class="card"><strong>2 个</strong>既有控制组完成4小时重放</div><div class="card"><strong>__CHECKS__ 项</strong>实际时钟、预算和输出核查</div><div class="card"><strong>57 项</strong>兼容修复后的生产读取回归</div></div></section><section><h2>全域账本与起雨过程</h2><img src="actual_clock_budget_controls.png" alt="两组同一步预算残差与降雨离散积分"><div class="scroll"><table><tr><th>控制组</th><th>实际步数</th><th>实际雨量m³</th><th>末端残差m³</th><th>在线负体积单元</th><th>空峰值窗口</th></tr>__TABLE__</table></div><p>预算用实际施加的每步源项与边界交换，不把输入曲线的解析积分代替离散施加量。原前缀1%、末端0.1%及1 m³下限保持不变。</p><p>__INTEGRATION__</p></section><section><h2>保住秒级时刻，逐条对接输出</h2><p>原控制文件的秒级节点完整保留；真实初始状态、每步时间与步长均已导出。map、history、peak以记录号逐条对应账本状态。地图体积为float32，写出量化与float64账本差异单独核验。实际峰值窗口显式保留，空窗口不作为完整事件峰值。</p></section><section><h2>发现并修复的两个接口缺口</h2><p>旧参考单位只写日期，首轮初始化被拒收；新版本在读取内部兼容午夜参考。旧配置中的ampr_block未被官方参数读取器识别，第二次在第一步诊断检查被拒收；本次运行副本增加正确的amprblock=0。原雨量、水位和源文件不改，两次失败记录保留。</p><p>兼容版本先完成50份原输入及7个格式／拒收回归，再运行这两个控制组。新镜像与补丁、运行身份绑定，旧冻结版本保持原样。</p></section><section><h2>天文潮表警告对当前配置的影响</h2><p>当前没有天文潮文件，默认潮汐分量为零，触发数组越界警告的潮表读取路径不可达；实际采用给定水位时序。日志没有潮汐分量加载记录。启用天文潮文件的情形仍未核验。</p></section><section><h2>结论边界与交付</h2><p class="note">这是两个既有控制组的数值接口核验；在线负体积保留隔离，参考标签和训练不自动放行。全域网格收敛与真实洪水准确性仍需各自的证据。58个未来副本继续未运行。</p><p><a href="README.md">说明与复现范围</a> · <a href="validation.json">检查明细</a> · <a href="control_summary.csv">两组汇总</a> · <a href="runtime_configuration_audit.json">配置键核查</a> · <a href="compatibility_v2_validation.json">兼容回归</a> · <a href="astronomical_tide_audit.json">潮表调用条件</a> · <a href="rain60_raised_onset_actual_steps.csv">起雨逐步账本</a></p></section></main></html>'''
 (folder/'review.html').write_text(page.replace('__STATUS__',status).replace('__CHECKS__',str(counts)).replace('__TABLE__',tab).replace('__INTEGRATION__',esc(integration_note)))
 dump(out/'qualification.json',{'two_open_controls_completed':True,'end_to_end_clock_budget_output_checks_passed':s['all_checks_passed'],'new_parent_events':0,'pending_58_executed':0,'future_library_released':False,'training_released':False,'existing_label_policy_changed':False,'old_controls_relabelled':False,'compatible_solver_image_id':image,'recorded_input_semantics_repairs':['date-only midnight origin','recognized linear interpolation key in staged copies'],'astronomical_tides_in_current_configuration':False,'physical_event_accuracy_verified':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--update-sop',action='store_true');a=p.parse_args()
 if(ROOT/OUT/'freeze.json').exists():raise FileExistsError('Frozen report')
 if a.update_sop:update_sop(ROOT)
 report(ROOT)
