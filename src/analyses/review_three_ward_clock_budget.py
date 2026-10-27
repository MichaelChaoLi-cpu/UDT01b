"""Render reproducible clock-interface evidence and narrow skill-routed AnaSOP updates."""
from pathlib import Path
import os,sys,json,csv,re,argparse,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from sfincs_clock_budget_interface import decode_legacy_fews_minutes,imported_rate,scheduler_curve
from audit_three_ward_clock_budget import OUT,REPORT,LIB,sha,dump


def update_sop(root):
    report=root/REPORT;path=root/'docs/AnaSOP.md';before=path.read_text();backup=report/'previous-AnaSOP.md'
    if not backup.exists():backup.write_text(before)
    parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);original=list(parts)
    for i,s in enumerate(parts):
        if s.startswith('## 4. '):
            old='File-reader-equivalent precipitation and boundary checks cover six representative exports, preserve valid zero rain and reproduce total input volume within the declared rounding gate. These checks do not execute or validate the hydraulic solver.'
            new='Earlier host-file readback checks covered six representative exports and preserved numeric file times, zero rain and volume. They did not exercise the internal solver date conversion. The subsequent compiled-interface audit found loss of fractional minutes, so host roundtrip success does not establish solver forcing equivalence. Existing scientific curves and roof allocations remain unchanged; their legacy transport requires requalification.'
            s=s.replace(old,new)
            old='This supports a clock-mismatch mechanism, not a demonstrated correction or proof of artificial water creation.'
            new='The later isolated official-converter test identified minute-fraction truncation and the following one-second meteo resampling. Conditional source-only rainfall integration at a capped 0.4-second step reproduces the rainy onset residual; including the existing boundary integration-rule difference leaves less than 0.001 cubic metres at that checkpoint. This does not recover the later adaptive step trace, prove full-event closure or alter the original labels.'
            s=s.replace(old,new)
            if '### Verified input-clock interface and minimum diagnostic export' not in s:
                s=s.rstrip()+"""

### Verified input-clock interface and minimum diagnostic export

The pinned official time converter was compiled and tested independently of any hydraulic model. It truncates fractional input minutes before converting to seconds. The existing one-second rain ramps consequently collapse into duplicate imported time knots. Taking the last value at each duplicate and then interpolating the meteo endpoints over one-second update brackets produces a ramp one second earlier than the file-defined curve. This is a verified interface transformation. A source-only forcing integral under a declared pre-onset capped-step assumption reproduces all six rainy whole-domain onset residuals; no fitted time shift or water-flow rerun is used. The explanation remains conditional on that local clock regime and does not establish exact actual clocks for the full event.

The same conversion was exercised for all 50 pending forcing exports. Twenty-five have imported time-knot collisions; 48 have a changed continuous rainfall integral, with maximum relative change approximately 2.884 percent and maximum knot advance 59 seconds. This audits mathematical transport, not new responses or physical flood prediction. The scientific curves, parent families and partitions remain frozen. All 58 pending hydraulic descendants require a qualified transport version before execution. The previous host-file receipt is retained, while its inference of internal reader equivalence is superseded. Renaming units to seconds does not repair a decoder that still multiplies by 60.

An undeployed integer-second conversion candidate recovers all intended integer-second knots in isolated tests. It is not installed in the official engine and does not qualify an operational image. A reader fix or an explicitly redesigned whole-minute forcing library requires a new transport or data version and renewed volume, timing and provenance checks. Do not silently adjust the frozen curves or use a fitted clock correction.

The minimum future numerical export comprises actual and scheduled state times, actual step length and step index; decoded forcing times, units and meteo brackets; same-step cumulative applied rainfall, gross boundary inflow/outflow and explicit other source/sink volumes; signed storage at matching checkpoints; and online negative-volume summaries plus an ever-negative native-cell mask. Record a true pre-step initial state and actual peak-window endpoints. Double-precision global and seven-group summaries at map checkpoints and forcing transitions suffice for the whole-domain peak-reference budget; a full internal-face time cube is reserved for a later dynamic target. The export schema and fail-closed arithmetic validator are implemented, but no running solver exports these new fields yet.

The raw scalar named as average time step in existing NetCDF is written before the accumulated step sum is divided for the log. It is not a step trace or a reliable mean. The public library clock and single-step branch exist in the pinned source, but neither was invoked here. Exact checkpoint instrumentation and operational source-term accumulation remain untested. Existing reference holds, candidate-variable status and training restrictions remain unchanged.

"""
            parts[i]=s
        if s.startswith('## 6. '):
            marker='### Connection to active measures'
            addition="""Before a new hydraulic reference run, qualify the actual rainfall time decoder in the selected solver image as well as the host file format. The current compiled audit demonstrates lost sub-minute knots in the legacy reader, affecting both rainfall timing and prospective event volumes. Mathematical input-curve validation and roof-volume conservation do not cover this internal transport transform. Keep input semantics and applied-step conservation as separate checks.

For minimum global diagnostic capture, bind each storage checkpoint to the live solver clock, actual step index and actual step duration; accumulate the same signed boundary and net-rainfall terms that continuity actually applies, including coarse/fine factors and any enabled source or sink. Preserve cumulative terms before output subsampling. Prefer double-precision reductions and explicitly measured residuals; clipping storage or inventing missing zero terms is prohibited. A source-only onset match or isolated date-conversion candidate does not release the reference library. The planned Reference uncertainty and Scenario coverage and split audit outputs must report the transport hold alongside numerical masks and coverage.

"""
            if 'Before a new hydraulic reference run, qualify the actual rainfall time decoder' not in s:s=s.replace(marker,addition+marker)
            parts[i]=s
        if s.startswith('## 7. '):
            s=s.replace('Forcings and family partitions: constructed and checked; response library: absent','Scientific curves and family partitions: constructed; legacy transport: held; response library: absent')
            s=s.replace('Keep frozen input hashes and family inheritance; separately authorize reference runs after numerical admission and output checks; no sealed-response inspection during setup.',
                'Keep frozen curves and family inheritance; qualify a versioned second-preserving reader and same-step diagnostic export before separately authorized reference runs; no sealed-response inspection during setup.')
            old='Numerical cell-admission and output contracts are implemented; actual-clock and process-budget qualification still hold the inspected controls from strict reference admission.'
            new='Numerical cell-admission and output contracts are implemented. The verified minute-truncation defect explains the onset discrepancy conditionally and holds all pending legacy rainfall exports; actual-clock and applied-step ledger qualification still hold the inspected controls from strict reference admission.'
            s=s.replace(old,new);parts[i]=s
    changed=[parts[i].splitlines()[0] for i in range(len(parts)) if parts[i]!=original[i]]
    assert all(x.startswith(('## 4. ','## 6. ','## 7. ')) for x in changed)
    after=''.join(parts)
    if after!=before:path.write_text(after)
    a=re.split(r'(?=^## [1-8]\. )',backup.read_text(),flags=re.M);b=re.split(r'(?=^## [1-8]\. )',after,flags=re.M)
    cumulative=[y.splitlines()[0] for x,y in zip(a,b) if x!=y]
    dump(report/'anasop-update.json',{'skills':{'4':'data-preprocessing','6':'estimation-framework-planning','7':'estimation-framework-planning'},'sections_changed_from_task_preimage':cumulative,'other_sections_unchanged':True,'preimage_sha256':sha(backup),'after_sha256':sha(path),'language':'English','decision':'MILI-D-20261015-007','candidate_variables_remain_no':True})
    return changed


def main(root=ROOT,freeze=False):
    root=Path(root);out=root/OUT;report=root/REPORT;v=json.loads((report/'validation.json').read_text());summary=json.loads((report/'summary.json').read_text())
    if not v['all_checks_passed']:raise ValueError('Failed interface validation')
    changed=update_sop(root)
    rows=list(csv.DictReader((report/'onset_residual_explanation.csv').open()));future=list(csv.DictReader((report/'prospective_export_impact.csv').open()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([0,3600,3601,7200,7201,14400.]);r=np.array([0,0,60,60,0,0.]);decoded=decode_legacy_fews_minutes(t/60);met,rate=scheduler_curve(decoded,r);q=np.linspace(3598,3602,801)
    fig,ax=plt.subplots(1,2,figsize=(13,4.2),layout='constrained')
    ax[0].plot(q,np.interp(q,t,r),label='File-defined curve',lw=2);ax[0].plot(q,imported_rate(decoded,r,q),label='After integer-minute conversion',ls=':');ax[0].plot(q,np.interp(q,met,rate),label='After meteo resampling',ls='--',lw=2)
    ax[0].set_xlabel('Model time (s)');ax[0].set_ylabel('Physical rainfall (mm/h)');ax[0].set_title('Verified input transform; no flow model');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    colors={'train':'#2c7da0','validation':'#3a8b65','sealed_test':'#75808d','challenge':'#c46b31','open_control':'#b49b52'}
    for split,color in colors.items():
        ix=[i for i,x in enumerate(future) if x['split']==split];y=[100*(float(future[i]['legacy_resampled_depth_mm'])-float(future[i]['intended_depth_mm']))/max(float(future[i]['intended_depth_mm']),1) for i in ix]
        ax[1].scatter(ix,y,label=split,color=color,s=23)
    ax[1].axhline(0,color='gray',lw=.7);ax[1].set_xlabel('Prospective forcing index');ax[1].set_ylabel('Continuous input depth change (%)');ax[1].set_title('Pending exports only; no responses');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    fig.savefig(report/'clock_and_transport_diagnostic.png',dpi=160);plt.close(fig)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [z['run'],f"{float(z['nominal_onset_residual_m3']):.6f}",f"{float(z['conditional_capped_step_rain_m3']):.6f}",f"{float(z['boundary_right_minus_trapezoid_at_onset_m3']):.6f}",f"{float(z['remaining_after_forcing_and_boundary_rule_m3']):.8f}"])+'</tr>' for z in rows)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区时钟与步积分预算接口核查</title><style>body{font:16px/1.65 system-ui;max-width:1120px;margin:auto;padding:30px;color:#203745;background:#f4f7f9}section{background:white;border:1px solid #dbe4eb;border-radius:12px;padding:23px;margin:20px 0}h1{font-size:29px}h2{font-size:22px}.note{border-left:4px solid #c28532;padding:17px;background:#fff2dd}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e0e7ed}img{width:100%}a{color:#1b6c91}code{overflow-wrap:anywhere}</style>
<h1>起雨残差已找到可复现的接口解释</h1><p>都心三区及现有跨区汇水范围 · 只读已有8次运行 · 官方日期函数独立测试 · 新水动力模拟0、训练0</p>
<p class="note"><strong>起雨点解释成立，不等于整场参考释放。</strong> 已核验输入分钟截断、气象插值和端点降雨积分的组合；没有恢复后来变化的真实步长，也没有修改旧标签、原生地形或冻结的情景曲线。8次严格准入限制保留。</p>
<section><h2>根因：文件时刻保住了，求解器读入时刻没有保住</h2><p>固定官方日期函数先对分钟数取整再乘60。独立编译其原始代码（Git blob校验一致）后，输入0、3600、3601、7200、7201、14400秒被读为0、3600、3600、7200、7200、14400秒。两段1秒斜坡因此变为重合节点。</p><p>气象读取选取同一时刻最后一个值，再在每1秒的气象更新端点之间插值；起雨斜坡因此出现在3599—3600秒，而不是输入定义的3600—3601秒。此后以时间步末端雨强累加，更早地带入了降雨体积。</p><p>此前六组文件回读检查只覆盖主机侧时间和坐标，没有经过这段内部转换。这是此前检查的遗漏；旧检查记录保留，本页替代其“内部读取等价”的推断。</p><img src="clock_and_transport_diagnostic.png" alt="实际输入时间转换与待运行情景雨量影响"></section>
<section><h2>已有全域残差能解释到什么程度</h2><p>仅对降雨接口做积分，明示采用起雨前封顶步长0.4秒，不求解任何水流，也不拟合时间平移。在3600秒的检查点，单位峰值雨率累计约0.720043秒，乘以实际峰值体积率，就得到约467.749686m³或779.582810m³。</p><div class="scroll"><table><tr><th>Existing run</th><th>Observed nominal residual / m³</th><th>Conditional rain integral / m³</th><th>Boundary right−trapezoid / m³</th><th>Remaining difference / m³</th></tr>__TABLE__</table></div><p>考虑已有边界通量在起雨点的右矩形／梯形差异后，6个有雨运行的最大剩余差约<strong>3.7×10⁻⁵ m³</strong>。这是源代码和封顶步长条件下的起雨点复现，不能证明全事件数值误差归零。冻结合同仍使用原规则；没有为了通过而切换整场积分方法。</p><p>全时段的输入语义敏感性另存CSV。实际步长以后会受CFL限制，已有平均值无法恢复每一步，因此该起雨解释不自动推广到全场时序。</p></section>
<section><h2>待运行的50份输入也需要重新核验</h2><ul><li><strong>25份</strong>有读入时间节点重合；各节点可提前最多<strong>59秒</strong>。</li><li><strong>48份</strong>的连续气象输入积分发生变化，最大相对差约<strong>2.884%</strong>。这是输入雨量的差别，不是已经算出的积水误差。</li><li>原始科学曲线、24／8／8＋8划分和谱系均未变。对应<strong>58个未运行变体</strong>继续保持执行限制，须先核验新的传输版本。</li></ul><p>孤立测试中，“先乘60再按整秒处理”的未部署候选可恢复全部50份整数秒节点；这不是已修好的官方水动力镜像。单把units改为seconds无效：原函数仍硬编码乘60，会把3601读成216060秒。</p><p>后续可选择保留秒级时刻的版本化读取修正，或另立整分钟雨型库并重新计算雨量和时序。不得默默重采样原冻结曲线；仅加密输出也修不了输入时间截断。</p></section>
<section><h2>现有导出接口：有实时变量，没有完整账本</h2><table><tr><th>接口／记录</th><th>已核验</th><th>限制</th></tr><tr><td>库实时t、dt与单步更新分支</td><td>固定源码公开t／dt；单步分支存在</td><td>本轮未调用水动力库；现有CLI镜像尚无合格的诊断导出</td></tr><tr><td>NetCDF time</td><td>写的是计划tout，状态已推进</td><td>不含实际状态／通量计算时刻</td></tr><tr><td>NetCDF average_dt</td><td>写出发生在总步长累加值除以步数之前</td><td>约14400的值不能当作0.4秒平均值，更不是逐步轨迹</td></tr><tr><td>timestep_analysis</td><td>每单元平均受限步长、限制比例等汇总</td><td>不恢复逐步时钟</td></tr><tr><td>全域储水与边界交换</td><td>有300s有符号储水和约0.4s瞬时边界流量</td><td>没有与每步实际应用源汇严格对应的累计预算</td></tr></table></section>
<section><h2>最少补出三组诊断</h2><ol><li><strong>时钟与输入解码：</strong>计划时刻、实际计算时刻、实际dt、步号，原输入与读入后的雨量节点、单位及气象插值区间；原生峰值累加的实际起止时刻。</li><li><strong>同一步水量账本：</strong>真正加进连续方程的雨量、逐面分正负后累计的边界入／出水、其他源汇，以及对应的有符号储水。初始状态须在第一步之前记录；按全域与7个水域编码汇总。</li><li><strong>数值健康记录：</strong>每步更新的最小原生体积、负体积数量／总量及“曾为负”的原生单元掩膜。不能从300s快照推断中间均正常。</li></ol><p>峰值参考只需在原300s输出点及强迫拐点附近写上述小型累计记录，不要求马上保存全域每个面的高频立方体。内部面通量留给后续动态路径／主动控制目标。已实现字段规范和缺项即拒绝的校验器，实际求解器导出尚未部署。</p></section>
<section><h2>交付与下一步</h2><p>__CHECKS__项接口、故障及数据保全检查通过。只补读一份固定版本官方日期源码，并独立执行60次日期函数调用；没有启动官方水动力程序或读取新测试响应。</p><p>下一步优先核验并版本化秒级时间读取修正，再接入上述实际时钟和同一步累计预算。随后仍需另行授权水动力端到端检查；当前不能开启训练。</p><p><a href="README.md">复现说明</a> · <a href="validation.json">检查明细</a> · <a href="source_evidence.json">源码证据</a> · <a href="onset_residual_explanation.csv">起雨全域对账</a> · <a href="prospective_export_impact.csv">50份输入影响</a> · <a href="remaining_diagnostics.csv">诊断缺口</a> · <a href="../../../processed/three_ward_clock_budget_audit/diagnostic_export_schema.json">最小导出规范</a></p></section></html>'''
    text=text.replace('__TABLE__',table).replace('__CHECKS__',str(len(v['checks'])));(report/'review.html').write_text(text)
    missing=[{'priority':'P0','diagnostic':'Second-preserving rainfall decode','present':'official converter truncates fractional minutes','required':'qualify a versioned reader or explicit redesigned data version; persist actual decoded knots','hook':'after convert_fewsdate','status':'candidate unit arithmetic verified; engine change not deployed'},
      {'priority':'P0','diagnostic':'Actual clock and peak-window endpoints','present':'nominal time only; public live t/dt in source','required':'float64 nominal/actual times, actual dt, step index, pre-step0 state and actual peak-window start/end','hook':'before write_output, after continuity','status':'schema and validator only'},
      {'priority':'P0','diagnostic':'Applied-step global budget','present':'signed snapshots and sampled instantaneous boundary flux only','required':'cumulative actual net-rain, gross boundary out/in, explicit other source/sink terms and same-step storage','hook':'same continuity dvol terms including coarse/fine factors','status':'not exported; do not substitute final cumprcp or unknown zeros'},
      {'priority':'P1','diagnostic':'Seven-group exchange','present':'historical sections can map to seven distinct codes','required':'optional same-step cumulative group volumes with per-face sign split','hook':'boundary contribution accumulator','status':'mapping available; same-step counters absent'},
      {'priority':'P1','diagnostic':'Every-step negative-volume health','present':'300s negative snapshot flags only','required':'online minimum, negative count/deficit and ever-negative mask','hook':'after storage update before output subsampling','status':'not exported'},
      {'priority':'P2','diagnostic':'Full internal-face dynamic budgets','present':'absent','required':'only if later path-duration or active-control target needs dynamic conservation','hook':'native interval-integrated flux and matched storage','status':'not required for current minimal global peak audit'}]
    from audit_three_ward_clock_budget import write_csv
    write_csv(report/'remaining_diagnostics.csv',missing)
    readme='''# 三区实际时钟与步积分预算接口核查

结论：固定官方分钟转换器丢失分钟小数；输入1秒雨量斜坡变重合节点，再经气象重采样提前一秒。独立编译原始函数，Git blob与固定commit一致；仅测试日期／降雨积分，没有水动力模拟。

起雨前封顶dt=0.4s是明示的接口复现条件，不是从文件恢复的实际步长轨迹。在此条件下，全域起雨残差约468/780m³可被源代码规定的输入和端点积分重现；另计已有边界积分口径差后，最大剩余约3.7e-5m³。没有改写原残差、按结果拟合时间偏移或切换冻结的整场预算政策。

50份待运行输入中25份有读入节点碰撞、48份连续雨量积分变化（最大约2.884%），节点可提前59s。58个变体全部继续限制执行；原科学曲线、谱系、划分和旧政策均未修改。此前host文件回读通过不等于内部读取合格，本次发现补足其遗漏。

## 复现

在项目根目录，使用已有numpy、netCDF4和matplotlib环境：

    python src/analyses/validate_three_ward_clock_budget.py
    python src/analyses/review_three_ward_clock_budget.py

日期函数的官方原始源码、Git blob清单和独立驱动保存在处理目录source_snapshots及date_probe.f90。独立测试可在临时目录用本机gfortran编译sfincs_date.f90与date_probe.f90；驱动只调用日期函数，不链接水动力求解器。当前本机gfortran需要使用系统ld，避免Conda旧ld；本次编译收据记录在fortran_probe_receipt.json。

完整审计入口是src/analyses/audit_three_ward_clock_budget.py。冻结交付版本拒绝覆盖；修订时另建版本。新代码源接口在src/preprocessing/sfincs_clock_budget_interface.py，包含日期算术、纯降雨积分和未来导出账本的缺项拒绝校验器。所有模拟状态读自此前已有结果，未启动Docker、SFINCS、BMI或任何学习器。

## 处理目录

- source_snapshots：10份固定官方源码；逐一校验commit树中的Git blob。
- interface_findings.json：时钟、输入和累计量的来源语义。
- diagnostic_export_schema.json：最小实际时钟／同一步预算规范；尚未接入求解器。
- transport_readiness.json：对旧host回读“内部等价”推断的替代说明，保留旧文件与原曲线。
- freeze.json：本次取证产品和代码哈希。

## 报告目录

- onset_residual_explanation.csv：8个已有运行的3600s全域残差、条件降雨积分、边界口径差与剩余。
- nominal_prefix_sensitivity.csv：所有49个名义检查点的已知输入语义差；不是实际已应用降雨或获准的修正。
- existing_rain_time_decode.csv：已有输入文件原秒值与官方读入秒值。
- prospective_export_impact.csv / prospective_knot_decode.csv：50个待运行强迫的转换影响。
- existing_station_onset.csv：已有单站在起雨附近的辅助记录；不代表全域高频场，也没有逐点地物核查。
- existing_clock_inventory.json：NetCDF average_dt的误名和日志均值；均不能恢复每步轨迹。
- source_evidence.json：带行号的源证据；源码原文另行保存。
- remaining_diagnostics.csv：最少补出的时钟、已应用预算与数值健康字段。

## 解释边界与下一步

源代码公开实时t/dt并有sfincs_update(0)单步分支，但本轮没有调用。累计预算必须来自连续方程实际使用的netprcp、边界面和其他源汇项，不能拿float32末端累计雨量代替，也不能裁负体积。保留原时间标签，另写实际时刻。峰值任务只需小型全域／7组累计记录及曾为负掩膜；暂不要求全内部面高频立方体。

整秒候选在孤立函数测试中恢复了50份输入的整数秒节点，尚未部署或验证官方计算镜像。只改units为seconds会被硬编码乘60，是错误方案。选择读取修正或重设计整分钟雨型时必须版本化并重新核验。

8次严格参考准入限制保持；33项规则测试通过不等于模型可训练。无新水动力模拟、无训练、无新封存响应读取。AnaSOP仅同步第4、6、7节；人类决策记录MILI-D-20261015-007。
'''
    (report/'README.md').write_text(readme)
    products={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='freeze.json'}
    code={str(p.relative_to(root)):sha(p) for p in [root/'src/preprocessing/sfincs_clock_budget_interface.py',root/'src/analyses/audit_three_ward_clock_budget.py',root/'src/analyses/validate_three_ward_clock_budget.py',Path(__file__)]}
    state={'state':'interface_findings_frozen_no_engine_deployment_no_reference_release','products_sha256':products,'code_sha256':code,'validation_sha256':sha(report/'validation.json'),'new_hydrodynamic_runs':0,'training':0}
    if freeze:
        f=out/'freeze.json'
        if f.exists():assert json.loads(f.read_text())==state
        else:dump(f,state)
    dump(report/'product_manifest.json',state)
    print(json.dumps({'review':str(report/'review.html'),'checks':len(v['checks']),'sop_sections_changed':changed,'frozen':(out/'freeze.json').exists()},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--freeze',action='store_true');a=p.parse_args();main(a.root,a.freeze)
