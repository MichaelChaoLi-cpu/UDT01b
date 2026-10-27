"""Human-reviewable proposed controlled experiment; no execution or fitted results."""
from pathlib import Path
import html,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write,sha
NAME='three_ward_unet_control_plan_v2';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def table(rows,cols):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for k,t in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,t in cols)+'</tr>' for r in rows)+'</tbody></table></div>'
def main():
 p=read(OUT/'protocol.json');v=read(REPORT/'validation.json');assert v['all_checks_passed'];probe=read(REPORT/'selection_rule_probe.json')
 arms=[{'arm':r['arm_id'][0],'loss':'原面积加权 MSE' if r['loss']=='mse' else '分层 2×Huber','lr':r['learning_rate'],'budget':'1 个种子 / 40 轮 / 480 步','status':'未开始'} for r in p['arms']]
 compare=[{'pair':'B − A；D − C','change':'损失组合','fixed':'分别固定原／较低学习率','meaning':'分层+稳健尾部作为整体；本轮不能分离两者各自作用'},{'pair':'C − A；D − B','change':'学习率','fixed':'分别固定 MSE／分层稳健损失','meaning':'只改变学习率，不同时更换调度器、初始化或网络'},{'pair':'两组损失差之差','change':'交互（描述性）','fixed':'共同数据、初始化和曝光顺序','meaning':'单种子、4 个验证事件，不作总体推断'}]
 lossrows=[{'part':'非湿区','ref':'参考水深 ≤0.10 m','w':'0.50','note':'保留对误报的损失压力；这不是把浅水当作真实零水深'},{'part':'普通湿区','ref':'0.10 m <参考水深≤1.00 m','w':'0.40','note':'独立计算区域内面积加权误差'},{'part':'深水尾部','ref':'参考水深>1.00 m','w':'0.10','note':'仍参与监督、单列深水 MAE，标签不截断也不删除'}]
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-ward next controlled trial: preparation</title><style>body{font:15px/1.7 system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 22px;background:#fafbfd;color:#172033}h1{font-size:28px}h2{font-size:21px;margin-top:30px}.box{padding:18px;background:#eaf5f1;border-left:5px solid #0f766e}.note{color:#596579}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:9px;border:1px solid #d9e1e9;text-align:left}th{background:#edf2f7}code{font-size:13px}li{margin:8px 0}a{color:#1762af}</style><h1>下一轮小规模对照：损失、选模与优化日志准备</h1>'''
 body+=f'<div class="box">接口和规则已实现，{len(v["checks"])} 项核验通过。新增拟合 0 次、优化器更新 0 步、封存测试未打开。<br>四组数值设置为本次准备形成的具体工程提案；尚未启动，也不表示这些数值已证明有效或由用户逐项选定。研究范围记录：MILI-D-20261016-004。</div>'
 body+='<h2>拟议四组对照</h2>'+table(arms,[('arm','组'),('loss','损失'),('lr','固定学习率'),('budget','共同预算'),('status','状态')])+table(compare,[('pair','比较'),('change','改变项'),('fixed','固定项'),('meaning','解释边界')])
 body+='<p>共最多 4 次拟合、1,920 次更新。四组共用已保存的同一份未训练初始化和同一份 40 轮训练父事件顺序。沿用原条件 U-Net（671,513 个参数）、Softplus、水深单位、输入编码、训练集归一化和原质量掩膜；每个训练父事件每轮一次全域更新。AdamW 权重衰减 0.00001、梯度裁剪 10 保持一致；固定 40 轮、没有早停，以免不同暴露次数干扰比较。40 轮是预算上限，不是收敛承诺。历史首轮只作背景，不冒充新的等预算 A 组。</p>'
 body+='<h2>分层稳健损失：保留深水，避免单一平方尾部主导</h2>'+table(lossrows,[('part','层'),('ref','按原精确参考划分'),('w','拟议混合权重'),('note','含义')])
 body+='<p>每层内部先按原 Ground Area 加权平均，再按 0.50/0.40/0.10 混合。候选误差函数为 2×Huber，转折残差固定 0.10 m：残差绝对值 ≤0.10 m 时等于平方误差，更大时转为线性，两种损失仍以 m² 计。分层依据参考水深，Huber 转折依据预测残差，两者含义不同。缺少某层时仅在实际存在层间重归一化，并记为缺层；不把空层错误记为零误差。整个事件无精确支持则失败。原未知、区间和隔离标签不参与监督，原深水标签不裁剪。</p><p class="note">权重与转折值是固定工程提案，没有用验证集拟合或网格搜索。分层与稳健残差在此作为一个损失组合；不能从这四组对照单独认定哪一个组件有效。</p>'
 body+='<h2>湿区与误报指标</h2><p>主阈值保持水深严格大于 0.10 m；0.05、0.30、0.50 m 仅作原有敏感性视图。TP、FP、FN、TN 都按开放地面面积计算。漏报率 FNR = FN/(TP+FN)；误报率 FPR = FP/(FP+TN)，其分母是参考非湿区；误报占预测湿区的比例另记为 false discovery，两者不混用。</p><p>主排序指标 B = 0.5 × (FNR + FPR)。完美预测 B=0；在真实干湿两类均存在时，全预测无积水和全预测积水均为 B=0.5。保留 IoU、召回率、精确率、误报/漏报 km²、MAE、RMSE、偏差，以及非湿区、普通湿区、深水和全部湿区 MAE。空类指标为 NA 并报告有效父事件数，不把它计成完美或零误差。</p>'
 body+='<h2>检查点选择与可用性分开</h2><ol><li>每轮全部 12 次更新后冻结模型，分别重算训练和验证数据；全域、三区、各区和跨区贡献区域，均保留原目标精确支持和检索配对支持。</li><li>主排序用三区内固定配对支持：目标原精确掩膜 ∩ 原训练检索来源精确掩膜 ∩ 有效地面 ∩ 三区。来源检索和掩膜预先固定，逐项 SHA256 校验；不随网络表现改变。</li><li>四个验证父事件先分别算指标，再等权平均；依次最小化 B、参考湿区 MAE、总体 MAE，同分取更早轮。比较前将这些值舍入到 6 位小数，形成固定全序。缺少父事件、关键类分母为空或掩膜不符时不能排名。</li><li>另作拟议开发基线比较：每一个验证父事件都需满足 B 不差于检索、总体 MAE 不差于无积水和检索、召回不差于检索、误报面积不大于检索。比较数值容差统一为 10⁻⁶（对应比率、m 或 km²），仅为工程数值容差。</li><li>可以保留“诊断最佳”检查点，同时明确标为 not_ready。未过基线比较不自动变成可用模型，也不会触发打开测试、追加训练或改变门槛。</li></ol><p>全部方法在同一个固定配对支持上比较；完整目标精确支持的分数和拒答面积另报。此处基线比较是拟议开发规则，不是既有数值标签准入政策，也不是实测洪水精度认证。</p>'
 pp=[]
 for q in probe['probes']:pp.append({'source':q['source'],'B':q['selection_key_if_this_rule_were_used'][0],'guard':q['readiness']['all_parents_pass'],'promoted':False})
 body+='<p>用现有两个保存检查点核验接口如下。表中是事后规则试算，不是新的模型选择或新训练结果。</p>'+table(pp,[('source','原检查点'),('B','按拟议规则的 B'),('guard','通过全部基线比较'),('promoted','本次提升为新模型')])
 body+='<h2>优化日志和逐轮评分合同</h2>'+table([{'level':'每一步','fields':'组、父事件/谱系/强迫指纹、种子、精确掩膜和协议指纹；更新前损失与各层贡献；裁剪前后梯度总/模块范数、有限性与缺梯度；裁剪次数；实际模块参数变化和参数范数；学习率、优化器前后真实步数、计时范围。'},{'level':'每轮固定模型','fields':'16 个父事件 × 6 个地理范围 × 原支持/配对支持，按父事件完整存储；全域原精确支持上的固定模型损失及分层项；训练/验证分开汇总，报告指标有效父事件数。'},{'level':'更新过程汇总','fields':'完整的 12 个父事件曝光、连续步数、更新前损失平均、梯度分位数、裁剪数和模块更新分布；与固定模型损失使用不同字段。'},{'level':'检查点与时钟','fields':'每轮模型状态及 SHA256、当前优化器/RNG 状态、固定评分与基线失败项；训练、评分、检查点 I/O 分开计时；评分前后模型指纹/步数相同。'},{'level':'失败处理','fields':'非有限有效预测、损失、梯度/参数，谱系/掩膜/版本不匹配，或优化器步数不连续时停止并保留失败记录；不靠删单元修复。'}],[('level','层级'),('fields','必须记录')])
 body+='<p>日志模块只观察，不调用 optimizer.step。当前两个逐步示例分别是已有模型的零更新反向接口核验和明确标记的人工大梯度裁剪夹具；逐轮示例来自已有第 15 轮预测的固定评分。它们均标为 interface_probe，不是实际训练步或新一轮训练历史。真实更新日志还必须在未来授权的实际拟合中验证，当前不能补造历史梯度或剪裁率。</p>'
 body+='<h2>已完成与仍待执行</h2><ul><li>已完成：损失与梯度的手算一致性、掩膜外 NaN 零梯度、空层/空事件处理、指标分母、空类处理、固定排序与拒绝规则、真实数据支持指纹、共享未训练初始化、日志顺序/裁剪/零参数变化及评分帧完整性核验。</li><li>未执行：四组拟合、任何新的模型性能比较、区间监督/GNN、追加水动力情景、封存测试和主动措施。下一步拟合需另行启动；本协议的 execution_authorized 为 false。</li><li>保留边界：4 个验证父事件已经用于诊断和规则设计，仍只是开发集；单种子对照不能证明统计泛化、容量充分性、实测精度或动态水量守恒。主动措施仍在后续第五阶段。</li></ul>'
 links=['preparation_receipt.json','validation.json','selection_rule_probe.json','frozen_prediction_loss_probe.json','step_interface_probe.jsonl','epoch_interface_probe.jsonl','synthetic_clipping_probe.jsonl','anasop_update.json']
 body+='<h2>文件与复现</h2><p>'+ ' · '.join('<a href="'+q+'">'+q+'</a>' for q in links)+'</p><p>输入引用、四组配置、共同初始化和逐轮顺序保存在同名 processed 目录。核心实现为 three_ward_surrogate_protocol_v2.py；准备、验证和报告脚本位于 src/analyses。没有新增依赖，也没有 Git/DVC 操作。</p></html>'
 (REPORT/'review.html').write_text(body)
 (REPORT/'README.md').write_text('''# Prepared next surrogate comparison (no fitting)

Status: proposed fixed 2x2 loss-by-learning-rate comparison, prepared and interface-tested only. Execution remains unauthorized. The four arms retain original 12/4 parents and exact masks; no sealed data is opened.

Components are in src/analyses/three_ward_surrogate_protocol_v2.py. Run preparation, validation, documentation and review scripts with the project Python. Preparation/validation refuse silent overwrite. The supplied code provides training primitives and telemetry interfaces, not an invoked fitting loop; a future authorized runner must call them and enforce the bound configuration. The original training script and frozen deliveries are unchanged.

The reference loss is original exact-area MSE. Candidate 2*Huber(delta0.10m) and 0.5/0.4/0.1 nonwet/ordinary/deep mixture are a concrete engineering proposal; they have not been optimized or evaluated in new fits. The loss contrast tests the combined stratification/robust-loss package, not each component separately.

Files named interface_probe are zero-update observations or explicit synthetic fixtures, never real training histories. Synthetic optimizer-clock dictionaries exist only in memory for unit checks. The preflight-draft folders preserve earlier interface checks while epoch aggregation and clock binding were completed; none performed fitting. Real future step records must be contiguous, cover all 12 parents, and retain separate fixed-model loss/metrics. Do not call a selected diagnostic checkpoint admitted or usable when baseline guards fail.

Protocol and shared initialization hashes must be verified before a separately authorized runner is implemented or started. No existing label registry or global release is changed.
''')
 write(REPORT/'summary.json',{'status':'prepared_not_executed','proposed_arms':4,'maximum_future_optimizer_updates':1920,'training_parents':12,'validation_parents':4,'shared_seed':p['frozen_common_settings']['seed'],'candidate_loss_weights':p['loss']['weights'],'selection':'B, reference-wet MAE, overall MAE, earliest epoch; paired three-ward original support','new_training_runs':0,'optimizer_updates':0,'sealed_opened':False,'checks':len(v['checks']),'execution_authorized':False,'no_new_runner_invoked':True,'settings_are_proposed_not_proven':True})
 print(json.dumps({'report':str(REPORT/'review.html'),'checks':len(v['checks']),'new_fits':0}))
if __name__=='__main__':main()
