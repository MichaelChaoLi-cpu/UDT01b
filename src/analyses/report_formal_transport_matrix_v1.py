"""Offline registered matrix and current bounded serial campaign status."""
from pathlib import Path
import sys,json,html,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_formal_transport_experiments_v1 import P,R,D,read,dump,sha

def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def main():
 p=json.loads((D/'protocol.json').read_text());m=json.loads((P/'run_manifest.json').read_text());s=json.loads((R/'campaign_state.json').read_text()) if (R/'campaign_state.json').exists() else dict(status='registered_inputs_preflight',completed_new_cases=[],current_case=None,active_wall_seconds=0,queue=m['new_execution_order']);validation=json.loads((R/'plan_validation.json').read_text()) if (R/'plan_validation.json').exists() else None
 factors=table(['因素','固定水平','意义'],[['降雨','R0：无雨；R1：原历史雷达雨场；R125：同雨场×1.25','均为9小时、相同108个5分钟源时段；增强不是另一独立实测暴雨或重现期'],['河道水位','L：既有低水位；H：七组共同＋1.5 m脉冲','18:00—18:30升、18:30—19:00平台、19:00—20:00降；均是假设'],['城市几何','T：仅原地形；U：既有城市约束组合','相同10 m网格、基础地形和固定求解器；U包含建筑阻水、道路输水和屋面分配的组合'],['停运假设','不假设轨道关闭／站口／整站通过／区段','道路积水限制仍适用；停运属于后处理假设，不重复跑水动力'],['道路和资料分支','严格分类道路／步道补缺 × 未知保留／移除 × 0.1／0.3 m × 两种站口归属','桥隧高程不明保持未知；每个分支有自己的正常基线']])
 rows=[]
 for r in m['runs']:
  case=r['case_id'];audit=R/'execution'/case/'audit.json';status='复用已通过结果' if r['reused'] else '数值审计通过' if case in s['completed_new_cases'] else '正在计算／审计' if case==s.get('current_case') and s['status'] not in ['held','paused_by_user'] else s['status'] if case==s.get('current_case') else '排队，尚未启动';batch='复用' if r['reused'] else next(b['batch_id'] for b in p['batches'] if case in b['new_cases']);rows.append([case,batch,r['rainfall'],r['stage'],r['arm'],status])
 body='<h1>正式交通情景实验：冻结矩阵与分批执行</h1><p class="lead">围绕暴雨下的交通联系、地面替代与城市几何信息价值。覆盖都心三区及已有跨区范围；不计算受困人数，不训练代理模型，不打开封存测试。</p>'
 body+='<div class="notice"><b>已注册12个水动力算例，复用2个，最多新增10个。</b><p>当前状态：'+html.escape(s['status'])+'；当前算例：'+html.escape(str(s.get('current_case')))+'；已完成新增：'+str(len(s['completed_new_cases']))+'/10。以下矩阵是计划，排队项不代表已执行结果。</p></div>'+factors
 body+='<h2>首批与完整水动力矩阵</h2>'+table(['算例','批次','雨型','水位','几何','当前状态'],rows)
 body+='<p>B1先补原历史雨场的T低水位与T抬升两例，与已有U两例配对；B2运行四个无雨控制；B3运行四个1.25倍雨场。每例通过原数值审计才进入下一例，每批全部通过才进入下一批。所有后代仍属于同一历史事件谱系，不能作为12场独立暴雨；原12＋4训练／验证划分保持不变。</p>'
 body+='<h2>预算与停止条件</h2>'+table(['项目','冻结上限／预计'],[['并发及资源','1实例；每例6核、7 GiB；固定已验证CPU镜像'],['首批预计','约2—3小时，地形组实际速度可能不同'],['整轮预计与上限','预计8—14小时活跃计算；总上限14小时，每例上限2小时；到限停队列，未完成项保留'],['输出与磁盘','新增输出最多120 GiB；剩余空间低于100 GiB即停'],['数值闸门','原正性、雨量守恒、同一步水量预算、完整双向边界交换、实际时钟和标签隔离阈值不变'],['停止行为','数值／版本／诊断失败、15分钟时钟停滞、进程异常或资源超限：停队列并保留现场；不自动重试、换情景或放宽阈值'],['用户暂停／停止','可通过本任务请求暂停或停止。控制器PAUSE标记冻结容器和活跃预算；STOP标记停止并保留现场']])
 body+='<p>计算预计参考既有9小时城市组约56.5／57.7分钟，不是完成承诺。预算不是无限后台队列；最多新增10个物理算例。交通对照只对通过数值闸门的结果后处理：每个时刻64组、12个同步时刻与单列峰值，共最多9,984组网络评估。此数量不是水动力运行数。</p>'
 body+='<h2>什么保持相同，什么可解释</h2><p>T与U共用源雨场总量和同一水位文件；T直接承接原地表源雨，U按既有屋面来源—接收关系保守分配，避免给T重复施加屋面径流。T恢复建筑占地储水并采用统一糙率，U保留建筑障碍、道路糙率、格面约束和瞬时屋面分配。因此对照评估的是城市信息组合，不是单独某个建筑高度的因果效应，也不是三维流体求解器对二维求解器的比较。</p>'
 body+='<p>交通图在T/U之间保持同一版本；差别来自积水与其假设关闭。原115点和补缺新增29点分层报告，144点为当前最大共同支持队列；209点总覆盖和65个原有证据不足点继续单列。新算例若增加隔离或缺测，按共同准入掩膜报告额外剔除，不临时换点。严格道路层、地面补缺层、未知分支和两种站口归属始终并列。</p>'
 body+='<h2>结果与后续措施</h2><p>预定输出为同步积水暴露、假设停运下的替代路径及步行距离、模型内稳健／条件敏感／证据不足片区，以及几何、雨量与水位的成对变化。阶段性输出先呈现覆盖和闸门，再解释差异；峰值包络不当作同一时刻断连。无雨控制专门分离假设水位影响；不能把全部积水归因于暴雨。</p>'
 body+='<p>主动措施在正式基线和城市几何对照后衔接，按可支持的站口防护或替代连接等措施另行冻结输入与能力假设。本批次不补造管网、泵站、地铁防洪设施能力或完整河床几何。历史雨场驱动、假设水位、不同年代城市资料以及无管网／泵站／入渗等局限不变；不声称真实灾情复现、实际停运概率或受困人数。</p>'
 if validation:body+='<p>启动前输入检查：'+str(len(validation['checks']))+'项，全部通过。最终每例仍需独立运行审计；输入通过不代表模拟通过。已用技术审计代替不了实测灾情验证。</p>'
 body+='<h2>版本与当前进度</h2><p>Framelog: MILI-D-20261017-004。独立执行协议以本次明确的交通研究目标为准；早期机器学习试验及其分区、数据与历史记录保留，没有恢复训练。</p><ul>'
 files=[D/'protocol.json',P/'run_manifest.json',P/'protected_sources.json',R/'input_validation.json',R/'plan_validation.json',R/'forcing_volume_preflight.csv']
 if (R/'campaign_state.json').exists():files.append(R/'campaign_state.json')
 for f in files:
  if f.exists():body+='<li><a href="'+os.path.relpath(f,D)+'">'+html.escape(f.name)+'</a></li>'
 body+='</ul>'
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区正式交通实验矩阵与进度</title><style>body{font:16px/1.8 system-ui;color:#243b49;background:#f3f6f9;max-width:1200px;margin:auto;padding:32px}h1,h2{line-height:1.4}h2{margin-top:36px}.lead{font-size:19px}.notice{background:#e7f1ec;border-left:5px solid #35806b;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border:1px solid #d5e0e8;padding:10px;text-align:left;vertical-align:top}th{background:#e5edf3}a{color:#17648b}</style>'+body+'</html>'
 tmp=D/'review.tmp';tmp.write_text(page);tmp.replace(D/'review.html');print('REPORT',s['status'],s.get('current_case'),flush=True)
if __name__=='__main__':main()
