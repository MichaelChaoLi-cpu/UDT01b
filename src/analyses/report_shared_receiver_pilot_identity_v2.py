"""Rebuild original review plus paired road effects after both cases qualify."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from shared_receiver_pilot_common_v1 import P,R,dump
import report_shared_receiver_pilot_v1 as original
def main():
 original.main()
 state=json.loads((R/'campaign_state.json').read_text())
 if state['status']!='qualified_complete':return
 v=json.loads((R/'postprocessing_validation.json').read_text());assert v['passed']
 roads=pd.read_csv(P/'road_trajectories.csv');transfer=pd.read_csv(P/'off_target_head_changes.csv');rows=[]
 for keys,g in roads.groupby(['Case','Stage','Depth Branch','Scope']):
  b=g[g.Arm=='baseline'].sort_values('Frame');d=g[g.Arm=='drainage'].sort_values('Frame');assert np.array_equal(b.Frame,d.Frame)
  t=b['Analysis Time'].to_numpy()/3600
  net=b['Closed Road Length'].to_numpy()-d['Closed Road Length'].to_numpy()
  restored=d['Same Time Restored Road Length'].to_numpy();added=d['Same Time Newly Closed Road Length'].to_numpy()
  assert np.max(abs(net-(restored-added)))<1e-7
  bi=int(np.argmax(b['Closed Road Length'].to_numpy()))
  base_integral=float(np.trapezoid(b['Closed Road Length'],t));change=float(np.trapezoid(net,t))
  row=dict(zip(['Case','Stage','Depth Branch','Scope'],keys))
  row.update({'Baseline Peak Closed km':float(b['Closed Road Length'].max()),'Drainage Peak Closed km':float(d['Closed Road Length'].max()),'Baseline Closure km h':base_integral,'Closure Reduction km h':change,'Closure Reduction Percent':100*change/base_integral if base_integral else None,'Baseline Peak Hour':float(t[bi]),'Restored At Baseline Peak km':float(restored[bi]),'Newly Closed At Baseline Peak km':float(added[bi]),'Net Restored At Baseline Peak km':float(net[bi]),'Maximum Same Time Restored km':float(restored.max()),'Maximum Same Time Newly Closed km':float(added.max()),'Maximum Same Time Net Improvement km':float(net.max()),'Minimum Same Time Net Improvement km':float(net.min()),'Restored Road km h':float(np.trapezoid(restored,t)),'Newly Closed Road km h':float(np.trapezoid(added,t))})
  rows.append(row)
 paired=pd.DataFrame(rows);paired.to_csv(P/'paired_road_effects.csv',index=False)
 shifts=transfer.groupby(['Case','Stage','Scope'],as_index=False).agg({'Outside Target Cells Certain Head Rise Area':'max','Outside Target Cells Certain Head Fall Area':'max'})
 shifts.to_csv(P/'off_target_summary.csv',index=False)
 selected=paired[(paired.Scope=='Three wards')&(paired['Depth Branch']=='lower')]
 effect={'passed':True,'paired_time_identity_checked':True,'quality_rules_unchanged':True,'road_effects':rows,'off_target':shifts.to_dict('records'),'limits':['Two stage assumptions of one historical parent,not independent observed events.','Only common admitted roads;unknown exposure remains unknown.','Off-target certain head rise threshold0.10m with original dry-censor bounds;smaller rises are not excluded.','Prescribed-stage receiving boundaries do not simulate added downstream river storage or acceptance.','No new congestion or affected-person results.']}
 dump(R/'effects_summary.json',effect)
 page=(R/'review.html').read_text()
 extra='<h2>同一时刻的道路改善汇总</h2><p>以下为三区、原水深下界分支。封路暴露减少量为同一时刻基线与措施差的时间积分；恢复与新增封路分别保留。峰值时刻由无措施基线预先确定，不用措施效果选时刻。</p><div class="scroll">'+selected.to_html(index=False,float_format=lambda x:f'{x:,.6f}')+'</div>'
 extra+='<h2>积水转移的识别范围</h2><p>下表为各范围内、目标道路控制单元以外，水位确定升高至少0.10m的最大同时面积，以及确定降低面积；保留原0—0.01m干单元删失界限。零值不排除更小变化。接收端规定水位，尚不能评价真实河道新增蓄水的下游影响。</p><div class="scroll">'+shifts.to_html(index=False,float_format=lambda x:f'{x:,.6f}')+'</div>'
 extra+='<p><a href="effects_summary.json">机器可读效果汇总</a> · <a href="identity_repair_v2/validation.json">身份修复核验</a> · <a href="identity_repair_v2/failed_high_stage/audit.json">保留的原失败审核</a></p>'
 (R/'review.html').write_text(page.replace('</html>',extra+'</html>'))
 print('PAIRED_ROAD_AND_TRANSFER_EFFECTS_SUMMARIZED',flush=True)
if __name__=='__main__':main()

