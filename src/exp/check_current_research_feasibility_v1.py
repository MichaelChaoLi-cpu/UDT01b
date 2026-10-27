"""Evidence-reviewed feasibility supplement to the bundled lexical skill.
Read-only scientific inputs;writes only the current feasibility output directory.
"""
from pathlib import Path
import csv,hashlib,html,json,re,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'data/exp/feasibility-check/three_ward_current_research_v1'
BRIEF=ROOT/'data/exp/data-briefing/three_ward_current_research_v1'
P=ROOT/'data/processed/three_ward_barrier_drainage_v1'
R=ROOT/'data/exp/estimation/three_ward_barrier_drainage_v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def save(name,rows):pd.DataFrame(rows).to_csv(O/name,index=False)

def main():
 doc=ROOT/'docs/AnaSOP.md';text=doc.read_text();before=read(O/'before.json');assert sha(doc)==before['anasop_sha256']
 s1=text.split('## 1. Research Objective\n',1)[1].split('## 2.',1)[0]
 central=re.search(r'(?m)^- Research question: (.+)$',s1).group(1)
 supporting=s1.split('### Supporting Research Questions\n',1)[1].split('### Scope of Analysis',1)[0]
 questions=[line.split('|')[1].strip() for line in supporting.splitlines() if line.startswith('| ') and not line.startswith('| research question')]
 assert len(questions)==4
 allq=[central]+questions
 bindings={str(doc.relative_to(ROOT)):sha(doc)}
 for name in ['coverage.json','validation.json','tables/dataset_inventory.csv','tables/variable_summary.csv','tables/variable_screening.csv']:
  p=BRIEF/name;bindings[str(p.relative_to(ROOT))]=sha(p)
 brief_sources=read(BRIEF/'source_bindings.json')
 assert all(sha(ROOT/n)==h for n,h in brief_sources.items()),'Briefing inputs changed'
 bindings.update(brief_sources)
 required={
 'Current briefing':BRIEF/'validation.json',
 'Whole-event road and resident analysis':ROOT/'data/exp/estimation/three_ward_five_event_temporal_analysis_v1/A1_validation.json',
 'Geometry reference comparison':ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1/B1_geometry_comparison/validation.json',
 'Geometry forcing comparison':ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1/B2_B3_forcing_geometry_comparison/validation.json',
 'Physical hydraulic delivery':R/'delivery_validation.json',
 'Physical road and population':R/'postprocessing_validation.json',
 'Physical traffic':R/'traffic_validation.json',
 'Physical paired effect summary':R/'effect_summary/validation.json'}
 evidence=[]
 for name,p in required.items():
  a=read(p);passed=bool(a.get('passed',a.get('all_passed',False)));assert passed,name
  bindings[str(p.relative_to(ROOT))]=sha(p);evidence.append(dict(evidence=name,path=str(p.relative_to(ROOT)),check_passed=passed,meaning='Qualification of its declared data/model operation;not independent observed-effect validation.'))
 # Verify every existing historical case's admission, clock export and actual source files.
 history=pd.read_csv(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv');cases=[]
 for r in history.itertuples():
  a=read(ROOT/r.audit_path);assert a['all_checks_passed'];assert(ROOT/r.map_path).exists()and(ROOT/r.clock_path).exists();assert r.map_frames==109 and r.nominal_spacing_s==300 and r.max_nominal_actual_difference_s<=.001
  bindings[r.audit_path]=sha(ROOT/r.audit_path)
  cases.append(dict(case=r.case_id,parent=r.event_id,role='Historical no-action state',passed=True,frames=r.map_frames,actual_clock_error_s=r.max_nominal_actual_difference_s))
 manifest=read(P/'run_manifest.json');assert len(manifest['runs'])==16 and len(manifest['new_execution_order'])==12
 for case in manifest['new_execution_order']:
  p=R/'execution'/case/'audit.json';a=read(p);assert a['all_checks_passed']and a['admission']['technical_reference'];assert a['summary']['negative_cells']==0 and a['summary']['map_records']==109
  assert all(a['physical_measure_checks'].values());bindings[str(p.relative_to(ROOT))]=sha(p)
  cases.append(dict(case=case,parent=case.split('__')[0],role='Physical intervention descendant',passed=True,frames=a['summary']['map_records'],actual_clock_error_s=abs(a['summary']['actual_final_time_s']-32400)))
 save('case_admission_inventory.csv',cases)
 assert history.event_id.nunique()==5 and len(history)==10
 # Check completed downstream interfaces without recomputing their models.
 roads=pd.read_csv(P/'road_summary.csv');pop=pd.read_csv(P/'paired_population_effects.csv');tr=pd.read_csv(P/'traffic/physical_traffic_outcomes.csv');risk=pd.read_csv(P/'neighbor_transfer_screen.csv')
 assert len(roads)==384 and len(tr)==16 and len(pop)==31392
 assert (pop.net_added_loss_lower<=pop.net_added_loss_upper).all()and(pop.new_loss_lower<=pop.new_loss_upper).all()
 assert not tr.duplicated(['event_id','stage','measure']).any()
 assert tr.common_survivor_original_request_proxy.gt(0).all()
 assert tr.groupby(['event_id','stage']).common_survivor_original_request_proxy.nunique().eq(1).all()
 assert np.allclose(tr.original_request_proxy,tr.served_proxy+tr.unreachable_proxy,atol=1e-6,rtol=0)
 assert all(set(g.measure)=={'none','barrier','drainage','combined'} for _,g in tr.groupby(['event_id','stage']))
 for keys,g in tr.groupby(['event_id','stage']):assert g.frame.nunique()==1
 for n in ['road_summary.csv','paired_population_effects.csv','traffic/physical_traffic_outcomes.csv','neighbor_transfer_screen.csv']:
  bindings[str((P/n).relative_to(ROOT))]=sha(P/n)
 c=read(BRIEF/'coverage.json');support=[]
 with np.load(P/'motor_common_support.npz')as z:
  q=z['qualified_roads'];assert q.dtype==bool and q.sum()==28064;support.append(dict(interface='Road hazard support',count=int(q.sum()),meaning='Common admitted source roads;not complete traffic-network exposure'))
 cov=c['unresolved_population_by_radius'];unknown10=next(x['min'] for x in cov if x['buffer_m']==10)
 assert unknown10==41308 and c['official_population']==496345
 support.extend([dict(interface='Official 2020 residents',count=496345,meaning='Conserved nighttime stock;not actual event-time persons'),dict(interface='Unresolved 10m baseline attachment',count=unknown10,meaning='Outside supported conditional access comparison;not safe or affected by default')])
 save('interface_checks.csv',support)
 base_evidence=['Registered source/case inventory;per-case numerical admission;actual output clocks;common hazard support','Directed routing and demand/capacity contract;request balance;all16restricted traffic checks','Official resident units;frozen candidates;paired access intervals;unresolved stock retained','Frozen physical protocol;all12new admissions;matched16arm states;paired effects and transfer maps']
 conditional=['Whole-area road exposure and sampled closure/recovery on qualified ground','Represented request reachability and restricted static model cost under stated demand/capacity','Nighttime resident-stock loss/restoration intervals on complete attachment support','Paired conditional physical-measure passage/access/cost effects and admitted outside-corridor deterioration']
 limits=['Unknown exposure remains277.920km of1114.878km clipped roads;dry starts and endpoint censoring;no observed geometry-accuracy verdict','Incomplete ordinary-road and through demand,vehicle classes,assumed capacity,unknown exposure retained available;not measured queues','41308 residents unresolved in10m branch;no validated daytime occupancy,stranded-person count or operating service-access model','Two rain parents,open selection-event placement,prescribed receivers,unverified construction feasibility;not broad engineering efficacy']
 assessments=[dict(id='RQ0',question=allq[0],status='partly-testable',supported_scope='Conditional coupled road,traffic-proxy,resident-access and measure analysis on declared support',evidence='Registered baseline and physical-measure deliveries plus data-briefing coverage checks',limits='Observed historical flood/congestion/people and broader day/service objectives remain unvalidated',next_skill='data-preprocessing',action='Confirm variables and reporting denominators;preserve unresolved constructs and unknown support')]
 for i,qtext in enumerate(questions,1):assessments.append(dict(id=f'RQ{i}',question=qtext,status='partly-testable',supported_scope=conditional[i-1],evidence=base_evidence[i-1],limits=limits[i-1],next_skill='data-preprocessing',action='Use available operational quantities with explicit units,coverage and publication-status confirmation;no mandatory new simulation is inferred'))
 save('question_feasibility.csv',assessments)
 # Decompose broad questions so available proxy fields do not certify absent outcomes.
 claims=[
 ('RQ1','Qualified road closure trajectories and unique length','partly-testable','Qualified109frame responses and clipping gates exist','Limit scope to qualified exposure;no recovery beyond horizon'),
 ('RQ1','Observed flood accuracy improvement from urban3D information','not-yet-testable','Matched geometry differences exist;independent matched observed flood accuracy is not established','Report geometry sensitivity rather than demonstrated accuracy superiority'),
 ('RQ2','Conditional vehicle reachability and unmet represented requests','partly-testable','Directed network and original-request ledger pass','Unknown exposure,external demand and classes remain explicit'),
 ('RQ2','Restricted modeled traffic cost on commonOD','partly-testable','All16traffic states pass with identical comparison cohort within event-stage','Report model cost;do not infer actual dynamic queues'),
 ('RQ2','Actual historical queues and congestion delay','not-yet-testable','No qualified event-matched dynamic queue or delay outcome','Observed congestion validation is a separate gap'),
 ('RQ3','Conditional nighttime resident-stock access intervals','partly-testable','Official496345stock,candidate bounds and paired effects exist','Unresolved attachment and census age retained'),
 ('RQ3','Actual unique people stranded or unable to return home','not-yet-testable','Stock access loss is available;event-time people and travel histories are not','Do not label stock intervals as actual stranded people'),
 ('RQ3','Daytime people present and day/night consequence contrast','not-yet-testable','Nighttime census exists;qualified daytime presence allocation is not constructed','Retain as open objective rather than substitute nighttime stock'),
 ('RQ3','Operating urban-function loss','not-yet-testable','No qualified destination,operation,capacity and access-standard model','Portal access alone is insufficient'),
 ('RQ3','Complete multimodal traffic isolation','not-yet-testable','Motor-road and complementary rail scenarios exist without a validated combined substitution consequence model','Report mode-specific conditional access rather than total district isolation'),
 ('RQ4','Paired physical intervention benefit at frozen locations','partly-testable','Twelve new physical runs and four baseline states have qualified paired outputs','Two parents and assumed capacities limit transfer'),
 ('RQ4','Outside-corridor adverse model depth and access change','partly-testable','Conservative0.10mcell screen and new-loss intervals exist','Cell screen is not measured newly flooded land;sub-threshold shifts remain possible'),
 ('RQ4','Observed engineering efficacy or universally optimal intervention','not-yet-testable','Frozen assumed measures,not surveyed pumps or optimized equal-cost designs','No effectiveness guarantee or universal ranking')]
 save('claim_feasibility.csv',[dict(question_id=q,claim=claim,status=status,current_evidence=e,interpretation_limit=limit)for q,claim,status,e,limit in claims])
 # Exact readable-to-stored mapping is documentation,not variable selection or renaming.
 mapping=[
 ('Rainfall Forcing','Rainfall intervals','source_area_mean_mm_h;start_elapsed_s;end_elapsed_s','Prescribed historical input;not rainfall prediction'),
 ('Closed Road Length','Road trajectories and summary','closed_km;peak_closed_km','Thresholded qualified length,not road area'),
 ('Closed Road Duration','Road summary','closed_km_hours','Length-time integral,not unique road count'),
 ('Vehicle Demand','Physical traffic outcomes','original_request_proxy','Original represented demand proxy,not persons'),
 ('Unreachable Vehicle Requests','Physical traffic outcomes','unreachable_proxy','Represented requests without a modeled route'),
 ('Common OD Cost','Physical traffic outcomes','common_survivor_cost_index','Fixed same-survivingOD cost,not measured delay'),
 ('Resident Population','Official resident units','official_2020_residents','Official2020stock denominator'),
 ('Paired Resident Benefit','Paired resident effects','net_added_loss_lower;net_added_loss_upper','Net avoided bounds require negation and endpoint reversal;already implemented effect summary retained'),
 ('New Resident Access Loss','Paired resident effects','new_loss_lower;new_loss_upper','Paired adverse stock interval,kept separate'),
 ('Outside Corridor Depth Increase','Transfer screen','outside_corridor_cell_area_head_increase_ge0_1m_km2','Native cell screening area,not observed flood footprint'),
 ('Daytime Population','No qualified constructed output','not constructed','Metadata or employment counts alone do not establish event-time presence'),
 ('Urban Function Access','No qualified constructed output','not constructed','Needs defined services and operating/access evidence')]
 save('readable_variable_mapping.csv',[dict(readable_name=a,source_role=b,stored_fields=c,interpretation=d,final_variable_selected=False)for a,b,c,d in mapping])
 # Preserve raw lexical outputs and document parser limits rather than inventing automated certainty.
 for name in ['dataset_availability.csv','variable_inventory.csv']:(O/name).write_bytes((O/'automatic'/name).read_bytes())
 save('evidence_inventory.csv',evidence);dump(O/'source_bindings.json',bindings)
 counts=pd.Series([a['status']for a in assessments]).value_counts().to_dict();cc=pd.Series([r[2]for r in claims]).value_counts().to_dict()
 lines=['# Current research feasibility check','','The current applied objective and four supporting questions were reviewed against the22table briefing,registered outputs and numerical/transport/population gates. This is a scoped evidence review,not a new empirical validation or data transformation.','','## Decision-facing result','','All five question chains are partly-testable as conditional scenario analysis. Six specific modeled consequences have current outputs; seven broader claims remain not-yet-testable under the available contract. This does not imply five independent hypotheses are proven or every requested human consequence is available.','','## Supported conditional core','','- Coverage-qualified road interruption and model geometry/stage response.','- Represented vehicle request access and restricted common-OD model cost.','- Nighttime resident-stock access intervals on complete baseline attachment.','- Frozen physical intervention contrasts and outside-corridor adverse changes.','','## Material gaps','','-277.920km of three-ward road length has unknown/unsupported exposure.','-41308residents remain unresolved under the10mattachment rule;they are not assigned safe or affected status.','-Actual congestion,daytime occupancy,unique stranded persons,operating services and complete multimodal isolation lack qualified outcomes.','-Observed flood improvement and engineering effectiveness are not established by numerical gates.','-The physical comparison has two rainfall parents and selected locations;it does not support universal intervention rankings.','','## Automatic-screen boundary','','The bundled lexical scan read22tables and351table-field occurrences and detected the central question only,because its default parser does not extract supporting questions from Markdown table rows. Its original output is retained in automatic/. This supplement directly extracts all four existing table questions without editing AnaSOP and assesses declared concepts against actual evidence. Lexical matches are not scientific validity.','','## Next step','','Proceed to data-preprocessing to confirm the operational variables,units,denominators,coverage masks and publication-final status. Reuse completed artifacts. Missing broader outcomes should be explicitly kept pending;any revision of questions,scope,assumptions or new experiment requires the corresponding later human decision. No objective is dropped here.','','AnaSOP,scientific source tables,original masks and outputs remain unchanged. No new simulations,training,time-series plots or sealed-test access.']
 (O/'README.md').write_text('\n'.join(lines)+'\n')
 def table(d):return '<div class="scroll">'+d.to_html(index=False,border=0,escape=True)+'</div>'
 overview=pd.DataFrame([dict(研究链条='主问题',状态='部分可行',当前支持='条件情景下的耦合影响比较',主要缺口='现实验证与更广人群／功能后果')]+[dict(研究链条=label,状态='部分可行',当前支持=cond,主要缺口=gap)for label,cond,gap in [
 ('道路中断','已有合格道路长度、时序及几何／水位对照','24.9%道路长度暴露未知；不证明真实浸水准确性'),
 ('车辆与交通','已有请求守恒、可达性和同一OD成本','需求／容量不完整；非实测排队'),
 ('人口与功能','已有夜间常住人口存量区间','4.13万人接入未解决；昼间、受困和功能缺口'),
 ('措施与风险转移','已有12组新模拟及配对收益／新增损失','仅两场雨及固定选址；实际工程效果未验证')]])
 page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区研究可行性核查</title><style>body{font:16px/1.8 system-ui;color:#203f4c;background:#edf2f5}main{max-width:1180px;margin:24px auto;padding:28px;background:white}h1{font-size:28px}h2{font-size:22px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd;min-width:110px}.note{background:#fff0d9;padding:16px}</style><main><h1>三区研究：哪些问题现在可以回答？</h1><p class="note">可以继续现有条件情景研究。数值与接口通过并不等于真实灾害／交通／人群后果已验证；不因更广目标有缺口而丢弃已合格结果。</p>',table(overview),'<h2>逐项主张</h2>',table(pd.DataFrame([dict(question_id=q,claim=claim,status=status,evidence=e,limit=limit)for q,claim,status,e,limit in claims])),'<h2>核验依据</h2>',table(pd.DataFrame(evidence)),'<h2>下一步</h2><p>使用data-preprocessing确认论文核心变量、单位、分母、掩膜、区间及最终使用状态，复用已有结果。昼间人口等缺口保留待定，不自动改研究范围或追加模拟。</p><p>自动字段匹配只检出主问题；已另行核读原有四个表格子问题。22张表共351个表内字段项，不代表351个最终变量。</p><ul>']
 for name in ['question_feasibility.csv','claim_feasibility.csv','dataset_availability.csv','variable_inventory.csv','readable_variable_mapping.csv','case_admission_inventory.csv','evidence_inventory.csv','source_bindings.json','validation.json']:page.append(f'<li><a href="{name}">{name}</a></li>')
 page.append('</ul></main></html>');(O/'review.html').write_text(''.join(page))
 assert sha(doc)==before['anasop_sha256']
 assert all(sha(ROOT/n)==h for n,h in bindings.items())
 dump(O/'validation.json',dict(passed=True,questions_reviewed=5,central_questions=1,supporting_questions=4,question_status_counts=counts,subclaims_reviewed=len(claims),subclaim_status_counts=cc,source_tables=22,table_field_occurrences=351,case_admissions_checked=len(cases),evidence_receipts_checked=len(evidence),original_anasop_unchanged=True,source_files_unchanged=True,no_variable_selection=True,no_data_transformation=True,no_new_simulations=True,no_new_training=True,no_new_time_series_figures=True,no_sealed_tests=True,next_skill='data-preprocessing',script_sha256=sha(Path(__file__))))
 print((O/'validation.json').read_text())
if __name__=='__main__':main()
