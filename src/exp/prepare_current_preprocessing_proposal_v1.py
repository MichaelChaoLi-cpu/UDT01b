"""Prepare a reviewable Phase1-4 proposal;no final selection or data export."""
from pathlib import Path
import csv,hashlib,html,importlib.util,json,re,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'data/exp/data-preprocessing/three_ward_analysis_contract_v1'
BR=ROOT/'data/exp/data-briefing/three_ward_current_research_v1'

ALIASES={
 'case':'Case','event_id':'Rain Parent','stage':'Receiving Water Scenario','measure':'Measure Configuration','frame':'Frame','elapsed_s':'Elapsed Time','bound':'Road Depth Bound','scope':'Reporting Scope','stratum':'Road Stratum',
 'open_km':'Open Road Length','slow_km':'Slow Road Length','closed_km':'Closed Road Length','qualified_km':'Qualified Road Length','unknown_km':'Unknown Road Length',
 'peak_closed_km':'Peak Closed Road Length','peak_frame':'Peak Closure Frame','closed_at9h_km':'Final Closed Road Length','closed_km_hours':'Closed Road Duration',
 'buffer_m':'Population Candidate Radius','total_stock':'Resident Population','loss_lower':'Resident Access Loss Lower','loss_upper':'Resident Access Loss Upper','unresolved_stock':'Unresolved Resident Stock',
 'net_added_loss_lower':'Paired Additional Resident Loss Lower','net_added_loss_upper':'Paired Additional Resident Loss Upper','restored_lower':'Restored Resident Access Lower','restored_upper':'Restored Resident Access Upper','new_loss_lower':'New Resident Access Loss Lower','new_loss_upper':'New Resident Access Loss Upper','net_avoided_loss_lower':'Paired Resident Benefit Lower','net_avoided_loss_upper':'Paired Resident Benefit Upper',
 'original_request_proxy':'Original Vehicle Demand','served_proxy':'Served Vehicle Requests','unreachable_proxy':'Unreachable Vehicle Requests','completion_fraction_original':'Vehicle Completion Fraction','common_survivor_original_request_proxy':'Common OD Vehicle Demand','baseline_reachable_excluded_from_cost_proxy':'Excluded Baseline Reachable Requests','common_survivor_cost_index':'Common OD Cost','total_network_cost':'Total Network Cost',
 'outside_corridor_cell_area_head_increase_ge0_1m_km2':'Outside Corridor Depth Increase Area','outside_corridor_cell_area_head_decrease_ge0_1m_km2':'Outside Corridor Depth Decrease Area','native_cell_area_not_measured_flood_area':'Native Cell Screen Flag',
 'ever_increased_cell_area_km2':'Any Time Depth Increase Area','ever_decreased_cell_area_km2':'Any Time Depth Decrease Area','center_assigned_10m_cell_area':'Cell Center Assignment Flag',
 'block_id':'Population Unit','ward':'Ward','town_id':'Town','name':'Statistical Unit Name','official_2020_residents':'Resident Population','count_role':'Population Count Role','within_block_allocation_class':'Population Allocation Class',
 'start_elapsed_s':'Interval Start','end_elapsed_s':'Interval End','source_area_mean_mm_h':'Mean Rainfall Rate'}
EXPORTS={
 'Road trajectories':'three_ward_barrier_drainage_v1/road_trajectories.csv',
 'Road summaries':'three_ward_barrier_drainage_v1/road_summary.csv',
 'Resident trajectories':'three_ward_barrier_drainage_v1/population_trajectories.csv',
 'Paired resident trajectories':'three_ward_barrier_drainage_v1/paired_population_effects.csv',
 'Fixed-time paired resident benefits':'three_ward_barrier_drainage_v1/effect_summary/paired_population_at_reference_time.csv',
 'Traffic states':'three_ward_barrier_drainage_v1/traffic/physical_traffic_outcomes.csv',
 'Instantaneous transfer screening':'three_ward_barrier_drainage_v1/neighbor_transfer_screen.csv',
 'Any-time transfer locations':'three_ward_barrier_drainage_v1/effect_summary/risk_transfer_ward_locations.csv',
 'Official resident baseline':'three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv',
 'Five-parent rain intervals':'three_ward_five_event_temporal_analysis_v1/rainfall_trajectories.csv'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def main():
 O.mkdir(parents=True,exist_ok=True);doc=ROOT/'docs/AnaSOP.md';before=sha(doc)
 source_inventory=pd.read_csv(BR/'tables/dataset_inventory.csv');paths=[ROOT/n for n in source_inventory.source_file]
 spec=importlib.util.spec_from_file_location('preprocess_skill',ROOT/'.codex/skills/data-preprocessing/scripts/prepare_variable_list.py');skill=importlib.util.module_from_spec(spec);spec.loader.exec_module(skill)
 skill.OUTPUT_DIR=str(O.relative_to(ROOT));skill.FEASIBILITY_DIR='data/exp/feasibility-check/three_ward_current_research_v1';skill.discover_data_files=lambda root:paths
 # The original inventory samples columns;attach known full-table missingness separately below.
 old_argv=sys.argv;sys.argv=['prepare_variable_list.py','--root',str(ROOT),'--pretty']
 try:skill.main()
 finally:sys.argv=old_argv
 variable_file=O/'variable_list.csv';v=pd.read_csv(variable_file,keep_default_na=False)
 full=pd.read_csv(BR/'tables/variable_summary.csv');lookup={(r.source_file,r.variable):r for r in full.itertuples()}
 for i,r in v.iterrows():
  x=lookup[(r.source_dataset,r.original_name)];v.at[i,'null_pct']=100*x.missing_pct;v.at[i,'non_null_count']=x.non_null;v.at[i,'feasibility_status']='not selected;see concept/evidence mapping'
  if r.original_name in ALIASES:v.at[i,'readable_name']=ALIASES[r.original_name];v.at[i,'full_name']=ALIASES[r.original_name]
 v.to_csv(variable_file,index=False)
 v.to_csv(O/'full_field_review.csv',index=False)
 # Regenerate grouped README with the full-table null percentage and provisional names.
 grouped={k:g.to_dict('records') for k,g in v.groupby('source_dataset',sort=False)};skill.write_readme(O,grouped,[])
 section=doc.read_text().split('## 4. Variable Construction  /  Key Variables\n',1)[1].split('## 5.',1)[0]
 concepts=[]
 for line in section.splitlines():
  if line.startswith('| ')and not line.startswith('| variable_name'):
   z=[a.strip() for a in line.strip('|').split('|')]
   if len(z)==6:concepts.append(dict(readable_name=z[0],full_name=z[1],role=z[2],definition=z[3],construction=z[4],existing_final=z[5],recommendation='quality_control_only' if z[0]=='Numerical Admission' else 'pending_construct' if z[0]in['Daytime Population','Urban Function Access'] else 'main_analysis',human_confirmed=False))
 assert len(concepts)==27 and sum(x['recommendation']=='main_analysis'for x in concepts)==24
 pd.DataFrame(concepts).to_csv(O/'concept_selection_proposal.csv',index=False)
 sources={str(doc.relative_to(ROOT)):before};proposed={};mappings=[]
 for label,n in EXPORTS.items():
  p=ROOT/'data/processed'/n;d=pd.read_csv(p,low_memory=False);sources[str(p.relative_to(ROOT))]=sha(p);assert set(d.columns)<=set(ALIASES),label
  variables=[]
  for c in d:
   variables.append(dict(original_name=c,readable_name=ALIASES[c],full_name=ALIASES[c],is_final_variable='pending',preprocessing=[],confirmation_required=True))
   mappings.append(dict(dataset=label,source_dataset=str(p.relative_to(ROOT)),original_name=c,proposed_readable_name=ALIASES[c],dtype=str(d[c].dtype),null_pct=float(d[c].isna().mean()*100),sample_values='; '.join(str(x)[:50] for x in d[c].dropna().head(3)),status='awaiting human selection/name/rule confirmation'))
  names=[a['readable_name'].strip().casefold()for a in variables];assert len(names)==len(set(names))
  assert all(re.fullmatch(r'[A-Za-z0-9 .,_/%()+-]+',a['readable_name'])for a in variables)
  stem=p.stem;slug=re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_')
  proposed[str(p.relative_to(ROOT))]=dict(label=label,rows=len(d),fields=len(d.columns),output=f'data/processed/three_ward_analysis_contract_v1/{slug}_preprocessed.parquet',script=f'src/preprocessing/preprocess_transport_{slug}.py',variables=variables)
 pd.DataFrame(mappings).to_csv(O/'export_field_proposal.csv',index=False)
 rules=[
  ('Export scope','Reuse existing approved model products;select and export readable columns only.'),
  ('Missingness','Keep nulls,unknown support and censored intervals;no mean/median/zero imputation.'),
  ('Extremes','Keep extreme values and negative net benefits;no clipping,standardization or log transformation.'),
  ('Road rules','Preserve existing0.10m slow and0.30m closure assumptions,quality masks,source-road clipping and unknown branches.'),
  ('Population rules','Preserve496345stock,0/10/25m candidate branches,unresolved stock and paired intervals;do not sum snapshots as unique people.'),
  ('Benefit sign','Existing net-added-loss fields remain additional loss;only existing net-avoided fields are named benefit.No sign change is silently introduced.'),
  ('Time','Preserve original seconds,frames and frozen same-time comparisons;no new resampling or midpoint labels.'),
  ('Traffic','Keep represented vehicle proxies and common-survivorOD denominator;model cost is not actual delay.'),
  ('Risk transfer','Keep instantaneous and any-time cell screens separate;retain increases and new access losses.'),
  ('Native inputs','Keep raster,CityGML,NetCDF and graph arrays in native/versioned form;reference them as inputs,do not fabricate flat measured attributes.'),
  ('Identifiers','Keep case,unit,ward and parent identifiers as identifiers,not predictors;preserve text semantics during export.'),
  ('Not available','Daytime Population and Urban Function Access stay pending;do not create substitutes.')]
 pd.DataFrame(rules,columns=['topic','proposed_rule']).to_csv(O/'rule_proposal.csv',index=False)
 dump(O/'proposed_decisions.json',dict(status='awaiting_human_confirmation',not_an_executable_final_decision=True,recommended_main_concepts=[x['readable_name']for x in concepts if x['recommendation']=='main_analysis'],quality_control_only=['Numerical Admission'],pending_constructs=['Daytime Population','Urban Function Access'],datasets=proposed,extra_transformations_proposed=False))
 dump(O/'source_bindings.json',sources)
 def table(d):return'<div class="scroll">'+d.to_html(index=False,border=0,escape=True,na_rep='—',float_format=lambda x:f'{x:,.3f}')+'</div>'
 page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分析变量与导出方案</title><style>body{font:16px/1.8 system-ui;background:#edf2f5;color:#23414d}main{max-width:1180px;margin:24px auto;padding:28px;background:white}h1{font-size:28px}h2{font-size:22px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;min-width:100px}.note{background:#fff0d9;padding:16px}</style><main><h1>正式分析变量、英文名和数据导出：待确认方案</h1><p class="note">已完成变量盘点，尚未选定最终变量、改写AnaSOP或生成新的分析Parquet。建议采用已有24项概念，数值准入仅作质控，昼间人口／城市功能继续待定。该选择需要研究者确认。</p><h2>27项现有概念的建议归类</h2>',table(pd.DataFrame(concepts)),'<h2>保留现有规则</h2>',table(pd.DataFrame(rules,columns=['主题','建议规则'])),'<h2>10张分析表的具体英文列名</h2><p>可以逐表核对原字段、建议英文名、类型、完整表空值率和样本。旧文件保持原样，确认后仅新增可复现的Parquet导出。</p>']
 for label,g in pd.DataFrame(mappings).groupby('dataset',sort=False):page.extend([f'<h3>{html.escape(label)}</h3>',table(g.drop(columns=['dataset','source_dataset']))])
 page.extend(['<h2>所有已盘点字段</h2><p>22张表、351个表内字段项的完整清单；标识符、诊断量与未实施构念不会自动升级成主要变量。</p>','<ul>'])
 for n in ['variable_list.csv','full_field_review.csv','concept_selection_proposal.csv','export_field_proposal.csv','rule_proposal.csv','proposed_decisions.json','source_bindings.json','preparation_validation.json']:page.append(f'<li><a href="{n}">{n}</a></li>')
 page.append('</ul></main></html>');(O/'review.html').write_text(''.join(page))
 assert sha(doc)==before and all(sha(ROOT/n)==h for n,h in sources.items())
 dump(O/'preparation_validation.json',dict(passed=True,phase='variable_discovery_and_reviewable_proposal',source_tables=22,table_field_occurrences=len(v),concepts=27,recommended_main_concepts=24,quality_control_only=1,pending_constructs=2,proposed_export_tables=len(proposed),proposed_export_fields=len(mappings),all_readable_names_valid=True,final_decisions_written=False,parquet_exports_created=False,anaSOP_unchanged=True,source_files_unchanged=True,next_step='Human confirmation of inclusion,names and no-extra-transformation rules',dependency_gap=['pyarrow','openpyxl'],project_dependency_registration_needed=['pandas','pyarrow','openpyxl']))
 print((O/'preparation_validation.json').read_text())
if __name__=='__main__':main()
