"""Confirm and export the human-approved transport analysis table contract.
Lossless typed CSV->readable Parquet exports;original scientific products unchanged.
"""
from pathlib import Path
import csv,hashlib,html,importlib.util,json,re,subprocess,sys
import pandas as pd
import pyarrow,openpyxl
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'data/exp/data-preprocessing/three_ward_analysis_contract_v1'
DEST=ROOT/'data/processed/three_ward_analysis_contract_v1'
DEC=ROOT/'data/exp/data-preprocessing/decisions.json'
IDENTIFIERS={'case','event_id','stage','measure','bound','scope','stratum','block_id','ward','town_id','count_role','within_block_allocation_class','name'}
REFERENCE={'case','event_id','stage','measure','frame','bound','scope','stratum','peak_frame','elapsed_s','buffer_m','block_id','ward','town_id','name','count_role','within_block_allocation_class','native_cell_area_not_measured_flood_area','center_assigned_10m_cell_area','net_added_loss_lower','net_added_loss_upper'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def unit(c):
 if c in IDENTIFIERS or c in ['frame','peak_frame']:return 'identifier or category' if c in IDENTIFIERS else 'frame index'
 if c in ['elapsed_s','start_elapsed_s','end_elapsed_s']:return 'elapsed seconds'
 if c=='buffer_m':return 'meters'
 if c=='source_area_mean_mm_h':return 'millimeters per hour'
 if 'km2' in c:return 'square kilometers of model-cell screen'
 if c=='closed_km_hours':return 'kilometer hours'
 if c.endswith('_km'):return 'kilometers'
 if c in ['total_stock','official_2020_residents','loss_lower','loss_upper','unresolved_stock','net_added_loss_lower','net_added_loss_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper','net_avoided_loss_lower','net_avoided_loss_upper']:return '2020 resident stock'
 if c=='common_survivor_cost_index':return 'dimensionless index;no-measure=100'
 if c=='completion_fraction_original':return 'fraction of original represented requests'
 if c=='total_network_cost':return 'vehicle-request proxy times generalized cost seconds'
 if c.endswith('_proxy'):return 'vehicle-request proxy per reference hour'
 if c in ['native_cell_area_not_measured_flood_area','center_assigned_10m_cell_area']:return 'boolean diagnostic'
 return 'documented in original source contract'
def typed_source(p,variables):
 dtype={v['original_name']:'string' for v in variables if v['original_name']in IDENTIFIERS}
 return pd.read_csv(p,encoding='utf-8-sig',low_memory=False,dtype=dtype)

def main():
 proposal=json.loads((O/'proposed_decisions.json').read_text());assert proposal['status']=='awaiting_human_confirmation'
 bindings=json.loads((O/'source_bindings.json').read_text());assert all(sha(ROOT/n)==h for n,h in bindings.items()),'Inputs changed since review'
 assert not DEC.exists(),'Preserve other confirmed decisions'
 assert not DEST.exists(),'Do not overwrite earlier analysis exports'
 before=(ROOT/'docs/AnaSOP.md').read_text();(O/'AnaSOP_before_confirmation.md').write_text(before)
 selected=proposal['recommended_main_concepts'];assert len(selected)==24
 decisions=dict(status='human_confirmed',decision_record='MILI-D-20261019-003',main_analysis_concepts=selected,quality_control_only=proposal['quality_control_only'],pending_constructs=proposal['pending_constructs'],extra_transformations=False,datasets={})
 for source,d in proposal['datasets'].items():
  assert not(ROOT/d['script']).exists(),'Preserve existing script'
  d=json.loads(json.dumps(d))
  d['identifier_columns']=[v['original_name']for v in d['variables']if v['original_name']in IDENTIFIERS]
  for v in d['variables']:
   v.pop('confirmation_required',None);v['human_confirmed']=True;v['is_final_variable']='no' if v['original_name']in REFERENCE else 'yes';v['preprocessing']=[];v['unit']=unit(v['original_name']);v['retention_role']='identifier or audit/reference' if v['is_final_variable']=='no'else'confirmed input,outcome or declared-support quantity'
  decisions['datasets'][source]=d
 dump(DEC,decisions);dump(O/'decisions.json',decisions)
 spec=importlib.util.spec_from_file_location('export_skill',ROOT/'.codex/skills/data-preprocessing/scripts/generate_preprocessing_script.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original=m.read_code_for_file
 def typed_reader(source):
  code=original(source)
  if Path(source).suffix.lower()=='.csv':
   dtype={c:'string' for c in decisions['datasets'][source]['identifier_columns']}
   code=f'df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False, dtype={dtype!r})'
  return code
 m.read_code_for_file=typed_reader
 oldargv=sys.argv;sys.argv=['generate_preprocessing_script.py','--root',str(ROOT),'--decisions',str(DEC.relative_to(ROOT)),'--pretty']
 try:m.main()
 finally:sys.argv=oldargv
 outputs=[];dictionary=[]
 for source,d in decisions['datasets'].items():
  p=ROOT/source;target=ROOT/d['output'];script=ROOT/d['script']
  assert sha(p)==bindings[source]
  expected=typed_source(p,d['variables']);rename={v['original_name']:v['readable_name']for v in d['variables']};expected=expected[list(rename)].rename(columns=rename)
  with(O/(script.stem+'.log')).open('w')as out:subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
  actual=pd.read_parquet(target)
  pd.testing.assert_frame_equal(expected,actual,check_exact=True,check_dtype=True)
  assert len(actual)==d['rows']and len(actual.columns)==d['fields']
  assert all(c.isascii()for c in actual)and not actual.columns.duplicated().any()
  for v in d['variables']:
   a=actual[v['readable_name']];dictionary.append(dict(dataset=d['label'],source_dataset=source,processed_dataset=d['output'],original_name=v['original_name'],readable_name=v['readable_name'],full_name=v['full_name'],unit=v['unit'],dtype=str(a.dtype),rows=len(a),nulls=int(a.isna().sum()),is_final_variable=v['is_final_variable'],retention_role=v['retention_role'],transformation='Column selection and readable rename only;identifiers read as text;values and nulls unchanged.'))
  outputs.append(dict(label=d['label'],source=source,output=d['output'],script=d['script'],rows=len(actual),columns=len(actual.columns),source_sha256=sha(p),output_sha256=sha(target),script_sha256=sha(script),exact_values_and_nulls=True))
  print('LOSSLESS EXPORT QUALIFIED',d['label'],len(actual),len(actual.columns),flush=True)
 # Scientific invariants on the exported interfaces,without new simulations or model fits.
 by={d['label']:pd.read_parquet(ROOT/d['output'])for d in decisions['datasets'].values()}
 r=by['Road trajectories'];assert (abs(r['Open Road Length']+r['Slow Road Length']+r['Closed Road Length']-r['Qualified Road Length'])<1e-7).all()
 p=by['Resident trajectories'];assert p.loc[p['Reporting Scope'].eq('Three wards'),'Resident Population'].eq(496345).all();assert(p['Resident Access Loss Lower']<=p['Resident Access Loss Upper']).all();assert p.loc[p['Reporting Scope'].eq('Three wards')&p['Population Candidate Radius'].eq(10),'Unresolved Resident Stock'].eq(41308).all()
 fixed=by['Fixed-time paired resident benefits'];assert(fixed['Paired Resident Benefit Lower']==-fixed['Paired Additional Resident Loss Upper']).all();assert(fixed['Paired Resident Benefit Upper']==-fixed['Paired Additional Resident Loss Lower']).all();assert fixed['Paired Resident Benefit Lower'].lt(0).any(),'Retain adverse effects'
 tr=by['Traffic states'];assert(abs(tr['Original Vehicle Demand']-tr['Served Vehicle Requests']-tr['Unreachable Vehicle Requests'])<1e-6).all();assert tr.groupby(['Rain Parent','Receiving Water Scenario'])['Common OD Vehicle Demand'].nunique().eq(1).all()
 official=by['Official resident baseline'];assert official['Resident Population'].sum()==496345 and official['Population Unit'].is_unique
 pd.DataFrame(outputs).to_csv(O/'export_validation.csv',index=False);pd.DataFrame(dictionary).to_csv(O/'data_dictionary.csv',index=False)
 pd.DataFrame(dictionary).to_csv(DEST/'data_dictionary.csv',index=False)
 # Native hydraulic,raster and graph inputs remain versioned inputs,not fabricated scalar tables.
 native={
 'Rainfall Forcing':['data/processed/three_ward_barrier_drainage_v1/run_manifest.json'],
 'Receiving Water Stage':['data/processed/three_ward_barrier_drainage_v1/run_manifest.json'],
 'Terrain':['data/processed/three_ward_barrier_drainage_v1/run_manifest.json'],
 'Urban Geometry':['data/processed/three_ward_barrier_drainage_v1/solver_binding.json','data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz'],
 'Surface Water State':['data/processed/three_ward_barrier_drainage_v1/run_manifest.json','data/exp/estimation/three_ward_barrier_drainage_v1/delivery_validation.json'],
 'Road Depth':['data/processed/three_ward_barrier_drainage_v1/motor_common_support.npz'],
 'Vehicle Network':['data/processed/three_ward_normal_vehicle_repair_v1/turn_router.npz'],
 'Vehicle Demand':['data/processed/three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz','data/processed/three_ward_barrier_drainage_v1/traffic/protocol.json'],
 'Road Capacity':['data/processed/three_ward_demand_protection_feedback_v1/capacity_assumptions.npz'],
 'Reference Travel Cost':['data/processed/three_ward_normal_vehicle_choice_v1/choice_costs.npz'],
 'Vehicle Access':['data/processed/three_ward_night_population_transport_v1/normal_node_access.npz'],
 'Population Access Candidates':['data/processed/three_ward_night_population_transport_v1/block_access_support.npz'],
 'Measure Configuration':['data/processed/three_ward_barrier_drainage_v1/protocol.json'],
 'Numerical Admission':['data/exp/estimation/three_ward_barrier_drainage_v1/delivery_validation.json']}
 native_receipt={k:[dict(path=n,sha256=sha(ROOT/n))for n in paths]for k,paths in native.items()}
 dump(DEST/'native_input_contract.json',dict(inputs=native_receipt,meaning='Existing native/versioned input resources and receipts retained;manifest paths locate unchanged case-specific grids,forcing and states.',no_binary_state_reconstruction=True))
 # Only Section4 is owned by this skill.
 start=before.index('## 4. Variable Construction  /  Key Variables\n');end=before.index('## 5. Identification Strategy\n');old4=before[start:end]
 lines=[];yes=0;no=0
 for line in old4.splitlines():
  if line.startswith('| ')and not line.startswith('| variable_name'):
   cells=[c.strip()for c in line.strip('|').split('|')]
   if len(cells)==6:
    cells[-1]='yes'if cells[0]in selected else'no';yes+=cells[-1]=='yes';no+=cells[-1]=='no';line='| '+' | '.join(cells)+' |'
  lines.append(line)
 new4='\n'.join(lines)+'\n\n';oldintro='The table documents article-facing names for already implemented quantities. It does not rename stored data, reprocess inputs, change thresholds or make a new final-variable selection. All entries remain unfinalized for publication; the existing experimental definitions continue to bind completed comparisons.'
 intro='The user confirmed twenty-four existing input and outcome concepts for the current main analysis. Numerical Admission remains quality control; Daytime Population and Urban Function Access remain unconstructed and are not final variables. Final status here binds the current analytical contract; manuscript presentation and final claims remain subject to output review. Existing experimental thresholds,source values,quality support and paired interval rules are unchanged.'
 assert oldintro in new4;new4=new4.replace(oldintro,intro)
 new4+='### Confirmed analysis-table conventions\n\nTen lossless analysis tables preserve the confirmed English column names,units,case keys,original row order and nulls. Their 109 column occurrences include repeated identifiers,interval endpoints,coverage quantities and summaries;they are not 109 independent final research constructs. Identifiers remain text or declared categories and are not physical predictors.\n\nDerived peak,final,lower,upper and paired-change fields remain variants of the named constructs. In particular,additional resident loss and avoided resident loss retain opposite signs;existing paired bounds are not converted into midpoint targets. Nulls,unknown roads,unresolved population and negative benefits are retained without imputation,clipping,resampling or recomputation.\n\nTerrain,three-dimensional city geometry,full hydraulic states,forcing curves and network arrays remain in their existing versioned native formats. Readable tables and a native-input contract reference these resources;the export does not invent flattened measurements or replace the source model. Unit,denominator,support and interpretation metadata accompany each exported field.\n\n'
 assert yes==24 and no==3
 after=before[:start]+new4+before[end:];(ROOT/'docs/AnaSOP.md').write_text(after)
 assert after[:start]==before[:start]and after[after.index('## 5. Identification Strategy\n'):]==before[end:]
 assert not re.search('[\u3400-\u9fff]',new4)
 # Confirm concept selection in the review registry;keep the original unconfirmed proposal intact.
 concepts=pd.read_csv(O/'concept_selection_proposal.csv');concepts['human_confirmed']=True;concepts['is_final_variable']=concepts.readable_name.isin(selected).map({True:'yes',False:'no'});concepts.to_csv(O/'confirmed_concepts.csv',index=False)
 v=pd.read_csv(O/'variable_list.csv',keep_default_na=False)
 index={(row['source_dataset'],row['original_name']):row for row in dictionary}
 for i,row in v.iterrows():
  key=(row.source_dataset,row.original_name)
  if key in index:
   z=index[key];v.at[i,'readable_name']=z['readable_name'];v.at[i,'full_name']=z['full_name'];v.at[i,'is_final_variable']=z['is_final_variable']
 v.to_csv(O/'variable_list.csv',index=False)
 globalvars=ROOT/'data/exp/data-preprocessing/variable_list.csv'
 if globalvars.exists():(O/'previous_root_variable_list.csv').write_bytes(globalvars.read_bytes())
 globalvars.write_bytes((O/'variable_list.csv').read_bytes())
 assert all(sha(ROOT/n)==h for n,h in bindings.items()if n!='docs/AnaSOP.md')
 assert all(sha(ROOT/n)==h for data in native_receipt.values()for n,h in[(item['path'],item['sha256'])for item in data])
 dump(DEST/'export_manifest.json',dict(decision_record='MILI-D-20261019-003',decision_sha256=sha(DEC),outputs=outputs,native_input_contract_sha256=sha(DEST/'native_input_contract.json')))
 dump(O/'validation.json',dict(passed=True,decision_record='MILI-D-20261019-003',confirmed_main_concepts=24,quality_control_only=1,pending_constructs=2,parquet_tables=len(outputs),column_occurrences=sum(x['columns']for x in outputs),rows_by_table={x['label']:x['rows']for x in outputs},all_values_nulls_row_order_and_logical_dtypes_exact=True,identifier_semantics_retained=True,population_total_conserved=True,paired_sign_and_interval_rules_preserved=True,negative_effects_retained=True,traffic_demand_and_common_cohort_preserved=True,source_scientific_products_unchanged=True,only_anasop_section4_updated=True,new_simulations=0,new_training=0,new_time_series_figures=0,pandas_version=pd.__version__,pyarrow_version=pyarrow.__version__,openpyxl_version=openpyxl.__version__,script_sha256=sha(Path(__file__)),anaSOP_sha256=sha(ROOT/'docs/AnaSOP.md'),next_skill='figure-table-planning'))
 def table(d):return'<div class="scroll">'+d.to_html(index=False,border=0,escape=True,na_rep='—')+'</div>'
 page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>正式分析数据合同已完成</title><style>body{font:16px/1.8 system-ui;background:#edf2f5;color:#24414e}main{max-width:1180px;margin:24px auto;background:white;padding:28px}h1{font-size:28px}h2{font-size:22px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}.note{background:#e4f0e9;padding:16px}</style><main><h1>分析变量与可复现导出：已完成</h1><p class="note">采用已确认的24项主分析概念；数值准入仅作质控，昼间人口／城市功能保留待定。10张Parquet、109个列项均与源数据逐值、空值、行序和逻辑类型一致。原模拟和源表保持原样，仅更新AnaSOP第4节。</p><h2>导出核验</h2>',table(pd.DataFrame(outputs).drop(columns=['source_sha256','output_sha256','script_sha256'])),'<h2>确认的概念</h2>',table(concepts[['readable_name','role','recommendation','is_final_variable']]),'<h2>保留规则与解释</h2><p>不填补、不截尾、不重采样、不重算；保留未知道路、未解决人口、负收益、质量掩膜和区间。附加人口损失与避免损失不混名。交通结果是模型成本，人口结果是2020存量接入区间。原生地形、三维城市与图数据继续通过版本合同引用。</p><h2>后续</h2><p>figure-table-planning：确认主图、主表与支持的研究问题。当前不自动追加实验或训练。</p><ul>']
 for name in ['decisions.json','confirmed_concepts.csv','data_dictionary.csv','export_validation.csv','validation.json']:page.append(f'<li><a href="{name}">{name}</a></li>')
 page.append('</ul></main></html>');(O/'review.html').write_text(''.join(page))
 (O/'README.md').write_text('# Confirmed transport-analysis preprocessing\n\nHuman decision:MILI-D-20261019-003.24confirmed analytical concepts,one quality-only diagnostic and two pending constructs.10lossless Parquet tables,109column occurrences.Original model products are unchanged.Only AnaSOP Section4is updated.No new imputation,clipping,resampling,simulation,training or time-series visualization.\n\nSee validation.json,export_validation.csv,data_dictionary.csv,decisions.json and the review page.Each table has its generated local Python export script;run using the project uv environment.\n\nNext skill:figure-table-planning.\n')
 print((O/'validation.json').read_text(),flush=True)
if __name__=='__main__':main()
