"""Current-scope data-briefing using existing source tables;no analytical transforms.
Runs the bundled skill with boolean-safe descriptive profiling,retains technical
lineage separately,and updates only AnaSOP Section3. Optional time plots need consent.
"""
import argparse,csv,datetime as dt,hashlib,html,importlib.util,json,os,re,sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'data/exp/data-briefing/three_ward_current_research_v1'
P=ROOT/'data/processed'
FILES={
 'Official resident units':'three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv',
 'Population allocation admission':'three_ward_population_block_stability_v1/population_input_admission_list.csv',
 'Census mesh 125 m':'three_ward_population_mesh_audit_v1/population_125m_source_rows.csv',
 'Census mesh 250 m':'three_ward_population_mesh_audit_v1/population_250m_source_rows.csv',
 'Building allocation attributes':'three_ward_population_mesh_audit_v1/building_allocation_readiness.csv',
 'Native road inventory':'three_ward_motor_flood_access_v1/road_inventory.csv',
 'Historical case inventory':'three_ward_five_event_temporal_analysis_v1/input_inventory.csv',
 'Historical rainfall intervals':'three_ward_five_event_temporal_analysis_v1/rainfall_trajectories.csv',
 'Rainfall acquisition snapshot':'three_ward_population_additional_radar_v1/five_event_rainfall_comparison.csv',
 'Frozen reference states':'three_ward_five_event_temporal_analysis_v1/reference_states.csv',
 'Conditional demand additions':'three_ward_vehicle_demand_recovery_v1/aggregated_endpoint_demands.csv',
 'Demand source conservation':'three_ward_vehicle_demand_recovery_v1/source_control_conservation.csv',
 'Normal-flow development comparison':'three_ward_vehicle_demand_recovery_v1/primary_small12h_metrics.csv',
 'Physical road trajectories':'three_ward_barrier_drainage_v1/road_trajectories.csv',
 'Physical road summary':'three_ward_barrier_drainage_v1/road_summary.csv',
 'Physical resident trajectories':'three_ward_barrier_drainage_v1/population_trajectories.csv',
 'Paired resident effects':'three_ward_barrier_drainage_v1/paired_population_effects.csv',
 'Outside-corridor transfer screen':'three_ward_barrier_drainage_v1/neighbor_transfer_screen.csv',
 'Physical traffic outcomes':'three_ward_barrier_drainage_v1/traffic/physical_traffic_outcomes.csv',
 'Pump scenario sites':'three_ward_barrier_drainage_v1/measures/pump_sites.csv',
 'Barrier scenario faces':'three_ward_barrier_drainage_v1/measures/barrier_faces.csv',
 'Transfer location summaries':'three_ward_barrier_drainage_v1/effect_summary/risk_transfer_ward_locations.csv',
}
META={
 'Elevation coverage audit':'data/exp/data-briefing/remote_sensing_fallback/terrain_fallback_coverage.csv',
 'Census generation and reference date':'data/exp/data-briefing/three_ward_population_mesh_audit_v1/summary.json',
 'Physical-measure protocol':'data/processed/three_ward_barrier_drainage_v1/protocol.json',
 'Physical-measure delivery':'data/exp/estimation/three_ward_barrier_drainage_v1/delivery_validation.json',
 'Historical consequence delivery':'data/exp/estimation/three_ward_five_event_population_validation_v1/delivery_validation.json',
 'Whole-event delivery':'data/exp/estimation/three_ward_five_event_temporal_analysis_v1/delivery_validation.json',
 'Effect summary checks':'data/exp/estimation/three_ward_barrier_drainage_v1/effect_summary/validation.json',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
def save(name,rows):pd.DataFrame(rows).to_csv(O/'tables'/name,index=False)
def read(label):return pd.read_csv(P/FILES[label],low_memory=False)
def sections(s):
 a=re.split(r'(?m)^(?=## [1-8]\. )',s);return a[0],{int(t.split('.')[0][3:]):t for t in a[1:]}
def run_skill():
 spec=importlib.util.spec_from_file_location('brief_skill',ROOT/'.codex/skills/data-briefing/scripts/create_data_briefing.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 native=m.summarize_dataset
 def boolean_safe(dataset,path,df):
  cols=[c for c in df if pd.api.types.is_bool_dtype(df[c])]
  vr,sr,tc=native(dataset,path,df.drop(columns=cols))
  for c in cols:
   a=df[c];n=len(a);vr.append(dict(dataset=dataset,source_file=str(path),variable=c,dtype=str(a.dtype),rows=n,non_null=int(a.notna().sum()),missing=int(a.isna().sum()),missing_pct=float(a.isna().mean()),unique=int(a.nunique()),role='boolean_diagnostic',true_count=int(a.eq(True).sum()),false_count=int(a.eq(False).sum())))
   sr.append(dict(dataset=dataset,source_file=str(path),variable=c,candidate_status='diagnostic_only',reason='Boolean diagnostic retained as categorical counts;no numeric quantile or predictor selection.',missing_pct=float(a.isna().mean()),unique=int(a.nunique())))
  return vr,sr,tc
 m.summarize_dataset=boolean_safe
 # Explicit elapsed clocks avoid interpreting row numbers as dates.
 def time_columns(df):return [c for c in df if c in ['elapsed_s','actual_elapsed_s','analysis_elapsed_s','start_elapsed_s','end_elapsed_s','actual_start_s','actual_end_s','window_start_jst','creationDate','surveyYear']]
 m.infer_time_columns=time_columns
 numeric=m.plot_numeric_distribution;categorical=m.plot_categorical_distribution
 allowed={'official_2020_residents','base_area_m2','area_inside_three_wards_m2','length_m','source_area_mean_mm_h','nine_hour_mm','max_rolling_hour_mm','prior_rain_outside_window_mm','daily','hour17','wape','prediction_observation_ratio','closed_km','peak_closed_km','closed_km_hours','loss_lower','loss_upper','unresolved_stock','net_added_loss_lower','net_added_loss_upper','common_survivor_cost_index','unreachable_proxy'}
 aliases={m.safe_name(str((P/n).relative_to(ROOT)).rsplit('.',1)[0]):label for label,n in FILES.items()}
 def friendly(title):
  dataset,field=title.rsplit(': ',1)
  return aliases.get(dataset,dataset)+'\n'+field.replace('_',' ')
 def numeric_plot(series,output,title):
  return numeric(series,output,friendly(title)) if str(series.name) in allowed and not pd.api.types.is_bool_dtype(series) else False
 def category_plot(series,output,title):
  return categorical(series,output,friendly(title)) if str(series.name) in {'use_group','allocation_input_class','highway','reason','measure'} else False
 m.plot_numeric_distribution=numeric_plot;m.plot_categorical_distribution=category_plot
 args=Namespace(root=str(ROOT),output=str(O.relative_to(ROOT)),inputs=[str((P/n).relative_to(ROOT)) for n in FILES.values()],include_time_series='no',max_distribution_plots=20,max_categorical_levels=50,max_time_columns=3,max_time_series_variables=6)
 result=m.create_briefing(args);dump(O/'skill_execution.json',result);return result

def coverage():
 census=read('Official resident units');pop=read('Physical resident trajectories');road=read('Physical road trajectories');q=road[(road.scope=='Three wards')&(road.stratum=='all')].iloc[0]
 unresolved=pop[pop.scope=='Three wards'].groupby('buffer_m',as_index=False).unresolved_stock.agg(['min','max']).reset_index()
 assert (unresolved['min']==unresolved['max']).all()
 save('unresolved_population_by_radius.csv',unresolved.to_dict('records'))
 for i in unresolved.to_dict('records'):i['fraction_of_total']=i['min']/496345
 known=float(q.qualified_km);unknown=float(q.unknown_km)
 cov=dict(official_units=len(census),official_population=float(census.official_2020_residents.sum()),qualified_road_km=known,unknown_road_km=unknown,road_exposure_known_fraction=known/(known+unknown),unresolved_population_by_radius=unresolved.to_dict('records'))
 blocks=read('Historical case inventory');rain=read('Historical rainfall intervals');cov.update(historical_cases=len(blocks),historical_parents=int(blocks.event_id.nunique()),historical_frames=int(blocks.map_frames.sum()),rain_interval_rows=len(rain),case_frames_each=sorted(blocks.map_frames.unique().tolist()))
 cov['selected_input_road_rows']=len(read('Native road inventory'))
 b=read('Building allocation attributes');cov['building_records']=len(b);cov['unique_building_ids']=int(b.gml_id.nunique());cov['three_ward_building_records']=int(b.intersects_three_wards.sum());cov['three_ward_valid_floor_records']=int(b.loc[b.intersects_three_wards,'floors_valid'].sum());cov['valid_floor_records']=int(b.floors_valid.sum());cov['building_use_counts']={str(k):int(v) for k,v in b.use_group.value_counts(dropna=False).items()}
 save('building_scope_quality.csv',[dict(scope=scope,records=len(g),unique_ids=g.gml_id.nunique(),valid_floor_records=int(g.floors_valid.sum()),unknown_use_records=int(g.use_group.eq('unknown').sum())) for scope,g in [('all_loaded_context',b),('intersects_three_wards',b[b.intersects_three_wards]),('outside_three_wards',b[~b.intersects_three_wards])]])
 # Semantic missingness is an audit only;no sentinel/suppression token is replaced.
 semantic=[]
 for label in ['Building allocation attributes','Census mesh 125 m','Census mesh 250 m']:
  d=read(label)
  for c in d:
   a=d[c];sentinel=int(pd.to_numeric(a,errors='coerce').eq(-9999).sum())
   suppressed=int(a.astype('string').isin(['*','Y']).sum())
   if sentinel or suppressed:semantic.append(dict(dataset=label,variable=c,rows=len(a),literal_missing=int(a.isna().sum()),sentinel_minus9999=sentinel,suppressed_or_transferred_token_count=suppressed,note='Audit only;do not interpret sentinel,suppressed or transferred age cells as zero.'))
 save('semantic_missingness.csv',semantic)
 flags=read('Population allocation admission').groupby('allocation_input_class',dropna=False).agg(units=('unit_id','size'),official_residents=('official_2020_residents','sum')).reset_index();flags.to_csv(O/'tables/allocation_quality_classes.csv',index=False)
 temporal=[]
 for label in ['Historical rainfall intervals','Physical road trajectories','Physical resident trajectories']:
  d=read(label);t='end_elapsed_s' if 'end_elapsed_s' in d else 'elapsed_s';groups=['event_id'] if label=='Historical rainfall intervals' else ['case','bound','scope']+(['stratum'] if 'stratum'in d else ['buffer_m'])
  for keys,g in d.groupby(groups,dropna=False):
   times=np.sort(g[t].unique());gaps=np.diff(times)
   temporal.append(dict(dataset=label,group=str(keys),rows=len(g),unique_times=len(times),start_s=float(times[0]),end_s=float(times[-1]),minimum_gap_s=float(gaps.min()) if len(gaps) else None,maximum_gap_s=float(gaps.max()) if len(gaps) else None,duplicate_key_rows=int(g.duplicated([t]).sum())))
 save('temporal_coverage.csv',temporal)
 old=read('Rainfall acquisition snapshot');delivery=json.loads((ROOT/META['Historical consequence delivery']).read_text());stale=old[~old.hydraulic_result_available]
 save('metadata_snapshot_differences.csv',[dict(dataset='Rainfall acquisition snapshot',event_id=r.event_id,acquisition_snapshot_available=bool(r.hydraulic_result_available),current_case_admission=True,note='Historical acquisition-stage flag predates completed solver delivery;source preserved;use current per-case audit for present admission.') for r in stale.itertuples()])
 cov['stale_acquisition_flags']=len(stale);cov['semantic_missing_fields']=len(semantic)
 assert cov['official_population']==496345 and cov['historical_parents']==5 and cov['historical_cases']==10
 assert all(x['duplicate_key_rows']==0 for x in temporal)
 dump(O/'coverage.json',cov);return cov

def draw_time_plots():
 import matplotlib.pyplot as plt
 out=O/'figures/time-series';out.mkdir(exist_ok=True)
 r=read('Historical rainfall intervals');events=sorted(r.event_id.unique());fig,axs=plt.subplots(5,1,figsize=(10,12),layout='constrained',sharex=True)
 for ax,event in zip(axs,events):
  g=r[r.event_id==event];ax.step(g.end_elapsed_s/3600,g.source_area_mean_mm_h,where='pre',color='#277b95');ax.set(ylabel='mm/h',title=event);ax.grid(alpha=.15)
 axs[-1].set_xlabel('Elapsed hours from registered dry-start window');fig.savefig(out/'rainfall_five_parents.png',dpi=140);plt.close(fig)
 for label,value,filtermask,name,ytitle in [
  ('Physical road trajectories','closed_km',lambda d:(d.scope=='Three wards')&(d.stratum=='all')&(d.bound=='lower'),'physical_closed_roads.png','Qualified closed length (km)'),
  ('Physical resident trajectories','loss_upper',lambda d:(d.scope=='Three wards')&(d.buffer_m==10)&(d.bound=='lower'),'physical_resident_intervals.png','Resident-stock access loss')]:
  d=read(label);d=d[filtermask(d)];fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=dict(none='#657784',barrier='#b57932',drainage='#23897b',combined='#86569a')
  for ax,(event,level) in zip(axs.ravel(),[(e,l) for e in ['E20250911','E20240618'] for l in ['H','L']]):
   for arm,color in colors.items():
    g=d[(d.event_id==event)&(d.stage==level)&(d.measure==arm)].sort_values('frame');x=g.elapsed_s/3600
    if 'loss_lower' in g:ax.fill_between(x,g.loss_lower,g.loss_upper,color=color,alpha=.12);ax.plot(x,g.loss_lower,color=color,label=arm);ax.plot(x,g.loss_upper,color=color,ls='--',lw=.9)
    else:ax.plot(x,g[value],color=color,label=arm)
   ax.set(title=event+'/'+level,xlabel='Elapsed hours',ylabel=ytitle,xlim=[0,9]);ax.legend(fontsize=8);ax.grid(alpha=.15)
  fig.savefig(out/name,dpi=140);plt.close(fig)
 return [str(p.relative_to(O)) for p in sorted(out.glob('*.png'))]

def update_section3(before,result,cov):
 pre,old=sections(before);section=old[3].rstrip()
 inv=pd.read_csv(O/'tables/dataset_inventory.csv');labels={str((P/n).relative_to(ROOT)):label for label,n in FILES.items()}
 section+='\n\n### Current scoped data briefing\n\nThe current briefing directly reads '+str(len(inv))+' source and processed tables, profiling '+str(result['variables_summarized'])+' table-field occurrences; field occurrences are not independent research variables. Native terrain, city-model, raster, hydraulic-array and graph sources are represented through existing provenance and admission audits; this is not an exhaustive rescan of historical simulation logs or raw archives.\n\n| article-facing data component | table rows | fields | observation and scope |\n|---|---|---|---|\n'
 for row in inv.itertuples():
  label=labels[row.source_file];unit='Existing source or scenario table;separate grouping and repeated-state structure retained.'
  if label=='Official resident units':unit='Official statistical units;2020 resident stock,not observed occupancy.'
  if label=='Native road inventory':unit='Source road segments across the wider network;not all have qualified three-ward exposure.'
  if label=='Historical rainfall intervals':unit='Five rain parents by five-minute interval;not independent storms per row.'
  if label=='Historical case inventory':unit='Ten event-stage model cases;five parent rainfall windows.'
  section+=f'| {label} | {row.rows:,} | {row.columns} | {unit} |\n'
 section+='\n### Coverage, missingness and snapshot interpretation\n\n'
 section+=f'- Qualified three-ward road length is {cov["qualified_road_km"]:.3f} km; another {cov["unknown_road_km"]:.3f} km has unknown or unsupported exposure. The qualified share of this clipped road-length denominator is {100*cov["road_exposure_known_fraction"]:.2f} percent. This is not the fraction of all network segments or proof of safety on unknown roads.\n'
 for row in cov['unresolved_population_by_radius']:section+=f'- Under the {int(row["buffer_m"])} m candidate search, unresolved normal-attachment stock is {int(row["min"]):,} residents ({100*row["min"]/496345:.2f} percent of the fixed official total); retain this denominator outside supported loss bounds.\n'
 section+=f'- Building profiling contains {cov["building_records"]:,} unique records across loaded context, of which {cov["three_ward_building_records"]:,} intersect the three wards and {cov["three_ward_valid_floor_records"]:,} of those carry the existing valid-floor flag. Literal nulls and semantic missing codes are reported separately; a numeric height sentinel is not a measured negative height. Suppressed or transferred census attributes are not silently made zero.\n'
 section+=f'- {cov["stale_acquisition_flags"]} acquisition-time rainfall availability flags predate the completed hydraulic deliveries. Their historical values are preserved; current numerical admission comes from versioned per-case and delivery checks. An old input-only flag is not a new solver failure.\n'
 section+='- The five rain parents provide 540 five-minute rainfall intervals and ten no-action case histories with 109 fields each. Physical-measure road and resident histories have the same 109-state window. Timing resolution, dry starts and end-of-window censoring remain material limitations.\n'
 section+='- Statistical distributions describe their stated table units. Repeated stages, measures, depth branches and reporting scopes do not increase event independence or population totals. Identifiers are not selected as predictors by an automatic numeric screen.\n'
 section+='- Daytime presence, service operation and destinations, event-specific observed traffic, complete vehicle demand and river receiving capacity remain unqualified. The briefing makes no final-variable selection or observed-effect claim.\n\n'
 after=pre+''.join(section if i==3 else old[i] for i in range(1,9));(ROOT/'docs/AnaSOP.md').write_text(after)
 pre2,new=sections(after);assert pre==pre2 and all(old[i]==new[i] for i in old if i!=3),'Other AnaSOP sections changed'
 return after

def report(result,cov,timeplots):
 def table(d):return '<div class="scroll">'+d.to_html(index=False,border=0,escape=True,na_rep='—',float_format=lambda x:f'{x:,.3f}')+'</div>'
 inv=pd.read_csv(O/'tables/dataset_inventory.csv');inv['source_file']=inv.source_file.map(lambda x:'<a href="../../../../'+html.escape(x)+'">'+html.escape(x)+'</a>')
 body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区研究数据简报</title><style>body{font:16px/1.8 system-ui;background:#edf2f5;color:#24414e}main{max-width:1160px;margin:24px auto;background:white;padding:28px}h1{font-size:28px}h2{font-size:22px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%;height:auto}.note{padding:15px;background:#fff0db}a{color:#17658a}</style><main><h1>三区道路、交通与人口影响：当前数据简报</h1>',f'<p>直接检查{result["datasets_read"]}张表、{result["variables_summarized"]}个表内字段；另读既有源信息与数值准入记录。字段数含重复概念，不是最终变量数。本次不新增模拟、不改源数据、不选择最终变量。</p>',f'<p class="note">三区合格道路{cov["qualified_road_km"]:.2f} km；未知暴露{cov["unknown_road_km"]:.2f} km。官方2020常住人口{cov["official_population"]:,.0f}；未解决的接入人口见下表。缺字段、保密编码、空间未知和现实验证不足是不同问题。</p>','<h2>覆盖与现有数据</h2>',table(pd.read_csv(O/'tables/dataset_inventory.csv')),'<h2>人口接入的未解决部分</h2>',table(pd.read_csv(O/'tables/unresolved_population_by_radius.csv')),'<h2>楼宇范围与属性</h2>',table(pd.read_csv(O/'tables/building_scope_quality.csv')),'<h2>楼宇分配准入类别</h2>',table(pd.read_csv(O/'tables/allocation_quality_classes.csv')),'<h2>字段缺失与语义缺失</h2><p>完整字段空值率见variable_summary；下表额外保留-9999及*／Y等编码。没有把它们改成零。</p>',table(pd.read_csv(O/'tables/semantic_missingness.csv')),'<h2>旧快照与当前状态</h2><p>部分雨量清单是在下载阶段生成，仍写着尚无模拟结果；当前交付审计已经通过。这是快照时点差异，不能当作当前失败，也不能修改历史源清单。</p>',table(pd.read_csv(O/'tables/metadata_snapshot_differences.csv')),'<h2>分布图</h2><p>按各自表单位解释。反复出现的情景、时刻和分支不是独立事件，不从分布图推断真实灾害频率。标识符分布已排除。</p>']
 for p in sorted((O/'figures/distributions').glob('*.png')):body.append(f'<img loading="lazy" src="{p.relative_to(O)}" alt="{html.escape(p.name)}">')
 if timeplots:
  body.append('<h2>经确认的时序图</h2><p>固定窗口内现有序列；人口阴影为分配边界，不是置信区间。</p>')
  for p in timeplots:body.append(f'<img loading="lazy" src="{p}" alt="{html.escape(p)}">')
 else:body.append('<p>时序可用性已登记；本次尚未生成新的时序图。</p>')
 body.append('<h2>文件与研究边界</h2><p>数值合格不等于实测验证。当前可用于条件封路、车行可达性、人口存量区间和措施比较；真实拥堵、实际受困、昼间人口和城市功能损失仍有证据缺口。</p><ul>')
 for p in sorted((O/'tables').glob('*.csv')):body.append(f'<li><a href="tables/{p.name}">{p.name}</a></li>')
 body.append('<li><a href="source_bindings.json">数据来源及校验值</a></li><li><a href="validation.json">本次简报核验</a></li></ul></main></html>');(O/'review.html').write_text(''.join(body))

def main():
 a=argparse.ArgumentParser();a.add_argument('--time-series',choices=['yes','no'],default='no');args=a.parse_args()
 O.mkdir(parents=True,exist_ok=True);(O/'tables').mkdir(exist_ok=True)
 before=(ROOT/'docs/AnaSOP.md').read_text()
 if not (O/'AnaSOP_before.md').exists():(O/'AnaSOP_before.md').write_text(before)
 else:before=(O/'AnaSOP_before.md').read_text()
 bindings={str((P/n).relative_to(ROOT)):sha(P/n) for n in FILES.values()}
 bindings.update({n:sha(ROOT/n) for n in META.values()});dump(O/'source_bindings.json',bindings)
 result=run_skill();cov=coverage();timeplots=draw_time_plots() if args.time_series=='yes' else []
 after=update_section3(before,result,cov);report(result,cov,timeplots)
 assert all(sha(ROOT/n)==h for n,h in bindings.items())
 v=dict(passed=True,datasets_read=result['datasets_read'],table_field_occurrences=result['variables_summarized'],distribution_plots=result['distribution_plots'],time_series_candidates=len(result['time_series_candidates']),time_series_visualization_authorized=args.time_series=='yes',time_series_plots=len(timeplots),skipped_files=result['skipped_files'],source_files_unchanged=True,only_anasop_section3_updated=True,no_new_final_variables=True,no_new_simulations=True,script_sha256=sha(Path(__file__)),anasop_sha256=sha(ROOT/'docs/AnaSOP.md'),coverage=cov)
 dump(O/'validation.json',v);print(json.dumps(v,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
