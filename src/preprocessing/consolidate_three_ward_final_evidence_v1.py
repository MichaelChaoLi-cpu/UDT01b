#!/usr/bin/env python3
"""Consolidate qualified road, traffic and resident evidence; never rerun a model.
Existing concepts, masks, original values and candidate-paired intervals are preserved.
"""
from pathlib import Path
import os,json,hashlib,re,sys,importlib.util
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_final_evidence_v1'
R=ROOT/'data/exp/data-preprocessing/three_ward_final_evidence_v1'
H=ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1'
T=ROOT/'data/processed/three_ward_s1r_september_traffic_phases_v1'
HR=ROOT/'data/exp/estimation'/H.name
TR=ROOT/'data/exp/estimation'/T.name
EVENTS=['E20240731','E20240830','E20250911']
BIND={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def csv(p):return pd.read_csv(bind(p))
def load(p):
 with np.load(bind(p),allow_pickle=False)as z:return dict(z)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def context(event,q):
 if event=='E20240731':
  hp=ROOT/'data/processed'/('three_ward_s1_reallocation_pilot_v1'if q=='Q2p5'else'three_ward_s1r_shared_allowance_pilot_v1')/'batches/O1'
  ap=H/'july'/q/'access'
 else:hp=H/'batches'/(event+'_'+q);ap=hp/'access'
 return hp,ap
CRITICAL={'event':'Rainfall Parent','case':'Case','stage':'Receiving Water Stage','branch':'Road Depth Branch','bound':'Road Depth Branch','scope':'Reporting Scope','reference':'Reference Portfolio','portfolio':'Measure Configuration','buffer_m':'Population Access Candidate Radius','frame':'Frame','elapsed_hours':'Elapsed Hours','reference_closure_km_h':'Reference Closed Road Duration','remaining_closure_km_h':'Remaining Closed Road Duration','net_avoided_km_h':'Avoided Closed Road Duration','restored_km_h':'Restored Road Exposure','newly_closed_km_h':'Newly Closed Road Exposure','peak_newly_closed_km':'Peak Newly Closed Road Length','original_core_request_proxy':'Original Core Vehicle Demand','served_core_request_proxy':'Served Core Vehicle Demand','unreachable_core_request_proxy':'Unreachable Vehicle Requests','common_cohort_mean_cost_min':'Common OD Cost','common_cohort_cost_index':'Common OD Cost Index','common_cohort_mean_model_cost_min':'Common OD Cost','common_cohort_cost_change_min':'Common OD Cost Change','common_cohort_cost_change_percent':'Relative Common OD Cost Change','restored_request_proxy':'Restored Vehicle Requests','newly_unreachable_request_proxy':'Newly Unreachable Vehicle Requests'}
def readable(c):
 if c in CRITICAL:return CRITICAL[c]
 c=re.sub(r'_(m3|m2|km|pct|percent|min|s)$','',c)
 return ' '.join({'od':'OD','q5':'Q5','q2p5':'Q2.5','h':'Hours','km':'Kilometers','m3':'Cubic Meters','m2':'Square Meters'}.get(w,w.capitalize())for w in c.split('_'))
def loss_source(scope):return scope.scope.eq('Three wards')&scope.bound.eq('lower')&scope.buffer_m.eq(10)
def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 assert read(HR/'final_summary_validation.json')['passed'] and read(HR/'delivery_validation.json')['passed']
 assert read(TR/'validation.json')['passed']and read(TR/'campaign_state.json')['status']=='qualified_complete'
 source=[H/n for n in ['three_parent_road_comparison.csv','three_parent_Q5_vs_Q2p5_population_summary.csv','three_parent_population_totals_all_arms.csv','three_parent_head_transfer_comparison.csv']]+[T/n for n in ['all_times_traffic_outcomes.csv','all_times_paired_traffic_effects.csv','fixed_all_times_cohort_costs.csv']]
 # Use the bundled variable discovery functions on this approved bounded source list.
 spec=importlib.util.spec_from_file_location('mili_variables',ROOT/'.codex/skills/data-preprocessing/scripts/prepare_variable_list.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 inventory=[];dictionaries=[];decisions={'decision':'MILI-D-20261024-002','selection':'All fields of the seven qualified result tables; existing approved concepts and their scope/interval variants. No new final concept, imputation or statistical transformation.','datasets':{}}
 exports=[]
 for f in source:
  d=csv(f);meta,err=mod.read_columns(f);assert err is None
  names={c:readable(c)for c in d};assert len(set(names.values()))==len(names)
  out=P/(f.stem+'_preprocessed.parquet');nd=d.rename(columns=names);nd.to_parquet(out,index=False)
  pd.testing.assert_frame_equal(pd.read_parquet(out).rename(columns={v:k for k,v in names.items()}),d)
  variables=[]
  for col in meta:
   c=col['column_name'];inventory.append({'source_dataset':str(f.relative_to(ROOT)),'original_name':c,**{k:v for k,v in col.items()if k!='column_name'},'readable_name':names[c],'feasibility_status':'qualified conditional result; no observed impact validation'})
   v={'original_name':c,'readable_name':names[c],'full_name':names[c],'is_final_variable':'yes','preprocessing':[]};variables.append(v)
   dictionaries.append({'Source Table':f.name,'Source Field':c,'Readable Field':names[c],'Role':'Existing concept variant or identifier; not an additional independent construct','Null Policy':'Preserve original nulls and undefined denominators','Output':out.name})
  decisions['datasets'][str(f.relative_to(ROOT))]={'output':str(out.relative_to(ROOT)),'script':'src/preprocessing/preprocess_final_'+f.stem+'.py','variables':variables}
  exports.append({'Source':f.name,'Rows':len(d),'Fields':len(d.columns),'Output':out.name,'Lossless':True})
 pd.DataFrame(inventory).to_csv(R/'variable_list.csv',index=False);pd.DataFrame(dictionaries).to_csv(R/'data_dictionary.csv',index=False);dump(R/'decisions.json',decisions)
 road=[];pairedpop=[];audits=[];common=None
 clip=load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');w=clip['union_length_m']/1000
 direct=csv(H/'three_parent_Q5_vs_Q2p5_population_summary.csv');head=csv(H/'three_parent_head_transfer_comparison.csv');totals=csv(H/'three_parent_population_totals_all_arms.csv')
 primary=totals[loss_source(totals)].copy();absolute=[]
 for event in EVENTS:
  for stage in ['H','L']:
   g=primary[primary.event.eq(event)&primary.stage.eq(stage)]
   a=g[g.arm.eq('baseline')].sort_values('portfolio');assert len(a)==2
   for c in ['loss_lower_resident_hours','loss_upper_resident_hours','unresolved_stock']:assert np.isclose(a[c].iloc[0],a[c].iloc[1],rtol=0,atol=1e-6)
   for arm,q in [('baseline','none'),('drainage','Q2p5'),('drainage','Q5')]:
    z=a.iloc[0]if arm=='baseline'else g[g.arm.eq(arm)&g.portfolio.eq(q)].iloc[0]
    absolute.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Measure Configuration':q,'Resident Access Loss Lower':z.loss_lower_resident_hours,'Resident Access Loss Upper':z.loss_upper_resident_hours,'Unresolved Resident Stock':z.unresolved_stock,'Resident Population':496345,'Population Access Candidate Radius':10,'Road Depth Branch':'lower','Window Hours':9,'Unit':'resident-hours; not unique people'})
  for q in ['Q2p5','Q5']:
   hp,ap=context(event,q);runs=read(hp/'run_manifest.json')['runs'];ps=csv(ap/'paired_population_summary.csv');ps=ps[loss_source(ps)]
   for run in runs:
    stage=run['stage'];case=run['case_id'];er=ROOT/'data/exp/estimation'/hp.relative_to(ROOT/'data/processed')/'execution'/case
    ad=read(er/'audit.json');pump=read(er/'pump_budget_summary.json');assert ad['all_checks_passed']and len(ad['checks'])==28 and all(x['pass']for x in ad['checks'])and ad['summary']['negative_cells']==0 and pump['passed']and len(pump['shared_receiver_checks'])==8 and all(pump['shared_receiver_checks'].values());audits.append({'Case':case,'Numerical Checks':28,'Pump Checks':8,'Negative Cells':0})
    a,b=[load(hp/'exposure'/(case+'__'+arm+'.npz'))for arm in ['baseline','drainage']];assert np.array_equal(a['road_indices'],b['road_indices']);ix=a['road_indices']
    if common is None:common=ix.copy()
    assert np.array_equal(common,ix)and len(ix)==28064
    ts=a['analysis_times'];assert np.array_equal(ts,b['analysis_times'])and len(ts)==109 and ts[0]==0 and ts[-1]==32400
    x,y=[z['road_state_codes'][:,0]==2 for z in [a,b]];wt=w[ix];lens=[v@wt for v in [x,y]];rest=(x&~y)@wt;new=(~x&y)@wt;integrate=lambda v:float(np.trapezoid(v,ts)/3600)
    base,rem=map(integrate,lens);benefit=base-rem;rr,nn=map(integrate,[rest,new]);assert abs(benefit-rr+nn)<1e-7
    road.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Measure Configuration':q,'Reference Portfolio':'none','No Action Closed Road Duration':base,'Remaining Closed Road Duration':rem,'Avoided Closed Road Duration':benefit,'Reduction Percent':100*benefit/base if base>0 else np.nan,'Restored Road Exposure':rr,'Newly Closed Road Exposure':nn,'Peak Newly Closed Road Length':float(new.max()),'Qualified Road Length':float(w[ix].sum()),'Unknown Road Length':float(w.sum()-w[ix].sum()),'Road Depth Branch':'lower','Window Hours':9,'Unit':'km h; peak length in km'})
    z=ps[ps.stage.eq(stage)];assert len(z)==1;z=z.iloc[0]
    pairedpop.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Reference Portfolio':'none','Measure Configuration':q,**{new:float(z[old])for old,new in [('net_avoided_loss_lower_resident_hours','Paired Resident Benefit Lower'),('net_avoided_loss_upper_resident_hours','Paired Resident Benefit Upper'),('restored_lower_resident_hours','Restored Resident Access Lower'),('restored_upper_resident_hours','Restored Resident Access Upper'),('new_loss_lower_resident_hours','New Resident Access Loss Lower'),('new_loss_upper_resident_hours','New Resident Access Loss Upper')]},'Unit':'resident-hours; candidate-paired'})
 for z in direct[loss_source(direct)].itertuples():
  pairedpop.append({'Rainfall Parent':z.event,'Receiving Water Stage':z.stage,'Reference Portfolio':'Q2p5','Measure Configuration':'Q5',**{new:float(getattr(z,old))for old,new in [('net_lower_resident_hours','Paired Resident Benefit Lower'),('net_upper_resident_hours','Paired Resident Benefit Upper'),('restored_lower_resident_hours','Restored Resident Access Lower'),('restored_upper_resident_hours','Restored Resident Access Upper'),('new_loss_lower_resident_hours','New Resident Access Loss Lower'),('new_loss_upper_resident_hours','New Resident Access Loss Upper')]},'Unit':'resident-hours; candidate-paired'})
 road=pd.DataFrame(road);absolute=pd.DataFrame(absolute);pairedpop=pd.DataFrame(pairedpop)
 assert len(road)==12 and len(absolute)==18 and len(pairedpop)==18 and len(audits)==12
 for prefix in ['Paired Resident Benefit','Restored Resident Access','New Resident Access Loss']:assert(pairedpop[prefix+' Lower']<=pairedpop[prefix+' Upper']+1e-8).all()
 assert absolute['Unresolved Resident Stock'].eq(41308).all()
 # Map only existing fixed-anchor cache states. No new shortest-path or routing calculation.
 support=load(ROOT/'data/processed/three_ward_night_population_transport_v1/block_access_support.npz');normal=load(ROOT/'data/processed/three_ward_night_population_transport_v1/normal_node_access.npz')['bidirectional']
 census=csv(ROOT/'data/processed/three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv');census['block_id']=census.block_id.astype(str)
 census=census.set_index('block_id').loc[support['block_ids']].reset_index();pop=census.official_2020_residents.to_numpy(dtype='i8');assert pop.sum()==496345
 ptr=support['offsets_10m'];ix=support['node_indices_10m'];lens=np.diff(ptr)
 def count(v):c=np.r_[0,np.cumsum(v,dtype='i8')];return c[ptr[1:]]-c[ptr[:-1]]
 complete=(lens>0)&(count(normal[ix])==lens);assert pop[~complete].sum()==41308
 maps=[]
 for stage in ['H','L']:
  values=[];anchors=[]
  for q in ['Q2p5','Q5']:
   hp,ap=context('E20240830',q);run=next(v for v in read(hp/'run_manifest.json')['runs']if v['stage']==stage);proto=read(ap/'protocol.json');anchors.append(proto['anchor_frame']);z=load(ap/'anchor_nodes'/(run['case_id']+'__lower__drainage.npz'));values.append(z['bidirectional'])
  assert anchors==[97,97]
  x,y=values;pos=normal&~x&y;neg=normal&x&~y;pc,nc=count(pos[ix]),count(neg[ix]);cl=np.full(len(pop),'No access change',dtype='U40');cl[complete&(pc>0)]='Possible restoration';cl[complete&(pc==lens)]='All-candidate restoration';cl[complete&(nc>0)]='Possible new loss';cl[complete&(nc==lens)]='All-candidate new loss';cl[complete&(pc>0)&(nc>0)]='Mixed changes';cl[~complete]='Unresolved attachment';cl[pop==0]='No recorded residents'
  # Independent anchor check against source candidate-paired intervals.
  lo=np.where(nc>0,-pop,np.where(pc==lens,pop,0))*complete;hi=np.where(pc>0,pop,np.where(nc==lens,-pop,0))*complete
  z=direct[loss_source(direct)&direct.event.eq('E20240830')&direct.stage.eq(stage)].iloc[0]
  assert lo.sum()==z.anchor_net_lower and hi.sum()==z.anchor_net_upper
  for j in range(len(pop)):maps.append({'Block ID':str(support['block_ids'][j]),'Receiving Water Stage':stage,'Resident Population':int(pop[j]),'Class':cl[j],'Candidate Complete':bool(complete[j]),'Candidate Count':int(lens[j]),'Restored Candidates':int(pc[j]),'New Loss Candidates':int(nc[j]),'Paired Resident Benefit Lower':int(lo[j]),'Paired Resident Benefit Upper':int(hi[j]),'Frame':97,'Rainfall Parent':'E20240830','Reference Portfolio':'Q2p5','Measure Configuration':'Q5'})
 tables={'Road Effects':road,'Resident Loss':absolute,'Paired Resident Effects':pairedpop,'Adverse Head Changes':head[head.scope.eq('Three wards')].copy(),'Traffic States':csv(T/'all_times_traffic_outcomes.csv'),'Paired Traffic Effects':csv(T/'all_times_paired_traffic_effects.csv'),'Fixed Temporal Cohort':csv(T/'fixed_all_times_cohort_costs.csv'),'Map Classifications':pd.DataFrame(maps),'Numerical Admission':pd.DataFrame(audits)}
 limits=pd.DataFrame([{'Item':k,'Value':v}for k,v in [('Decision','MILI-D-20261024-002'),('Hydraulic scope','Three inspected rainfall parents; 12 treatment cases, four July reused and eight new'),('Traffic scope','September only, four snapshots x two stages x three arms; fixed 17:00 request proxy x2'),('Population','Official 2020 nighttime stock 496345; 41308 unresolved; resident-hours are not unique people'),('Road support','28064 qualified roads, 836.958326 km; 277.920092 km unknown exposure retained conditionally available'),('Receiver limits','Hypothetical 2.5 or 5 m3/s shared allowance, not observed equipment or downstream safety'),('Intervals','Frozen 10 m candidate branch, lower road-depth main branch; no interval midpoint or marginal-bound subtraction'),('Numerical status','Original 28 hydraulic and eight pump checks for all treatment cases; 18 new traffic fits and 36 path sweeps passed'),('Empirical limits','No independent flood/congestion validation, daytime presence or dynamic queues'),('Stopping','Declared batch complete; no new simulation, fitting, optimization, training or sealed-test access')]])
 tables['Scope and Units']=limits
 for name,d in tables.items():
  slug=name.lower().replace(' ','_');d.to_csv(P/(slug+'.csv'),index=False);d.to_parquet(P/(slug+'.parquet'),index=False)
 workbook=R/'Three_Ward_Final_Evidence.xlsx'
 with pd.ExcelWriter(workbook,engine='openpyxl')as writer:
  for name,d in tables.items():d.to_excel(writer,sheet_name=name,index=False)
 wb=load_workbook(workbook)
 for sheet in wb:
  sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
  for cells in sheet.columns:
   letter=cells[0].column_letter;sheet.column_dimensions[letter].width=min(48,max(16,max(len(str(c.value or''))for c in list(cells)[:40])+2))
 wb.save(workbook)
 assert all(sha(ROOT/n)==h for n,h in BIND.items())
 dump(R/'source_bindings.json',BIND)
 validation={'passed':True,'source_tables':7,'lossless_exports':exports,'hydraulic_cases_independently_checked':12,'original_28_plus_8_preserved':True,'road_trajectories_independently_integrated':12,'population_total':496345,'unresolved_stock':41308,'paired_intervals_not_marginal_subtraction':True,'anchor_map_pair_reproduced':True,'traffic_states':24,'new_simulations':0,'new_traffic_fits':0,'original_sources_unchanged':True,'workbook':str(workbook.relative_to(ROOT))}
 dump(R/'validation.json',validation)
 (R/'README.md').write_text('# Final evidence consolidation\n\nSeven existing tables are losslessly exported to readable Parquet fields. No missing-value imputation, clipping, resampling, interval midpoint, changed mask or new final research construct is introduced. Existing twenty-four main concepts remain authoritative. All raw fields and mappings appear in variable_list.csv, decisions.json and data_dictionary.csv. Original time, stage, population stock and demand denominators are preserved.\n\nThe final workbook separately contains twelve reconstructed road comparisons, eighteen absolute resident-loss states, eighteen candidate-paired resident contrasts, adverse head changes, twenty-four September traffic states, paired traffic effects and the fixed temporal OD cohort. Reference, units and unknown support remain explicit. This is an analytical evidence ledger, not approval of final manuscript tables.\n\nMaps use existing August anchor caches at frame97 and compare Q5 with Q2.5 at identical candidates. They do not reroute traffic or allocate population to centroids.\n')
 print(json.dumps({'passed':True,'road_rows':len(road),'resident_loss_rows':len(absolute),'paired_resident_rows':len(pairedpop),'map_rows':len(maps),'workbook':str(workbook)},indent=2))
if __name__=='__main__':main()
