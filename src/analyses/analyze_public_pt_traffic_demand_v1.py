"""Execute the registered descriptive traffic-demand plan on official public CSVs.
Does not construct OD pairs from marginals, assign vehicles, or count affected people.
"""
from pathlib import Path
import os,json,hashlib,datetime as dt,re,unicodedata,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_traffic_demand_analysis_v1';S=P/'sources'
B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1'
R=ROOT/'data/exp/estimation/three_ward_traffic_demand_analysis_v1'
MODES=['鉄道・地下鉄','路線バス・都電','自動車','２輪車','自転車','徒歩','その他','不明'];CAR='自動車';TOTAL='合計'
WARD_EN={'千代田区':'Chiyoda','中央区':'Chuo','港区':'Minato'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def save(name,rows):pd.DataFrame(rows).to_csv(P/name,index=False,encoding='utf-8-sig')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 R.mkdir(parents=True,exist_ok=True);plan=json.loads((P/'analysis_plan.json').read_text());assert len(plan['questions'])==5
 man=json.loads((S/'acquisition_manifest.json').read_text());assert all(digest(S/x['name'])==x['sha256']for x in man['files'])
 core=pd.read_csv(B/'pt_2018_core_zone_codes.csv',dtype=str);zm=core.drop_duplicates(['planning_zone','municipality']).set_index('planning_zone')['municipality'].to_dict();assert len(zm)==14
 frames={};inventory=[]
 for f in sorted(S.glob('*.csv')):
  d=pd.read_csv(f,dtype=str,keep_default_na=False);d['source_row_1_based']=np.arange(len(d))+2
  if 'code'in d:
   d['planning_zone']=d['code'].str.extract(r'^:(\d{4}):$',expand=False);d['ward']=d.planning_zone.map(zm)
   for m in MODES+[TOTAL]:d[m]=pd.to_numeric(d[m],errors='raise').astype('int64')
  else:d['trip']=pd.to_numeric(d['trip'],errors='raise').astype('int64')
  frames[f.name]=d
  inventory.append({'source_file':f.name,'rows':len(d),'core_rows':int(d.ward.notna().sum())if'ward'in d else None,'negative_numeric_cells':int((d[MODES+[TOTAL]]<0).sum().sum())if'ward'in d else int((d.trip<0).sum()),'sha256':digest(f)})
 save('source_inventory.csv',inventory)
 rows=[];purpose=[];profile=[];dur=[];qa=[];sourcekeys=[];allpurpose=['通勤','通学','業務','帰宅','私事','不明'];direction_summary={};all_raw_core=[]
 for flow,suffix in [('departure','generation'),('arrival','concentrated')]:
  base=frames[f'trip_data_{suffix}.csv'];base=base[base.ward.notna()].copy();assert not base.duplicated(['planning_zone','purpose']).any()
  clock=frames[f'trip_movetime_data_{suffix}.csv'];clock=clock[clock.ward.notna()].copy();assert not clock.duplicated(['planning_zone','purpose','time']).any()
  duration=frames[f'trip_time_data_{suffix}.csv'];duration=duration[duration.ward.notna()].copy();assert not duration.duplicated(['planning_zone','purpose','time']).any()
  for family,df in [('trip_end',base),('clock',clock),('duration',duration)]:
   assert (df[MODES+[TOTAL]]>=0).all().all()
   output=df.copy();output['flow']=flow;output['source_file']=f'trip_{"data"if family=="trip_end"else("movetime_data"if family=="clock"else"time_data")}_{suffix}.csv'
   all_raw_core.extend(output.to_dict('records'))
  totals=base[base.purpose=='計'].set_index('planning_zone');assert set(totals.index)==set(zm)
  for zone,r in totals.iterrows():
   z={'level':'planning_zone','zone_or_ward':zone,'ward':r.ward,'flow':flow,'car_person_trip_ends':int(r[CAR]),'all_mode_person_trip_ends':int(r[TOTAL]),'car_share':float(r[CAR]/r[TOTAL])if r[TOTAL]>0 else None,'source_file':f'trip_data_{suffix}.csv','source_row_1_based':int(r.source_row_1_based)};rows.append(z)
  for ward,group in base.groupby('ward',sort=False):
   t=group[group.purpose=='計'];car=int(t[CAR].sum());total=int(t[TOTAL].sum());rows.append({'level':'ward','zone_or_ward':ward,'ward':ward,'flow':flow,'car_person_trip_ends':car,'all_mode_person_trip_ends':total,'car_share':car/total,'source_file':f'trip_data_{suffix}.csv','source_row_1_based':None})
   for pur,g in group[group.purpose!='計'].groupby('purpose',sort=False):purpose.append({'ward':ward,'flow':flow,'purpose':pur,'car_person_trip_ends':int(g[CAR].sum()),'share_of_published_car_total':float(g[CAR].sum()/car)if car else None})
  car=int(totals[CAR].sum());total=int(totals[TOTAL].sum());direction_summary[flow]={'car_person_trip_ends':car,'all_mode_person_trip_ends':total,'car_share':car/total}
  # Audit all modes/purposes. Preserve source differences rather than forcing sums.
  for zone,r in totals.iterrows():
   g=base[base.planning_zone==zone];parts=g[g.purpose!='計']
   for mode in MODES+[TOTAL]:qa.append({'flow':flow,'planning_zone':zone,'check':'purpose_components_minus_published_total','mode':mode,'difference':int(parts[mode].sum()-r[mode])})
  for _,r in base.iterrows():qa.append({'flow':flow,'planning_zone':r.planning_zone,'check':'mode_components_minus_published_total__'+r.purpose,'mode':'all_modes','difference':int(sum(r[m]for m in MODES)-r[TOTAL])})
  for family,df in [('clock',clock),('duration',duration)]:
   for (zone,pur),g in df.groupby(['planning_zone','purpose'],sort=False):
    n=g[g.time=='合計'];assert len(n)==1;n=n.iloc[0]
    ref=base[(base.planning_zone==zone)&(base.purpose==pur)].iloc[0]
    for mode in MODES+[TOTAL]:
     qa.extend([{'flow':flow,'planning_zone':zone,'check':family+'_components_minus_native_total__'+pur,'mode':mode,'difference':int(g.loc[g.time!='合計',mode].sum()-n[mode])},{'flow':flow,'planning_zone':zone,'check':family+'_native_total_minus_daily__'+pur,'mode':mode,'difference':int(n[mode]-ref[mode])}])
   # Overall native all-purpose car distributions. No separate purpose totals are added.
   for label,g in df[df.purpose=='計'].groupby('time',sort=False):
    if label=='合計':continue
    val=int(g[CAR].sum());plain=unicodedata.normalize('NFKC',label)
    item={'flow':flow,'native_bin':label,'car_person_trip_ends':val,'fraction_of_published_daily_car_total':val/car if car else None,'bin_class':'unknown'if label=='不明'else('open_ended'if label.startswith('～')or label.endswith('～')else'bounded')}
    if family=='clock':
     m=re.fullmatch(r'(\d+)時台',plain);hour=int(m.group(1))if m else None
     item.update({'survey_hour_start':hour,'civil_hour_jst':hour%24 if hour is not None else None,'survey_day_offset':int(hour>=24)if hour is not None else None});profile.append(item)
    else:dur.append(item)
  for flow_summary in [direction_summary[flow]]:
   cp=[r for r in profile if r['flow']==flow];known=[r for r in cp if r['survey_hour_start']is not None];peak=max(known,key=lambda x:x['car_person_trip_ends']);unknown=sum(r['car_person_trip_ends']for r in cp if r['bin_class']=='unknown');openended=sum(r['car_person_trip_ends']for r in cp if r['bin_class']=='open_ended')
   flow_summary.update({'clock_unknown_person_trip_ends':unknown,'clock_open_ended_person_trip_ends':openended,'clock_known_exact_hour_person_trip_ends':sum(r['car_person_trip_ends']for r in known),'clock_known_peak_native_bin':peak['native_bin'],'clock_peak_person_trip_ends':peak['car_person_trip_ends'],'clock_peak_share_of_published_daily':peak['car_person_trip_ends']/car,'clock_sum_minus_published_total':sum(r['car_person_trip_ends']for r in cp)-car})
 save('ward_and_planning_zone_trip_ends.csv',rows);save('purpose_composition.csv',purpose);save('car_clock_profile.csv',profile);save('car_duration_profile.csv',dur);save('clock_and_duration_quality.csv',qa);save('core_native_rows_with_source_keys.csv',all_raw_core)
 # Coarse 13-region exchanges are not three-ward OD. Negative source codes remain unknown.
 reg=frames['sankey.csv'];reg=reg[(reg.year=='平成３０年')&(reg.purpose=='8_合計')&(reg.transportation=='3:自動車')&((reg.o_zone=='東京区部')|(reg.d_zone=='東京区部'))].copy();assert not reg.duplicated(['o_zone','d_zone']).any()
 regions=[]
 for _,r in reg.iterrows():regions.append({'origin_region':r.o_zone,'destination_region':r.d_zone,'car_person_trips_reported':int(r.trip)if r.trip>=0 else None,'source_value_raw':int(r.trip),'status':'reported_nonnegative'if r.trip>=0 else'negative_code_uninterpreted_unknown','source_row_1_based':int(r.source_row_1_based),'scope':'Tokyo23wards,NOTthreewardOD'})
 save('tokyo_23ward_regional_car_flows_context.csv',regions)
 # Native census panel: one row per count-unit/direction/class, complete24h, observed only.
 h=pd.read_csv(B/'core_count_hourly_long.csv',dtype={'count_authority_code':str,'count_unit_id':str,'direction_code':str,'vehicle_class_code':str,'observed_estimated_code':str})
 q=pd.read_csv(B/'hourly_profile_quality.csv',dtype={'count_authority_code':str,'count_unit_id':str,'direction_code':str,'vehicle_class_code':str,'observed_estimated_code':str})
 eligible=set(q.loc[(q.observed_estimated_code=='1')&(q.available_hour_count==24)&(q.sum_24h_minus_native_veh==0),'source_row_1_based'].astype(int))
 panel=h[h.source_row_1_based.isin(eligible)].copy();assert not panel.duplicated(['source_row_1_based','hour_start_jst']).any();assert len(panel)==24*len(eligible)
 panel_units=panel[['count_authority_code','count_unit_id']].drop_duplicates();byhour=panel.groupby('hour_start_jst').hourly_volume_veh_per_hour.sum();panel_total=float(byhour.sum());roadprofile=[{'hour_start_jst':int(hr),'panel_vehicle_passages_per_hour':float(v),'fraction_of_panel_24h_passages':float(v/panel_total),'count_units':len(panel_units),'direction_vehicle_class_profiles':len(eligible)}for hr,v in byhour.items()]
 save('road_observation_panel.csv',roadprofile);save('road_observation_panel_members.csv',q[q.source_row_1_based.isin(eligible)].to_dict('records'))
 quality=pd.DataFrame(qa);nonzero=quality[quality.difference!=0];mismatch_by=nonzero.groupby('check').difference.agg(['count','min','max']).reset_index().to_dict('records')
 # Reconciliation of public-table daily totals/clock aggregation is a diagnostic, not correction.
 results={'analysis_completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_registered_utc':plan['registered_utc'],'source_csvs':len(inventory),'source_csv_rows':sum(r['rows']for r in inventory),'source_original_csv_bytes':sum(r['bytes']for r in man['files']if r['name'].endswith('.csv')),'planning_zones':len(zm),'wards':list(WARD_EN),'trip_ends':direction_summary,'ward_totals':[r for r in rows if r['level']=='ward'],'trip_end_count_interpretation':'Separate departure and arrival person-trip endpoints; internal trips appear once in each margin,not unique people or car counts.','quality':{'reconciliation_checks':len(qa),'nonzero_differences':len(nonzero),'maximum_absolute_difference':int(quality.difference.abs().max()),'mismatch_groups':mismatch_by},'road_observed_24h_panel':{'count_units':len(panel_units),'direction_vehicle_class_profiles':len(eligible),'peak_hour_jst':int(byhour.idxmax()),'peak_share':float(byhour.max()/panel_total),'counting_unit':'vehicle passages summed over a fixed observation panel,not unique regional vehicles or traffic demand'},'full_three_ward_od_available':False,'public_coarse_region_context_rows':len(regions),'public_coarse_unknown_cells':sum(r['car_person_trips_reported']is None for r in regions),'traffic_assignment_run':False,'congestion_computed':False,'people_impacts_computed':False}
 dump(R/'summary.json',results)
 checks={'plan_registered_before_results':plan['registered_utc']<results['analysis_completed_utc'],'source_hashes_match':all(digest(S/x['name'])==x['sha256']for x in man['files']),'14_zones_covered_both_margins':all(len([r for r in rows if r['flow']==f and r['level']=='planning_zone'])==14 for f in ['departure','arrival']),'3_wards_covered_both_margins':len(results['ward_totals'])==6,'ward_sum_equals_core_total':all(sum(r['car_person_trip_ends']for r in results['ward_totals']if r['flow']==f)==results['trip_ends'][f]['car_person_trip_ends']for f in ['departure','arrival']),'no_negative_native_core_counts':all(x['negative_numeric_cells']==0 for x in inventory if x['source_file']!='sankey.csv'),'daily_clock_duration_reconciliations_exact':len(nonzero)==0,'native_rows_have_source_keys':all('source_row_1_based'in x and 'source_file'in x for x in all_raw_core),'fixed_panel24h_and_observed_only':len(panel)==24*len(eligible)and set(panel.observed_estimated_code)=={'1'},'panel_units_unique':len(panel_units)==len(panel_units.drop_duplicates()),'panel_fraction_sums1':abs(sum(r['fraction_of_panel_24h_passages']for r in roadprofile)-1)<1e-12,'no_derived_od_pairs':not results['full_three_ward_od_available'],'no_congestion_or_person_impacts':not results['congestion_computed']and not results['people_impacts_computed']}
 # Nonzero source reconciliation is quarantined from capacity/demand fitting; do not rewrite source totals.
 dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks,'source_reconciliation_policy':'Record exact differences; no rescaling,zero-fill or repair. A failure blocks model-input release,not descriptive inspection of original reported counts.'})
 dump(P/'analysis_contract.json',{'plan_sha256':digest(P/'analysis_plan.json'),'data_sources':inventory,'mode_mapping':{'car':CAR,'all_modes':TOTAL},'flow_mapping':{'departure':'generation','arrival':'concentrated'},'purposes':'total=計; components=通勤,通学,業務,帰宅,私事,不明; never combine both','time_mapping':'movetime files are departure/arrival CLOCK categories; time files are journey DURATION bins. Retain3–26hour labels,open-ended andunknown categories.','aggregation':'14 official planning zones into three wards. Internal journeys appear in both margins; margins must not be added to obtain distinct journeys.','road_panel':'Observed flag1,exact complete24h records only; one survey-unit/direction/vehicle-class profile; no multiplication by linked basic-section count.','units':'PT automobile person-trip ends; census vehicle passages. No occupancy or driver-only conversion applied.','negative_codes':'Coarse-region -1 values remain unknown; do not treat as negative flow or zero. Core chart zeros retained as reported,not proof of absence.','interpretation':'Sample-expanded2018 normal weekday; census2021. No causal result,no event-specific traffic reconstruction,no congestion or affectedpeople.','formal_model_input_release':all(checks.values())and results['full_three_ward_od_available']})
 print(json.dumps(results,ensure_ascii=False,indent=2));print('CHECKS',checks)
if __name__=='__main__':main()
