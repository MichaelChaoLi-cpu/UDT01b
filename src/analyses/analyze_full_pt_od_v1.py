"""Execute predeclared full-OD descriptive analysis; no vehicle assignment or congestion.
Hierarchical geography totals are excluded from pair sums through an explicit partition.
"""
from pathlib import Path
import json,re,hashlib,datetime as dt,collections,unicodedata
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_full_pt_od_analysis_v1';D=P/'decoded_tables';S=P/'sources';R=ROOT/'data/exp/estimation/three_ward_full_pt_od_analysis_v1';B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1';PRE=ROOT/'data/processed/three_ward_traffic_demand_analysis_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def save(name,rows):pd.DataFrame(rows).to_csv(P/name,index=False,encoding='utf-8-sig')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def code(s):
 m=re.match(r'[:：]([^:\s]+)',str(s).strip());return m.group(1)if m else None
def book(name):
 d=pd.read_csv(D/name,skiprows=4,header=None,keep_default_na=False,dtype=str);d['source_row_1_based']=np.arange(len(d))+5;return d
def num(series):return pd.to_numeric(series.astype(str).str.replace(',','',regex=False),errors='raise')
def main():
 R.mkdir(parents=True,exist_ok=True)
 dependencies=[B/'pt_2018_zone_code_crosswalk.csv',B/'pt_2018_core_zone_codes.csv',B/'pt_2018_candidate_external_zones_6677.geojson',PRE/'ward_and_planning_zone_trip_ends.csv',PRE/'car_clock_profile.csv']
 dependency_hashes={str(f.relative_to(ROOT)):sha(f)for f in dependencies}
 dump(P/'dependency_bindings.json',{'sha256':dependency_hashes,'role':'Earlier source-preserving zone and chart inputs referenced by this analysis'})
 plan=json.loads((P/'analysis_plan.json').read_text());man=json.loads((S/'acquisition_manifest.json').read_text());dec=json.loads((P/'decode_manifest.json').read_text());assert len(man['files'])==8
 assert all(sha(S/x['name'])==x['sha256']for x in man['files']);assert all(sha(D/x['decoded_file'])==x['sha256']for x in dec['files'])
 lookup=pd.read_csv(B/'pt_2018_zone_code_crosswalk.csv',dtype=str);known=set(lookup.planning_zone);zoneinfo={z:{'municipalities':sorted(set(g.municipality)),'prefectures':sorted(set(g.prefecture))}for z,g in lookup.groupby('planning_zone')};corelookup=pd.read_csv(B/'pt_2018_core_zone_codes.csv',dtype=str);cw=corelookup.drop_duplicates('planning_zone').set_index('planning_zone').municipality.to_dict();core=set(cw);midward=corelookup.drop_duplicates('middle_zone').set_index('middle_zone').municipality.to_dict();midknown=set(lookup.middle_zone)
 od=pd.read_csv(D/'d-1.cp932.csv',skiprows=3,encoding='cp932',dtype=str,keep_default_na=False);od['source_row_1_based']=np.arange(len(od))+5;od['origin_code']=od['発地'].map(code);od['destination_code']=od['着地'].map(code)
 for m in ['鉄道','バス','自動車','２輪車','自転車','徒歩','その他','不明','計']:od[m]=num(od[m]).astype('int64')
 assert not od.duplicated(['origin_code','destination_code','目的種類']).any();assert (od.iloc[:,6:15]>=0).all().all()
 totals=od[od['目的種類']=='計'].copy();unique_labels=dict(zip(od['origin_code'],od['発地']));extras={c for c in unique_labels if '-'in c};partition_codes=known|extras|{'8700','9999'}
 hierarchy=[{'code':c,'native_label':label,'role':'known_fine_zone'if c in known else'location_unresolved_partition_component'if c in partition_codes else'geographic_subtotal_excluded','in_od_partition':c in partition_codes}for c,label in unique_labels.items()];save('od_geographic_hierarchy.csv',hierarchy)
 part=totals[totals.origin_code.isin(partition_codes)&totals.destination_code.isin(partition_codes)].copy();part['origin_ward']=part.origin_code.map(cw);part['destination_ward']=part.destination_code.map(cw);part['car_person_trips']=part['自動車'];part['origin_located']=part.origin_code.isin(known);part['destination_located']=part.destination_code.isin(known)
 touched=part[part.origin_code.isin(core)|part.destination_code.isin(core)].copy()
 def classify(r):
  oc=r.origin_code in core;dc=r.destination_code in core
  if oc and dc:return 'within_same_ward'if cw[r.origin_code]==cw[r.destination_code]else'between_core_wards'
  if oc:return 'known_external_departure'if r.destination_code in known else'location_unresolved_departure'
  return 'known_external_arrival'if r.origin_code in known else'location_unresolved_arrival'
 touched['flow_class']=touched.apply(classify,axis=1);fields=['origin_code','destination_code','origin_ward','destination_ward','origin_located','destination_located','flow_class','car_person_trips','発地','着地','source_row_1_based'];save('core_touching_car_person_od.csv',touched[fields]);save('all_partition_car_person_od.csv',part[['origin_code','destination_code','car_person_trips','source_row_1_based']])
 decomp=touched.groupby('flow_class').car_person_trips.sum().astype(int).to_dict();save('flow_decomposition.csv',[{'flow_class':k,'car_person_trips_per_normal_weekday':v}for k,v in decomp.items()]);matrix=touched[touched.flow_class.isin(['within_same_ward','between_core_wards'])].groupby(['origin_ward','destination_ward']).car_person_trips.sum().reset_index();save('three_ward_car_person_matrix.csv',matrix)
 # Marginal checks against native all-geography8888 cells and prior public chart exports.
 prior=pd.read_csv(PRE/'ward_and_planning_zone_trip_ends.csv',dtype={'zone_or_ward':str});reconcile=[]
 for flow,axis,other in [('departure','origin_code','destination_code'),('arrival','destination_code','origin_code')]:
  for z in sorted(core):
   total=int(totals.loc[(totals[axis]==z)&(totals[other]=='8888'),'自動車'].sum());derived=int(part.loc[part[axis]==z,'自動車'].sum());pr=int(prior.loc[(prior.level=='planning_zone')&(prior.zone_or_ward==z)&(prior.flow==flow),'car_person_trip_ends'].iloc[0]);reconcile.append({'flow':flow,'planning_zone':z,'partition_margin':derived,'native_grand_margin':total,'prior_public_chart_margin':pr,'partition_minus_grand':derived-total,'grand_minus_chart':total-pr})
 save('od_margin_reconciliation.csv',reconcile)
 # All-geography count crosschecks, important for ambiguous8700/9999 categories.
 root_checks=[]
 for axis,other in [('origin_code','destination_code'),('destination_code','origin_code')]:
  native_totals=totals.loc[totals[other]=='8888'].groupby(axis)['自動車'].sum().to_dict()
  partition_totals=part.groupby(axis)['自動車'].sum().to_dict()
  for z in sorted(known):
   native=int(native_totals.get(z,0));actual=int(partition_totals.get(z,0));root_checks.append({'axis':axis,'planning_zone':z,'partition_margin':actual,'grand_margin':native,'difference':actual-native})
 save('all_zone_partition_reconciliation.csv',root_checks)
 # External known partners: do not distribute flow over municipalities in multi-municipality zones.
 ext=touched[touched.flow_class.isin(['known_external_departure','known_external_arrival'])].copy();ext['external_code']=np.where(ext.origin_code.isin(core),ext.destination_code,ext.origin_code);ext['external_municipality']=ext.external_code.map(lambda z:zoneinfo[z]['municipalities'][0]if len(zoneinfo[z]['municipalities'])==1 else'MULTI: '+' / '.join(zoneinfo[z]['municipalities']));ext['external_prefecture']=ext.external_code.map(lambda z:' / '.join(zoneinfo[z]['prefectures']));partners=ext.groupby(['flow_class','external_municipality','external_prefecture']).car_person_trips.sum().reset_index().sort_values(['flow_class','car_person_trips'],ascending=[True,False]);save('known_external_municipality_partners.csv',partners)
 ep=ext.groupby(['flow_class','external_code']).car_person_trips.sum().reset_index();save('known_external_planning_zone_partners.csv',ep)
 # Existing10km pool means zones merely intersecting the buffer, not exact origin coverage.
 gj=json.loads((B/'pt_2018_candidate_external_zones_6677.geojson').read_text());pool={str(f['properties']['planning_zone'])for f in gj['features']};halo=[]
 for flow,g in ext.groupby('flow_class'):
  total=int(g.car_person_trips.sum());inside=int(g.loc[g.external_code.isin(pool),'car_person_trips'].sum());halo.append({'flow_class':flow,'known_external_car_person_trips':total,'zone_intersects_candidate_pool_car_person_trips':inside,'zone_outside_pool_car_person_trips':total-inside,'zone_pool_share':inside/total})
 save('candidate_halo_demand_coverage.csv',halo)
 # b3 is explicitly tripcounts, not per-capita units; purpose=計 and vehicle=合計 only for totals.
 b3=book('H30_b3.csv');b3['zone']=b3[3].map(code)
 for c in [6,7,8,9]:b3[c]=num(b3[c]).astype('int64')
 resident=[]
 for w in sorted(set(cw.values())):
  g=b3[b3.zone.map(cw).eq(w)&b3[4].eq('計')&b3[5].eq('合計')];resident.append({'ward':w,'driver_trips_of_residents':int(g[6].sum()),'passenger_trips_of_residents':int(g[7].sum()),'driving_status_unknown_trips_of_residents':int(g[8].sum()),'three_vehicle_class_person_trips_of_residents':int(g[9].sum()),'basis':'residence,NOT origin/destination;classes passenger car/light car,goods/light goods,rental/car share'})
 save('resident_driver_trip_reference.csv',resident)
 car=book('H30_5_3.csv');car['origin_code']=car[1].map(code);car['destination_code']=car[2].map(code)
 for c in range(3,12):car[c]=num(car[c]).astype('int64')
 # Validate driver interpretation by every purpose; the native5-3 file itself labels trips.
 purpose_names=['自宅－勤務','自宅－通学','自宅－業務','自宅－私事','帰宅','勤務・業務','私事','不明','計'];cg=car[car[0].eq('合計')&car.origin_code.eq('888')&car.destination_code.eq('888')];assert len(cg)==1;cg=cg.iloc[0];driver_evidence=[]
 for c,pur in zip(range(3,12),purpose_names):
  br=b3[b3.zone.eq('8600')&b3[4].eq(pur)&b3[5].eq('合計')];assert len(br)==1;bv=int(br.iloc[0][6]);driver_evidence.append({'purpose':pur,'b3_resident_driver_total':bv,'table5_3_whole_region_total':int(cg[c]),'difference':int(cg[c])-bv})
 save('car_od_driver_interpretation_evidence.csv',driver_evidence)
 # Native middle-zone subtotal checks. Do not manufacture finer vehicle flows from personOD.
 mc={c for c in car.origin_code.unique()if c in midknown or '-'in c}|{'870','999'};mt=car[car[0].eq('合計')&car.origin_code.isin(mc)&car.destination_code.isin(mc)].copy();coremid=set(midward);mcored=mt[mt.origin_code.isin(coremid)|mt.destination_code.isin(coremid)].copy();mcored['source_total_driver_candidate_trips']=mcored[11];mcored['origin_ward']=mcored.origin_code.map(midward);mcored['destination_ward']=mcored.destination_code.map(midward);save('middle_zone_car_od_driver_candidate.csv',mcored[['origin_code','destination_code','origin_ward','destination_ward','source_total_driver_candidate_trips','source_row_1_based']])
 middlechecks=[];middlerows=[]
 for flow,axis,other in [('departure','origin_code','destination_code'),('arrival','destination_code','origin_code')]:
  for z,w in midward.items():
   reference=int(car.loc[car[0].eq('合計')&car[axis].eq(z)&car[other].eq('888'),11].sum());partn=int(mt.loc[mt[axis].eq(z),11].sum());middlechecks.append({'flow':flow,'middle_zone':z,'partition_total':partn,'native_total':reference,'difference':partn-reference});middlerows.append({'flow':flow,'ward':w,'middle_zone':z,'driver_candidate_trips_per_day':partn})
 save('middle_zone_driver_candidate_margins.csv',middlerows);save('middle_zone_partition_reconciliation.csv',middlechecks)
 # c1/c3 comparisons retain original row and column positions.
 c1=book('H30_c1.csv');c1['zone']=c1[0].map(code);c3=book('H30_c3.csv');c3['zone']=c3[2].map(code);supp=[]
 prevclock=pd.read_csv(PRE/'car_clock_profile.csv')
 for flow,basecol,c3col in [('departure',4,6),('arrival',13,15)]:
  for z in sorted(core):
   rr=c1[c1.zone.eq(z)&c1[1].eq('計')];assert len(rr)==1;reference=int(prior.loc[prior.level.eq('planning_zone')&prior.zone_or_ward.eq(z)&prior.flow.eq(flow),'car_person_trip_ends'].iloc[0]);supp.append({'table':'c1','flow':flow,'key':z,'value':int(rr.iloc[0][basecol]),'reference':reference,'difference':int(rr.iloc[0][basecol])-reference})
  for label,g in c3[c3.zone.isin(core)].groupby(3):
   if label in ['計','合計']:continue
   matched=prevclock[prevclock.flow.eq(flow)&prevclock.native_bin.eq(label)]
   if len(matched)!=1:raise ValueError('Unknownnativeclocklabel '+label)
   value=int(num(g[c3col]).sum());reference=int(matched.iloc[0].car_person_trip_ends);supp.append({'table':'c3','flow':flow,'key':label,'value':value,'reference':reference,'difference':value-reference})
 save('companion_chart_reconciliation.csv',supp)
 e=book('H30_e2.csv');e['origin_code']=e[0].map(code);e['destination_code']=e[1].map(code);e['native_car_mean_minutes']=num(e[4]);positive=touched[touched.car_person_trips>0].merge(e[['origin_code','destination_code','native_car_mean_minutes','source_row_1_based']],on=['origin_code','destination_code'],how='left',suffixes=['_od','_mean']);positive['mean_reference_status']=np.where(positive.native_car_mean_minutes.gt(0),'reported_positive_mean','missing_or_nonpositive_mean_not_freeflow_zero');save('core_positive_od_mean_time_reference.csv',positive[['origin_code','destination_code','car_person_trips','native_car_mean_minutes','mean_reference_status','source_row_1_based_od','source_row_1_based_mean']])
 ward_matrix=matrix.pivot(index='origin_ward',columns='destination_ward',values='car_person_trips').fillna(0).astype(int);internal=int(matrix.car_person_trips.sum());total_touch=int(touched.car_person_trips.sum());known_out=decomp.get('known_external_departure',0);known_in=decomp.get('known_external_arrival',0);unknown_out=decomp.get('location_unresolved_departure',0);unknown_in=decomp.get('location_unresolved_arrival',0)
 checks={'dependent_inputs_unchanged_during_analysis':all(sha(ROOT/name)==digest for name,digest in dependency_hashes.items()),'eight_source_files_hash_verified':len(man['files'])==8 and all(sha(S/x['name'])==x['sha256']for x in man['files']),'all_decoded_hashes_verified':all(sha(D/x['decoded_file'])==x['sha256']for x in dec['files']),'14_core_zones_match_official_codes':len(core)==14,'615_known_fine_zones':len(known)==615,'source_od_keys_unique':not od.duplicated(['origin_code','destination_code','目的種類']).any(),'28_core_margins_equal_source_and_prior_chart':all(r['partition_minus_grand']==0 and r['grand_minus_chart']==0 for r in reconcile),'1230_all_zone_partition_checks':len(root_checks)==1230 and all(r['difference']==0 for r in root_checks),'partition_excludes_geographic_subtotals':not part.origin_code.isin(set(unique_labels)-partition_codes).any()and not part.destination_code.isin(set(unique_labels)-partition_codes).any(),'touching_total_not_double_counted':total_touch==233508+221296-internal,'ward_matrix_equals_internal':int(ward_matrix.values.sum())==internal,'nine_driver_purpose_totals_match':len(driver_evidence)==9 and all(r['difference']==0 for r in driver_evidence),'six_middle_zone_margins_match':len(middlechecks)==6 and all(r['difference']==0 for r in middlechecks),'companion_tables_equal_prior_charts':all(r['difference']==0 for r in supp),'known_halo_partition_conserved':all(r['known_external_car_person_trips']==r['zone_intersects_candidate_pool_car_person_trips']+r['zone_outside_pool_car_person_trips']for r in halo),'no_vehicle_disaggregation_or_traffic_assignment':True}
 summary={'analysis_completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_registered_utc':plan['registered_utc'],'source_files':len(man['files']),'original_download_bytes':sum(x['bytes']for x in man['files']),'native_d1_data_rows':len(od),'native_allpurpose_rows_including_subtotals':len(totals),'disjoint_partition_pair_rows':len(part),'core_touching_pair_rows_including_reported_zero_car':len(touched),'core_touching_positive_car_pairs':int(touched.car_person_trips.gt(0).sum()),'core_touching_car_person_trips_per_day':total_touch,'internal_core_car_person_trips_per_day':internal,'decomposition':decomp,'departure_car_person_trips':internal+known_out+unknown_out,'arrival_car_person_trips':internal+known_in+unknown_in,'external_known_share_of_known_core_touching':(known_out+known_in)/(internal+known_out+known_in),'unlocated_partner_share_of_all_core_touching':(unknown_out+unknown_in)/total_touch,'ward_matrix':ward_matrix.to_dict(orient='index'),'top_known_external_partners':partners.groupby('flow_class',sort=False).head(8).to_dict('records'),'candidate_halo':halo,'resident_driver_reference':resident,'middle_driver_candidate_margins':middlerows,'driver_interpretation':{'status':'strong_numerical_consistency_with_driver_trips; specific5_3driver_only_definition_not_explicit_in_workbook','whole_region_driver_trips':int(cg[11]),'all_nine_purposes_match':all(r['difference']==0 for r in driver_evidence),'native_workbook_unit':'トリップ数','vehicle_classes':'passenger/lightcar,goods/lightgoods,rental/carshare;privatebus,taxi coverage not established','not_applied':'No globaloccupancyfactor,no residence-ratio applied to origins,no finer vehicleOD invented'},'mean_time_reference':{'positive_core_car_pair_rows':len(positive),'missing_or_nonpositive_mean_rows':int(positive.mean_reference_status.ne('reported_positive_mean').sum())},'checks_passed':sum(checks.values()),'checks_total':len(checks),'formal_vehicle_demand_released':False,'traffic_assignment_run':False,'congestion_computed':False,'affected_people_computed':False,'hydraulic_campaign_modified':False}
 dump(R/'summary.json',summary);dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks})
 dump(P/'analysis_contract.json',{'plan_sha256':sha(P/'analysis_plan.json'),'source_acquisition_manifest_sha256':sha(S/'acquisition_manifest.json'),'decoded_manifest_sha256':sha(P/'decode_manifest.json'),'native_od_columns':{'origin':'発地','destination':'着地','purpose':'目的種類','car':'自動車','purpose_total':'計'},'source_rows':'CSV/XLSX row1based; derived columns preserve native keys and leading zeros','geographic_partition':{'fine_zone_codes':sorted(known),'unlocated_partition_codes':sorted(extras|{'8700','9999'}),'excluded_geographic_subtotals':sorted(set(unique_labels)-partition_codes),'validation':'615origin+615destination checks against native8888margins;28core margins against previouschart','8700_9999_note':'Native8700labels include含不明,but numerical margins require8700and9999 as separate components. Both kept in unresolved category without claiming precise geography. No nested subtotals added.'},'marginal_semantics':'within_core+outgoing=departure;within_core+incoming=arrival;touching=departure+arrival-within_core. Touching trips are not unique people or unique vehicles.','missing_pair_policy':'Do not generate unlisted OD cells or interpret unlisted pairs as measuredzero; published table is sparse. Reported zero means sample-expanded publishedzero,not proofno travel.','driver_interpretation':summary['driver_interpretation'],'halo':'Existing candidate zones intersect10km buffer; inclusiondoesnotmean full endpoint orroad coverage. Outside-zone flows retained. External-externalthroughtraffic requiresrouting,not derivedfrom core margins.','time':'DailyOD plus separate hourlymargins; no OD-by-hour jointinvented','population':'b4stayingpopulation downloadedbutnot substitutedfor2020nightbaseline,nopersonimpacts','next_steps':['Confirm5_3driver-onlyspecific convention before releasing vehicle-demandmodel','Preserve middle-zone drivercandidatetrip totals if later disaggregatedusingfinepersonOD; explicit allocationassumptions required','Build batchzone-network connectors and census-road correspondence','Calibrate normal volumes/speeds before evaluating floodedtraffic capacity and delay'],'formal_vehicle_demand_released':False})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('CHECKS',checks)
 if not all(checks.values()):raise SystemExit('ODanalysisvalidationfailed; preserveoutputsfordiagnosis')
if __name__=='__main__':main()
