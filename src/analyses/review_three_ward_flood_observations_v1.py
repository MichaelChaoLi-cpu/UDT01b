"""Freeze public road-flood evidence without reading flood-model predictions.

Acquisition is separate. Manually reviewed assertions below preserve the difference
between a source claim, a validation label, and a geotemporally admissible sample.
Run locally: .venv/bin/python src/analyses/review_three_ward_flood_observations_v1.py
"""
from pathlib import Path
import csv, datetime as dt, hashlib, html, json, shutil
ROOT=Path(__file__).resolve().parents[2]
TAG='three_ward_road_flood_observations_v1'
P=ROOT/'data/processed'/TAG
R=ROOT/'data/exp/data-briefing'/TAG

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(path,rows,fields=None):
 fields=fields or list(rows[0])
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

# All notes are paraphrases of the listed public sources; none are simulated facts.
NOTES={
'S001':('candidate','Resident reports a flooded Takanawadai school route on September 11. Municipality confirms dismissal began about 14:15, not the flood onset. Road segment, observation time and depth are missing. Updated 2025-12-03.','resident_account_on_official_site'),
'S002':('coverage_context','Catalog says reports are incomplete. No selected-event negative observations can be inferred from absent map entries.','official_catalog_reviewed'),
'S003':('coverage_context','Rendered map legend has nine events, latest 2021-07-11; combines basement inundation and road flooding. August 2026 map update is not an observation date. None of the five selected events is separately mapped.','pdf_text_and_visual_review'),
'S004':('coverage_context','Historical 1982–2008 maps. Inundation areas are inferred from road records rather than surveyed flood boundaries. Does not cover the selected five events.','official_catalog_reviewed'),
'S005':('coverage_context','Catalog last updated 2025-06-24. Listed 2024 events July6, July20–21 and August21 differ from selected dates. Resident-phone reports are incomplete and approximate. Predates both 2025 events.','official_catalog_reviewed'),
'S006':('external_duplicate','Secondary report describes R357 Tokyo Port tunnel flooding; 15:49 announcement, 16:26 one direction reopened, other direction still closed at 18:00. Not a Minato surface-road observation. Direction/stage must be reconciled with S019.','secondary_text_reviewed'),
'S007':('external_duplicate','Secondary headline says all restrictions lifted by 19:25. Published 23:11:55. Do not substitute this for original operator log or infer continuous inundation between closure and reopening.','secondary_text_reviewed'),
'S008':('external_duplicate','Regional July31 article published 19:31. Followed original image credit to S025: Nerima, Kawagoe Kaido near Kitamachi1. Not a three-ward scene.','article_credit_traced_to_original'),
'S009':('unverified_video_lead','Search metadata mentions Odaiba heavy rain about 15:00–15:10, not road inundation. Downloaded HTML has generic YouTube title and no usable video description; no video/frame review. Not admitted.','metadata_only_content_not_retrieved'),
'S010':('external_context','Next-day ANN report published 2025-09-12 15:15. Named flood scenes include wards outside the three-ward scope. No matched core road/time pair extracted. Current weather widgets are not event evidence.','article_text_only_no_frame_review'),
'S011':('coverage_context','Tokyo historical address search covers 1974–2021 at neighborhood scale. Major-event maps have reporting thresholds and approximate locations, not complete road wet/dry observations.','official_catalog_reviewed'),
'S012':('unavailable_forecast','HTTP429; no retry bypass. Forecast possible closures, not proof an actual road flooded.','not_retrieved'),
'S013':('forecast_only','June18 NEXCO notice concerns possible weather closures. No confirmed three-ward flooded road observation.','notice_reviewed'),
'S014':('external_candidate','TBS reports flooded/closed R246 in Setagaya on August30. Outside the three wards; exact observation time is not supplied by the article publication time 16:57.','primary_media_text_only'),
'S015':('acquisition_route','Weathernews retrospective describes archived report/photo/video maps. No licensed historical export acquired; public article is not a three-ward observation table.','article_reviewed_no_archive_export'),
'S016':('outside_event','Minato August21 2024 actual-inundation map belongs to a different storm; excluded from these five events.','pdf_event_identity_reviewed'),
'S017':('susceptibility_only','List of underpasses susceptible to flooding, not an event record. A listed site is neither a positive nor a negative observation.','document_role_reviewed'),
'S018':('acquisition_route','Chiyoda MyCityReport reporting service records location and photos; page discourages posting during disasters and routes emergencies to telephone. No historical case export was obtained.','official_service_page_reviewed'),
'S019':('external_candidate','Primary operator notice explicitly confirms R357 Tokyo Port tunnel road flooding and Kanagawa-bound closure from 16:30 September11. Restriction starts at Koto Aomi1/Odaiba Chuo junction; not a core surface-road observation. Closure start is not flood onset.','pdf_text_and_visual_review'),
'S020':('warning_only','Minato July10 19:03 heavy-rain/flood warning. Warning does not demonstrate road flooding or dry conditions.','warning_text_reviewed'),
'S021':('rail_context','Mainichi July10 event video page, published July11, describes rail suspensions. No timestamped core road scene obtained; video not downloaded or frame-reviewed.','video_page_text_only'),
'S022':('external_context','August30 Weathernews image credits include Setagaya and Ninomiya. No verified core-ward photo/time pair. Publication 15:58 is not capture time.','article_text_and_credits_only'),
'S023':('rail_context','June18 ANN rail-service report published 17:12 after the 16:45 simulation end; not a core road inundation observation.','article_text_reviewed'),
'S024':('unavailable_locator','SSL certificate verification failed. No certificate bypass or alternate private access. A repost locator, not an independently admitted source.','not_retrieved'),
'S025':('external_candidate','Original user report identifies Kawagoe Kaido near Kitamachi1, Nerima; display timestamp July31 19:20. Not within the three wards. Report time is not independently verified camera capture time.','original_report_text_and_metadata_no_image_review'),
}

def case(cid,event,location,ward,scope,layer,reported,state,sources,reason,**extra):
 r=dict(candidate_id=cid,event_id=event,location_text=location,ward_code=ward,scope=scope,road_layer=layer,
  reported_state=reported,verification_state=state,admission='held' if scope=='core_ward' else 'excluded',
  source_ids=sources,duplicate_family=cid,observation_start_jst=None,observation_end_jst=None,
  observation_time_basis='unknown',publication_or_update_jst=None,report_time_jst=None,
  operational_time_jst=None,operational_time_meaning=None,geometry=None,road_id=None,
  location_precision='named_area_or_corridor_only',depth_lower_m=None,depth_upper_m=None,
  reason=reason,required_for_admission='',frame_reviewed=False)
 r.update(extra);return r
CASES=[
 case('C001','E20250911','高輪台小学校の通学路（具体路段不明）','13103','core_ward','surface_road_reported',
  'resident_reports_flooding','unknown',['S001'],
  '居民冠水报告；非区方现场测量。缺具体道路和冠水时刻，无法配对10 m网格。14:15仅是开始放学时间。',
  publication_or_update_jst='2025-12-03',operational_time_jst='2025-09-11T14:15:00+09:00',
  operational_time_meaning='school dismissal began approximately; NOT flood onset',
  required_for_admission='Original image/incident record with road segment and capture/observation time; no need to identify children.'),
 case('C002','E20250911','国道357号 東京港トンネル 海側・神奈川方面／青海1丁目規制起点','13108','outside_core','tunnel',
  'operator_confirms_flooding_and_closure','flooded_confirmed',['S019','S006','S007'],
  '官方确认冠水并封路，但属于外围隧道；不是三区地面水深标签。16:30是该方向管制时间，不是冠水开始时间。',
  publication_or_update_jst='2025-09-11',operational_time_jst='2025-09-11T16:30:00+09:00',
  operational_time_meaning='Kanagawa-bound restriction start in S019; secondary 15:49/16:26/19:25 claims retained separately',
  required_for_admission='Not admissible to core surface-road benchmark. Future tunnel/outer-network study would require original direction-specific closure logs and inundation location/time.'),
 case('C003','E20240731','川越街道 北町一丁目付近（練馬区）','13120','outside_core','surface_road_reported',
  'original_user_reports_flooding','unknown',['S025','S008'],
  '沿署名追到原帖后确认在练马区，不能因转载标题写“都心”而归入三区；19:20是报告显示时间，拍摄时刻未独立确认。',
  report_time_jst='2024-07-31T19:20:00+09:00',publication_or_update_jst='2024-07-31T19:31:00+09:00',
  required_for_admission='Outside scope. Original photo capture time and precise segment would still be required for independent image validation.'),
 case('C004','E20240830','国道246号 世田谷区（报道未给具体路段）','13112','outside_core','surface_road',
  'primary_media_reports_flooded_road_and_closure','flooded_confirmed',['S014'],
  '原始新闻正文明确道路冠水、封路，但在世田谷区；16:57是文章发布时间，不是现场观测时刻。视频逐帧未核验。',
  publication_or_update_jst='2024-08-30T16:57:00+09:00',
  required_for_admission='Outside scope; named road alone cannot identify 10 m cells. Need exact scene/segment and observation time.'),
]


def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(R/'qa').mkdir(exist_ok=True)
 sources=json.loads((P/'source_acquisition.json').read_text())
 assert set(NOTES)=={r['source_id'] for r in sources}
 for r in sources:
  r['evidence_role'],r['review_note'],r['review_method']=NOTES[r['source_id']]
  r['retrieval_is_observation_time']=False
  if r['status']=='downloaded':assert sha(ROOT/r['local_path'])==r['sha256']
 dump(P/'source_register.json',sources);table(P/'source_register.csv',sources,sorted(set().union(*(r.keys() for r in sources))))
 event_inputs=[ROOT/'data/processed/three_ward_five_event_population_validation_v1/event_selection.json',ROOT/'data/processed/three_ward_cross_event_protection_v1/event_selection.json']
 events=[]
 for f in event_inputs:
  for e in json.loads(f.read_text())['events']:
   events.append({k:e[k] for k in ['event_id','parent_id','window_start_jst','window_end_jst'] }|{'metadata_source':str(f.relative_to(ROOT)),'metadata_sha256':sha(f),'prewindow_rain_mm':e.get('rain_before_window_mm',e.get('rain_before_window_reconstructed_mm',0))})
 events.sort(key=lambda e:e['event_id']);assert len(events)==5
 for e in events:assert (dt.datetime.fromisoformat(e['window_end_jst'])-dt.datetime.fromisoformat(e['window_start_jst'])).total_seconds()==32400
 dump(P/'event_windows.json',events);table(P/'event_windows.csv',events)
 protocol=dict(version='1.0',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope_wards={'13101':'Chiyoda','13102':'Chuo','13103':'Minato'},events=[e['event_id'] for e in events],
  purpose='Find independent observed road-flood benchmarks; do not validate the model with model-generated labels.',
  state_definitions={
   'flooded_confirmed':'Primary operator or primary contemporaneous news explicitly confirms road inundation. Source-level confirmation is separate from geotemporal admission; resident-only uncorroborated accounts remain unknown.',
   'not_flooded_confirmed':'Explicit contemporaneous view or observation of an identifiable visible road footprint without standing inundation; applies only to that footprint and time, never to the whole event.',
   'unknown':'Missing/ambiguous road identity, actual road state, or uncorroborated report. Warnings, rain, moving vehicles and silence do not establish a dry road.'},
  admission_requirements=['Selected event identity independently verified','Road footprint/segment can be spatially matched, with documented positional uncertainty','Observation/capture time or bounded interval overlaps model window; publication/dismissal/closure times kept separately','Confirmed road state supported by original record or reviewed frame','Correct surface/tunnel/bridge level; core ground benchmark excludes tunnel-only observations','Duplicate families counted once; depth only where independently measurable'],
  temporal_rule='Retain observation uncertainty intervals and compare only overlapping saved model times. Do not round an unknown capture time to a 5-minute frame. Date-only evidence remains qualitative.',
  dry_rule='No reports or an unmarked catalog location never constitute a negative sample.',
  independence='No simulated water depth, predicted wet location, predicted road closure or intervention result is read by this script. Candidates were selected from public source searches.',
  restrictions=['No parameter tuning','No new hydrodynamic simulations','No training','No sealed-test access','No affected-person calculation','No cloud upload','No messages/applications sent'],
  caveat='Search is non-exhaustive and indexing/access are incomplete. 15 event–ward coverage rows are evidence gaps, not 15 dry observations. No accuracy statistics can be computed from this inventory.',
  inspection_limits='PDFs S003/S019 rendered and visually reviewed; media article text/credits reviewed. No road image/video frames geolocated or visually admitted in this pass.')
 dump(P/'protocol.json',protocol)
 dump(P/'candidate_observations.json',CASES);table(P/'candidate_observations.csv',CASES)
 admitted=[c for c in CASES if c['admission']=='admitted']
 table(P/'admitted_observations.csv',admitted,list(CASES[0]))
 dump(P/'admitted_observations.geojson',{'type':'FeatureCollection','features':[]})
 dump(P/'candidate_locations.geojson',{'type':'FeatureCollection','features':[{'type':'Feature','geometry':c['geometry'],'properties':{k:v for k,v in c.items() if k!='geometry'}} for c in CASES]})
 # Null geometries are deliberate. Never place the school centroid at the reported flood.
 coverage=[]
 for e in events:
  for ward,name in protocol['scope_wards'].items():
   matched=[c for c in CASES if c['event_id']==e['event_id'] and c['ward_code']==ward and c['scope']=='core_ward']
   coverage.append(dict(event_id=e['event_id'],ward_code=ward,ward=name,window_start_jst=e['window_start_jst'],window_end_jst=e['window_end_jst'],
    admissible_flooded=sum(c['admission']=='admitted' and c['verification_state']=='flooded_confirmed' for c in matched),admissible_not_flooded=0,
    held_core_candidates=len(matched),candidate_ids=[c['candidate_id'] for c in matched],evidence_status='reported_flood_unmatched' if matched else 'no_admissible_evidence_found',
    road_state='unknown',catalog_sources={'13101':['S004','S011','S018'],'13102':['S002','S003','S011'],'13103':['S005','S011']}[ward],
    interpretation='检索证据缺口，不代表整区未浸水；未对该区所有道路作观测。'))
 table(P/'coverage_matrix.csv',coverage)
 dump(P/'coverage_matrix.json',coverage)
 closure_assertions=[
  dict(candidate_id='C002',source_id='S019',time_jst='2025-09-11T16:30:00+09:00',assertion='Kanagawa-bound closure starts',evidence='primary_operator',use_as_flood_onset=False),
  dict(candidate_id='C002',source_id='S006',time_jst='2025-09-11T15:49:00+09:00',assertion='Operator announcement of both-direction closure as relayed by secondary article',evidence='secondary_operator_quote',use_as_flood_onset=False),
  dict(candidate_id='C002',source_id='S006',time_jst='2025-09-11T16:26:00+09:00',assertion='One direction reopened as relayed by secondary article',evidence='secondary',use_as_flood_onset=False),
  dict(candidate_id='C002',source_id='S007',time_jst='2025-09-11T19:25:00+09:00',assertion='All restrictions lifted by headline time, original log not acquired',evidence='secondary',use_as_flood_onset=False)]
 table(P/'closure_assertions.csv',closure_assertions)
 exclusions=[dict(source_id=r['source_id'],event_id=r['event_id'],role=r['evidence_role'],reason=r['review_note']) for r in sources if r['evidence_role'] not in ['candidate','external_candidate']]
 table(P/'excluded_and_context_sources.csv',exclusions)
 requests=[
  dict(priority=1,target='Minato / Takanawadai September11 record holders',event_id='E20250911',scope='Only the reported road footprint; exclude personal/child identifiers',needed='Original incident photo or road/complaint log: street segment, observation time interval, surface versus underground, wet/dry evidence, origin/reuse conditions',reason='Resolve C001 without treating 14:15 school dismissal as flood onset',contacted=False),
  dict(priority=2,target='Three ward road-maintenance / disaster-response archives',event_id='all_five',scope='13101;13102;13103; windows in event_windows.csv',needed='Patrol/callout/closure logs with cause, location, start/end and any timestamped inspection or camera evidence. Request explicitly dry inspections as well as positive flood reports.',reason='Paired positive and observed negative samples; absence of incident is not a negative',contacted=False),
  dict(priority=3,target='Original media or Weathernews historical report archive',event_id='all_five',scope='Three ward polygons only; separate Tokyo Chuo from other cities and Togoshi Ginza from Ginza',needed='Original scene ID, date, capture/report/publication timestamps separately, camera location/direction, original image rights. No license/export acquired in this task.',reason='Recover geotemporal metadata; count repost families once',contacted=False),
  dict(priority=4,target='Tokyo National Highway Office R357 original operator log',event_id='E20250911',scope='External tunnel; separate from core ground-road labels',needed='Direction-specific restriction/relief times and actual flooded segment; reconcile S019 with secondary reports',reason='Optional external transport evidence, not an excuse to enlarge core benchmark',contacted=False)]
 table(P/'evidence_gap_requests.csv',requests)
 query_batches=json.loads((P/'search_log.json').read_text())
 queries=[dict(batch_id=f'Q{i+1:02}',query_index=j+1,query=q['q'],syntax_caveat='First two queries in Q16 use malformed site syntax; not counted as strict domain searches' if i==15 and j<2 else '') for i,b in enumerate(query_batches) for j,q in enumerate(b['queries'])]
 table(P/'search_queries.csv',queries)
 for src,dst in [('/tmp/udt_chuo_actual_flood.png','S003_reviewed_map.png'),('/tmp/udt_r357_closure.png','S019_reviewed_notice.png')]:
  if Path(src).exists():shutil.copyfile(src,R/'qa'/dst)
 review_receipt={'S003':{'review':'Text and rendered map checked; nine event legend entries, latest 2021-07-11. Mixed basement/road symbols.','image':'qa/S003_reviewed_map.png'},'S019':{'review':'Text and rendered one-page notice checked. September11 2025, 16:30 Kanagawa-bound restriction; Koto Aomi1 start.','image':'qa/S019_reviewed_notice.png'}}
 dump(R/'pdf_review_receipt.json',review_receipt)
 checked=[]
 def check(name,ok):
  assert ok,name;checked.append({'check':name,'passed':True})
 check('Exactly five frozen 9-hour windows; JST offsets preserved',len(events)==5)
 check('All source hashes match downloaded snapshots',all(r['status']!='downloaded' or sha(ROOT/r['local_path'])==r['sha256'] for r in sources))
 check('Exactly fifteen event–ward gap rows',len(coverage)==15 and len({(r['event_id'],r['ward_code']) for r in coverage})==15)
 check('Every candidate has reviewed source identifiers',all(set(c['source_ids'])<=set(NOTES) for c in CASES))
 check('No unknown state admitted',all(c['verification_state']!='unknown' for c in admitted))
 check('No closure/publication/dismissal time substituted for observation',all(c['observation_start_jst'] is None and c['observation_end_jst'] is None for c in CASES))
 check('No coordinates/depths/road IDs fabricated',all(c['geometry'] is None and c['depth_lower_m'] is None and c['depth_upper_m'] is None and c['road_id'] is None for c in CASES))
 check('No unsupported dry labels created',not any(c['verification_state']=='not_flooded_confirmed' for c in CASES))
 check('School flood assertion kept as uncorroborated resident account',CASES[0]['verification_state']=='unknown' and CASES[0]['scope']=='core_ward')
 check('Tunnel kept outside core surface labels',CASES[1]['scope']=='outside_core' and CASES[1]['road_layer']=='tunnel')
 check('Weathernews original location resolved to Nerima',CASES[2]['ward_code']=='13120' and CASES[2]['admission']=='excluded')
 check('No candidate accepted with missing geometry/time',not admitted)
 check('All saved PDFs have separate review status',all(r['review_method'] in ['pdf_text_and_visual_review','pdf_event_identity_reviewed','document_role_reviewed'] for r in sources if r.get('local_path','').endswith('.pdf')))
 summary=dict(source_records=len(sources),downloaded_source_files=sum(r['status']=='downloaded' for r in sources),unavailable_sources=[r['source_id'] for r in sources if r['status']!='downloaded'],
  candidate_records=len(CASES),core_ward_candidates=1,core_ward_unknown=1,external_candidates=3,admitted_core_flooded=0,admitted_core_not_flooded=0,admitted_core_total=0,
  external_source_confirmed_flooded=2,event_ward_scopes=15,query_batches=len(query_batches),query_entries=len(queries),
  interpretation='An evidence inventory and gap audit, not completed observational validation. No inference that any event/ward was dry.',new_simulations_started=0,parameters_changed=0)
 dump(R/'summary.json',summary)
 dump(R/'validation.json',{'passed':True,'checks':checked,'scope':'Inventory integrity/classification only, NOT validation of hydraulic predictions','summary':summary})
 text='''# Road-flood observation inventory v1

This is an evidence inventory, not a calibrated/validated flood model. Source acquisition
and processing scripts never read simulated depths or predicted road closures.

- `event_windows.csv`: five existing 9-hour JST windows and metadata hashes.
- `source_register.csv`: 25 source attempts, retrieval receipts, content inspection limits.
- `candidate_observations.csv`: one core unresolved resident account plus three external records.
- `closure_assertions.csv`: original versus secondary operational time assertions, not flood onset.
- `coverage_matrix.csv`: fifteen event–ward evidence gaps, not road observations or dry labels.
- `admitted_observations.csv/.geojson`: intentionally empty. No road geometry was fabricated.
- `candidate_locations.geojson`: null geometry is deliberate; school/corridor names are not flood points.
- `evidence_gap_requests.csv`: prioritized data requirements; no contacts or requests sent.
- `search_queries.csv` and `search_log.json`: non-exhaustive query receipts, not proof of completeness.

`verification_state` describes source-supported road state. `admission` separately
requires correct scope/layer and a matchable road footprint and observation time.
For example, an official tunnel closure confirms a flooded road at event level but
is excluded from core surface-depth validation. An uncorroborated resident account
remains unknown pending precise location/time and original evidence.

No recorded inundation is never equivalent to recorded non-inundation. A dry label
requires an actually observed road footprint/time; moving cars or rain imagery alone
are insufficient. All depth fields are null. No F1/IoU/depth error is computed.

Media frames were not independently reviewed. Two PDFs were rendered and inspected;
source role and figure legends were checked. Failed HTTP/TLS access was not bypassed.

Run: `.venv/bin/python src/analyses/review_three_ward_flood_observations_v1.py`
Acquisition: `.venv/bin/python src/preprocessing/acquire_three_ward_flood_observations_v1.py --ids S001`
Existing source snapshots are reused by default; use --refresh only intentionally.
'''
 (P/'README.md').write_text(text)
 # Standalone local review; no external JS, telemetry, map tiles, uploads or video embeds.
 esc=lambda x:html.escape(str(x))
 def source_links(ids):return ' · '.join(f'<a target="_blank" rel="noopener" href="{esc(next(s["url"] for s in sources if s["source_id"]==sid))}">{sid}</a>' for sid in ids)
 cards=''.join(f'<div class="card"><strong>{v}</strong><span>{k}</span></div>' for k,v in [('可配对的三区浸水样本',0),('已观测未浸水样本',0),('三区内待核实线索',1),('保留的外围道路案例',3)])
 eventrows=''.join(f'<tr><td>{e["event_id"]}</td><td>{e["window_start_jst"][:10]}</td><td>{e["window_start_jst"][11:16]}–{e["window_end_jst"][11:16]}</td><td>{e["prewindow_rain_mm"]:.2f} mm</td></tr>' for e in events)
 caserows=''.join(f'<tr data-event="{c["event_id"]}" data-scope="{c["scope"]}"><td><b>{c["candidate_id"]}</b><br>{c["event_id"]}</td><td>{esc(c["location_text"])}</td><td>{esc(c["verification_state"])}<br><small>{esc(c["reported_state"])}</small></td><td>{esc(c["reason"])}</td><td>{source_links(c["source_ids"])}</td></tr>' for c in CASES)
 sourcerows=''.join(f'<tr><td>{s["source_id"]}</td><td><a href="{esc(s["url"])}" target="_blank" rel="noopener">{esc(s["title"])}</a></td><td>{esc(s["status"])}<br>{esc(s["evidence_role"])}</td><td>{esc(s["review_note"])}</td></tr>' for s in sources)
 gaprows=''.join(f'<tr><td>{e["event_id"]}</td>'+''.join('<td class="gap">居民冠水线索，待定位定时</td>' if e['event_id']=='E20250911' and w=='13103' else '<td class="gap">未知／未找到合格观测</td>' for w in protocol['scope_wards'])+'</tr>' for e in events)
 reqrows=''.join(f'<li><b>{esc(r["target"])}</b><p>{esc(r["needed"])}</p></li>' for r in requests)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road Flood Evidence — Five Historical Events</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#18283a;font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1320px;margin:auto;padding:32px 26px 70px}h1{font-size:32px;line-height:1.25;margin:12px 0}h2{font-size:22px;margin:34px 0 12px}p{max-width:1050px}a{color:#126098}small{font-size:12px;color:#5c6877}header small{letter-spacing:.1em}.notice{background:#fff4da;border-left:5px solid #c98a19;padding:15px 20px;border-radius:5px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card{background:white;border:1px solid #dae1e9;border-radius:10px;padding:18px}.card strong{display:block;font-size:36px;color:#245a81}.card span{font-size:14px}.wrap{overflow:auto;border:1px solid #dae1e9;border-radius:7px;background:#fff}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid #e3e8ee}th{background:#eaf0f5}td:first-child{white-space:nowrap}.gap{color:#785b22;background:#fffaf0}select{padding:7px;margin:0 12px 12px 0;border:1px solid #bdc9d5;border-radius:4px}details{margin-top:22px}summary{cursor:pointer;font-weight:650}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.qa img{width:100%;border:1px solid #dde3eb}.qa{background:#fff;padding:12px;border-radius:8px}footer{margin-top:30px;color:#637184;font-size:13px}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.cols{grid-template-columns:1fr}main{padding:20px 14px}h1{font-size:26px}table{min-width:680px}}
</style><main><header><small>OBSERVATIONAL EVIDENCE AUDIT · CHIYODA / CHUO / MINATO</small><h1>五场历史暴雨：道路冠水验证证据盘点</h1><p>先核实真实道路状态，再与模拟配对。检索覆盖道路冠水、媒体现场线索和封路记录；不调参、不追加模拟。</p></header><div class="notice"><b>本轮未形成可用于10 m网格和统一时刻对照的三区验证样本。</b><br>“暂无合格观测”不等于“未浸水”，也不意味着现有模型已经验证通过。外围确有冠水事件，但不能移作三区地面道路标签。</div>CARDS
<h2>五个事件与既有模拟窗口</h2><p>全部为日本时间 JST。只读取既有事件元数据，未读取预测水深来挑选样本。窗口前降雨仅提醒初始状态解释限制，本次不修改模拟。</p><div class="wrap"><table><thead><tr><th>Event</th><th>Date</th><th>9-hour window (JST)</th><th>Pre-window rain</th></tr></thead><tbody>EVENTS</tbody></table></div>
<h2>候选与外围案例：状态和准入分开记录</h2><p>只有 C001 是三区内道路冠水线索，尚未确认到具体路段与观测时刻。C002、C004 的来源明确确认道路冠水，但位于核心范围外；C003 为原始居民报告，追溯后确认在练马区。下表不等于4条已准入样本。</p><label>事件 <select id="event"><option value="all">All events</option>OPTIONS</select></label><label>范围 <select id="scope"><option value="all">All records</option><option value="core_ward">Core three wards</option><option value="outside_core">Outside core</option></select></label><div class="wrap"><table id="cases"><thead><tr><th>ID / Event</th><th>Location</th><th>Evidence state</th><th>为何保留／隔离</th><th>原始来源</th></tr></thead><tbody>CASES</tbody></table></div><p id="empty" hidden>当前筛选没有候选记录；不表示该事件道路未浸水。</p>
<h2>三区 × 五场事件：证据覆盖缺口</h2><p>这15格是检索覆盖清单，不是15个道路观测。公开检索并不完备；官方目录未收录、网页无法访问、搜索未命中都不能生成负标签。</p><div class="wrap"><table><thead><tr><th>Event</th><th>Chiyoda</th><th>Chuo</th><th>Minato</th></tr></thead><tbody>GAPS</tbody></table></div>
<h2>哪些资料不能直接拿来验证</h2><div class="cols"><section><b>官方履历图：先看事件和图例</b><p>中央区2026年更新的图例仍只列到2021年的历史事件，并混合地下室进水与道路冠水。千代田区目录覆盖1982–2008年；港区2025年6月更新的目录既不含本次选定的2024日期，也早于两场2025事件。均不能给五场暴雨提供“未浸水”标签。</p></section><section><b>时间、地名与路层：避免错配</b><p>14:15放学、16:30管制、新闻发布时间和实际拍摄时间分别保存。户越银座不属于中央区银座；东京港隧道不等于港区地面道路。隧道冠水需要独立路层处理，不能直接检验地面二维水深。</p></section></div>
<h2>准入与三类状态</h2><ul><li><b>已确认浸水：</b>原始运营记录或原始新闻明确报告道路冠水。进入定量验证还须有可匹配路段、正确路层和观测时间。</li><li><b>已确认未浸水：</b>有具体时刻、具体可见路面范围的明确观测；不能由“没有新闻”“车辆在行驶”推断。本轮0条。</li><li><b>未知：</b>居民未获佐证的报告、地点或时间不明、只有雨情／预警、或者根本没有道路状态观测。</li></ul><p>原始影像页面已追查，但本轮没有完成任何三区道路视频的逐帧定位或水深量测。两份关键PDF已核查文字与图面。没有可靠空间范围，故没有捏造经纬度点，也没有绘制伪观测地图。</p>
<h2>最有价值的补证据顺序</h2><ol>REQUESTS</ol><p>以上是待获取字段清单；尚未联系机构、发送申请或购买影像。拿到同路段、同时间的正负样本后，才能评估命中、漏报、误报；独立水深证据不足时不计算水深误差。</p>
<details><summary>25条来源尝试与核查说明</summary><p>文件下载成功不等于内容可用。S009仅获取通用页面；S012受HTTP429限制，S024证书校验失败，未绕过访问限制。不同媒体转载不重复计数。</p><div class="wrap"><table><thead><tr><th>ID</th><th>Source</th><th>Status / role</th><th>Review notes</th></tr></thead><tbody>SOURCES</tbody></table></div></details>
<details><summary>PDF图面复核</summary><div class="cols"><div class="qa"><p>S003 · Chuo historical map</p><a href="../../../processed/three_ward_road_flood_observations_v1/sources/S003.pdf"><img src="qa/S003_reviewed_map.png" alt="Reviewed Chuo map with dated historical-event legend"></a></div><div class="qa"><p>S019 · Primary R357 closure notice</p><a href="../../../processed/three_ward_road_flood_observations_v1/sources/S019.pdf"><img src="qa/S019_reviewed_notice.png" alt="Reviewed official September11 R357 closure notice"></a></div></div></details>
<h2>可复用文件</h2><p><a href="../../../processed/three_ward_road_flood_observations_v1/candidate_observations.csv">Candidate records CSV</a> · <a href="../../../processed/three_ward_road_flood_observations_v1/coverage_matrix.csv">15-scope coverage CSV</a> · <a href="../../../processed/three_ward_road_flood_observations_v1/source_register.csv">Source register CSV</a> · <a href="../../../processed/three_ward_road_flood_observations_v1/evidence_gap_requests.csv">Evidence gaps CSV</a> · <a href="validation.json">Inventory validation</a></p><footer>Public-source retrieval: 2026-10-20 · 5 existing events · Non-exhaustive search · No new simulations, tuning, training, population counts or cloud uploads · Framelog MILI-D-20261020-019</footer></main><script>
function filter(){let n=0;document.querySelectorAll('#cases tbody tr').forEach(r=>{let show=(eventFilter.value==='all'||r.dataset.event===eventFilter.value)&&(scopeFilter.value==='all'||r.dataset.scope===scopeFilter.value);r.hidden=!show;if(show)n++});document.getElementById('empty').hidden=n!==0}const eventFilter=document.getElementById('event'),scopeFilter=document.getElementById('scope');eventFilter.addEventListener('change',filter);scopeFilter.addEventListener('change',filter);
</script></html>'''
 for key,val in [('CARDS','<div class="cards">'+cards+'</div>'),('EVENTS',eventrows),('OPTIONS',''.join(f'<option>{e["event_id"]}</option>' for e in events)),('CASES',caserows),('GAPS',gaprows),('REQUESTS',reqrows),('SOURCES',sourcerows)]:page=page.replace(key,val)
 (R/'review.html').write_text(page)
 dump(R/'code_binding.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__).resolve(),ROOT/'src/preprocessing/acquire_three_ward_flood_observations_v1.py']})
 dump(R/'artifact_hashes.json',{str(f.relative_to(ROOT)):sha(f) for f in sorted(P.rglob('*')) if f.is_file()})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('Review:',R/'review.html')
if __name__=='__main__':main()
