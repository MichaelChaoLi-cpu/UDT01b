"""Verify public traffic-data availability and fields; no demand assignment or impact counts."""
from pathlib import Path
import requests,csv,io,json,hashlib,datetime as dt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_traffic_data_search_v1';R=ROOT/'data/exp/feasibility-check/three_ward_traffic_data_search_v1';S=P/'sources'

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
 S.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 files=[('mlit_tokyo_2021_sections.csv','https://www.mlit.go.jp/road/census/r3/data/csv/kasyo13.csv'),('mlit_tokyo_2021_hourly.csv','https://www.mlit.go.jp/road/census/r3/data/csv/zkntrf13.csv'),('tokyo_wards_2021_hourly.csv','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000028577')];records=[]
 for name,url in files:
  f=S/name
  if not f.exists():q=requests.get(url,timeout=45);q.raise_for_status();f.write_bytes(q.content)
  b=f.read_bytes();text=None;encoding=None
  for enc in ['utf-8-sig','cp932']:
   try:text=b.decode(enc);encoding=enc;break
   except UnicodeDecodeError:pass
  assert text is not None;rows=list(csv.reader(io.StringIO(text)));records.append(dict(url=url,path=str(f.relative_to(ROOT)),bytes=len(b),sha256=sha(b),encoding=encoding,csv_rows_including_headers=len(rows),first_row_columns=len(rows[0]),first_row_preview=rows[0][:12]));print(name,len(b),'bytes',len(rows),'rows',flush=True)
 section=list(csv.DictReader(io.StringIO((S/files[0][0]).read_bytes().decode('cp932'))));header=list(section[0]);core=[r for r in section if r['市区町村コード']in ['13101','13102','13103']]
 rawrows=list(csv.reader(io.StringIO((S/files[0][0]).read_bytes().decode('cp932'))))
 with(P/'three_ward_census_sections.csv').open('w',newline='',encoding='utf-8-sig')as f:w=csv.writer(f);w.writerow(rawrows[0]);w.writerows(row for row in rawrows[1:]if row[18]in ['13101','13102','13103'])
 field_groups={k:[h for h in header if any(p in h for p in needles)]for k,needles in {'traffic_volume':['自動車類交通量'],'observed_estimated_flag':['観測・非観測'],'travel_speed':['旅行速度'],'geometry':['幅員','車線数','交差点密度','青時間比','指定最高速度'],'congestion':['混雑度'],'direct_capacity':['交通容量'],'location':['市区町村コード','観測地点地名']}.items()};dump(P/'census_field_inventory.json',field_groups)
 counts={c:sum(r['市区町村コード']==c for r in core)for c in ['13101','13102','13103']}
 # Do not equate a nonempty numeric entry with a verified measurement or infer usable capacity here.
 summary=dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),downloaded_sources=records,tokyo_census_section_rows=len(section),three_ward_section_rows=len(core),three_ward_section_rows_by_code=counts,core_census_sections_not_all_local_roads=True,field_groups=field_groups,direct_capacity_column_present=bool(field_groups['direct_capacity']),capacity_interpretation='Volume,lanes,width,speeds,intersection fields andcongestion degree can constrain a capacity model; no per-link veh/hour capacity has been estimated here. Survey/estimated flags,original time units and methodology must be retained.',pt_catalog=dict(url='https://www.tokyo-pt.jp/data/H30',year=2018,status='Official catalog andFAQ verified by web search; dataset bytes not obtained in this check.',confirmed_tables=['d-1: purpose x main mode OD trips,planning basic zones','c-2/c-3: departure/arrival time-of-day trip ends,planning basic zones','b-4: time-of-day staying population,planning basic zones','e-2: mean zone-to-zone travel time','5-3: purpose x tolled-road-use car OD,middle zones','500m mesh trip-end aggregates; incomplete geocoding and suppression cautions apply'],access='Public predefined tables are separate from custom data aggregation,which needs registration. No account created orpersonal information submitted.',transport_units='Trips/persons and driver-only vehicle-trip counts differ; main-mode car person trips are not directly vehicle counts.',access_limitation='Direct requests andsystemcurl encountered TLS chain verification errors on PT host; do not mark catalogs missing anddo not disable verification.'),census_od_population=dict(url='https://www.e-stat.go.jp/stat-search/database?layout=datalist&toukei=00200521&tstat=000001136464&tclass1=000001136469',year=2020,role='Commuter/student origin-destination and usual transport-mode constraints; not all-purpose daily trips,concurrent daytime presence or observed vehicles.'),hypotheses_if_missing=['Within-zone origin/destination placement andzone-to-road connectors','Unobserved minor-road capacity andspeed assumptions','Conversion from daily OD to event-hour trips where suitable joint time-by-OD tables are unavailable','Storm-related trip cancellation,departure shifts,rerouting,rail-to-car switching and vehicle occupancy'],not_done=['No traffic assignment','No congestion ordelay estimates','No affected-person counts','No capacity values orOD invented','No changes to running8case campaign'])
 dump(R/'summary.json',summary);dump(S/'acquisition_manifest.json',dict(files=records));print('TRAFFIC SOURCE AUDIT',len(core),'core surveysections;capacitycolumn',bool(field_groups['direct_capacity']),flush=True)
if __name__=='__main__':main()
