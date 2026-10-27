"""Acquire two independent-date5min radar parents; choose9h windows using rainfall alone."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,datetime as dt,concurrent.futures as cf,time,csv,re
import requests
import numpy as np
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from acquire_tokyo_historical_radar_v1 import decode,sha,self_test
from prepare_historical_surface_access_v1 import dump,write,load
TAG='three_ward_cross_event_protection_v1';RAW=ROOT/'data/raw'/TAG;P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG;D=ROOT/'data/exp/research-design'/TAG
OLD=ROOT/'data/processed/tokyo_historical_radar_20240821_v1';UTC=dt.timezone.utc;JST=dt.timezone(dt.timedelta(hours=9));BASE='http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/'
EVENTS=[dict(event_id='E20240830',parent_id='HIST-RADAR-TOKYO-20240829-30',survey_start_jst='2024-08-29T00:00:00+09:00',survey_end_jst='2024-08-31T00:00:00+09:00',candidate_role='Sustained-rainfall candidate; numerical profile will be reported, not presumed.',gauge_context='https://ds.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=44&block_no=47662&year=2024&month=8&view=p1'),dict(event_id='E20250911',parent_id='HIST-RADAR-TOKYO-20250911',survey_start_jst='2025-09-11T00:00:00+09:00',survey_end_jst='2025-09-12T00:00:00+09:00',candidate_role='Concentrated-rainfall candidate; numerical profile will be reported, not presumed.',gauge_context='https://ds.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=44&block_no=47662&year=2025&month=9&view=p1')]

def register_selection():
 for p in [RAW,P,R,D]:p.mkdir(parents=True,exist_ok=True)
 f=D/'rainfall_selection_protocol.json'
 protocol=dict(human_decision='MILI-D-20261017-011',events=EVENTS,source_product='Unchanged JMA parameter203 five-minute1km composite representative intensity archived by RISH; embedded representative table and original decoder.',rejected_early_candidates='2023-06-02 and2023-09-08 archives inspected: available tar Prr10lv has10min sampling; not used to avoid mixing product cadence.',window_rule='For each complete survey, select earliest9h/108consecutive5min interval window maximizing total source-to-recipient contributing-area-weighted reconstructed rainfall. No flooding, route loss or intervention benefit considered.',rate_rule='Hold end-stamped representative intensity over preceding300s, as in original. Not a measured five-minute accumulation.',missing_policy='Any missing or saturated source with positive catchment area blocks delivery; no zero filling or sample replacement.',geometry='Reuse frozen terrain/urban conservative source-area matrices and same0.5m roof redistribution; no change to city geometry.',initial_condition='Same dry initial surface and9h horizon for controlled comparison; survey rainfall outside selected window and prior wetness not applied. This is not a full historic flood reconstruction.',stage='Use the same elapsed-time low/high1.5m7group stage scenarios, not observed event river levels or tide. No optimization of stage alignment for a favorable result.',frozen_measures=str((ROOT/'data/processed/three_ward_motor_road_protection_v1/frozen_selection_binding.json').relative_to(ROOT)),acquisition_caps=dict(total_max_files=864,total_max_bytes=2*1024**3,workers=4,retries_per_file=2),hydraulic_runs_started=0,not_done=['No training','No sealed-test opening','No affected-person counts','No actual historical traffic closure claim'])
 if f.exists():assert json.loads(f.read_text())==protocol
 else:dump(f,protocol)
 return protocol

def fetch(stamp):
 name=f'Z__C_RJTD_{stamp:%Y%m%d%H%M%S}_RDR_JMAGPV_Ggis1km_Prr05lv_ANAL_grib2.bin';url=BASE+f'{stamp:%Y/%m/%d}/'+name;path=RAW/'grib2'/name
 if path.exists():data=path.read_bytes();reused=True
 else:
  reused=False
  for attempt in range(3):
   try:
    r=requests.get(url,timeout=45);r.raise_for_status();data=r.content;assert data[:4]==b'GRIB'and data[-4:]==b'7777';assert len(data)<10*1024**2;break
   except Exception:
    if attempt==2:raise
    time.sleep(attempt+1)
  with path.open('xb')as f:f.write(data)
 level,rate,meta=decode(data);assert meta['utc']==stamp.isoformat()
 return level,rate,dict(**meta,url=url,path=str(path.relative_to(ROOT)),bytes=len(data),sha256=sha(data),reused=reused)

def main():
 start=time.monotonic();protocol=register_selection();self_test();(RAW/'grib2').mkdir(exist_ok=True);mat=load_npz(OLD/'terrain_source_area_matrix.npz');areas=np.asarray(mat.sum(axis=0)).ravel();used=areas>0;mask=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ward Code'].ravel();wardareas={str(code):np.asarray(mat[mask==code].sum(axis=0)).ravel()for code in [1,2,3]}
 # Preserve source index and numeric gauge context pages; only rainfall files drive selection.
 docs=[]
 for event in EVENTS:
  stamps=[dt.datetime.fromisoformat(event['survey_start_jst']).astimezone(UTC)+dt.timedelta(minutes=5*(i+1))for i in range(int((dt.datetime.fromisoformat(event['survey_end_jst'])-dt.datetime.fromisoformat(event['survey_start_jst'])).total_seconds()/300))];assert len(stamps)in [288,576]
  out=P/'rainfall'/event['event_id'];out.mkdir(parents=True,exist_ok=True)
  dates=sorted({s.date()for s in stamps})
  for day in dates:
   u=BASE+day.strftime('%Y/%m/%d/');f=RAW/(event['event_id']+'_'+day.strftime('%Y%m%d')+'_index.html')
   if not f.exists():q=requests.get(u,timeout=45);q.raise_for_status();f.write_bytes(q.content)
   docs.append(dict(url=u,path=str(f.relative_to(ROOT)),sha256=sha(f.read_bytes())))
  f=RAW/(event['event_id']+'_jma_daily.html')
  if not f.exists():q=requests.get(event['gauge_context'],timeout=45);q.raise_for_status();f.write_bytes(q.content)
  docs.append(dict(url=event['gauge_context'],path=str(f.relative_to(ROOT)),sha256=sha(f.read_bytes()),role='context only; no gauge calibration or independent validation'))
  results=[]
  with cf.ThreadPoolExecutor(max_workers=4)as pool:
   for k,x in enumerate(pool.map(fetch,stamps)):
    results.append(x)
    if(k+1)%48==0:
     dump(R/'acquisition_state.json',dict(status='acquiring_radar',event_id=event['event_id'],frames_done=k+1,event_total=len(stamps),elapsed_s=time.monotonic()-start,hydraulic_runs_started=0));print('RADAR',event['event_id'],k+1,'/',len(stamps),round(time.monotonic()-start),flush=True)
  levels=np.stack([x[0]for x in results]);rates=np.stack([x[1]for x in results]);manifest=[x[2]for x in results];flat=rates.reshape(len(stamps),-1);assert np.isfinite(flat[:,used]).all(),'Missing positive-support radar';assert not(levels.reshape(len(stamps),-1)[:,used]==251).any(),'Saturated positive-support radar'
  avg=flat[:,used].astype('f8')@areas[used]/areas.sum();totals=np.convolve(avg/12,np.ones(108),mode='valid');i=int(np.argmax(totals));selected=np.arange(i,i+108);ends=np.array([s.timestamp()for s in stamps],dtype='i8');window_start=stamps[i]-dt.timedelta(seconds=300);window_end=stamps[i+107]
  np.savez_compressed(out/'survey_radar_intervals.npz',representative_intensity_mm_h=rates,level=levels,interval_end_unix_s=ends,interval_start_unix_s=ends-300)
  np.savez_compressed(out/'radar_intervals.npz',representative_intensity_mm_h=rates[selected],level=levels[selected],interval_end_unix_s=ends[selected],interval_start_unix_s=ends[selected]-300,latitude=48-(np.arange(1452,1500)+.5)/120,longitude=118+(np.arange(1716,1756)+.5)/80)
  chosen=avg[selected];roll60=np.convolve(chosen/12,np.ones(12),'valid');source_total=np.nan_to_num(flat[selected],nan=0).sum(axis=0)/12
  event_summary=dict(**event,window_start_utc=window_start.isoformat(),window_end_utc=window_end.isoformat(),window_start_jst=window_start.astimezone(JST).isoformat(),window_end_jst=window_end.astimezone(JST).isoformat(),window_start_index=i,survey_frames=len(stamps),window_frames=108,positive_catchment_radar_cells=int(used.sum()),catchment_area_m2=float(areas.sum()),reconstructed_rain_mm=float(chosen.sum()/12),peak5min_representative_mm_h=float(chosen.max()),peak_rolling60min_reconstructed_mm=float(roll60.max()),peak60_fraction_of9h=float(roll60.max()/(chosen.sum()/12)),wet_minutes_above1mm_h=int((chosen>1).sum()*5),window_fraction_of_survey_rain=float(chosen.sum()/avg.sum()),rain_before_window_reconstructed_mm=float(avg[:i].sum()/12),rain_after_window_reconstructed_mm=float(avg[i+108:].sum()/12),ward_mean_reconstructed_mm={code:float(source_total@wa/wa.sum())for code,wa in wardareas.items()},unweighted_positive_source_total_cv=float(source_total[used].std()/source_total[used].mean()),source_bytes=sum(x['bytes']for x in manifest),missing_supported=0,saturated_supported=0,hydraulic_runs_started=0)
  dump(out/'event_summary.json',event_summary);dump(out/'acquisition_manifest.json',dict(files=manifest,decoder=str((ROOT/'src/preprocessing/acquire_tokyo_historical_radar_v1.py').relative_to(ROOT)),decoder_sha256=sha((ROOT/'src/preprocessing/acquire_tokyo_historical_radar_v1.py').read_bytes())));write(out/'rainfall_profile.csv',[dict(interval_end_jst=stamps[j].astimezone(JST).isoformat(),catchment_mean_mm_h=float(avg[j]),in_selected_window=bool(i<=j<i+108))for j in range(len(stamps))]);print('SELECTED',event['event_id'],event_summary['window_start_jst'],event_summary['reconstructed_rain_mm'],flush=True)
  assert sum(p.stat().st_size for p in(RAW/'grib2').iterdir())<=protocol['acquisition_caps']['total_max_bytes']
 dump(RAW/'source_documents.json',docs);dump(P/'event_selection.json',dict(events=[json.loads((P/'rainfall'/e['event_id']/'event_summary.json').read_text())for e in EVENTS],selection_protocol_sha256=sha((D/'rainfall_selection_protocol.json').read_bytes()),selection_uses_rainfall_only=True,measures_not_reselected=True));dump(R/'acquisition_state.json',dict(status='radar_acquisition_qualified',events=2,files=864,elapsed_s=time.monotonic()-start,hydraulic_runs_started=0));print('RADAR ACQUISITION QUALIFIED',flush=True)
if __name__=='__main__':main()
