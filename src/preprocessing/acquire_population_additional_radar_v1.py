"""Acquire three predeclared historical radar days; no hydraulic simulation.
Keep native 5-minute representative intensity; select nine-hour rain-only windows.
"""
from pathlib import Path
import os,sys,json,datetime as dt,time,concurrent.futures as cf
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np, requests, pandas as pd
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from acquire_tokyo_historical_radar_v1 import decode,sha,self_test
TAG='three_ward_population_additional_radar_v1';P=ROOT/'data/processed'/TAG;RAW=ROOT/'data/raw'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
BASE='http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/'
DATES=['2024-06-18','2024-07-31','2025-07-10'];UTC=dt.timezone.utc;JST=dt.timezone(dt.timedelta(hours=9))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda z:z.item()if isinstance(z,np.generic)else str(z),allow_nan=False)+'\n')
def get(stamp):
 name=f'Z__C_RJTD_{stamp:%Y%m%d%H%M%S}_RDR_JMAGPV_Ggis1km_Prr05lv_ANAL_grib2.bin';p=RAW/'grib2'/name;url=BASE+stamp.strftime('%Y/%m/%d/')+name
 if p.exists():data=p.read_bytes()
 else:
  for attempt in range(3):
   try:
    r=requests.get(url,timeout=40);r.raise_for_status();data=r.content;assert data[:4]==b'GRIB'and data[-4:]==b'7777'and len(data)<10*1024**2;break
   except Exception:
    if attempt==2:raise
    time.sleep(attempt+1)
  with p.open('xb')as f:f.write(data)
 level,rate,meta=decode(data);assert meta['utc']==stamp.isoformat()
 return level,rate,dict(**meta,url=url,path=str(p.relative_to(ROOT)),bytes=len(data),sha256=sha(data))
def main():
 for p in [P,RAW/'grib2',R]:p.mkdir(parents=True,exist_ok=True)
 script=sha(Path(__file__).read_bytes());decoder=ROOT/'src/preprocessing/acquire_tokyo_historical_radar_v1.py';self_test()
 protocol=dict(decision='MILI-D-20261018-012',dates_predeclared=DATES,product='Same JMA parameter203 1km Prr05lv representative intensity archived by Kyoto University RISH; not measured5min accumulation',cadence_s=300,survey='Complete calendar day JST; no outcome-based selection',window='Earliest maximum contributing-area-weighted rainfall9h/108frames; same source matrices as original cases',integration='End-stamped representative rate held over preceding300s; integrate rate/12; no calibration or scaling',missing='Any missing/saturated positive-support source blocks eligibility; no zero filling',budget=dict(files=864,max_bytes=2*1024**3,workers=4),new_hydraulics=False,training=False,sealed_test=False,source_url=BASE,script_sha256=script,decoder_sha256=sha(decoder.read_bytes()),dates_interpretation='Additional rain dates are candidate event parents, not independent random samples or validated flood reconstructions')
 f=P/'protocol.json'
 if f.exists():assert json.loads(f.read_text())==protocol
 else:dump(f,protocol)
 old=ROOT/'data/processed/tokyo_historical_radar_20240821_v1';mat=load_npz(old/'terrain_source_area_matrix.npz');area=np.asarray(mat.sum(0)).ravel();used=area>0;ward=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ward Code'].ravel();wa={str(k):np.asarray(mat[ward==k].sum(0)).ravel()for k in [1,2,3]};summaries=[];begun=time.monotonic()
 for day in DATES:
  out=P/('E'+day.replace('-',''));out.mkdir(exist_ok=True);start=dt.datetime.fromisoformat(day).replace(tzinfo=JST).astimezone(UTC);stamps=[start+dt.timedelta(seconds=300*(i+1))for i in range(288)]
  docs=[]
  for d in sorted({s.date()for s in stamps}):
   u=BASE+d.strftime('%Y/%m/%d/');f=RAW/(d.strftime('%Y%m%d')+'_index.html')
   if not f.exists():r=requests.get(u,timeout=35);r.raise_for_status();f.write_bytes(r.content)
   docs.append(dict(url=u,path=str(f.relative_to(ROOT)),sha256=sha(f.read_bytes())))
  gauge=f'https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no=44&block_no=47662&year={day[:4]}&month={int(day[5:7])}&day={int(day[8:10])}&view=p1';f=RAW/(day+'_jma_hourly.html')
  if not f.exists():r=requests.get(gauge,timeout=35);r.raise_for_status();f.write_bytes(r.content)
  docs.append(dict(url=gauge,path=str(f.relative_to(ROOT)),sha256=sha(f.read_bytes()),role='numeric gauge context only; no calibration or independent flood validation'))
  results=[]
  with cf.ThreadPoolExecutor(max_workers=4)as pool:
   for i,z in enumerate(pool.map(get,stamps),1):
    results.append(z)
    if i%48==0:print(json.dumps(dict(date=day,frames=i,total=288,elapsed_s=round(time.monotonic()-begun))),flush=True)
  rates=np.stack([z[1]for z in results]);levels=np.stack([z[0]for z in results]);flat=rates.reshape(288,-1);assert np.isfinite(flat[:,used]).all();assert not(levels.reshape(288,-1)[:,used]==251).any();avg=flat[:,used].astype('f8')@area[used]/area.sum();roll=np.convolve(avg/12,np.ones(108),'valid');i=int(roll.argmax());ends=np.array([s.timestamp()for s in stamps],dtype='i8');sel=slice(i,i+108);total=flat[sel].astype('f8').sum(0)/12
  np.savez_compressed(out/'survey_radar_intervals.npz',representative_intensity_mm_h=rates,level=levels,interval_end_unix_s=ends,interval_start_unix_s=ends-300)
  np.savez_compressed(out/'radar_intervals.npz',representative_intensity_mm_h=rates[sel],level=levels[sel],interval_end_unix_s=ends[sel],interval_start_unix_s=ends[sel]-300,latitude=48-(np.arange(1452,1500)+.5)/120,longitude=118+(np.arange(1716,1756)+.5)/80)
  summary=dict(event_id='E'+day.replace('-',''),parent_id='HIST-RADAR-TOKYO-'+day.replace('-',''),date_jst=day,window_start_jst=(stamps[i]-dt.timedelta(seconds=300)).astimezone(JST).isoformat(),window_end_jst=stamps[i+107].astimezone(JST).isoformat(),survey_frames=288,window_frames=108,window_start_index=i,survey_reconstructed_mm=float(avg.sum()/12),reconstructed_rain_mm=float(avg[sel].sum()/12),peak_rolling60min_reconstructed_mm=float(np.convolve(avg[sel]/12,np.ones(12),'valid').max()),peak5min_representative_mm_h=float(avg[sel].max()),rain_before_window_mm=float(avg[:i].sum()/12),rain_after_window_mm=float(avg[i+108:].sum()/12),ward_mean_reconstructed_mm={k:float(total@v/v.sum())for k,v in wa.items()},positive_support_source_cells=int(used.sum()),missing_supported=0,saturated_supported=0,raw_bytes=sum(z[2]['bytes']for z in results),eligible_same_product_rain_input=True,new_hydraulics=False,gauge_url=gauge)
  dump(out/'acquisition_manifest.json',dict(files=[z[2]for z in results],documents=docs,decoder_sha256=protocol['decoder_sha256']));dump(out/'event_summary.json',summary);summaries.append(summary)
  pd.DataFrame(dict(interval_end_jst=[s.astimezone(JST).isoformat()for s in stamps],catchment_mean_representative_mm_h=avg,in_selected_window=[i<=j<i+108 for j in range(288)])).to_csv(out/'rainfall_profile.csv',index=False)
  assert sum(p.stat().st_size for p in(RAW/'grib2').iterdir())<=2*1024**3
  print(json.dumps(summary),flush=True)
 dump(P/'event_inventory.json',dict(events=summaries,new_hydraulics=False));dump(R/'validation.json',dict(passed=True,events=3,frames=864,raw_bytes=sum(s['raw_bytes']for s in summaries),complete_support=True,seconds=time.monotonic()-begun,script_sha256=script,selection_uses_rainfall_only=True,new_hydraulics=False))
if __name__=='__main__':main()
