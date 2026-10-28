
# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
from urllib.parse import urljoin
from html.parser import HTMLParser
import requests,re,csv,json,shutil,hashlib
ROOT=Path('/Volumes/Extension/Research/UDT01b');RAW=moga_raw(ROOT, 'river_boundary_review/2026-10-14');PRO=ROOT/'data/processed/three_ward_road_connectivity';OUT=ROOT/'data/exp/data-briefing/three_ward_road_connectivity';RAW.mkdir(parents=True,exist_ok=True)
base='https://www.data.jma.go.jp/kaiyou/db/tide/genbo/genbo.php?stn=TK&year=2024&month=08&G_HOUR=on&LV='
entries=[('jma_tide_hourly_DL.html',base+'DL'),('jma_tide_hourly_TP.html',base+'TP'),('jma_tide_202408.html','https://www.data.jma.go.jp/kaiyou/db/tide/genbo/genbo.php?stn=TK&year=2024&month=08'),('jma_tide_explanation.html','https://www.data.jma.go.jp/kaiyou/db/tide/genbo/explanation.html'),('shibuya_furukawa.html','https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/shibuya'),('kanda_management.html','https://www.kensetsu.metro.tokyo.lg.jp/jimusho/sanken/kasen/kasen_kanri')]
s=Path('/private/tmp/jma_tide_hourly_DL.html').read_text();txt=re.search(r'href="([^"]+hry202408TK.txt)"',s)[1];entries.append(('hry202408TK.txt',urljoin(base,txt)));entries.append(('jma_tide_format.html','https://www.data.jma.go.jp/kaiyou/db/tide/genbo/format.html'));manifest=[]
for name,url in entries:
 p=RAW/name
 if not p.exists():
  cache=Path('/private/tmp')/name
  if cache.exists():shutil.copyfile(cache,p)
  else:r=requests.get(url,timeout=(15,35));r.raise_for_status();p.write_bytes(r.content)
 manifest.append({'file':moga_source(p, ROOT),'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
class T(HTMLParser):
 def __init__(self):super().__init__();self.active=False;self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t=='table' and dict(a).get('summary')=='毎時潮位' and dict(a).get('border')=='1':self.active=True
  if self.active and t=='tr':self.row=[]
  if self.active and t in ['th','td']:self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if self.active and t in ['td','th'] and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
  if self.active and t=='tr':self.rows.append(self.row);self.row=None
  if self.active and t=='table':self.active=False
parsed={}
for lv in ['DL','TP']:
 t=T();t.feed((RAW/f'jma_tide_hourly_{lv}.html').read_text());parsed[lv]={int(r[0]):r[1:] for r in t.rows if len(r)==25 and r[0].isdigit()};assert len(parsed[lv])==31
rows=[];dlzero=-188.4
assert '-188.4(cm)' in (RAW/'jma_tide_hourly_DL.html').read_text()
for day in [21,22]:
 for h in range(24):
  dlraw=parsed['DL'][day][h];tpraw=parsed['TP'][day][h];dl=float(dlraw) if dlraw!='-' else None;tp=float(tpraw) if tpraw!='-' else None
  if dl is not None and tp is not None:assert abs(tp-(dl+dlzero))<=.6
  rows.append({'station':'Tokyo_Harumi_TK','time_jst':f'2024-08-{day:02d}T{h:02d}:00:00+09:00','observed_hourly_smoothed_cm_above_DL':dl,'official_display_elevation_cm':tp,'elevation_m_from_DL_and_2024_reference':(dl+dlzero)/100 if dl is not None else None,'DL_elevation_cm_for_2024':dlzero,'raw_DL':dlraw,'raw_TP_display':tpraw,'quality':'normal' if dl is not None else 'missing','terrain_epoch_alignment':'not yet verified with PLATEAU source height version','use':'Sea-side background only; hourly filtered values are not river-stage hydrographs or minute-scale storm maxima.'})
with (PRO/'tokyo_harumi_tide_20240821_22.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'boundary_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');note={'records':len(rows),'event_reference_DL_cm':dlzero,'DL_TP_display_consistency':'all48 within display rounding tolerance0.6cm','all48_hourly_values_available':all(r['quality']=='normal' for r in rows),'smoothed_observations':True,'known_filter_effect':'JMA removes periods up toabout3h; does not preserve rapid storm/surge response.','not_incorporated_into_road_on_off_test':True};(OUT/'tide_acquisition_validation.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n');print(note,flush=True)
