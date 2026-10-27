"""Acquire exact-ID official census map geometries andattributes for core observations."""
from pathlib import Path
import json,csv,hashlib,datetime as dt,concurrent.futures,requests,time
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';S=P/'sources';B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1'
BASE='https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/'
def main():
 S.mkdir(exist_ok=True);core=list(csv.DictReader((B/'core_road_attributes.csv').open(encoding='utf-8-sig')));obs=list(csv.DictReader((B/'core_directional_observations.csv').open(encoding='utf-8-sig')));ids=sorted({r['section_id']for r in core}|{r['native_observation_section_id']for r in obs if r['native_observation_section_id'].isdigit()});jobs=[(i,typ)for i in ids for typ in ['geometry','attributes']]
 def get(job):
  i,typ=job;rel=('census/'+i[:2]+'/'+i+'.geojson')if typ=='geometry'else'censusprop/'+i[:2]+'/'+i+'.json';url=BASE+rel;f=S/(i+('.geojson'if typ=='geometry'else'.json'));status=200;last=None
  if not f.exists():
   r=requests.get(url,timeout=35);status=r.status_code
   if status==200:
    data=json.loads(r.content.decode("utf-8-sig"));f.write_bytes(r.content);last=r.headers.get('last-modified')
   elif status!=404:r.raise_for_status()
  rec={'section_id':i,'kind':typ,'url':url,'status':status,'path':str(f.relative_to(ROOT))if f.exists()else None,'bytes':f.stat().st_size if f.exists()else 0,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()if f.exists()else None,'last_modified':last}
  if f.exists():
   raw=f.read_text(encoding='utf-8-sig');data=json.loads(raw)
   if typ=='geometry':assert all(str(x['properties']['R03cen'])==i for x in data['features'])
   else:assert data['交通調査基本区間番号']==i
  return rec
 out=[];start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
  for n,r in enumerate(pool.map(get,jobs),1):
   out.append(r)
   if n%100==0:print('CENSUS SOURCE',n,'/',len(jobs),'seconds',round(time.monotonic()-start),flush=True)
 features=[]
 for r in out:
  if r['kind']=='geometry'and r['status']==200:
   d=json.loads((ROOT/r['path']).read_text(encoding='utf-8-sig'));features.extend(d['features'])
 (P/'official_census_geometries_4326.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n');(S/'acquisition_manifest.json').write_text(json.dumps({'acquired_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'official_map':BASE+'webmap.html','target':'Core391census sections andtheir actualobservationsectionIDs,not nationwidecollection','files':out,'unique_section_ids':len(ids),'http_404':sum(r['status']==404 for r in out)},ensure_ascii=False,indent=2)+'\n');print('CENSUS GEOMETRIES READY',len(features),'features',flush=True)
if __name__=='__main__':main()
