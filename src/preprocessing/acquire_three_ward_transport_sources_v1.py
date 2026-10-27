"""Acquire bounded official transport audit inputs; no credentials or accounts.
The road tiles are a schema sample, not a complete routable network.
"""
from pathlib import Path
import requests,json,re,hashlib,datetime,math
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_transport_framework_v1';S=OUT/'sources';S.mkdir(parents=True,exist_ok=True)
manifest=[]
def fetch(name,url):
 p=S/name
 if p.exists():
  b=p.read_bytes();status=403 if b'403' in b[:500] or b'AccessDenied' in b[:2000] or b'Access Denied' in b[:2000] else 200;cached=True;headers={}
 else:
  r=requests.get(url,timeout=90);b=r.content;status=r.status_code;headers=dict(r.headers);p.write_bytes(b);cached=False
 row={'file':str(p.relative_to(ROOT)),'url':url,'http_status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_or_verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cache_reused':cached,'etag':headers.get('ETag'),'last_modified':headers.get('Last-Modified')};manifest.append(row);print(name,status,len(b),flush=True)
 (S/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 return b
fetch('walking_catalog.json','https://www.geospatial.jp/ckan/api/3/action/package_show?id=0401')
fetch('ksj_n02_catalog.html','https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-2025.html')
fetch('gsi_road_readme.md','https://raw.githubusercontent.com/gsi-cyberjapan/experimental_rdcl/master/README.md')
fetch('N02-25_GML.zip','https://nlftp.mlit.go.jp/ksj/gml/data/N02/N02-25/N02-25_GML.zip')
cat=json.loads((S/'walking_catalog.json').read_text())['result'];lookup={r['id']:r for r in cat['resources']}
selected=[('walking_tokyo_2018_links.geojson','1e1e2208-0b2b-4038-ac04-9c0ac011eacf'),('walking_tokyo_2018_nodes.geojson','4b783ac2-fe76-4894-81e9-069223f7cc95'),('walking_central_2017_links.geojson','8182f873-a400-43e2-86a3-27b16538fa6f'),('walking_central_2017_nodes.geojson','c2b96fb8-5c4f-4673-a274-76bbee3e225a'),('walking_minato_2010_2011.zip','3a162f67-f1bc-4f86-83a8-24b81761613b')]
selected += [('walking_daimon_2018spec_links.geojson','52ad2ba4-dcb1-4715-8b44-6f3276ef4d44'),('walking_daimon_2018spec_nodes.geojson','bcaf44aa-ff76-4196-bbe4-20bc7b10f093')]
for name,rid in selected:fetch(name,lookup[rid]['url'])
(S/'selected_walking_metadata.json').write_text(json.dumps([lookup[rid] for name,rid in selected],ensure_ascii=False,indent=2)+'\n')
# Three dispersed sample tiles per ward, including likely cross-ward approaches.
coords=[(139.75,35.69),(139.775,35.70),(139.735,35.69),(139.78,35.67),(139.80,35.675),(139.78,35.65),(139.75,35.65),(139.73,35.66),(139.76,35.63)]
for lon,lat in coords:
 x=int((lon+180)/360*2**16);y=int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*2**16)
 fetch(f'gsi_rdcl_16_{x}_{y}.geojson',f'https://cyberjapandata.gsi.go.jp/xyz/experimental_rdcl/16/{x}/{y}.geojson')
# Current provider pages establish availability; legacy API docs do not prove a
# present-day station-entrance endpoint or complete floor/threshold attributes.
for name,url in [('metro_station_catalog.html','https://ckan.odpt.org/dataset/r_station-tokyometro'),('toei_station_catalog.html','https://ckan.odpt.org/dataset/r_station-toei'),('metro_otemachi_station.html','https://www.tokyometro.jp/station/otemachi/index.html'),('jr_tokyo_station.html','https://www.jreast.co.jp/estation/stations/1039.html'),('gsi_road_style.js','https://cyberjapandata.gsi.go.jp/xyz/experimental_rdcl/style_canvas.js')]:fetch(name,url)
(S/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

# Public metadata and field specification; no API registration or transit data request.
for name,url in [('walking_spec_2018.pdf','https://www.geospatial.jp/ckan/dataset/12fed482-7b89-4441-90ab-dcef0b753b7b/resource/a084d521-5f4d-4708-9d0d-02834b387f92/download/description201803.pdf'),('odpt_jreast_listing.html','https://ckan.odpt.org/dataset/?organization=jreast'),('odpt_routes_listing.html','https://ckan.odpt.org/en/dataset/?tags=%E8%B7%AF%E7%B7%9A%E7%B3%BB%E7%B5%B1%E6%83%85%E5%A0%B1-route_information'),('odpt_jr_station_catalog.html','https://ckan.odpt.org/dataset/jreast__r_station'),('odpt_jr_route_catalog.html','https://ckan.odpt.org/dataset/jreast__r_route'),('odpt_metro_route_catalog.html','https://ckan.odpt.org/dataset/r_route-tokyometro'),('odpt_toei_route_catalog.html','https://ckan.odpt.org/dataset/r_route-toei')]:fetch(name,url)
