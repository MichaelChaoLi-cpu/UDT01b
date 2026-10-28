"""Audit official KSJ layers against the existing study boundaries, without changing heights."""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for key in ('PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA'):
    os.environ.pop(key, None)
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from hashlib import sha256
from datetime import datetime, timezone
from collections import Counter
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
import json
from shapely.geometry import shape, mapping, Point, LineString, Polygon, box
from shapely.ops import transform, unary_union
from shapely import make_valid
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RAW = moga_raw(ROOT, 'ksj')
OUT = ROOT / 'data/exp/data-briefing/ksj_tokyo'
PROC = ROOT / 'data/processed/ksj_tokyo'
for d in (RAW, OUT, PROC): d.mkdir(parents=True, exist_ok=True)
G = '{http://www.opengis.net/gml/3.2}'
K = '{http://nlftp.mlit.go.jp/ksj/schemas/ksj-app}'
X = '{http://www.w3.org/1999/xlink}'
old_to_local = Transformer.from_crs(4612, 6677, always_xy=True)
new_to_local = Transformer.from_crs(6668, 6677, always_xy=True)
local_to_new = Transformer.from_crs(6677, 6668, always_xy=True)
boundaries = json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())
wards = {f['properties']['ward']: transform(new_to_local.transform, shape(f['geometry'])) for f in boundaries['features']}
union = unary_union(list(wards.values()))
corridor = box(-7200, -36600, -6400, -35400)
sources = [
    ('W05', 'KsjTmplt-W05.html', 'W05-08_13_GML.zip', '2008 FY', 'river centerlines and nodes', 'JGD2000', 'non-commercial'),
    ('W09', 'KsjTmplt-W09-v2_2.html', 'W09-05_GML.zip', '2005-09-01', 'selected lake polygons', 'JGD2000', 'commercial use permitted'),
    ('W07', 'KsjTmplt-W07.html', None, '2009-03-27', '100 m basin grid', 'JGD2000', None),
    ('P22', 'KsjTmplt-P22.html', 'P22-12_13_GML.zip', '2012 FY', 'sewer-law pump and treatment facilities', 'JGD2000', 'non-commercial'),
    ('A51', 'KsjTmplt-A51-2025.html', 'A51-25_13_GML.zip', '2025 edition; Tokyo members named 2024', 'scenario inundation-depth polygons', 'JGD2011', 'CC BY 4.0'),
]
archive_urls = {
    'W05':'https://nlftp.mlit.go.jp/ksj/gml/data/W05/W05-08/W05-08_13_GML.zip',
    'W09':'https://nlftp.mlit.go.jp/ksj/gml/data/W09/W09-05/W09-05_GML.zip',
    'P22':'https://nlftp.mlit.go.jp/ksj/gml/data/P22/P22-12/P22-12_13_GML.zip',
    'A51':'https://nlftp.mlit.go.jp/ksj/gml/data/A51/A51-25/A51-25_13_GML.zip',
}
def snapshot(s):
    code,page,*_=s
    url = 'https://nlftp.mlit.go.jp/ksj/gml/datalist/'+page
    b = urlopen(url, timeout=45).read()
    p=RAW/page; p.write_bytes(b)
    return code, {'metadata_url':url,'snapshot':moga_source(p, ROOT),'sha256':sha256(b).hexdigest()}
meta = dict(ThreadPoolExecutor(5).map(snapshot, sources))

def xml_features(path, tag):
    with ZipFile(path) as z:
        root=ET.fromstring(z.read(next(n for n in z.namelist() if n.endswith('.xml') and not n.startswith('KS-'))))
    ids={e.get(G+'id'):e for e in root if e.get(G+'id')}
    def coords(e):
        vals=[float(v) for v in ''.join(e.itertext()).split()]
        return [(vals[i+1], vals[i]) for i in range(0,len(vals),2)] # GML is latitude,longitude
    def curve(ref):
        e=ids[ref.lstrip('#')]
        if e.tag==G+'OrientableCurve':
            c=curve(e.find(G+'baseCurve').get(X+'href'))
            return c[::-1] if e.get('orientation')=='-' else c
        return coords(e.find('.//'+G+'posList'))
    def ring(e):
        result=[]
        for m in e.findall('.//'+G+'curveMember'):
            c=curve(m.get(X+'href'))
            result.extend(c[1:] if result and c[0]==result[-1] else c)
        return result
    result=[]
    for e in root.findall(K+tag):
        properties={'source_id':e.get(G+'id'), 'feature_type':tag}
        def preserve(element, prefix=''):
            for child in element:
                key=prefix+child.tag.split('}')[-1]
                if len(child): preserve(child,key+'.')
                elif child.text and child.text.strip():properties[key]=child.text.strip()
                elif child.get(X+'href'):properties[key]=child.get(X+'href')
        preserve(e)
        if tag=='Lake':
            surface=ids[e.find(K+'bounds').get(X+'href').lstrip('#')]
            patch=surface.find('.//'+G+'PolygonPatch')
            geo=Polygon(ring(patch.find(G+'exterior')), [ring(x) for x in patch.findall(G+'interior')])
        elif tag=='Stream':geo=LineString(curve(e.find(K+'location').get(X+'href')))
        else:
            pt=ids[e.find(K+'position').get(X+'href').lstrip('#')]
            geo=Point(coords(pt.find(G+'pos'))[0])
        if not geo.is_valid:geo=make_valid(geo)
        result.append((transform(old_to_local.transform,geo),properties))
    return result

def save_fc(path, features):
    path.write_text(json.dumps({'type':'FeatureCollection','features':[
        {'type':'Feature','geometry':mapping(transform(local_to_new.transform,g)), 'properties':p} for g,p in features]},ensure_ascii=False),encoding='utf-8')

audit=[]
for code,page,filename,epoch,role,crs,license_name in sources:
    entry={'code':code,'catalog_epoch':epoch,'role':role,'horizontal_crs':crs,'license':license_name,**meta[code]}
    if filename:
        p=RAW/filename; b=p.read_bytes()
        entry['archive_url']=archive_urls[code]
        with ZipFile(p) as z:
            entry.update({'archive':moga_source(p, ROOT), 'bytes':len(b), 'sha256':sha256(b).hexdigest(), 'crc_passed':z.testzip() is None, 'members':z.namelist()})
        if code=='A51':
            with ZipFile(p) as z:
                j=json.loads(z.read(next(n for n in z.namelist() if n.endswith('.geojson'))))
            features=[(transform(new_to_local.transform,shape(f['geometry'])),f['properties']) for f in j['features']]
            entry['municipalities']=dict(Counter(p['A51_003'] for g,p in features))
            entry['member_crs']=j.get('crs')
        elif code=='W05': features=xml_features(p,'Stream')
        elif code=='W09': features=xml_features(p,'Lake')
        else:features=xml_features(p,'PumpingPlantFacility')+xml_features(p,'SewageTreatmentPlantFacility')
        entry['feature_count']=len(features)
        selected=[(g.intersection(union),p) for g,p in features if g.intersects(union)]
        entry['central_three_wards_count']=len(selected)
        entry['by_ward']={name:sum(g.intersects(boundary) for g,p in features) for name,boundary in wards.items()}
        entry['hibiya_corridor_count']=sum(g.intersects(corridor) for g,p in features)
        namekey={'W05':'riverName','W09':'lakeName','P22':'facilityName','A51':'A51_003'}[code]
        entry['central_three_wards_names']=sorted(set(p.get(namekey,'') for g,p in selected))
        entry['hibiya_corridor_features']=[p for g,p in features if g.intersects(corridor)]
        if code=='P22':
            midpoint=corridor.centroid
            entry['nearest_five_facilities_to_corridor_center']=[{'distance_m':round(g.distance(midpoint),1),**p} for g,p in sorted(features,key=lambda f:f[0].distance(midpoint))[:5]]
        save_fc(PROC/(code+'_central_three_wards.geojson'),selected)
        if code=='W05':
            save_fc(PROC/'W05_river_nodes_central_three_wards.geojson',[(g,p) for g,p in xml_features(RAW/filename,'RiverNode') if g.intersects(union)])
    else:entry['downloaded']=False
    audit.append(entry)
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'layers':audit,'overlay_crs':'EPSG:6677','outputs_crs':'EPSG:6668',
        'method':'Parse official GML references and latitude/longitude axis order; project JGD2000 or JGD2011 to EPSG:6677; intersect existing ward boundaries. No height transformation.',
        'limitations':['Horizontal datum conversion uses installed PROJ transformations; not a claim of survey-grade epoch reconciliation.', 'W05 node elevation is derived from 50m DEM, not bank crest, riverbed or water level.', 'P22 facility location or proximity does not prove service area or hydraulic connectivity.', 'A51 absence in these wards is lack of package coverage, not zero inundation risk.']}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:e.get(k) for k in ('code','feature_count','central_three_wards_count','central_three_wards_names','hibiya_corridor_count','municipalities')} for e in audit],ensure_ascii=False,indent=2))
