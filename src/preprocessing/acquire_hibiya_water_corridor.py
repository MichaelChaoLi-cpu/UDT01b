"""Discover adjacent official source sheets from official index, verify archives.

Run with the project Python. No guessed archive names or height correction.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import hashlib,json,math,re,shutil,urllib.request,zipfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import pyproj,rasterio

ROOT=Path(__file__).resolve().parents[2]
RAW=moga_raw(ROOT, 'tokyo_lidar_dem/2024')
DEST=RAW/'hibiya_water_corridor'
OUT=ROOT/'data/exp/data-briefing/hibiya_water_corridor'
BASE='https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/'
TOLL=pyproj.Transformer.from_crs(6677,6668,always_xy=True)

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix(path.suffix+'.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as f:
                    shutil.copyfileobj(response,f,1024*1024)
                    expected=response.headers.get('Content-Length')
                if expected and part.stat().st_size!=int(expected):raise ValueError('HTTP length mismatch')
                part.replace(path);break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    return {'source_url':url,'local_path':moga_source(path, ROOT),'bytes':path.stat().st_size,'sha256':digest(path)}

def index_urls(kind,x,y):
    lon,lat=TOLL.transform(x,y);n=2**18
    tx=int((lon+180)/360*n);ty=int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
    rel=f'Vectortile/23ku/{kind}/18/{tx}/{ty}.pbf'
    path=DEST/'index'/f'{kind}_18_{tx}_{ty}.pbf'
    entry=fetch(BASE+rel,path)
    urls=sorted(set(v.decode() for v in re.findall(rb'https?://[^\s\x00-\x20<>\"]+?\.zip',path.read_bytes())))
    if not urls:raise ValueError('No download URL in official index: '+rel)
    return urls,entry

def acquire_zip(url,kind,tile):
    # Reuse original pilot files without duplicating large source data.
    existing={'grd50':RAW/'pilot_hibiya'/f'{tile}.zip','lp':RAW/'pilot_hibiya'/f'{tile}_lp.zip','ortho':RAW/'pilot_hibiya'/f'{tile}_ortho.zip'}[kind]
    path=existing if existing.exists() else DEST/kind/f'{tile}.zip'
    entry=fetch(url,path);entry.update(kind=kind,tile=tile)
    members=[];extracted=[]
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():continue
            if Path(info.filename).name!=info.filename:raise ValueError('Unexpected nested ZIP member')
            target=(RAW/'pilot_hibiya'/info.filename if kind in ['lp','grd50'] else RAW/'pilot_hibiya/orthophoto'/info.filename) if tile=='09LD2812' else DEST/kind/info.filename
            target.parent.mkdir(parents=True,exist_ok=True)
            # A full read validates each ZIP member's CRC, including existing files.
            with z.open(info) as src:
                h=hashlib.sha256()
                if target.exists():
                    while chunk:=src.read(1024*1024):h.update(chunk)
                    if h.hexdigest()!=digest(target):raise ValueError('Existing original differs from archive')
                else:
                    part=target.with_suffix(target.suffix+'.part')
                    with part.open('wb') as dst:
                        while chunk:=src.read(1024*1024):h.update(chunk);dst.write(chunk)
                    part.replace(target)
            members.append({'name':info.filename,'bytes':info.file_size,'zip_timestamp':list(info.date_time),'crc32':f'{info.CRC:08x}','sha256':h.hexdigest()})
            extracted.append(moga_source(target, ROOT))
    entry.update(zip_crc_verified=True,members=members,extracted=extracted)
    print('Verified',kind,tile,entry['bytes'],flush=True)
    return entry

def main():
    OUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    entries=[];candidate_urls=set()
    # 800 x 1200 m rectangle: pilot, both park ponds and complete nearby moat.
    for y in [-36450,-36150,-35850,-35550]:
        for x in [-7000,-6600]:
            urls,e=index_urls('grd50',x,y);candidate_urls.update(urls);entries.append(e)
    dem_entries=[];sheets=[]
    for url in sorted(candidate_urls):
        tile=Path(url).stem;e=acquire_zip(url,'grd50',tile);dem_entries.append(e)
        target=moga_project_path(ROOT, next(p for p in e['extracted'] if p.endswith('.tif')))
        with rasterio.open(target) as d:
            if d.res!=(.5,.5) or d.shape!=(600,800):raise ValueError('Unexpected native DEM grid')
            b=d.bounds
            if not (b.left<-6400 and b.right>-7200 and b.bottom<-35400 and b.top>-36600):continue
            sheets.append({'tile':tile,'bounds':list(b),'dem_path':moga_source(target, ROOT),'dem_crs_wkt':d.crs.to_wkt(),'dem_nodata':d.nodata})
    entries.extend(dem_entries)
    (OUT/'acquisition_manifest.json').write_text(json.dumps({'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'catalog':'https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024','resources':entries,'sheets':sheets,'acquisition_complete':False},ensure_ascii=False,indent=2)+'\n')
    jobs=[]
    for sheet in sheets:
        l,b,r,t=sheet['bounds'];x=(l+r)/2;y=(b+t)/2
        for kind in ['ortho','lp']:
            urls,e=index_urls(kind,x,y);entries.append(e)
            matches=[u for u in urls if Path(u).stem==sheet['tile']]
            if len(matches)!=1:raise ValueError('Matching official source URL not found for '+sheet['tile']+' '+kind)
            jobs.append((matches[0],kind,sheet['tile']))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(acquire_zip,*job) for job in jobs]):entries.append(future.result())
    manifest={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'catalog':'https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024','requested_bounds':[-7200,-36600,-6400,-35400],'resources':entries,'sheets':sheets,'acquisition_complete':True,'limits':['Product/ZIP timestamps are not flight dates.','DEM/LAS are a common survey source; point checks assess consistency, not independent accuracy.','No height-datum conversion or inferred water elevation.']}
    (OUT/'acquisition_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Acquisition complete',len(sheets),'DEM/ortho/LAS sheets',flush=True)

if __name__=='__main__':main()
