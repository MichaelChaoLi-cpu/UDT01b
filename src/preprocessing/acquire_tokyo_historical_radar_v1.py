"""Acquire the public JMA/RISH numerical radar archive; no hydrodynamic runs.

A five-minute representative-intensity hold is an explicit reconstruction rule,
not a claim that radar level values are observed accumulated precipitation.
"""
from pathlib import Path
import os
for _key in ["PROJ_LIB", "PROJ_DATA", "GDAL_DATA"]: os.environ.pop(_key, None)
import concurrent.futures as cf
import csv, datetime as dt, hashlib, json, re, time
import numpy as np
import requests
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
TAG = 'tokyo_historical_radar_20240821_v1'
RAW = ROOT / 'data/raw' / TAG
OUT = ROOT / 'data/processed' / TAG
REPORT = ROOT / 'data/exp/data-preprocessing' / TAG
JST = dt.timezone(dt.timedelta(hours=9))
UTC = dt.timezone.utc
BASE = 'http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/2024/08/21/'
START = dt.datetime(2024, 8, 21, 15, tzinfo=JST)
END = dt.datetime(2024, 8, 22, 0, tzinfo=JST)
# Exact published rational grid; do not propagate the rounded 8333-microdegree header increment.
R0, R1 = 1452, 1500  # 35.9 to 35.5 N, north first
C0, C1 = 1716, 1756  # 139.45 to 139.95 E


def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def u(s, a, b): return int.from_bytes(s[a:b], 'big')


def expand_rle(payload, vmax, n):
    values, counts = [], []
    current, count, power = None, 0, 1
    base = 255 - vmax
    if not 1 <= base <= 254: raise ValueError('Unsupported RLE base')
    total = 0
    for b in payload:
        if b <= vmax:
            if current is not None:
                values.append(current); counts.append(count); total += count
                if total > n: raise ValueError('RLE exceeds grid')
            current, count, power = b, 1, 1
        else:
            if current is None: raise ValueError('RLE count before first level')
            count += (b - vmax - 1) * power
            power *= base
            if count > n or power > (n + 1) * base: raise ValueError('Invalid run length')
    if current is None: raise ValueError('Empty RLE payload')
    values.append(current); counts.append(count)
    if sum(counts) != n: raise ValueError('RLE decoded point count mismatch')
    return np.repeat(np.asarray(values, dtype=np.uint8), counts)


def decode(data):
    if data[:4] != b'GRIB' or data[7] != 2 or u(data, 8, 16) != len(data) or data[-4:] != b'7777':
        raise ValueError('Invalid GRIB2 framing')
    sections = {}; offset = 16
    while offset < len(data) - 4:
        length = u(data, offset, offset + 4)
        if length < 5 or offset + length > len(data) - 4: raise ValueError('Invalid section length')
        s = data[offset:offset + length]; number = s[4]
        if number in sections: raise ValueError('Unexpected multiple fields')
        sections[number] = s; offset += length
    if offset != len(data)-4 or set(sections) != {1,3,4,5,6,7}: raise ValueError('Unexpected GRIB layout')
    s1, s3, s4, s5, s6 = [sections[k] for k in [1,3,4,5,6]]
    stamp = dt.datetime(u(s1,12,14), *s1[14:19], tzinfo=UTC)
    end = dt.datetime(u(s4,34,36), *s4[36:41], tzinfo=UTC)
    assert u(s1,5,7)==34 and s1[11]==0 and s1[20]==0
    assert len(s3)==72 and u(s3,12,14)==0 and s3[14]==4 and s3[71]==0
    assert (u(s3,30,34), u(s3,34,38), u(s3,6,10)) == (2560,3360,8601600)
    assert (u(s3,46,50),u(s3,50,54),u(s3,55,59),u(s3,59,63)) == (47995833,118006250,20004167,149993750)
    assert (u(s3,63,67),u(s3,67,71)) == (12500,8333)
    assert len(s4)==82 and u(s4,7,9)==50008 and s4[9:13]==bytes([1,203,0,201])
    assert s4[17]==0 and u(s4,18,22)==0x80000005 and end==stamp
    assert s4[46:49]==bytes([1,2,0]) and u(s4,49,53)==5
    assert u(s5,5,9)==8601600 and u(s5,9,11)==200 and s5[11]==8
    vmax, m, scale = u(s5,12,14), u(s5,14,16), s5[16]
    assert m==251 and len(s5)==17+2*m and scale==2 and vmax<=m
    assert len(s6)==6 and s6[5]==255
    representative = np.r_[np.nan, np.frombuffer(s5[17:],dtype='>u2').astype(float)/10**scale]
    assert representative[1]==0
    level = expand_rle(sections[7][5:],vmax,8601600).reshape(3360,2560)[R0:R1,C0:C1].copy()
    metadata = {'utc':stamp.isoformat(), 'vmax':vmax, 'parameter':203,
        'radar_operation_hex':s4[58:66].hex(), 'rain_coefficient_operation_hex':s4[66:74].hex(),
        'embedded_representative_table_sha256':sha(s5[17:]),'section4_hex':s4.hex()}
    return level, representative[level].astype(np.float32), metadata


def fetch_file(stamp):
    name = f'Z__C_RJTD_{stamp:%Y%m%d%H%M%S}_RDR_JMAGPV_Ggis1km_Prr05lv_ANAL_grib2.bin'
    path = RAW/'grib2'/name; url = BASE+name
    existing = path.exists()
    if existing: data=path.read_bytes()
    else:
        for attempt in range(3):
            try:
                response=requests.get(url,timeout=60); response.raise_for_status(); data=response.content
                assert data[:4]==b'GRIB' and data[-4:]==b'7777'
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2*(attempt+1))
        with path.open('xb') as f: f.write(data)
    level, rate, meta = decode(data)
    assert meta['utc']==stamp.isoformat()
    return level, rate, dict(meta, url=url, local_path=str(path.relative_to(ROOT)), sha256=sha(data), bytes=len(data),
        retrieved_or_verified_utc=dt.datetime.now(UTC).isoformat(), reused=existing)


def self_test():
    # vmax=250 -> base5; [2,253] means three copies of level2; second run11 copies.
    got=expand_rle(bytes([2,253,0,251,253]),250,14)
    assert np.array_equal(got,np.array([2,2,2]+[0]*11,dtype=np.uint8))
    for payload,n in [(bytes([251]),1),(bytes([1]),2),(bytes([1,255,255,255]),1)]:
        try: expand_rle(payload,250,n)
        except ValueError: pass
        else: raise AssertionError('Invalid stream was accepted')
    # Independent run-by-run slice fill for the downloaded full sample, without np.repeat.
    sample=Path('/private/tmp/udt_hist_radar_sample.grib2')
    if sample.exists():
        b=sample.read_bytes(); secs={};p=16
        while p<len(b)-4:
            n=u(b,p,p+4);secs[b[p+4]]=b[p:p+n];p+=n
        vmax=u(secs[5],12,14); codes=secs[7][5:]; target=np.empty(8601600,'u1'); i=0;cursor=0
        while i<len(codes):
            v=codes[i];assert v<=vmax;i+=1;run=1;digit=1
            while i<len(codes) and codes[i]>vmax:
                run+=(codes[i]-vmax-1)*digit;digit*=255-vmax;i+=1
            target[cursor:cursor+run]=v;cursor+=run
        assert cursor==target.size
        assert np.array_equal(target,expand_rle(codes,vmax,target.size))


def main():
    self_test()
    for p in [RAW/'grib2',OUT,REPORT]:p.mkdir(parents=True,exist_ok=True)
    source_docs = [
        ('rish_archive_information.html','http://database.rish.kyoto-u.ac.jp/arch/jmadata/'),
        ('rish_20240821_index.html',BASE),
        ('jma_568.pdf','https://www.data.jma.go.jp/suishin/jyouhou/pdf/568.pdf')]
    sources=[]
    for name,url in source_docs:
        p=RAW/name
        if not p.exists():
            response=requests.get(url,timeout=90);response.raise_for_status()
            with p.open('xb') as f:f.write(response.content)
        sources.append(dict(url=url,path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
    stamps=[(START+dt.timedelta(minutes=5*(i+1))).astimezone(UTC) for i in range(108)]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch_file,stamps))
    levels=np.stack([x[0] for x in results]);rates=np.stack([x[1] for x in results]);manifest=[x[2] for x in results]
    lat=48-(np.arange(R0,R1)+.5)/120
    lon=118+(np.arange(C0,C1)+.5)/80
    ends=np.asarray([s.timestamp() for s in stamps],dtype=np.int64)
    np.savez_compressed(OUT/'radar_intervals.npz',representative_intensity_mm_h=rates,level=levels,
        interval_end_unix_s=ends,interval_start_unix_s=ends-300,latitude=lat,longitude=lon,
        global_row=np.arange(R0,R1),global_col=np.arange(C0,C1))
    # Preserve missingness: never count no observation as zero rain.
    complete=np.isfinite(rates).all(axis=0)
    total=np.where(complete,np.sum(rates.astype(float)/12,axis=0),np.nan)
    peak=np.where(complete,np.max(rates,axis=0),np.nan)
    profile=dict(driver='GTiff',height=len(lat),width=len(lon),count=1,dtype='float32',crs=6668,
        transform=from_origin(118+C0/80,48-R0/120,1/80,1/120),compress='deflate',nodata=np.nan)
    for name,a in [('integrated_representative_rain_mm',total),('peak_representative_intensity_mm_h',peak)]:
        with rasterio.open(OUT/f'{name}.tif','w',**profile) as f:f.write(a.astype('f4'),1)
    stations=list(csv.DictReader((ROOT/'data/processed/tokyo_rain_events/2024-08-21/jma_station_event_summary.csv').open(encoding='utf-8-sig')))
    gauges=list(csv.DictReader((ROOT/'data/processed/tokyo_rain_events/2024-08-21/jma_rain_10min.csv').open(encoding='utf-8-sig')))
    comparisons=[];pairs=[]
    for st in stations:
        rr=int(np.floor((48-float(st['lat']))*120))-R0;cc=int(np.floor((float(st['lon'])-118)*80))-C0
        assert 0<=rr<len(lat) and 0<=cc<len(lon)
        obs=[];pred=[]
        for g in gauges:
            if g['station_id']!=st['station_id']:continue
            a=dt.datetime.fromisoformat(g['interval_start_jst']);b=dt.datetime.fromisoformat(g['interval_end_jst'])
            if a<START or b>END:continue
            sel=(ends>a.timestamp())&(ends<=b.timestamp());assert int(sel.sum())==2
            estimate=float(rates[sel,rr,cc].astype(float).sum()/12)
            measured=float(g['precip_mm']);obs.append(measured);pred.append(estimate)
            pairs.append(dict(station_id=g['station_id'],station_name=g['station_name'],interval_end_jst=b.isoformat(),gauge_10min_mm=measured,
                radar_reconstructed_10min_mm=estimate,gauge_quality=g['quality'],source_global_row=rr+R0,source_global_col=cc+C0))
        obs=np.array(obs);pred=np.array(pred);assert len(obs)==54 and np.isfinite(pred).all()
        comparisons.append(dict(station_id=st['station_id'],station_name=st['station_name'],latitude=float(st['lat']),longitude=float(st['lon']),
            samples_10min=54,gauge_total_mm=float(obs.sum()),radar_reconstructed_total_mm=float(pred.sum()),
            bias_total_mm=float((pred-obs).sum()),rmse_10min_mm=float(np.sqrt(np.mean((pred-obs)**2))),
            gauge_peak10_mm=float(obs.max()),radar_peak10_mm=float(pred.max()),
            gauge_rolling60_peak_mm=float(np.convolve(obs,np.ones(6),'valid').max()),radar_rolling60_peak_mm=float(np.convolve(pred,np.ones(6),'valid').max()),
            note='Point versus grid representative comparison; gauge input to radar coefficients means independence is not assured; no calibration applied.'))
    for name,rows in [('gauge_comparison.csv',comparisons),('gauge_aligned_pairs.csv',pairs)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=dict(event='2024-08-21 Tokyo',window_start_jst=START.isoformat(),window_end_jst=END.isoformat(),
        frames=108,interval_seconds=300,missing_frames=0,roi_shape=list(rates.shape),roi_bounds_lon_lat=[139.45,35.5,139.95,35.9],
        original_global_shape=[3360,2560],nominal_resolution='Approximately 1 km; longitude 45 arcseconds, latitude 30 arcseconds',
        product='JMA 5-minute 1 km nationwide composite radar intensity, parameter203, analysis, archived by Kyoto University RISH',
        product_is_forecast=False,missing_level=0,no_echo_level=1,saturated_level=251,
        missing_values_in_roi=int((levels==0).sum()),saturated_values_in_roi=int((levels==251).sum()),
        reconstruction='Hold the embedded representative intensity over the preceding 300 seconds (t-300,t]; integrate rate/12. This is an explicit rainfall reconstruction, not a measured five-minute accumulated product.',
        http_transport='Published archive uses HTTP. No certificate verification disabled. SHA256 secures local identity, not authenticated remote provenance.',
        horizontal_reference='Product declares GRS80, not a surveyed epoch; assigned JGD2011 geographic (EPSG6668) for model alignment as an explicit approximation.',
        missing_policy='No observation remains NaN; positive catchment weight on missing data blocks forcing delivery.',
        gauge_comparison=comparisons,roi_integrated_representative_max_mm=float(np.nanmax(total)),
        source_documents=sources,raw_download_bytes=sum(m['bytes'] for m in manifest),
        not_done=['No hydrodynamic simulation','No affected population','No ML training','No sealed-test access'],
        limitations=['Radar intensity is not ground truth of flooded depth.','Public 1 km file is a lower-resolution product, not native 250 m XRAIN.',
            'No observed river-level hydrograph with common elevation datum acquired; river stages must remain explicit scenarios.'])
    dump(RAW/'acquisition_manifest.json',dict(files=manifest,source_documents=sources,script_sha256=sha(Path(__file__).read_bytes())))
    dump(OUT/'rainfall_contract.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
