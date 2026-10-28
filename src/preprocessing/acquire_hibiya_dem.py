"""Acquire the official Hibiya 0.50 m DEM and a bounded LAS metadata prefix.

No third-party download mirror is used. A LAS prefix is not a full point-cloud
download and cannot receive a full ZIP CRC check. Existing raw files are kept.
"""
from __future__ import annotations

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path

import hashlib
import json
import struct
import urllib.request
import zipfile
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE="https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/"
RAW=moga_raw(ROOT, 'tokyo_lidar_dem/2024')
OUT=ROOT/"data/exp/data-briefing/plateau_terrain_pilot"


def main():
    (RAW/"pilot_hibiya").mkdir(parents=True,exist_ok=True)
    (RAW/"source_documents").mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    resources=[]
    for relative,name in [("Vectortile/23ku/grd50/18/232839/103235.pbf","source_documents/grd50_index_18_232839_103235.pbf"),("Vectortile/23ku/lp/18/232839/103235.pbf","source_documents/lp_index_18_232839_103235.pbf"),("grd50/09LD2812.zip","pilot_hibiya/09LD2812.zip")]:
        path=RAW/name; url=BASE+relative
        if not path.exists():
            with urllib.request.urlopen(url,timeout=60) as response: path.write_bytes(response.read())
        resources.append({"source_url":url,"local_path":moga_source(path, ROOT),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    with zipfile.ZipFile(RAW/"pilot_hibiya/09LD2812.zip") as archive:
        if archive.testzip() is not None: raise ValueError("DEM ZIP CRC check failed")
        target=RAW/"pilot_hibiya/09LD2812.tif"
        if not target.exists(): target.write_bytes(archive.read("09LD2812.tif"))
        elif target.read_bytes()!=archive.read("09LD2812.tif"): raise ValueError("Existing TIFF differs from original ZIP")
    path=RAW/"source_documents/lp_09LD2812_zip_first_1MiB.bin"
    url=BASE+"lp/09LD2812.zip"
    if not path.exists():
        request=urllib.request.Request(url,headers={"Range":"bytes=0-1048575"})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=206: raise ValueError("Server did not honor bounded range request")
            path.write_bytes(response.read(1048576))
    data=path.read_bytes()
    fields=struct.unpack_from("<IHHHHHIIIHH",data)
    _,version,flag,method,time,datebits,crc,compressed,uncompressed,nlen,xlen=fields
    filename=data[30:30+nlen].decode()
    if filename!="09LD2812.las" or method!=8: raise ValueError("Unexpected LAS ZIP layout")
    decoded=zlib.decompressobj(-15).decompress(data[30+nlen+xlen:],4*1024*1024)
    if decoded[:4]!=b"LASF": raise ValueError("LAS header missing")
    prefix_path=RAW/"source_documents/lp_09LD2812_first_member_prefix.bin"
    if not prefix_path.exists(): prefix_path.write_bytes(decoded)
    header_size=struct.unpack_from("<H",decoded,94)[0]
    vlr_count=struct.unpack_from("<I",decoded,100)[0]
    point_offset=struct.unpack_from("<I",decoded,96)[0]
    record_size=struct.unpack_from("<H",decoded,105)[0]
    position=header_size;vlrs=[]
    for _ in range(vlr_count):
        _,user,rid,size,description=struct.unpack_from("<H16sHH32s",decoded,position)
        payload=decoded[position+54:position+54+size]
        item={"user_id":user.rstrip(b"\0").decode(),"record_id":rid,"record_length":size,"description":description.rstrip(b"\0").decode()}
        if rid==34735:item["geokey_directory"]=list(struct.unpack("<"+"H"*(len(payload)//2),payload))
        elif rid==34736:item["geodoubles"]=list(struct.unpack("<"+"d"*(len(payload)//8),payload))
        elif rid==34737:item["text"]=payload.rstrip(b"\0").decode()
        vlrs.append(item);position+=54+size
    if decoded[104]!=3 or record_size!=34:raise ValueError("Unexpected LAS point format")
    count=min(1000,(len(decoded)-point_offset)//record_size)
    gps=[struct.unpack_from("<d",decoded,point_offset+i*record_size+20)[0] for i in range(count)]
    year=struct.unpack_from("<H",decoded,92)[0];day=struct.unpack_from("<H",decoded,90)[0]
    header={"header_version":list(decoded[24:26]),"global_encoding":struct.unpack_from("<H",decoded,6)[0],"file_creation_date":str(date(year,1,1)+timedelta(days=day-1)),"point_format":decoded[104],"point_record_length":record_size,"vlrs":vlrs,"sampled_gps_time_min":min(gps),"sampled_gps_time_max":max(gps),"sampled_point_count":count,"note":"Partial LAS prefix only. File creation date is not survey date. GPS week time without a week number cannot establish a calendar acquisition date."}
    (OUT/"las_header_metadata.json").write_text(json.dumps(header,ensure_ascii=False,indent=2)+"\n")
    resources.append({"source_url":url,"range":"bytes=0-1048575","local_path":moga_source(path, ROOT),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"partial_only":True})
    (OUT/"download_provenance.json").write_text(json.dumps({"recorded_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),"catalog_url":"https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024","dem_zip_crc_verified":True,"resources":resources},indent=2)+"\n")
    print("Original DEM verified; bounded LAS header inspected; provenance saved.")


if __name__=="__main__":main()
