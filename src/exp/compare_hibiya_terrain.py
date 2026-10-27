"""Compare a native Tokyo 0.50 m ground DEM sheet with original PLATEAU TIN.

Run from any local terminal after uv sync:
    uv run python src/exp/compare_hibiya_terrain.py
Inputs are preserved; outputs are exploratory, not a flood simulation.
"""

from __future__ import annotations

import os
# Use the project's bundled PROJ/GDAL data rather than inherited Anaconda paths.
for variable in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(variable, None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/udt01b-mpl")

import csv
import hashlib
import heapq
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as PlotPath
import numpy as np
import pyproj
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-briefing/plateau_terrain_pilot"
RAW = ROOT / "data/raw/tokyo_lidar_dem/2024/pilot_hibiya/09LD2812.tif"
ARCHIVE = ROOT / "data/raw/plateau/2025/13101-chiyoda-ku/13101_chiyoda-ku_pref_2025_citygml_1_op.zip"
GML = "http://www.opengis.net/gml"
CORE = "http://www.opengis.net/citygml/2.0"
TO_XY = pyproj.Transformer.from_crs(6668, 6677, always_xy=True)
TO_LL = pyproj.Transformer.from_crs(6677, 6668, always_xy=True)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean_m": float(np.mean(values)), "median_m": float(np.median(values)), "mean_absolute_m": float(np.mean(np.abs(values))), "rms_difference_m": float(np.sqrt(np.mean(values**2))), "p05_m": float(np.quantile(values,.05)), "p95_m": float(np.quantile(values,.95)), "p95_absolute_m": float(np.quantile(np.abs(values),.95)), "min_m": float(np.min(values)), "max_m": float(np.max(values)), "fraction_abs_over_0_10m": float(np.mean(np.abs(values)>.1)), "fraction_abs_over_0_30m": float(np.mean(np.abs(values)>.3))}


def load_tin(bounds):
    """Keep original triangles intersecting the raster and a 10 m margin."""
    corners = [TO_LL.transform(x,y) for x in (bounds.left-10,bounds.right+10) for y in (bounds.bottom-10,bounds.top+10)]
    west,east=min(p[0] for p in corners),max(p[0] for p in corners)
    south,north=min(p[1] for p in corners),max(p[1] for p in corners)
    nodes=[]; lookup={}; triangles=[]; seen=set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        filename="udx/dem/533946_dem_6697_op.gml"
        with archive.open(filename) as stream:
            for event,e in ET.iterparse(stream,events=("end",)):
                if e.tag==f"{{{GML}}}posList":
                    v=np.fromstring(e.text or "",sep=" ")
                    if len(v)==12:
                        pts=v.reshape(4,3)[:3]
                        if not (pts[:,1].max()<west or pts[:,1].min()>east or pts[:,0].max()<south or pts[:,0].min()>north):
                            ids=[]
                            for p in pts:
                                key=tuple(p)
                                if key not in lookup:
                                    lookup[key]=len(nodes); nodes.append(key)
                                ids.append(lookup[key])
                            key=tuple(sorted(ids))
                            if len(set(ids))==3 and key not in seen:
                                triangles.append(ids); seen.add(key)
                e.clear()
    nodes=np.asarray(nodes)
    x,y=TO_XY.transform(nodes[:,1],nodes[:,0])
    projected=np.column_stack((x,y,nodes[:,2]))
    triangulation=mtri.Triangulation(x,y,np.asarray(triangles))
    return projected,triangulation,mtri.LinearTriInterpolator(triangulation,nodes[:,2])


def polygon_points(element, feature):
    """Resolve same-feature GML polygon references; retain XY/Z geometry."""
    by_id={e.get(f"{{{GML}}}id"):e for e in feature.iter() if e.get(f"{{{GML}}}id")}
    polygons=list(element.iter(f"{{{GML}}}Polygon"))
    for e in element.iter():
        ref=e.get("{http://www.w3.org/1999/xlink}href","")
        if ref.startswith("#") and ref[1:] in by_id:
            target=by_id[ref[1:]]
            if target.tag==f"{{{GML}}}Polygon": polygons.append(target)
    result=[]; seen=set()
    for p in polygons:
        rings=[]
        for ring in p.iter(f"{{{GML}}}LinearRing"):
            pos=ring.find(f"{{{GML}}}posList")
            if pos is None: continue
            values=np.fromstring(pos.text or "",sep=" ")
            if len(values)%3: continue
            points=values.reshape(-1,3)
            if len(points)<4: continue
            x,y=TO_XY.transform(points[:,1],points[:,0])
            rings.append(np.column_stack((x,y,points[:,2])))
        if rings:
            key=rings[0].tobytes()
            if key not in seen: result.append(rings); seen.add(key)
    return result


def load_overlays(bounds):
    roads=[]; buildings=[]; lod3vertices=[]; road_metadata=[]
    def intersects(rings):
        points=rings[0]
        return not(points[:,0].max()<bounds.left or points[:,0].min()>bounds.right or points[:,1].max()<bounds.bottom or points[:,1].min()>bounds.top)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for family in ("tran","bldg"):
            filename=f"udx/{family}/53394600_{family}_6697_op.gml"
            print(f"Reading {filename}",flush=True)
            with archive.open(filename) as stream:
                for event,e in ET.iterparse(stream,events=("end",)):
                    if e.tag!=f"{{{CORE}}}cityObjectMember": continue
                    feature=next(iter(e),None)
                    if feature is None: e.clear(); continue
                    property_name="lod1MultiSurface" if family=="tran" else "lod0RoofEdge"
                    candidates=[q for q in feature if local(q.tag)==property_name]
                    polygons=[]
                    for q in candidates: polygons.extend(polygon_points(q,feature))
                    matches=[p for p in polygons if intersects(p)]
                    if matches:
                        target=roads if family=="tran" else buildings
                        target.extend({"type":"Polygon","coordinates":[r[:,:2].tolist() for r in p]} for p in matches)
                        if family=="tran":
                            dates=sorted({q.text.strip() for q in feature.iter() if local(q.tag)=="creationDate" and q.text})
                            road_metadata.append({"gml_id":feature.get(f"{{{GML}}}id"),"creation_dates":dates})
                            for q in feature.iter():
                                if local(q.tag)=="lod3MultiSurface":
                                    for p in polygon_points(q,feature):
                                        for ring in p: lod3vertices.extend(ring.tolist())
                    e.clear()
    return roads,buildings,np.unique(np.asarray(lod3vertices).reshape(-1,3),axis=0),road_metadata


def sample_native(array,transform,x,y):
    col=(np.asarray(x)-transform.c)/transform.a-.5
    row=(np.asarray(y)-transform.f)/transform.e-.5
    return ndimage.map_coordinates(array,[row,col],order=1,mode="constant",cval=np.nan)


def depression_fill(surface):
    """Priority flood with all four raster edges as outlets; 8-neighbor grid."""
    rows,cols=surface.shape
    visited=np.zeros(surface.shape,dtype=bool); filled=surface.copy(); heap=[]
    for r in range(rows):
        for c in (0,cols-1):
            if not visited[r,c]: visited[r,c]=True; heapq.heappush(heap,(float(surface[r,c]),r,c))
    for c in range(cols):
        for r in (0,rows-1):
            if not visited[r,c]: visited[r,c]=True; heapq.heappush(heap,(float(surface[r,c]),r,c))
    while heap:
        level,r,c=heapq.heappop(heap)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr,cc=r+dr,c+dc
                if (dr or dc) and 0<=rr<rows and 0<=cc<cols and not visited[rr,cc]:
                    visited[rr,cc]=True
                    new=max(level,float(surface[rr,cc])); filled[rr,cc]=new
                    heapq.heappush(heap,(new,rr,cc))
    return filled-surface


def depression_candidates(depth,surface,transform):
    interior=np.ones(surface.shape,dtype=bool); interior[:20]=False; interior[-20:]=False; interior[:,:20]=False; interior[:,-20:]=False
    labels,n=ndimage.label((depth>.1)&interior,structure=np.ones((3,3)))
    counts=np.bincount(labels.ravel())
    candidates=[]
    for label in np.argsort(counts[1:])[::-1][:30]+1:
        if counts[label]<16: continue
        r,c=np.where(labels==label)
        bottom=int(np.argmin(surface[r,c])); rr,cc=int(r[bottom]),int(c[bottom])
        x,y=rasterio.transform.xy(transform,rr,cc)
        lon,lat=TO_LL.transform(x,y)
        candidates.append({"area_m2":float(len(r)*.25),"max_fill_depth_m":float(depth[r,c].max()),"bottom_height_m":float(surface[rr,cc]),"row":rr,"col":cc,"easting":x,"northing":y,"longitude":lon,"latitude":lat})
    return candidates


def write_raster(name,array,transform):
    with rasterio.open(OUT/name,"w",driver="GTiff",height=array.shape[0],width=array.shape[1],count=1,dtype="float32",crs="EPSG:6677",transform=transform,nodata=-32767,compress="deflate") as dest:
        dest.write(np.where(np.isfinite(array),array,-32767).astype("float32"),1)
        dest.update_tags(AREA_OR_POINT="Point")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with rasterio.open(RAW) as src:
        dem=src.read(1).astype(float); transform=src.transform; bounds=src.bounds
        if src.res!=(.5,.5) or src.tags().get("AREA_OR_POINT")!="Point": raise ValueError("Unexpected DEM grid")
        if np.any(dem==src.nodata): raise ValueError("Pilot contains nodata; revise comparison handling")
        original_wkt=src.crs.to_wkt(); original_epsg=src.crs.to_epsg()
    xs=transform.c+(np.arange(dem.shape[1])+.5)*transform.a
    ys=transform.f+(np.arange(dem.shape[0])+.5)*transform.e
    xx,yy=np.meshgrid(xs,ys)
    print("Extracting original PLATEAU triangles",flush=True)
    nodes,triangulation,interpolate=load_tin(bounds)
    plateau=np.asarray(interpolate(xx,yy).filled(np.nan))
    if not np.all(np.isfinite(plateau)): raise ValueError("PLATEAU does not cover all DEM pixels")
    roads,buildings,road3,road_metadata=load_overlays(bounds)
    roadmask=rasterize([(g,1) for g in roads],out_shape=dem.shape,transform=transform,fill=0).astype(bool)
    buildingmask=rasterize([(g,1) for g in buildings],out_shape=dem.shape,transform=transform,fill=0).astype(bool)
    difference=dem-plateau
    node_dem=sample_native(dem,transform,nodes[:,0],nodes[:,1])
    inside=np.isfinite(node_dem)
    r3dem=sample_native(dem,transform,road3[:,0],road3[:,1]) if len(road3) else np.array([])
    r3ok=np.isfinite(r3dem)
    print("Comparing terrain depression candidates",flush=True)
    depths={"tokyo_0_50m":depression_fill(dem),"plateau_tin":depression_fill(plateau)}
    candidates={key:depression_candidates(depth,dem if key=="tokyo_0_50m" else plateau,transform) for key,depth in depths.items()}
    for key,points in candidates.items():
        if points:
            with (OUT/f"{key}_depression_candidates.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    inside_boundary=None
    border_path=OUT/"chiyoda_boundary_source.geojson"
    if border_path.exists():
        border=json.loads(border_path.read_text())
        points=np.column_stack((xx.ravel(),yy.ravel())); hit=np.zeros(len(points),dtype=bool)
        for feature in border["features"]:
            geometry=feature["geometry"]
            if geometry["type"]=="LineString":
                coords=geometry["coordinates"]
                if not np.allclose(coords[0],coords[-1],rtol=0,atol=1e-10):
                    raise ValueError("Administrative boundary LineString is not closed")
                polys=[[coords]]
            elif geometry["type"]=="MultiPolygon":
                polys=geometry["coordinates"]
            elif geometry["type"]=="Polygon":
                polys=[geometry["coordinates"]]
            else:
                raise ValueError(f"Unsupported boundary geometry: {geometry['type']}")
            for polygon in polys:
                rings=[]
                for ring in polygon:
                    arr=np.asarray(ring); x,y=TO_XY.transform(arr[:,0],arr[:,1]); rings.append(np.column_stack((x,y)))
                contained=PlotPath(rings[0]).contains_points(points)
                for hole in rings[1:]: contained &= ~PlotPath(hole).contains_points(points)
                hit |= contained
        inside_boundary=float(hit.mean())
    with zipfile.ZipFile(RAW.with_suffix(".zip")) as archive:
        zip_date=archive.getinfo(RAW.name).date_time
    las_metadata_path=OUT/"las_header_metadata.json"
    las_metadata=json.loads(las_metadata_path.read_text()) if las_metadata_path.exists() else None
    report={
        "comparison_date":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pilot":"Hibiya, Chiyoda; sheet 09LD2812", "area_m2":120000,"shape":list(dem.shape),"pixel_spacing_m":.5,
        "bounds_epsg6677":list(bounds),"bounds_corner_lonlat":[TO_LL.transform(bounds.left,bounds.bottom),TO_LL.transform(bounds.right,bounds.top)],
        "fraction_pixel_centers_in_chiyoda_boundary":inside_boundary,
        "original_tiff_wkt":original_wkt,"original_tiff_to_epsg":original_epsg,
        "horizontal_crs_assignment":"EPSG:6677 from original GeoTIFF ProjectedCSType key=6677, matching TM projection parameters and official catalog; the source geographic datum name is incomplete. No source coordinate values changed.",
        "height_operation":"None. Original elevations preserved; vertical-reference equivalence remains unconfirmed.",
        "pixel_registration":"Source RasterPixelIsPoint; GDAL half-pixel normalization retained. Pixel centers start at (-7199.75,-36300.25).",
        "tokyo_source_url":"https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/grd50/09LD2812.zip",
        "tokyo_raw_zip_sha256":hashlib.sha256(RAW.with_suffix(".zip").read_bytes()).hexdigest(),
        "tokyo_zip_member_timestamp_unverified":list(zip_date),"tokyo_exact_survey_date":"Not recorded in TIFF or one-file ZIP; official campaign FY2022-FY2023, publication 2024-10-31.",
        "corresponding_las_header_metadata":las_metadata,
        "tokyo_vertical_datum":"Not explicitly encoded: only vertical unit metre, no VerticalCSType/VerticalDatum GeoKey. Catalog confirms horizontal JGD2011 IX only.",
        "plateau_source_archive":str(ARCHIVE.relative_to(ROOT)),"plateau_terrain_member":"udx/dem/533946_dem_6697_op.gml",
        "plateau_terrain_creation_date":"2024-03-15; not acquisition date","plateau_declared_vertical_reference":"EPSG:6697 / JGD2011 + TP from GML and metadata; no confirmed 2025 height revision.",
        "comparison_sign":"Tokyo 0.50m elevation minus PLATEAU TIN elevation, metres",
        "terrain_sampling":"Piecewise-linear interpolation of original TIN triangles onto native DEM pixel centers; no new Delaunay surface, height correction, pit filling or smoothing applied to source comparison.",
        "retained_tin_nodes":len(nodes),"retained_tin_triangles":len(triangulation.triangles),
        "difference_all_pixels":stats(difference),"difference_road_pixels":stats(difference[roadmask]),"difference_outside_roof_outline_pixels":stats(difference[~buildingmask]),
        "difference_at_original_tin_nodes":stats(node_dem[inside]-nodes[inside,2]),
        "road_lod3_vertex_difference":stats(r3dem[r3ok]-road3[r3ok,2]) if len(road3) else {"n":0},
        "road_polygon_count":len(roads),"roof_outline_polygon_count":len(buildings),"road_area_m2":float(roadmask.sum()*.25),"road_metadata":road_metadata,
        "depression_method":"8-neighbor priority flood with raster edges as outlets, threshold fill depth>0.10m, candidate connected area>=4m2, exclude 10m edge strip. Pure terrain candidates, not predicted inundation. Building footprints, ponds, sewer inlets and engineered drainage are not removed from this diagnostic.",
        "depression_candidates":candidates,
        "depression_area_over_0_10m_m2":{key:float(np.sum(depth>.1)*.25) for key,depth in depths.items()},
        "depression_area_overlap_iou":float(np.sum((depths['tokyo_0_50m']>.1)&(depths['plateau_tin']>.1))/np.sum((depths['tokyo_0_50m']>.1)|(depths['plateau_tin']>.1))),
        "limitations":["One deliberately small pilot; not representative of the three wards.","Elevation differences are not error against independent surveyed truth; datasets may share an original survey.","Precise acquisition dates and vertical height-result versions are not confirmed.","Road mask uses LOD1 polygons; building exclusion uses roof outlines, not verified ground footprints.","Terrain low points/depressions may include ponds, artifacts and areas drained by infrastructure.","No vertical offset is inferred or removed."],
        "runtime_versions":{"numpy":np.__version__,"rasterio":rasterio.__version__,"pyproj":pyproj.__version__,"proj":pyproj.__proj_version__},
    }
    # Inspect a reproducible E-W transect crossing a road-rich row within the sheet.
    row_scores=ndimage.uniform_filter1d(roadmask.sum(axis=1).astype(float),size=21)
    row=int(np.argmax(row_scores[100:-100])+100)
    report["road_transect"]={"orientation":"East-west across the whole sheet","row":row,"northing":float(ys[row]),"selection":"Row with greatest smoothed LOD1 road coverage, excluding 50m northern/southern strips; exploratory agent selection.","road_pixel_count":int(roadmask[row].sum())}
    with (OUT/"road_transect.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["distance_from_west_m","easting","northing","tokyo_dem_height_m","plateau_tin_height_m","tokyo_minus_plateau_m","inside_lod1_road"])
        writer.writerows(zip(xs-bounds.left,xs,np.repeat(ys[row],len(xs)),dem[row],plateau[row],difference[row],roadmask[row].astype(int)))
    (OUT/"comparison_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (OUT/"roads_source.geojson").write_text(json.dumps({"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:6677"}},"features":[{"type":"Feature","properties":{},"geometry":g} for g in roads]})+"\n")
    for name,array in [("tokyo_dem_native_0_50m.tif",dem),("plateau_tin_on_0_50m_grid.tif",plateau),("tokyo_minus_plateau.tif",difference)]:write_raster(name,array,transform)
    extent=(bounds.left,bounds.right,bounds.bottom,bounds.top)
    zlo,zhi=np.quantile(dem,[.01,.99]); limit=max(.3,float(np.quantile(np.abs(difference),.98)))
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),layout="constrained")
    for ax,array,title in zip(axes,[dem,plateau,difference],["Tokyo ground DEM: native 0.50 m","PLATEAU: original nominal 5 m TIN","Tokyo minus PLATEAU (m)"]):
        im=ax.imshow(array,extent=extent,origin="upper",cmap="RdBu_r" if array is difference else "terrain",vmin=-limit if array is difference else zlo,vmax=limit if array is difference else zhi)
        for geometry in roads:
            ring=np.asarray(geometry["coordinates"][0]); ax.plot(ring[:,0],ring[:,1],color="black",lw=.3,alpha=.5)
        ax.axhline(ys[row],color="#a21a7d",lw=1,ls="--")
        ax.set_xlim(bounds.left,bounds.right);ax.set_ylim(bounds.bottom,bounds.top)
        ax.set_title(title,fontsize=11);ax.set_xlabel("Easting (m), JGD2011 IX");ax.set_ylabel("Northing (m)")
        fig.colorbar(im,ax=ax,shrink=.7,extend="both",label="Elevation difference (m)" if array is difference else "Elevation (m)")
    fig.suptitle("Hibiya pilot: 400 x 300 m; source heights preserved; vertical reference not fully confirmed",fontsize=13)
    fig.savefig(OUT/"terrain_comparison.png",dpi=170);plt.close(fig)
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),layout="constrained",sharex=True)
    distance=xs-bounds.left
    ax1.plot(distance,dem[row],label="Tokyo native 0.50 m DEM",lw=1)
    ax1.plot(distance,plateau[row],label="PLATEAU original TIN",lw=1.5)
    ax1.fill_between(distance,zlo,zhi,where=roadmask[row],color="#aaa",alpha=.2,label="PLATEAU LOD1 road footprint")
    ax1.set_ylabel("Elevation (m)");ax1.legend(loc="upper right",fontsize=9)
    ax1.set_title(f"Road-rich E-W transect at northing {ys[row]:.2f} m; no vertical correction")
    ax2.plot(distance,difference[row],color="#a21a7d",lw=1);ax2.axhline(0,color="#777",lw=.8)
    ax2.set_xlabel("Distance from west sheet edge (m)");ax2.set_ylabel("Tokyo - PLATEAU (m)")
    fig.savefig(OUT/"road_transect.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout="constrained")
    for ax,key in zip(axes,depths):
        im=ax.imshow(depths[key],extent=extent,origin="upper",cmap="magma",vmin=0,vmax=.5)
        for point in candidates[key][:5]:ax.scatter(point["easting"],point["northing"],s=20,facecolors="none",edgecolors="cyan")
        ax.set_title(key.replace("_"," "));ax.set_xlabel("Easting (m)");ax.set_ylabel("Northing (m)")
        fig.colorbar(im,ax=ax,shrink=.7,label="Terrain-only fill depth (m)")
    fig.suptitle("Depression candidates: sheet edges are outlets; this is not simulated flooding",fontsize=11)
    fig.savefig(OUT/"depression_comparison.png",dpi=170);plt.close(fig)
    print(json.dumps({key:report[key] for key in ["fraction_pixel_centers_in_chiyoda_boundary","difference_all_pixels","difference_road_pixels","difference_at_original_tin_nodes","road_lod3_vertex_difference","depression_area_over_0_10m_m2","depression_area_overlap_iou","road_transect"]},indent=2),flush=True)
    print(f"Outputs: {OUT}",flush=True)


if __name__=="__main__":main()
