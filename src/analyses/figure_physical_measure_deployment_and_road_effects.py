#!/usr/bin/env python3
"""Physical Measure Deployment and Road Effects.
AnaSOP Sections 5-7: matched two-parent/two-stage physical-arm trajectories,
common qualified roads, original lower-depth branch, synchronized actual clocks.
Approved aee/bee/cee/dee layout: unchanged trajectories and frozen deployment.
Presentation only: no simulation, training, new statistical model or source edits.
"""
from __future__ import annotations
import hashlib
import json
import os
import html
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/udt01b-article-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Physical Measure Deployment and Road Effects'
AUDIT = ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
BASE = ROOT/'data/processed/three_ward_analysis_contract_v1'
PHYSICAL = ROOT/'data/processed/three_ward_barrier_drainage_v1'
OUTPUT = ROOT/'data/results/figures/Figure_physical_measure_deployment_and_road_effects.png'
EVENTS = ['E20240618', 'E20250911']
STAGES = ['L', 'H']
ARMS = ['none', 'barrier', 'drainage', 'combined']
LABELS = ['No measure', 'Barrier', 'Drainage', 'Combined']
COLORS = ['#555555', '#ee7733', '#4477aa', '#228833']
STYLES = ['-', (0,(5,2)), (0,(1,1.5)), (0,(5,2,1,2))]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import Transformer
import rasterio
from shapely.geometry import shape
SLUG='physical_measure_deployment_and_road_effects'
MAPBASE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
BIND={}
def bind(p):
 p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def js(p):return json.loads(bind(p).read_text())
def prepare_curves():
    doc=ROOT/'docs/AnaSOP.md'; doc_sha=sha(doc)
    planpath=ROOT/'data/exp/research-design/three_ward_figure_table_plan_v1/confirmed_plan.json'
    assert f'| {TITLE} |' in doc.read_text()
    prior=js(AUDIT/'physical_measure_road_effects_validation.json');assert prior['passed']
    assert sha(bind(ROOT/'data/results/figures/Figure_physical_measure_road_effects.png'))==prior['output_sha256']
    sources=[BASE/'road_trajectories_preprocessed.parquet',BASE/'road_summaries_preprocessed.parquet',BASE/'export_manifest.json',PHYSICAL/'road_trajectories.csv',PHYSICAL/'road_summary.csv',planpath]
    for rel in ['data/exp/data-preprocessing/three_ward_analysis_contract_v1/validation.json','data/exp/research-design/three_ward_estimation_framework_v1/validation.json','data/exp/estimation/three_ward_barrier_drainage_v1/delivery_validation.json','data/exp/estimation/three_ward_barrier_drainage_v1/postprocessing_validation.json']:
        p=ROOT/rel;assert json.loads(p.read_text())['passed'];sources.append(p)
    sources += [PHYSICAL/'protocol.json', PHYSICAL/'measures/barrier_faces.csv',
                PHYSICAL/'measures/pump_sites.csv', PHYSICAL/'measures/barrier_exclusions.csv',
                ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz',
                ROOT/'src/preprocessing/prepare_barrier_drainage_inputs_v1.py',
                ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc',
                PHYSICAL/'solver_binding.json']
    bindings={str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol=json.loads((PHYSICAL/'protocol.json').read_text())
    barriers=pd.read_csv(PHYSICAL/'measures/barrier_faces.csv')
    pumps=pd.read_csv(PHYSICAL/'measures/pump_sites.csv')
    exclusions=pd.read_csv(PHYSICAL/'measures/barrier_exclusions.csv')
    with np.load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz') as z:
        protected=z['protected']; qualified=z['qualified']; weights=z['union_length_m']
        protected_km=float(weights[protected].sum()/1000)
        corridor_share=protected_km/float(weights[qualified].sum()/1000)
        assert protected.sum()==118 and not np.any(protected & ~qualified)
    assert len(barriers)==protocol['barrier_faces']==696
    assert len(pumps)==protocol['pumps']==19
    assert np.allclose(barriers.crest_m-barriers.ground_max_m,.5,atol=1e-8)
    assert np.allclose(barriers.Cd,.6) and np.allclose(pumps.capacity_m3_s,.5)
    assert pumps.capacity_m3_s.sum()==protocol['total_capacity_m3_s']==9.5
    assert np.allclose(pumps.trigger_depth_above_native_min_m,.05)
    assert np.allclose(pumps.maximum_lift_m,5) and pumps.straight_link_length_m.max()<=500
    assert (exclusions.reason=='mapped_road_opening_retained').sum()==610
    assert protocol['engineering_parameters_observed'] is False

    manifest=json.loads((BASE/'export_manifest.json').read_text())
    for e in manifest['outputs']:
        if e['label'] in ['Road trajectories','Road summaries']:
            assert sha(ROOT/e['source'])==e['source_sha256']
            assert sha(ROOT/e['output'])==e['output_sha256'] and e['exact_values_and_nulls']
    df=pd.read_parquet(sources[0]); summary=pd.read_parquet(sources[1])
    data=df[(df['Road Depth Bound']=='lower')&(df['Reporting Scope']=='Three wards')&(df['Road Stratum']=='all')]
    sums=summary[(summary['Road Depth Bound']=='lower')&(summary['Reporting Scope']=='Three wards')&(summary['Road Stratum']=='all')]
    assert len(data)==1744 and len(sums)==16
    known=836.9583262602582;unknown=277.9200922406389
    assert np.allclose(data['Qualified Road Length'],known,rtol=0,atol=1e-8)
    assert np.allclose(data['Unknown Road Length'],unknown,rtol=0,atol=1e-8)
    assert np.allclose(data['Open Road Length']+data['Slow Road Length']+data['Closed Road Length'],known,rtol=0,atol=1e-7)
    curves={};checks=[];clocks={}
    for event in EVENTS:
        for stage in STAGES:
            for arm in ARMS:
                case=f'{event}__{stage}__U__{arm}'
                g=data[data.Case==case].sort_values('Frame')
                assert len(g)==109 and np.array_equal(g.Frame,np.arange(109))
                # Read only the tiny clock members, never the native-depth cube.
                with np.load(PHYSICAL/'exposure'/f'{case}.npz') as z:
                    actual=z['actual_times'];analysis=z['analysis_times']
                times=g['Elapsed Time'].to_numpy();values=g['Closed Road Length'].to_numpy()
                # CSV-to-Parquet export preserves source decimals; binary clock roundtrip differs by at most 4e-12 s.
                assert np.allclose(times,analysis,rtol=0,atol=1e-10) and np.all(np.diff(actual)>0)
                assert np.max(abs(actual-times))<=.001 and times[0]==0 and times[-1]==32400
                assert np.all(values>=0) and np.isfinite(values).all()
                row=sums[sums.Case==case];assert len(row)==1;row=row.iloc[0]
                integral=float(np.trapezoid(values,times)/3600)
                assert np.isclose(integral,row['Closed Road Duration'],rtol=0,atol=1e-8)
                assert np.isclose(values.max(),row['Peak Closed Road Length'],rtol=0,atol=1e-8)
                assert int(np.argmax(values))==int(row['Peak Closure Frame'])
                assert np.isclose(values[-1],row['Final Closed Road Length'],rtol=0,atol=1e-8)
                curves[event,stage,arm]=(actual/3600,values)
                clocks[case]={'file':str((PHYSICAL/'exposure'/f'{case}.npz').relative_to(ROOT)),'actual_clock_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'analysis_clock_sha256':hashlib.sha256(analysis.tobytes()).hexdigest(),'max_actual_analysis_difference_s':float(np.max(abs(actual-times))),'table_clock_roundtrip_error_s':float(np.max(abs(times-analysis)))}
                checks.append({'case':case,'frames':len(g),'peak_closed_km':float(values.max()),'peak_frame':int(np.argmax(values)),'closure_exposure_km_h':integral,'closed_at_9h_km':float(values[-1]),'end_window_right_censored':bool(values[-1]>0)})
    contrasts=[]
    for event in EVENTS:
        for stage in STAGES:
            baseline=next(x for x in checks if x['case']==f'{event}__{stage}__U__none')
            for arm in ARMS[1:]:
                treatment=next(x for x in checks if x['case']==f'{event}__{stage}__U__{arm}')
                contrasts.append({'event':event,'stage':stage,'measure':arm,'avoided_peak_closed_km':baseline['peak_closed_km']-treatment['peak_closed_km'],'avoided_closure_exposure_km_h':baseline['closure_exposure_km_h']-treatment['closure_exposure_km_h']})
    differences={}; difference_checks=[]
    for event in EVENTS:
        for stage in STAGES:
            base_t,base_y=curves[event,stage,'none']
            for arm in ARMS[1:]:
                t,y=curves[event,stage,arm]
                assert np.max(abs(t-base_t))*3600<=.001
                # Join by the unchanged frame and approved analysis time, not nearest timestamps.
                case=f'{event}__{stage}__U__{arm}'
                g=data[data.Case==case].sort_values('Frame')
                analysis=g['Elapsed Time'].to_numpy()
                bg=data[data.Case==f'{event}__{stage}__U__none'].sort_values('Frame')
                assert np.allclose(analysis,bg['Elapsed Time'],rtol=0,atol=.001)
                delta=base_y-y
                integral=float(np.trapezoid(delta,analysis)/3600)
                original=next(x for x in contrasts if x['event']==event and x['stage']==stage and x['measure']==arm)
                assert np.isclose(integral,original['avoided_closure_exposure_km_h'],rtol=0,atol=1e-8)
                differences[event,stage,arm]=(base_t,delta)
                difference_checks.append({'event':event,'stage':stage,'measure':arm,'minimum_net_closed_length_reduction_km':float(delta.min()),'maximum_net_closed_length_reduction_km':float(delta.max()),'integrated_net_reduction_km_h':integral,'original_exposure_difference_matched':True})
    for row in difference_checks:
        previous=next(x for x in prior['difference_checks'] if all(x[k]==row[k] for k in ['event','stage','measure']))
        for key in ['minimum_net_closed_length_reduction_km','maximum_net_closed_length_reduction_km','integrated_net_reduction_km_h']:
            np.testing.assert_allclose(row[key],previous[key],rtol=0,atol=1e-8)
    for case,row in clocks.items():
        assert row['actual_clock_sha256']==prior['actual_clocks_checked'][case]['actual_clock_sha256']
        assert row['analysis_clock_sha256']==prior['actual_clocks_checked'][case]['analysis_clock_sha256']
    BIND.update(bindings)
    return locals()


def line_parts(g):return [g] if g.geom_type=='LineString' else list(g.geoms)
def poly_parts(g):return [g] if g.geom_type=='Polygon' else list(g.geoms)
def prepare_map(z):
    to_geo=Transformer.from_crs(6677,4326,always_xy=True)
    to_metric=Transformer.from_crs(4326,6677,always_xy=True)
    roads=js(MAPBASE/'clipped_roads_6677.geojson')['features']
    wards=[shape(f['geometry']) for f in js(MAPBASE/'wards_6677.geojson')['features']]
    with np.load(bind(MAPBASE/'clipped_road_lengths.npz')) as q:
        protected=q['protected'].copy();weights=q['union_length_m'].copy()
    with np.load(bind(PHYSICAL/'measures/spatial_contract.npz')) as q:
        native_xy=q['native_xy'].copy();kcs=q['native_kcs'].copy();tube=q['corridor_native'].copy()
        assert np.array_equal(q['protected_source_roads'],protected)
        assert np.array_equal(q['barrier_face_indices'],z['barriers'].face.to_numpy())
    pumps=z['pumps'];barriers=z['barriers'];protocol=z['protocol']
    sources=pumps.source_native_cell.to_numpy()-1;receivers=pumps.receiver_native_cell.to_numpy()-1
    np.testing.assert_allclose(native_xy[sources],pumps[['source_x','source_y']],rtol=0,atol=1e-8)
    np.testing.assert_allclose(native_xy[receivers],pumps[['receiver_x','receiver_y']],rtol=0,atol=1e-8)
    assert (kcs[sources]==1).all() and (kcs[receivers]==2).all() and tube[sources].all()
    lengths=np.hypot(pumps.receiver_x-pumps.source_x,pumps.receiver_y-pumps.source_y)
    np.testing.assert_allclose(lengths,pumps.straight_link_length_m,rtol=0,atol=1e-8);assert lengths.max()<=500
    ends=barriers[['x1','y1','x2','y2']].to_numpy().reshape(-1,2,2)
    np.testing.assert_allclose(np.linalg.norm(ends[:,1]-ends[:,0],axis=1),10,rtol=0,atol=1e-9)
    assert abs(float(np.linalg.norm(ends[:,1]-ends[:,0],axis=1).sum())-protocol['barrier_length_m'])<1e-8
    deployment=js(PHYSICAL/'measures/deployment_6677.geojson')['features'];assert len(deployment)==len(barriers)+len(pumps)
    for row in deployment:
        v=row['properties'];coordinates=np.array(row['geometry']['coordinates'])
        if v['kind']=='barrier':
            original=barriers[barriers.face.eq(v['face'])].iloc[0]
            np.testing.assert_allclose(coordinates,np.array([original.x1,original.y1,original.x2,original.y2]).reshape(2,2),rtol=0,atol=1e-8)
        else:
            original=pumps[pumps.pump_id.eq(v['pump_id'])].iloc[0]
            np.testing.assert_allclose(coordinates,np.array([original.source_x,original.source_y,original.receiver_x,original.receiver_y]).reshape(2,2),rtol=0,atol=1e-8)
    background=[];targets=[];ids=[]
    for f in roads:
        i=int(f['properties']['road_index']);g=shape(f['geometry']);ids.append(i)
        assert abs(g.length-weights[i])<1e-6
        for part in line_parts(g):
            xy=np.array(part.coords);x,y=to_geo.transform(xy[:,0],xy[:,1]);seg=np.column_stack([x,y]);background.append(seg)
            if protected[i]:targets.append(seg)
    assert len(ids)==len(set(ids));np.testing.assert_allclose(sum(shape(f['geometry']).length for f in roads if protected[int(f['properties']['road_index'])])/1000,z['protected_km'],rtol=0,atol=1e-8)
    coords=np.concatenate([ends.reshape(-1,2),pumps[['source_x','source_y']].to_numpy(),pumps[['receiver_x','receiver_y']].to_numpy()])
    lon,lat=to_geo.transform(coords[:,0],coords[:,1]);xx,yy=to_metric.transform(lon,lat)
    error=float(np.max(np.abs(np.column_stack([xx,yy])-coords)));assert error<1e-5
    barrier_lonlat=[]
    for seg in ends:
        x,y=to_geo.transform(seg[:,0],seg[:,1]);barrier_lonlat.append(np.column_stack([x,y]))
    links=[]
    for row in pumps.itertuples():
        x,y=to_geo.transform([row.source_x,row.receiver_x],[row.source_y,row.receiver_y]);links.append(np.column_stack([x,y]))
    source_lon,source_lat=to_geo.transform(pumps.source_x.to_numpy(),pumps.source_y.to_numpy())
    receiver_lon,receiver_lat=to_geo.transform(pumps.receiver_x.to_numpy(),pumps.receiver_y.to_numpy())
    water=js(ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson')['features']
    water_polygons=[]
    for f in water:
        g=shape(f['geometry'])
        for part in poly_parts(g):
            coords0=np.asarray(part.exterior.coords);assert (coords0[:,0]>139).all() and (coords0[:,0]<141).all()
            water_polygons.append(part)
    dem=ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif'
    dem_meta=(dem.stat().st_size,dem.stat().st_mtime_ns)
    with rasterio.open(dem) as ds:bb=ds.bounds
    lon0,lat0=to_geo.transform([bb.left,bb.left,bb.right,bb.right],[bb.bottom,bb.top,bb.bottom,bb.top])
    extent=[min(lon0),max(lon0),min(lat0),max(lat0)]
    assert (lon>=extent[0]).all() and (lon<=extent[1]).all() and (lat>=extent[2]).all() and (lat<=extent[3]).all()
    aspect=1/np.cos(np.deg2rad((extent[2]+extent[3])/2))
    return locals()

def add_polygon(ax,poly,color):
    vertices=[];codes=[]
    for ring in [poly.exterior]+list(poly.interiors):
        xy=np.array(ring.coords)[:,:2];vertices.extend(xy);codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(np.array(vertices),codes),facecolor=color,edgecolor='none',zorder=1))

def render(z,m):
    sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.05)
    plt.rcParams.update({'savefig.facecolor':'white','axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False})
    width=14.;left=.09;curvewidth=.24;mapleft=.435;mapwidth=.48;footer=1.7;head=.65;gap=.95
    mapheight=mapwidth*width*(m['extent'][3]-m['extent'][2])/(m['extent'][1]-m['extent'][0])*m['aspect']
    height=footer+mapheight+head;bottom=footer/height;mh=mapheight/height;ch=(mh-3*gap/height)/4
    fig=plt.figure(figsize=(width,height));axes={}
    configs=[('a','E20240618','L'),('b','E20240618','H'),('c','E20250911','L'),('d','E20250911','H')]
    lines=[]
    for i,(label,event,stage) in enumerate(configs):
        ax=fig.add_axes([left,bottom+(3-i)*(ch+gap/height),curvewidth,ch]);axes[label]=ax
        ax.axhspan(-.75,0,color='#fcf0ee',zorder=0);ax.axhline(0,color='#555555',lw=1,zorder=2)
        for ai,arm in enumerate(ARMS[1:],1):
            t,y=z['differences'][event,stage,arm]
            line,=ax.plot(t,y,color=COLORS[ai],linestyle=STYLES[ai],lw=1.8,label=LABELS[ai],zorder=2+ai)
            assert np.array_equal(line.get_xdata(),t) and np.array_equal(line.get_ydata(),y);lines.append(line)
        ax.set(xlim=(0,9),ylim=(-.75,4),xlabel='Elapsed model time (h)');ax.set_xticks([0,3,6,9]);ax.set_yticks([-.5,0,1,2,3,4]);ax.tick_params(labelsize=9.2)
        ax.grid(axis='y',color='#dce3e8',lw=.65);ax.set_axisbelow(True)
        day=f'{event[1:5]}-{event[5:7]}-{event[7:9]}';stage_label='low stage' if stage=='L' else 'raised stage'
        ax.annotate(f'{label}: {day}, {stage_label}',(0,1),xycoords='axes fraction',xytext=(-22,11),textcoords='offset points',fontsize=11.2,fontweight='bold',va='bottom',ha='left',annotation_clip=False)
    ylabel=fig.text(.025,bottom+mh/2,'Net closed-road length reduction (km)',rotation=90,ha='center',va='center',fontsize=11.5)
    ax=fig.add_axes([mapleft,bottom,mapwidth,mh]);axes['e']=ax
    for poly in m['water_polygons']:add_polygon(ax,poly,'#c8e6f0')
    ax.add_collection(LineCollection(m['background'],colors='#d7dfe5',linewidths=.28,zorder=2,rasterized=True))
    ax.add_collection(LineCollection(m['targets'],colors='#99337d',linewidths=2.2,zorder=4,rasterized=True))
    ax.add_collection(LineCollection(m['barrier_lonlat'],colors='#ee7733',linewidths=1.5,zorder=5,rasterized=True))
    ax.add_collection(LineCollection(m['links'],colors='#4477aa',linewidths=.9,linestyles=(0,(3,2)),alpha=.8,zorder=6))
    ax.scatter(m['receiver_lon'],m['receiver_lat'],marker='D',s=27,facecolor='#56b4e9',edgecolor='#1c648c',linewidth=.6,zorder=7)
    ax.scatter(m['source_lon'],m['source_lat'],marker='o',s=38,facecolor='#4477aa',edgecolor='white',linewidth=.7,zorder=8)
    for g in m['wards']:
        for part in poly_parts(g):
            xx,yy=part.exterior.xy;x,y=m['to_geo'].transform(xx,yy);ax.plot(x,y,color='#53616b',lw=.7,zorder=3)
    ax.set_xlim(m['extent'][:2]);ax.set_ylim(m['extent'][2:]);ax.set_aspect(m['aspect'],adjustable='box');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}°E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:.2f}°N'))
    ax.tick_params(labelsize=9.5);ax.set_axisbelow('line');ax.grid(color='#75838b',lw=.55,linestyle=(0,(3,3)),alpha=.55)
    for spine in ax.spines.values():spine.set_visible(True);spine.set_edgecolor('#303030');spine.set_linewidth(.8)
    ax.annotate('e: Hypothetical local deployment',(0,1),xycoords='axes fraction',xytext=(-22,11),textcoords='offset points',fontsize=12,fontweight='bold',va='bottom',ha='left',annotation_clip=False)
    measures=[Line2D([0],[0],color=COLORS[i],ls=STYLES[i],lw=1.9,label=LABELS[i]) for i in [1,2,3]]
    ml=fig.legend(handles=measures,loc='upper center',bbox_to_anchor=(left+curvewidth/2,1.08/height),ncol=1,frameon=False,fontsize=10,handlelength=3)
    map_handles=[Line2D([0],[0],color='#99337d',lw=2.2,label='Target roads (4.74 km)'),Line2D([0],[0],color='#ee7733',lw=1.6,label='Barriers (6.96 km)'),Line2D([0],[0],color='#4477aa',marker='o',ls='none',label='Pump intakes (19)'),Line2D([0],[0],color='#56b4e9',marker='D',ls='none',label='Receiving cells'),Line2D([0],[0],color='#4477aa',lw=.9,ls='--',label='Assumed transfer links'),Patch(facecolor='#c8e6f0',label='Water bodies')]
    gl=fig.legend(handles=map_handles,loc='upper center',bbox_to_anchor=(mapleft+mapwidth/2,1.08/height),ncol=2,frameon=False,fontsize=9.3,columnspacing=2,handlelength=2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();pos={k:ax.get_position() for k,ax in axes.items()}
    errors={'a_e_top':abs(pos['a'].y1-pos['e'].y1),'d_e_bottom':abs(pos['d'].y0-pos['e'].y0),'map_double_curve_width':abs(pos['e'].width-2*pos['a'].width)}
    for k in 'bcd':errors['a_'+k+'_left']=abs(pos['a'].x0-pos[k].x0);errors['a_'+k+'_right']=abs(pos['a'].x1-pos[k].x1)
    assert max(errors.values())<1e-9,errors
    assert fig._suptitle is None and all(not ax.get_title() for ax in axes.values())
    labels=[ax.get_tightbbox(renderer) for ax in axes.values()];legends=[ml.get_window_extent(renderer),gl.get_window_extent(renderer)]
    for i,b in enumerate(labels):
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,('clipped panel',list(axes)[i])
        for j,other in enumerate(labels):
            if i<j:assert not b.overlaps(other),('panel overlap',list(axes)[i],list(axes)[j])
    for b in legends:
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,'clipped legend'
        assert all(not b.overlaps(other) for other in labels),'legend overlaps panel'
    assert not legends[0].overlaps(legends[1]);assert not any(ylabel.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer)) for ax in axes.values())
    OUTPUT.parent.mkdir(exist_ok=True,parents=True);fig.savefig(OUTPUT,dpi=450);fig.savefig(AUDIT/f'{SLUG}_preview.png',dpi=135);plt.close(fig)
    with Image.open(OUTPUT) as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<=x<=451 for x in dpi)
    return {'layout':['aee','bee','cee','dee'],'figure_inches':[width,height],'dimensions_px':dimensions,'dpi':dpi,'panel_positions':{k:list(v.bounds)for k,v in pos.items()},'alignment_errors':errors,'shared_y_label':True,'common_curve_axes':True,'curve_lines_identical_to_source':len(lines),'geographic_aspect_preserved':True,'map_graticules_and_four_borders':True,'labels_and_legends_inside_image':True,'panel_and_legend_overlaps':False}

def main():
    existing={str(p.relative_to(ROOT)):sha(p) for folder in ['data/results/figures','data/results/tables'] for p in (ROOT/folder).iterdir() if p.is_file() and p!=OUTPUT}
    z=prepare_curves();m=prepare_map(z);layout=render(z,m)
    assert all(sha(ROOT/k)==v for k,v in BIND.items()),'Map input changed'
    assert all(sha(ROOT/k)==v for k,v in z['bindings'].items()),'Curve source changed'
    assert sha(z['doc'])==z['doc_sha'];assert (m['dem'].stat().st_size,m['dem'].stat().st_mtime_ns)==m['dem_meta']
    assert all(sha(ROOT/k)==v for k,v in existing.items()),'Original publication output changed'
    plot=[]
    for (event,stage,arm),(t,y) in z['differences'].items():
        for frame,(hour,value) in enumerate(zip(t,y)):plot.append({'event':event,'stage':stage,'measure':arm,'frame':frame,'actual_elapsed_hours':hour,'net_closed_road_reduction_km':value})
    pd.DataFrame(plot).to_csv(AUDIT/f'{SLUG}_plot_data.csv',index=False)
    validation={'passed':True,'status':'generated_awaiting_human_review','decision':'MILI-D-20261024-021','panels':5,'layout':layout,'rain_parents':2,'event_stage_contexts':4,'original_case_states':16,'original_trajectory_rows':1744,'difference_rows':len(plot),'all12_difference_integrals_reproduced':True,'original16_clock_digests_unchanged':True,'barrier_faces':len(z['barriers']),'barrier_length_m':6960,'pump_intakes':len(z['pumps']),'nominal_capacity_ceiling_m3_s':9.5,'target_source_roads':118,'target_road_length_km':z['protected_km'],'source_and_receiver_native_coordinates_verified':True,'receivers_prescribed_stage':True,'deployment_not_benefit_extent':True,'coordinates_roundtrip_error_m':m['error'],'existing_publication_files_preserved':len(existing),'new_simulations':0,'new_routing':0,'new_fits':0,'sealed_test_access':False,'source_files_unchanged':True,'source_sha256':BIND,'output':str(OUTPUT.relative_to(ROOT)),'output_sha256':sha(OUTPUT),'script_sha256':sha(Path(__file__)),'difference_checks':z['difference_checks']}
    write_json(AUDIT/f'{SLUG}_validation.json',validation);write_json(AUDIT/f'{SLUG}_source_bindings.json',BIND)
    caption=f'''Local hypothetical deployment and road-closure effects. Panels a-d show simultaneous no-measure minus treatment closed-road length for 18 June 2024 at low and raised stage, then 11 September 2025 at low and raised stage. Orange dashed, blue dotted and green dash-dot lines indicate barrier, drainage and combined arms. Positive values indicate fewer closed kilometers; negative values indicate more. Same-time aggregate net reduction is not individual restored-road length or a difference between separately timed peaks. All panels use unchanged 0.30 m closure rules, the lower depth branch, native clocks, 109 sampled states and 836.958 km qualified road support, with 277.920 km unknown exposure separate. Panel e maps the shared frozen local deployment: 118 target source roads totaling {z['protected_km']:.3f} ward-clipped km; 696 ten-meter barrier faces totaling 6.96 km; nineteen hypothetical intakes; saved prescribed-stage receiving cells; assumed direct transfer links; and contextual PLATEAU water bodies. Road and structure lengths describe different objects. Map links are not surveyed pipes or hose routes; symbol widths and point sizes are enlarged for visibility, not physical dimensions. Each intake has capacity at most 0.5 cubic meters per second, totaling a nominal ceiling of 9.5, with 0.05 m trigger, 5 m lift cutoff and prior-storage constraints. Barriers stand 0.5 m above sampled static face ground, permit overtopping with coefficient 0.6 and retain existing road-crossing openings. Barrier arms enable the mapped barriers; drainage arms enable the pumps; combined arms enable both using the same locations. The map depicts treatment inputs, not the area benefiting from treatment. The original sixteen states comprise four reused no-measure and twelve treatment histories. The September parent supplied prior corridor-selection context; June is a limited transfer check. River storage, downstream capacity, actual construction feasibility and event-time traffic congestion are unvalidated. Dry starts and nine-hour end-window censoring remain. All values and locations are unchanged; no hydraulic simulation, routing or model fitting is added.'''
    (AUDIT/f'{SLUG}_caption.md').write_text(caption+'\n')
    (AUDIT/f'{SLUG}_methods.md').write_text('''# Physical Measure Deployment and Road Effects

AnaSOP Sections 5-8 preserve the original local two-parent physical-measure estimand and authorize a five-panel deployment-aware revision. The generator reads the existing analysis Parquet exports and independently verifies their manifests, original numerical admission, sixteen actual-clock digests and twelve baseline-minus-treatment exposure integrals. All twelve difference ranges and integrals agree with the preceding accepted image receipt. Four trajectory panels retain identical axes, original values and line styles; a shared ordinate and bottom key reduce repeated labels.

The deployment panel uses the frozen barrier and pump CSVs, matching GeoJSON and native spatial contract. Every barrier coordinate and length agrees with the saved deployment; every intake and receiving endpoint agrees with its indexed native cell. Sources are dynamic corridor cells and receivers are prescribed-stage cells. All transfer lengths reproduce saved values and remain within the original 500 m screen. Ward-clipped target geometries reproduce the saved 4.743 km total. Longitude/latitude display is a coordinate transformation without source resampling, with roundtrip error below 0.00001 m. PLATEAU water polygons are contextual geometry; receiving endpoints remain the model's authoritative transfer targets. No new target selection, capacity, threshold, mask or engineering conclusion is introduced. Map marker sizes are display symbols rather than facility footprints.

The accepted four-panel master and every other current publication output remain byte-identical. The revised output remains pending human review. This rendering introduces no hydraulic run, routing, assignment, training or sealed-test access.
''')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Physical Measure Deployment and Road Effects</title><style>body{{font:16px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;background:#f7f9fb;color:#203b4b}}img{{width:100%;height:auto}}aside{{padding:18px;background:#e6eef5}}a{{color:#176a96}}</style><h1>Physical Measure Deployment and Road Effects</h1><aside>Five-panel revision awaiting visual acceptance. Layout: aee / bee / cee / dee. Original curves and hypothetical local deployment; English; 450 dpi.</aside><img src="../../../results/figures/{OUTPUT.name}?v={validation['output_sha256'][:12]}" alt="Four stacked intervention-effect trajectories beside a full-height deployment map"><h2>Caption and scope</h2><p>{html.escape(caption)}</p><p><a href="../../../results/figures/{OUTPUT.name}">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Numerical and layout checks</a> · <a href="{SLUG}_plot_data.csv">Plot data</a> · <a href="{SLUG}_methods.md">Methods</a></p></html>'''
    (AUDIT/f'{SLUG}_review.html').write_text(page)
    print('GENERATED',OUTPUT,flush=True);print(json.dumps({'passed':True,'layout':layout,'protected_km':z['protected_km'],'original_outputs_preserved':len(existing),'pump_intakes':19,'barrier_faces':696},indent=2),flush=True)
if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'physical_measure_deployment_and_road_effects')
