#!/usr/bin/env python3
"""Receiving Allowance Road and Resident Effects.
AnaSOP5: same three parents, stages and frozen sites; allowance-only comparison.
AnaSOP6: original absolute road exposure, candidate loss bounds and fixed map.
AnaSOP7-8: acc/bcc/dde/ddf assembly. No solver, routing, fitting or training.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import hashlib,json,html
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,PathPatch
from matplotlib.ticker import MaxNLocator,FuncFormatter
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from figure_resident_access_loss_and_restoration import polygon_path,MAP
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
P=ROOT/'data/processed/three_ward_final_evidence_v1'
B=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
SLUG='receiving_allowance_road_and_resident_effects'
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'
TITLE='Receiving Allowance Road and Resident Effects'
EVENTS=['E20240731','E20240830','E20250911']
ARMS=['none','Q2p5','Q5'];COLORS={'none':'#252b30','Q2p5':'#4477aa','Q5':'#228833'}
POOLS=['Furukawa','Kamejima','Kanda','Nihonbashi','Sumida']
POOLCOLORS=dict(zip(POOLS,['#cc6677','#ddaa33','#117733','#44aadd','#aa4499']))
BIND={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def js(p):return json.loads(bind(p).read_text())
def csv(p):return pd.read_csv(bind(p),float_precision='round_trip',dtype={'Block ID':str})
def pq(p):return pd.read_parquet(bind(p))
def date(e):return f'{e[1:5]}-{e[5:7]}-{e[7:9]}'
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def prepare():
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert f'| {TITLE} |' in doc
 bind(Path(__file__));bind(ROOT/'src/analyses/figure_resident_access_loss_and_restoration.py')
 assert js(ROOT/'data/exp/data-preprocessing/three_ward_final_evidence_v1/validation.json')['passed']
 assert js(R/'deployment_assumptions_and_receiving_limits_validation.json')['passed']
 road=pq(P/'road_effects.parquet');population=pq(P/'resident_loss.parquet');paired=pq(P/'paired_resident_effects.parquet');maps=pq(P/'map_classifications.parquet')
 assert len(road)==12 and len(population)==18 and len(paired)==18 and len(maps)==9492
 for name,df in [('road_effects',road),('resident_loss',population),('paired_resident_effects',paired),('map_classifications',maps)]:pd.testing.assert_frame_equal(csv(P/(name+'.csv')),df,check_dtype=False,rtol=1e-12,atol=1e-8)
 assert not road.duplicated(['Rainfall Parent','Receiving Water Stage','Measure Configuration']).any()
 assert population['Resident Population'].eq(496345).all() and population['Unresolved Resident Stock'].eq(41308).all()
 assert (population['Resident Access Loss Lower']<=population['Resident Access Loss Upper']).all()
 np.testing.assert_allclose(road['No Action Closed Road Duration']-road['Remaining Closed Road Duration'],road['Avoided Closed Road Duration'],rtol=0,atol=1e-8)
 np.testing.assert_allclose(road['Restored Road Exposure']-road['Newly Closed Road Exposure'],road['Avoided Closed Road Duration'],rtol=0,atol=1e-8)
 assert road['Qualified Road Length'].eq(road['Qualified Road Length'].iloc[0]).all() and road['Road Depth Branch'].eq('lower').all()
 main=csv(R/'physical_road_and_resident_comparison_numeric_main_values.csv')
 assert len(main)==12
 for row in road.to_dict('records'):
  z=main[main['Rainfall Parent'].eq(row['Rainfall Parent'])&main['Receiving Water Stage'].eq(row['Receiving Water Stage'])&main['Measure Configuration'].eq(row['Measure Configuration'])].iloc[0]
  np.testing.assert_allclose([z['No Action Road Exposure (km h)'],z['Remaining Road Exposure (km h)']],[row['No Action Closed Road Duration'],row['Remaining Closed Road Duration']],rtol=0,atol=1e-8)
 # Preserve the complete directly paired effect ledger separately from absolute loss.
 direct=paired[paired['Reference Portfolio'].eq('Q2p5')&paired['Measure Configuration'].eq('Q5')].copy()
 assert len(direct)==6 and not direct.duplicated(['Rainfall Parent','Receiving Water Stage']).any()
 direct_support=csv(R/'physical_road_and_resident_comparison_direct_allowance_contrasts.csv')
 keys=['Rainfall Parent','Receiving Water Stage','Reference Portfolio','Measure Configuration']
 pd.testing.assert_frame_equal(direct.sort_values(keys).reset_index(drop=True),direct_support.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-8)
 assert (direct['Paired Resident Benefit Lower']<=direct['Paired Resident Benefit Upper']).all()
 assert set(maps.Frame)=={97} and set(maps['Rainfall Parent'])=={'E20240830'} and set(maps['Reference Portfolio'])=={'Q2p5'} and set(maps['Measure Configuration'])=={'Q5'}
 mh=maps[maps['Receiving Water Stage'].eq('H')].set_index('Block ID').sort_index();ml=maps[maps['Receiving Water Stage'].eq('L')].set_index('Block ID').sort_index()
 assert mh.Class.equals(ml.Class) and mh['Resident Population'].equals(ml['Resident Population'])
 assert len(mh)==4746 and mh['Resident Population'].sum()==496345 and mh.loc[mh.Class.eq('Unresolved attachment'),'Resident Population'].sum()==41308
 geo=js(ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson')['features'];assert len(geo)==4746
 to_geo=Transformer.from_crs(6677,4326,always_xy=True);to_metric=Transformer.from_crs(4326,6677,always_xy=True)
 geoms={str(f['properties']['unit_id']):transform(to_geo.transform,shape(f['geometry']))for f in geo};assert set(geoms)==set(mh.index)
 wards=[transform(to_geo.transform,shape(f['geometry']))for f in js(B/'wards_6677.geojson')['features']]
 bounds=np.array([g.bounds for g in wards]);xmin,ymin=bounds[:,:2].min(0);xmax,ymax=bounds[:,2:].max(0);dx=(xmax-xmin)*1.05;dy=(ymax-ymin)*1.05
 extent=[(xmin+xmax-dx)/2,(xmin+xmax+dx)/2,(ymin+ymax-dy)/2,(ymin+ymax+dy)/2];aspect=1/np.cos(np.deg2rad((ymin+ymax)/2))
 manifests=[ROOT/'data/processed/three_ward_s1_reallocation_pilot_v1/batches/O1/run_manifest.json',ROOT/'data/processed/three_ward_s1r_shared_allowance_pilot_v1/batches/O1/run_manifest.json',ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1/run_manifest.json']
 sitefile=manifests[0].parent/'pump_sites.csv';sites=csv(sitefile).sort_values('Pump ID').reset_index(drop=True)
 assert len(sites)==319 and sites['Source Native Cell One Based'].is_unique and not sites['Field Deployment Verified'].any()
 assert sites.groupby('Receiving Pool').size().reindex(POOLS).tolist()==[58,79,37,80,65]
 assert sites['Standalone Rate Ceiling'].eq(.5).all()
 runchecks=[]
 for manifest in manifests:
  for run in js(manifest)['runs']:
   folder=ROOT/run['folder'];batch=folder.parent.parent
   pd.testing.assert_frame_equal(csv(batch/'pump_sites.csv').sort_values('Pump ID').reset_index(drop=True),sites)
   pumpfile=bind(folder/'udt_pumps.dat');assert sha(pumpfile)==run['input_sha256']['udt_pumps.dat']
   z=np.loadtxt(pumpfile,skiprows=1,ndmin=2);assert z.shape==(319,5)
   np.testing.assert_array_equal(z[:,0],sites['Source Native Cell One Based']);np.testing.assert_array_equal(z[:,1],sites['Receiver Native Cell One Based']);np.testing.assert_allclose(z[:,2],sites['Reserved Rate'],rtol=0,atol=1e-15)
   assert (z[:,3]==.05).all() and (z[:,4]==5).all()
   rulesfile=bind(folder/'udt_pool_rules.dat');assert sha(rulesfile)==run['input_sha256']['udt_pool_rules.dat']
   lines=rulesfile.read_text().splitlines();assert lines[0]=='UDT_WEIGHTED_POOL_V1' and int(lines[1])==5 and int(lines[7])==319
   caps=[float(s.split()[1])for s in lines[2:7]];assert caps==[.5]*5 or caps==[1.]*5
   rules=np.array([list(map(float,s.split())) for s in lines[8:]]);assert rules.shape==(319,3) and (rules[:,2]==.5).all()
   for i,pool in enumerate(POOLS,1):np.testing.assert_array_equal(rules[:,1]==i,sites['Receiving Pool'].eq(pool))
   runchecks.append({'case':run['case_id'],'total_receiving_allowance_m3_s':sum(caps),'same_319_sources_and_receivers':True})
 assert len(runchecks)==12
 with np.load(bind(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz')) as z:
  xy=z['native_xy'].copy();kcs=z['native_kcs'].copy();source=sites['Source Native Cell One Based'].to_numpy()-1;receiver=sites['Receiver Native Cell One Based'].to_numpy()-1
  assert np.array_equal(z['native_grid_north_index'][source],sites['Grid Index'])
 assert (kcs[source]==1).all() and (kcs[receiver]==2).all()
 distance=np.linalg.norm(xy[source]-xy[receiver],axis=1);np.testing.assert_allclose(distance,sites['Straight Distance Lower Bound'],rtol=0,atol=1e-8);assert distance.max()<=1000+1e-8
 sx,sy=to_geo.transform(xy[source,0],xy[source,1]);rx,ry=to_geo.transform(xy[receiver,0],xy[receiver,1]);xx,yy=to_metric.transform(sx,sy)
 roundtrip=float(np.max(np.abs(np.column_stack([xx,yy])-xy[source])));assert roundtrip<1e-5
 alllon=np.r_[sx,rx];alllat=np.r_[sy,ry]
 extent=[min(extent[0],float(alllon.min())-.0005),max(extent[1],float(alllon.max())+.0005),min(extent[2],float(alllat.min())-.0005),max(extent[3],float(alllat.max())+.0005)]
 assert ((alllon>=extent[0])&(alllon<=extent[1])&(alllat>=extent[2])&(alllat<=extent[3])).all()
 links=[np.array([[a,b],[c,d]])for a,b,c,d in zip(sx,sy,rx,ry)]
 water=[shape(f['geometry'])for f in js(ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson')['features']]
 background=[]
 for f in js(B/'clipped_roads_6677.geojson')['features']:
  g=shape(f['geometry'])
  for part in [g] if g.geom_type=='LineString' else list(g.geoms):
   p=np.array(part.coords);x,y=to_geo.transform(p[:,0],p[:,1]);background.append(np.column_stack([x,y]))
 return locals()

def render(z):
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.05)
 plt.rcParams.update({'axes.linewidth':.8,'savefig.facecolor':'white','hatch.linewidth':.35})
 width=14.;left=.09;small=.245;large=.49;right=.95;mapright=right-large;statright=right-small
 ratio=(z['extent'][3]-z['extent'][2])/(z['extent'][1]-z['extent'][0])*z['aspect'];mapheight=large*width*ratio
 footer=1.8;blockgap=2.25;top=.7;height=footer+2*mapheight+blockgap+top
 lower=footer/height;upper=(footer+mapheight+blockgap)/height;mh=mapheight/height;statgap=1.35/height;sh=(mh-statgap)/2
 fig=plt.figure(figsize=(width,height));axes={
 'a':fig.add_axes([left,upper+sh+statgap,small,sh]),'b':fig.add_axes([left,upper,small,sh]),
 'c':fig.add_axes([statright,lower+sh+statgap,small,sh]),'d':fig.add_axes([statright,lower,small,sh]),
 'e':fig.add_axes([mapright,upper,large,mh]),'f':fig.add_axes([left,lower,large,mh])}
 roadplot=[];intervalplot=[];road_bars=0;intervals=0
 for key,stage in [('a','H'),('b','L')]:
  ax=axes[key]
  for ei,event in enumerate(EVENTS):
   group=z['road'][z['road']['Rainfall Parent'].eq(event)&z['road']['Receiving Water Stage'].eq(stage)].set_index('Measure Configuration');assert len(group)==2
   base=float(group['No Action Closed Road Duration'].iloc[0]);np.testing.assert_allclose(group['No Action Closed Road Duration'],base,rtol=0,atol=1e-8)
   for arm,offset in zip(ARMS,[.23,0,-.23]):
    value=base if arm=='none' else float(group.loc[arm,'Remaining Closed Road Duration']);y=2-ei+offset
    bar=ax.barh(y,value,height=.17,color=COLORS[arm],zorder=3)[0];assert abs(bar.get_width()-value)<1e-10
    ax.annotate(f'{value:.1f}',(value,y),xytext=(4,0),textcoords='offset points',fontsize=8.7,ha='left',va='center',color=COLORS[arm]);road_bars+=1
    roadplot.append({'event':event,'stage':stage,'portfolio':arm,'cumulative_road_closure_km_h':value})
   if ei<2:ax.axhline(1.5-ei,color='#e4e9ed',lw=.6,zorder=0)
  ax.set(xlim=(0,1050),ylim=(-.55,2.55),xlabel='Cumulative road closure (km h)');ax.set_xticks([0,250,500,750,1000]);ax.set_yticks([2,1,0],[date(e)for e in EVENTS]);ax.tick_params(labelsize=8.8);ax.grid(axis='x',color='#dce2e6',lw=.6);ax.set_axisbelow(True)
 for key,stage in [('c','H'),('d','L')]:
  ax=axes[key]
  ax.axvline(0,color='#64717b',ls='--',lw=.9,zorder=1)
  for ei,event in enumerate(EVENTS):
   group=z['direct'][z['direct']['Rainfall Parent'].eq(event)&z['direct']['Receiving Water Stage'].eq(stage)]
   assert len(group)==1;row=group.iloc[0]
   lo=float(row['Paired Resident Benefit Lower'])/1000;hi=float(row['Paired Resident Benefit Upper'])/1000;y=2-ei
   line=ax.hlines(y,lo,hi,color='#aa4499',lw=2.2,zorder=3);np.testing.assert_allclose(line.get_segments()[0],[[lo,y],[hi,y]],rtol=0,atol=1e-10)
   ax.plot([lo,hi],[y,y],ls='none',marker='|',markersize=7,color='#aa4499',markeredgewidth=1.3,zorder=4);intervals+=1
   # Outward-rounded display bounds; raw directly paired values are unchanged.
   for value,label in [(lo,np.floor(lo*1000)/1000),(hi,np.ceil(hi*1000)/1000)]:
    ax.annotate(f'{label:.3f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',va='bottom',fontsize=8.8,color='#88367a')
   intervalplot.append({'event':event,'stage':stage,'reference':'Q2p5','treatment':'Q5','paired_benefit_lower_resident_h':lo*1000,'paired_benefit_upper_resident_h':hi*1000})
   if ei<2:ax.axhline(1.5-ei,color='#e4e9ed',lw=.6,zorder=0)
  ax.set(xlim=(-2,60),ylim=(-.55,2.55),xlabel='Additional access benefit\n(thousand resident-hours)');ax.set_xticks([0,15,30,45,60]);ax.set_yticks([2,1,0],[date(e)for e in EVENTS]);ax.tick_params(labelsize=8.8);ax.grid(axis='x',color='#dce2e6',lw=.6);ax.set_axisbelow(True)
 def map_style(ax):
  for g in z['wards']:
   for poly in [g] if g.geom_type=='Polygon' else list(g.geoms):
    p=np.asarray(poly.exterior.coords);ax.plot(p[:,0],p[:,1],color='#445762',lw=.75,zorder=7)
  ax.set_xlim(z['extent'][:2]);ax.set_ylim(z['extent'][2:]);ax.set_aspect(z['aspect'],adjustable='box');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}°E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:.2f}°N'));ax.tick_params(labelsize=9)
  ax.set_axisbelow('line');ax.grid(color='#8d9da9',lw=.45,ls=(0,(3,3)),alpha=.55)
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
 ax=axes['e']
 patches=[PathPatch(polygon_path(g))for g in z['water']];ax.add_collection(PatchCollection(patches,facecolor='#c8e6f0',edgecolor='none',zorder=1))
 ax.add_collection(LineCollection(z['background'],colors='#dce4e9',linewidths=.24,zorder=2,rasterized=True))
 ax.add_collection(LineCollection(z['links'],colors='#617e94',linewidths=.48,linestyles=(0,(3,2)),alpha=.45,zorder=3))
 for pool in POOLS:
  use=z['sites']['Receiving Pool'].eq(pool).to_numpy();ax.scatter(z['sx'][use],z['sy'][use],s=10,color=POOLCOLORS[pool],edgecolor='white',linewidth=.15,zorder=6)
 unique=np.unique(z['receiver']);rec=Transformer.from_crs(6677,4326,always_xy=True).transform(z['xy'][unique,0],z['xy'][unique,1]);ax.scatter(rec[0],rec[1],s=28,marker='D',facecolor='#ffffff',edgecolor='#34566c',linewidth=.8,zorder=5)
 map_style(ax)
 ax=axes['f'];seen=[];mapstats=[]
 for name,(color,hatch)in MAP.items():
  ids=z['mh'].index[z['mh'].Class.eq(name)].tolist()
  if ids:
   seen.append(name);patches=[PathPatch(polygon_path(z['geoms'][i]))for i in ids];ax.add_collection(PatchCollection(patches,facecolor=color,edgecolor='#60717e'if hatch else '#b6c0c8'if name in ['No access change','No recorded residents']else color,linewidth=.13,hatch=hatch,zorder=2))
  mapstats.append({'class':name,'basic_units':len(ids),'resident_stock_in_units':int(z['mh'].loc[ids,'Resident Population'].sum())if ids else 0})
 map_style(ax)
 titles={'a':'a: High stage, road closure','b':'b: Low stage, road closure','c':'e: High stage, Q5 relative to Q2.5','d':'f: Low stage, Q5 relative to Q2.5','e':'c: Common hypothetical deployment','f':'d: Additional access restoration'}
 for key,ax in axes.items():
  ax.annotate(titles[key],(0,1),xycoords='axes fraction',xytext=(-22,12),textcoords='offset points',fontsize=11.4 if key in 'abcd' else 12.2,fontweight='bold',va='bottom',ha='left',annotation_clip=False)
 poolhandles=[Line2D([0],[0],ls='none',marker='o',markersize=5,color=POOLCOLORS[p],label=p)for p in POOLS]+[Line2D([0],[0],ls='none',marker='D',markerfacecolor='white',markeredgecolor='#34566c',label='Receiving cells'),Line2D([0],[0],ls='--',lw=.8,color='#617e94',label='Assumed links'),Patch(facecolor='#c8e6f0',label='Water bodies')]
 upperlegend=fig.legend(handles=poolhandles,loc='upper center',bbox_to_anchor=(mapright+large/2,upper-.68/height),ncol=3,frameon=False,fontsize=9,columnspacing=1.5,handlelength=1.5)
 fhandles=[Patch(facecolor=MAP[n][0],edgecolor='#60707c',hatch=MAP[n][1],label=n)for n in seen]
 maplegend=fig.legend(handles=fhandles,loc='upper center',bbox_to_anchor=(left+large/2,1.13/height),ncol=2,frameon=False,fontsize=9,columnspacing=1.4,handlelength=1.8)
 labels={'none':'No action','Q2p5':'Q2.5: lower allowance','Q5':'Q5: higher allowance'}
 roadlegend=fig.legend(handles=[Line2D([0],[0],color=COLORS[a],lw=2.1,label=labels[a])for a in ARMS],loc='upper center',bbox_to_anchor=(left+small/2,upper-.68/height),ncol=1,frameon=False,fontsize=9.2,handlelength=2)
 benefitlegend=fig.legend(handles=[Line2D([0],[0],color='#aa4499',lw=2.1,marker='|',markersize=7,label='Q5 relative to Q2.5'),Line2D([0],[0],color='#64717b',lw=.9,ls='--',label='Zero additional benefit')],loc='upper center',bbox_to_anchor=(statright+small/2,1.13/height),ncol=1,numpoints=2,frameon=False,fontsize=9.2,handlelength=2)
 assert [t.get_text()for t in roadlegend.get_texts()]==[labels[a]for a in ARMS]
 assert [t.get_text()for t in benefitlegend.get_texts()]==['Q5 relative to Q2.5','Zero additional benefit']
 fig.canvas.draw();renderer=fig.canvas.get_renderer();pos={k:ax.get_position()for k,ax in axes.items()}
 errors={'a_c_top':abs(pos['a'].y1-pos['e'].y1),'b_c_bottom':abs(pos['b'].y0-pos['e'].y0),'e_d_top':abs(pos['c'].y1-pos['f'].y1),'f_d_bottom':abs(pos['d'].y0-pos['f'].y0),'maps_equal_width':abs(pos['e'].width-pos['f'].width),'maps_equal_height':abs(pos['e'].height-pos['f'].height),'a_b_left':abs(pos['a'].x0-pos['b'].x0),'e_f_right':abs(pos['c'].x1-pos['d'].x1)}
 assert max(errors.values())<1e-9,errors
 assert road_bars==18 and intervals==6 and fig._suptitle is None and all(not a.get_title()for a in axes.values())
 boxes=[a.get_tightbbox(renderer)for a in axes.values()];legends=[l.get_window_extent(renderer)for l in [upperlegend,maplegend,roadlegend,benefitlegend]]
 for i,b in enumerate(boxes):
  assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,('clipped axes',list(axes)[i])
  for j,b2 in enumerate(boxes):
   if i<j:assert not b.overlaps(b2),('panel collision',list(axes)[i],list(axes)[j])
 for i,b in enumerate(legends):
  assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,'clipped legend'
  assert not any(b.overlaps(x)for x in boxes),'legend/panel collision'
  for b2 in legends[i+1:]:assert not b.overlaps(b2),'legend collision'
 # Check every value annotation against all other annotations and panels.
 valueboxes=[t.get_window_extent(renderer)for k in 'abcd'for t in axes[k].texts if t.get_text()[0].isdigit()]
 assert not any(a.overlaps(b)for i,a in enumerate(valueboxes)for b in valueboxes[i+1:]),'value-label collision'
 OUT.parent.mkdir(exist_ok=True,parents=True);R.mkdir(exist_ok=True,parents=True);fig.savefig(OUT,dpi=450);fig.savefig(R/f'{SLUG}_preview.png',dpi=100);plt.close(fig)
 with Image.open(OUT) as im:dpi=list(im.info['dpi']);size=list(im.size);assert all(449<=d<=451 for d in dpi)
 pd.DataFrame(roadplot).to_csv(R/f'{SLUG}_road_plot_data.csv',index=False);pd.DataFrame(intervalplot).to_csv(R/f'{SLUG}_resident_plot_data.csv',index=False);pd.DataFrame(mapstats).to_csv(R/f'{SLUG}_map_summary.csv',index=False)
 return {'layout':['acc','bcc','dde','ddf'],'figure_inches':[width,height],'dimensions_px':size,'dpi':dpi,'road_bars':road_bars,'resident_intervals':intervals,'interval_midpoints':False,'panel_positions':{{'a':'a','b':'b','c':'e','d':'f','e':'c','f':'d'}[k]:list(p.bounds)for k,p in pos.items()},'alignment_errors':errors,'maps_identical_geographic_extent_and_aspect':True,'panel_and_legend_collisions':False,'axis_text_inside_image':True,'value_labels_nonoverlapping':True}

def main():
 existing={str(p.relative_to(ROOT)):sha(p)for folder in ['data/results/figures','data/results/tables']for p in (ROOT/folder).iterdir()if p.is_file()and p!=OUT}
 z=prepare();layout=render(z)
 assert all(sha(ROOT/n)==h for n,h in BIND.items()),'Source changed'
 assert all(sha(ROOT/n)==h for n,h in existing.items()),'Prior publication output changed'
 caption='''Receiving-allowance effects on roads and resident vehicle access. Panels a and b show absolute nine-hour cumulative closed-road exposure for three historical radar rainfall parents under high and low prescribed water stages. Black, blue and green road bars indicate no action, Q2.5 and Q5; smaller road totals are preferable. Units are kilometer-hours, not unique closed kilometers. Unlike the earlier Static/O1/S1 plot, these road values are the later matched receiving-allowance results used for the resident comparison. Panels e and f now show the six original directly candidate-paired additional resident-access benefits of Q5 relative to Q2.5, in thousand resident-hours. The road-only black/blue/green key is placed below a/b. The separate key below e/f denotes purple candidate-paired benefit intervals and the gray dashed zero-benefit reference. Purple intervals and their outward-rounded endpoint labels show net reduction in access-loss exposure; positive values indicate added benefit and larger values more reduction. No midpoint or statistical confidence interpretation is supplied. All six lower bounds are positive within the declared candidate model, supporting a positive net direction for this comparison while retaining uncertain magnitudes and separate local new-loss outcomes. These values are inherited direct pairs, never differences of marginal absolute interval endpoints. Original absolute losses and benefits relative to no action remain in supporting data. Panel c maps the same 319 hypothetical modeled intakes and fixed receiving endpoints under both allowances, colored by their five assigned pools. Straight dashed links are assumed geometric connections, not surveyed pipes. Q2.5 and Q5 assume pool caps 0.5 and 1.0 cubic meters per second, totaling 2.5 and 5.0, with the same locations, weights, allocator, eligibility and per-intake ceiling. Locations are not machine counts or validated downstream receiving capacity. Panel d shows the original August frame-97 higher-relative-to-lower allowance access-restoration classification, elapsed 8:05. High- and low-stage classifications are identical at this fixed anchor, permitting one display without asserting equal full-window histories. Dark green denotes all-candidate restoration and light green possible restoration. Unchanged access can remain disconnected; hatched attachment is unresolved and white units have no recorded residents. Reporting-unit resident stocks are not confirmed restored people. Whole-area official nighttime stock is 496,345 and unresolved stock is 41,308. The road denominator is 836.958 km qualified with 277.920 km unknown; the lower mapped-depth branch, 0.30 m closure rule and 10 m population candidate radius are unchanged. Full-window statistics and the snapshot map use different declared temporal summaries. Direct paired benefits and local adverse outcomes remain separately retained. Historical forcing, dry starts and assumed stages support conditional scenarios, not observed flood, stranded-person or traffic-congestion reconstruction. River storage and downstream acceptance remain unmodeled. No solver, routing, fitting, new population allocation or training is run for this assembly.'''
 (R/f'{SLUG}_caption.md').write_text(caption+'\n')
 (R/f'{SLUG}_methods.md').write_text('''# Receiving Allowance Road and Resident Effects

AnaSOP Sections 5-8 authorize this staggered six-panel trial, labeled acc / bcc / dde / ddf. Panel letters follow the revised reading order; axes, data and locations remain unchanged. It aligns the later receiving-allowance road ledger with direct higher-relative-to-lower allowance resident benefits, rather than relabeling the older Static/O1/S1 results. Road exposure identities, six shared no-action totals, both allowance outcomes and the already accepted main-table numeric register are checked. All eighteen absolute resident-loss states remain unchanged in the final ledger. The six direct higher-relative-to-lower allowance benefit intervals are independently matched to the existing supporting contrast register and plotted without endpoint subtraction. The complete no-action, restoration and new-loss ledgers are retained. Midpoints are not used; labels round outward to resident-hours before scaling to thousands. Road and resident keys are separated: the three portfolio colors apply to a/b; the purple paired-benefit interval and gray dashed zero reference apply to e/f.

Every source/receiver identity, frozen weight, trigger and lift is checked across twelve hydraulic input manifests and their pump files. Five pool memberships and weighted rules remain fixed; only the five cap values change. The native spatial contract maps the indexed cells to metric coordinates and its north-grid indices agree with the site registry. Intake and receiver masks remain dynamic and prescribed-stage respectively. Straight distances reproduce the saved 1000 m screen. Coordinate transforms preserve scientific geometry, with roundtrip error below 0.00001 m. Water and road vectors provide geographic context; the nineteen-unit local pilot contributes no deployment locations to this figure.

The 4746 official reporting geometries are retained without centroid reassignment. The complete high- and low-stage August classifications are compared by unit identifier and population before one common restoration map is drawn. Map time is frame 97, whereas statistical panels integrate nine hours. Secondary text beneath map headings is omitted from the image at the user's request; full comparison and time metadata remain in the caption. Numerical and visual layout checks include mirrored map sizes, common geographic extent/aspect, synchronized plotting edges, legend separation and complete label bounds. All previous publication files remain byte-identical. The trial stays pending human visual acceptance, with no new simulation or route computation.
''')
 validation={'passed':True,'status':'trial_awaiting_human_visual_review','decision':'MILI-D-20261024-024','revision_decision':'MILI-D-20261024-028','separate_road_and_resident_legends':True,'map_panel_subtitles':False,'panels':6,'layout':layout,'same_allowance_comparison_for_roads_and_residents':True,'older_Static_O1_S1_not_relabelled':True,'road_ledger_rows':12,'resident_ledger_rows':18,'resident_panel_estimand':'direct_Q5_relative_to_Q2p5_net_benefit','direct_intervals_independently_checked':6,'paired_resident_rows_preserved':18,'road_total_checks_against_existing_numeric_register':12,'shared_site_and_receiver_cases_verified':z['runchecks'],'intake_locations':319,'pool_counts':[58,79,37,80,65],'intakes_not_machine_counts':True,'native_cell_coordinates_verified':True,'coordinate_roundtrip_error_m':z['roundtrip'],'map_basic_units':4746,'population_total':496345,'unresolved_population':41308,'H_L_anchor_classifications_identical':True,'restoration_frame':97,'restoration_elapsed_hours':97/12,'original_publication_files_preserved':len(existing),'source_files_unchanged':True,'new_simulations':0,'new_routing':0,'new_fits':0,'training':False,'sealed_test_access':False,'source_sha256':BIND,'output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT),'script_sha256':sha(Path(__file__))}
 dump(R/f'{SLUG}_validation.json',validation);dump(R/f'{SLUG}_source_bindings.json',BIND)
 page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{TITLE}</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:30px auto;padding:0 20px;background:#f7f9fb;color:#203b4b}}img{{width:100%;height:auto}}aside{{padding:18px;background:#e6eef5}}a{{color:#176a96}}</style><h1>{TITLE}</h1><aside>Staggered six-panel trial; awaiting visual review. Layout acc / bcc / dde / ddf. Roads: absolute burdens. Residents: directly paired Q5-relative-to-Q2.5 benefit. English; 450 dpi.</aside><img src="../../../results/figures/{OUT.name}?v={validation['output_sha256'][:12]}" alt="Road bars and deployment map above a restoration map and directly paired resident-benefit intervals"><h2>Caption and evidence limits</h2><p>{html.escape(caption)}</p><p><a href="../../../results/figures/{OUT.name}">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Checks</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_road_plot_data.csv">Road data</a> · <a href="{SLUG}_resident_plot_data.csv">Resident data</a></p></html>'''
 (R/f'{SLUG}_review.html').write_text(page)
 print('GENERATED',OUT,flush=True);print(json.dumps({'passed':True,'layout':layout,'original_outputs_preserved':len(existing),'same_intake_locations':319,'map_classes':z['mh'].Class.value_counts().to_dict()},indent=2),flush=True)
if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'receiving_allowance_road_and_resident_effects')
