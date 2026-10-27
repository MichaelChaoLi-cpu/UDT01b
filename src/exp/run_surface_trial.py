"""Exploratory mass-conserving storage/diffusive surface routing, not event replay.
Synthetic equal-volume rain fields; artificial candidate water connections/stages.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,csv,time
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt,label,find_objects
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b');PRO=ROOT/'data/processed/three_ward_surface_trial';OUT=ROOT/'data/exp/data-briefing/three_ward_surface_trial';DX=20.

def csvwrite(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def jswrite(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def edge_flux(z,h,phi,roads,src,dst):
 eta=z+h;delta=eta[src]-eta[dst];crest=np.maximum(z[src],z[dst]);hf=np.maximum(np.maximum(eta[src],eta[dst])-crest,0);n=.05-.02*(roads[src]+roads[dst])/2;return np.sign(delta)*DX*np.minimum(phi[src],phi[dst])*hf**(5/3)/n*np.sqrt(np.abs(delta)/DX)

def step(z,h,phi,roads,src,dst,bs,base_stage,stage_add,bank_extra,rain_m3_s,dt,loss_mm_h,opened):
 area=phi*DX*DX;v=h*area;rain=rain_m3_s*dt;v+=rain;loss=np.minimum(v,loss_mm_h/1000/3600*area*dt);v-=loss;h=v/area;eta=z+h;q=edge_flux(z,h,phi,roads,src,dst)
 if opened:
  stage=base_stage+stage_add;delta=eta[bs]-stage;crest=z[bs]+bank_extra;hf=np.maximum(np.maximum(eta[bs],stage)-crest,0);qb=np.sign(delta)*DX*phi[bs]*hf**(5/3)/.05*np.sqrt(np.abs(delta)/(DX/2))
 else:qb=np.zeros(len(bs))
 # Monotone face-volume control: at most20% of a pair's level-equalizing
 # volume per step. Four neighbours + boundary faces cannot overshoot stage.
 equalize=np.abs(eta[src]-eta[dst])/(1/area[src]+1/area[dst])
 q=np.sign(q)*np.minimum(np.abs(q),.2*equalize/dt)
 if opened:qb=np.sign(qb)*np.minimum(np.abs(qb),.2*np.abs(eta[bs]-stage)*area[bs]/dt)
 outgoing=np.bincount(src,weights=np.maximum(q,0),minlength=len(z))+np.bincount(dst,weights=np.maximum(-q,0),minlength=len(z))+np.bincount(bs,weights=np.maximum(qb,0),minlength=len(z));factor=np.minimum(1,np.divide(.95*v,outgoing*dt,out=np.ones_like(v),where=outgoing>0));q*=np.where(q>=0,factor[src],factor[dst]);qb*=np.where(qb>=0,factor[bs],1)
 change=np.bincount(dst,weights=q*dt,minlength=len(z))-np.bincount(src,weights=q*dt,minlength=len(z))-np.bincount(bs,weights=qb*dt,minlength=len(z));v+=change
 if v.min()<-1e-7:raise AssertionError('negative volume')
 h=np.maximum(v,0)/area;return h,float(rain.sum()),float(loss.sum()),float(np.maximum(-qb,0).sum()*dt),float(np.maximum(qb,0).sum()*dt),q,int(np.sum(factor<.999))

def numerical_checks():
 src=np.array([0]);dst=np.array([1]);empty=np.array([],dtype=int);z=np.array([0.,1.]);h=np.array([2.,1.]);phi=np.ones(2);roads=np.zeros(2);assert np.max(np.abs(edge_flux(z,h,phi,roads,src,dst)))==0
 h=np.array([.3,0.]);z=np.array([1.,0.]);vol0=float(h.sum()*DX*DX)
 for i in range(30):h,*_=step(z,h,phi,roads,src,dst,empty,np.array([]),0,.25,np.zeros(2),5,0,False)
 assert h.min()>=0 and abs(h.sum()*DX*DX-vol0)<1e-8
 # Flat plane: 60mm over one hour gives 60mm storage, including under small dt.
 z=np.zeros(2);h=np.zeros(2)
 for i in range(360):h,*_=step(z,h,phi,roads,src,dst,empty,np.array([]),0,.25,np.ones(2)*.06/3600*DX*DX,10,0,False)
 assert np.allclose(h,.06,atol=1e-10)
 # Four simultaneously incoming faces must not exceed the largest water level.
 src=np.zeros(4,dtype=int);dst=np.arange(1,5);z=np.zeros(5);h=np.array([.1,2,2,2,2.]);phi=np.array([.125,1,1,1,1.]);roads=np.zeros(5)
 h,*_=step(z,h,phi,roads,src,dst,empty,np.array([]),0,.25,np.zeros(5),5,0,False);assert h.min()>=0 and h.max()<=2+1e-10
 h=np.zeros(1);net=0
 for i in range(60):
  h,ri,lo,wi,wo,q,lim=step(np.zeros(1),h,np.ones(1),np.zeros(1),empty,empty,np.array([0]),np.array([1.]),0,.25,np.zeros(1),2.5,0,True);net+=wi-wo;assert 0<=h[0]<=1+1e-9
 assert abs(h[0]*400-net)<1e-8
 return {'lake_at_rest':'pass','wet_dry_positivity_and_mass':'pass','flat_uniform_rain_depth':'pass','four_face_no_surface_level_overshoot':'pass','prescribed_stage_no_overshoot_and_mass':'pass'}

def main():
 audit=json.loads((OUT/'input_audit.json').read_text());a=dict(np.load(PRO/'surface_inputs.npz'));land=a['land'];shape=land.shape;N=int(land.sum());idx=np.full(shape,-1,dtype=np.int32);idx[land]=np.arange(N);z=a['z'][land];phi=1-a['building_fraction'][land].astype(float);roads=a['road_fraction'][land].astype(float);ward=a['ward'][land];edge_uncertain=a['edge_uncertain'][land];water=a['water'];candidate=a['candidate_water'];tf=Affine(*audit['transform']);sr=[];ds=[];orientation=[];banknodes=[];bankids=[]
 # Four-neighbour finite-volume faces: no transport across diagonal contacts.
 for axis in [1,0]:
  left=(slice(None),slice(None,-1)) if axis==1 else (slice(None,-1),slice(None));right=(slice(None),slice(1,None)) if axis==1 else (slice(1,None),slice(None));il=idx[left];ir=idx[right];keep=(il>=0)&(ir>=0);sr.append(il[keep]);ds.append(ir[keep]);orientation.extend([axis]*int(keep.sum()))
  for si,other,ow,oc in [(il,ir,water[right],candidate[right]),(ir,il,water[left],candidate[left])]:
   k=(si>=0)&(other<0)&oc;banknodes.extend(si[k].tolist());bankids.extend(ow[k].tolist())
 src=np.concatenate(sr);dst=np.concatenate(ds);orientation=np.array(orientation);bs=np.array(banknodes,dtype=int);bi=np.array(bankids,dtype=int);stages={r['id']:r['synthetic_stage_low_m'] for r in audit['synthetic_water_stages']};valid=np.array([stages.get(int(i)) is not None for i in bi]);bs=bs[valid];bi=bi[valid];stage=np.array([stages[int(i)] for i in bi]);bank_extra=.25
 catch=a['domain']&np.isfinite(a['z'])&(water==0);near=distance_transform_edt(~land,return_distances=False,return_indices=True);target=idx[near[0][catch],near[1][catch]];assert (target>=0).all();cw=a['ward'][catch];rawrain=np.where(cw==3,100.,np.where(cw==1,30.,np.where(cw==2,20.,40.)));norm=60/rawrain.mean();rainfield={'uniform':np.ones(len(cw))*60,'zoned':rawrain*norm};rainvec={k:np.bincount(target,weights=r/1000/3600*DX*DX,minlength=N) for k,r in rainfield.items()};assert np.isclose(rainvec['uniform'].sum(),rainvec['zoned'].sum())
 config={'grid_m':20,'time_step_seconds':2.5,'duration_minutes':90,'rain_duration_minutes':60,'rain_total_domain_average_mm':60,'rain_hyetograph':'0-15min and45-60min:0.5 times average;15-45min:1.5 times average','rain_is_observed_20240821':False,'zoned_synthetic_mean_rates_mm_h':{'千代田区':30*norm,'中央区':20*norm,'港区':100*norm,'buffer':40*norm},'roughness_n':'0.05 minus0.02 times road fraction, uncalibrated','loss_capacity_mm_h':15,'loss_kind':'Combined assumed surface loss/drainage capacity on open area; no actual pipe capacity or infiltration map.','water_connection_barrier_extra_m':bank_extra,'water_boundary_stage_definition':'Per-object lower adjacent-land quantile minus0.5m; artificial low stage. High scenario adds up to1m. Original WTR Z never used.','full_building_runoff':'Cells >=90% covered by roof/footprint: route gross roof rainfall to nearest active cell, not to sewers. Partial-cell rain retained in open-area storage.','water_interior_rainfall':'Excluded: no pond/river storage or catchment-to-river model.','edge_condition':'Closed, no external upstream hydrographs. 1km requested buffer is intersected with available terrain;500m near uncovered edges flagged. Not a full basin model.','hydraulic_method':'Explicit finite-volume Manning/diffusive storage-routing prototype with monotone face-equalization cap plus available-volume outflow limiter, no inertia, no pipe network, no engine claim. Porosity/open face widths approximated from building cover.','land_cells':N,'land_to_land_faces':len(src),'candidate_land_to_water_faces':len(bs),'measured_bank_crest':False,'scenario_definitions':{'A':'uniform rain, all water exchange closed','B':'equal-volume zoned rain, all water exchange closed','C':'same zoned rain, candidate river/sea bank faces open to artificial low stage','D':'same as C, artificial water stage rises1m (20-40min), holds(40-70min), falls(70-90min)'}};jswrite(OUT/'model_assumptions.json',config)
 validation=numerical_checks();summ=[];hydros=[];outputs={}
 def run(key,rkind,opened,high,dt,save=True):
  h=np.zeros(N);peak=np.zeros(N);initial=0.;rain=loss=win=wout=0.;qx=np.zeros(N);qy=np.zeros(N);maxres=0.;limited=0;start=time.monotonic();hist=[]
  for it in range(int(90*60/dt)):
   t=it*dt;m=t/60;shape_factor=.5 if m<15 or 45<=m<60 else 1.5 if 15<=m<45 else 0;rise=0 if m<20 else min((m-20)/20,1) if m<70 else max((90-m)/20,0);add=rise if high else 0
   h,ri,lo,wi,wo,q,lim=step(z,h,phi,roads,src,dst,bs,stage,add,bank_extra,rainvec[rkind]*shape_factor,dt,15,opened);rain+=ri;loss+=lo;win+=wi;wout+=wo;limited+=lim;np.maximum(peak,h,out=peak);storage=float((h*phi*DX*DX).sum());res=storage-(rain+win-loss-wout);maxres=max(maxres,abs(res))
   if save:
    ex=q*(orientation==1)*dt;ey=-q*(orientation==0)*dt;qx+=np.bincount(src,weights=ex,minlength=N);qy+=np.bincount(src,weights=ey,minlength=N)
   if (it+1)*dt%300==0:
    hist.append({'scenario':key,'time_min':(it+1)*dt/60,'rain_m3':rain,'loss_m3':loss,'water_in_m3':win,'water_out_m3':wout,'surface_storage_m3':storage,'mass_residual_m3':res})
   if (it+1)*dt%1800==0:print(key,'minute',int((it+1)*dt/60),'maxdepth',round(float(h.max()),3),'mass',round(res,7),flush=True)
  relative=maxres/max(rain+win,1);assert relative<1e-9
  if save:
   p=np.full(shape,np.nan,dtype='float32');p[land]=peak;end=p.copy();end[land]=h;xx=np.zeros(shape,dtype='float32');yy=xx.copy();xx[land]=qx;yy[land]=qy
   np.savez_compressed(PRO/f'{key}_result.npz',peak=p,final=end,flow_east_m3=xx,flow_north_m3=yy)
   with rasterio.open(PRO/f'{key}_peak_depth.tif','w',driver='GTiff',height=shape[0],width=shape[1],count=1,dtype='float32',crs=6677,transform=tf,nodata=np.nan,compress='deflate') as ds:ds.write(p,1)
   for w,name in [(1,'千代田区'),(2,'中央区'),(3,'港区')]:
    for interior in [False,True]:
     k=(ward==w)&(~edge_uncertain if interior else True);summ.append({'scenario':key,'ward':name,'edge_zone_excluded':interior,'land_area_km2':float(k.sum()*DX*DX/1e6),'screen_area_peak_ge_0_10m_km2':float(np.sum(k&(peak>=.1))*DX*DX/1e6),'screen_area_peak_ge_0_30m_km2':float(np.sum(k&(peak>=.3))*DX*DX/1e6),'p95_peak_depth_m':float(np.quantile(peak[k],.95)) if k.any() else None,'max_peak_depth_m':float(peak[k].max()) if k.any() else None})
   hydros.extend(hist);outputs[key]={'peak':p,'final':end,'flow_east':xx,'flow_north':yy};validation[key]={'relative_mass_residual':relative,'max_mass_residual_m3':maxres,'negative_depths':int(np.sum(h<0)),'limiter_cell_step_count':limited,'rain_m3':rain,'loss_m3':loss,'water_in_m3':win,'water_out_m3':wout,'end_storage_m3':storage,'runtime_s':time.monotonic()-start}
  return peak
 for key,kind,opened,high in [('A','uniform',False,False),('B','zoned',False,False),('C','zoned',True,False),('D','zoned',True,True)]:run(key,kind,opened,high,2.5)
 finer=run('C_dt1_25','zoned',True,False,1.25,False);fine_grid=np.full(shape,np.nan,dtype='float32');fine_grid[land]=finer;np.savez_compressed(PRO/'C_timestep_check_1_25s.npz',peak=fine_grid);coarse=outputs['C']['peak'][land].astype(float);diff=np.abs(coarse-finer);validation['timestep_comparison']={'scenario':'C','dt_s':[2.5,1.25],'mean_abs_peak_difference_m':float(diff.mean()),'p95_abs_peak_difference_m':float(np.quantile(diff,.95)),'max_abs_peak_difference_m':float(diff.max()),'area_ge_0_30m_relative_difference':float(abs(np.sum(coarse>=.3)-np.sum(finer>=.3))/max(np.sum(finer>=.3),1)),'interpretation':'Numerical timestep check only, not calibration, grid convergence or physical validation.'};jswrite(OUT/'numerical_validation.json',validation);csvwrite(OUT/'ward_scenario_summary.csv',summ);csvwrite(OUT/'volume_balance_timeseries.csv',hydros)
 hotspots=[]
 # Priority polygons: >=0.30m in >=10 cells and outside data-edge flag, sorted by area.
 for key in outputs:
  p=outputs[key]['peak'];mask=land&~a['edge_uncertain']&(a['building_fraction']<=.75)&(p>=.3)&(p<=2);labels,nlab=label(mask);counts=np.bincount(labels.ravel());largest=sorted(range(1,nlab+1),key=lambda i:counts[i],reverse=True)[:20]
  for i in largest:
   k=labels==i
   if counts[i]<10:continue
   rr,cc=np.where(k);j=np.argmax(p[k]);r,c=int(rr[j]),int(cc[j]);x,y=tf*(c+.5,r+.5);hotspots.append({'scenario':key,'rank':len([q for q in hotspots if q['scenario']==key])+1,'cluster_cells':int(counts[i]),'screen_cluster_area_m2':int(counts[i]*DX*DX),'peak_depth_m':float(p[k].max()),'representative_x':x,'representative_y':y,'location_role':'Highest model cell in screening cluster, not a surveyed flood point','ward_id':int(a['ward'][r,c]),'needs_check':'Synthetic screening only; >2m or >75%roof covered cells excluded from ranking. Terrain interpolation, road/building connectivity and outlets not validated.'})
 if hotspots:csvwrite(OUT/'screening_hotspot_candidates.csv',hotspots)
 plots(outputs,summ,hydros,a,audit,tf);report(config,validation,summ,audit,hotspots)
 print('DONE',json.dumps(validation['timestep_comparison']),flush=True)

def plots(outputs,summ,hydros,a,audit,tf):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 from matplotlib.colors import ListedColormap,BoundaryNorm
 from matplotlib.patches import Polygon
 from pyproj import Transformer
 from shapely.geometry import shape
 from shapely.ops import transform
 font=FontProperties(fname='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 nr,nc=a['land'].shape;extent=[tf.c/1000,(tf.c+nc*DX)/1000,(tf.f-nr*DX)/1000,tf.f/1000];wardg=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];toxy=Transformer.from_crs(6668,6677,always_xy=True);bound=[transform(toxy.transform,shape(f['geometry'])) for f in wardg]
 def boundaries(ax):
  for g in bound:
   for p in (g.geoms if g.geom_type=='MultiPolygon' else [g]):xy=np.array(p.exterior.coords)/1000;ax.plot(xy[:,0],xy[:,1],color='#555',lw=.5)
  ax.set_aspect('equal');ax.set_xlabel('東西 km');ax.set_ylabel('南北 km')
 fig,ax=plt.subplots(1,3,figsize=(15,7));im=ax[0].imshow(np.where(a['land'],a['z'],np.nan),extent=extent,cmap='terrain',vmin=0,vmax=35);fig.colorbar(im,ax=ax[0],shrink=.6,label='地形標高 m (PLATEAU T.P.宣言)');ax[0].set_title('20m計算地形／元TIN約5m');im=ax[1].imshow(a['building_fraction'],extent=extent,cmap='Greys',vmin=0,vmax=1);fig.colorbar(im,ax=ax[1],shrink=.6,label='建物被覆率');ax[1].set_title('建物被覆／屋根縁の代理を含む');ax[2].imshow(np.where(a['edge_uncertain'],1,np.nan),extent=extent,cmap=ListedColormap(['#ef8b51']));ax[2].imshow(np.where(a['water']>0,1,np.nan),extent=extent,cmap=ListedColormap(['#6fb4dc']));ax[2].set_title('橙：データ縁500m／青：水域')
 for x in ax:boundaries(x)
 fig.suptitle('三区地表試算の入力・計算範囲（上流流域全体ではない）',fontsize=15);fig.tight_layout();fig.savefig(OUT/'surface_input_overview.png',dpi=160);plt.close(fig)
 cm=ListedColormap(['#eef4fa','#c4dff0','#77b6d7','#2c7eaf','#754a9b','#bb2857']);norm=BoundaryNorm([0,.05,.1,.3,.5,1,2],cm.N);fig,ax=plt.subplots(2,2,figsize=(12,13));titles={'A':'A 一様降雨／水域交換閉鎖','B':'B 同じ総雨量を分区／交換閉鎖','C':'C 分区／仮の河渠・海側出口を開放','D':'D 同じ出口／仮の水位を最大1m上げる'}
 for x,(key,v) in zip(ax.ravel(),outputs.items()):
  im=x.imshow(np.where(v['peak']<=2,v['peak'],np.nan),extent=extent,cmap=cm,norm=norm);x.imshow(np.where(v['peak']>2,1,np.nan),extent=extent,cmap=ListedColormap(['#333333']));x.imshow(np.where(a['water']>0,1,np.nan),extent=extent,cmap=ListedColormap(['#6295a9']),alpha=.7);boundaries(x);x.set_title(titles[key],fontsize=12)
 fig.colorbar(im,cax=fig.add_axes([.865,.18,.025,.60]),label='試算最大水深 m（観測・予測精度は未検証）',ticks=[0,.05,.1,.3,.5,1]);fig.suptitle('三区の地表貯留・流下スクリーニング\n合成降雨60mm平均／損失15mm/h／20m格子／90分\n黒：>2mのモデル疑義格（実際の浸水深とは読まない）',fontsize=15);fig.subplots_adjust(top=.85,wspace=.24,hspace=.18,right=.82);fig.savefig(OUT/'surface_scenario_comparison.png',dpi=170);plt.close(fig)
 # Net accumulated face transport, aggregated to500m arrows; not flow velocity.
 v=outputs['C'];fig,ax=plt.subplots(figsize=(9,9));im=ax.imshow(np.where(v['peak']<=2,v['peak'],np.nan),extent=extent,cmap=cm,norm=norm);ax.imshow(np.where(v['peak']>2,1,np.nan),extent=extent,cmap=ListedColormap(['#333333']));ax.imshow(np.where(a['water']>0,1,np.nan),extent=extent,cmap=ListedColormap(['#6295a9']));ar=[]
 for r in range(0,nr,25):
  for c in range(0,nc,25):
   ex=float(v['flow_east'][r:r+25,c:c+25].sum());ey=float(v['flow_north'][r:r+25,c:c+25].sum());mag=np.hypot(ex,ey)
   if mag>5000:
    br,bc=np.where(a['land'][r:r+25,c:c+25]);j=int(np.argmin((br-12)**2+(bc-12)**2));x,y=tf*(c+bc[j]+.5,r+br[j]+.5);ar.append((x/1000,y/1000,ex/mag*.25,ey/mag*.25))
 if ar:
  ar=np.array(ar);ax.quiver(ar[:,0],ar[:,1],ar[:,2],ar[:,3],angles='xy',scale_units='xy',scale=1,width=.002,color='#34404e',alpha=.8)
 boundaries(ax);fig.colorbar(im,ax=ax,shrink=.7,label='C試算最大深さ m（未検証）',ticks=[0,.05,.1,.3,.5,1]);ax.set_title('C：合成降雨での地表輸水方向\n500mブロック累積通量／矢印は固定長、速度ではない',fontsize=13);fig.tight_layout();fig.savefig(OUT/'surface_flow_directions.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,5));groups=['千代田区','中央区','港区'];colors=['#4777ac','#eea541','#63a68d','#bd638f'];positions=np.arange(3)
 for j,(key,col) in enumerate(zip(outputs,colors)):
  vals=[next(r['screen_area_peak_ge_0_30m_km2'] for r in summ if r['scenario']==key and r['ward']==w and r['edge_zone_excluded']) for w in groups];ax[0].bar(positions+(j-1.5)*.18,vals,width=.18,label=key,color=col);h=[r for r in hydros if r['scenario']==key];ax[1].plot([r['time_min'] for r in h],[r['surface_storage_m3']/1e6 for r in h],label=key,color=col)
 ax[0].set_xticks(positions,groups);ax[0].set_ylabel('最大深さ>=0.30mの試算面積 km²');ax[0].set_title('データ縁500mを除く／実績面積ではない');ax[1].set_xlabel('開始後 min');ax[1].set_ylabel('地表貯留量 百万m³');ax[1].set_title('域内地表貯留（河川貯留は含まない）')
 for x in ax:x.legend();x.grid(axis='y',alpha=.2)
 fig.suptitle('分区雨量と水域境界の感度：仮定付きの比較',fontsize=14);fig.tight_layout();fig.savefig(OUT/'surface_scenario_metrics.png',dpi=170);plt.close(fig)

def report(config,validation,summ,audit,hotspots):
 text=f'''三区地表试算：第一版（2026-10-14）
已运行四个合成情景，输出最大水深栅格、分区筛查面积和水量平衡。不是2024-08-21实况复现，也未检验水深/道路通行精度。
采用20m格网，原始PLATEAU约5m TIN重心坐标线性取样；未填洼、未加入建筑屋顶高度、未合并0.50m DEM。
建筑解析器优先LOD0 FootPrint；实际包内此次全部使用LOD0 RoofEdge（屋顶轮廓），以其作为建筑占地代理，在5m中间栅格计覆盖率，再汇总20m。不能分辨窄道路、路缘或各建筑出入口。建筑覆率>=90%格不流通，其屋顶雨量转给最近开放格；其余按开放比例计储水和输水宽度。这个近似不能替代真实街道通道。
粗糙率0.03–0.05随道路覆率变化，排水/下渗合并假设15mm/h，仅在有水开放面扣减；均未标定。没有完整管网也能启动这一步，但这不证明管网不重要。

计算域：三区外请求1km缓冲与已下载TIN可用范围的交集。可用地表面积{audit['land_km2']:.2f}km²；三区缺少地形的格数{audit['terrain_cells_missing_by_ward']}（含水域缺值）；非水域缺值另见input_coverage_validation.json。
缓冲并非完整古川/神田川上游流域，缺外部来水过程；外缘关闭，距缺数据边缘500m的格标为不可靠并另给剔除汇总。行政界内部允许流动。边界影响可能向500m以外传播，剔除500m不是消除误差的证明。下一版应扩到实际汇水域或接上游边界流量。

情景A/B总雨量相同：计算域非水域面平均60mm，持续60min，前后各15min为0.5倍、中间30min为1.5倍，再退水30min。
A为均匀空间分布；B采用港100/千代田30/中央20/缓冲40的相对强度后归一到域平均60mm。这些是合成对照，不是官方事件插值。归一后的分区平均mm/h见model_assumptions.json。
C用同一分区雨量，将已命名河渠及源分类海域的相邻候选格面开放；其余池塘、皇居濠和无名水体不当出口。
D用C的候选连接，将人工边界水位在20–40min升1m、40–70min维持、70–90min回落。
每个水体人工低水位=相邻20m地表高程10%分位数-0.5m，格面岸障=相邻地形+0.25m。这不是实测岸顶、水位、入口或高程基准转换。WTR原始Z从未使用。C/D只能检验候选连接/抬水位假设敏感性，不能证明街道水能排入某河渠；未把古川约4.3m上涨套入模型。
水域内部不储水、不模拟河道/海水传播，也没有泵站、潮汐或调节池过程；只作外边界交换，其流入/流出单独记账。调节池42200m³没有被当排水能力。

求解：自写显式有限体积地表储水/扩散型Manning通量；无惯性、地下排水、建筑内部或地下通道。水流只沿四邻接格面。每格面每步限制在两侧水面趋于平衡所需水量的20%，四邻面合计保持水面单调；另用可用水量限制保证非负。限流会影响时间步精度，因此必须比较时间步。
已验证平水面静止、干湿界非负与守恒、平地均匀降雨累积。每情景动态雨量+边界流入-损失-流出=储水，质量残差见numerical_validation.json。
C的2.5s/1.25s时步比较：平均峰值差{validation['timestep_comparison']['mean_abs_peak_difference_m']:.6f}m，95%差{validation['timestep_comparison']['p95_abs_peak_difference_m']:.6f}m；最大差{validation['timestep_comparison']['max_abs_peak_difference_m']:.3f}m，>=0.30m筛查面积相对差{validation['timestep_comparison']['area_ge_0_30m_relative_difference']:.2%}。局部深度尚未数值收敛，>0.10m差异格另存掩膜；不能声称局部深度准确。这只是数值时间步检查；没有网格收敛或观测验证。初次未加水面单调控制的计算产生异常大水深，已弃用并覆盖，不纳入交付。

部分格仍出现数米深水，主要涉及建筑高覆盖和粗格网地形/通道表达。这些不是已证实真实洼地；>2m格标黑，且不进入热点排序；屋顶覆率>75%也不进入排序。黑格不是无风险或无数据，而是模型疑义信号。阈值仅为探索性筛查，不能代替验证。

可用结果：三区尺度下哪些较低区域对雨量分布、河道边界敏感；为下一轮数据补齐排序。不可用结果：具体街道预测水深、未来风险概率、精确岸顶阈值、2024事件拟合优劣。
优先补数据：跨区汇水域裸地DEM与建筑/路网；分布式实际降雨；古川/神田川/日本桥川/隅田川与海侧水位过程及统一基准；候选河岸交换位置的设施拓扑。先补会改变大片区域结果的数据，再做局部岸顶/路缘测量。
这一步保存原包CRC/成员来源，不改原ZIP，不做Git/DVC提交或推送。
方法背景（本脚本不是HEC-HMS/RAS实现）：https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/2d-diffusion-wave-model
'''
 (OUT/'assessment.txt').write_text(text)
 links=['surface_input_overview.png','surface_scenario_comparison.png','surface_scenario_metrics.png','surface_flow_directions.png'];imgs=''.join(f'<img src="{n}" alt="地表试算图">' for n in links)
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区地表试算</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:28px auto;padding:0 16px;color:#203044}}h1{{font-size:26px}}img{{max-width:100%;height:auto}}.note{{background:#fff1d3;padding:15px}}a{{color:#1667a1}}</style><h1>三区地表试算：第一版</h1><p class="note">四个合成情景已运行。结果用于比较地形、雨量分布和水域边界的敏感性；不是2024暴雨复现，也不是具体街道水深预测。20m格网，未标定。黑格代表>2m的模型疑义结果，不能读作真实洪深。</p><p>A：均匀降雨，水域交换关闭；B：同样总量的分区降雨，交换关闭；C：分区降雨，候选河渠/海侧连接开放；D：同C，人工水位抬升1m。</p><p>域平均60mm雨量持续60分钟，再退水30分钟；假设损失容量15mm/h。池塘、皇居濠和无名水体没有被自动设成出口。C/D的水位、岸障和入口是假设，原水域Z未作为水位。</p>{imgs}<p><a href="assessment.txt">完整解释与下一步</a> · <a href="ward_scenario_summary.csv">三区情景指标</a> · <a href="volume_balance_timeseries.csv">水量平衡</a> · <a href="model_assumptions.json">全部模型假设</a> · <a href="input_audit.json">输入来源与覆盖</a> · <a href="numerical_validation.json">数值验证</a> · <a href="../../../processed/three_ward_surface_trial/A_peak_depth.tif">A最大水深栅格</a> · <a href="../../../processed/three_ward_surface_trial/B_peak_depth.tif">B栅格</a> · <a href="../../../processed/three_ward_surface_trial/C_peak_depth.tif">C栅格</a> · <a href="../../../processed/three_ward_surface_trial/D_peak_depth.tif">D栅格</a></p><p>缺完整上游汇水域及实际来水、分布式雨量和基准水位；边缘500m仅作影响提示。下一版先补这些大尺度输入，再针对敏感区核查。</p></html>''')
if __name__=='__main__':main()
