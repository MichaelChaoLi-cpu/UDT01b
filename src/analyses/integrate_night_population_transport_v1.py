"""Census-resident stock and conditional vehicle-access consequences, not stranded people.
No resident is assigned to a centroid or converted from a vehicle-demand proxy.
"""
from pathlib import Path
import os,sys,json,time,datetime as dt,hashlib,html
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt-population-mpl')
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import shape,Point,mapping
from shapely import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import load,dump,sha
TAG='three_ward_night_population_transport_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
POP=ROOT/'data/processed/three_ward_population_block_stability_v1';N=ROOT/'data/processed/three_ward_normal_transport_network_v1';REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1';F=ROOT/'data/processed/three_ward_cross_event_protection_v1';MEASURE=ROOT/'data/processed/three_ward_demand_protection_feedback_v1';C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1'
CASES=['E20250911__H__U','E20240830__H__U'];BUFFERS=[0.,10.,25.];BOUNDS=['lower','upper'];WARD={'13101':'Chiyoda','13102':'Chuo','13103':'Minato'}
def save(p,df):pd.DataFrame(df).to_csv(p,index=False,encoding='utf-8-sig')
def distances(rr,pr,cost,closed,gates):
 """Original turn-state graph with two boundary super-sources; no planar noding.
 Outward traversal uses transposed graph. Inbound starts pay their own first arc.
 """
 s=rr['state_physical_arc'];S=len(s);comp=rr['node_component'];start=comp[pr['from_index'][s]];end=comp[rr['state_arc_end_node']];row=np.repeat(np.arange(S,dtype='i4'),np.diff(rr['indptr']));col=rr['indices'];openstate=~closed[pr['arc_to_road'][s]];keep=openstate[row]&openstate[col]
 rows=row[keep];cols=col[keep];data=cost[s[cols]];endgate=np.flatnonzero(np.isin(end,gates)&openstate);startgate=np.flatnonzero(np.isin(start,gates)&openstate)
 # In the transposed graph super-source reaches states terminating at a boundary.
 go=coo_matrix((np.r_[data,np.zeros(len(endgate))],(np.r_[cols,np.full(len(endgate),S)],np.r_[rows,endgate])),shape=(S+1,S+1)).tocsr()
 gi=coo_matrix((np.r_[data,cost[s[startgate]]],(np.r_[rows,np.full(len(startgate),S)],np.r_[cols,startgate])),shape=(S+1,S+1)).tocsr()
 out=dijkstra(go,indices=S,directed=True)[:S];inn=dijkstra(gi,indices=S,directed=True)[:S];out[~openstate]=np.inf;inn[~openstate]=np.inf
 nc=int(comp.max())+1;outc=np.full(nc,np.inf);inc=np.full(nc,np.inf);np.minimum.at(outc,start,cost[s]+out);np.minimum.at(inc,end,inn)
 # Zero distance when already at a boundary portal is valid.
 outc[gates]=0;inc[gates]=0
 return outc[comp],inc[comp]
def selfcheck():
 # Two-way chain with a dead spur; closing the middle arc removes outward access.
 rr={'state_physical_arc':np.arange(4),'node_component':np.arange(3),'state_arc_end_node':np.array([1,0,2,1]),'indptr':np.array([0,2,3,4,6]),'indices':np.array([1,2,0,3,1,2])}
 pr={'from_index':np.array([0,1,1,2]),'arc_to_road':np.array([0,0,1,1])};c=np.ones(4)
 o,i=distances(rr,pr,c,np.array([False,False]),np.array([2]));assert np.allclose(o,[2,1,0])and np.allclose(i,[2,1,0])
 o,i=distances(rr,pr,c,np.array([False,True]),np.array([2]));assert np.isinf(o[:2]).all()and np.isinf(i[:2]).all()
 return True

def main():
 t=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);assert selfcheck()
 inputs=[POP/'official_2020_block_count_baseline.csv',POP/'block_population_stability_6677.geojson',POP/'population_input_admission_list.csv',POP/'admission_policy.json',POP/'block_population_intervals.npz',N/'surface_nodes.csv',REP/'turn_router.npz',REP/'multi_network_endpoints.csv',M/'source_motor_graph.npz',M/'road_projection.npz',M/'road_inventory.csv',F/'motor_common_support.npz',C/'choice_costs.npz',MEASURE/'protection_masks.npz',ROOT/'data/processed/three_ward_combined_transport_measures_v1/protocol.json']+[F/'motor_exposures'/(c+'.npz')for c in CASES]
 binding={str(f.relative_to(ROOT)):sha(f)for f in inputs};dump(P/'input_bindings.json',dict(sha256=binding,script_sha256=sha(Path(__file__))))
 protocol=dict(decision='MILI-D-20261018-012',population='Official2020-10-01 census basic-unit resident stocks,4746units,496345residents; historical residence-location scenario,not simultaneous occupancy or2026population',cases=CASES,elapsed_seconds=14400,hydraulics='Existing radar rainfall plus same hypothetical1.5m raised river stage,Ugeometry; not historical flood reconstruction',population_clock='Apply a hypothetical at-residence stock to both existing4h snapshots; these snapshots are not both nighttime. No observed affected-person claim.',access='Conditional vehicle access in both directions to ANY of16frozen10km road-boundary portals. Outward/inward may reach different portals. Boundary crossing does not establish safety or access to a specific hospital/workplace.',candidates='Existing permitted ground motor nodes of non-motorway/non-trunk roads within official polygon or buffer;0/10/25m sensitivity,10mprimary. Candidate road access only;not verified building driveway/parking. No centroid assignment, snapping, new edges or planar crossing.',within_block='No equal resident distribution to nodes. All-candidate loss supports whole-block conditional loss; mixed candidate states retain interval0..officialcount. Baseline-mixed/unsupported units held. These are candidate-location bounds,not statistical confidence bounds or bounds on true stranded people.',road_evidence='Original common-qualified lower/upper flood depth and0.1/0.3m rules;unknown road exposure remains unknown and assumed available in main branch. No closed surface applied to bridge/tunnel elevation. Every summary is conditional on this road assumption.',measure='Same frozen118road/4743.231745m ideal passability restoration;no reselection. Demand deferral and guidance do not change physical connectivity;vehicle-trip gains are not converted to residents.',detour='Reference-cost shortest route to any boundary in each direction;slow roads2x,not congested queue delay.1.5ratio only descriptive scenario threshold. Worst of inward/outward ratios;not actual travel duration.',mask_and_allocation='Final population allocation-admission class retained. Original125/250m building intervals unchanged by reference andSHA;never sum marginal allocation upperbounds or rescale them to census totals.',daytime='Residence-based commuting fields are not destination daytime population;no automatic daytime population conversion.',new_hydraulics=False,new_training=False,sealed_test=False)
 dump(P/'protocol.json',protocol)
 df=pd.read_csv(POP/'official_2020_block_count_baseline.csv',dtype={'block_id':str,'ward':str,'town_id':str});assert df.block_id.is_unique and df.official_2020_residents.sum()==496345
 adm=pd.read_csv(POP/'population_input_admission_list.csv',dtype={'unit_id':str});ad=adm.set_index('unit_id');df['allocation_admission']=df.block_id.map(ad.allocation_input_class);assert df.allocation_admission.notna().all()
 gj=json.loads((POP/'block_population_stability_6677.geojson').read_text());assert gj['crs']['properties']['name']=='EPSG:6677';gm={f['properties']['unit_id']:shape(f['geometry'])for f in gj['features']};polys=[gm[k]for k in df.block_id]
 rr=load(REP/'turn_router.npz');pr=load(M/'road_projection.npz');native=load(M/'source_motor_graph.npz');inv=pd.read_csv(M/'road_inventory.csv');qual=load(F/'motor_common_support.npz')['qualified_roads'];protected=load(MEASURE/'protection_masks.npz')['targeted'];assert protected.sum()==118
 nodes=pd.read_csv(N/'surface_nodes.csv',usecols=['node_id','x_m','y_m']).set_index('node_id').reindex(native['node_ids']);assert nodes.notna().all().all();xy=nodes.to_numpy();roadok=(inv.grade_key.eq('0|0|surface')&~inv.highway.isin(['motorway','motorway_link','trunk','trunk_link'])).to_numpy();arcok=roadok[pr['arc_to_road']];eligible=np.unique(np.r_[pr['from_index'][arcok],pr['to_index'][arcok]]);points=[Point(*xy[j])for j in eligible];tree=STRtree(points)
 offsets={};indices={};supports=[]
 for buf in BUFFERS:
  lens=[];parts=[]
  for k,g in enumerate(polys):
   idx=tree.query(g if buf==0 else g.buffer(buf),predicate='intersects');ix=np.sort(eligible[idx]);parts.extend(ix.tolist());lens.append(len(ix));supports.append(dict(block_id=df.block_id.iloc[k],buffer_m=buf,candidate_nodes=len(ix)))
  offsets[buf]=np.r_[0,np.cumsum(lens)];indices[buf]=np.array(parts,dtype='i4')
 np.savez_compressed(P/'block_access_support.npz',block_ids=df.block_id.to_numpy(dtype=str),**{f'offsets_{int(b)}m':offsets[b]for b in BUFFERS},**{f'node_indices_{int(b)}m':indices[b]for b in BUFFERS})
 save(P/'candidate_support.csv',supports);ep=pd.read_csv(REP/'multi_network_endpoints.csv');gates=ep.loc[ep.kind.eq('conditional_outer_boundary_portal'),'component'].to_numpy(dtype='i4');assert len(gates)==16;save(P/'boundary_portals.csv',ep[ep.kind.eq('conditional_outer_boundary_portal')])
 tref=load(C/'choice_costs.npz')['travel_seconds'];normal=distances(rr,pr,tref,np.zeros(len(inv),bool),gates);normalok=np.isfinite(normal[0])&np.isfinite(normal[1]);results={};checks=[]
 for case in CASES:
  ex=load(F/'motor_exposures'/(case+'.npz'));assert abs(ex['actual_times'][5]-14400)<.001
  for bound in BOUNDS:
   depth=ex[bound][5];closed=qual&(depth>=.3);slow=qual&(depth>=.1)&~closed
   for arm in ['flood','targeted']:
    cl=closed.copy();sl=slow.copy()
    if arm=='targeted':cl[protected]=False;sl[protected]=False
    cost=tref.copy();cost[sl[pr['arc_to_road']]]*=2
    oo,ii=distances(rr,pr,cost,cl,gates);ok=np.isfinite(oo)&np.isfinite(ii);assert not(ok&~normalok).any(),'Flood cannot restore normal disconnection'
    out_ratio=np.divide(oo,normal[0],out=np.full_like(oo,np.inf),where=np.isfinite(normal[0])&(normal[0]>0));in_ratio=np.divide(ii,normal[1],out=np.full_like(ii,np.inf),where=np.isfinite(normal[1])&(normal[1]>0));ratio=np.maximum(out_ratio,in_ratio);results[(case,bound,arm)]=(ok,ratio)
    print(json.dumps(dict(case=case,bound=bound,arm=arm,normal_accessible_nodes=int(normalok.sum()),current_accessible_nodes=int(ok.sum()),seconds=round(time.monotonic()-t))),flush=True)
    np.savez_compressed(P/f'{case}__{bound}__{arm}_node_access.npz',outward_reference_seconds=oo,inward_reference_seconds=ii,normal_bidirectional=normalok,bidirectional=ok,reference_detour_ratio=ratio)
 np.savez_compressed(P/'normal_node_access.npz',outward_reference_seconds=normal[0],inward_reference_seconds=normal[1],bidirectional=normalok)
 knownnode=np.zeros(len(xy),bool);q=qual[pr['arc_to_road']];knownnode[pr['from_index'][q]]=True;knownnode[pr['to_index'][q]]=True
 rows=[];restored=[]
 for (case,bound,arm),(ok,ratio)in results.items():
  for buf in BUFFERS:
   ptr=offsets[buf];ix=indices[buf]
   for k,base in df.iterrows():
    ns=ix[ptr[k]:ptr[k+1]];n=len(ns);bn=int(normalok[ns].sum());cn=int(ok[ns].sum());complete=n>0 and bn==n;lost=normalok[ns]&~ok[ns];nloss=int(lost.sum());pop=float(base.official_2020_residents);detour=normalok[ns]&ok[ns]&(ratio[ns]>=1.5)
    status='no_candidate_support'if n==0 else 'normal_access_missing'if bn==0 else 'normal_support_mixed'if bn<n else 'all_candidates_lost'if cn==0 else 'partial_candidate_loss'if cn<n else 'connected_detour_all'if detour.all() else 'connected_detour_mixed'if detour.any() else 'connected_below_detour_threshold'
    lower=pop if complete and nloss==n else 0.;upper=pop if complete and nloss>0 else 0.;unresolved=pop if not complete else 0.
    rows.append(dict(case=case,bound=bound,arm=arm,buffer_m=buf,block_id=base.block_id,ward=base.ward,town_id=base.town_id,name=base['name'],residents_2020=pop,allocation_admission=base.allocation_admission,candidate_nodes=n,normal_bidirectional_candidates=bn,current_bidirectional_candidates=cn,candidates_with_local_qualified_road=int(knownnode[ns].sum()),lost_candidates=nloss,detour_candidates=int(detour.sum()),access_class=status,conditional_loss_population_lower=lower,conditional_loss_population_upper=upper,baseline_unresolved_population=unresolved,qualified_local_support_absent=bool(n>0 and not knownnode[ns].any())))
 for case in CASES:
  for bound in BOUNDS:
   f=results[(case,bound,'flood')][0];p=results[(case,bound,'targeted')][0];assert not(f&~p).any(),'Protection cannot remove access';recover=normalok&~f&p
   for buf in BUFFERS:
    ptr=offsets[buf];ix=indices[buf]
    for k,base in df.iterrows():
     ns=ix[ptr[k]:ptr[k+1]];complete=len(ns)>0 and normalok[ns].all();pop=float(base.official_2020_residents)
     restored.append(dict(case=case,bound=bound,buffer_m=buf,block_id=base.block_id,ward=base.ward,normal_support_complete=bool(complete),residents_2020=pop,recovered_candidates=int(recover[ns].sum()),conditional_restored_population_lower=pop if complete and recover[ns].all()else 0.,conditional_restored_population_upper=pop if complete and recover[ns].any()else 0.))
 tab=pd.DataFrame(rows);re=pd.DataFrame(restored);save(P/'block_population_access.csv',tab);save(P/'block_population_restoration.csv',re)
 summary=tab.groupby(['case','bound','arm','buffer_m','access_class'],as_index=False).agg(blocks=('block_id','size'),residents_2020=('residents_2020','sum'));save(P/'population_class_summary.csv',summary)
 metrics=tab.groupby(['case','bound','arm','buffer_m'],as_index=False).agg(total_population=('residents_2020','sum'),conditional_loss_lower=('conditional_loss_population_lower','sum'),conditional_loss_upper=('conditional_loss_population_upper','sum'),baseline_unresolved_population=('baseline_unresolved_population','sum'));rrr=re.groupby(['case','bound','buffer_m'],as_index=False).agg(conditional_restored_lower=('conditional_restored_population_lower','sum'),conditional_restored_upper=('conditional_restored_population_upper','sum'));save(P/'population_effect_summary.csv',metrics);save(P/'population_restoration_summary.csv',rrr)
 wardmetrics=tab.groupby(['case','bound','arm','buffer_m','ward'],as_index=False).agg(residents_2020=('residents_2020','sum'),conditional_loss_lower=('conditional_loss_population_lower','sum'),conditional_loss_upper=('conditional_loss_population_upper','sum'),baseline_unresolved_population=('baseline_unresolved_population','sum'));save(P/'ward_population_effect_summary.csv',wardmetrics)
 # Store official counts with exact polygons and final, rather than preliminary, admission.
 main=tab[(tab.buffer_m==10)&(tab.bound=='lower')];features=[]
 for k,base in df.iterrows():
  props=base.to_dict()
  for case in CASES:
   for arm in ['flood','targeted']:
    row=main[(main.block_id==base.block_id)&(main.case==case)&(main.arm==arm)].iloc[0];props[case+'__'+arm]=row.access_class
  features.append(dict(type='Feature',properties=props,geometry=mapping(polys[k])))
 dump(P/'population_access_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=features))
 # Daytime source semantics and allocation intervals are intentionally not modified.
 save(P/'population_input_readiness.csv',[dict(input='2020officialnightresidentstock',status='integrated',interpretation='Historical at-residence stock;not actualnighttimeoccupancy'),dict(input='125/250mbuildingallocation',status='retained_by_reference',interpretation='Use final admission and original conditional intervals;no rescaling'),dict(input='residencecommutingfields',status='available_but_not_daytimepopulation',interpretation='Not workplace/school destination occupancy'),dict(input='daytime_destination_population',status='not_integrated',interpretation='Need compatible destination jobs/students and presence assumptions'),dict(input='age_and_vulnerability',status='not_integrated',interpretation='Aggregate separately after compatible geography is verified')])
 checks=dict(population_identity=bool((metrics.total_population==496345).all()),unique_state_block=not tab.duplicated(['case','bound','arm','buffer_m','block_id']).any(),intervals_ordered=bool((tab.conditional_loss_population_lower<=tab.conditional_loss_population_upper).all()),unresolved_never_counted_as_loss=bool((tab.loc[tab.baseline_unresolved_population>0,'conditional_loss_population_upper']==0).all()),source_unchanged=all(sha(ROOT/k)==v for k,v in binding.items()),turn_graph_synthetic_chain=selfcheck(),no_new_graph_connections=True,no_vehicle_to_people_conversion=True)
 assert all(checks.values()),checks;dump(R/'validation.json',dict(passed=True,checks=checks,elapsed_s=time.monotonic()-t))
 brief=dict(population_total=496345,blocks=4746,states=8,snapshot_elapsed_s=14400,main_buffer_m=10,main_effects=metrics[metrics.buffer_m==10].to_dict('records'),main_restoration=rrr[rrr.buffer_m==10].to_dict('records'),interpretation=protocol)
 dump(R/'summary.json',brief)
 fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
 colors={'all_candidates_lost':'#b32635','partial_candidate_loss':'#ec9946','normal_support_mixed':'#b1b4b8','no_candidate_support':'#ddd','normal_access_missing':'#ddd','connected_detour_all':'#835bb4','connected_detour_mixed':'#b9a3d5','connected_below_detour_threshold':'#8dc1ad'}
 for ax,case in zip(axes,CASES):
  d=main[(main.case==case)&(main.arm=='flood')].set_index('block_id')
  for k,g in enumerate(polys):
   gs=list(g.geoms)if g.geom_type=='MultiPolygon'else[g]
   for poly in gs:
    if poly.geom_type!='Polygon':continue
    x,y=poly.exterior.xy;ax.fill(x,y,color=colors[d.loc[df.block_id.iloc[k],'access_class']],lw=0)
  ax.set_aspect('equal');ax.set_title(case.split('__')[0]+' | no protection');ax.set_axis_off()
 fig.suptitle('Conditional vehicle access of census blocks | same elapsed 4 h\nUnknown-exposure roads assumed available; no actual stranded-person estimate',fontsize=10);fig.savefig(R/'population_access_map.png',dpi=160);plt.close(fig)
 txt='<h1>三区人口与车行可达性：首轮接入</h1><p>2020年常住人口496,345人，4,746个基本单位区。使用住地人口情景；两场均取积水计算第4小时，包含假设河道抬升，不能解释为当时实际人口或历史受困人数。</p><p>保留原建筑分配区间与最终准入类；不把整片人口放到中心点。使用地面车行候选接入节点，0／10／25m范围对照，以10m为主。未知高程／积水覆盖道路仍按可用假设，结果均有条件。</p><h2>人口影响区间</h2><p>上下端点只约束“候选住地接入位置”：所有正常可达候选均失联才计入下端点；部分失联为0至整片人口。正常支持混合或缺失人口单列。这不是实际受困人数置信区间，也不是完整多方式交通孤岛。</p>'+metrics[metrics.buffer_m==10].to_html(index=False)+'<h2>固定道路保护的恢复区间</h2>'+rrr[rrr.buffer_m==10].to_html(index=False)+'<h2>空间分类</h2><p>红：候选全失联；橙：部分候选失联；紫：参考通行成本至少1.5倍（全／部分候选）；绿：保持连接且低于该阈值；灰：正常接入支持不足。阈值为情景描述，不是交通服务标准。未模拟真实拥堵排队。</p><img src="population_access_map.png"><h2>主分析分类人口账本</h2>'+summary[(summary.buffer_m==10)&(summary.bound=='lower')].to_html(index=False)+'<h2>后续资料</h2><p>白天工作地／学校所在地人口尚未接入。已有通勤字段以居住地统计，不能代替白天目的地人口。未从车辆出行或需求减量推算人数；步行、轨道替代、医院就业目的地需后续分别评估。</p><p><a href="../../../data-preprocessing/three_ward_population_additional_radar_v1/review.html">新增暴雨资料核验</a></p>'
 # Relative link from estimation/TAG to sibling data-preprocessing.
 txt=txt.replace('../../../data-preprocessing/','../../data-preprocessing/')
 style='body{font-family:system-ui,sans-serif;max-width:1150px;margin:30px auto;padding:0 18px;color:#233044;line-height:1.7}img{max-width:100%}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto;max-width:100%}th,td{padding:7px;border:1px solid #ddd}h1{font-size:27px}p{max-width:100%}'
 (R/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>三区人口与车行可达性</title><style>'+style+'</style>'+txt+'</html>')
 print(json.dumps(brief,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
