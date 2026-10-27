#!/usr/bin/env python3
"""Screen all target-road cells against modeled receivers; no solver or pump-count target."""
from pathlib import Path
import json,hashlib,subprocess,html
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_intake_receiver_candidates_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/research-design'/TAG
OLD=ROOT/'data/processed/three_ward_protection_requirements_register_v1'
PLAN=ROOT/'data/processed/three_ward_protection_service_demand_plan_v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 paths=[PLAN/'policy_control_cells_preprocessed.parquet',PLAN/'road_policy_membership_preprocessed.parquet',OLD/'road_control_cell_group_links_preprocessed.parquet',ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz',ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz',ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz',ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv',ROOT/'docs/AnaSOP.md',ROOT/'data/processed/three_ward_ground_flux_replay_v1/zone_ground_budget_trajectories_preprocessed.parquet']
 ctl=pd.read_parquet(paths[0]);groups=pd.read_parquet(paths[1]);cg=pd.read_parquet(paths[2]);inv=pd.read_csv(paths[6]);inv=inv[inv.case_id.str.contains('20240731')];assert len(inv)==2
 cells=ctl[ctl.Policy=='Combined'].sort_values('Grid Index');grid=cells['Grid Index'].to_numpy();target=cells['Target Storage'].to_numpy();assert len(grid)==13214 and np.isfinite(target).all()
 with np.load(paths[5]) as z:native_grid=z['native_grid_north_index'];kcs=z['native_kcs'];xy=z['native_xy']
 grid_native=np.full(1012*770,-1,int);grid_native[native_grid]=np.arange(len(native_grid));src=grid_native[grid];assert (src>=0).all() and (kcs[src]==1).all()
 with np.load(paths[4]) as z:
  names=z['Static Channel Names'].tolist();tensor=z['Static Tensor'].reshape(28,-1);valid=z['Static Feature Valid'].reshape(28,-1);ward=z['Ward Code'].ravel();features={n:tensor[names.index(n),grid] for n in ['Road Ground Fraction','Structure Uncertainty Fraction','Building Occupancy Fraction']}
  known=np.logical_and.reduce([valid[names.index(n),grid] for n in features]);struct=features['Structure Uncertainty Fraction']>0;road=features['Road Ground Fraction']>0
  eligible=known&~struct&road
 sources=pd.DataFrame({'Source Candidate':['S'+str(x).zfill(7) for x in grid],'Grid Index':grid,'Native Cell Zero Based':src,'Source Easting':xy[src,0],'Source Northing':xy[src,1],'Ward Code':ward[grid],'Road Ground Fraction':features['Road Ground Fraction'],'Structure Uncertainty Fraction':features['Structure Uncertainty Fraction'],'Building Occupancy Fraction':features['Building Occupancy Fraction'],'Ground Source Screen Passed':eligible,'Structural Unknown':struct,'Mapped Ground Road Missing':~road,'Field Access Verified':False,'Deployment Feasible':None})
 masks={}
 for name in ['Backbone','Area Access','Combined']:
  masks[name]=np.isin(grid,ctl.loc[ctl.Policy==name,'Grid Index']);sources[name+' Candidate']=masks[name]
  assert masks[name].sum()==len(ctl[ctl.Policy==name])
 sourcegroups=cg[cg['Grid Index'].isin(grid)].merge(groups[['Road Group','Backbone Candidate','Area Access Candidate','Combined Candidate']],on='Road Group',validate='many_to_one');sourcegroups=sourcegroups[sourcegroups['Combined Candidate']].copy()
 with np.load(paths[3]) as z:
  kept_boundary=np.unique(z['Receiver Link Index'][1]);bp=z['Boundary Point Position'][kept_boundary];codes=z['Boundary Group Codes'];bn=z['Boundary Group Names'];bi=z['Boundary Group Index'][kept_boundary];low=z['Boundary Low Stage'][kept_boundary];water_grid=(z['Boundary Point Row']*770+z['Boundary Point Column'])[kept_boundary]
 water_native=grid_native[water_grid];assert (water_native>=0).all() and (kcs[water_native]==2).all() and np.max(abs(xy[water_native]-bp))<1e-6
 assert len(np.unique(water_native))==len(water_native)
 receivers=pd.DataFrame({'Receiver Candidate':['W'+str(x).zfill(7) for x in water_native],'Native Cell Zero Based':water_native,'Grid Index':water_grid,'Receiver Easting':bp[:,0],'Receiver Northing':bp[:,1],'Receiver Group':codes[bi],'Receiver Name':bn[bi],'Boundary Low Stage':low,'Stage Evidence':'existing prescribed model scenario;not observed river measurement','Receiving Permission Verified':False,'Receiving Capacity Known':False,'Shared Receiving Capacity':None})
 # Keep nearest modeled receiving point in EVERY group;no global nearest-only exclusion.
 source_ix=[];rec_ix=[];distances=[]
 for g in range(len(codes)):
  rr=np.flatnonzero(bi==g);dist,ix=cKDTree(bp[rr]).query(xy[src]);source_ix.append(np.arange(len(src)));rec_ix.append(rr[ix]);distances.append(dist)
 si=np.concatenate(source_ix);ri=np.concatenate(rec_ix);dist=np.concatenate(distances)
 link=pd.DataFrame({'Link Candidate':[f'{sources["Source Candidate"].iloc[s]}-R{codes[bi[r]]}' for s,r in zip(si,ri)],'Source Candidate':sources['Source Candidate'].to_numpy()[si],'Receiver Candidate':receivers['Receiver Candidate'].to_numpy()[ri],'Receiver Group':codes[bi[ri]],'Receiver Name':bn[bi[ri]],'Straight Distance Lower Bound':dist,'Transfer Route Verified':False,'Receiving Permission Verified':False,'Deployment Feasible':None,'Selection Meaning':'closest modeled point per group;farther bank or engineered receiving points not exhaustively searched'})
 assert len(link)==len(src)*len(codes) and not link['Link Candidate'].duplicated().any()
 np.savez_compressed(P/'candidate_index.npz',source_grid=grid,source_native=src,receiver_grid=water_grid,receiver_native=water_native,original_boundary_index=kept_boundary,link_source=si,link_receiver=ri,link_distance=dist,target_storage=target,source_eligible=eligible)
 for row in inv.itertuples():paths.extend([ROOT/row.map_path,(ROOT/row.map_path).parent/'sfincs_subgrid.nc',OLD/'zones'/(row.case_id+'.npz'),(ROOT/row.map_path).parent/'sfincs.bnd',(ROOT/row.map_path).parent/'sfincs.bzs',(ROOT/row.map_path).parent/'sfincs.inp',(ROOT/row.map_path).parent/'udt_step_budget.csv'])
 binding={str(p.relative_to(ROOT)):sha(p) for p in paths}
 # Export only original saved states;not hydraulic integration.
 native=r'''
import sys,csv
import pandas as pd
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
root=Path(sys.argv[1]);out=Path(sys.argv[2])
with np.load(out/'candidate_index.npz') as z:sg=z['source_grid'];sn=z['source_native'];rg=z['receiver_grid'];rn=z['receiver_native'];original_boundary_index=z['original_boundary_index']
with (root/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
for row in rows:
 case=row['case_id']
 if '20240731' not in case:continue
 p=root/row['map_path'];source=[];receiver=[];volume=[]
 with Dataset(p) as d:
  for frame in range(109):
   eta=np.flipud(np.ma.asarray(d['zs'][frame],dtype='float64').filled(np.nan)).ravel();v=np.flipud(np.ma.asarray(d['subgrid_volume'][frame],dtype='float64').filled(np.nan)).ravel()
   source.append(eta[sg]);receiver.append(eta[rg]);volume.append(v[sg])
 with Dataset(p.parent/'sfincs_subgrid.nc') as d:
  zmin=np.asarray(d['z_zmin'][:],dtype=float)[sn];rmin=np.asarray(d['z_zmin'][:],dtype=float)[rn];vmax=np.asarray(d['z_volmax'][:],dtype=float)[sn];zmax=np.asarray(d['z_zmax'][:],dtype=float)[sn];levels=np.asarray(d['z_level'][:],dtype='float32')[sn]
 def eta_of_volume(v):
  result=np.maximum(zmin,-20.).copy();full=v>=vmax*.999;result[full]=np.maximum(zmax[full],-20.)+(v[full]-vmax[full])/100.
  loc=np.flatnonzero(~full&(v>1e-6));frac=v[loc]/(vmax[loc]/252.);j=np.minimum(np.floor(frac).astype(int),251);frac-=j;result[loc]=levels[loc,j]+frac*(levels[loc,j+1]-levels[loc,j]);return result
 with np.load(root/'data/processed/three_ward_protection_requirements_register_v1/zones'/f'{case}.npz') as z:times=z['actual_times'];analysis=z['analysis_times']
 s=np.array(source);v=np.array(volume);reconstructed=np.array([eta_of_volume(vv) for vv in v]);finite=np.isfinite(s);stage_error=float(np.max(abs(s[finite]-reconstructed[finite]),initial=0.));assert stage_error<1e-4
 missing_count=int((~finite).sum());s[~finite]=reconstructed[~finite]
 with np.load(out/'candidate_index.npz') as z:receiver_xy=np.c_[-11360+(z['receiver_grid']%770+.5)*10,-42440+(1011-z['receiver_grid']//770+.5)*10]
 bnd=np.loadtxt(p.parent/'sfincs.bnd');assert np.max(abs(bnd[original_boundary_index]-receiver_xy))<1e-6
 bzs=np.loadtxt(p.parent/'sfincs.bzs');assert bzs.shape[1]==len(bnd)+1 and np.all(np.diff(bzs[:,0])>0)
 forcing=bzs[:,1:].astype('float32')[:,original_boundary_index];r=np.array([np.interp(times,bzs[:,0],forcing[:,i]) for i in range(len(rn))]).T;r=np.maximum(r,rmin[None,:])
 maxdt=float(pd.read_csv(p.parent/'udt_step_budget.csv',usecols=['actual_step_dt'])['actual_step_dt'].max());maxslope=float(np.max(abs(np.diff(forcing.astype(float),axis=0)/np.diff(bzs[:,0])[:,None]),initial=0.));alignment_bound=maxslope*maxdt+4*np.finfo('float32').eps*max(1.,float(abs(r).max()))+stage_error
 inp=(p.parent/'sfincs.inp').read_text();assert 'tspinup                  = 0.0' in inp and 'pavbnd                   = 0' in inp and 'bca' not in inp and 'bzi' not in inp
 assert np.isfinite(s).all() and np.isfinite(r).all() and np.isfinite(v).all() and (v>=0).all()
 np.savez_compressed(out/(case+'_saved_states.npz'),source_stage=s,receiver_stage=r,source_volume=v,source_minimum=zmin,actual_time=times,analysis_time=analysis,receiver_alignment_bound=alignment_bound,source_reconstruction_error=stage_error,source_missing_reconstructed=missing_count)
 print('EXISTING_STATES_READ',case,flush=True)
'''
 subprocess.run([str(ROOT/'.frame-control/runtimes/physical-measures-py312/bin/python'),'-c',native,str(ROOT),str(P)],check=True)
 envelopes=[('Anchor',.5,5.,500.),('Rate Half',.25,5.,500.),('Rate Double',1.,5.,500.),('Lift Half',.5,2.5,500.),('Lift Double',.5,10.,500.),('Reach Half',.5,5.,250.),('Reach Double',.5,5.,1000.)]
 env=pd.DataFrame([{'Capability Scenario':n,'Standalone Rate Ceiling':q,'Static Lift Cutoff':h,'Straight Distance Screen':l,'Depth Trigger Above Native Minimum':.05,'Parameter Evidence':'hypothetical;anchor from legacy pilot,half/double brackets are agent-proposed sensitivity cases','Pump Curve Available':False} for n,q,h,l in envelopes])
 receiver_scenarios=pd.DataFrame([{'Receiving Scenario':'Open Conditional','Shared Group Rate Allowance':np.nan,'Meaning':'assume unconstrained prescribed-stage disposal;not evidence of river capacity'},{'Receiving Scenario':'Finite 0.5','Shared Group Rate Allowance':.5,'Meaning':'hypothetical total per model receiver group,shared by all active sites'},{'Receiving Scenario':'Finite 2.0','Shared Group Rate Allowance':2.,'Meaning':'hypothetical total per model receiver group,shared by all active sites'},{'Receiving Scenario':'Blocked','Shared Group Rate Allowance':0.,'Meaning':'no permitted disposal to this receiver group'}])
 linkrows=[];frame_rows=[];group_rows=[];case_stats=[]
 for row in inv.itertuples():
  case=row.case_id
  with np.load(P/(case+'_saved_states.npz')) as z:s=z['source_stage'];rec=z['receiver_stage'];vol=z['source_volume'];minimum=z['source_minimum'];actual=z['actual_time'];analysis=z['analysis_time'];alignment_bound=float(z['receiver_alignment_bound']);source_stage_error=float(z['source_reconstruction_error']);source_missing=int(z['source_missing_reconstructed'])
  excess=np.maximum(vol-target[None,:],0.);depth=np.maximum(s-minimum[None,:],0.);lift=np.maximum(rec[:,ri]-s[:,si],0.);wet=depth[:,si]>=.05
  a=link[['Link Candidate','Receiver Group']].copy();a.insert(0,'Case',case);a['Minimum Saved Static Lift']=lift.min(axis=0);a['Maximum Saved Static Lift']=lift.max(axis=0);a['Saved Wet Trigger Fraction']=wet.mean(axis=0);a['Ground Source Screen Passed']=eligible[si];a['Receiver Stage Alignment Bound']=alignment_bound;a['Receiver Stage Evidence']='prescribed forcing at actual map time,bed-clamped;not saved boundary measurement';a['Head Losses Included']=False;a['Field Deployment Confirmed']=False;linkrows.append(a)
  feasible_counts={}
  for name,q,head,reach in envelopes:
   passes=(lift+alignment_bound<=head)&wet&(dist[None,:]<=reach)&eligible[si][None,:]
   reached=passes.reshape(109,len(codes),len(src)).any(axis=1);named=np.array(['unnamed' not in str(n) for n in bn]);named_reached=passes.reshape(109,len(codes),len(src))[:,named,:].any(axis=1);feasible_counts[name]=reached.sum(axis=1)
   for policy,mask in masks.items():
    for frame in range(109):
     core=mask&(ward[grid]>0);buffer=mask&~(ward[grid]>0);e=excess[frame];covered=core&reached[frame];uncovered=core&~reached[frame]
     frame_rows.append({'Case':case,'Policy':policy,'Capability Scenario':name,'Frame':frame,'Actual Time':float(actual[frame]),'Analysis Time':float(analysis[frame]),'Three Ward Road Excess':float(e[core].sum()),'Excess At Sources With Screened Link':float(e[covered].sum()),'Excess At Sources Without Screened Link':float(e[uncovered].sum()),'Excess With Named Receiver Link':float(e[core&named_reached[frame]].sum()),'Excess With Only Unnamed Receiver Link':float(e[covered&~named_reached[frame]].sum()),'Buffer Road Excess':float(e[buffer].sum()),'Possible Source Cells At Saved Time':int((mask&reached[frame]).sum()),'Strict Unmet Drainage Requirement':None,'Meaning':'baseline source-cell coverage proxy;not water removed,capacity sufficiency or guaranteed road service'})
   for group,code in enumerate(codes):
    pp=passes[:,group*len(src):(group+1)*len(src)];count=pp.sum(axis=1)
    for rr in receiver_scenarios.to_dict('records'):
     allowance=rr['Shared Group Rate Allowance'];group_rows.append({'Case':case,'Capability Scenario':name,'Receiver Group':int(code),'Receiver Name':str(bn[group]),'Receiving Scenario':rr['Receiving Scenario'],'Shared Group Rate Allowance':allowance,'Possible Source Cells Ever Passing':int(pp.any(axis=0).sum()),'Maximum Simultaneous Possible Source Cells':int(count.max()),'Standalone Rate Ceiling':q,'Actual Allocated Capacity':None,'Selected Deployment Locations':None,'Meaning':'possible alternatives,not deployed pump count;finite allowance must constrain aggregate allocation across sites and model groups sharing physical reaches'})
  assert np.array_equal(feasible_counts['Anchor'],feasible_counts['Rate Half']) and np.array_equal(feasible_counts['Anchor'],feasible_counts['Rate Double'])
  for lo,hi in [('Lift Half','Anchor'),('Anchor','Lift Double'),('Reach Half','Anchor'),('Anchor','Reach Double')]:assert (feasible_counts[lo]<=feasible_counts[hi]).all()
  case_stats.append({'case':case,'saved_frames':109,'source_missing_stages_reconstructed':source_missing,'maximum_source_curve_difference':source_stage_error,'conservative_receiver_stage_alignment_bound':alignment_bound,'baseline_source_volume_peak':float(vol.sum(axis=1).max()),'source_control_excess_peak':float(excess[:,ward[grid]>0].sum(axis=1).max())})
 frames=pd.DataFrame(frame_rows);rate_cols=['Case','Policy','Frame','Excess At Sources With Screened Link','Excess At Sources Without Screened Link','Possible Source Cells At Saved Time']
 anchor=frames[frames['Capability Scenario']=='Anchor'];summary=[]
 for (case,policy,scenario),a in frames.groupby(['Case','Policy','Capability Scenario']):
  z=a.loc[a['Three Ward Road Excess'].idxmax()];total=float(z['Three Ward Road Excess']);summary.append({'Case':case,'Policy':policy,'Capability Scenario':scenario,'Road Excess Peak Analysis Time':float(z['Analysis Time']),'Three Ward Road Excess At Peak':total,'Excess With Screened Link At Same Peak':float(z['Excess At Sources With Screened Link']),'Excess Without Screened Link At Same Peak':float(z['Excess At Sources Without Screened Link']),'Screened Link Share Of Excess At Peak':float(z['Excess At Sources With Screened Link']/total) if total else None,'Maximum Excess Without Screened Link':float(a['Excess At Sources Without Screened Link'].max()),'Excess With Named Receiver Link At Peak':float(z['Excess With Named Receiver Link']),'Excess With Only Unnamed Receiver Link At Peak':float(z['Excess With Only Unnamed Receiver Link']),'Derived Active Intakes':None,'Unmet Drainage Requirement':None})
 summ=pd.DataFrame(summary);assert np.max(abs(frames['Three Ward Road Excess']-frames['Excess At Sources With Screened Link']-frames['Excess At Sources Without Screened Link']))<1e-6
 assert np.max(abs(frames['Excess At Sources With Screened Link']-frames['Excess With Named Receiver Link']-frames['Excess With Only Unnamed Receiver Link']))<1e-6
 # No transfer amount is predicted from snapshots or repeated storage availability.
 tables={'source_candidates':sources,'source_policy_group_links':sourcegroups,'receiver_candidates':receivers,'intake_receiver_links':link,'link_stage_screen':pd.concat(linkrows,ignore_index=True),'capability_scenarios':env,'receiving_scenarios':receiver_scenarios,'source_coverage_trajectories':frames,'source_coverage_summary':summ,'receiver_allocation_readiness':pd.DataFrame(group_rows)}
 exports=[];dictionary=[]
 for name,a in tables.items():
  a=a.reset_index(drop=True);assert all(c.isascii() and c==c.strip() for c in a.columns);f=P/(name+'_preprocessed.parquet');a.to_parquet(f,index=False);pd.testing.assert_frame_equal(a,pd.read_parquet(f));exports.append({'path':str(f.relative_to(ROOT)),'rows':len(a),'columns':len(a.columns),'sha256':sha(f)})
  for c in a:
   unit='identifier,category,count or boolean'
   if any(x in c for x in ['Easting','Northing','Lift','Distance','Minimum','Stage']):unit='m in existing projected/model datum'
   if any(x in c for x in ['Excess','Volume','Drainage Requirement']):unit='m3;screening only or unresolved'
   if any(x in c for x in ['Fraction','Share']):unit='fraction'
   if 'Rate' in c or 'Capacity' in c:unit='m3/s;hypothetical ceiling,shared limit or unresolved'
   if 'Time' in c:unit='s since scenario start' if c in ['Actual Time','Analysis Time','Road Excess Peak Analysis Time'] else 'count/category'
   dictionary.append({'Dataset':f.name,'Readable Field':c,'Unit':unit,'Final Variable':'no','Role':'reference-only candidate/assumption/screen;not engineering verification'})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 protocol={'status':'candidate_register_complete;deployment_not_sized','scope':'all 13214 combined-policy road-control cells;nearest modeled receiving point within each of seven groups;not fixed number of pumps','physics_runs':0,'training':0,'people_counts':False,'parameters':'hypothetical anchor0.5m3/s,5mstaticlift,500mstraight-distance,0.05mtrigger copied only as scenario reference;half/double one-factor alternatives','receiving':'open conditional,finite0.5or2m3/s per modeled group,blocked;all assumptions,not measured capacity','limits':['straight line is a lower bound,not a verified hose route','static lift omits friction,bank crossing and equipment curve;baseline source stage falls after pumping','all nodes are candidates;one native cell is not one equipment unit or final installation','same physical river can span multiple model groups;shared physical-reach budget must supersede group caps before a deployment run','wet control-zone connectivity is not a contributing catchment','source-cell excess coverage does not prove drainage volume or protection service coverage','uncovered source-cell excess is a gap marker,not unavoidable unmet drainage;another intake may drain it through modeled paths','candidate banks are model boundary points,not officially approved outfalls','only July H/L examined dynamically;other events require their own head and exchange checks','all109 snapshots checked;sub-five-minute operating duty and instantaneous peaks unknown'],'unresolved':['field access and footprint','transfer alignment and crossing feasibility','permission and shared physical-reach acceptance','effective equipment curve and losses','target-stage and path response under extraction','finite receiver cap and fair allocation implementation qualification'],'next_step':'choose conditional source-receiver portfolios using these assumptions and linked control zones;derive active positions by allocation,then declare run matrix,budget and unchanged admission checks before simulation'}
 dump(R/'protocol.json',protocol);dump(R/'decisions.json',{'basis':'requirements-led deployment decision retained;latest user authorizes next candidate-construction step','agent_proposed_sensitivity_parameters':env.to_dict('records'),'reference_only':True,'human_final_deployment_freeze':False,'counts_not_prespecified':True})
 assert all(sha(ROOT/n)==h for n,h in binding.items());dump(P/'manifest.json',{'input_sha256':binding,'outputs':exports,'script_sha256':sha(Path(__file__)),'source_count':len(sources),'receiver_stage_method':'original bzs forcing with exact bnd coordinate matching,subgrid minimum clamp and conservative one-step alignment bound;source dry maps restored by original storage curve','link_count':len(link)})
 checks={'passed':True,'source_cells':len(sources),'ground_source_screen_passed':int(eligible.sum()),'structural_unknown_sources':int(struct.sum()),'mapped_ground_road_missing':int((~road).sum()),'receiver_points':len(receivers),'receiver_groups':len(codes),'candidate_links':len(link),'cases':case_stats,'source_native_and_receiver_native_masks_checked':True,'source_ground_and_boundary_positions_exact':True,'single_link_per_source_per_group':True,'snapshot_excess_partition_closes':True,'one_factor_nesting_checked':True,'rate_only_does_not_artificially_change_geometric_coverage':True,'all_parquet_roundtrips_exact':True,'source_hashes_and_AnaSOP_unchanged':True,'no_capacity_sufficiency_or_unmet_volume_inferred':True,'new_hydraulic_runs':0};dump(R/'validation.json',checks)
 display=summ[(summ.Policy=='Combined')&summ['Capability Scenario'].isin(['Anchor','Lift Double','Reach Double'])].copy()
 display['Screened Link Share Of Excess At Peak']*=100
 cols=['Case','Capability Scenario','Three Ward Road Excess At Peak','Excess With Screened Link At Same Peak','Excess Without Screened Link At Same Peak','Screened Link Share Of Excess At Peak']
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>取水—接收候选与能力情景</title><style>body{{font:16px/1.8 system-ui;max-width:1320px;margin:30px auto;padding:0 24px;color:#243d48}}aside{{background:#eaf4f5;border-left:4px solid #427d90;padding:18px}}.warn{{background:#fff0d7;padding:16px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #d4e0e4;padding:8px;text-align:left}}</style><h1>取水位置—接收水体—有效能力：候选准备</h1><aside>在三区联合候选道路的全部{len(sources):,}个控制单元上筛选，没有先指定泵点数。每个源点保留七个模型水域组各自最近接收点，共{len(link):,}条连接候选。<b>本轮读取既有结果，没有新模拟，也没有把候选当成部署。</b></aside><h2>1. 哪些已有证据，哪些仍是假设</h2><p>源点均核对为原生地面单元，接收点均核对为模型规定水位单元；投影坐标沿用EPSG:6677，水位沿用同一模型高程基准。7月31日两组均读取109个实际时刻的源区储量与有效水位。地图未保存的干单元水位由同一储量曲线恢复，已有有效水位用于交叉核验；边界水位由原始bzs文件按实际地图时刻重建，并按原生最低底高程截断。扬程筛查额外扣除最大实际步长对应的边界变幅和浮点误差界。模型内部基准一致不等于外部现场高程已复测。</p><p>地面来源筛查通过{int(eligible.sum()):,}处；结构不确定{int(struct.sum()):,}处，无映射地面道路{int((~road).sum()):,}处，后两类可能重叠。它们保留在表中，不按可部署处理。即使来源筛查通过，现场停车／作业空间、设备进场、管线穿路和许可仍全部未核验。</p><p>取水候选仅覆盖目标道路控制单元；水区内远处更合适的集水井或设备作业面尚未穷尽。每水域组最近点是筛查代理，不能排除更远但岸线条件更好的接收位置。</p><h2>2. 能力和接收条件怎样比较</h2><p>Anchor只沿用旧试验的0.5m³/s、5m静扬程门槛、500m直线距离、0.05m启动水深作为假设参考。每次只把单点流量、扬程门槛或距离门槛减半／加倍；它们不是设备实测能力，也不是东京官方行动标准。当前模型算子在扬程门槛内按恒定流量上限抽取、超门槛停止，尚无真实泵曲线和摩阻损失。</p><div class="scroll">{env.to_html(index=False,float_format=lambda v:f'{v:g}')}</div><p>接收情景分别为：条件开放；每模型水域组共享0.5或2m³/s的假设额度；受阻时0。有限额度约束所有激活位置之和，不能逐点重复享有额度。神田川、古川等实际河段若跨多个模型组，也必须共享物理河段总额度；不能靠拆分组号增加排水能力。这些额度尚未作为新措施算子运行。</p><h2>3. 能接到哪里：基线储量覆盖筛查</h2><p>下表在各情景联合候选的同一储量峰值时刻统计。通过条件为地面来源可用、直线距离在门槛内、静水位差满足扬程门槛且源点超过启动水深。表中m³为道路单元原有超阈值储量，最后一列为百分比；尚未施加现场路线许可或有限接收额度。</p><div class="scroll">{display[cols].to_html(index=False,float_format=lambda v:f'{v:,.2f}')}</div><div class="warn"><b>“有筛选连接”不等于能抽走这些水，也不等于道路获保护；“无筛选连接”也不等于必然无法排水。</b>相邻位置可能通过地形连通共同排水，抽排后源水位、来水和河道交换也会变化。因此实际激活取水点数、实际分配能力和严格未满足需求继续留空。没有把不同帧储量重复相加来推算抽排量。</div><p>仅改变单点流量的两组，几何覆盖率按预期不变；流量能否缩短封路、减少拥堵，需要下一轮实际分配和措施模拟。扬程与距离放宽只表示条件覆盖变化，并非已经证明措施收益。接收组51仍未命名，数据中已把仅有未命名接收连接的储量单列；模型名称也不代表官方允许排放。</p><h2>4. 下一步怎样真正形成部署</h2><ol><li>以目标道路及其关联水区为单位，利用候选表选择源—接收组合；不以候选单元数作为安装台数。</li><li>对开放、有限共享、受阻三类接收条件分别分配有效流量；保留无法支持的道路目标和未知路线。</li><li>激活位置数由分配结果得出；同一点多个设备的台数另行定义。挡水方案保留车辆开口，并在挡排结合情景重新核查入水和滞水。</li><li>明确小批次情景与计算预算后，才能运行新措施对照并检查原数值准入、水量守恒、三区封路、交通及风险转移。</li></ol><p>未计算新的受影响人口、未训练、未访问封存测试；AnaSOP和已认可图件保持不变。</p><h2>数据与复现</h2><ul>'''
 for e in exports:page+=f'<li><a href="../../../processed/{TAG}/{Path(e["path"]).name}">{html.escape(Path(e["path"]).name)}</a>（{e["rows"]:,}行）</li>'
 page+='</ul><p><a href="validation.json">核验</a> · <a href="protocol.json">条件与边界</a> · <a href="data_dictionary.csv">字段单位</a> · <a href="README.md">复现</a> · <a href="../three_ward_protection_service_demand_plan_v1/review.html">保护目标和需求筛查</a></p></html>'
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Intake-receiver candidate screen\n\nRun `.venv/bin/python src/analyses/build_intake_receiver_candidates_v1.py` from the project root. Uses the existing native netCDF runtime to read saved maps only;no solver or new physics. Source cells:all13214 Combined policy control cells. Candidate water points:all5327 open-ground-adjacent prescribed-stage cells in seven model groups. Retain nearest point per group,without imposing a pump count. A source is screened from known static features with mapped ground road fraction>0 and zero structure uncertainty;field access remains unknown.\n\nAnchor is a hypothetical legacy reference. One-factor half/double rate,lift and distance cases are proposals. Head is positive static receiving minus source stage at saved frames,no friction or performance curve. Map dry source stages are reconstructed from the exact source storage operator and checked against all finite saved stages. Prescribed receiver stages are read from original bzs forcing with exact bnd coordinate matching and native subgrid minimum clamp,using an added conservative one-step timing and float uncertainty margin. No receiver map pixels are filled with zero. Receiver acceptance is open-conditional,finite hypothetical aggregate0.5or2m3/s per model group,or blocked. No allocation is performed. Model groups sharing real receiving reaches must not obtain independent extra physical allowance. All actual capacities,selected deployment locations and strict unmet drainage remain null.\n\nRoad-cell baseline excess is partitioned by possible screened links per saved frame. This is not predicted transfer,drainage sufficiency,guaranteed service,or unavoidable unmet demand. Other intakes may serve uncovered cells. Do not sum policies or repeated snapshots. JulyH/L alone have completed ground-flux evidence. Other events must not inherit these stages or links' dynamic qualification. No new official/engineering claim,no raw-data,AnaSOP,accepted-figure,Git orDVC modifications.\n''')
 print(json.dumps({'status':'complete','source_candidates':len(sources),'source_ground_screen_passed':int(eligible.sum()),'links':len(link),'summary':display[cols].to_dict('records')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
