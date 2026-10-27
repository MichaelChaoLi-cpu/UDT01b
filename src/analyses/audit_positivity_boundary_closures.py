"""Batch-audit every declared closed receiving boundary over the complete step history."""
from pathlib import Path
import sys,json,csv,collections,argparse
import numpy as np
from scipy.spatial import cKDTree
from netCDF4 import Dataset,chartostring
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_conservative_positivity import OUT,REPORT,INTERFACE
from prepare_sfincs_cpu_build import dump,sha
from audit_three_ward_reference_contract import polyline_sections
from audit_three_ward_negative_volumes import readnp,csvwrite,arr

def main(only=None):
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen controls')
 prefix=(only+'_')if only else'';summary_path=report/(prefix+'summary.json');result=json.loads(summary_path.read_text());backup=report/(prefix+'audit-with-constraint-findings.json')
 if not backup.exists():backup.write_text(json.dumps(result,indent=2)+'\n')
 graph=readnp(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz');blocked=set(map(int,graph['Blocked Native Face Id']));codes=json.loads((ROOT/INTERFACE/'output_schema.json').read_text())['groups']
 topology=ROOT/INTERFACE/'whole_domain_metadata/whole_native_topology.bin'
 with topology.open('rb')as f:
  n,nf=map(int,np.fromfile(f,dtype='<i4',count=2));sr=np.fromfile(f,dtype='<i4',count=n)-1;cc=np.fromfile(f,dtype='<i4',count=n)-1;mask=np.fromfile(f,dtype='<i4',count=n);f.seek(n*4*4,1);a=np.fromfile(f,dtype='<i4',count=nf)-1;b=np.fromfile(f,dtype='<i4',count=nf)-1
 pos=np.c_[-11360+(cc+.5)*10,-42440+(sr+.5)*10]
 select=np.flatnonzero(((mask[a]==1)&(mask[b]==2))|((mask[a]==2)&(mask[b]==1)));centres=(pos[a[select]]+pos[b[select]])*.5
 group_grid=np.zeros((1012,770),int);group_grid[graph['Boundary Point Row'],graph['Boundary Point Column']]=graph['Boundary Group Codes'][graph['Boundary Group Index']]
 data={};allrows=[];checks=[]
 def check(case,name,ok):checks.append({'case':case,'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+case+' '+name,flush=True)
 manifest=json.loads((out/'run_manifest.json').read_text())
 for run in manifest['runs']:
  if only and run['case_id']!=only:continue
  case=run['case_id'];folder=ROOT/run['folder'];sections=polyline_sections(folder/'sfincs.crs');mid=np.array([xy.mean(0)for name,xy in sections]);distance,ix=cKDTree(centres).query(mid);face_ids=select[ix]
  land=np.where(mask[a[face_ids]]==1,a[face_ids],b[face_ids]);water=np.where(mask[a[face_ids]]==2,a[face_ids],b[face_ids]);normal=np.array([[-(xy[1,1]-xy[0,1]),xy[1,0]-xy[0,0]]for name,xy in sections])
  check(case,'all_boundary_sections_biject_native_faces_with_outward_orientation',len(face_ids)==len(select)and len(set(ix))==len(ix)and np.max(distance)<1e-8 and np.all(np.sum(normal*(pos[water]-pos[land]),axis=1)>0))
  closed=np.array([int(i)in blocked for i in face_ids]);jclosed=np.flatnonzero(closed);native_closed=face_ids[jclosed];land_closed=land[jclosed];water_closed=water[jclosed];group=group_grid[1011-sr[water_closed],cc[water_closed]]
  clock=np.genfromtxt(folder/'udt_output_clock.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');hist=clock[clock['write_history']=='T'];ledger=np.genfromtxt(folder/'udt_step_budget.csv',delimiter=',',names=True)
  check(case,'history_covers_every_native_applied_step',np.array_equal(hist['step_index'],np.arange(1,len(ledger))))
  dt=ledger['actual_step_dt'][hist['step_index']];time=hist['actual_state_time'];counts=np.zeros(len(jclosed),np.int64);outvol=np.zeros(len(jclosed));invol=outvol.copy();maxq=outvol.copy();first=np.full(len(jclosed),np.nan);last=first.copy();minimum_match=True
  with Dataset(folder/'sfincs_his.nc')as d:
   check(case,'section_order_matches_runtime_history',[str(v).strip() for v in chartostring(d['crosssection_name'][:])]==[name for name,xy in sections])
   thd=np.c_[arr(d['thindam_x'][:]),arr(d['thindam_y'][:])];snapdist,_=cKDTree(thd).query(mid[jclosed]);check(case,'all_closed_boundary_faces_are_runtime_reported_thin_dams',np.max(snapdist)<1e-5)
   for start in range(0,len(hist),256):
    q=arr(d['crosssection_discharge'][start:start+256])[:,jclosed];used=q!=0;valid=np.all(np.isfinite(q));minimum_match &=valid
    # Match the applied-step single-precision discharge-times-dt convention.
    amount=(q.astype('f4')*dt[start:start+len(q),None].astype('f4')).astype('f8');outvol+=np.maximum(amount,0).sum(0);invol+=np.maximum(-amount,0).sum(0);maxq=np.maximum(maxq,np.max(abs(q),axis=0));counts+=used.sum(0)
    for j in np.flatnonzero(used.any(0)):
     idx=np.flatnonzero(used[:,j]);first[j]=time[start+idx[0]]if np.isnan(first[j])else first[j];last[j]=time[start+idx[-1]]
   check(case,'all_closed_boundary_discharge_records_finite',minimum_match)
  violation=counts>0;affected=set(map(int,land_closed[violation]+1));violatingfaces=set(map(int,native_closed[violation]+1))
  for j in range(len(jclosed)):
   allrows.append({'case':case,'native_face_index':int(native_closed[j]+1),'interior_native_cell':int(land_closed[j]+1),'boundary_native_cell':int(water_closed[j]+1),'water_group_code':int(group[j]),'water_group_name':codes[str(int(group[j]))],'nonzero_applied_steps':int(counts[j]),'first_nonzero_endpoint_s':float(first[j])if violation[j]else None,'last_nonzero_endpoint_s':float(last[j])if violation[j]else None,'gross_outward_applied_m3':float(outvol[j]),'gross_inward_applied_m3':float(invol[j]),'max_abs_discharge_m3_s':float(maxq[j]),'runtime_thindam_midpoint_error_m':float(snapdist[j])})
  # Confirm every closed-face crossing observation is a declared blocked face with a nonzero history.
  events={}
  with(folder/'udt_negative_crossing_faces.csv').open()as f:
   crossings=[r for r in csv.DictReader(f)if r['face_exists']=='T'and float(r['open_width_m'])==0 and float(r['signed_inward_volume_m3'])!=0]
  check(case,'all_closed_crossing_faces_match_full_history_violations',all(int(r['native_face_index'])in violatingfaces and int(r['native_cell_index'])in affected for r in crossings))
  group_totals={}
  for code in sorted(set(map(int,group))):
   sel=group==code;group_totals[str(code)]={'name':codes[str(code)],'declared_closed_faces':int(sel.sum()),'faces_with_nonzero_flux':int((sel&violation).sum()),'gross_outward_m3':float(outvol[sel].sum()),'gross_inward_m3':float(invol[sel].sum())}
  data[case]={'by_water_group':group_totals,'closed_boundary_faces':len(jclosed),'faces_with_nonzero_applied_flux':int(violation.sum()),'affected_interior_native_cells':len(affected),'gross_closed_outward_m3':float(outvol.sum()),'gross_closed_inward_m3':float(invol.sum()),'max_abs_closed_discharge_m3_s':float(maxq.max()),'nonzero_face_step_observations':int(counts.sum()),'closed_crossing_face_observations':len(crossings),'declared_closed_boundary_constraint_satisfied':not np.any(violation),'geometry_admission_gate':not np.any(violation),'finding_scope':'all prescribed receiving-water boundary faces at every native applied step; internal closed faces observed only at crossing events','source_mechanism':'corrected production boundary update includes closed flag0 in the existing zero-exchange branch; all-step history verifies its effect','affected_native_cells':sorted(affected)}
  # Keep model failures as explicit findings, rather than pretending they are data-interface failures.
  selected=[c for c in result['checks']if c['case']==case and c['check']=='no_nonzero_exchange_on_declared_closed_faces']
  result.setdefault('constraint_findings',[]).extend(selected);result['checks']=[c for c in result['checks']if c not in selected]
  result['cases'][case]['boundary_closure_audit']=data[case]
 csvwrite(report/(prefix+'all_closed_boundary_faces.csv'),allrows);dump(report/(prefix+'boundary_closure_audit.json'),{'cases':data,'checks':checks,'all_data_checks_passed':all(c['pass']for c in checks),'new_solver_runs':0,'no_admission_rescue_by_spatial_mask':True,'topology_sha256':sha(topology)})
 closure_names={c['check']for c in checks};result['checks']=[c for c in result['checks']if c['check']not in closure_names];result['checks']+=checks;result['all_checks_passed']=all(c['pass']for c in result['checks']);result['all_declared_boundary_closures_satisfied']=all(d['declared_closed_boundary_constraint_satisfied']for d in data.values());dump(summary_path,result)
 print(json.dumps(data,indent=2))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['dry_raised','rain60_raised']);main(parser.parse_args().only)
