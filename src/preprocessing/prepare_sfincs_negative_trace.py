"""Stage compact diagnostic interface and bind it to the current frozen CPU source."""
from pathlib import Path
import sys,json,csv,subprocess,shutil,difflib
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump,COMMIT
OUT='data/processed/three_ward_negative_trace_interface_v1'
REPORT='data/exp/data-preprocessing/three_ward_negative_trace_interface'
TAG='udt01b/sfincs-cpu:galibier-negative-trace-v1'
RUNTIME='udt01b/sfincs-cpu:galibier-second-clock-budget-v2'
BUILDER='udt01b/sfincs-cpu:builder-v1'
CODES=[17,51,71,79,106,110,111]

def main():
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen interface')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True);context=out/'context';context.mkdir(exist_ok=True)
 protected={}
 def bind(p):protected[str(p.relative_to(ROOT))]=sha(p)
 sources=ROOT/'src/preprocessing/sfincs_negative_trace_v1'
 for name in ['udt_clock_budget.f90','negative_declarations.inc','negative_routines.inc']:bind(sources/name)
 shutil.copyfile(sources/'udt_clock_budget.f90',context/'udt_clock_budget.f90')
 rain=ROOT/'src/preprocessing/sfincs_second_clock_budget_v2/udt_rain_clock.f90';bind(rain);shutil.copyfile(rain,context/rain.name)
 oldbase=ROOT/'data/processed/three_ward_second_clock_budget_v1'
 lines=[]
 for p in sorted((oldbase/'patched').iterdir()):
  prior=oldbase/'base'/p.name;new=context/p.name if p.name in ['udt_clock_budget.f90','udt_rain_clock.f90']else p
  lines.extend(difflib.unified_diff(prior.read_text().splitlines(True)if prior.exists()else[],new.read_text().splitlines(True),fromfile='a/source/src/'+p.name,tofile='b/source/src/'+p.name))
 patch=out/'negative_trace_full.patch';patch.write_text(''.join(lines))
 base=ROOT/'data/processed/three_ward_cpu_e2e_controls_v1';run=json.loads((base/'run_manifest.json').read_text())['runs'][0];folder=ROOT/run['folder']
 index=np.fromfile(folder/'sfincs.ind',dtype='<i4');n=int(index[0]);ix=index[1:]-1;sr=ix%1012;cc=ix//1012;rr=1011-sr;mask=np.fromfile(folder/'sfincs.msk',dtype='u1')
 static=ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz';graphpath=ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz'
 with np.load(static,allow_pickle=False)as d:features=d['Static Tensor'];valid=d['Static Feature Valid'];names=list(d['Static Channel Names']);native_mask=d['Native Hydraulic Mask']
 assert np.array_equal(mask,native_mask[rr,cc])and len(ix)==n
 with np.load(graphpath,allow_pickle=False)as d:
  group=np.zeros((1012,770),np.int32);group[d['Boundary Point Row'],d['Boundary Point Column']]=d['Boundary Group Codes'][d['Boundary Group Index']]
  groupnames=dict(zip(d['Boundary Group Codes'].tolist(),d['Boundary Group Names'].tolist()))
 channels=[names.index(direction+' '+kind)for kind in ['Open Face Width','Road Face Width','Face Crest']for direction in ['West','East','South','North']]
 meta=features[channels][:,rr,cc].astype('f8');known=valid[channels][:,rr,cc];meta[~known]=np.nan
 mapping=out/'whole_domain_metadata';mapping.mkdir(exist_ok=True)
 columns=['native_cell_index','row_south_one_based','col_east_one_based','kcs','water_group_code']+[d+'_'+k for k in ['open_width_m','road_width_m','face_minimum_m']for d in ['west','east','south','north']]
 with(mapping/'udt_negative_native_map.csv').open('w',newline='')as f:
  w=csv.writer(f);w.writerow(columns)
  for i in range(n):w.writerow([i+1,int(sr[i])+1,int(cc[i])+1,int(mask[i]),int(group[rr[i],cc[i]]),*meta[:,i]])
 (mapping/'udt_negative_trace.request').write_text('schema=negative-cell-face-v1\n')
 # Reconstruct pinned native east-then-north face order for a read-only full topology fixture.
 grid=np.zeros((1012,770),np.int32);grid[sr,cc]=np.arange(1,n+1,dtype=np.int32)
 minus=[];plus=[];direction=[]
 for i,(r,c)in enumerate(zip(sr,cc),1):
  if c+1<770 and grid[r,c+1]>0:minus.append(i);plus.append(int(grid[r,c+1]));direction.append(0)
  if r+1<1012 and grid[r+1,c]>0:minus.append(i);plus.append(int(grid[r+1,c]));direction.append(1)
 minus=np.array(minus,dtype='<i4');plus=np.array(plus,dtype='<i4');direction=np.array(direction,dtype='i1');nf=len(minus)
 pointers=np.full((4,n),nf+1,dtype='<i4')
 for d in [0,1]:
  sel=np.flatnonzero(direction==d);face=sel+1
  pointers[1 if d==0 else 3,minus[sel]-1]=face;pointers[0 if d==0 else 2,plus[sel]-1]=face
 with Dataset(folder/'sfincs_subgrid.nc')as nc:crests=np.array(nc['uv_zmin'][:],dtype='<f4')
 assert len(crests)==nf
 with(mapping/'whole_native_topology.bin').open('wb')as f:
  for a in [np.array([n,nf],dtype='<i4'),(sr+1).astype('<i4'),(cc+1).astype('<i4'),mask.astype('<i4'),*pointers,minus,plus,direction,crests]:a.tofile(f)
 dump(mapping/'topology_fixture.json',{'native_cells':n,'native_faces':nf,'all_water_volumes':'prescribed zero bookkeeping fixture; not a hydraulic state','source':'pinned native index and east-then-north face order, original subgrid minima','solver_calls':0,'topology_sha256':sha(mapping/'whole_native_topology.bin')})

 # Do not put new files in completed controls or alter their identities.
 for p in [folder/'sfincs_subgrid.nc',folder/'sfincs.ind',folder/'sfincs.msk',folder/'sfincs.inp',folder/'sfincs.bzs',static,graphpath,base/'freeze.json',ROOT/'data/processed/three_ward_negative_volume_audit_v1/freeze.json',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json',ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json']:bind(p)
 builder=json.loads(subprocess.run(['docker','image','inspect',BUILDER],capture_output=True,text=True,check=True).stdout)[0]
 runtime=json.loads(subprocess.run(['docker','image','inspect',RUNTIME],capture_output=True,text=True,check=True).stdout)[0]
 expected=json.loads((base/'compatibility_v2/source_binding.json').read_text());active=json.loads((base/'compatibility_v2/active_solver_binding.json').read_text())
 assert builder['Id']==expected['base_builder_image_id']and runtime['Id']==active['new_image_id']
 binding={'source_commit':COMMIT,'base_builder_image_id':builder['Id'],'base_runtime_image_id':runtime['Id'],'full_patch_sha256':sha(patch),'budget_module_sha256':sha(context/'udt_clock_budget.f90'),'rain_module_sha256':sha(context/'udt_rain_clock.f90'),'scope':'opt-in regular projected CPU subgrid; no state equation changes','target_tag':TAG,'hydraulic_runs_authorized_in_this_step':False}
 dump(out/'source_binding.json',binding);dump(context/'NEGATIVE_BINDING.json',binding)
 dockerfile=f'''FROM {BUILDER} AS trace_builder
COPY udt_clock_budget.f90 udt_rain_clock.f90 /usr/src/sfincs/src/
COPY NEGATIVE_BINDING.json /opt/udt-build/NEGATIVE_BINDING.json
COPY negative_trace_fixture.f90 /opt/udt-build/negative_trace_fixture.f90
RUN cd /usr/src/sfincs/src && make -j1 > /opt/udt-build/receipts/negative-make.log 2>&1 && make install > /opt/udt-build/receipts/negative-install.log 2>&1 && ldconfig && cp udt_clock_budget.mod udt_rain_clock.mod /opt/udt-build/modules/ && gfortran -fopenmp -I/opt/udt-build/modules /opt/udt-build/negative_trace_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/negative_trace_fixture
FROM {RUNTIME}
COPY --from=trace_builder /usr/local /usr/local
COPY --from=trace_builder /opt/udt-build /opt/udt-build
RUN ldconfig && /opt/udt-build/netcdf_link_probe && /opt/udt-build/loader_probe
LABEL org.opencontainers.image.version="negative-cell-face-v1"
WORKDIR /data
CMD ["sfincs"]
'''
 (context/'Dockerfile').write_text(dockerfile)
 fixture=sources/'negative_trace_fixture.f90';shutil.copyfile(fixture,context/fixture.name);bind(fixture)
 schema={'schema':'negative-cell-face-v1','threshold_m3':-.001,'comparison':'strict less than; endpoint observation, continuous crossing bracket is [before,after]','files':{'udt_negative_cell_summary.csv':'all interior cells, first/last below endpoints, earliest minimum and step, negative poststep count and crossings; blank unknown times','udt_negative_crossings.csv':'every downward and upward threshold crossing with same-step sources and signed storage','udt_negative_crossing_faces.csv':'four directions per crossing; absent geometric faces explicit; q multiplied by native transverse width once, never by open width twice','udt_negative_cell_group_links.csv':'static direct open boundary groups, multiple memberships explicit; no nearest-water assignment','udt_negative_trace_receipt.txt':'actual enabled profile and native count'},'initial_negative':'recorded in summary at initial endpoint; not a fabricated crossing with face exchange','minimum_ties':'earliest state retained','net_face_arithmetic':'sum individually rounded face terms is diagnostic; grouped native continuity delta stays authoritative, preserve difference','storage':'signed volume never clipped; source-add and storage-update remainders retained','duration':'negative step count only, no exact continuous negative-duration claim','groups':{'0':'No direct open receiver',**{str(k):str(v)for k,v in groupnames.items()}},'group_counting':'one unique cell per linked group, nonexclusive across groups; overall unique count reported separately','output_size':'O(cells) summaries and O(crossings) event faces; no all-face time cube','activation':'request plus validated native map; deployment bindings must require both and verify hashes','old_control_records_backfilled':False,'new_hydraulic_runs':0,'training':False,'batch_58':False}
 dump(out/'output_schema.json',schema);dump(report/'provenance.json',{'protected_sha256':protected,'human_record':'MILI-D-20261015-011','completed_run_inputs_untouched':True});dump(out/'metadata_binding.json',{'native_cells':n,'interior_cells':int((mask==1).sum()),'boundary_cells':int((mask==2).sum()),'water_codes':CODES,'map_sha256':sha(mapping/'udt_negative_native_map.csv'),'request_sha256':sha(mapping/'udt_negative_trace.request'),'source_index_sha256':sha(folder/'sfincs.ind'),'source_mask_sha256':sha(folder/'sfincs.msk'),'no_new_control_staged':True})
 print(json.dumps({'context':str(context),'native_cells':n,'patch':sha(patch)}))
if __name__=='__main__':main()
