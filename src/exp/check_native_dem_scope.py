"""Check provisional acquisition scope against available native-source terrain.

20m cell-center sampling is a scope diagnostic, not a final urban flow grid.
Unfilled D8 stops at local pits; filled D8 forces drainage across pits and is
an upper screening case. Neither defines storm-time bidirectional exchange.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('acq',ROOT/'src/preprocessing/acquire_three_ward_dem.py');acq=importlib.util.module_from_spec(spec);spec.loader.exec_module(acq)
def main():
 with rasterio.open(acq.PROC/'native_dem_20m_center_audit.tif') as d:z=d.read(1);tf=d.transform
 with rasterio.open(acq.PROC/'ward_id.tif') as d:ward=d.read(1)
 with rasterio.open(acq.PROC/'channel_stop_mask.tif') as d:sinks=d.read(1)>0
 with rasterio.open(acq.PROC/'acquisition_mask.tif') as d:old=d.read(1)>0
 valid=np.isfinite(z);target=(ward>0)&valid&~sinks
 rec,_=acq.receivers(z,valid,sinks,False);lower=acq.upstream(rec,target)
 rec,filled=acq.receivers(z,valid,sinks,True);upper=acq.upstream(rec,target)
 external_lower=lower&(ward==0);external_upper=upper&(ward==0)
 newly_required=external_upper&~old
 boundary_sensitive=external_upper&binary_dilation(~valid)&~sinks
 unfilled_add=external_lower&~old
 result={'source':'available native0.50m DEM sampled at20m centers; no GSI gap fill','unfilled_external_candidate_km2':float(external_lower.sum())*.0004,'filled_external_candidate_km2':float(external_upper.sum())*.0004,'filled_candidate_beyond_initial_acquisition_km2':float(newly_required.sum())*.0004,'unfilled_candidate_beyond_initial_acquisition_km2':float(unfilled_add.sum())*.0004,'external_candidate_cells_adjacent_to_NoData':int(boundary_sensitive.sum()),'scope_is_final_hydraulic_boundary':False,'limitations':['NoData can truncate contributing areas, especially around imperial-property data gaps.','Priority fill is only a scope bound; no native/model elevation edited.','Channel stops are centerline approximations, not bank/culvert verification.','Available source footprint limits this check; domain expansion decisions must include possible incoming surface flow at those edges.']}
 acq.dump(acq.OUT/'native_scope_check.json',result)
 acq.write_tif('native_unfilled_scope_mask.tif',lower.astype('uint8'),tf,0);acq.write_tif('native_filled_scope_mask.tif',upper.astype('uint8'),tf,0);acq.write_tif('native_scope_extension_candidates.tif',newly_required.astype('uint8'),tf,0)
 if newly_required.any():acq.fc(acq.PROC/'native_scope_extension_candidates.geojson',[(acq.geometry(newly_required,tf),{'role':'available-native-terrain scope extension candidate','not_certified_catchment':True})])
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
