"""Portable NumPy-only loading helpers; no normalization fitting or ML execution."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def arrays(path):
    with np.load(path,allow_pickle=False) as d:
        return {k:d[k] for k in d.files}

def load_static(root=ROOT):
    return arrays(Path(root)/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz')

def load_graph(root=ROOT):
    return arrays(Path(root)/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz')

def load_control(case,root=ROOT):
    """Return prescribed predictors separately from response-derived supervision.

    Existing controls are development evidence; this function never assigns a split.
    Static/graph arrays may be loaded once and reused across controls.
    """
    folder=Path(root)/'data/processed/three_ward_surrogate_inputs'
    schema=json.loads((folder/'input_schema.json').read_text())
    if case not in schema['cases']:
        raise ValueError('Only registered existing controls can be loaded by this package')
    prescribed=arrays(folder/(case+'_inputs.npz'))
    supervision=arrays(folder/(case+'_label_admission.npz'))
    return prescribed,supervision

def forcing_rasters(prescribed,graph,shape=(1012,770)):
    """Lossless raster views for uniform physical rain and typed boundary stages.

    Stage zero outside Stage Known is a numeric placeholder. Never interpolate a
    receiving level across the ground or use this view as a response water level.
    Physical uniform rain is broadcast, not confused with roof-redistributed rain.
    """
    rain=prescribed['Physical Rainfall Rate']
    n=len(prescribed['Event Time'])
    if rain.shape!=(n,):raise ValueError('Physical rainfall must be a uniform time vector')
    physical=np.broadcast_to(rain[:,None,None],(n,*shape))
    stage=np.zeros((n,*shape),np.float32)
    known=np.zeros(shape,bool)
    r=graph['Boundary Point Row'];c=graph['Boundary Point Column']
    stage[:,r,c]=prescribed['River Stage'];known[r,c]=True
    return {'Physical Rainfall Raster':physical,'River Stage Raster':stage,'Stage Known':known}
