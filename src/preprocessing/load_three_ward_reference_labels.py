"""Load numerical reference diagnostics without bypassing whole-event admission."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FOLDER='data/processed/three_ward_reference_contract_v1'

def load_reference(case,root=ROOT,*,diagnostic_only=False):
 root=Path(root);folder=root/FOLDER
 registry=json.loads((folder/'reference_registry.json').read_text())
 record=registry['controls'].get(case)
 if record is None:raise KeyError('Only registered open controls are available; no new or sealed responses exist')
 if not diagnostic_only:
  if not record['technical_reference']:raise PermissionError('Whole-event reference admission is on hold: '+','.join(record['hold_reasons']))
  raise PermissionError('Formal reference/training release is closed in this preparation-only package')
 with np.load(folder/record['diagnostic_masks'],allow_pickle=False) as d:return {k:d[k] for k in d.files}
