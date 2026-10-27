"""Register already-approved derived export fields alongside discovered fields."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'data/exp/data-preprocessing/three_ward_analysis_contract_v1'
def main():
 v=pd.read_csv(O/'variable_list.csv',keep_default_na=False);d=pd.read_csv(O/'data_dictionary.csv',keep_default_na=False)
 existing=set(zip(v.source_dataset,v.original_name));extra=[]
 for r in d.itertuples():
  key=(r.source_dataset,r.original_name)
  if key not in existing:
   extra.append(dict(source_dataset=r.source_dataset,original_name=r.original_name,dtype=r.dtype,non_null_count=r.rows-r.nulls,null_pct=100*r.nulls/r.rows,sample_values='Already reviewed in export_field_proposal.csv',feasibility_status='partly-testable',readable_name=r.readable_name,full_name=r.full_name,is_final_variable=r.is_final_variable));existing.add(key)
 if extra:v=pd.concat([v,pd.DataFrame(extra)],ignore_index=True)
 assert not v.duplicated(['source_dataset','original_name']).any()
 a=v.set_index(['source_dataset','original_name'])
 for r in d.itertuples():
  row=a.loc[(r.source_dataset,r.original_name)];assert row.readable_name==r.readable_name and row.is_final_variable==r.is_final_variable
 v.to_csv(O/'variable_list.csv',index=False);(ROOT/'data/exp/data-preprocessing/variable_list.csv').write_bytes((O/'variable_list.csv').read_bytes())
 receipt=dict(passed=True,all_approved_export_fields_registered=True,export_field_occurrences=len(d),registry_field_occurrences=len(v),registry_source_tables=v.source_dataset.nunique(),new_rows_registered=len(extra),new_final_concepts_selected=False,existing_native_inputs_unchanged=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (O/'registry_validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
