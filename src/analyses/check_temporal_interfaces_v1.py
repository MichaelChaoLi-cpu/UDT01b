"""Finite temporal contract fixtures;no hydraulics or traffic fitting."""
import io,csv
import numpy as np,pandas as pd
from run_temporal_road_population_v1 import envelope,signed_envelope,episodes
from temporal_transport_common_v1 import R,dump

def main():
 info=dict(ptr=np.array([0,2,2]),ix=np.array([0,1]),lens=np.array([2,0]),complete=np.array([True,False]));pop=np.array([100,20]);loss=np.array([[True,False],[False,True]])
 for x in loss:
  lo,up=envelope(x,info,pop);assert int(lo@pop)==0 and int(up@pop)==100
 lo,up=envelope(loss.any(0),info,pop);assert int(lo@pop)==100 and int(up@pop)==100
 lo,up=signed_envelope(np.array([-1,1]),info,pop);assert lo.tolist()==[-100,0]and up.tolist()==[100,0]
 f=io.StringIO();writer=csv.DictWriter(f,fieldnames=['case','bound','arm','road_index','source_edge_id','episode','onset_lower_s','onset_upper_s','recovery_lower_s','recovery_upper_s','left_censored','right_censored','observed_span_min','conditional_span_upper_min','three_ward_clipped_length_m']);writer.writeheader()
 z=episodes('fixture','lower','flood',np.array([[False,True],[True,True],[True,False],[False,True]]),np.array([0,300,600,900]),np.array([0,1]),np.array([10.,20.]),pd.DataFrame({'source_edge_id':['a','b']}),writer)
 rows=list(csv.DictReader(io.StringIO(f.getvalue())));assert len(rows)==3
 assert rows[0]['onset_lower_s']=='0'and rows[0]['onset_upper_s']=='300'and rows[0]['recovery_lower_s']=='600'and rows[0]['recovery_upper_s']=='900'
 assert rows[-1]['right_censored']=='True'and rows[-1]['recovery_upper_s']==''
 assert z['sampled_closed_min'].tolist()==[10.,10.]and z['closed_at_horizon'].tolist()==[False,True]
 dump(R/'temporal_interface_selfcheck.json',dict(passed=True,checks=['alternating_candidates_ever_before_envelope','unresolved_population_held','paired_signed_candidate_difference','onset_recovery_brackets','right_censoring','sampled_trapezoid_integral'],new_hydraulics=False,training=False));print('TEMPORAL INTERFACE FIXTURES PASSED')
if __name__=='__main__':main()
