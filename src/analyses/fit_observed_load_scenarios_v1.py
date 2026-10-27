"""Minimum-change observation-constrained scenarios, not true vehicle OD recovery."""
from observed_load_common_v1 import *
from scipy.sparse import load_npz
from scipy.optimize import linprog,minimize
import time,datetime as dt

def main():
 started=time.monotonic();basis=json.loads((R/'basis_validation.json').read_text());assert basis['passed'];order=load(P/'basis_order.npz')['source_expanded_od_index'];dem=load(REP/'multi_routing_demands.npz');base=dem['base'][order];prior=dem['period_weights'][order];A=load_npz(P/'screenline_incidence.npz');o=pd.read_csv(P/'observation_route_support.csv',dtype={'observation_section_id':str});y=o.observed_small_daytime12h.to_numpy();raw=np.asarray(A@(prior[:,4]*base)).ravel();supported=raw>1e-10;ii=np.flatnonzero(base);patterns,inverse=np.unique(A[:,ii].toarray().T,axis=0,return_inverse=True);q0=np.bincount(inverse,weights=prior[ii,4],minlength=len(patterns));Q=float(q0.sum());V=patterns[:,supported].T.astype(float)*Q/y[supported,None];p0=q0/Q;L=np.full(supported.sum(),.8);U=np.full(supported.sum(),1.2);feas=linprog(np.zeros(len(p0)),A_ub=np.vstack([V,-V]),b_ub=np.r_[U,-L],bounds=(0,None),method='highs',options={'time_limit':30});trace=[];mode='minimum_relative_entropy'
 def budget():
  if time.monotonic()-started>60:raise RuntimeError('Registered inverse budget reached')
 if feas.success:
  n=len(L)
  def objective(lam):
   budget();eta=-V.T@(lam[:n]-lam[n:]);p=p0*np.exp(np.clip(eta,-60,60));pred=V@p;val=float((p-p0).sum()+U@lam[:n]-L@lam[n:]);grad=np.r_[U-pred,pred-L];return val,grad
  def callback(lam):
   if len(trace)%10==0:
    _,g=objective(lam);trace.append({'iteration_sample':len(trace),'objective':objective(lam)[0],'projected_gradient_inf':float(abs(np.minimum(lam,g)).max())})
   else:trace.append({'iteration_sample':len(trace)})
  result=minimize(objective,np.zeros(2*n),jac=True,bounds=[(0,None)]*(2*n),method='L-BFGS-B',callback=callback,options={'maxiter':2500,'ftol':1e-14,'gtol':1e-9,'maxls':40});p=p0*np.exp(np.clip(-V.T@(result.x[:n]-result.x[n:]),-60,60));factors=p/p0
 else:
  mode='infeasible_band_least_residual_diagnostic';weights=np.array([.5/sum(o.loc[supported].road_class.eq(c))for c in o.loc[supported].road_class]);X=V*p0[None,:]
  def objective(logf):
   budget();f=np.exp(logf);pred=X@f;error=pred-1;kl=float(np.sum(p0*(f*logf-f+1)));val=float(.5*np.sum(weights*error**2)+1e-4*kl);grad=f*(X.T@(weights*error))+1e-4*p0*f*logf;return val,grad
  result=minimize(objective,np.zeros(len(p0)),jac=True,bounds=[(np.log(.02),np.log(50.))]*len(p0),method='L-BFGS-B',options={'maxiter':1500,'ftol':1e-13,'gtol':1e-9,'maxls':40});factors=np.exp(result.x)
 ratio=np.ones(len(order));ratio[base]=factors[inverse];central=prior*ratio[:,None]*base[:,None];pred=np.asarray(A@central[:,4]).ravel();within=np.abs(pred/y-1)<=.20001;three=np.array([central*.8,central,central*1.2]);np.savez_compressed(P/'load_scenarios.npz',period_weights=three,source_expanded_od_index=order,relative_to_prior=ratio,prior_period_weights=prior,base=base,original_through_plus=dem['through_plus'][order],original_remote_through_weights=prior*(dem['through_plus'][order]&~base)[:,None]);o['supported']=supported;o['low_small12h']=pred*.8;o['central_small12h']=pred;o['high_small12h']=pred*1.2;o['central_observed_ratio']=pred/y;o['within20_supported']=within&supported;save('normal_observation_constraints.csv',o);save('group_adjustment_diagnostics.csv',[{'group_id':j,'prior_candidate12h':q0[j],'central_candidate12h':q0[j]*factors[j],'ratio_to_prior':factors[j],'constrained_screenlines':int((patterns[j]>0).sum()),'unidentified_pattern':not(patterns[j]>0).any()}for j in range(len(patterns))]);save('inverse_optimization_log.csv',trace or [{'iteration_sample':0,'mode':mode}]);led=[]
 for k,label in enumerate(['low','central','high']):
  for t,period in enumerate(['daily','hour8','hour10','hour17','daytime12h']):led.append({'load':label,'period':period,'candidate_trip_weight':float(three[k,:,t].sum()),'original_prior_weight':float((prior[:,t]*base).sum()),'meaning':'conditioned scenario weight; not measured source vehicle count'})
 save('scenario_demand_totals.csv',led);g=[]
 for cls in ['ordinary','expressway']:
  mask=o.road_class.eq(cls).to_numpy()
  for k,label in enumerate(['low','central','high']):
   pp=pred*[.8,1,1.2][k];g.append({'load':label,'road_class':cls,'all_screens':int(mask.sum()),'supported_screens':int((mask&supported).sum()),'observed_small12h':float(y[mask].sum()),'scenario_small12h_passages':float(pp[mask].sum()),'wape_all':float(abs(pp[mask]-y[mask]).sum()/y[mask].sum()),'wape_supported':float(abs(pp[mask&supported]-y[mask&supported]).sum()/y[mask&supported].sum()),'supported_within20_count':int((mask&supported&(abs(pp/y-1)<=.20001)).sum())})
 save('normal_constraint_metrics.csv',g);checks={'finite_nonnegative_scenario_weights':bool(np.isfinite(three).all()and(three>=0).all()),'central_supported_observations_in_preregistered_band':bool(within[supported].all()),'unconstrained_route_patterns_kept_at_prior':bool(np.allclose(factors[~patterns.any(axis=1)],1)),'three_loads_monotone_same_OD_shape':bool(np.allclose(three[0],three[1]*.8)and np.allclose(three[2],three[1]*1.2)),'unsupported_counts_not_relabelled_observed_zero':bool((y[~supported]>0).all()),'original_unknown_through_pool_not_reweighted':True};weights12=central[:,4];summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'solver_mode':mode,'lp_band_feasible':bool(feas.success),'lp_status':str(feas.message),'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'groups':len(patterns),'supported_screens':int(supported.sum()),'unsupported_screens':int((~supported).sum()),'central_supported_within20':int(within[supported].sum()),'full_baseline_gate_passed':bool(supported.all()and within.all()),'partial_supported_constraint_gate_passed':bool(within[supported].all()),'maximum_source_ratio':float(ratio[base].max()),'minimum_source_ratio':float(ratio[base].min()),'prior_weight_share_at_ratio_above10':float(prior[base,4][ratio[base]>10].sum()/prior[base,4].sum()),'central_weight_share_at_ratio_above10':float(weights12[base][ratio[base]>10].sum()/weights12.sum()),'central_scenario12h_total':float(weights12.sum()),'source_prior12h_total':Q,'inverse_seconds':time.monotonic()-started,'checks':checks,'new_hydraulics':False,'affected_people':False,'is_true_OD':False,'independent_validation':False};dump(R/'inverse_summary.json',summary);print(json.dumps(summary,indent=2))
 if not checks['finite_nonnegative_scenario_weights']:raise SystemExit('Invalid inverse numerical output')
if __name__=='__main__':main()
