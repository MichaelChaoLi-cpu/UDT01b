"""Brute-force synthetic original-ground checks for asynchronous peaks and local5m bounds."""
import os
os.environ['NUMBA_NUM_THREADS']='2'
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import REPORT
from reconstruct_three_ward_common_depth import ground_tables,bounds
from reconstruct_adaptive_positivity_labels import quarters
from prepare_sfincs_cpu_build import dump

def main():
 rng=np.random.default_rng(1729);checks=[]
 for trial in range(24):
  z=(rng.integers(0,40,(20,20))*.025).astype('f4');cl=np.where(rng.random((20,20))<.15,1,2).astype('u1');inside=np.ones((1,1),bool);zz,pre,count,road,unknown=ground_tables(z,cl,inside);own=np.arange(4,dtype='i4')[None,:];amin=np.array([float(np.min(z[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10][cl[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10]!=1]))for q in range(4)])
  states=(amin[None,:]+rng.random((9,4))*2).astype('f4');states[:,0]=(amin[0]+.005);truepeak=states.max(0);ap=np.where(truepeak-amin>.01,truepeak,np.nan);rm=np.array([float(np.min(z[cl!=1]))]);rp=np.array([float(rm[0]+(0.005 if trial%3==0 else .7))]);rpencoded=np.where(rp-rm>.01,rp,np.nan);sample=states[::2].copy();sample[sample-amin[None,:]<=.01]=np.nan
  result,nowners=bounds(zz,pre,count,own,rpencoded,rm,ap,amin,sample);local,bar=quarters(zz,pre,count,own,rpencoded,rm,ap,amin,np.where(cl==1,20.,np.inf).astype('f4'))
  groundarea=float((cl!=1).sum()*.25);full=[];localtrue=[];regulartrue=[]
  for t in range(len(states)):
   total=0.
   for q in range(4):
    zs=z[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10];cs=cl[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10];values=zs[cs!=1].astype('f8');total+=float(np.maximum(float(states[t,q])-values,0).sum()*.25)
   full.append(total/groundarea)
  for q in range(4):
   zs=z[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10];cs=cl[(q//2)*10:(q//2+1)*10,(q%2)*10:(q%2+1)*10];values=zs[cs!=1].astype('f8');localtrue.append(float(np.maximum(float(truepeak[q])-values,0).mean()));regulartrue.append(float(np.maximum(float(rp[0])-values,0).mean()))
  true=max(full);eps=1e-12
  ok=bool(result[2,0]<=true+eps and true<=result[3,0]+eps and np.all(local[2,0]<=np.array(localtrue)+eps)and np.all(np.array(localtrue)<=local[3,0]+eps)and np.all(local[0,0]<=np.array(regulartrue)+eps)and np.all(np.array(regulartrue)<=local[1,0]+eps)and np.all(count[0]>.0)and np.sum(count)*.25==groundarea)
  checks.append({'case':trial,'pass':ok,'true_common_peak_m':true,'common_lower_m':float(result[2,0]),'common_upper_m':float(result[3,0]),'censored_local_interval_retained':bool(local[3,0,0]>local[2,0,0])})
 result={'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'scope':'Known9-state native hydrographs, four asynchronous5m cells, excluded buildings, one shallow-censored child and regular dry/wet states; compare to brute-force sums over original ground. No solver or training.'};dump(ROOT/REPORT/'depth_bound_fixture_validation.json',result);print('PASS'if result['all_checks_passed']else'FAIL',len(checks),'brute force interval fixtures');assert result['all_checks_passed']
if __name__=='__main__':main()
