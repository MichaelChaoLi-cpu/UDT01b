"""Single bounded full-domain conditional U-Net fit and two cheap baselines.
Run prepare_three_ward_unet_exact_pilot.py once before this script.
"""
from pathlib import Path
import csv,json,math,random,sys,time,os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import OUT,REPORT,sha,read,write,arrays,load_label

class Block(nn.Module):
 def __init__(self,cin,cout):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU());self.film=nn.Linear(64,cout*2)
  nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
 def forward(self,x,c):
  z=self.net(x);scale,offset=self.film(c).chunk(2,dim=1)
  return z*(1+scale[:,:,None,None])+offset[:,:,None,None]
class ConditionalUNet(nn.Module):
 def __init__(self):
  super().__init__();w=[8,16,32,64,128]
  self.condition=nn.Sequential(nn.Linear(1920,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
  self.enc=nn.ModuleList([Block(58 if i==0 else w[i-1],n) for i,n in enumerate(w)])
  self.dec=nn.ModuleList([Block(w[i+1]+w[i],w[i]) for i in range(3,-1,-1)])
  self.head=nn.Conv2d(8,1,1);nn.init.constant_(self.head.bias,-3.)
 def forward(self,x,c):
  h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%16,0,(-h)%16));c=self.condition(c);skips=[]
  for i,b in enumerate(self.enc):
   if i:x=F.avg_pool2d(x,2)
   x=b(x,c);skips.append(x)
  for b,s in zip(self.dec,reversed(skips[:-1])):
   x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False);x=b(torch.cat([x,s],dim=1),c)
  return F.softplus(self.head(x))[..., :h,:w]

def synchronize(device):
 if device.type=='mps':torch.mps.synchronize()

def metrics(y,p,area,mask):
 m=mask&np.isfinite(y)&np.isfinite(p)&(area>0);a=area[m];t=y[m];v=p[m];total=float(a.sum())
 if not total:return {'support_km2':0.,'mae_m':None,'rmse_m':None,'bias_m':None,'wet_mae_m':None}
 e=v-t;wet=t>.1
 d={'support_km2':total/1e6,'mae_m':float(np.sum(a*np.abs(e))/total),'rmse_m':float(np.sqrt(np.sum(a*e*e)/total)),'bias_m':float(np.sum(a*e)/total),'wet_mae_m':float(np.sum(a[wet]*np.abs(e[wet]))/a[wet].sum()) if wet.any() else None}
 for threshold in [.05,.1,.3,.5]:
  truth=t>threshold;pred=v>threshold;tp=float(a[truth&pred].sum());fp=float(a[~truth&pred].sum());fn=float(a[truth&~pred].sum());key=f'{threshold:g}'
  d.update({f'iou_{key}':tp/(tp+fp+fn) if tp+fp+fn else None,f'recall_{key}':tp/(tp+fn) if tp+fn else None,f'precision_{key}':tp/(tp+fp) if tp+fp else None,f'missed_km2_{key}':fn/1e6,f'false_km2_{key}':fp/1e6,f'true_wet_km2_{key}':(tp+fn)/1e6,f'predicted_wet_km2_{key}':(tp+fp)/1e6})
 return d

def writecsv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
 p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json')
 if (REPORT/'completion.json').exists():raise FileExistsError('Completed pilot cannot be retrained in place')
 if (OUT/'last.pt').exists():raise FileExistsError('Use an explicitly implemented checkpoint resume; refuse silent second fit')
 for name,h in p['implementation_hashes'].items():assert sha(ROOT/name)==h, 'Implementation differs from frozen protocol'
 for name,h in p['source_hashes'].items():assert sha(ROOT/name)==h, name
 assert sha(OUT/'prepared_inputs.npz')==p['prepared_inputs_sha256'] and sha(OUT/'normalization.json')==p['normalization_sha256']
 seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(6)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 if device.type=='mps':torch.mps.manual_seed(seed)
 z=arrays(OUT/'prepared_inputs.npz');records=p['records'];ids=[r['case_id'] for r in records]
 tr=[i for i,r in enumerate(records) if r['split']=='train'];va=[i for i,r in enumerate(records) if r['split']=='validation']
 labels=[load_label(r['case_id'],r['split']) for r in records]
 y=np.array([d['Peak Mean Depth Lower'] for d in labels],dtype='f4');m=np.array([d['Exact Label Mask'] for d in labels])&z['Eligible Ground Mask'][None]
 assert all(np.isfinite(y[i,m[i]]).all() for i in range(16))
 # Numeric placeholders outside the loss mask are never labels and have zero weight.
 safe=np.where(m,y,0).astype('f4');weights=(m*z['Ground Area'][None]).astype('f4');weights/=weights.sum(axis=(1,2))[:,None,None]
 x=torch.from_numpy(z['Static Input'][None]).to(device);c=torch.from_numpy(z['Forcing Input']).to(device)
 target=torch.from_numpy(safe[:,None]).to(device);weight=torch.from_numpy(weights[:,None]).to(device)
 model=ConditionalUNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
 env={'torch':torch.__version__,'python':sys.version,'device':str(device),'cpu_threads':torch.get_num_threads(),'parameters':sum(v.numel() for v in model.parameters()),'protocol_sha256':ph,'seed':seed,'repeatability':'Seeded single run; Metal kernel bitwise determinism is not certified.','start_time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'fit_count':1}
 write(REPORT/'environment.json',env)
 best=math.inf;best_epoch=None;stale=0;history=[];steps=0;start=time.perf_counter();rng=np.random.default_rng(seed)
 for epoch in range(1,p['maximum_epochs']+1):
  model.train();running=0.;order=rng.permutation(tr).tolist();es=time.perf_counter()
  for i in order:
   optimizer.zero_grad(set_to_none=True);pred=model(x,c[i:i+1]);loss=((pred-target[i:i+1]).square()*weight[i:i+1]).sum()
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step();running+=float(loss.detach().cpu());steps+=1
  synchronize(device);row={'epoch':epoch,'train_mse_m2':running/len(tr),'validation_mae_m':None,'epoch_seconds':time.perf_counter()-es,'steps':steps}
  if epoch in p['validation_epochs']:
   model.eval();vals=[]
   with torch.inference_mode():
    for i in va:
     pred=model(x,c[i:i+1]);vals.append(float(((pred-target[i:i+1]).abs()*weight[i:i+1]).sum().cpu()))
   score=float(np.mean(vals));row['validation_mae_m']=score
   if score<best-1e-6:
    best=score;best_epoch=epoch;stale=0
    torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation_mae_m':best,'protocol_sha256':ph,'normalization_sha256':sha(OUT/'normalization.json'),'seed':seed},OUT/'best.pt')
   else:stale+=1
  history.append(row)
  # Save restartable training state without modifying the first fit's recipe.
  ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'steps':steps,'history':history,'best_epoch':best_epoch,'best_validation_mae_m':best,'stale_checks':stale,'numpy_generator_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'protocol_sha256':ph}
  if device.type=='mps':ck['mps_rng']=torch.mps.get_rng_state()
  torch.save(ck,OUT/'last.pt');writecsv(REPORT/'training_history.csv',history)
  write(REPORT/'progress.json',{'status':'training','epoch':epoch,'maximum_epochs':p['maximum_epochs'],'best_epoch':best_epoch,'best_validation_mae_m':best,'elapsed_seconds':time.perf_counter()-start})
  print(json.dumps(row|{'best_epoch':best_epoch,'best_validation_mae_m':best}),flush=True)
  if stale>=p['early_stop_validation_checks']:break
 fit_seconds=time.perf_counter()-start
 bestck=torch.load(OUT/'best.pt',map_location=device,weights_only=False);model.load_state_dict(bestck['state_dict']);model.eval()
 predictions=[];inference_times=[]
 with torch.inference_mode():
  # One untimed warmup. No hydraulic runtime or compilation comparisons claimed.
  model(x,c[tr[0]:tr[0]+1]);synchronize(device)
  for i in range(16):
   synchronize(device);tic=time.perf_counter();pred=model(x,c[i:i+1]);synchronize(device);inference_times.append(time.perf_counter()-tic)
   a=pred[0,0].cpu().numpy().copy();a[~z['Eligible Ground Mask']]=np.nan;predictions.append(a)
 predictions=np.array(predictions);np.savez_compressed(OUT/'predictions.npz',**{'Conditional UNet Peak Mean Depth':predictions,'Case ID':np.array(ids),'Eligible Ground Mask':z['Eligible Ground Mask']})
 # Retrieve exclusively by prescribed input curves. Mean over redundant stage channels
 # gives the water-level modality weight 1/2 rather than seven times rainfall.
 curves=z['Forcing Input'].reshape(16,240,8);retrieval=[];metric_rows=[];timings=[]
 for i,r in enumerate(records):
  cand=[j for j in tr if records[j]['family_id']!=r['family_id']]
  ds=np.array([.5*np.mean((curves[i,:,0]-curves[j,:,0])**2)+.5*np.mean((curves[i,:,1:]-curves[j,:,1:])**2) for j in cand]);j=cand[int(np.argmin(ds))]
  assert records[j]['split']=='train' and records[j]['family_id']!=r['family_id']
  nn_pred=np.where(m[j],y[j],np.nan);pair=m[i]&m[j];area=z['Ground Area'];exact_area=float(area[m[i]].sum());pair_area=float(area[pair].sum())
  retrieval.append({'case_id':ids[i],'split':r['split'],'source_case_id':ids[j],'source_parent_id':records[j]['parent_id'],'distance':float(ds.min()),'target_exact_km2':exact_area/1e6,'paired_km2':pair_area/1e6,'paired_fraction_of_exact':pair_area/exact_area,'abstained_km2':(exact_area-pair_area)/1e6})
  for scope,region in [('domain',np.ones_like(m[i])) ,('three_wards',z['Ward Code']>0)]+[(f'ward_{w}',z['Ward Code']==w) for w in [1,2,3]]:
   exact=m[i]&region;shared=pair&region;den=float(area[exact].sum())
   for method,pred in [('dry',np.zeros_like(y[i])),('retrieval',nn_pred),('conditional_unet',predictions[i])]:
    for support,mask in [('paired_exact',shared),('own_exact',exact&np.isfinite(pred))]:
     values=metrics(y[i],pred,area,mask)
     metric_rows.append({'case_id':ids[i],'parent_id':r['parent_id'],'split':r['split'],'scope':scope,'method':method,'support':support,'target_exact_km2':den/1e6,'coverage_fraction':values['support_km2']*1e6/den if den else None,**values})
  timings.append({'case_id':ids[i],'split':r['split'],'forward_seconds_prepared_features':inference_times[i]})
 writecsv(REPORT/'event_metrics.csv',metric_rows);writecsv(REPORT/'retrieval_choices.csv',retrieval);writecsv(REPORT/'inference_runtime.csv',timings)
 macro=[]
 for split in ['train','validation']:
  for scope in ['domain','three_wards','ward_1','ward_2','ward_3']:
   for method in ['dry','retrieval','conditional_unet']:
    for support in ['paired_exact','own_exact']:
     items=[r for r in metric_rows if r['split']==split and r['scope']==scope and r['method']==method and r['support']==support]
     keys=[k for k in items[0] if isinstance(items[0][k],(int,float)) or items[0][k] is None]
     agg={};counts={}
     for k in keys:
      vv=[r[k] for r in items if r[k] is not None];agg[k]=float(np.mean(vv)) if vv else None;counts[k]=len(vv)
     macro.append({'split':split,'scope':scope,'method':method,'support':support,'parents':len(items),**agg})
 writecsv(REPORT/'macro_metrics.csv',macro)
 done={'status':'complete','fit_count':1,'epochs_completed':epoch,'optimizer_steps':steps,'best_epoch':best_epoch,'best_validation_mae_m_own_exact':best,'fit_elapsed_seconds':fit_seconds,'forward_median_seconds_prepared_features':float(np.median(inference_times)),'runtime_scope':'Warm full-domain forward only, prepared features already resident on device; not end-to-end or hydraulic speedup.','protocol_sha256':ph,'best_checkpoint_sha256':sha(OUT/'best.pt'),'predictions_sha256':sha(OUT/'predictions.npz'),'sealed_response_access':False,'source_integrity_after_fit':all(sha(ROOT/k)==v for k,v in p['source_hashes'].items()),'stop_reason':'early_stop' if stale>=p['early_stop_validation_checks'] else 'maximum_epochs','validation_reused_for_selection':True}
 assert done['source_integrity_after_fit'];write(REPORT/'completion.json',done);write(REPORT/'progress.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
