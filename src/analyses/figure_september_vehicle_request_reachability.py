#!/usr/bin/env python3
"""September Vehicle Request Reachability, relocated supporting null result.
AnaSOP5-8: original represented request denominator, fixed snapshots and masks.
No new hydraulic simulation, route search or traffic fitting.
"""
from pathlib import Path
import os,json,hashlib,sys,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import figure_vehicle_access_and_traffic_cost as traffic
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
SLUG='september_vehicle_request_reachability'
TITLE='September Vehicle Request Reachability'
OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1024**2),b''):h.update(z)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def generate(data,source_bindings):
 bindings=dict(source_bindings)
 for p in [Path(__file__),Path(traffic.__file__),ROOT/'docs/AnaSOP.md']:
  bindings[str(p.relative_to(ROOT))]=sha(p)
 assert '| '+TITLE+' |' in (ROOT/'docs/AnaSOP.md').read_text()
 assert len(data)==24
 for _,g in data.groupby(['Frame','Receiving Water Stage']):
  assert set(g['Measure Configuration'])=={'none','Q2p5','Q5'}
  assert g['Routable Requests Percent'].nunique()==1 and g['Routable Request Proxy'].nunique()==1
  np.testing.assert_allclose(g['Routable Request Proxy']+g['Unreachable Request Proxy'],g['Original Request Proxy'],rtol=0,atol=1e-8)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.linewidth':.8,'savefig.facecolor':'white','axes.edgecolor':'#52616c'})
 fig=plt.figure(figsize=(7.5,4.7));ax=fig.add_axes([.12,.25,.84,.70]);traces=[]
 for stage,label,color,style,marker in [('H','High stage','#263746','-','o'),('L','Low stage','#9b6f9f','--','s')]:
  z=data[data['Receiving Water Stage'].eq(stage)&data['Measure Configuration'].eq('none')].sort_values('Frame')
  assert z.Frame.tolist()==[29,48,72,108]
  line,=ax.plot(z['Elapsed Hours'],z['Routable Requests Percent'],label=label,color=color,ls=style,marker=marker,markersize=5,lw=1.6)
  np.testing.assert_array_equal(line.get_ydata(),z['Routable Requests Percent'])
  traces.append({'stage':stage,'all_three_portfolios_coincide':True,'values_percent':z['Routable Requests Percent'].tolist()})
 ax.set(xlim=(2,9.5),ylim=(0,100),xlabel='Elapsed model time (h:mm)',ylabel='Requests with a modeled route (%)');ax.set_xticks([29/12,4,6,9],['2:25','4:00','6:00','9:00']);ax.set_yticks([0,25,50,75,100]);ax.grid(color='#dce3e8',lw=.6);ax.set_axisbelow(True)
 legend=ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False,fontsize=10)
 for sp in ax.spines.values():sp.set_visible(True)
 fig.canvas.draw();box=ax.get_tightbbox(fig.canvas.get_renderer());assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
 assert ax.get_title()==''
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT) as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<x<451 for x in dpi)
 data.to_csv(R/(SLUG+'_plot_data.csv'),index=False)
 caption='''September represented vehicle-request reachability at the four original snapshots, elapsed model time 2:25, 4:00, 6:00 and 9:00. Solid circles denote high prescribed receiving-water stage and dashed squares low stage. No action, Q2.5 and Q5 have exactly identical request reachability at every stage-time, so each stage is drawn once after all twenty-four source-state equalities and original request balances are checked. Both stages retain the same original core request proxy denominator, 37,113.1985; this is not the smaller fixed traffic-cost cohort. The vertical axis remains zero to one hundred percent. Rising curves show modeled temporal recovery, not treatment-restored demand. Original paired restored and newly unreachable request weights are zero across all sampled treatment comparisons. Lines connect four snapshots only; no intervening traffic state, dynamic queue, observed trip completion or unique-person count is inferred. The original doubled 17:00 small-vehicle driver-request proxy, route pool, capacity, reference-cost, masks and 0.30 m closure rule are unchanged. This supporting figure relocates the former main panel without omitting its null result or adding a model run.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 (R/(SLUG+'_methods.md')).write_text('# September Vehicle Request Reachability\n\nAnaSOP Sections 5-8 and decision MILI-D-20261024-032 relocate the former main request-reachability panel to supporting material. The independent script uses the original traffic-master preparation routine to reconstruct all twenty-four qualified saved fits, paired accounting and cohort rules. The main assembly may call the same rendering function with already verified arrays and source hashes. Both paths check exact configuration equality before collapsing curves and verify the original request balances. Full zero-to-one-hundred percent scale, four source times, stage encodings and English 450 dpi export are preserved. No simulation, optimizer, route search or sealed test is run. Output remains pending individual human review.\n')
 assert all(sha(ROOT/n)==h for n,h in bindings.items())
 val={'passed':True,'decision':'MILI-D-20261024-032','traffic_states_checked':24,'all_three_portfolios_identical_at_each_stage_time':True,'request_denominator':float(data['Original Request Proxy'].iloc[0]),'original_request_balances_preserved':True,'zero_to100_axis':True,'single_panel_without_title_or_letter':True,'source_files_unchanged':True,'new_simulations':0,'new_traffic_fits':0,'awaiting_human_review':True,'dimensions':dimensions,'dpi':dpi,'traces':traces,'output_sha256':sha(OUT)}
 dump(R/(SLUG+'_validation.json'),val);dump(R/(SLUG+'_source_bindings.json'),bindings);dump(R/(SLUG+'_completion.json'),{'title':TITLE,'status':'generated_pending_human_review','output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT)})
 page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><style>body{{font:16px/1.7 system-ui;max-width:1000px;margin:25px auto;padding:0 20px;color:#24404f}}img{{width:100%;height:auto}}</style><h1>{TITLE}</h1><p>Supporting null result; pending individual review. No action, Q2.5 and Q5 coincide at every sampled stage-time.</p><img src="../../../results/figures/Figure_{SLUG}.png?v={val['output_sha256'][:12]}" alt="Two stage curves with exactly identical reachability across all portfolios"><p>{html.escape(caption)}</p><p><a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_plot_data.csv">All twenty-four source states</a></p></html>'''
 (R/(SLUG+'_review.html')).write_text(page)
 return val

def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
 data,_,_,paired,_,_=traffic.prepare()
 assert paired.restored_request_proxy.eq(0).all() and paired.newly_unreachable_request_proxy.eq(0).all()
 print(json.dumps(generate(data,traffic.BIND),indent=2))
if __name__=='__main__':main()
