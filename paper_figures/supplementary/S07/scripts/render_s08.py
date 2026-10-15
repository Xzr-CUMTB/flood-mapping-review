from plot_helpers import *
from matplotlib.colors import LogNorm
def main():
 f=Canvas(8,210,args());joint=f.read('joint_score_density.csv');dec=f.read('risk_deciles.csv');summary=f.read('risk_distribution_summary.csv');r=f.read('fixed_map_review_results.csv');sim=f.read('simultaneous_intervals.csv');gaps=f.read('oracle_gaps.csv')
 f.title(3,4,'a','Risk score and observed omissions');f.title(96,4,'b','Core-test risk strata')
 for z,reg in enumerate(REG):
  x=15+z*40;a=f.ax(x,24,28,28);top=f.ax(x,19,28,3);side=f.ax(x+29,24,3,28);q=joint[joint.region==reg];mat=q.area_km2.to_numpy().reshape(30,30);im=a.pcolormesh(np.linspace(0,1,31),np.linspace(0,1,31),np.ma.masked_less_equal(mat.T,0),norm=LogNorm(.01,10000),cmap='Blues',shading='flat',rasterized=False);f.gid(im,f'a_joint_{z}');a.set(xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Risk score S');a.plot([0,1],[0,1],lw=.6,ls=':',color='#888888');top.fill_between(np.linspace(1/60,1-1/60,30),0,mat.sum(1),color=C['neutral']);side.fill_betweenx(np.linspace(1/60,1-1/60,30),0,mat.sum(0),color=C['neutral']);top.set(xlim=(0,1));side.set(ylim=(0,1));top.set_axis_off();side.set_axis_off();f.text(x+14,13,RN[z],ha='center')
  if z==0:a.set_ylabel('Observed FN density D');a.set_yticklabels(['','.5','1'])
  else:a.set_yticklabels([])
  q=summary[summary.region==reg].iloc[0];f.text(x,65,f'Zero-FN area: {q.zero_FN_area_fraction*100:.1f}%');f.text(x,69,f'{q.units:,} review units')
 cbar(f,im,15,74,25,'Area (km²)',ticks=[.01,10,10000])
 a=f.ax(108,20,65,31)
 for z,reg in enumerate(REG):
  q=dec[dec.region==reg];col=[C['sar'],C['value']][z];f.gid(a.plot(q.mean_score,q.observed_density,color=col,marker='o',ms=2.7,lw=1,label=RN[z])[0],f'b_strata_{z}')
 a.plot([0,1],[0,1],color='#999999',ls=':',lw=.7);a.set(xlim=(0,.85),ylim=(0,.09),xticks=[0,.4,.8],yticks=[0,.04,.08],yticklabels=['','.04','.08'],xlabel='Area-weighted mean score',ylabel='Observed FN density');legend(f,106,13,[(RN[0],C['sar'],'-'),(RN[1],C['value'],'-')],42)
 f.text(99,64,'Ten equal-area risk strata; frozen head.');f.text(99,69,'Training input: event-held-out map predictions.');f.text(99,74,'Independent OOF risk-head scores: unavailable.')
 f.title(3,84,'c','Complete fixed-map policy audit');f.title(117,84,'d','Captured missed-flood area')
 cols=[(reg,b) for reg in REG for b in [.01,.05,.1,.2]];values=np.array([[r[(r.region==reg)&(r.policy==p)&(r.budget_fraction==b)].residual_fn_km2.iloc[0] for reg,b in cols] for p in POL]);colors=np.array([[r[(r.region==reg)&(r.policy==p)&(r.budget_fraction==b)].delta_FN_vs_entropy_km2.iloc[0] for reg,b in cols] for p in POL]);a=f.ax(34,96,78,53);im=heat(f,a,values,colour=-colors,cmap=DIV,vmin=-130,vmax=130,role='c_all_120_cells',size=7);a.set(yticks=range(15),yticklabels=PN,xticks=range(8),xticklabels=['1','5','10','20']*2);a.axvline(3.5,color='white',lw=2);f.text(53.5,91,RN[0],ha='center');f.text(92.5,91,RN[1],ha='center');f.text(73,155,'Area budget (%)',ha='center');cbar(f,im,35,160,24,'Colour: FN reduction vs entropy (km²)',ticks=[-100,0,100])
 prim=['mean_entropy','fn_probability_density','learned_omission_density','stacker_omission_density','oracle_fn_density']
 for z,reg in enumerate(REG):
  a=f.ax(133,98+z*32,40,21);f.text(153,92+z*32,RN[z],ha='center')
  for p in POL:
   q=r[(r.region==reg)&(r.policy==p)].sort_values('actual_area_fraction');color=PC[p] if p in prim else '#C5C7C9';a.plot(np.r_[0,q.actual_area_fraction*100],np.r_[0,q.captured_FN_fraction*100],color=color,lw=1.1 if p in prim else .5,marker='o' if p in prim else None,ms=2,ls='--' if p=='oracle_fn_density' else '-')
  a.set(xlim=(0,20.5),ylim=(0,102),xticks=[0,10,20],yticks=[0,50,100],ylabel='Captured FN (%)');
  if z==1:a.set_xlabel('Actual reviewed area (%)')
 legend(f,119,163,[(s,PC[p],'-') for s,p in zip(['H','P','L','S','O'],prim)],11)
 f.title(3,171,'e','Paired risk-policy contrasts');f.title(117,171,'f','Distance from the oracle')
 for z,reg in enumerate(REG):
  a=f.ax(23+z*44,182,36,17);f.text(41+z*44,176,RN[z],ha='center')
  for j,p in enumerate(['mean_entropy','fn_probability_density','stacker_omission_density']):
   for k,end in enumerate(['residual_fn_km2','residual_fp_km2']):
    q=sim[(sim.region==reg)&(sim.first_model=='value_gate')&(sim.first_policy=='learned_omission_density')&(sim.second_model=='value_gate')&(sim.second_policy==p)&(sim.endpoint==end)].iloc[0];interval(f,a,j+(k-.5)*.23,q.difference_first_minus_second,q.simultaneous_lower95,q.simultaneous_upper95,[C['value'],C['error']][k],['v','s'][k],f'e_{z}_{j}_{k}')
  a.axvline(0,color='black',lw=.6);a.set(ylim=(2.6,-.6),yticks=[0,1,2],yticklabels=['L − H','L − P','L − S'] if z==0 else ['']*3,xlim=(-10,165),xticks=[0,80,160]);a.set_xlabel('Residual-area difference (km²)')
 a=f.ax(134,183,39,16)
 for z,reg in enumerate(REG):
  q=gaps[(gaps.region==reg)&(gaps.budget_fraction==.1)].set_index('policy');x=np.arange(15);a.plot(x,q.loc[POL,'FN_gap_to_oracle_km2'],color=[C['sar'],C['value']][z],marker='o',ms=2,lw=.8)
 a.set(xticks=[0,4,9,14],xticklabels=['H','L','Neg. V','Oracle'],ylabel='FN gap (km²)',ylim=(-3,170));f.text(117,207,'10% budget; policy order matches c')
 f.text(3,208,'▼ FN  ■ FP; simultaneous 95% intervals. H: entropy; L: learned; P: proxy; S: stacking.')
 for p,b in zip('abcdef',[(0,0,94,80),(94,0,86,80),(0,82,115,85),(115,82,65,85),(0,169,115,41),(115,169,65,41)]):tag(f,p,*b)
 f.finish()
if __name__=='__main__':main()
