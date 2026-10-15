from plot_helpers import *
def main():
 f=Canvas(9,220,args());r=f.read('all_review_results.csv');w=f.read('loss_winners.csv');rk=f.read('rank_trajectories.csv');check=f.read('selection_recheck.csv');bud=[.01,.05,.1,.2];keys=[(reg,b) for reg in REG for b in bud]
 f.title(3,4,'a','All maps under mean-entropy review');f.title(117,4,'b','All Value-map policies')
 vals=np.array([[r[(r.region==reg)&(r.model==model)&(r.policy=='mean_entropy')&(r.budget_fraction==b)].residual_fn_km2.iloc[0] for reg,b in keys] for model in MODELS]);frac=np.array([[r[(r.region==reg)&(r.model==model)&(r.policy=='mean_entropy')&(r.budget_fraction==b)].residual_fn_km2.iloc[0]/r[(r.region==reg)&(r.model==model)&(r.policy=='mean_entropy')&(r.budget_fraction==b)].initial_fn_km2.iloc[0] for reg,b in keys] for model in MODELS]);a=f.ax(34,19,78,62);im=heat(f,a,vals,colour=frac,cmap='Blues',vmin=0,vmax=1,role='a_all_128_cells');a.set(yticks=range(16),yticklabels=NAMES,xticks=range(8),xticklabels=['1','5','10','20']*2);a.axvline(3.5,color='white',lw=2);f.text(53.5,13,RN[0],ha='center');f.text(92.5,13,RN[1],ha='center');cbar(f,im,35,89,22,'Colour: residual / initial FN; numbers: km²',ticks=[0,.5,1])
 fixed=r[r.model=='value_gate'];v=[]
 for p in POL:
  row=[]
  for reg,b in keys:
   q=fixed[(fixed.region==reg)&(fixed.budget_fraction==b)].set_index('policy');row.append(q.loc[p,'residual_fn_km2']-q.loc['mean_entropy','residual_fn_km2'])
  v.append(row)
 v=np.array(v);a=f.ax(129,19,47,58);im=heat(f,a,v,colour=-v,cmap=DIV,vmin=-130,vmax=130,fmt='+.0f',role='b_all_policy_deltas');a.set(yticks=range(15),yticklabels=[f'P{i+1}' for i in range(15)],xticks=range(8),xticklabels=['1','5','10','20']*2);a.axvline(3.5,color='white',lw=1.5);f.text(140.7,13,'BB',ha='center');f.text(164.2,13,'HB',ha='center');f.text(119,86,'ΔFN vs entropy (km²); teal: reduction');f.text(119,91,'Columns: 1%, 5%, 10%, 20% budget')
 f.title(3,100,'c','Pareto geometry at 10% review');f.title(96,100,'d','Cost-sensitive winners: all 16 maps')
 for z,reg in enumerate(REG):
  a=f.ax(16+z*41,113,30,28);q=r[(r.region==reg)&(r.budget_fraction==.1)];deploy=q[q.policy!='oracle_fn_density'];oracle=q[q.policy=='oracle_fn_density'];f.gid(a.scatter(deploy.residual_fp_km2,deploy.residual_fn_km2,s=5,color=C['neutral'],alpha=.65),f'c_candidates_{z}');eff=deploy[deploy.pareto_efficient_fn_fp].sort_values('residual_fp_km2');f.gid(a.plot(eff.residual_fp_km2,eff.residual_fn_km2,'o-',color=C['value'],ms=2.5,lw=1)[0],f'c_pareto_{z}');a.scatter(oracle.residual_fp_km2,oracle.residual_fn_km2,s=12,facecolors='none',edgecolors=C['gold'],marker='D',lw=.6);a.set(xscale='symlog',yscale='symlog',xlim=(0,1500),ylim=(0,750),xticks=[0,10,1000],yticks=[0,10,100],xlabel='Residual FP (km²)');a.set_xticklabels(['0','10','1000']);a.set_yticklabels(['0','10','100'] if z==0 else ['','','']);f.text(31+z*41,107,RN[z],ha='center')
  if z==0:a.set_ylabel('Residual FN (km²)')
 f.text(3,151,'Teal: deployable frontier; diamonds: oracle')
 # Winner identity plus absolute gap to second-best; identical entropy policy in each cell.
 winner_names=sorted(w.winner.unique(),key=lambda q:MODELS.index(q));code={n:chr(65+i) for i,n in enumerate(winner_names)};palette=['#446A9C','#80669D','#147D83','#D5A34A','#AC88AA','#495C77','#BD5B4B']
 a=f.ax(111,114,65,25);arr=np.array([[winner_names.index(w[(w.region==reg)&(w.budget_fraction==b)&(w.lambda_FP==l)].winner.iloc[0]) for reg,b in keys] for l in [.25,1.,4.]]);from matplotlib.colors import ListedColormap
 f.gid(a.pcolormesh(np.arange(9)-.5,np.arange(4)-.5,arr,cmap=ListedColormap(palette[:len(winner_names)]),vmin=-.5,vmax=len(winner_names)-.5,shading='flat',rasterized=False),'d_winner_matrix');a.set(xlim=(-.5,7.5),ylim=(2.5,-.5),yticks=range(3),yticklabels=['0.25','1','4'],xticks=range(8),xticklabels=['1','5','10','20']*2,ylabel='FP penalty λ');a.tick_params(length=0);a.axvline(3.5,color='white',lw=2);f.text(127,107,'BB',ha='center');f.text(160,107,'HB',ha='center')
 for i,l in enumerate([.25,1.,4.]):
  for j,(reg,b) in enumerate(keys):
   q=w[(w.region==reg)&(w.budget_fraction==b)&(w.lambda_FP==l)].iloc[0];a.text(j,i-.18,code[q.winner],ha='center',va='center',color='white',fontsize=7,fontweight='bold');a.text(j,i+.20,f'{q.gap_km2:.1f}',ha='center',va='center',color='white',fontsize=7)
 f.text(98,146,'Letter: winner; number: next-best gap (km²)')
 for i,n in enumerate(winner_names):f.text(98+(i%2)*40,151+(i//2)*4,f'{code[n]}: {NAMES[MODELS.index(n)]}',color=palette[i])
 legend(f,3,158,[(n,MCO[m],'-') for n,m in [('SAR','sar'),('Fusion','fusion'),('Ordinary','ordinary_gate'),('Value','value_gate')]],22)
 f.title(3,163,'e','FN ranks; end labels: gap (km²)');f.title(96,163,'f','Budget and correction audit')
 for z,reg in enumerate(REG):
  a=f.ax(16+z*41,174,30,17);q=rk[rk.region==reg]
  for model in ['sar','fusion','ordinary_gate','value_gate']:
   b=q[q.model==model].sort_values('budget_fraction');f.gid(a.plot(b.budget_fraction*100,b['rank'],color=MCO[model],marker='o',ms=2.5,lw=1)[0],f'e_rank_{z}_{model}')
   a.text(20.8,b['rank'].iloc[-1],f'{b.gap_to_best_km2.iloc[-1]:.1f}',fontsize=7,va='center',color=MCO[model])
  a.set(xlim=(0,26),ylim=(4.4,.6),xticks=[1,10,20],yticks=[1,2,3,4],xlabel='Budget (%)');f.text(31+z*41,170,RN[z],ha='center')
  if z==0:a.set_ylabel('FN rank')
  else:a.set_yticklabels([])
 a=f.ax(109,175,29,16)
 for i,b in enumerate(bud):
  q=check[check.budget_fraction==b];short=(b-q.actual_area_fraction.to_numpy())*1e6;a.plot([i,i],[short.min(),short.max()],color=C['sar'],lw=1);a.scatter([i],[np.median(short)],color=C['value'],s=9)
 a.set(xticks=range(4),xticklabels=['1','5','10','20'],xlabel='Budget (%)',ylabel='Unused area (ppm)');f.text(146,176,'1,824 selections');f.text(146,181,'Independently rebuilt');err=max(check.fn_conservation_error_km2.abs().max(),check.fp_conservation_error_km2.abs().max());f.text(146,187,f'Max |area error|');f.text(146,192,f'{err:.1e} km²')
 # Complete policy key, shared with the b row order.
 for i,p in enumerate(POL):f.text(3+(i//4)*45,203+(i%4)*4,f'P{i+1}: {PN[i]}',color=PC[p] if p in ['mean_entropy','learned_omission_density','fn_probability_density'] else 'black')
 for p,b in zip('abcdef',[(0,0,115,96),(115,0,65,96),(0,98,94,60),(94,98,86,63),(0,161,94,40),(94,161,86,40)]):tag(f,p,*b)
 f.finish()
if __name__=='__main__':main()
