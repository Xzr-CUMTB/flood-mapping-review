from plot_helpers import *
def main():
 f=Canvas(6,170,args());r=f.read('reliability.csv');m=f.read('calibration_metrics.csv');s=f.read('confidence_spectrum.csv');op=f.read('operating_points.csv');cols=[C['sar'],C['fusion'],C['value']];models=['sar','fusion','value_gate']
 f.title(3,4,'a','Probability calibration');f.title(96,4,'b','Confident-omission response')
 legend(f,14,11,[(n,c,'-') for n,c in zip(['SAR','Fusion','Value'],cols)],23);f.text(99,11,'Solid: raw   Dashed: calibrated')
 for z,reg in enumerate(REG):
  x=14+z*41;a=f.ax(x,27,31,32);b=f.ax(x,21,31,4);f.text(x+15.5,17,RN[z],ha='center');a.plot([0,1],[0,1],color='#A0A0A0',ls=':',lw=.7)
  for model,col in zip(models,cols):
   for typ,ls in [('Raw','-'),('Calibrated','--')]:
    q=r[(r.region==reg)&(r.model==model)&(r.version==typ)&(r.area_m2>0)];f.gid(a.plot(q.mean_probability,q.flood_fraction,color=col,ls=ls,lw=.9)[0],f'a_{z}_{model}_{typ}');b.plot((q.bin+.5)/20,100*q.area_m2/q.area_m2.sum(),color=col,ls=ls,lw=.7)
  a.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted p');b.set(xlim=(0,1),ylim=(0,100));b.set_axis_off()
  if z==0:a.set_ylabel('Observed flood fraction');a.set_yticklabels(['','.5','1'])
  else:a.set_yticklabels([])
  f.text(x,72,'Brier: raw → calibrated',size=7)
  for j,model in enumerate(models):
   q=m[(m.region==reg)&(m.base_model==model)].set_index('version');f.text(x,77+j*4,f"{q.loc['Raw','brier']:.4f} → {q.loc['Calibrated','brier']:.4f}",color=cols[j])
  x=106+z*39;a=f.ax(x,23,30,34);f.text(x+15,17,RN[z],ha='center')
  for model,col in zip(models,cols):
   for typ,ls in [('Raw','-'),('Calibrated','--')]:
    q=s[(s.region==reg)&(s.model==model)&(s.version==typ)];f.gid(a.plot(q.tau,100*q.omission_fraction,color=col,ls=ls,lw=1,marker='o',ms=2)[0],f'b_{z}_{model}_{typ}')
  a.set(xlim=(.04,.21),ylim=(0,18),xticks=[.05,.1,.2],yticks=[0,5,10,15],xlabel='Threshold τ')
  if z==0:a.set_ylabel('C(τ), % of reference flood')
  else:a.set_yticklabels([])
  a=f.ax(x,74,30,11)
  for model,base,col in [('fusion','sar',C['fusion']),('value_gate','fusion',C['value'])]:
   for typ,ls in [('Raw','-'),('Calibrated','--')]:
    q=s[(s.region==reg)&(s.model==model)&(s.version==typ)].sort_values('tau');b=s[(s.region==reg)&(s.model==base)&(s.version==typ)].sort_values('tau');a.plot(q.tau,100*(q.omission_fraction.to_numpy()-b.omission_fraction.to_numpy()),color=col,ls=ls,lw=.9)
  a.axhline(0,color='black',lw=.6);a.set(xlim=(.04,.21),ylim=(-4,7),xticks=[],yticks=[0,5]);
  if z==0:a.set_ylabel('ΔC (pp)')
  else:a.set_yticklabels([])
 f.text(106,90,'Purple: Fusion − SAR; teal: Value − Fusion')
 f.title(3,97,'c','Development-to-core transfer');f.title(96,97,'d','Frozen operating points')
 f.text(15,104,'Filled: development target; open: core')
 for j,(scenario,metric,lim,ticks,label) in enumerate([('matched_fpr','false_positive_rate',(0,1.6),[0,.8,1.6],'FPR (%)'),('matched_recall','recall',(20,95),[20,50,80],'Recall (%)')]):
  a=f.ax(29+j*34,114,27,34);a.set(ylim=(5.7,-.7),yticks=np.arange(6),yticklabels=['BB  SAR','Fusion','Value','HB  SAR','Fusion','Value'] if j==0 else ['']*6,xlim=lim,xticks=ticks,xlabel=label)
  for z,region in enumerate(REG):
   for k,model in enumerate(models):
    row=op[(op.region==region)&(op.model==model)&(op.scenario==scenario)].iloc[0];y=z*3+k;target=100*row.development_target_rate;actual=100*row[metric];a.plot([target,actual],[y,y],color=cols[k],lw=1);f.gid(a.scatter([target],[y],color=cols[k],s=9),f'c_target_{j}_{z}_{k}');f.gid(a.scatter([actual],[y],facecolors='white',edgecolors=cols[k],s=13,zorder=4),f'c_core_{j}_{z}_{k}')
 for z,region in enumerate(REG):
  x=108+z*39;a=f.ax(x,114,28,34);f.text(x+14,107,RN[z],ha='center')
  for model,col in zip(models,cols):
   q=op[(op.region==region)&(op.model==model)].sort_values('threshold');a.plot(q.fp_km2,q.fn_km2,color=col,lw=.8,alpha=.8)
   for scenario,mk in [('raw_0.5','o'),('matched_fpr','s'),('matched_recall','^')]:
    q=op[(op.region==region)&(op.model==model)&(op.scenario==scenario)];f.gid(a.scatter(q.fp_km2,q.fn_km2,s=14,marker=mk,facecolors='white' if scenario!='raw_0.5' else col,edgecolors=col,zorder=4),f'd_{z}_{model}_{scenario}')
  a.set(xlim=(-30,880),ylim=(0,600),xticks=[0,400,800],yticks=[0,200,400,600],xlabel='FP area (km²)')
  if z==0:a.set_ylabel('FN area (km²)')
  else:a.set_yticklabels([])
 for x,label,mk in [(98,'0.5','o'),(117,'FPR-matched','s'),(147,'Recall-matched','^')]:
  f.fig.add_artist(Line2D([x/180],[1-159/170],transform=f.fig.transFigure,marker=mk,markersize=3,markerfacecolor='white' if mk!='o' else 'black',markeredgecolor='black',linestyle='none'));f.text(x+2.5,159,label)
 f.text(3,165,'Positive temperatures preserve the 0.5 binary maps. Matching is performed on development data.')
 for p,b in zip('abcd',[(0,0,93,93),(94,0,86,93),(0,95,93,67),(94,95,86,67)]):tag(f,p,*b)
 f.finish()
if __name__=='__main__':main()
