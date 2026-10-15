from plot_helpers import *
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
def main():
 f=Canvas(10,175,args());blocks=f.read('spatial_blocks.csv');pat=f.read('patch_assignments.csv');s=f.read('scale_sensitivity_intervals.csv');cont=f.read('block_contributions.csv.gz');influence=f.read('influence_summary.csv');key=f.read('contrast_key.csv')
 f.title(3,4,'a','Observed spatial blocks');f.title(76,4,'b','All primary paired effects')
 for z,reg in enumerate(REG):
  x=12+z*33;a=f.ax(x,33,27,29);q=blocks[blocks.region==reg];xx=q.left_m.min()/1000;yy=q.bottom_m.min()/1000;rects=[Rectangle((r.left_m/1000-xx,r.bottom_m/1000-yy),10,10) for r in q.itertuples()];pc=PatchCollection(rects,cmap='Blues',norm=LogNorm(1,100),edgecolors='white',linewidth=.1);pc.set_array(q.valid_area_km2.to_numpy());f.gid(a.add_collection(pc),f'a_blocks_{z}');a.autoscale_view();a.set_aspect('equal');a.set(xlabel='ΔEasting (km)',ylabel='ΔNorthing (km)' if z==0 else None);a.locator_params(nbins=3);f.text(x+13.5,12,['BB: 252 blocks','HB: 148 blocks'][z],ha='center')
  ec=f.ax(x,17,27,7)
  for field,col in [('valid_area_km2',C['sar']),('flood_area_km2',C['value'])]:
   v=np.sort(q[field].to_numpy());ec.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=col,lw=.8)
  ec.set(xscale='symlog',xlim=(0,150),ylim=(0,1),xticks=[0,10,100],xticklabels=['0','10','100'],yticks=[0,1],yticklabels=['','1'] if z==0 else ['','']);ec.tick_params(pad=1)
 f.text(3,29,'Area CDF: blue valid, teal flood (km²)')
 cbar(f,pc,14,75,22,'Valid area (km²)',ticks=[1,10,100]);f.text(3,84,'10 km blocks; patch-centroid assignment.')
 # 24 primary simultaneous estimates: six comparisons x two endpoints x two regions.
 for z,reg in enumerate(REG):
  x=87+z*47;a=f.ax(x,20,40,57);f.text(x+20,12,RN[z],ha='center');q=s[(s.region==reg)&(s.block_size_m==10000)]
  for j in range(6):
   for k,end in enumerate(['residual_fn_km2','residual_fp_km2']):
    r=q[(q.contrast==f'C{j+1}')&(q.endpoint==end)].iloc[0];y=j*2+k;interval(f,a,y,r.difference_first_minus_second,r.simultaneous_lower95,r.simultaneous_upper95,[C['value'],C['error']][k],['v','s'][k],f'b_{z}_{j}_{k}')
  a.axvline(0,color='black',lw=.65);a.set(ylim=(11.6,-.6),xlim=(-25,165),xticks=[0,80,160],yticks=np.arange(12),yticklabels=[f'C{j+1} '+e for j in range(6) for e in ['FN','FP']] if z==0 else ['']*12,xlabel='First − second (km²)')
 f.text(77,87,'▼ FN  ■ FP; 95% simultaneous intervals, 12 per region')
 # Complete contrast key in a compact two-column ledger.
 for i,row in key.iterrows():f.text(3+(i//3)*89,91+(i%3)*4,f'{row.contrast}  {row.label}')
 f.text(3,104,'H: mean entropy; L: learned omission; P: probability proxy; S: stacking risk. Budget: 10%.')
 f.title(3,112,'c','Block-scale sensitivity');f.title(96,112,'d','Concentration of block effects')
 for z,reg in enumerate(REG):
  x=20+z*39;a=f.ax(x,126,31,28);f.text(x+15.5,119,RN[z],ha='center')
  for j,(con,end) in enumerate([('C1','residual_fn_km2'),('C1','residual_fp_km2'),('C3','residual_fn_km2'),('C3','residual_fp_km2')]):
   for k,(size,col) in enumerate([(5000,C['sar']),(10000,C['value']),(20000,C['gold'])]):
    q=s[(s.region==reg)&(s.block_size_m==size)&(s.contrast==con)&(s.endpoint==end)].iloc[0];interval(f,a,j+(k-1)*.2,q.difference_first_minus_second,q.simultaneous_lower95,q.simultaneous_upper95,col,['o','s','^'][k],f'c_{z}_{j}_{k}',2.5)
  a.axvline(0,color='black',lw=.6);a.set(xlim=(-20,190),xticks=[0,80,160],ylim=(3.6,-.6),yticks=range(4),yticklabels=['C1 FN','C1 FP','C3 FN','C3 FP'] if z==0 else ['']*4,xlabel='Effect (km²)')
 legend(f,9,168,[('5 km',C['sar'],'-'),('10 km',C['value'],'-'),('20 km',C['gold'],'-')],25)
 for z,reg in enumerate(REG):
  a=f.ax(109+z*36,126,28,27);f.text(123+z*36,119,RN[z],ha='center')
  for con,col in [('C1',C['fusion']),('C3',C['value'])]:
   q=cont[(cont.region==reg)&(cont.contrast==con)&(cont.endpoint=='FN')];v=q.signed_contribution_km2.to_numpy();v=v[np.argsort(-np.abs(v),kind='stable')];x=np.arange(1,len(v)+1)/len(v)*100;f.gid(a.plot(x,np.cumsum(v),color=col,lw=1,label=con)[0],f'd_signed_{z}_{con}');a.axhline(v.sum(),color=col,lw=.5,ls=':')
  a.axhline(0,color='black',lw=.5);a.set(xlim=(0,100),xticks=[0,50,100],xlabel='Blocks included (%)',ylabel='Cumulative ΔFN (km²)' if z==0 else None);a.locator_params(axis='y',nbins=3)
 f.text(97,165,'Sorted by |block effect|; purple C1, teal C3.')
 f.text(97,170,'38 fits · 142 frozen files · 8,233 core patches')
 for p,b in zip('abcd',[(0,0,74,86),(74,0,106,86),(0,110,94,65),(94,110,86,65)]):tag(f,p,*b)
 f.finish()
if __name__=='__main__':main()
