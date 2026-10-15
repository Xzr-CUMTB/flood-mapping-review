from figure_common import *
REG=['Beijing_Baoding','Harbin'];RN=['Beijing–Baoding','Harbin'];MODELS=['sar','fusion','rgb_fusion','optical','value_gate','ordinary_gate','stacker','probability_average','lower_entropy_expert','inverse_entropy_blend','terramind_tiny','terramind_tiny_random','fusion_ensemble','sar_temperature','fusion_temperature','value_gate_temperature']
NAMES=['SAR','Fusion','RGB fusion','Optical','Value','Ordinary','Stacker','Probability mean','Lower entropy','Inverse entropy','TerraMind tiny','TerraMind random','Fusion ensemble','SAR + T','Fusion + T','Value + T']
POL=['mean_entropy','total_entropy','mean_disagreement','fn_probability_density','learned_omission_density','stacker_omission_density','negative_entropy_density','ensemble_variance','ensemble_mutual_information','negative_ensemble_variance','negative_ensemble_mutual_information','random_seed_17','random_seed_29','random_seed_43','oracle_fn_density']
PN=['Mean entropy','Total entropy','Disagreement','Probability proxy','Learned risk','Stacking risk','Negative entropy','Ensemble variance','Mutual information','Negative variance','Negative MI','Random 17','Random 29','Random 43','Oracle density']
PC={p:c for p,c in zip(POL,[C['fusion'],C['ordinary'],C['sar'],C['gold'],C['value'],C['error'],C['average'],'#6685A0','#868B55','#849BA7','#ADA77A',C['neutral'],C['neutral'],C['neutral'],'#262626'])}
MCO={m:c for m,c in zip(MODELS,[C['sar'],C['fusion'],C['random'],C['optical'],C['value'],C['ordinary'],C['error'],C['average'],C['neutral'],'#637C77',C['pretrained'],'#8B939F','#886134',C['sar'],C['fusion'],C['value']])}
def legend(f,x,y,items,spacing=25):
 for i,(label,col,ls) in enumerate(items):f.line(x+i*spacing,y,x+3+i*spacing,y,col,1.4,ls);f.text(x+4+i*spacing,y,label)
def tag(f,p,x,y,w,h):f.panel_bounds[p]=[x,y,w,h]
def heat(f,a,values,colour=None,cmap='Blues',vmin=None,vmax=None,fmt='.1f',size=7,role='matrix'):
 colour=values if colour is None else colour;nr,nc=values.shape;im=a.pcolormesh(np.arange(nc+1)-.5,np.arange(nr+1)-.5,colour,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',rasterized=False);a.set(xlim=(-.5,nc-.5),ylim=(nr-.5,-.5));f.gid(im,role);a.tick_params(length=0)
 norm=im.norm
 for (i,j),v in np.ndenumerate(values):
  if not np.isfinite(v):s='NA';color='black'
  else:
   s=format(v,fmt);r,g,b,_=im.cmap(norm(colour[i,j]));color='white' if .2126*r+.7152*g+.0722*b<.43 else 'black'
  a.text(j,i,s,ha='center',va='center',fontsize=size,color=color)
 return im
def cbar(f,im,x,y,w,label,ticks=None):
 a=f.ax(x,y,w,1.8);cb=f.fig.colorbar(im,cax=a,orientation='horizontal',ticks=ticks);cb.solids.set_rasterized(False);cb.outline.set_linewidth(.8);cb.ax.tick_params(length=1.5,pad=1);f.text(x+w+3.5,y+.9,label);return cb
def interval(f,a,y,point,lo,hi,color,marker='o',role='interval',ms=3):
 f.gid(a.plot([lo,hi],[y,y],color=color,lw=1)[0],role+'_bar');f.gid(a.scatter([point],[y],s=ms*ms,color=color,marker=marker,zorder=4),role+'_point')
