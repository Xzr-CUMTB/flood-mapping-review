"""Fig03: native-vector architecture and full-population analytical graphics."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm,TwoSlopeNorm
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image
OUT=Path(os.path.abspath(__file__)).parents[1];SD=OUT/'source_data';AS=OUT/'assets';AS.mkdir(exist_ok=True)
W,H=180,200
C={'sar':'#446A9C','fusion':'#80669D','value_gate':'#147D83','ordinary_gate':'#AC88AA','negative':'#BD5B4B','neutral':'#B9C2CA'}
EVENTS=['2020_flood_season','2021_Henan','2024_April','2024_June'];EN=['2020 season','2021 Henan','2024 April','2024 June']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--font',default='Times New Roman');ap.add_argument('--round',default='1');ap.add_argument('--final',action='store_true');a=ap.parse_args();suffix='' if a.font=='Times New Roman' else '_Arial'
    plt.rcParams.update({'font.family':a.font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.1,'axes.edgecolor':'black','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.8,'ytick.major.width':.8,'xtick.major.size':2,'ytick.major.size':2,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'Fig03-OOF','mathtext.fontset':'custom','mathtext.rm':a.font,'mathtext.it':a.font+':italic','mathtext.bf':a.font+':bold','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(W/25.4,H/25.4));commands=[]
    def ax(x,y,w,h,**kw):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H],**kw)
    def txt(x,y,s,size=7,bold=False,ha='left',native=False,plain=None,**kw):
        obj=fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='top',**kw)
        if native:commands.append(dict(type='text',x=x,y=y,text=plain or s,size=size,bold=bold,ha=ha))
        return obj
    def rect(x,y,w,h,fill='white',lw=.6,native=False):
        fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=fill,edgecolor='black',lw=lw,zorder=-2))
        if native:commands.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,lw=lw))
    def line(points,arrow=True,dash=False,lw=.8,native=False):
        for j,(p,q) in enumerate(zip(points[:-1],points[1:])):
            if arrow and j==len(points)-2:fig.add_artist(FancyArrowPatch((p[0]/W,1-p[1]/H),(q[0]/W,1-q[1]/H),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=5,color='black',lw=lw,linestyle='--' if dash else '-',shrinkA=0,shrinkB=0))
            else:fig.add_artist(Line2D([p[0]/W,q[0]/W],[1-p[1]/H,1-q[1]/H],transform=fig.transFigure,color='black',lw=lw,ls='--' if dash else '-'))
        if native:commands.append(dict(type='line',points=points,arrow=arrow,dash=dash,lw=lw))
    def title(x,y,label,s):txt(x,y,label,8,True,native=label=='a');txt(x+4,y+.1,s,7,True,native=label=='a')
    def key(x,y,s,col):
        fig.add_artist(Line2D([x/W,(x+3)/W],[1-(y+1)/H]*2,color=col,lw=1.7));txt(x+4,y,s)
    # a: all geometry and text recorded as editable native objects.
    title(2,1,'a','Nested cross-event learning of signed expert value')
    for x,w,col in [(2,32,'#E8F1F2'),(38,42,'#EEEAF3'),(84,50,'#E8F1F2'),(138,40,'#F4EADF')]:rect(x,7,w,39 if x==84 else 38,col,lw=0,native=True)
    txt(4,8,'Event exclusion',bold=True,native=True);txt(40,8,'Inner expert cross-fitting',bold=True,native=True);txt(86,8,'Value learning and routing',bold=True,native=True);txt(140,8,'Outer-event evaluation',bold=True,native=True)
    for k in range(4):
        txt(12+k*5,12.5,f'E{k+1}',ha='center',native=True);txt(4,17+k*4.5,f'E{k+1}',native=True)
        for j in range(4):rect(10+j*5,16+k*4.5,4,3.5,C['value_gate'] if j==k else '#C9DDD8',lw=.35,native=True)
    rect(4,35,2.2,2,C['value_gate'],lw=0,native=True);txt(7,34.6,'Held out',native=True);txt(4,38,'Excluded from both\nexpert and head fitting',native=True)
    txt(40,13,'Two experts · 20 epochs',native=True)
    rect(40,18,15,6,'white',native=True);rect(60,18,18,6,'white',native=True);txt(47.5,19.5,'SAR',ha='center',native=True);txt(69,19.5,'Fusion',ha='center',native=True)
    rect(40,28,38,6,'white',native=True)
    line([(47.5,24),(47.5,28)],native=True);line([(69,24),(69,28)],native=True)
    txt(59,28.9,r'$v^*=(s-y)^2-(f-y)^2$',size=8,ha='center',native=True,plain='v* = (s − y)² − (f − y)²')
    txt(40,35,'Inner held-out predictions',native=True);txt(40,39,'+ Fusion benefit / − SAR benefit',native=True)
    line([(34,24),(38,24)],native=True);line([(78,31),(82,31),(82,22),(86,22)],dash=True,native=True)
    txt(86,13,'z: 6 bands + 7 expert / support maps',native=True)
    rect(86,16.5,46,9,'white',native=True);txt(109,17.5,'Spatial head · 13,417 parameters',ha='center',native=True);txt(109,20.6,r'$L_v=\mathrm{mean}_{\mathrm{valid}}(\hat v-v^*)^2$',size=8,ha='center',native=True,plain='Lv = meanvalid(v̂ − v*)²')
    rect(86,28,46,5.5,'white',native=True)
    txt(109,28.7,r'$\hat v=\tanh h_\theta(z),\quad g=\sigma(\hat v/0.05)$',size=8,ha='center',native=True,plain='v̂ = tanh hθ(z),  g = σ(v̂ / 0.05)')
    rect(86,36,46,5.5,'white',native=True)
    txt(109,37,r'$p=(1-g)s+gf$',size=8,ha='center',native=True,plain='p = (1 − g)s + gf')
    txt(86,42,'Optical unavailable: g = 0',native=True)
    line([(109,25.5),(109,28)],native=True);line([(109,33.5),(109,36)],native=True)
    line([(132,38.75),(136,38.75),(136,23),(140,23)],native=True)
    rect(140,16,36,13,'white',native=True);txt(158,17,'Outer experts + fitted head',ha='center',native=True);txt(158,21,'Predict excluded event E',ha='center',native=True);txt(158,25,'Four OOF folds → b–e',ha='center',native=True)
    rect(140,34,36,9,'white',native=True);txt(158,35,'Final development fit → freeze',ha='center',native=True);txt(158,39,'2023 core test → f',ha='center',native=True)
    txt(4,47,'E1  2020 flood season     E2  2021 Henan     E3  2024 April     E4  2024 June',native=True)
    txt(4,51,'Solid: prediction flow    Dashed: supervision    |    Head: 10 epochs, seed 17; event identity and age are not inputs',native=True)
    (AS/'panel_a_native_commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    z=np.load(SD/'joint_statistics.npz');be=z['b_edges'];ce=z['c_edges'];B=sum(z[f'b_{j}'] for j in range(4));CC=sum(z[f'c_{j}'] for j in range(4));CW=sum(z[f'benefit_sum_{j}'] for j in range(4));cal=sum(z[f'cal_{j}'] for j in range(4));summ=pd.read_csv(SD/'panel_bc_event_summary.csv')
    seq=LinearSegmentedColormap.from_list('area',['#F4F6F7','#A7C6D2','#446A9C','#15384C']);div=LinearSegmentedColormap.from_list('benefit',[C['negative'],'#FCFBF8',C['value_gate']])
    # b: all-pixel density, shared marginals and bin-wise observed-value calibration.
    title(2,57,'b','Predicted versus observed expert value')
    ba=ax(13,71,65,32);top=ax(13,65,65,4,sharex=ba);right=ax(80,71,7,32,sharey=ba)
    q=B/1e6;mesh=ba.pcolormesh(be,be,q.T,cmap=seq,norm=LogNorm(vmin=max(q[q>0].min(),.001),vmax=q.max()),rasterized=False,shading='flat')
    for aa in [ba,top]:aa.set_xscale('symlog',linthresh=.01,linscale=.5);aa.set_xlim(-1,1)
    for aa in [ba,right]:aa.set_yscale('symlog',linthresh=.01,linscale=.5);aa.set_ylim(-1,1)
    ba.plot([-1,1],[-1,1],'--',color='black',lw=.8);ba.axhline(0,color='black',lw=.6);ba.axvline(0,color='black',lw=.6)
    use=cal[:,0]>0;ba.plot(cal[use,1]/cal[use,0],cal[use,2]/cal[use,0],color=C['value_gate'],lw=1.6)
    top.stairs(B.sum(1)/B.sum()*100,be,fill=True,color=C['neutral']);right.stairs(B.sum(0)/B.sum()*100,be,orientation='horizontal',fill=True,color=C['neutral'])
    top.axis('off');right.axis('off');ba.set_xticks([-1,-.01,0,.01,1],labels=['−1','−.01','0','.01','1']);ba.set_yticks([-1,-.01,0,.01,1],labels=['−1','−.01','0','.01','1']);ba.set_xlabel('Predicted value v̂',labelpad=1);ba.set_ylabel('Observed value v*',labelpad=1)
    txt(13,61,'Area-weighted pixels · event-wise r = 0.28–0.82');txt(79,65,'Area %')
    cb=fig.colorbar(mesh,cax=ax(13,115,28,2),orientation='horizontal');cb.set_ticks([.01,1,100,10000]);cb.set_ticklabels(['.01','1','100','10⁴']);cb.ax.tick_params(pad=1);txt(44,114,'Area per bin (km²)');key(13,110,'Bin mean observed value',C['value_gate'])
    # c: routing density and genuine realized loss benefit occupy matched coordinates.
    title(99,57,'c','Routing behaviour and realized benefit')
    ca=ax(110,71,29,32);cr=ax(141,71,4,32,sharey=ca);ea=ax(151,71,26,32,sharey=ca)
    ckm=CC/1e6;cm=ca.pcolormesh(ce,ce,ckm.T,cmap=seq,norm=LogNorm(vmin=.001,vmax=ckm.max()),shading='flat')
    ben=np.divide(CW,CC,out=np.full_like(CC,np.nan),where=CC>0);lim=.1;em=ea.pcolormesh(ce,ce,ben.T,cmap=div,norm=TwoSlopeNorm(0,-lim,lim),shading='flat')
    cr.stairs(CC.sum(0)/CC.sum()*100,ce,orientation='horizontal',color=C['neutral'],fill=True);cr.axis('off')
    for aa in [ca,ea]:aa.set_xlim(0,1);aa.set_ylim(0,1);aa.set_xticks([0,.5,1]);aa.set_xlabel('|f − s|',labelpad=1)
    ca.set_yticks([0,.5,1]);ca.set_ylabel('Fusion routing weight g',labelpad=1);ea.tick_params(labelleft=False)
    txt(110,66,'Area density');txt(150,66,'Realized effect');txt(110,61,'Same OOF pixels; raw predictions')
    cb=fig.colorbar(cm,cax=ax(110,115,28,2),orientation='horizontal');cb.set_ticks([.01,1,100,10000]);cb.set_ticklabels(['.01','1','100','10⁴']);cb.ax.tick_params(pad=1)
    cb2=fig.colorbar(em,cax=ax(151,115,26,2),orientation='horizontal',extend='both');cb2.set_ticks([-lim,0,lim]);cb2.set_ticklabels([f'−{lim:.1f}','0',f'{lim:.1f}']);cb2.ax.tick_params(pad=1)
    txt(110,110,'Area (km²)');txt(149,109.5,'Brier(F) − Brier(G)')
    # d: one-to-one alignment of three-model IoU and signed effects vs fusion.
    title(2,124,'d','Leave-one-event-out generalization');df=pd.read_csv(SD/'panel_de_event_metrics.csv');eff=pd.read_csv(SD/'panel_de_paired_effects.csv').set_index('event_group').loc[EVENTS]
    da=ax(28,138,35,22);ha=ax(67,138,27,22)
    for i,event in enumerate(EVENTS):
        part=df[df.event_group==event].set_index('model');vs=[part.loc[m,'flood_iou'] for m in ['sar','fusion','value_gate']];da.plot(vs,[i]*3,color='black',lw=.7,zorder=1)
        for m,v0,mark in zip(['sar','fusion','value_gate'],vs,['o','s','D']):da.scatter(v0,i,s=14,color=C[m],marker=mark,edgecolor='white',linewidth=.25,zorder=3)
    da.set_yticks(range(4),EN);da.set_ylim(3.6,-.6);da.set_xlim(.15,.72);da.set_xticks([.2,.4,.6]);da.set_xlabel('Flood IoU',labelpad=1)
    # Effects have separate scales and are numerically annotated; normalized colour encodes each column's own symmetric range.
    mat=np.column_stack([eff.delta_iou*100,-eff.delta_brier*1000]);scales=np.max(abs(mat),axis=0);ha.imshow(mat/scales,aspect='auto',cmap=div,vmin=-1,vmax=1,extent=[-.5,1.5,3.5,-.5]);ha.set_ylim(3.6,-.6);ha.set_yticks([]);ha.set_xticks([0,1],['ΔIoU\n(pp)','−ΔBrier\n(×10³)']);ha.tick_params(axis='x',length=0,pad=2)
    for i in range(4):
        for j in range(2):ha.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.82,edgecolor='none',pad=.45))
    txt(66,133,'Value minus Fusion');key(3,130,'SAR',C['sar']);key(23,130,'Fusion',C['fusion']);key(48,130,'Value',C['value_gate'])
    txt(3,168,'Teal: improvement; each effect column scaled separately')
    # e: paired FN and FP changes, same event order and km² zero reference.
    title(99,124,'e','Error-area trade-offs across events');ee=ax(125,136,50,26)
    for i,event in enumerate(EVENTS):
        vals=[eff.loc[event,'delta_fn_km2'],eff.loc[event,'delta_fp_km2']];ee.plot(vals,[i,i],color='black',lw=.7)
        ee.scatter(vals[0],i,s=20,marker='v',color=C['value_gate'],zorder=3);ee.scatter(vals[1],i,s=19,marker='s',color=C['negative'],zorder=3)
        for k,v0 in enumerate(vals):ee.annotate(f'{v0:+.2f}',(v0,i),xytext=(0,(-5 if k==0 else 5)),textcoords='offset points',ha='center',va='center',fontsize=7,zorder=6,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    ee.axvline(0,color='black',lw=.8);ee.set_yticks(range(4),['E1','E2','E3','E4']);ee.set_ylim(3.7,-.7);extent=max(abs(eff.delta_fn_km2).max(),abs(eff.delta_fp_km2).max());ee.set_xlim(-extent*1.4,extent*1.4);ee.set_xlabel('Value − Fusion error area (km²)',labelpad=1)
    txt(104,130,'▼ ΔFN');txt(129,130,'■ ΔFP');txt(104,170,'Negative: less error area; positive: more')
    # f: frozen matched three-model metrics; labels on points expose close differences.
    title(2,175,'f','Frozen 2023 core test');key(56,175,'Fusion',C['fusion']);key(83,175,'Ordinary gate',C['ordinary_gate']);key(123,175,'Value gate',C['value_gate'])
    core=pd.read_csv(SD/'panel_f_frozen_metrics.csv')
    for region,x,lab in [('Beijing_Baoding',2,'Beijing–Baoding'),('Harbin',94,'Harbin')]:
        txt(x,180,lab,bold=True);r=core[core.region==region].set_index('model')
        for k,(met,name,fmt) in enumerate([('flood_iou','IoU','.4f'),('fn_km2','FN (km²)','.2f'),('fp_km2','FP (km²)','.2f')]):
            xx=x+26+k*20;txt(xx,180,name,ha='center')
            values=np.array([r.loc[m,met] for m in ['fusion','ordinary_gate','value_gate']]);lo,hi=values.min(),values.max();span=max(hi-lo,1e-6)
            fa=ax(xx-8,185,16,9)
            for j,(m,v0) in enumerate(zip(['fusion','ordinary_gate','value_gate'],values)):
                fa.plot([lo,v0],[j,j],color='black',lw=.5);fa.scatter(v0,j,s=12,color=C[m],zorder=3);fa.annotate(format(v0,fmt),(v0,j),xytext=(3,0),textcoords='offset points',va='center',fontsize=7)
            fa.set_xlim(lo-span*.1,hi+span*2.1);fa.set_ylim(2.7,-.7);fa.axis('off')
        for j,name in enumerate(['Fusion','Ordinary','Value']):txt(x,185+9*(j+.7)/3.4-1.15,name)
    txt(2,197,'Raw probabilities; p ≥ 0.5. OOF: four disaster groups. Core: models and decisions frozen before evaluation.')
    (AS/'render_metadata.json').write_text(json.dumps(dict(width_mm=W,height_mm=H,font=a.font,body_minimum_pt=7,panels=6,density='complete-population area per unequal-width value bin; symlog axes, linear core ±0.01',effect_colour_limits=[-lim,lim]),indent=2))
    fig.savefig(AS/f'Fig03_round{a.round}{suffix}.png',dpi=220)
    fig.savefig(AS/('panel_a'+suffix+'.svg'),bbox_inches=Bbox.from_extents(0,(H-55)/25.4,W/25.4,H/25.4),pad_inches=0)
    if a.final:
        stem=OUT/('Fig03_main'+suffix);fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.png',dpi=1200)
        with Image.open(str(stem)+'.png') as im:im.convert('RGB').save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig);print('Rendered',a.round,suffix,flush=True)
if __name__=='__main__':main()
