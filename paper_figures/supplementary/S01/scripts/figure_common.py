import os,json,argparse
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,to_rgba
from PIL import Image
C=dict(sar='#446A9C',optical='#9D8BB5',fusion='#80669D',value='#147D83',ordinary='#AC88AA',error='#BD5B4B',gold='#D5A34A',neutral='#B9C2CA',average='#637C77',random='#AA7D55',pretrained='#495C77')
EVENTS=['2020_flood_season','2021_Henan','2024_April','2024_June'];GROUPS=EVENTS+['2023_holdout'];GC=[C['value'],C['fusion'],C['sar'],C['neutral'],C['gold']]
GN=['2020 season','2021 Henan','2024 April','2024 June','2023 core'];EC=dict(zip(GROUPS,GC))
DIV=LinearSegmentedColormap.from_list('effect',[C['error'],'#FDFCF9',C['value']]);MODEL=['sar','fusion','probability_average','sigmoid_value','analytic_value','ridge_value_001'];MN=['SAR','Fusion','Probability average','Sigmoid Value','Analytic Value','Ridge Value'];MC=[C['sar'],C['fusion'],C['average'],C['value'],C['random'],C['ordinary']]
class Canvas:
    def __init__(self,n,h,args):
        parent=Path(os.path.abspath(__file__)).parents[1]
        self.n=n;self.original_height=h;self.compact_factor=0.96;self.h=h*self.compact_factor;self.w=180;self.args=args;self.out=Path(args.output) if args.output else (parent if parent.name==f'S{n:02d}' else parent/f'S{n:02d}');self.sd=self.out/'source_data';self.assets=self.out/'assets';self.assets.mkdir(exist_ok=True);self.roles=[];self.panel_bounds={}
        mpl.rcParams.update({'font.family':args.font,'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':1.15,'axes.edgecolor':'black','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.85,'ytick.major.width':.85,'xtick.major.size':2.3,'ytick.major.size':2.3,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','legend.frameon':False,'legend.fontsize':8,'mathtext.fontset':'custom','mathtext.rm':args.font,'mathtext.it':args.font+':italic','mathtext.bf':args.font+':bold','path.simplify':False})
        self.fig=plt.figure(figsize=(180/25.4,self.h/25.4))
    def y(self,v):return v*self.compact_factor
    def read(self,n):return pd.read_csv(self.sd/n)
    def js(self,n):return json.loads((self.sd/n).read_text(encoding='utf-8'))
    def text(self,x,y,s,size=8,ha='left',va='center',bold=False,color='black',**kw):return self.fig.text(x/180,1-self.y(y)/self.h,s,ha=ha,va=va,fontsize=size,fontweight='bold' if bold else 'normal',color=color,**kw)
    def title(self,x,y,p,s):self.text(x,y,p,9,bold=True);self.text(x+4,y,s,9,bold=True)
    def ax(self,x,y,w,h):
        a=self.fig.add_axes([x/180,1-self.y(y+h)/self.h,w/180,self.y(h)/self.h]);a.tick_params(pad=2);return a
    def box(self,x,y,w,h,color,alpha=1,edge='none',lw=.6,rounded=False):
        kw=dict(transform=self.fig.transFigure,facecolor=to_rgba(color,alpha),edgecolor=edge,linewidth=lw,zorder=-5)
        p=FancyBboxPatch((x/180,1-self.y(y+h)/self.h),w/180,self.y(h)/self.h,boxstyle='round,pad=0.001',**kw) if rounded else Rectangle((x/180,1-self.y(y+h)/self.h),w/180,self.y(h)/self.h,**kw);self.fig.add_artist(p);return p
    def line(self,x0,y0,x1,y1,c='black',lw=.7,ls='-'):
        p=Line2D([x0/180,x1/180],[1-self.y(y0)/self.h,1-self.y(y1)/self.h],color=c,lw=lw,ls=ls,transform=self.fig.transFigure);self.fig.add_artist(p);return p
    def arrow(self,x0,y0,x1,y1,dashed=False):
        a=FancyArrowPatch((x0/180,1-self.y(y0)/self.h),(x1/180,1-self.y(y1)/self.h),transform=self.fig.transFigure,arrowstyle='-|>',mutation_scale=6,linewidth=.7,color='black',linestyle='--' if dashed else '-',shrinkA=1.5,shrinkB=1.5);self.fig.add_artist(a);return a
    def gid(self,a,name):
        if hasattr(a,'set_gid'):a.set_gid(name);self.roles.append(name)
        else:
            for i,part in enumerate(a):part.set_gid(name+'_'+str(i));self.roles.append(name+'_'+str(i))
        return a
    def finish(self):
        for t in self.fig.findobj(mpl.text.Text):
            if t.get_fontsize()<8:t.set_fontsize(8)
        self.panel_bounds={k:[v[0],self.y(v[1]),v[2],self.y(v[3])] for k,v in self.panel_bounds.items()}
        suffix='_Arial' if self.args.font=='Arial' else '';stem=f'S{self.n:02d}_main'+suffix
        self.fig.savefig(self.assets/f'S{self.n:02d}_round{self.args.round}{suffix}.png',dpi=240);self.fig.savefig(self.assets/f'S{self.n:02d}_round{self.args.round}{suffix}.pdf')
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();bounds=[]
        not_drawn=set()
        for ax in self.fig.axes:
            if not ax.axison:
                not_drawn.update(ax.findobj(mpl.text.Text));continue
            for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                lo,hi=sorted(lim)
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo-1e-12<=tick.get_loc()<=hi+1e-12:not_drawn.update([tick.label1,tick.label2])
        for t in self.fig.findobj(mpl.text.Text):
            if t in not_drawn or not t.get_visible() or not t.get_text():continue
            b=t.get_window_extent(renderer).transformed(self.fig.transFigure.inverted());bounds.append(dict(text=t.get_text(),x0_mm=b.x0*180,x1_mm=b.x1*180,y0_mm=(1-b.y1)*self.h,y1_mm=(1-b.y0)*self.h,font_pt=t.get_fontsize(),figure_level=t.axes is None))
        axes=[]
        for ax in self.fig.axes:
            collections=[]
            for col in ax.collections:
                if not isinstance(col,mpl.collections.PathCollection):continue
                xy=np.asarray(col.get_offsets());finite=xy[np.isfinite(xy).all(axis=1)];lo,hi=sorted(ax.get_xlim());bot,top=sorted(ax.get_ylim())
                collections.append(dict(role=col.get_gid(),points=len(xy),outside=int(((finite[:,0]<lo)|(finite[:,0]>hi)|(finite[:,1]<bot)|(finite[:,1]>top)).sum())))
            axes.append(dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),aspect=ax.get_aspect(),collections=collections,images=[dict(shape=list(im.get_array().shape),extent=[float(v) for v in im.get_extent()],clim=[float(v) for v in im.get_clim()]) for im in ax.images]))
        (self.assets/('text_bounds'+suffix+'.json')).write_text(json.dumps(bounds,indent=2),encoding='utf-8');(self.assets/('render_metadata'+suffix+'.json')).write_text(json.dumps(dict(width_mm=180,height_mm=self.h,font=self.args.font,original_height_mm=self.original_height,compact_factor=self.compact_factor,background='white',body_minimum_pt=8,roles=self.roles,panels=self.panel_bounds,axes=axes),indent=2),encoding='utf-8')
        if self.args.final:
            self.fig.savefig(self.out/(stem+'.pdf'));self.fig.savefig(self.out/(stem+'.svg'));self.fig.savefig(self.out/(stem+'.png'),dpi=1200);Image.MAX_IMAGE_PIXELS=190_000_000
            with Image.open(self.out/(stem+'.png')) as im:im.convert('RGB').save(self.out/(stem+'.tiff'),compression='tiff_lzw',dpi=(1200,1200))
        plt.close(self.fig);print(stem,'round',self.args.round,'final',self.args.final,flush=True)
def args():
    p=argparse.ArgumentParser();p.add_argument('--font',default='Times New Roman',choices=['Times New Roman','Arial']);p.add_argument('--round',type=int,default=1);p.add_argument('--final',action='store_true');p.add_argument('--output');return p.parse_args()
