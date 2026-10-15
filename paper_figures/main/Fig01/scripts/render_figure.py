"""Reproducible Fig. 1 vector figure; derives no fitted model or threshold."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json, argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from PIL import Image
from pyproj import Geod
from scipy import ndimage

OUT=Path(__file__).resolve().parents[1]; SD=OUT/'source_data'; AS=OUT/'assets'
W,H=180.,220.
C={'sar':'#446A9C','fusion':'#80669D','value':'#147D83','harm':'#BD5B4B','review':'#D5A34A',
   'neutral':'#B9C2CA','teal':'#E8F1F2','lav':'#EEEAF3','sand':'#F4EADF'}
REGIONS=['Beijing_Baoding','Harbin'];SHORT=['BB','HB']; LABEL=['Beijing–Baoding','Harbin']
PANELS={'a':[1,1,44,218],'b':[48,1,131,37],'c':[48,40,131,55],
        'd':[48,97,131,32],'e':[48,132,85,55],'f':[133,132,46,54],
        'g':[48,190,68,29],'h':[120,190,59,29]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font',default='Times New Roman');parser.add_argument('--final',action='store_true');parser.add_argument('--round',default='1');args=parser.parse_args()
    font=args.font; suffix='' if font=='Times New Roman' else '_Arial'
    plt.rcParams.update({'font.family':font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':1.1,'axes.edgecolor':'black','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'xtick.direction':'in','ytick.direction':'in','xtick.major.width':.8,'ytick.major.width':.8,
        'xtick.major.size':2.2,'ytick.major.size':2.2,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','svg.hashsalt':'Fig01-frozen-093a72b5','path.simplify':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4)); commands=[]
    def ax(x,y,w,h,**kw):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H],**kw)
    def txt(x,y,s,size=7,weight='normal',ha='left',color='black',**kw):
        return fig.text(x/W,1-y/H,s,fontsize=max(size,7),fontweight=weight,ha=ha,va='top',color=color,**kw)
    def label(p,title):
        x,y,_,_=PANELS[p];txt(x,y,p,8,'bold');txt(x+4,y+.1,title,7,'bold')
    def rect(x,y,w,h,fill='none',lw=.7,dash=False,native=False):
        p=Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=fill,edgecolor='black' if lw else 'none',linewidth=lw,linestyle='--' if dash else '-',zorder=0)
        fig.add_artist(p)
        if native:commands.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,lw=lw,dash=dash))
    def line(points,lw=1.,dash=False,arrow=False,native=False):
        xx=np.array([p[0]/W for p in points]); yy=np.array([1-p[1]/H for p in points])
        if not arrow or len(xx)>2:
            fig.add_artist(Line2D(xx[:-1] if arrow else xx,yy[:-1] if arrow else yy,transform=fig.transFigure,color='black',lw=lw,ls='--' if dash else '-',zorder=2))
        if arrow:
            p=FancyArrowPatch((xx[-2],yy[-2]),(xx[-1],yy[-1]),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=5,lw=lw,linestyle='--' if dash else '-',shrinkA=0,shrinkB=0,color='black',zorder=3)
            fig.add_artist(p)
        if native:commands.append(dict(type='line',points=points,lw=lw,dash=dash,arrow=arrow))
    def image_at(array,x,y,w,h,name=None,native=False):
        aa=ax(x,y,w,h);aa.imshow(array,interpolation='nearest',aspect='auto');aa.axis('off')
        if native:
            path=AS/(name+'.png');Image.fromarray(array).save(path);commands.append(dict(type='image',x=x,y=y,w=w,h=h,path=str(path)))
        return aa
    def rgb(case,gray=False):
        raw=np.moveaxis(case['x'][2:5],0,-1).astype(float)
        valid=case['support']==7
        lo=np.percentile(raw[valid],2,axis=0);hi=np.percentile(raw[valid],98,axis=0)
        a=np.clip((raw-lo)/np.maximum(hi-lo,1),0,1)
        if gray:a=.45*a+.55*np.mean(a,axis=2,keepdims=True)
        a[~valid]=1
        return np.round(a*255).astype('uint8')
    def scale_geo(aa,bounds,km,loc=(.07,.08),color='black',size=7):
        l,r,b,t=bounds;lat=b+(t-b)*loc[1];lon=l+(r-l)*loc[0]
        lon2=Geod(ellps='WGS84').fwd(lon,lat,90,km*1000)[0]
        yy=lat;dy=(t-b)*.012
        aa.plot([lon,lon2],[yy,yy],color=color,lw=1.2)
        aa.plot([lon,lon],[yy-dy,yy+dy],color=color,lw=.8);aa.plot([lon2,lon2],[yy-dy,yy+dy],color=color,lw=.8)
        aa.text((lon+lon2)/2,yy+(t-b)*.027,f'{km:g} km',ha='center',va='bottom',fontsize=7,color=color,
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':.4})
    cases={r:np.load(SD/f'panel_c_{r}.npz') for r in REGIONS}
    case_meta={v['region']:v for v in json.loads((SD/'panel_c_selected_cases.json').read_text('utf-8'))}
    metrics=pd.read_csv(SD/'panel_d_h_map_metrics.csv')
    get=lambda r,m,col:float(metrics[(metrics.region==r)&(metrics.model==m)].iloc[0][col])
    # a. All structural commands are reused as native PowerPoint elements.
    label('a','Observation to review')
    rect(1,7,43,211,'white',0,native=True)
    txt(3,9,'SATELLITE OBSERVATIONS',7,'bold')
    txt(4,14,'GF-3',7,'bold');txt(25,14,'Sentinel-2',7,'bold')
    ca=cases[REGIONS[0]]
    image_at(rgb(ca),25,18,17,17,'workflow_optical',True)
    hh=ca['x'][0].astype(float);lo,hi=np.percentile(hh[ca['support']==7],[2,98]);hh=np.clip((hh-lo)/max(hi-lo,1),0,1)
    image_at(np.repeat((hh*255).astype('uint8')[:,:,None],3,axis=2),4,18,17,17,'workflow_sar',True)
    txt(4,36,'Post-flood\nHH / HV',7,linespacing=1.0);txt(25,36,'Pre-flood\nR / G / B / NIR',7,linespacing=1.0)
    # Optical feeds the fusion expert, SAR branches to both experts.
    line([(11.5,43),(11.5,48)],native=True,arrow=True)
    line([(11.5,45),(30,45),(30,48)],native=True,arrow=True)
    line([(35,43),(35,48)],native=True,arrow=True)
    rect(4,48,15,14,C['teal'],native=True);rect(25,48,17,14,C['lav'],native=True)
    txt(11.5,51,'SAR\nexpert',7,'bold',ha='center');txt(33.5,51,'Optical–SAR\nfusion expert',7,'bold',ha='center')
    line([(11.5,62),(11.5,65),(2.8,65),(2.8,113),(10,113),(10,115)],native=True,arrow=True)
    line([(33.5,62),(33.5,65),(43,65),(43,113),(35,113),(35,115)],native=True,arrow=True)
    line([(11.5,62),(11.5,69),(14,69),(14,74)],dash=True,native=True,arrow=True)
    line([(33.5,62),(33.5,69),(31,69),(31,74)],dash=True,native=True,arrow=True)
    rect(5,74,36,26,C['sand'],.7,True,True)
    txt(23,76,'Cross-fitted evidence value',7,'bold',ha='center')
    txt(23,81,'Training-only reference y',7,ha='center')
    txt(23,86,'v* = (pSAR − y)²\n       − (pFusion − y)²',7,ha='center',linespacing=1.25)
    txt(23,96,'Event-held-out predictions',6.6,ha='center')
    line([(23,100),(23,115)],dash=True,native=True,arrow=True)
    txt(25,104,'Learned\nvalue head',6.6)
    rect(5,115,36,18,C['teal'],native=True)
    txt(23,117,'Value-guided routing',7,'bold',ha='center')
    txt(23,121,'Predictions + available features',7,ha='center')
    txt(23,125,'g = sigmoid(v̂ / 0.05)\np = (1−g)pSAR + gpFusion',7,ha='center',linespacing=1.1)
    line([(23,133),(23,140)],native=True,arrow=True)
    rect(5,140,36,25,'white',native=True)
    pcol=colors.LinearSegmentedColormap.from_list('prob',['#FFFFFF',C['value']])
    pp=(pcol(ca['value_gate'])*255).astype('uint8');pp[ca['support']!=7]=255
    image_at(pp,7,142,18,20,'workflow_probability',True)
    txt(27,143,'Flood\nprobability\nmap',7,'bold')
    txt(27,157,'0 ≤ p ≤ 1',6.6)
    line([(23,165),(23,171)],native=True,arrow=True)
    rect(5,171,36,23,C['lav'],native=True)
    # Real probability image with exact 4x4 unit boundaries, not a generated risk field.
    image_at(pp,7,176,15,15,'workflow_units',True)
    for j in range(5):
        line([(7+j*15/4,176),(7+j*15/4,191)],.4,native=True)
        line([(7,176+j*15/4),(22,176+j*15/4)],.4,native=True)
    txt(23,173,'Omission risk\n& review units',7,'bold')
    txt(23,182,'64 × 64 pixels\nEntropy or\nlearned risk',6.6)
    line([(23,194),(23,201)],native=True,arrow=True)
    rect(5,201,36,15,C['sand'],native=True)
    txt(23,203,'Budgeted whole-unit review',7,'bold',ha='center')
    txt(23,208,'Ideal correction → residual FN\n10% of valid area',6.7,ha='center',linespacing=1.2)

    # b. Two georeferenced observations, imagery on actual observed support only.
    label('b','2023 core observations')
    for j,r in enumerate(REGIONS):
        g=np.load(SD/f'panel_b_{r}.npz');ext=g['extent'];x0=54+j*65
        aa=ax(x0,9.5,55,24)
        img=g['rgb'].copy();imf=img[:,:,:3]/255.;gray=imf.mean(2,keepdims=True);imf=.35*imf+.65*gray
        imf=.8*imf+.2
        aa.imshow(np.dstack([imf,img[:,:,3]/255.]),extent=ext,origin='upper',interpolation='nearest')
        ff=g['flood_fraction'];om=g['observed'];over=np.zeros((*ff.shape,4));over[:,:,:3]=colors.to_rgb(C['sar']);over[:,:,3]=np.sqrt(ff)*.9
        aa.imshow(over,extent=ext,origin='upper',interpolation='nearest')
        aa.contour(np.linspace(ext[0],ext[1],om.shape[1]),np.linspace(ext[3],ext[2],om.shape[0]),om.astype(float),[.5],colors='black',linewidths=.15)
        aa.set_aspect(1/np.cos(np.deg2rad((ext[2]+ext[3])/2)))
        aa.set_xlim(ext[:2]);aa.set_ylim(ext[2:])
        aa.xaxis.set_major_locator(plt.MaxNLocator(2));aa.yaxis.set_major_locator(plt.MaxNLocator(2))
        aa.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v:.0f}°E'));aa.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,pos:f'{v:.1f}°N'))
        aa.tick_params(labelsize=7,pad=1)
        m=case_meta[r];aa.scatter([m['lon']],[m['lat']],s=23,marker='s',facecolors='none',edgecolors='black',linewidths=.8,zorder=5)
        aa.annotate('Case '+str(j+1),(m['lon'],m['lat']),xytext=(5,6),textcoords='offset points',fontsize=7,
            bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':.5},arrowprops={'arrowstyle':'-','color':'black','lw':.6})
        scale_geo(aa,ext,30 if j==0 else 20,loc=(.05,.06),size=6)
        txt(x0,6,LABEL[j]+f' ({SHORT[j]})',7,'bold')
        valid_km=get(r,'sar','valid_area_km2');flood_km=get(r,'sar','true_flood_km2')
        txt(51+j*65,14,f'{valid_km:,.0f} km²\nvalid support\n\n{flood_km:,.0f} km²\nreference\nflood',7,linespacing=1.0)
    txt(51,36,'RGB context',6.5);rect(70,36.2,2,1.6,C['sar'],0);txt(73,36,'Reference flood',6.5)
    rect(99,36.2,2,1.6,'none',.6);txt(102,36,'Observed support / case windows',6.5)

    # c. Identical entire patch window in all four maps; neutral RGB underlay.
    label('c','Spatial cases of confident omission')
    rect(143.5,44.5,1.8,1.6,C['harm'],0)
    column_x=[62,90,118,146]
    for j,r in enumerate(REGIONS):
        dat=cases[r];ext=dat['bounds'];valid=dat['support']==7;yy=dat['label']==1;base=rgb(dat,True)
        top=48+j*23.5
        txt(49,top+5,SHORT[j]+'\nCase '+str(j+1),7,'bold')
        omission=yy&valid&(dat['fusion']<=.1)
        labs,nlabs=ndimage.label(omission,np.ones((3,3),int));aw=np.broadcast_to(dat['row_area_m2'][:,None],(256,256))
        areas=np.bincount(labs.ravel(),weights=np.where(omission,aw,0).ravel());areas[0]=0
        ij=np.argwhere(labs==areas.argmax());cy,cx=np.round(ij.mean(0)).astype(int)
        side=max(32,int(np.max(np.ptp(ij,axis=0)))+12);side=min(96,side)
        r0=int(np.clip(cy-side//2,0,256-side));c0=int(np.clip(cx-side//2,0,256-side))
        roi=[ext[0]+c0*(ext[1]-ext[0])/256,ext[0]+(c0+side)*(ext[1]-ext[0])/256,
             ext[3]-(r0+side)*(ext[3]-ext[2])/256,ext[3]-r0*(ext[3]-ext[2])/256]
        for k,(name,col) in enumerate(zip(['Reference','SAR','Fusion','Omission'],[C['sar'],C['sar'],C['fusion'],C['harm']])):
            aa=ax(column_x[k],top,21,21)
            aa.imshow(base,extent=ext,origin='upper',interpolation='nearest',aspect='equal')
            if k==0:mask=yy&valid
            elif k==1:mask=(dat['sar']>=.5)&valid
            elif k==2:mask=(dat['fusion']>=.5)&valid
            else:mask=yy&(dat['fusion']<=.1)&valid
            ov=np.zeros((256,256,4));ov[:,:,:3]=colors.to_rgb(col);ov[:,:,3]=mask*.8
            aa.imshow(ov,extent=ext,origin='upper',interpolation='nearest',aspect='equal')
            aa.add_patch(Rectangle((roi[0],roi[2]),roi[1]-roi[0],roi[3]-roi[2],fill=False,edgecolor='white',linewidth=1.55))
            aa.add_patch(Rectangle((roi[0],roi[2]),roi[1]-roi[0],roi[3]-roi[2],fill=False,edgecolor='black',linewidth=.65))
            if k==3:
                zoom=ax(168,top+6,10,10)
                zoom.imshow(base[r0:r0+side,c0:c0+side],extent=roi,origin='upper',interpolation='nearest')
                zoom.imshow(ov[r0:r0+side,c0:c0+side],extent=roi,origin='upper',interpolation='nearest')
                zoom.set_xticks([]);zoom.set_yticks([])
                for sp in zoom.spines.values():sp.set_visible(True);sp.set_linewidth(.6)
                txt(168,top+2,'Zoom',6)
                rx=column_x[k]+21*(roi[1]-ext[0])/(ext[1]-ext[0])
                ry=top+21*(ext[3]-(roi[2]+roi[3])/2)/(ext[3]-ext[2])
                line([(rx,ry),(168,top+11)],.55)
                scale_geo(zoom,roi,.1,loc=(.08,.08),size=5.2)
            aa.set_xticks([]);aa.set_yticks([])
            for sp in aa.spines.values():sp.set_visible(True);sp.set_linewidth(.5)
            if j==0:txt(column_x[k]+10.5,44,name,6.7,ha='center',linespacing=.9)
            if k==0:scale_geo(aa,ext,.5,size=6)
        if j==1:txt(62,93,'Common window; reference y = 1 and fusion p ≤ 0.1 defines confident omission.',6.3)

    # d. Independent scales, same region rows; colour encodes model across all axes.
    label('d','Higher accuracy coexists with more confident omissions')
    specs=[(61,33,'flood_iou',1,'IoU',[.3,.7],['sar','fusion']),
           (105,29,'confident_omission_01',100,'Omission (%)',[0,12],['sar','fusion']),
           (148,27,'confident_omission_01',100,'Omission (%)',[0,8],['sar_temperature','fusion_temperature'])]
    for k,(xx,ww,col,factor,xlabel,xlim,mods) in enumerate(specs):
        aa=ax(xx,108,ww,11)
        for j,r in enumerate(REGIONS):
            ss,ff=[get(r,m,col)*factor for m in mods];yy=1-j
            aa.plot([ss,ff],[yy,yy],color='black',lw=.7)
            aa.scatter(ss,yy,s=17,c=C['sar'],marker='o',zorder=3);aa.scatter(ff,yy,s=18,c=C['fusion'],marker='D',zorder=3)
            for val,dy in [(ss,-.28),(ff,.23)]:aa.text(val,yy+dy,f'{val:.4f}' if k==0 else f'{val:.2f}',ha='center',va='center',fontsize=7)
        aa.set_ylim(-.55,1.55);aa.set_xlim(xlim);aa.set_yticks([1,0]);aa.set_yticklabels(SHORT if k==0 else [])
        aa.set_xticks([.3,.5,.7] if k==0 else ([0,6,12] if k==1 else [0,4,8]));aa.tick_params(pad=1)
        aa.set_xlabel(xlabel,labelpad=1)
        txt(xx,103,['Map agreement','Raw probability','Dev.-calibrated'][k],7,'bold')
    # Shared colour/shape key for d-f; no repeated legends inside plots.
    txt(52,127,'SAR',6.5,color='black');txt(70,127,'Fusion',6.5,color='black')
    fig.add_artist(Line2D([49/W],[1-128/H],transform=fig.transFigure,marker='o',color=C['sar'],ms=3,ls='none'))
    fig.add_artist(Line2D([67/W],[1-128/H],transform=fig.transFigure,marker='D',color=C['fusion'],ms=3,ls='none'))
    txt(95,127,'Omission denominator: reference-flooded area',6.5)

    # e. Full area-weighted histogram, top/right marginals share the same edges.
    label('e','Confidence–loss discordance')
    dat=np.load(SD/'panel_e_pooled.npz');hist=dat['area_m2'];xe=dat['xedges'];ye=dat['yedges'];pct=100*hist/hist.sum()
    joint=ax(60,148,51,30);topax=ax(60,139,51,7,sharex=joint);rightax=ax(113,148,8,30,sharey=joint)
    density=colors.LinearSegmentedColormap.from_list('density',['#E8EDF3','#B6C9DD','#7D9EBC','#446A9C','#20364E'])
    norm=colors.LogNorm(vmin=.0001,vmax=max(1.,pct.max()))
    joint.axvspan(-np.log(2),0,color='#FAEFEC',zorder=-3)
    mesh=joint.pcolormesh(xe,ye,np.ma.masked_where(pct.T<=0,pct.T),cmap=density,norm=norm,rasterized=False,shading='flat',linewidth=0)
    # A single area-mass contour adds reference-flood context without changing the all-valid density.
    hf=dat['flood_area_m2'];hh=hf.ravel();order=np.argsort(hh)[::-1];mass=np.cumsum(hh[order])/hh.sum();lev=hh[order][np.searchsorted(mass,.8)]
    joint.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,hf.T,[lev],colors='black',linewidths=.5,linestyles='--')
    joint.axhline(0,color='black',lw=.75);joint.axvline(0,color='black',lw=.75)
    joint.set_xlim(xe[[0,-1]]);joint.set_ylim(-1,1);joint.set_xticks([-.6,0,.6]);joint.set_yticks([-1,0,1])
    joint.set_xlabel('ΔEntropy (nats)',labelpad=1);joint.set_ylabel('ΔBrier',labelpad=1);joint.tick_params(pad=1)
    shares=pd.read_csv(SD/'panel_e_quadrant_shares.csv');pool=shares[shares.region=='Pooled_area_weighted'].iloc[0]
    joint.text(-.66,.92,f'Lower entropy\nHigher loss\n{pool.quadrant_share_all_valid*100:.2f}% valid area\n{pool.quadrant_share_reference_flood*100:.2f}% flood area',fontsize=7,va='top',linespacing=1.1,bbox={'facecolor':'#FAEFEC','edgecolor':'none','pad':1})
    joint.text(.035,.92,'Higher entropy\nHigher loss',fontsize=7,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1})
    joint.text(-.66,-.92,'Lower entropy\nLower loss',fontsize=7,va='bottom',bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1})
    joint.text(.035,-.92,'Higher entropy\nLower loss',fontsize=7,va='bottom',bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1})
    topax.stairs(pct.sum(1),xe,color=C['neutral'],fill=True,lw=.4);rightax.stairs(pct.sum(0),ye,orientation='horizontal',color=C['neutral'],fill=True,lw=.4)
    topax.set_yscale('log');topax.set_ylim(.001,100);topax.set_yticks([.01,1,100],labels=['.01','1','100']);topax.tick_params(labelbottom=False,labelsize=7,pad=1)
    rightax.set_xscale('log');rightax.set_xlim(.001,100);rightax.set_xticks([.01,100],labels=['.01','100']);rightax.tick_params(labelleft=False,labelsize=7,pad=1)
    topax.set_ylabel('%',rotation=0,labelpad=1,fontsize=7);rightax.set_xlabel('%',labelpad=0,fontsize=7)
    txt(60,135.5,'Δ = Fusion − SAR; pooled valid area',7)
    cbax=ax(124,148,1.6,30);cb=fig.colorbar(mesh,cax=cbax,ticks=[.0001,.01,1]);cb.ax.set_yticklabels(['.0001','.01','1']);cb.ax.tick_params(labelsize=7,pad=1,length=1.5);cb.set_label('Area / bin (%)',fontsize=7,labelpad=1)

    txt(60,184,'Dashed contour: 80% reference-flood area (binned)',6.2)
    # f. Exact area-weighted CDF at retained steps, each region kept separate.
    label('f','Flood-probability tails')
    ec=pd.read_csv(SD/'panel_f_ecdf_display.csv')
    a_full=ax(142,141,33,16);a_tail=ax(142,166,33,14)
    # Display each region with line style; no smoothing or density fitting.
    for j,r in enumerate(REGIONS):
        for m in ['sar','fusion']:
            data=ec[(ec.region==r)&(ec.model==m)]
            ls='-' if j==0 else (0,(2,1.2))
            for aa in [a_full,a_tail]:aa.step(data.probability,100*data.ecdf,where='post',color=C[m],lw=1.4,ls=ls)
    for aa in [a_full,a_tail]:
        aa.axvline(.1,color='black',lw=.7,ls=':');aa.tick_params(pad=1)
    a_full.set_xlim(0,1);a_full.set_ylim(0,100);a_full.set_xticks([0,.5,1]);a_full.set_yticks([0,50,100]);a_full.set_ylabel('Flood area (%)',labelpad=1)
    a_full.set_xlabel('Flood probability',labelpad=0)
    a_tail.set_xlim(0,.2);a_tail.set_ylim(0,16);a_tail.set_xticks([0,.1,.2]);a_tail.set_yticks([0,8,16]);a_tail.set_ylabel('Flood area (%)',labelpad=1)
    a_tail.set_xlabel('Flood probability (tail)',labelpad=0)
    txt(141,137,'BB solid; HB dashed',6.5)
    txt(148,163,'p = 0.1',6.5)

    # g. Only the frozen 10km pointwise intervals, within-region composition.
    label('g','Paired spatial effects')
    intervals=pd.read_csv(SD/'panel_g_intervals.csv')
    for k,(metric,xx,ww,limits,ticks,xlabel) in enumerate([
        ('flood_iou',58,23,(-.01,.34),[0,.15,.3],'ΔIoU'),
        ('confident_omission_01',91,22,(-.2,6.2),[0,3,6],'ΔOmission (pp)')]):
        aa=ax(xx,200,ww,10)
        for j,r in enumerate(REGIONS):
            row=intervals[(intervals.region==r)&(intervals['first']=='fusion')&(intervals.metric==metric)].iloc[0];factor=1 if k==0 else 100
            p,lo,hi=[row[z]*factor for z in ['difference_first_minus_second','ci025','ci975']]
            aa.errorbar(p,1-j,xerr=np.array([[p-lo],[hi-p]]),fmt='o',color='black',mfc=C['fusion'] if k==0 else C['harm'],mec='black',ms=3.5,lw=1,capsize=2)
        aa.axvline(0,color='black',ls=':',lw=.7);aa.set_ylim(-.6,1.6);aa.set_xlim(limits);aa.set_xticks(ticks)
        aa.set_yticks([1,0]);aa.set_yticklabels(SHORT if k==0 else []);aa.tick_params(pad=1);aa.set_xlabel(xlabel,labelpad=0)
    txt(53,195,'Fusion − SAR; 10 km paired blocks',6.5)
    txt(49,217,'95% pointwise intervals; within-region composition',6.2)

    # h. Compact inversion preview with actual absolute endpoints, independent axes.
    label('h','Review-stage inversion')
    txt(121,195,'10% valid-area review budget',6.5)
    review=pd.read_csv(SD/'panel_h_review.csv')
    for k,(xx,ww,limits,ticks,xlab) in enumerate([(130,18,(.61,.68),[.62,.66],'IoU'),(156,20,(35,120),[40,80,120],'Residual FN (km²)')]):
        aa=ax(xx,202,ww,8)
        for j,r in enumerate(REGIONS):
            if k==0:values=[get(r,m,'flood_iou') for m in ['fusion','value_gate']]
            else:values=[float(review[(review.region==r)&(review.model==m)].iloc[0].residual_fn_km2) for m in ['fusion','value_gate']]
            aa.plot(values,[1-j,1-j],color='black',lw=.75)
            aa.scatter(values,[1-j,1-j],c=[C['fusion'],C['value']],s=15,marker='D',zorder=3)
            if k==1:aa.text(np.clip(np.mean(values),52,106),1-j+.35,f'+{values[1]-values[0]:.2f}',fontsize=7,ha='center',va='center')
        aa.set_ylim(-.5,1.6);aa.set_xlim(limits);aa.set_xticks(ticks);aa.set_yticks([1,0]);aa.set_yticklabels(SHORT if k==0 else []);aa.tick_params(pad=1,labelsize=7);aa.set_xlabel(xlab,labelpad=0,fontsize=7)
    for xx,col in [(123,C['fusion']),(147,C['value'])]:fig.add_artist(Line2D([xx/W],[1-199/H],transform=fig.transFigure,marker='D',color=col,ms=3,ls='none'))
    txt(126,198,'Fusion',7);txt(150,198,'Value gate',7)
    txt(121,217,'Ideal correction simulation; mean entropy',6.2)

    fig.canvas.draw()
    # Full vector files retain true data and text. Independent panel assets share exact page geometry.
    stem='Fig01_main'+suffix
    fig.savefig(OUT/(stem+'.svg'),dpi=450)
    fig.savefig(OUT/(stem+'.pdf'),dpi=450)
    preview=AS/f'Fig01_round{args.round}{suffix}.png';fig.savefig(preview,dpi=220)
    if not suffix:
        for p,(xx,yy,ww,hh) in PANELS.items():
            bbox=Bbox.from_bounds(xx/25.4,(H-yy-hh)/25.4,ww/25.4,hh/25.4)
            fig.savefig(AS/f'panel_{p}.svg',bbox_inches=bbox,dpi=450)
            fig.savefig(AS/f'panel_{p}.png',bbox_inches=bbox,dpi=350)
        (AS/'workflow_native_commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
        (AS/'panel_layout_mm.json').write_text(json.dumps({'width':W,'height':H,'panels':PANELS},indent=2),encoding='utf-8')
        texts=[(obj,obj.get_color()) for obj in fig.findobj(match=Text)]
        for obj,_ in texts:obj.set_color((0,0,0,0))
        for p,(xx,yy,ww,hh) in PANELS.items():
            if p=='a':continue
            bbox=Bbox.from_bounds(xx/25.4,(H-yy-hh)/25.4,ww/25.4,hh/25.4)
            fig.savefig(AS/f'panel_{p}_geometry.svg',bbox_inches=bbox,dpi=450)
            fig.savefig(AS/f'panel_{p}_geometry.png',bbox_inches=bbox,dpi=450)
            ET.register_namespace('','http://www.w3.org/2000/svg');ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
            tree=ET.parse(AS/f'panel_{p}_geometry.svg')
            for parent in tree.iter():
                for child in list(parent):
                    if child.tag=='{http://www.w3.org/2000/svg}text':parent.remove(child)
            tree.write(AS/f'panel_{p}_geometry.svg',encoding='utf-8',xml_declaration=True)
        for obj,colour in texts:obj.set_color(colour)
    if args.final:
        fig.savefig(OUT/(stem+'.png'),dpi=1200)
        with Image.open(OUT/(stem+'.png')) as im:
            im.convert('RGB').save(OUT/(stem+'.tiff'),compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig)
    print('Rendered',preview,flush=True)

if __name__=='__main__':main()
