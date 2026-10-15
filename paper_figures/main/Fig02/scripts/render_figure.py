"""Six-panel geospatial Fig. 2; quantitative graphics remain vector and images lossless."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap,TwoSlopeNorm,to_rgb
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image
from pyproj import Geod
from scipy import ndimage

OUT=Path(__file__).resolve().parents[1];SD=OUT/'source_data';AS=OUT/'assets';W,H=180,195
C={'sar':'#446A9C','fusion':'#80669D','conf':'#BD5B4B','neutral':'#B9C2CA','recovered':'#7EADA3','persistent':'#B3A4BD','stable':'#E8EEF2'}
PANELS={'a':[1,1,105,88],'b':[110,1,69,39],'c':[110,43,69,45],'d':[1,92,178,49],'e':[1,146,88,48],'f':[95,146,84,48]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--font',default='Times New Roman');ap.add_argument('--round',default='1');ap.add_argument('--final',action='store_true');args=ap.parse_args();suffix='' if args.font=='Times New Roman' else '_Arial'
    plt.rcParams.update({'font.family':args.font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.1,'axes.edgecolor':'black','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.8,'ytick.major.width':.8,'xtick.major.size':2,'ytick.major.size':2,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'Fig02-spatial-frozen','path.simplify':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    def ax(x,y,w,h,**kw):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H],**kw)
    def txt(x,y,s,size=7,bold=False,ha='left',**kw):return fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='top',color='black',**kw)
    def title(p,s):
        x,y,_,_=PANELS[p];txt(x,y,p,8,True);txt(x+4,y+.1,s,7,True)
    def key(x,y,label,color,ls=None):
        if ls:fig.add_artist(Line2D([x/W,(x+3)/W],[1-(y+1.3)/H]*2,color=color,lw=1.5,ls=ls))
        else:fig.add_artist(Rectangle((x/W,1-(y+2)/H),2/W,1.6/H,transform=fig.transFigure,facecolor=color,edgecolor='none'))
        txt(x+4,y,label)
    def scale(aa,extent,km,loc=(.08,.09)):
        l,r,b,t=extent;lon=l+(r-l)*loc[0];lat=b+(t-b)*loc[1];end=Geod(ellps='WGS84').fwd(lon,lat,90,km*1000)[0];dy=(t-b)*.012
        aa.plot([lon,end],[lat,lat],color='black',lw=1.2);aa.plot([lon,lon],[lat-dy,lat+dy],color='black',lw=.8);aa.plot([end,end],[lat-dy,lat+dy],color='black',lw=.8)
        aa.text((lon+end)/2,lat+2*dy,f'{km:g} km',fontsize=7,ha='center',va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.6))
    z=np.load(SD/'panel_a_geographic_window.npz');extent=z['extent'];truth=z['label']==1;valid=z['label']!=255;s=z['sar'];f=z['fusion'];conf=truth&(f<=.1);sarfn=truth&(s<.5);fusfn=truth&(f<.5)
    raw=np.moveaxis(z['x'][2:5],0,-1).astype(float);lo=np.percentile(raw[valid],2,axis=0);hi=np.percentile(raw[valid],98,axis=0);rgb=np.clip((raw-lo)/np.maximum(hi-lo,1),0,1);rgb=.25*rgb+.75*rgb.mean(2,keepdims=True);rgb=.72*rgb+.28;rgb[~valid]=1
    overlay=np.zeros((*truth.shape,4));overlay[sarfn]=[*to_rgb(C['sar']),.65];overlay[fusfn]=[*to_rgb(C['fusion']),.66];overlay[conf]=[*to_rgb(C['conf']),.9]
    # a — continuous native-grid window, exact shared-coordinate marginals and two zooms.
    title('a','Geographic structure of omissions');txt(5,5.3,'Beijing–Baoding | native-grid window')
    map_width=64*np.cos(np.deg2rad(np.mean(extent[2:])))
    ma=ax(12,20,map_width,64);ma.imshow(rgb,extent=extent,origin='upper',interpolation='nearest',aspect='auto');ma.imshow(overlay,extent=extent,origin='upper',interpolation='nearest',aspect='auto')
    xx=np.linspace(extent[0],extent[1],1280,endpoint=False)+(extent[1]-extent[0])/2560;yy=np.linspace(extent[3],extent[2],1280,endpoint=False)-(extent[3]-extent[2])/2560
    ma.contour(xx,yy,truth,levels=[.5],colors='black',linewidths=.22);ma.contour(xx,yy,valid,levels=[.5],colors='black',linewidths=.55)
    lat=float(pd.read_csv(SD/'panel_d_transect.csv').latitude.iloc[0]);ma.plot(extent[:2],[lat,lat],ls='--',lw=.8,color='black');ma.text(extent[0]+.002,lat,'A',ha='left',va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=.2));ma.text(extent[1]-.002,lat,"A′",ha='right',va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=.2))
    scale(ma,extent,2);ma.set_xlim(extent[:2]);ma.set_ylim(extent[2:]);ma.set_xticks(np.linspace(extent[0],extent[1],3)[1:]);ma.set_xticklabels([f'{v:.2f}°E' for v in ma.get_xticks()]);ma.set_yticks(np.linspace(extent[2],extent[3],4)[1:-1]);ma.set_yticklabels([f'{v:.2f}°N' for v in ma.get_yticks()],rotation=90,va='center');ma.tick_params(pad=1)
    marg=pd.read_csv(SD/'panel_a_marginals.csv');top=ax(12,11,map_width,7,sharex=ma);right=ax(65,20,8,64,sharey=ma)
    for m,color in [('sar_fn',C['sar']),('fusion_fn',C['fusion']),('fusion_conf',C['conf'])]:
        for axis,aa in [('longitude',top),('latitude',right)]:
            q=marg[marg.axis==axis];cent=(q.lower+q.upper)/2;v=q[m+'_km2']
            if axis=='longitude':aa.plot(cent,v,color=color,lw=1)
            else:aa.plot(v,cent,color=color,lw=1)
    top.tick_params(labelbottom=False);right.tick_params(labelleft=False,pad=1);top.set_ylabel('km²',labelpad=0);txt(65,16,'km²');right.set_xticks([round(right.get_xlim()[1],1)])
    ma.get_xticklabels()[-1].set_ha('right')
    # Fixed local zooms: selected component's centre and largest other confident component in the window.
    labs,n=ndimage.label(conf,np.ones((3,3)));sizes=np.bincount(labs.ravel());sizes[0]=0;selected_mask=z['selected_component'];rr,cc=np.nonzero(selected_mask);centres=[(float(rr.mean()),float(cc.mean()))]
    used=set(np.unique(labs[selected_mask]));order=np.argsort(sizes)[::-1];other=None
    for candidate in order:
        if candidate==0 or candidate in used:continue
        cr,cc2=np.nonzero(labs==candidate)
        if np.hypot(cr.mean()-centres[0][0],cc2.mean()-centres[0][1])>=384:other=int(candidate);break
    if other:
        r2,c2=np.nonzero(labs==other);centres.append((float(r2.mean()),float(c2.mean())))
    else:centres.append((float(rr[np.argmax(rr)]),float(cc[np.argmax(rr)])))
    zoom_records=[]
    for j,(cy,cx) in enumerate(centres):
        ri=int(np.clip(cy-96,0,1088));ci=int(np.clip(cx-148,0,984));ze=[xx[ci]-(xx[1]-xx[0])/2,xx[ci+295]+(xx[1]-xx[0])/2,yy[ri+191]-(yy[0]-yy[1])/2,yy[ri]+(yy[0]-yy[1])/2]
        zoom_width=23*(296/192)*np.cos(np.deg2rad(np.mean(ze[2:])))
        zm=ax(78,22+j*32,zoom_width,23);cut=np.s_[ri:ri+192,ci:ci+296];zm.imshow(rgb[cut],extent=ze,aspect='auto',interpolation='nearest');zm.imshow(overlay[cut],extent=ze,aspect='auto',interpolation='nearest');zm.contour(xx[ci:ci+296],yy[ri:ri+192],truth[cut],[.5],colors='black',linewidths=.35);zm.set_xticks([]);zm.set_yticks([]);scale(zm,ze,.3)
        ma.add_patch(Rectangle((ze[0],ze[2]),ze[1]-ze[0],ze[3]-ze[2],fill=False,color='white',lw=1.7));ma.add_patch(Rectangle((ze[0],ze[2]),ze[1]-ze[0],ze[3]-ze[2],fill=False,color='black',lw=.6));txt(78,18.4+j*32,f'Window {j+1}')
        # Known figure geometry makes connectors survive cropped panel exports.
        ymap=20+64*(extent[3]-(ze[2]+ze[3])/2)/(extent[3]-extent[2]);xmap=12+map_width*(ze[1]-extent[0])/(extent[1]-extent[0]);fig.add_artist(Line2D([xmap/W,76/W,78/W],[1-ymap/H,1-ymap/H,1-(33.5+j*32)/H],color='black',lw=.45,transform=fig.transFigure))
        zoom_records.append(dict(window=j+1,rows=[ri,ri+192],cols=[ci,ci+296],extent=ze))
    key(77,80,'SAR FN',C['sar']);key(77,84,'Fusion FN',C['fusion']);key(5,88,'Confident FN',C['conf']);txt(44,88,'Black: reference / footprint')
    # b — omission-area survival curve with number and area-contribution strips.
    title('b','Connected omission scales');comp=pd.read_csv(SD/'panel_b_components.csv.gz');summary=[]
    ba=ax(119,10,57,17)
    for region,col,style,short in [('Beijing_Baoding',C['conf'],'-','BB'),('Harbin',C['conf'],'--','HB')]:
        q=comp[(comp.region==region)&(comp.connectivity==8)].sort_values('area_m2');area=q.area_m2.to_numpy()/1e4;surv=100*(np.cumsum(area[::-1])[::-1]/area.sum());ba.step(area,surv,where='pre',color=col,lw=1.5,ls=style)
        summary.append(dict(region=region,n=len(q),largest_ha=area.max(),fraction_area_ge_1ha=float(area[area>=1].sum()/area.sum()),censored_fraction=float(q.loc[q.touches_unknown,'area_m2'].sum()/q.area_m2.sum())))
    ba.set_xscale('log');ba.set_ylim(0,102);ba.set_yticks([0,50,100]);ba.set_ylabel('Omitted area ≥ size (%)',labelpad=1);ba.set_xlabel('Component area (ha)',labelpad=0);ba.tick_params(pad=1)
    key(119,5,'BB',C['conf'],'-');key(145,5,'HB',C['conf'],'--')
    txt(159,5,'8-neighbour')
    for j,rec in enumerate(summary):
        txt(111,34+j*3.2,f"{['BB','HB'][j]}: n={rec['n']:,}; ≥1 ha: {rec['fraction_area_ge_1ha']*100:.1f}%")
    # c — full paired state matrix; numbers are not inferred from marginal FN totals.
    title('c','Detection-state transfers');tc=pd.read_csv(SD/'panel_c_transitions.csv');states=['stable_detection','newly_missed','recovered','persistent_omission'];cols=[C['stable'],C['conf'],C['recovered'],C['persistent']]
    txt(117,48,'Fusion: detected / missed')
    for j,(region,short) in enumerate([('Beijing_Baoding','BB'),('Harbin','HB')]):
        q=tc[tc.region==region].set_index('state');ca=ax(119+j*31,55,27,20);ca.imshow(np.arange(4).reshape(2,2),cmap=ListedColormap(cols),vmin=-.5,vmax=3.5,aspect='auto')
        for k,state in enumerate(states):
            row=q.loc[state];ca.text(k%2,k//2,f'{row.area_m2/1e6:.1f}\n{row.fraction_flood*100:.1f}%',ha='center',va='center',fontsize=7)
        ca.set_xticks([]);ca.set_yticks([0,1]);ca.set_yticklabels(['TP','FN'] if j==0 else []);ca.tick_params(length=0,pad=2);ca.set_title(short,pad=2,fontsize=7);ca.spines[['left','bottom']].set_visible(False)
        delta=(q.loc['newly_missed'].area_m2-q.loc['recovered'].area_m2)/1e6;txt(119+j*31,76.4,f'ΔFN {delta:+.2f} km²')
    txt(110,59,'SAR',ha='center',rotation=90);txt(111,81,'ΔFN = newly missed − recovered');txt(113,85,'Cells: km² / reference flood %')
    # d — every layer follows the same real horizontal path; NaNs preserve gaps.
    title('d','A–A′: aligned observations and prediction response');tr=pd.read_csv(SD/'panel_d_transect.csv');distance=tr.distance_km.to_numpy();reference=tr.reference.to_numpy()==1;validtr=tr.valid.to_numpy(bool);cl=tr.fusion_confident.to_numpy(bool)
    da=[]
    for y0,hh in [(99,6),(107,5),(114,1.8),(118,7),(127,5),(135,3)]:da.append(ax(19,y0,157,hh))
    for aa in da:
        aa.set_xlim(distance[0],distance[-1]);aa.tick_params(labelbottom=False,pad=1);aa.spines['right'].set_visible(False);aa.spines['top'].set_visible(False)
        starts=np.flatnonzero(np.diff(np.r_[False,cl,False].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[False,cl,False].astype(int))==-1)
        for st,en in zip(starts,ends):aa.axvspan(distance[st],distance[min(en,len(distance)-1)],color=C['conf'],alpha=.10,lw=0,zorder=0)
    da[0].plot(distance,tr.HH,color='black',lw=.65);da[0].plot(distance,tr.HV,color=C['neutral'],lw=.65);da[0].set_ylim(0,255);da[0].set_yticks([0,255]);txt(3,100,'SAR\nDN')
    da[1].plot(distance,tr.NIR,color=C['fusion'],lw=.7);da[1].set_ylim(0,255);da[1].set_yticks([0,255]);txt(3,107.5,'NIR\nDN')
    strip=np.where(~validtr,0,np.where(reference,2,1))[None,:];da[2].imshow(strip,extent=[distance[0],distance[-1],0,1],aspect='auto',cmap=ListedColormap(['#FFFFFF','#E8EEF2','#446A9C']),vmin=0,vmax=2,interpolation='nearest');da[2].set_yticks([]);da[2].set_ylabel('Ref.',rotation=0,labelpad=3,va='center');da[2].spines[['left','bottom']].set_visible(False)
    da[3].plot(distance,tr.p_sar,color=C['sar'],lw=1);da[3].plot(distance,tr.p_fusion,color=C['fusion'],lw=1);da[3].axhline(.5,color='black',lw=.5,ls=':');da[3].set_ylim(0,1);da[3].set_yticks([0,1]);da[3].set_ylabel('p(flood)',labelpad=1)
    da[4].fill_between(distance,0,tr.delta_probability,where=tr.delta_probability>=0,color=C['recovered'],lw=0);da[4].fill_between(distance,0,tr.delta_probability,where=tr.delta_probability<0,color=C['conf'],lw=0);da[4].axhline(0,color='black',lw=.5);da[4].set_ylim(-1,1);da[4].set_yticks([-1,1]);da[4].set_ylabel('Δp',labelpad=1)
    stat=np.zeros((3,len(distance),4))
    for j,(name,col) in enumerate([('sar_fn',C['sar']),('fusion_fn',C['fusion']),('fusion_confident',C['conf'])]):stat[j,tr[name].to_numpy(bool)]=[*to_rgb(col),1]
    da[5].imshow(stat,extent=[distance[0],distance[-1],3,0],aspect='auto',interpolation='nearest');da[5].set_yticks([]);da[5].set_ylabel('FN',rotation=0,labelpad=4);da[5].tick_params(labelbottom=True);da[5].set_xlabel('Distance along transect (km)',labelpad=0)
    key(125,94,'HH', 'black','-');key(141,94,'HV',C['neutral'],'-');txt(160,94,'8-bit DN')
    txt(5,96,'Post-flood SAR / pre-flood NIR; Δp = fusion − SAR')
    # e — no missing-support edge is treated as a flood boundary.
    title('e','Reference-boundary proximity');bd=pd.read_csv(SD/'panel_e_boundary_distance.csv');em=ax(13,155,67,7);ee=ax(13,165,67,21,sharex=em)
    for region,style,short in [('Beijing_Baoding','-','BB'),('Harbin','--','HB')]:
        q=bd[(bd.region==region)&(bd.bin<10)&(bd.flood_m2>0)].copy();mid=np.sqrt(np.maximum(q.left_m.to_numpy(),2)*q.right_m.to_numpy());mid=np.where(np.isfinite(mid),mid,q.left_m.to_numpy()*1.4)
        em.step(mid,100*q.flood_m2/bd[bd.region==region].flood_m2.sum(),where='mid',color='black',ls=style,lw=1)
        for col,color in [('fusion_conf_m2',C['conf']),('fusion_other_fn_m2',C['fusion'])]:
            rate=100*q[col]/q.flood_m2;ee.plot(mid,rate,color=color,ls=style,lw=1.5)
            large=q.flood_m2.to_numpy()>=1e6
            ee.scatter(mid[large],rate[large],color=color,s=7,zorder=4)
            ee.scatter(mid[~large],rate[~large],facecolors='white',edgecolors=color,s=14,zorder=5,linewidths=.8)
    for aa in [em,ee]:aa.set_xscale('log');aa.tick_params(pad=1)
    em.tick_params(labelbottom=False);em.set_ylabel('Area (%)',labelpad=1);ee.set_ylabel('Flood area (%)',labelpad=1);ee.set_xlabel('Distance to reference boundary (m)',labelpad=0);ee.set_xticks([10,100,1000]);ee.set_xticklabels(['10','100','1,000']);ee.set_ylim(bottom=0)
    txt(6,150,'BB solid; HB dashed');key(43,150,'Confident',C['conf'],'-');key(62,150,'Other FN',C['fusion'],'-')
    # f — fixed encoding bins are observed covariates, not inferred land-cover classes.
    title('f','Observed intensity and error shifts');cf=pd.read_csv(SD/'panel_f_conditions.csv').sort_values(['region','hh_bin']);mat=cf[['delta_fn_pp','delta_conf_pp']].to_numpy();vmax=max(1,np.ceil(abs(mat).max()));cmap=LinearSegmentedColormap.from_list('effect',[C['sar'],'#FFFFFF',C['conf']]);fa=ax(122,156,26,29);im=fa.imshow(mat,cmap=cmap,norm=TwoSlopeNorm(0,-vmax,vmax),aspect='auto')
    for i in range(8):
        for j in range(2):fa.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',fontsize=7)
    rowlabels=[s+('†' if area<1e6 else '') for s,area in zip(['0–63','64–127','128–191','192–255']*2,cf.flood_m2)]
    fa.set_xticks([0,1]);fa.set_xticklabels(['ΔFN','ΔConf.']);fa.xaxis.tick_top();fa.tick_params(length=0,pad=2);fa.set_yticks(range(8));fa.set_yticklabels(rowlabels);fa.spines[['left','bottom']].set_visible(False);fa.axhline(3.5,color='black',lw=.6)
    txt(97,159,'BB',bold=True);txt(97,173.5,'HB',bold=True);txt(105,150,'HH DN');txt(150,151,'Flood km²');aa=ax(150,156,26,29,sharey=fa);aa.barh(np.arange(8),cf.flood_m2/1e6,color=C['neutral'],height=.65);aa.tick_params(axis='y',labelleft=False,length=0);aa.tick_params(pad=1);aa.set_ylim(7.5,-.5);aa.axhline(3.5,color='black',lw=.6)
    for i,area in enumerate(cf.flood_m2/1e6):aa.text(area-8 if area>100 else area+12,i,f'{area:.2f}',ha='right' if area>100 else 'left',va='center',fontsize=7)
    cbax=ax(144,190,30,1.2);cb=fig.colorbar(im,cax=cbax,orientation='horizontal');cb.set_ticks([-vmax,0,vmax]);cb.ax.tick_params(labelsize=7,pad=1,length=1.5);txt(98,190,'Fusion − SAR (pp)')
    fig.canvas.draw();stem='Fig02_main'+suffix
    fig.savefig(OUT/(stem+'.pdf'),dpi=450);fig.savefig(OUT/(stem+'.svg'),dpi=450);fig.savefig(AS/f'Fig02_round{args.round}{suffix}.png',dpi=220)
    if not suffix:
        for p,(x0,y0,ww,hh) in PANELS.items():fig.savefig(AS/f'panel_{p}.svg',bbox_inches=Bbox.from_bounds(x0/25.4,(H-y0-hh)/25.4,ww/25.4,hh/25.4),dpi=450)
        (AS/'layout_mm.json').write_text(json.dumps(dict(width=W,height=H,panels=PANELS),indent=2));(SD/'display_adjustments.json').write_text(json.dumps(dict(rgb_percentile_2=lo.tolist(),rgb_percentile_98=hi.tolist(),desaturation='25% colour +75% mean gray; then 0.72*image+0.28',zoom_windows=zoom_records,image_interpolation='nearest',component_summary=summary),indent=2))
    if args.final:
        fig.savefig(OUT/(stem+'.png'),dpi=1200)
        with Image.open(OUT/(stem+'.png')) as image:image.convert('RGB').save(OUT/(stem+'.tiff'),compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig);print('Rendered',stem,flush=True)
if __name__=='__main__':main()
