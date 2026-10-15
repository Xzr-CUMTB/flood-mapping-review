"""Native Python six-panel quantitative figure from independently audited source data."""
import os,json,argparse
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,PathPatch,ConnectionPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from PIL import Image
from pyproj import Geod
OUT=Path(os.path.abspath(__file__)).parents[1];SD=OUT/'source_data';AS=OUT/'assets';W,H=180,225
C=dict(sar='#446A9C',fusion='#80669D',ordinary_gate='#AC88AA',value_gate='#147D83',corrected='#147D83',remaining='#BD5B4B',review='#D5A34A',neutral='#B9C2CA')
LABEL=dict(sar='SAR',fusion='Fusion',ordinary_gate='Ordinary',value_gate='Value')
MARK=dict(sar='s',fusion='o',ordinary_gate='^',value_gate='D')
REGIONS=['Beijing_Baoding','Harbin'];RL={'Beijing_Baoding':'Beijing–Baoding','Harbin':'Harbin'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font',choices=['Times New Roman','Arial'],default='Times New Roman');parser.add_argument('--round',type=int,default=1);parser.add_argument('--final',action='store_true');a=parser.parse_args();suffix='_Arial' if a.font=='Arial' else ''
    assert json.loads((SD/'scientific_audit.json').read_text())['passed']
    mpl.rcParams.update({'font.family':a.font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.15,'axes.edgecolor':'black','axes.labelcolor':'black','text.color':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.85,'ytick.major.width':.85,'xtick.major.size':2.5,'ytick.major.size':2.5,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'custom','mathtext.rm':a.font,'mathtext.it':a.font+':italic','mathtext.bf':a.font+':bold','path.simplify':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4));artists=[]
    def txt(x,y,s,size=7,bold=False,ha='left',va='center',color='black',**kw):
        t=fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va=va,color=color,**kw);artists.append(t);return t
    def title(x,y,p,s):txt(x,y,p,8,True);txt(x+4,y,s,8,True)
    def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    def key(x,y,s,c,marker=None):
        fig.add_artist(Line2D([x/W,(x+3.3)/W],[1-y/H]*2,transform=fig.transFigure,color=c,lw=1.6,marker=marker,markersize=3,markevery=[1] if marker else None));txt(x+4,y,s)
    def box(x,y,w,h,color,alpha=1):fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=color,edgecolor='none',alpha=alpha,zorder=-5))
    def blackzero(axis,vertical=False):
        (axis.axvline if vertical else axis.axhline)(0,color='black',lw=.8,zorder=0)
    budget=pd.read_csv(SD/'panel_a_budget_response.csv');delta=pd.read_csv(SD/'panel_a_paired_differences.csv');ranks=pd.read_csv(SD/'panel_b_rankings.csv');decomp=pd.read_csv(SD/'panel_c_decomposition.csv');flows=pd.read_csv(SD/'panel_d_conservation_flows.csv');endpoints=pd.read_csv(SD/'panel_f_endpoints_and_costs.csv');interval=pd.read_csv(SD/'panel_f_paired_intervals.csv')
    # a: exact observed budget points, paired difference axes, primary-budget guide.
    title(2,4,'a','Budget-dependent residual omissions')
    key(4,11,'Fusion',C['fusion'],'o');key(25,11,'Ordinary',C['ordinary_gate'],'^');key(50,11,'Value',C['value_gate'],'D');key(70,11,'SAR',C['sar'],'s')
    txt(4,16,'Mean-entropy ranking · ideal correction')
    for j,region in enumerate(REGIONS):
        x=13+j*47;g=budget[budget.region==region];up=ax(x,24,39,24);lo=ax(x,55,39,12);txt(x+19.5,21,RL[region],bold=True,ha='center')
        up.axvspan(9.65,10.35,color='#F4EADF',zorder=-1);lo.axvspan(9.65,10.35,color='#F4EADF',zorder=-1)
        for m in ['fusion','ordinary_gate','value_gate']:
            q=g[g.model==m].sort_values('budget_fraction');up.plot(q.budget_fraction*100,q.residual_fn_km2,color=C[m],marker=MARK[m],ms=3,lw=1.55,zorder=3)
        up.set_xlim(-.7,20.7);up.set_xticks([0,1,5,10,20]);up.set_xticklabels([]);up.set_ylim((45,225) if j==0 else (19,99));up.set_yticks([50,100,150,200] if j==0 else [25,50,75]);up.tick_params(pad=2)
        d=delta[delta.region==region].sort_values('budget_fraction');xx=d.budget_fraction.to_numpy()*100;yy=d.delta_residual_fn_km2.to_numpy();lo.plot(xx,yy,color='black',lw=1.25,zorder=2);lo.scatter(xx,yy,c=np.where(yy>0,C['remaining'],C['corrected']),s=15,zorder=3,edgecolors='white',linewidths=.3);blackzero(lo);lo.set_xlim(up.get_xlim());lo.set_ylim(-10,8);lo.set_yticks([-8,0,6]);lo.set_xticks([0,5,10,20]);lo.tick_params(pad=2);lo.set_xlabel('Review area budget (%)',labelpad=1)
        if j==0:up.set_ylabel('Residual FN (km²)',labelpad=2);lo.set_ylabel('ΔFN',labelpad=2)
        txt(x+18,51,'Value − Fusion',ha='center');v=float(d.loc[d.budget_fraction==.1,'delta_residual_fn_km2'].iloc[0]);lo.annotate(f'{v:+.2f}',(10,v),xytext=(5,0),textcoords='offset points',fontsize=7,ha='left',va='center',bbox=dict(facecolor='white',edgecolor='none',pad=.2))
    txt(4,76,'Points: 0, 1, 5, 10, 20%; shaded band: 10%')
    # b: ordinal ranks prevent tiny metric gaps from becoming unreadable. Exact metrics label every node.
    title(109,4,'b','Accuracy–action inversion');txt(113,10,'Four models · 10% review budget');txt(119,16,'IoU ↑',bold=True);txt(175,16,'Residual FN ↓',bold=True,ha='right')
    for region,y in zip(REGIONS,[27,54]):
        txt(113,y-6,RL[region],bold=True);ra=ax(109,y,68,17);q=ranks[ranks.region==region].set_index('model');ra.set_xlim(0,1);ra.set_ylim(4.5,.5);ra.axis('off')
        for m in ['sar','fusion','ordinary_gate','value_gate']:
            r=q.loc[m];r1=int(r.iou_rank);r2=int(r.residual_fn_rank);line=ra.plot([.47,.70],[r1,r2],color=C[m],lw=1.55,alpha=.95,zorder=1)[0];line.set_gid(f'rank_link_{region}_{m}');ra.scatter([.47,.70],[r1,r2],color=C[m],s=16,marker=MARK[m],zorder=2)
            ra.text(.01,r1,f'{LABEL[m]} {r.flood_iou:.4f}',color=C[m],va='center',fontsize=7);ra.text(.75,r2,f'{r.residual_fn_km2:.2f}',color=C[m],va='center',fontsize=7)
            ra.text(.46,r1,str(r1),ha='right',va='center',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',pad=.1));ra.text(.71,r2,str(r2),ha='left',va='center',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',pad=.1))
    txt(113,74,'Rank 1 is best; FN in km²')
    # c: a signed, conserved waterfall; second increment is -delta corrected FN.
    title(2,83,'c','Recovery explains the inversion');txt(7,90,'ΔR = ΔF − ΔQ    (Value − Fusion)',bold=True)
    for j,region in enumerate(REGIONS):
        rr=decomp[decomp.region==region].iloc[0];wa=ax(12+j*34,100,28,30);txt(26+j*34,96,RL[region],ha='center',bold=True)
        f=float(rr.delta_initial_fn_km2);q=float(rr.negative_delta_corrected_fn_km2);r=float(rr.delta_residual_fn_km2)
        for x0,bottom,height,col in [(0,0,f,C['corrected']),(1,f,q,C['remaining']),(2,0,r,C['remaining'])]:
            bars=wa.bar(x0,height,bottom=bottom,color=col,width=.62,edgecolor='none',zorder=2);bars.patches[0].set_gid(f'waterfall_{region}_{x0}')
        wa.plot([.31,.69],[f,f],color='black',lw=.7);wa.plot([1.31,1.69],[r,r],color='black',lw=.7);blackzero(wa);wa.set_xlim(-.65,2.65);wa.set_ylim(-11,11);wa.set_yticks([-10,0,10]);wa.set_xticks([0,1,2],['ΔF','−ΔQ','ΔR']);wa.tick_params(pad=2)
        wa.text(0,f-.7,f'{f:+.2f}',ha='center',va='top');wa.text(1,r+.7,f'{q:+.2f}',ha='center',va='bottom');wa.text(2,r+.7,f'{r:+.2f}',ha='center',va='bottom',color='black')
        if j==0:wa.set_ylabel('FN-area difference (km²)',labelpad=1)
        else:wa.set_yticklabels([])
    txt(5,138,'F: initial   Q: corrected   R: remaining');txt(5,143,'Less initial FN; less FN recovered at 10%')
    # d: four separate bifurcations; no cross-model pixel transitions. Same physical width scale.
    title(81,83,'d','Where the omitted area goes');box(81,92,96,52,'white',1)
    key(85,90,'Corrected FN',C['corrected']);key(129,90,'Remaining FN',C['remaining'])
    flowax=ax(0,0,W,H);flowax.set_xlim(0,W);flowax.set_ylim(H,0);flowax.axis('off');flowax.set_zorder(-1);geometry=[];scale=.072
    def ribbon(x0,x1,source_top,target_top,thickness,colour,gid):
        mid=(x0+x1)/2
        verts=[(x0,source_top),(mid,source_top),(mid,target_top),(x1,target_top),(x1,target_top+thickness),(mid,target_top+thickness),(mid,source_top+thickness),(x0,source_top+thickness),(x0,source_top)]
        codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
        patch=PathPatch(MPath(verts,codes),facecolor=colour,edgecolor='none',alpha=.80);patch.set_gid(gid);flowax.add_patch(patch)
    for region,cy,labely in [('Beijing_Baoding',109,94),('Harbin',135,123)]:
        txt(84,labely,RL[region],bold=True)
        for m,x0 in [('fusion',85),('value_gate',133)]:
            rows=flows[(flows.region==region)&(flows.model==m)].set_index('target');f=float(rows.initial_fn_km2.iloc[0]);q=float(rows.loc['corrected FN','area_km2']);r=float(rows.loc['remaining FN','area_km2']);h=f*scale;hq=q*scale;hr=r*scale;top=cy-h/2;x1=x0+20
            txt(x0,cy-h/2-3,f'{LABEL[m]} · {f:.2f}',color=C[m]);flowax.add_patch(Rectangle((x0-1.7,top),1.7,h,facecolor=C['neutral'],edgecolor='none'))
            for state,amount,st,tt in [('corrected',q,top,top-1.8),('remaining',r,top+hq,top+hq+1.8)]:
                thick=amount*scale;ribbon(x0,x1,st,tt,thick,C[state],f'fn_flow_{region}_{m}_{state}');flowax.add_patch(Rectangle((x1,tt),1,thick,facecolor=C[state],edgecolor='none'));txt(x1+2.2,tt+thick/2,f'{amount:.2f}');geometry.append(dict(region=region,model=m,state=state,area_km2=amount,width_mm=thick,scale_mm_per_km2=scale,source_top_mm=st,target_top_mm=tt,initial_width_mm=h))
    txt(84,146,'Model · initial FN; one width scale (km²)')
    pd.DataFrame(geometry).to_csv(SD/'panel_d_rendered_flow_geometry.csv',index=False)
    # e: native-grid, contiguous cases; selected units remain those from regional 10% allocation.
    title(2,153,'e','Geographic review allocation')
    key(5,159,'Reviewed',C['review']);key(36,159,'Corrected FN',C['corrected']);key(72,159,'Residual FN',C['remaining'])
    caseinfo=json.loads((SD/'panel_e_case_selection.json').read_text());display=[];zoomrows=[];geod=Geod(ellps='WGS84')
    for region,y in [('Beijing_Baoding',169),('Harbin',199)]:
        z=np.load(SD/f'panel_e_{region}_window.npz');extent=z['extent'];midlat=(extent[2]+extent[3])/2;aspect=1/np.cos(np.deg2rad(midlat));width=44;height=width*(extent[3]-extent[2])/(extent[1]-extent[0])*aspect
        rgb=z['rgb'].astype(float)/255;valid=z['valid'];lo=np.quantile(rgb[valid],.02,axis=0);hi=np.quantile(rgb[valid],.98,axis=0);rgb=np.clip((rgb-lo)/(hi-lo+1e-12),0,1)**.85;grey=rgb.mean(axis=2,keepdims=True);rgb=.24*rgb+.76*grey;rgb=.57*rgb+.43;rgb[~valid]=1
        display.append(dict(region=region,rgb_low=lo.tolist(),rgb_high=hi.tolist(),gamma=.85,colour_fraction=.24,white_mix=.43,longitude_latitude_aspect=aspect,rendered_width_mm=width,rendered_height_mm=height))
        # Deterministic shared 128 x 128 native-pixel zoom, aligned to whole review units.
        sf=z['selected_fusion'].repeat(64,axis=0).repeat(64,axis=1);sv=z['selected_value_gate'].repeat(64,axis=0).repeat(64,axis=1)
        shared_error=valid&(z['label']==1)&(z['fusion']<.5)&(z['value_gate']<.5)&sf&~sv
        candidates=[(float(z['area_m2'][rr:rr+128,cc:cc+128][shared_error[rr:rr+128,cc:cc+128]].sum()),rr,cc) for rr in range(0,385,64) for cc in range(0,1153,64)]
        score,rr,cc=sorted(candidates,key=lambda v:(-v[0],v[1],v[2]))[0];assert score>0
        dx=(extent[1]-extent[0])/1280;dy=(extent[3]-extent[2])/512;zoom=[extent[0]+cc*dx,extent[0]+(cc+128)*dx,extent[3]-(rr+128)*dy,extent[3]-rr*dy]
        zoomrows.append(dict(region=region,row_start=rr,col_start=cc,pixels=128,area_corrected_by_fusion_remaining_under_value_km2=score/1e6,left=zoom[0],right=zoom[1],bottom=zoom[2],top=zoom[3],rule='Maximize reference FN common to both maps, corrected by Fusion selection and remaining under Value selection, over 128x128 windows aligned to 64 pixels; row/column stable ties. Same crop for both models.'))
        for m,x0 in [('fusion',5),('value_gate',57)]:
            ma=ax(x0,y,width,height);txt(x0,y-3,f'{RL[region]} · {LABEL[m]}',color=C[m]);ma.imshow(rgb,extent=extent,origin='upper',interpolation='nearest');ma.set_aspect(aspect);fn=(z[m]<.5)&(z['label']==1)&valid;sel=z['selected_'+m].repeat(64,axis=0).repeat(64,axis=1);overlay=np.zeros((*fn.shape,4));overlay[fn&sel]=to_rgba(C['corrected'],.92);overlay[fn&~sel]=to_rgba(C['remaining'],.92);ma.imshow(overlay,extent=extent,origin='upper',interpolation='nearest')
            # Reference outlines and exact 64-pixel rectangles stay vector-editable.
            lon=np.linspace(extent[0],extent[1],fn.shape[1],endpoint=False)+(extent[1]-extent[0])/fn.shape[1]/2;lat=np.linspace(extent[3],extent[2],fn.shape[0],endpoint=False)-(extent[3]-extent[2])/fn.shape[0]/2
            contour=ma.contour(lon,lat,np.ma.masked_where(~valid,z['label']==1),levels=[.5],colors='black',linewidths=.28,alpha=.70);contour.set_gid(f'reference_context_{region}_{m}')
            if np.any(~valid):ma.contour(lon,lat,valid.astype(int),levels=[.5],colors='black',linewidths=.4,linestyles='dashed')
            dy=(extent[3]-extent[2])/8;dx=(extent[1]-extent[0])/20
            for rr,cc in np.argwhere(z['selected_'+m]):
                rect=Rectangle((extent[0]+cc*dx,extent[3]-(rr+1)*dy),dx,dy,edgecolor=C['review'],facecolor=to_rgba(C['review'],.12),linewidth=.7);rect.set_gid(f'review_unit_{region}_{m}_{rr}_{cc}');ma.add_patch(rect)
            xb=extent[0]+(.43 if region=='Beijing_Baoding' else .38)*(extent[1]-extent[0]);yb=extent[2]+(.095 if region=='Beijing_Baoding' else .05)*(extent[3]-extent[2]);xe,_,_=geod.fwd(xb,yb,90,2000);ma.plot([xb,xe],[yb,yb],color='black',lw=1.6);ma.text((xb+xe)/2,yb+(extent[3]-extent[2])*.04,'2 km',ha='center',va='bottom',fontsize=7,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=.2));ma.text(.71,.80 if region=='Beijing_Baoding' else .95,'N ↑',transform=ma.transAxes,ha='right',va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=.2));ma.set_xlim(extent[:2]);ma.set_ylim(extent[2:]);ma.set_xticks([]);ma.set_yticks([])
            ma.set_aspect(aspect)
            for sp in ma.spines.values():sp.set_visible(True);sp.set_linewidth(.65)
            # Insets render the same native arrays and exact vector review rectangles.
            zw=12;zh=zw*aspect;za=ax(x0+1.1,y+1.1,zw,zh);za.imshow(rgb,extent=extent,origin='upper',interpolation='nearest');za.imshow(overlay,extent=extent,origin='upper',interpolation='nearest');za.contour(lon,lat,np.ma.masked_where(~valid,z['label']==1),levels=[.5],colors='black',linewidths=.35)
            for rr,cc in np.argwhere(z['selected_'+m]):
                l=extent[0]+cc*dx;b=extent[3]-(rr+1)*dy
                if l<zoom[1] and l+dx>zoom[0] and b<zoom[3] and b+dy>zoom[2]:za.add_patch(Rectangle((l,b),dx,dy,edgecolor=C['review'],facecolor=to_rgba(C['review'],.12),linewidth=.9))
            za.set_xlim(zoom[:2]);za.set_ylim(zoom[2:]);za.set_aspect(aspect);za.set_xticks([]);za.set_yticks([])
            for sp in za.spines.values():sp.set_visible(True);sp.set_linewidth(.8)
            ma.add_patch(Rectangle((zoom[0],zoom[2]),zoom[1]-zoom[0],zoom[3]-zoom[2],fill=False,edgecolor='black',linewidth=.8))
            # Both connector ends touch the corresponding source and inset boundaries.
            for src,dest in [((zoom[0],zoom[3]),(1,1)),((zoom[0],zoom[2]),(1,0))]:
                fig.add_artist(ConnectionPatch(xyA=src,coordsA='data',axesA=ma,xyB=dest,coordsB='axes fraction',axesB=za,color='black',linewidth=.5,zorder=3))
    (AS/'geographic_display_parameters.json').write_text(json.dumps(display,indent=2))
    pd.DataFrame(zoomrows).to_csv(SD/'panel_e_shared_zoom_selection.csv',index=False)
    # f: paired FN/FP view plus familywise spatial bands. No pointwise bands transplanted to other budgets.
    title(110,153,'f','Error trade-offs and uncertainty')
    for j,region in enumerate(REGIONS):
        x0=120+j*32;ea=ax(x0,165,24,20);txt(x0+12,161,'BB' if j==0 else 'Harbin',ha='center',bold=True);q=endpoints[endpoints.region==region].set_index('model')
        for m in ['fusion','ordinary_gate','value_gate']:ea.scatter(q.loc[m,'residual_fp_km2'],q.loc[m,'residual_fn_km2'],s=23,color=C[m],marker=MARK[m],zorder=3)
        ea.set_xlim((33,52) if j==0 else (29,37));ea.set_ylim((95,113) if j==0 else (42,51));ea.set_xticks([35,50] if j==0 else [30,35]);ea.set_yticks([100,110] if j==0 else [45,50]);ea.tick_params(pad=2);ea.set_xlabel('FP (km²)',labelpad=1)
        if j==0:ea.set_ylabel('FN (km²)',labelpad=1)
    txt(111,195,'Simultaneous 95% intervals · 10 km blocks')
    fa=ax(131,201,45,17);fa.axvline(0,color='black',lw=.8);fa.set_ylim(3.65,-.65);fa.set_xlim(-14,13);fa.set_xticks([-10,0,10]);fa.set_yticks([0,1,2,3],['BB  FN','BB  FP','HB  FN','HB  FP']);fa.tick_params(pad=2);fa.set_xlabel('Value − Fusion area (km²)',labelpad=1)
    for i,(region,endpoint) in enumerate([(r,e) for r in REGIONS for e in ['residual_fn_km2','residual_fp_km2']]):
        r=interval[(interval.region==region)&(interval.endpoint==endpoint)].iloc[0];v=r.difference_first_minus_second;lo=r.simultaneous_lower95;hi=r.simultaneous_upper95;col=C['remaining'] if 'fn_' in endpoint else C['neutral'];fa.plot([lo,hi],[i,i],color=col,lw=1.7,zorder=2);fa.scatter(v,i,marker='v' if 'fn_' in endpoint else 's',s=20,color=col,edgecolors='black',linewidths=.3,zorder=3)
    metadata=dict(width_mm=W,height_mm=H,font=a.font,body_minimum_pt=7,panels=6,flow_width_mm_per_km2=scale,geographic_case_pixels=[1280,512],axis_width_pt=1.15,tick_direction='in',models_in_rank_comparison=['sar','fusion','ordinary_gate','value_gate'])
    (AS/('render_metadata'+suffix+'.json')).write_text(json.dumps(metadata,indent=2))
    fig.savefig(AS/f'Fig04_round{a.round}{suffix}.png',dpi=240)
    # Record figure-level label extents for final geometric review.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
    for t in artists:
        bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted());bounds.append(dict(text=t.get_text(),x0_mm=bb.x0*W,y0_mm=(1-bb.y1)*H,x1_mm=bb.x1*W,y1_mm=(1-bb.y0)*H))
    (AS/('text_bounds'+suffix+'.json')).write_text(json.dumps(bounds,indent=2))
    if a.final:
        stem=OUT/('Fig04_main'+suffix);fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.png',dpi=1200)
        Image.MAX_IMAGE_PIXELS=160_000_000
        with Image.open(str(stem)+'.png') as im:im.convert('RGB').save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig);print('Rendered',a.round,a.font,flush=True)
if __name__=='__main__':main()
