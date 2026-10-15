"""Fig05: native quantitative and geographic artists, from verified frozen sources."""
import os,json,argparse
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,LinearSegmentedColormap,to_rgba
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Geod
OUT=Path(os.path.abspath(__file__)).parents[1];SD=OUT/'source_data';AS=OUT/'assets';W,H=180,205
C=dict(H='#80669D',L='#147D83',P='#D5A34A',random='#B9C2CA',FN='#BD5B4B',both='#446A9C',neither='#F0F0ED')
REGIONS=['Beijing_Baoding','Harbin'];RL={'Beijing_Baoding':'Beijing–Baoding','Harbin':'Harbin'}
POL={'H':'mean_entropy','L':'learned_omission_density','P':'fn_probability_density'}
DENSITY=LinearSegmentedColormap.from_list('area',['#FAFCFD','#DCE9EE','#91B4C3','#446A9C','#183D4A'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font',choices=['Times New Roman','Arial'],default='Times New Roman');parser.add_argument('--round',type=int,default=1);parser.add_argument('--final',action='store_true');args=parser.parse_args();suffix='_Arial' if args.font=='Arial' else ''
    assert json.loads((SD/'scientific_audit.json').read_text())['passed']
    mpl.rcParams.update({'font.family':args.font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.15,'axes.edgecolor':'black','axes.labelcolor':'black','text.color':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.85,'ytick.major.width':.85,'xtick.major.size':2.3,'ytick.major.size':2.3,'xtick.minor.visible':False,'ytick.minor.visible':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'custom','mathtext.rm':args.font,'mathtext.it':args.font+':italic','mathtext.bf':args.font+':bold','path.simplify':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4));texts=[];roles=[]
    def txt(x,y,s,size=7,bold=False,ha='left',va='center',color='black',**kw):
        t=fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va=va,color=color,**kw);texts.append(t);return t
    def title(x,y,p,s):txt(x,y,p,8,True);txt(x+4,y,s,8,True)
    def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    def box(x,y,w,h,color,alpha=1):fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=color,edgecolor='none',alpha=alpha,zorder=-10))
    def key(x,y,s,c,ls='-',square=False):
        if square:fig.add_artist(Rectangle((x/W,(H-y-1)/H),2.7/W,2/H,facecolor=c,edgecolor='black',linewidth=.35,transform=fig.transFigure))
        else:fig.add_artist(Line2D([x/W,(x+3.2)/W],[1-y/H]*2,transform=fig.transFigure,color=c,lw=1.6,ls=ls))
        txt(x+4,y,s)
    def gid(artist,name):artist.set_gid(name);roles.append(name);return artist
    def clean(axis):axis.tick_params(pad=1.8);axis.minorticks_off()
    def marginal(axis,xe,ye,mat,x,y,w,h,color=C['random']):
        top=ax(x,y-5,w,4);top.stairs(mat.sum(axis=1),xe,fill=True,color=color,lw=0);top.set_xlim(axis.get_xlim());top.axis('off')
        side=ax(x+w+1,y,4,h);side.stairs(mat.sum(axis=0),ye,fill=True,color=color,lw=0,orientation='horizontal');side.set_yscale(axis.get_yscale(),**({'linthresh':.001,'linscale':.8} if axis.get_yscale()=='symlog' else {}));side.set_ylim(axis.get_ylim());side.axis('off')
    hist=np.load(SD/'density_statistics.npz');stats=pd.read_csv(SD/'panel_a_distribution_summary.csv');budget=pd.read_csv(SD/'panel_b_budget_capture.csv');means=pd.read_csv(SD/'panel_c_score_bin_means.csv');quant=pd.read_csv(SD/'panel_c_conditional_quantiles.csv');upset=pd.read_csv(SD/'panel_d_disjoint_intersections.csv');interval=pd.read_csv(SD/'panel_f_simultaneous_intervals.csv')
    # Functional zones are subtle continuous strips, not six separate cards.
    box(1,1,178,76,'white',1);box(1,79,178,60,'white',1);box(1,142,178,62,'white',1)
    title(2,4,'a','Uncertainty–omission structure');txt(6,10,'Fixed Value Gate · joint plots: positive FN units')
    for j,region in enumerate(REGIONS):
        x=12+j*48;r=stats[stats.region==region].iloc[0];xx=hist['entropy_edges'];yy=hist['positive_D_edges'];hh=hist[region+'_joint_H'];main=ax(x,25,31,21)
        main.pcolormesh(xx,yy,np.ma.masked_equal(hh.T,0),cmap=DENSITY,norm=LogNorm(.0001,10000),rasterized=True);main.set_yscale('log');main.set_ylim(1e-4,1);main.set_xlim(0,np.log(2));main.set_xticks([0,.3,.6]);main.set_yticks([.0001,.01,1],[r'$10^{-4}$','0.01','1']);clean(main)
        if j==0:main.set_ylabel('Observed FN density',labelpad=1)
        else:main.set_yticklabels([])
        main.set_xlabel('Mean entropy (nats)',labelpad=1);marginal(main,xx,yy,hh,x,25,31,21)
        txt(x+15.5,16,RL[region],ha='center',bold=True)
        txt(x,54,f'Zero FN: {100*r.zero_FN_area_fraction:.1f}% of area')
        rk=ax(x,60,12,11);rk.pcolormesh(hist['rank_edges'],hist['rank_edges'],np.ma.masked_equal(hist[region+'_copula'].T,0),cmap=DENSITY,norm=LogNorm(.0001,10000),rasterized=True);rk.set_xlim(0,1);rk.set_ylim(0,1);rk.set_xticks([0,1]);rk.set_yticks([0,1]);clean(rk)
        if j==0:rk.set_ylabel('FN rank',labelpad=0)
        else:rk.set_yticklabels([])
        txt(x+16,61,'H →; FN ↑');txt(x+16,65,f'ρ = {r.entropy_FN_weighted_midrank_correlation:.2f}');txt(x+16,69,f'Low-H FN: {100*r.FN_fraction_in_lower_entropy_half:.1f}%')
    ca=ax(37,74,25,1.5);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=LogNorm(.0001,10000),cmap=DENSITY),cax=ca,orientation='horizontal',ticks=[.0001,1,10000]);cb.ax.set_xticklabels([r'$10^{-4}$','1',r'$10^{4}$']);cb.ax.tick_params(pad=0,length=1.5);cb.outline.set_linewidth(.65);txt(67,74.5,'km² / bin (a, c)')
    # b: measured budgets, line segments as guides; actual random realizations.
    title(108,4,'b','Budgeted omission capture');key(111,10,'H entropy',C['H']);key(143,10,'L learned',C['L']);key(111,15,'P proxy',C['P']);key(143,15,'Random × 3',C['random']);key(111,20,'Oracle-guided reference','black','--')
    for j,region in enumerate(REGIONS):
        x=117+j*34;ba=ax(x,29,25,25);txt(x+12.5,26,'BB' if j==0 else 'Harbin',ha='center',bold=True);ba.axvspan(9.7,10.3,color='#F4EADF',zorder=-1)
        for code in ['random_seed_17','random_seed_29','random_seed_43','oracle_fn_density','H','L','P']:
            policy=POL.get(code,code);q=budget[(budget.region==region)&(budget.policy==policy)].sort_values('budget_fraction');col=C[code] if code in POL else C['random'] if code.startswith('random') else 'black';isrand=code.startswith('random');oracle=code.startswith('oracle')
            gid(ba.plot(q.actual_area_fraction*100,q.capture_fraction*100,color=col,lw=.65 if isrand else 1.55,ls='--' if oracle else '-',marker='o' if not isrand else '.',ms=2.7 if not isrand else 1.5,zorder=1 if isrand else 3)[0],f'capture_{region}_{code}')
        ba.set_xlim(-.7,20.8);ba.set_ylim(-3,103);ba.set_xticks([0,10,20]);ba.set_yticks([0,50,100]);clean(ba)
        if j==0:ba.set_ylabel('Initial FN captured (%)',labelpad=1)
        else:ba.set_yticklabels([])
        ba.set_xlabel('Review area (%)',labelpad=1)
        q=budget[(budget.region==region)&budget.budget_fraction.eq(.1)].set_index('policy');h=q.loc[POL['H'],'capture_fraction']*100
        for k,code in enumerate(['L','P']):
            dv=q.loc[POL[code],'capture_fraction']*100-h;txt(x+1,65+4*k,f'{code} − H: {dv:+.1f} pp',color=C[code])
    txt(112,61,'Capture difference at 10%',bold=True);txt(112,75,'Ideal correction in selected units')
    # c: score correspondence, all-unit joint density + positive-FN conditional quartiles.
    title(2,83,'c','Learned score–outcome correspondence');key(7,89,'Weighted bin mean',C['L']);txt(43,89,'ρ = '+' / '.join(f'{stats.loc[stats.region==r,"risk_FN_weighted_midrank_correlation"].iloc[0]:.2f}'.replace('-','−') for r in REGIONS))
    for j,region in enumerate(REGIONS):
        x=12+j*40;xx=hist['score_edges'];yy=hist['all_D_edges'];hh=hist[region+'_joint_L'];ca=ax(x,99,26,20);ca.pcolormesh(xx,yy,np.ma.masked_equal(hh.T,0),cmap=DENSITY,norm=LogNorm(.0001,10000),rasterized=True);ca.set_yscale('symlog',linthresh=.001,linscale=.8);ca.set_ylim(0,1);ca.set_xlim(0,1);ca.set_xticks([0,.5,1]);ca.set_yticks([0,.01,1],['0','0.01','1']);clean(ca)
        if j==0:ca.set_ylabel('Observed FN density',labelpad=1)
        else:ca.set_yticklabels([])
        ca.set_xlabel('Learned score L',labelpad=1);marginal(ca,xx,yy,hh,x,99,26,20);q=means[means.region==region];gid(ca.plot(q.mean_score,q.mean_D,color=C['L'],lw=1.55,marker='.',ms=2)[0],f'risk_bin_mean_{region}');txt(x+13,92,'BB' if j==0 else 'Harbin',bold=True,ha='center')
        qt=quant[quant.region==region];qa=ax(x,130,26,7);qa.vlines(qt.risk_decile,qt.positive_D_q25,qt.positive_D_q75,color=C['L'],lw=1.4);gid(qa.plot(qt.risk_decile,qt.positive_D_median,'o',color=C['L'],ms=2)[0],f'risk_conditional_{region}');qa.set_xlim(.5,10.5);qa.set_ylim(0,.052);qa.set_yticks([0,.04]);qa.set_xticks([1,5,10]);clean(qa);qa.set_xlabel('Risk decile',labelpad=0)
        if j==0:qa.set_ylabel('D | D > 0',labelpad=1)
        else:qa.set_yticklabels([])
    txt(7,126,'Positive FN: median and interquartile range')
    # d: three aligned axes, one disjoint-membership order, separate area quantities.
    title(88,83,'d','Review-selection intersections');txt(92,89,'Same 10% budget · exact H / L / P sets')
    order=[1,2,4,3,5,6,7];cellcols=[C['H'],C['L'],C['P'],'#8F9FA9','#A59CA5','#7EADA1',C['both']]
    for j,region in enumerate(REGIONS):
        x=102+j*39;q=upset[upset.region==region].set_index('intersection_id').loc[order];aa=ax(x,98,34,16);ff=ax(x,116,34,8);mat=ax(x,128,34,8);txt(x+17,94,RL[region] if j==0 else 'Harbin',ha='center',bold=True)
        for k,row in enumerate(q.itertuples()):
            gid(aa.bar(k,row.valid_area_km2,width=.68,color=cellcols[k],edgecolor='none')[0],f'upset_area_{region}_{row.Index}');gid(ff.plot([k,k],[0,row.fn_area_km2],color=C['FN'],lw=1.4)[0],f'upset_fn_{region}_{row.Index}');ff.scatter(k,row.fn_area_km2,s=9,color=C['FN'],zorder=3)
        aa.set_xlim(-.6,6.6);aa.set_ylim(0,1480);aa.set_yticks([0,600,1200]);ff.set_xlim(-.6,6.6);ff.set_ylim(0,80);ff.set_yticks([0,60]);aa.set_xticks([]);ff.set_xticks([]);clean(aa);clean(ff)
        peak=int(np.argmax(q.valid_area_km2.to_numpy()));aa.text(peak,q.valid_area_km2.iloc[peak]+35,f'{q.valid_area_km2.iloc[peak]:,.0f}',ha='center',va='bottom',fontsize=7)
        for k,v in enumerate(q.fn_area_km2):ff.text(k,v+4,('0' if v==0 else '<0.1' if v<.1 else f'{v:.1f}'),ha='center',va='bottom',fontsize=7,color='black')
        if j==0:aa.set_ylabel('Area (km²)',labelpad=1);ff.set_ylabel('FN (km²)',labelpad=1)
        else:aa.set_yticklabels([]);ff.set_yticklabels([])
        mat.set_xlim(-.6,6.6);mat.set_ylim(2.5,-.5);mat.axis('off')
        for row,code in enumerate(['H','L','P']):mat.text(-.78,row,code,ha='right',va='center',fontsize=7,color=C[code])
        for k,cell in enumerate(order):
            yy=[r for r,bit in enumerate([1,2,4]) if cell&bit];mat.scatter([k]*3,[0,1,2],color='#D7DEE0',s=9);gid(mat.plot([k]*len(yy),yy,'o-',color='black',lw=.9,ms=2.7)[0],f'upset_membership_{region}_{cell}')
        none=upset[(upset.region==region)&(upset.intersection_id==0)].iloc[0];txt(x,140,f'Outside union: {none.fn_area_km2:.1f} km² FN')
    # e: native geographical windows, selected by recorded extrema in both directions.
    title(2,147,'e','Geography of selection disagreement');key(5,153,'H only',C['H'],square=True);key(26,153,'L only',C['L'],square=True);key(47,153,'Both',C['both'],square=True);key(65,153,'Neither',C['neither'],square=True)
    key(5,158,'Initial FN',C['FN'],square=True);key(32,158,'Reference flood','black');txt(67,158,'Exclusive FN below')
    cases=json.loads((SD/'panel_e_case_selection.json').read_text())['cases'];geod=Geod(ellps='WGS84');display=[]
    for j,region in enumerate(REGIONS):
        x=6+j*52;txt(x+23,163,RL[region],bold=True,ha='center')
        for k,direction in enumerate(['H_favouring','L_favouring']):
            case=next(r for r in cases if r['region']==region and r['direction']==direction);z=np.load(SD/f'panel_e_{case["case_id"]}.npz');ex=z['extent'];lat=(ex[2]+ex[3])/2;aspect=1/np.cos(np.deg2rad(lat));width=44;height=width*(ex[3]-ex[2])/(ex[1]-ex[0])*aspect;y=166+19*k;ma=ax(x,y,width,height);valid=z['valid'];rgb=z['rgb'].astype(float)/255;lo=np.quantile(rgb[valid],.02,axis=0);hi=np.quantile(rgb[valid],.98,axis=0);rgb=np.clip((rgb-lo)/(hi-lo+1e-12),0,1)**.85;grey=rgb.mean(axis=2,keepdims=True);rgb=.12*rgb+.88*grey;rgb=.55*rgb+.45;rgb[~valid]=1;ma.imshow(rgb,extent=ex,origin='upper',interpolation='nearest')
            dx=(ex[1]-ex[0])/16;dy=(ex[3]-ex[2])/4;cats=z['selected_H'].astype(int)+z['selected_L'].astype(int)*2
            raster_cats=cats.repeat(64,0).repeat(64,1);selection_layer=np.zeros((*valid.shape,4))
            for category,col in enumerate([C['neither'],C['H'],C['L'],C['both']]):selection_layer[(raster_cats==category)&valid]=to_rgba(col,.48 if category else .02)
            ma.imshow(selection_layer,extent=ex,origin='upper',interpolation='nearest')
            for rr in range(4):
                for cc in range(16):
                    rect=Rectangle((ex[0]+cc*dx,ex[3]-(rr+1)*dy),dx,dy,facecolor='none',edgecolor='black',linewidth=.22);gid(rect,f'review_cell_{region}_{direction}_{rr}_{cc}');ma.add_patch(rect)
            fn=valid&(z['label']==1)&(z['probability']<.5);overlay=np.zeros((*valid.shape,4));overlay[fn]=to_rgba(C['FN'],.95);ma.imshow(overlay,extent=ex,origin='upper',interpolation='nearest');lon=np.linspace(ex[0],ex[1],valid.shape[1],endpoint=False)+(ex[1]-ex[0])/(2*valid.shape[1]);latitude=np.linspace(ex[3],ex[2],valid.shape[0],endpoint=False)-(ex[3]-ex[2])/(2*valid.shape[0]);cont=ma.contour(lon,latitude,np.ma.masked_where(~valid,z['label']==1),levels=[.5],colors='black',linewidths=.25);gid(cont,f'reference_flood_{region}_{direction}');ma.set_xlim(ex[:2]);ma.set_ylim(ex[2:]);ma.set_aspect(aspect);ma.set_xticks([]);ma.set_yticks([])
            for sp in ma.spines.values():sp.set_visible(True);sp.set_linewidth(.65)
            # The bar is an actual geodesic at the displayed latitude.
            xb=ex[0]+.69*(ex[1]-ex[0]);yb=ex[2]+.08*(ex[3]-ex[2]);xe,_,_=geod.fwd(xb,yb,90,1000);ma.plot([xb,xe],[yb,yb],color='black',lw=1.6);ma.text((xb+xe)/2,yb+.05*(ex[3]-ex[2]),'1 km',ha='center',va='bottom',fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.15));ma.text(.025,.94,'N ↑',ha='left',va='top',transform=ma.transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.15))
            hfn=case['FN_by_selection_km2']['H_only'];lfn=case['FN_by_selection_km2']['L_only'];txt(x,y+height+2.2,f'H {hfn:.3f} / L {lfn:.3f} km² FN')
            display.append(dict(case_id=case['case_id'],rgb_low=lo.tolist(),rgb_high=hi.tolist(),gamma=.85,colour_fraction=.12,white_mix=.45,selection_alpha=.48,longitude_latitude_aspect=aspect,rendered_width_mm=width,rendered_height_mm=height,scale_bar_metres=1000))
    (AS/'geographic_display_parameters.json').write_text(json.dumps(display,indent=2))
    # f: eight independently preserved contrasts, same frozen 12-estimate family per region.
    title(116,147,'f','Paired ranking effects');txt(119,153,'Learned minus simple ranking');txt(136,158,'ΔFN',bold=True,ha='center');txt(165,158,'ΔFP',bold=True,ha='center')
    combos=[(region,policy) for region in REGIONS for policy in ['mean_entropy','fn_probability_density']]
    for j,endpoint in enumerate(['residual_fn_km2','residual_fp_km2']):
        fa=ax(130+j*28,163,20,29);fa.axvline(0,color='black',lw=.8);fa.set_ylim(3.65,-.65);fa.set_xlim((-8,69) if j==0 else (-15,114));fa.set_xticks([0,30,60] if j==0 else [0,50,100]);fa.set_yticks([]);clean(fa)
        for i,(region,policy) in enumerate(combos):
            r=interval[(interval.region==region)&interval.second_policy.eq(policy)&interval.endpoint.eq(endpoint)].iloc[0];v=r.difference_first_minus_second;lo=r.simultaneous_lower95;hi=r.simultaneous_upper95
            gid(fa.plot([lo,hi],[i,i],color=C['FN'],lw=1.6)[0],f'paired_interval_{region}_{policy}_{endpoint}');fa.scatter(v,i,color=C['FN'],s=15,marker='o' if policy=='mean_entropy' else 's',edgecolors='black',linewidths=.3,zorder=3);fa.text(v,i-.17,f'{v:+.2f}',fontsize=7,ha='center',va='bottom')
            if j==0:fa.text(-13,i,('BB ' if i==0 else 'HB ' if i==2 else '')+('L−H' if policy=='mean_entropy' else 'L−P'),fontsize=7,ha='right',va='center',clip_on=False)
        fa.set_xlabel('Residual area (km²)',labelpad=1)
    txt(118,199,'Positive: more residual error');txt(118,203,'95% simultaneous · 10 km blocks')
    metadata=dict(width_mm=W,height_mm=H,font=args.font,body_minimum_pt=7,panels=6,axis_width_pt=1.15,tick_direction='in',density_norm=dict(kind='log',vmin=.0001,vmax=10000,units='km2 per bin'),vector_roles=roles,case_pixels=[1024,256],fixed_base_map='value_gate seed 17',risk_scores_are_not_calibrated_probabilities=True)
    (AS/('render_metadata'+suffix+'.json')).write_text(json.dumps(metadata,indent=2));fig.savefig(AS/f'Fig05_round{args.round}{suffix}.png',dpi=240)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
    for t in texts:
        bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted());bounds.append(dict(text=t.get_text(),x0_mm=bb.x0*W,y0_mm=(1-bb.y1)*H,x1_mm=bb.x1*W,y1_mm=(1-bb.y0)*H))
    (AS/('text_bounds'+suffix+'.json')).write_text(json.dumps(bounds,indent=2))
    fig.savefig(AS/f'Fig05_round{args.round}{suffix}.pdf',dpi=600)
    if args.final:
        stem=OUT/('Fig05_main'+suffix);fig.savefig(str(stem)+'.pdf',dpi=600);fig.savefig(str(stem)+'.svg',dpi=600);fig.savefig(str(stem)+'.png',dpi=1200);Image.MAX_IMAGE_PIXELS=160_000_000
        with Image.open(str(stem)+'.png') as im:im.convert('RGB').save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig);print('Rendered',args.round,args.font,flush=True)
if __name__=='__main__':main()
