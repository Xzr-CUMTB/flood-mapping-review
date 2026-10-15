"""Four-panel, fully vector decision figure from audited retrospective source tables."""
import os,json,argparse
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LinearSegmentedColormap,to_rgba
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
OUT=Path(os.path.abspath(__file__)).parents[1];SD=OUT/'source_data';AS=OUT/'assets';W,H=180,170
MODELS=['sar','fusion','ordinary_gate','value_gate'];REGIONS=['Beijing_Baoding','Harbin'];BUDGETS=[.01,.05,.1,.2];WEIGHTS=[.25,1.,4.]
C=dict(sar='#446A9C',fusion='#80669D',ordinary_gate='#AC88AA',value_gate='#147D83',FN='#BD5B4B',FP='#B9C2CA',budget='#D5A34A')
NAME=dict(sar='SAR',fusion='Fusion',ordinary_gate='Ordinary',value_gate='Value');RL={'Beijing_Baoding':'Beijing–Baoding','Harbin':'Harbin'}
REGRET=LinearSegmentedColormap.from_list('regret',['#FEFDFC','#F2E1D5','#DDA38F','#BD5B4B','#993B34'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font',choices=['Times New Roman','Arial'],default='Times New Roman');parser.add_argument('--round',type=int,default=1);parser.add_argument('--final',action='store_true');args=parser.parse_args();suffix='_Arial' if args.font=='Arial' else ''
    assert json.loads((SD/'scientific_audit.json').read_text())['passed']
    mpl.rcParams.update({'font.family':args.font,'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.15,'axes.edgecolor':'black','axes.labelcolor':'black','text.color':'black','xtick.color':'black','ytick.color':'black','xtick.direction':'in','ytick.direction':'in','xtick.major.width':.85,'ytick.major.width':.85,'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'custom','mathtext.rm':args.font,'mathtext.it':args.font+':italic','mathtext.bf':args.font+':bold','path.simplify':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4));texts=[];roles=[];geometry=[]
    def txt(x,y,s,size=7,bold=False,ha='left',va='center',color='black',**kw):
        t=fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',color=color,ha=ha,va=va,**kw);texts.append(t);return t
    def title(x,y,p,s):txt(x,y,p,8,True);txt(x+4,y,s,8,True)
    def ax(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    def gid(a,name):a.set_gid(name);roles.append(name);return a
    def box(x,y,w,h,c,alpha=1,z=-10):
        p=Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=c,edgecolor='none',alpha=alpha,zorder=z);fig.add_artist(p);return p
    def line(x0,x1,y,c='black',lw=.6):fig.add_artist(Line2D([x0/W,x1/W],[1-y/H]*2,color=c,lw=lw,transform=fig.transFigure))
    def key(x,y,s,c):box(x,y-1,2.8,2,c,z=1);txt(x+4,y,s)
    def clean(a):a.tick_params(pad=2);a.minorticks_off()
    atlas=pd.read_csv(SD/'panel_a_decision_atlas.csv');params=pd.read_csv(SD/'panel_b_analytical_parameters.csv');curves=pd.read_csv(SD/'panel_b_analytical_lines.csv');markers=pd.read_csv(SD/'panel_b_preset_weight_markers.csv');regret=pd.read_csv(SD/'panel_c_primary_regret.csv');ledger=pd.read_csv(SD/'panel_d_displayed_scenarios.csv')
    box(1,1,178,71,'white',1);box(1,74,178,59,'white',1);box(1,135,178,34,'white',1)
    # a: strictly categorical 4x3 atlases; microbar geometry shares a physical scale.
    title(2,4,'a','Budget–cost decision atlas');txt(127,4,'L = FN + λFP',8,True)
    for m,x in zip(MODELS,[6,34,65,114]):key(x,10,NAME[m]+(' Gate' if 'gate' in m else ''),C[m])
    txt(6,16,'Observed lowest loss among four candidates · mean-entropy review')
    margin_scale=.30
    for j,region in enumerate(REGIONS):
        x=14+j*90;width=68;cw=width/3;aa=ax(x,26,width,34);aa.set_xlim(0,3);aa.set_ylim(4,0);txt(x+width/2,21,RL[region],bold=True,ha='center')
        for ri,b in enumerate(BUDGETS):
            for ci,w in enumerate(WEIGHTS):
                r=atlas[atlas.region.eq(region)&atlas.budget_fraction.eq(b)&atlas.lambda_fp.eq(w)].iloc[0];assert not r.tied_best
                gid(aa.add_patch(Rectangle((ci,ri),1,1,facecolor=to_rgba(C[r.best_model],.31),edgecolor='black',linewidth=.45)),f'decision_cell_{region}_{ri}_{ci}')
                aa.text(ci+.5,ri+.29,NAME[r.best_model],fontsize=7,fontweight='bold',ha='center',va='center');aa.text(ci+.91,ri+.73,f'{r.margin_km2eq:.2f}',fontsize=7,ha='right',va='center')
                length_mm=float(r.margin_km2eq)*margin_scale;gid(aa.add_patch(Rectangle((ci+.10,ri+.70),length_mm/cw,.065,facecolor='black',edgecolor='none')),f'margin_bar_{region}_{ri}_{ci}')
                geometry.append(dict(role='margin_bar',region=region,budget_fraction=b,lambda_fp=w,value=r.margin_km2eq,length_mm=length_mm,scale_mm_per_unit=margin_scale))
        gid(aa.add_patch(Rectangle((0,2),3,1,facecolor='none',edgecolor=C['budget'],linewidth=1.15,zorder=3)),f'primary_budget_{region}')
        aa.set_xticks([.5,1.5,2.5],['0.25','1','4']);aa.set_yticks(np.arange(4)+.5,['1','5','10','20']);aa.set_xlabel('FP-to-FN cost weight (λ)',labelpad=1);clean(aa)
        if j==0:aa.set_ylabel('Review area budget (%)',labelpad=2)
        else:aa.set_yticklabels([])
    txt(6,70,'Numbers / bars: gap to runner-up (km²-eq)');line(128,128+20*margin_scale,70,lw=1.5);txt(136,70,'20 km²-eq')
    # b: fixed-outcome analytical reweighting, with a separate two-model preference strip.
    title(2,77,'b','Analytical cost crossover');txt(6,83,'10% review · Value − Fusion')
    txt(14,88,'Harbin: pairwise lower loss')
    hb=params[params.region.eq('Harbin')].iloc[0];root=float(hb.positive_break_even_lambda);strip=ax(14,90,71*4/4.12,3);strip.set_xlim(0,4);strip.set_ylim(0,1);strip.axis('off')
    for lo,hi,m in [(0,root,'fusion'),(root,4,'value_gate')]:
        gid(strip.add_patch(Rectangle((lo,0),hi-lo,1,facecolor=to_rgba(C[m],.30),edgecolor='black',linewidth=.45)),f'preference_strip_{m}');strip.text((lo+hi)/2,.5,NAME[m],ha='center',va='center',fontsize=7)
    ba=ax(14,96,71,24);ba.set_xlim(0,4.12);ba.set_ylim(-8,36);ba.axhspan(0,36,color='#F5E9E5',alpha=.55,zorder=-5);ba.axhspan(-8,0,color='#E8F1F2',alpha=.8,zorder=-5);gid(ba.axhline(0,color='black',lw=.85), 'pairwise_zero')
    for region,ls,mk in [('Beijing_Baoding','-','o'),('Harbin','--','D')]:
        r=params[params.region.eq(region)].iloc[0];q=curves[curves.region.eq(region)];pts=markers[markers.region.eq(region)];gid(ba.plot(q.lambda_fp,q.delta_weighted_loss_km2eq,color='black',lw=1.65,ls=ls)[0],f'analytical_curve_{region}');gid(ba.scatter(pts.lambda_fp,pts.delta_weighted_loss_km2eq,s=17,marker=mk,color='black',zorder=4),f'preset_markers_{region}')
        formula=f"{'BB' if region=='Beijing_Baoding' else 'HB'}: {r.delta_FN_km2:.2f} {'+' if r.delta_FP_km2>0 else '−'} {abs(r.delta_FP_km2):.2f}λ"
        ba.text(.16 if region=='Beijing_Baoding' else 1.68,30 if region=='Beijing_Baoding' else 13,formula,fontsize=7,va='center')
    gid(ba.scatter([root],[0],s=24,color='white',edgecolors='black',linewidths=.9,zorder=5),'break_even_point');ba.annotate(f'λ* = {root:.2f}',xy=(root,0),xytext=(2.50,6.4),fontsize=7,ha='left',va='center',arrowprops=dict(arrowstyle='->',color='black',lw=.7,shrinkA=1,shrinkB=3));ba.set_xticks([0,1,2,3,4]);ba.set_yticks([-5,0,15,30]);ba.set_xlabel('FP-to-FN cost weight (λ)',labelpad=1);ba.set_ylabel('ΔL (km²-eq)',labelpad=2);clean(ba);txt(6,131,'BB: no non-negative crossover; point estimates')
    # c: all four candidates retained; square-root colour scale exposes small positive gaps.
    title(95,77,'c','Candidate regret at 10%');txt(99,83,'Gap from observed minimum');norm=PowerNorm(.5,vmin=0,vmax=500)
    for j,region in enumerate(REGIONS):
        x=113+j*34;ca=ax(x,94,30,24);ca.set_xlim(0,3);ca.set_ylim(4,0);txt(x+15,90,'BB' if j==0 else 'Harbin',bold=True,ha='center')
        for ri,m in enumerate(MODELS):
            for ci,w in enumerate(WEIGHTS):
                r=regret[regret.region.eq(region)&regret.model.eq(m)&regret.lambda_fp.eq(w)].iloc[0];v=float(r.regret_km2eq);assert 0<=v<=500;gid(ca.add_patch(Rectangle((ci,ri),1,1,facecolor=REGRET(norm(v)),edgecolor='black',linewidth=.4)),f'regret_cell_{region}_{ri}_{ci}');ca.text(ci+.53,ri+.5,'0' if v==0 else f'{v:.2f}',fontsize=7,ha='center',va='center',color='white' if norm(v)>.70 else 'black')
                if v==0:gid(ca.scatter([ci+.21],[ri+.5],color='black',marker='D',s=7,zorder=3),f'minimum_symbol_{region}_{ri}_{ci}')
        ca.set_xticks([.5,1.5,2.5],['0.25','1','4']);ca.set_yticks(np.arange(4)+.5,[NAME[m] for m in MODELS] if j==0 else []);ca.set_xlabel('λ',labelpad=0);clean(ca)
        if j==0:
            for t,m in zip(ca.get_yticklabels(),MODELS):t.set_color(C[m])
    cbax=ax(113,127,42,1.5);cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=REGRET),cax=cbax,orientation='horizontal',ticks=[0,25,100,500]);cb.ax.tick_params(length=1.5,pad=.5);cb.outline.set_linewidth(.65);cb.solids.set_rasterized(False)
    fig.add_artist(Line2D([162/W],[1-128/H],marker='D',color='black',markersize=3,ls='none',transform=fig.transFigure));txt(165,128,'Minimum');txt(109,132,'Regret (km²-eq; square-root colour scale)')
    # d: fixed endpoint scenarios in both regions, all six primary scenarios retained in data.
    title(2,136,'d','Retrospective scenario ledger')
    txt(6,142,'10% area · areas: km²; Gap / L: km²-eq');line(138,144,142,lw=1.7);txt(146,142,'50 km² (FN / FP)')
    for x,s,ha in [(6,'Region','left'),(24,'λ','center'),(43,'Lowest','center'),(65,'Second','center'),(86,'Gap','right'),(111,'Reviewed','right'),(129,'FN','center'),(151,'FP','center'),(178,'L','right')]:txt(x,147,s,bold=True,ha=ha,color=C['FN'] if s=='FN' else 'black')
    line(4,178,149.3,lw=.55);glyph_scale=.12
    for i,(region,w) in enumerate([(r,w) for r in REGIONS for w in [.25,4.]]):
        r=ledger[ledger.region.eq(region)&ledger.lambda_fp.eq(w)].iloc[0];y=153+4.5*i
        txt(6,y,'BB' if region=='Beijing_Baoding' else 'HB');txt(24,y,f'{w:g}',ha='center');txt(43,y,NAME[r.best_model],ha='center',color=C[r.best_model],bold=True);txt(65,y,NAME[r.second_model],ha='center',color=C[r.second_model]);txt(86,y,f'{r.margin_km2eq:.2f}',ha='right');txt(111,y,f'{r.reviewed_valid_area_km2:,.2f}',ha='right');txt(178,y,f'{r.best_loss_km2eq:.2f}',ha='right')
        for endpoint,x,c in [('FN',119,C['FN']),('FP',142,C['FP'])]:
            v=float(r['residual_'+endpoint+'_km2']);length_mm=v*glyph_scale;gid(box(x,y+.9,length_mm,.55,c,z=1),f'ledger_area_{region}_{w}_{endpoint}');txt(x+10,y-.5,f'{v:.2f}',ha='center');geometry.append(dict(role='ledger_area',region=region,budget_fraction=.1,lambda_fp=w,endpoint=endpoint,value=v,length_mm=length_mm,scale_mm_per_unit=glyph_scale))
        if i==1:line(4,178,y+2.2,lw=.35)
    pd.DataFrame(geometry).to_csv(SD/'rendered_quantitative_geometry.csv',index=False)
    metadata=dict(width_mm=W,height_mm=H,font=args.font,body_minimum_pt=7,panels=4,axis_width_pt=1.15,tick_direction='in',fully_vector=True,margin_mm_per_km2eq=margin_scale,ledger_mm_per_km2=glyph_scale,crossover_strip_x_mm=14+root/4*(71*4/4.12),crossover_plot_x_mm=14+root/4.12*71,regret_colour=dict(transform='square root',gamma=.5,minimum=0,maximum=500),vector_roles=roles)
    (AS/('render_metadata'+suffix+'.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8');fig.savefig(AS/f'Fig06_round{args.round}{suffix}.png',dpi=240);fig.savefig(AS/f'Fig06_round{args.round}{suffix}.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
    for t in texts:
        bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted());bounds.append(dict(text=t.get_text(),x0_mm=bb.x0*W,y0_mm=(1-bb.y1)*H,x1_mm=bb.x1*W,y1_mm=(1-bb.y0)*H))
    (AS/('text_bounds'+suffix+'.json')).write_text(json.dumps(bounds,indent=2),encoding='utf-8')
    all_bounds=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted());all_bounds.append(dict(text=t.get_text(),x0_mm=bb.x0*W,y0_mm=(1-bb.y1)*H,x1_mm=bb.x1*W,y1_mm=(1-bb.y0)*H))
    (AS/('all_text_bounds'+suffix+'.json')).write_text(json.dumps(all_bounds,indent=2),encoding='utf-8')
    if args.final:
        stem=OUT/('Fig06_main'+suffix);fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.png',dpi=1200);Image.MAX_IMAGE_PIXELS=160_000_000
        with Image.open(str(stem)+'.png') as im:im.convert('RGB').save(str(stem)+'.tiff',compression='tiff_lzw',dpi=(1200,1200))
    plt.close(fig);print('Rendered',args.round,args.font,flush=True)
if __name__=='__main__':main()
