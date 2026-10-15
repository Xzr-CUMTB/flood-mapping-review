from figure_common import *
def main():
    f=Canvas(4,220,args());m=f.read('all_oof_metrics.csv');effects=f.read('effects_vs_fusion.csv');groups=f.read('event_composition.csv').set_index('event_group');ranks=f.read('core_model_ranks.csv');f.panel_bounds=dict(a=[0,0,180,71],b=[0,72,91,130],c=[92,72,180,130],d=[0,131,99,182],e=[100,131,180,182],f=[0,183,180,220])
    f.box(1,1,178,70,'white',1);f.box(1,73,178,57,'white',1);f.box(1,132,178,48,'white',1)
    f.title(2,4,'a','Complete outer-event performance');metrics=['flood_iou','brier','fn_km2','fp_km2'];a=f.ax(30,21,147,39);a.set_xlim(0,16);a.set_ylim(6,0)
    for i,model in enumerate(MODEL):
        for j,event in enumerate(EVENTS):
            for k,metric in enumerate(metrics):
                r=effects[effects.model.eq(model)&effects.event_group.eq(event)&effects.metric.eq(metric)].iloc[0];col=j*4+k;bg=DIV((r.colour_value+1)/2);f.gid(a.add_patch(Rectangle((col,i),1,1,facecolor=bg,edgecolor='black',lw=.3)),f'performance_cell_{i}_{j}_{k}');a.text(col+.5,i+.5,f'{r.value:.4f}' if k<2 else f'{r.value:.1f}',ha='center',va='center',color='white' if abs(r.colour_value)>.75 else 'black')
    a.axhline(4,color='black',lw=1);a.set_yticks(np.arange(6)+.5,MN);a.set_xticks([]);a.tick_params(axis='y',length=0,pad=3)
    for j,label in enumerate(['E1 · 2020 season','E2 · 2021 Henan','E3 · 2024 April','E4 · 2024 June']):
        f.text(30+(j+.5)*147/4,12,label,ha='center',bold=True)
        for k,label2 in enumerate(['IoU','Brier','FN km²','FP km²']):f.text(30+(j*4+k+.5)*147/16,18,label2,ha='center')
        if j:a.axvline(j*4,color='black',lw=.9)
    f.text(4,65,'Last two rows: exploratory development controls');f.text(4,70,'Colour: favourable effect from Fusion / per-metric maximum |effect|')
    cb=f.fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1,1),cmap=DIV),cax=f.ax(135,65,30,2),orientation='horizontal',ticks=[-1,0,1]);cb.ax.tick_params(length=1,pad=1);cb.solids.set_rasterized(False);f.text(177,62.5,'+ favourable',ha='right')
    f.title(2,77,'b','Value minus Fusion');q=effects[effects.model.eq('sigmoid_value')];eventnames=['E1','E2','E3','E4']
    for j,(metric,scale,label) in enumerate([('flood_iou',100,'ΔIoU (pp)'),('brier',1000,'ΔBrier × 1000')]):
        aa=f.ax(17+j*43,90,29,27);v=[q[q.event_group.eq(e)&q.metric.eq(metric)].difference.iloc[0]*scale for e in EVENTS];aa.axvline(0,color='black',lw=.7);f.gid(aa.scatter(v,range(4),s=21,color=[C['value'] if x*(1 if metric=='flood_iou' else -1)>0 else C['error'] for x in v]),f'paired_effect_{metric}');aa.set_ylim(3.6,-.6);aa.set_yticks(range(4),eventnames if j==0 else []);aa.set_xlabel(label,labelpad=1);lo,hi=min(v),max(v);pad=(hi-lo)*.30;aa.set_xlim(lo-pad,hi+pad);aa.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        for i,x in enumerate(v):aa.annotate(f'{x:+.2f}',(x,i),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    f.text(4,126,'Four event estimates; observed signs retained')
    f.title(96,77,'c','FN–FP trade-off');xy=[]
    for e in EVENTS:xy.append([q[q.event_group.eq(e)&q.metric.eq(mm)].difference.iloc[0] for mm in ['fp_km2','fn_km2']])
    xy=np.array(xy);aa=f.ax(113,92,58,28);aa.set_xlim(-20,15);aa.set_ylim(-3,16);aa.axhspan(-3,0,color=C['value'],alpha=.07);aa.axvspan(-20,0,color=C['value'],alpha=.07);aa.axvline(0,color='black',lw=.75);aa.axhline(0,color='black',lw=.75)
    for i,(x,y) in enumerate(xy):
        f.gid(aa.scatter(x,y,s=22,color=GC[i],edgecolor='black',lw=.4),f'joint_effect_{i}')
        offset=(5,0) if i==0 else (-5,5) if i==3 else (4,4)
        aa.annotate('E'+str(i+1),(x,y),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if i==3 else 'left',va='center' if i==0 else 'bottom')
    aa.set_xlabel('ΔFP (km²)',labelpad=1);aa.set_ylabel('ΔFN (km²)',labelpad=1);top=f.ax(113,87,58,3);top.set_xlim(aa.get_xlim());top.plot(xy[:,0],np.zeros(4),'|',color='black',ms=5);top.axis('off');side=f.ax(174,92,3,28);side.set_ylim(aa.get_ylim());side.plot(np.zeros(4),xy[:,1],'_',color='black',ms=5);side.axis('off');f.text(100,126,'Negative = less error area')
    f.title(2,136,'d','Probability quality and event context');aa=f.ax(17,149,45,25);aa.set_ylim(3.5,-.5)
    for i,event in enumerate(EVENTS):
        sel=m[m.event_group.eq(event)].set_index('model');vs=[sel.loc[k,'brier'] for k in ['fusion','probability_average','sigmoid_value']];aa.plot([min(vs),max(vs)],[i,i],c='black',lw=.6)
        for model,col,mk in [('fusion',C['fusion'],'o'),('probability_average',C['average'],'s'),('sigmoid_value',C['value'],'D')]:aa.scatter(sel.loc[model,'brier'],i,c=col,s=15,marker=mk)
        r=groups.loc[event];f.text(78,149+(i+.5)/4*25,f'{int(r.tiles):,}',ha='center');f.text(95,149+(i+.5)/4*25,f'{r.flood_fraction*100:.1f}%',ha='right')
    aa.set_yticks(range(4),eventnames);aa.set_xlabel('Brier score',labelpad=1);aa.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3));f.text(78,145,'Tiles',ha='center',bold=True);f.text(95,145,'Flood',ha='right',bold=True)
    f.title(102,136,'e','IoU rank transitions');aa=f.ax(115,149,58,25)
    for i,model in enumerate(MODEL[:4]):
        vals=[ranks[ranks.event_group.eq(e)&ranks.model.eq(model)].rank_IoU.iloc[0] for e in EVENTS];f.gid(aa.plot(range(4),vals,color=MC[i],lw=1.2,marker=['o','s','^','D'][i],ms=3)[0],f'rank_braid_{model}')
    aa.set_ylim(4.3,.7);aa.set_yticks([1,2,3,4]);aa.set_xticks(range(4),eventnames);aa.set_ylabel('Rank',labelpad=1)
    for i,(name,col) in enumerate(zip(['SAR','Fusion','Average','Value'],MC[:4])):f.line(10+i*42,181,14+i*42,181,col,1.4);f.text(16+i*42,181,name)
    f.title(2,188,'f','Complete event-level effect lattice');cols=[('Fusion − SAR','ΔIoU (pp)'),('Value − Fusion','ΔIoU (pp)'),('Value − Fusion','ΔBrier × 1000'),('Value − Fusion','ΔFN (km²)'),('Value − Fusion','ΔFP (km²)')];vals=[]
    for e in EVENTS:
        mm=m[m.event_group.eq(e)].set_index('model');vals.append([(mm.loc['fusion','flood_iou']-mm.loc['sar','flood_iou'])*100,(mm.loc['sigmoid_value','flood_iou']-mm.loc['fusion','flood_iou'])*100,(mm.loc['sigmoid_value','brier']-mm.loc['fusion','brier'])*1000,mm.loc['sigmoid_value','fn_km2']-mm.loc['fusion','fn_km2'],mm.loc['sigmoid_value','fp_km2']-mm.loc['fusion','fp_km2']])
    vals=np.array(vals);sc=np.max(abs(vals),axis=0);aa=f.ax(18,201,159,16);aa.set_xlim(0,5);aa.set_ylim(4,0)
    for i in range(4):
        for j in range(5):v=vals[i,j];signed=v*(1 if j<2 else -1)/sc[j];f.gid(aa.add_patch(Rectangle((j,i),1,1,facecolor=DIV((signed+1)/2),edgecolor='black',lw=.3)),f'synthesis_{i}_{j}');aa.text(j+.5,i+.5,f'{v:+.2f}',ha='center',va='center',color='white' if abs(signed)>.75 else 'black')
    aa.set_xticks([]);aa.set_yticks(np.arange(4)+.5,eventnames)
    for j,(one,two) in enumerate(cols):f.text(18+(j+.5)*159/5,194,one,ha='center');f.text(18+(j+.5)*159/5,198,two,ha='center')
    f.finish()
if __name__=='__main__':main()
