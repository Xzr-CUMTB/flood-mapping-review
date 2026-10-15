from figure_common import *
def main():
    f=Canvas(5,185,args());data=f.read('all_core_map_metrics.csv');cells=f.read('benchmark_cells.csv');meta=f.read('model_provenance.csv');models=meta.model.tolist();metrics=['flood_iou','fn_km2','fp_km2','brier','confident_omission_01'];regions=['Beijing_Baoding','Harbin'];names=['SAR','Optical','RGB + SAR','RGB-NIR + SAR','Ordinary Gate','Value Gate','Probability average','Lower-entropy expert','Inverse-entropy blend','Stacker','TerraMind tiny','TerraMind tiny (random)','Fusion ensemble','SAR + temperature','Fusion + temperature','Value + temperature'];f.panel_bounds=dict(a=[0,0,130,185],b=[131,0,180,96],c=[131,97,180,185]);f.box(1,1,128,183,'white',1);f.box(131,1,48,95,'white',1);f.box(131,98,48,86,'white',1)
    f.title(2,4,'a','Complete frozen-test benchmark');aa=f.ax(36,24,91,128);aa.set_xlim(0,10);aa.set_ylim(16,0)
    for i,model in enumerate(models):
        for j,region in enumerate(regions):
            for k,metric in enumerate(metrics):
                r=cells[cells.model.eq(model)&cells.region.eq(region)&cells.metric.eq(metric)].iloc[0];col=j*5+k;effect=r.colour_value;bg=DIV((effect+1)/2);f.gid(aa.add_patch(Rectangle((col,i),1,1,facecolor=bg,edgecolor='black',lw=.3)),f'benchmark_cell_{i}_{j}_{k}');s=f'{r.value:.1f}' if metric in ['fn_km2','fp_km2'] else f'{r.value:.4f}' if metric in ['flood_iou','brier'] else f'{r.value:.3f}';aa.text(col+.5,i+.5,s,ha='center',va='center',fontsize=8,color='white' if abs(effect)>.75 else 'black')
    aa.set_xticks([]);aa.set_yticks(np.arange(16)+.5,names);aa.tick_params(axis='y',length=0,pad=3);aa.axvline(5,c='black',lw=1)
    for i in [2,4,6,10,12,13]:aa.axhline(i,c='black',lw=1)
    for j,region in enumerate(['Beijing–Baoding','Harbin']):
        f.text(36+(j+.5)*91/2,13,region,ha='center',bold=True)
        for k,label in enumerate(['IoU','FN','FP','Brier','C(0.1)']):f.text(36+(j*5+k+.5)*91/10,20,label,ha='center',bold=True)
    blocks=[(0,2,'I'),(2,4,'II'),(4,6,'III'),(6,10,'IV'),(10,12,'V'),(12,13,'VI'),(13,16,'VII')]
    for lo,hi,tag in blocks:f.box(1,24+lo*8,1,8*(hi-lo),C['neutral'],.6);f.text(4,24+(lo+hi)*4,tag,ha='center')
    f.text(5,158,'FN / FP: km² · C(0.1): confident omission fraction at p ≤ 0.1')
    labels=['I Single modality','II Multimodal experts','III Learned routing','IV Combinations','V Foundation controls','VI Ensemble','VII Calibrated outputs']
    for i,label in enumerate(labels):f.text(5+(i%2)*62,164+(i//2)*4,label)
    f.text(5,181,'Colour: favourable effect from Fusion, separately scaled by metric')
    cb=f.fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1,1),cmap=DIV),cax=f.ax(86,174,30,2),orientation='horizontal',ticks=[-1,0,1]);cb.ax.tick_params(length=1,pad=.5);cb.solids.set_rasterized(False)
    f.title(133,4,'b','Pretraining / ensemble')
    pairs=f.read('paired_controls.csv');m=data.set_index(['region','model'])
    for comparison,yy in [('Pretrained minus random',17),('Ensemble minus fusion',57)]:
        f.text(135,yy,'TerraMind pretraining' if yy==17 else 'Three-member ensemble',bold=True)
        a=f.ax(145,yy+10,31,19);a.set_ylim(1.6,-.6);a.set_xlim(.25,.70);a.set_yticks([0,1],['BB','HB']);a.set_xticks([.3,.5,.7]);a.set_xlabel('Flood IoU',labelpad=1)
        for i,region in enumerate(regions):
            r=pairs[pairs.region.eq(region)&pairs.comparison.eq(comparison)&pairs.metric.eq('flood_iou')].iloc[0];f.gid(a.plot([r.second_value,r.first_value],[i,i],color='black',lw=.7)[0],f'paired_control_{yy}_{i}');a.scatter(r.second_value,i,s=18,facecolors='white',edgecolors='black',lw=.8,zorder=3);a.scatter(r.first_value,i,s=18,color=C['value'] if r.difference>0 else C['error'],edgecolors='black',lw=.4,zorder=3);a.annotate(f'{100*r.difference:+.2f} pp',(np.mean([r.first_value,r.second_value]),i),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    f.text(134,95,'Open: baseline; filled: modified')
    f.title(133,102,'c','Capacity and IoU');capacity=f.read('capacity_comparison.csv');colmap=dict(sar=C['sar'],optical=C['optical'],rgb_fusion=C['random'],fusion=C['fusion'],terramind_tiny=C['value'],terramind_tiny_random=C['neutral']);markers=dict(sar='s',optical='^',rgb_fusion='v',fusion='D',terramind_tiny='o',terramind_tiny_random='X')
    for j,region in enumerate(regions):
        yy=114+j*28;f.text(135,yy-4,'Beijing–Baoding' if j==0 else 'Harbin',bold=True);a=f.ax(145,yy,31,20);a.set_xlim(1.2,6.6);a.set_ylim(0,.75);a.set_xticks([1.964529,5.873617],['1.96','5.87']);a.set_yticks([0,.3,.6]);a.set_ylabel('IoU',labelpad=1)
        for r in capacity[capacity.region.eq(region)].itertuples():f.gid(a.scatter(r.parameters/1e6,r.flood_iou,s=19,color=colmap[r.model],marker=markers[r.model],edgecolor='black',lw=.35),f'capacity_{region}_{r.model}')
    f.text(160,168,'Network parameters (million)',ha='center')
    for i,(label,model) in enumerate([('SAR','sar'),('Optical','optical'),('RGB+SAR','rgb_fusion'),('Fusion','fusion'),('TM pre.','terramind_tiny'),('TM rand.','terramind_tiny_random')]):
        x=134+(i%2)*23;y=173+(i//2)*4;aa=f.ax(x,y-1,2,2);aa.scatter([0],[0],s=13,color=colmap[model],marker=markers[model],edgecolor='black',lw=.3);aa.axis('off');f.text(x+3,y,label)
    f.finish()
if __name__=='__main__':main()
