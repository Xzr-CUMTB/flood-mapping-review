from figure_common import *
def main():
    f=Canvas(3,150,args());p=f.read('crossfit_provenance.csv');f.panel_bounds=dict(a=[0,0,99,150],b=[100,0,180,107],c=[100,108,180,150]);f.box(1,1,97,148,'white',1);f.box(101,1,78,106,'white',1)
    f.title(2,4,'a','Verified outer–inner exclusion');f.text(4,11,'Outer experts and value head');aa=f.ax(27,19,48,24);aa.set_xlim(0,4);aa.set_ylim(4,0)
    for i in range(4):
        for j in range(4):
            f.gid(aa.add_patch(Rectangle((j,i),1,1,facecolor=to_rgba(C['gold'] if i==j else C['value'],.24),edgecolor='black',lw=.5)),f'outer_matrix_{i}_{j}');aa.text(j+.5,i+.5,'H' if i==j else 'T',ha='center',va='center')
    aa.set_yticks(np.arange(4)+.5,['Out E1','Out E2','Out E3','Out E4']);aa.set_xticks(np.arange(4)+.5,['E1','E2','E3','E4']);f.text(27,15,'Training event groups');f.text(4,51,'T: training   H: held out; inner grids omit the outer event')
    counts=f.read('oof_product_index.csv.gz').groupby('outer_fold').size()
    for fold in range(4):
        x=5+(fold%2)*47;yy=62+(fold//2)*37;target=EVENTS[fold];others=[e for e in EVENTS if e!=target];f.text(x,yy,f'Outer E{fold+1} · {counts[fold]:,} OOF tiles',bold=True)
        a=f.ax(x+11,yy+10,27,18);a.set_xlim(0,3);a.set_ylim(3,0)
        for i,t in enumerate(others):
            actual=p[p.outer_fold.eq(fold)&p.role.eq('inner')&p.target_event.eq(t)&p.modality.eq('sar')].iloc[0];train=actual.training_groups.split(';')
            for j,e in enumerate(others):
                used=e in train;f.gid(a.add_patch(Rectangle((j,i),1,1,facecolor=to_rgba(C['value'] if used else C['gold'],.24),edgecolor='black',lw=.5)),f'inner_matrix_{fold}_{i}_{j}');a.text(j+.5,i+.5,'T' if used else 'H',ha='center',va='center')
        labels=['E'+str(EVENTS.index(e)+1) for e in others];a.set_xticks(np.arange(3)+.5,labels);a.set_yticks(np.arange(3)+.5,labels);f.text(x,yy+6,'Inner target × expert training')
    for i,label in enumerate(['E1  2020 flood season','E2  2021 Henan','E3  2024 April','E4  2024 June']):f.text(5+(i%2)*47,136+(i//2)*5,label)
    f.text(5,147,'8,786 distinct development tiles → full-system OOF')
    f.title(102,4,'b','Supervision provenance')
    def node(x,y,w,h,s,col,rounded=False):f.gid(f.box(x,y,w,h,col,.15,edge='black',rounded=rounded),f'node_{s.splitlines()[0]}');f.text(x+w/2,y+h/2,s,ha='center')
    node(104,15,38,10,'SAR / Fusion\ninner OOF probabilities',C['fusion']);node(149,15,27,10,'Development\nreference y',C['neutral'])
    node(104,32,72,9,'v* = (s − y)² − (f − y)²',C['gold']);f.arrow(123,25,123,32);f.arrow(162,25,162,32,True)
    node(104,47,32,10,'Value-head weights\ng = sigmoid(v̂/0.05)',C['value'],True);node(145,47,31,10,'Full-system\nouter OOF p',C['fusion']);f.arrow(120,41,120,47,True);f.arrow(136,52,145,52)
    node(104,64,72,10,'Omission head: BCE(y) on valid p < 0.5\nScore = sigmoid(logit) × 1[p < 0.5]',C['value'],True);f.arrow(161,57,161,64);f.line(176,20,178,20,ls='--');f.line(178,20,178,69,ls='--');f.arrow(178,69,176,69,True)
    node(104,81,43,9,'Frozen models + heads',C['sar'],True);node(151,81,25,9,'Core labels',C['gold']);f.arrow(125,74,125,81);node(104,96,72,8,'2023 core evaluation',C['sar']);f.arrow(125,90,125,96);f.arrow(164,90,164,96,True)
    f.text(104,109,'Solid: prediction/dependency · dashed: supervision')
    f.title(102,115,'c','Computational integrity');rows=f.read('integrity_counts.csv');f.text(104,121,'Evidence',bold=True);f.text(154,121,'Expected',ha='right',bold=True);f.text(176,121,'Observed',ha='right',bold=True)
    for i,r in rows.iterrows():
        yy=125+i*2.8;f.text(104,yy,r.evidence);f.text(154,yy,f'{r.expected:,}',ha='right');f.text(176,yy,f'{r.observed:,}',ha='right',color=C['value']);f.gid(f.line(103,yy+1.3,177,yy+1.3,lw=.25),f'integrity_row_{i}')
    f.text(104,148,'Release: 093a72b5…ff242 · seed 17; ensemble 29/43')
    f.finish()
if __name__=='__main__':main()
