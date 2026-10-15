from figure_common import *
def main():
    f=Canvas(1,160,args());d=f.read('site_dates.csv');groups=f.read('event_composition.csv').set_index('event_group').loc[GROUPS];f.panel_bounds=dict(a=[0,0,180,78],b=[0,79,95,127],c=[96,79,180,127],d=[0,128,180,160])
    f.box(1,1,178,76,'white',1);f.title(2,4,'a','Published site-level observation chronology')
    for x,m,label,col in [(5,'o','Optical',C['optical']),(33,'D','Flood reference',C['error']),(75,'s','SAR',C['sar'])]:
        a=f.ax(x,8,3,3);a.scatter([0],[0],marker=m,s=16,facecolors='none' if m=='s' else col,edgecolors=col);a.axis('off');f.text(x+4,9.5,label)
    for x,s in [(123,'Optical date'),(145,'Reference date'),(167,'SAR date')]:f.text(x,15,s,ha='center',bold=True)
    a=f.ax(28,19,79,48);a.set_xlim(-218,15);a.set_ylim(8.6,-.6)
    for i,r in d.iterrows():
        c=EC[r.conservative_group];a.axhspan(i-.47,i+.47,color=c,alpha=.09);f.gid(a.plot([r.optical_relative_reference_days,r.sar_delay_days],[i,i],color='black',lw=.65)[0],f'chronology_{r.site}');a.scatter([r.optical_relative_reference_days],[i],s=18,color=C['optical'],edgecolor='black',lw=.4,zorder=3);a.scatter([0],[i],marker='D',s=12,color=C['error'],zorder=4);a.scatter([r.sar_delay_days],[i],marker='s',s=35,facecolors='none',edgecolors=C['sar'],lw=.9,zorder=5)
        yy=19+(i+.6)/9.2*48;f.text(26,yy,r.site+' '+str(pd.to_datetime(r.sar_date).year)[2:],ha='right',color='black');
        for x,col in [(123,'optical_date'),(145,'paper_flood_date'),(167,'sar_date')]:f.text(x,yy,str(r[col])[:10],ha='center')
        if r.membership=='frozen core':f.line(110,yy-1.5,110,yy+1.5,C['gold'],2)
    a.set_yticks([]);a.set_xticks([-200,-100,0]);a.set_xlabel('Days relative to site reference flood date',labelpad=1);a.axvline(0,color='black',lw=.6,ls=':');f.text(4,76,'Gold edge: 2023 core · publication dates; provisional geographic site association')
    f.title(2,83,'b','Rotating event roles');roles=f.read('event_roles.csv');aa=f.ax(17,94,68,25);aa.set_xlim(0,6);aa.set_ylim(5,0)
    stage=['Outer 1','Outer 2','Outer 3','Outer 4','Final fit','Core test'];colors={'training':C['value'],'held out':C['gold'],'evaluation':C['fusion'],'unused':'white'};symbol={'training':'T','held out':'H','evaluation':'E','unused':'–'}
    for i,e in enumerate(GROUPS):
        for j,s in enumerate(stage):
            role=roles[roles.event_group.eq(e)&roles.stage.eq(s)].role.iloc[0];f.gid(aa.add_patch(Rectangle((j,i),1,1,facecolor=to_rgba(colors[role],.24) if role!='unused' else 'white',edgecolor='black',lw=.45)),f'role_{i}_{j}');aa.text(j+.5,i+.5,symbol[role],ha='center',va='center')
    aa.set_yticks(np.arange(5)+.5,['E1','E2','E3','E4','C']);aa.set_xticks(np.arange(6)+.5,['O1','O2','O3','O4','Final','Core']);f.text(5,90,'O1–O4: outer folds; T train, H held out, E evaluate');f.text(4,125,'E1–E4: development; C: entire frozen 2023 core')
    f.title(97,83,'c','Age versus observation delay');xx=np.arange(9);codes=['Nan','Hef','Zhe','Heb','Bei','Bao','Har','Wuz','Gui']
    for name,y,h,vals,col,lim in [('Optical age',95,11,d.optical_age_days,C['optical'],235),('SAR delay',114,8,d.sar_delay_days,C['sar'],10)]:
        a=f.ax(110,y,66,h);a.vlines(xx,0,vals,color=col,lw=1.0);f.gid(a.scatter(xx,vals,s=12,color=col),name.replace(' ','_'));a.set_xlim(-.6,8.6);a.set_ylim(0,lim);a.set_yticks([0,200] if lim>20 else [0,9]);a.set_xticks(xx,codes if lim<20 else []);f.text(108,y-4,name+' (days)');
        for i,v in enumerate(vals):a.annotate(str(v),(i,v),xytext=(0,2),textcoords='offset points',ha='center',fontsize=8)
    f.title(2,132,'d','Retained sample and geographic composition');f.text(178,132,'8,786 development + 8,233 core = 17,019 tiles',ha='right')
    for x,s,ha in [(5,'Group','left'),(47,'Tiles','right'),(70,'Count scale','center'),(106,'Valid area (km²)','right'),(143,'Longitude extent','center'),(169,'Latitude extent','center')]:f.text(x,138,s,bold=True,ha=ha)
    for i,(e,r) in enumerate(groups.iterrows()):
        yy=142+3.5*i;f.text(5,yy,('E'+str(i+1)+': ' if i<4 else 'C: ')+GN[i]);f.text(47,yy,f'{int(r.tiles):,}',ha='right');f.gid(f.box(50,yy-.65,r.tiles*.0042,1.3,GC[i]),f'count_bar_{i}');f.text(106,yy,f'{r.valid_area_km2:,.1f}',ha='right');f.text(143,yy,f'{r.lon_min:.2f}–{r.lon_max:.2f}°E',ha='center');f.text(169,yy,f'{r.lat_min:.2f}–{r.lat_max:.2f}°N',ha='center')
    f.finish()
if __name__=='__main__':main()
