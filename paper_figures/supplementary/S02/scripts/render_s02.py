from figure_common import *
from matplotlib.ticker import ScalarFormatter,FuncFormatter
from pyproj import Geod
GEOD=Geod(ellps='WGS84')
def main():
    f=Canvas(2,200,args());d=f.read('tile_audit.csv.gz');keep=d.retained;groups=f.read('event_composition.csv').set_index('event_group').loc[GROUPS];z=np.load(f.sd/'displayed_scene.npz');extent=z['extent'];f.panel_bounds=dict(a=[0,0,94,62],b=[94,0,180,62],c=[0,63,96,151],d=[96,63,180,151],e=[0,152,97,200],f=[97,152,180,200]);f.box(1,1,178,60,'white',1);f.box(1,63,178,87,'white',1);f.box(1,152,178,47,'white',1)
    f.title(2,4,'a','Exact-ID inclusion audit');st=f.read('inclusion_stages.csv');a=f.ax(35,14,49,30);names=['Raw paired','Named metadata','Valid / unique','Non-overlap'];vals=st['count'].to_numpy()[:4]
    a.set_xlim(0,24000);a.set_ylim(3.7,-.6)
    for i,(v,c) in enumerate(zip(vals,[C['neutral'],C['neutral'],C['sar'],C['value']])):
        f.gid(a.barh(i,v,height=.46,color=c,edgecolor='black',lw=.4)[0],f'inclusion_bar_{i}');a.text(v+350,i,f'{v:,}',va='center',fontsize=8)
    a.set_yticks(range(4),names);a.set_xticks([0,10000,20000],['0','10k','20k']);a.set_xlabel('Tiles',labelpad=1);f.text(5,53,'Retained partition',bold=True);f.text(38,53,'Development 8,786',color=C['value']);f.text(38,58,'Frozen core 8,233',color=C['gold']);f.text(5,10,'85,932 original files · four files per paired tile')
    f.title(97,4,'b','Offset and common support');f.text(102,10,'Retained 17,019',color=C['value']);f.text(139,10,'Excluded 4,464',color=C['error'])
    a=f.ax(112,23,53,25);top=f.ax(112,15,53,6);side=f.ax(168,23,8,25)
    for mask,col,label in [(~keep,C['error'],'Excluded'),(keep,C['value'],'Retained')]:
        f.gid(a.scatter(d.loc[mask,'offset_px'],d.loc[mask,'coverage_fraction'],s=.5,c=col,alpha=.4,linewidths=0),f'coverage_points_{label}');top.hist(d.loc[mask,'offset_px'],bins=np.r_[np.linspace(0,1.1,24),np.geomspace(1.2,600,14)],histtype='step',color=col,lw=.75);side.hist(d.loc[mask,'coverage_fraction'],bins=np.linspace(0,1,21),orientation='horizontal',histtype='step',color=col,lw=.75)
    a.set_xscale('symlog',linthresh=1);top.set_xscale('symlog',linthresh=1);a.set_xlim(0,620);top.set_xlim(0,620);a.set_ylim(-.03,1.03);side.set_ylim(-.03,1.03);a.set_xticks([0,1,100,600],['0','1','100','600']);a.set_yticks([0,.5,1]);a.set_xlabel('Original grid offset (pixels; symlog)',labelpad=1);a.set_ylabel('Valid pixel fraction',labelpad=1);top.axis('off');side.axis('off');f.text(102,58,'Retained maximum: 0.966 pixels',color=C['value'])
    # Exact geographical plate, fixed 0--255 intensity, all six layers share extent and aspect.
    f.title(2,67,'c','Co-registered six-channel observations');sid=f.js('scene_provenance.json')['sample_id'];f.text(5,73,f'Beijing–Baoding · tile {sid} · native 256 × 256 window')
    xx=z['x'];labels=z['label'];valid=z['support']==7;lat=(extent[2]+extent[3])/2;aspect=1/np.cos(np.deg2rad(lat));width=22;height=width*(extent[3]-extent[2])/(extent[1]-extent[0])*aspect
    def image_axes(x,y,w,h,img,cmap=None,clim=None,boundary=False,scale=True):
        aa=f.ax(x,y,w,h);aa.imshow(img,extent=extent,cmap=cmap,interpolation='nearest',vmin=clim[0] if clim else None,vmax=clim[1] if clim else None,aspect=aspect);aa.set_xticks([]);aa.set_yticks([])
        if boundary:
            xcoord=np.linspace(extent[0]+(extent[1]-extent[0])/512,extent[1]-(extent[1]-extent[0])/512,256);ycoord=np.linspace(extent[3]-(extent[3]-extent[2])/512,extent[2]+(extent[3]-extent[2])/512,256);f.gid(aa.contour(xcoord,ycoord,(labels==1).astype(float),levels=[.5],colors=[C['error']],linewidths=.55),f'reference_edge_{x}_{y}')
        if scale:
            left=extent[0]+.06*(extent[1]-extent[0]);bottom=extent[2]+.08*(extent[3]-extent[2]);right,_,_=GEOD.fwd(left,bottom,90,500);aa.plot([left,right],[bottom,bottom],color='white',lw=2.2);aa.plot([left,right],[bottom,bottom],color='black',lw=.85);aa.text((left+right)/2,bottom+.025*(extent[3]-extent[2]),'500 m',ha='center',va='bottom',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.2))
        return aa
    layers=[('HH',xx[0],'gray',(0,255)),('HV',xx[1],'gray',(0,255)),('RGB',xx[2:5].transpose(1,2,0)/255,None,None),('NIR',xx[5],'gray',(0,255)),('Reference',np.where(labels==1,1,0),LinearSegmentedColormap.from_list('ref',['white',C['sar']]),(0,1)),('Valid mask',valid.astype(int),LinearSegmentedColormap.from_list('valid',['#D7D7D7',C['value']]),(0,1))]
    for i,(label,img,cmap,clim) in enumerate(layers):
        x=7+(i%3)*29;y=81+(i//3)*(height+7);f.text(x+width/2,y-3,label,ha='center',bold=True);image_axes(x,y,width,height,img,cmap,clim)
    f.text(5,148.5,f'{extent[0]:.3f}–{extent[1]:.3f}°E, {extent[2]:.3f}–{extent[3]:.3f}°N · uint8')
    f.title(98,67,'d','Area-weighted composition')
    for yi,col,label in [(79,'valid_area_fraction','Valid area / tile footprint'),(105,'flood_fraction','Flood area / common-valid area')]:
        aa=f.ax(112,yi,64,16)
        for event in GROUPS:
            q=d[keep&d.event_group.eq(event)].sort_values(col);xval=q[col].to_numpy();w=q.valid_area_km2.to_numpy();cdf=np.cumsum(w)/w.sum();f.gid(aa.step(np.r_[0,xval],np.r_[0,cdf],where='post',color=EC[event],lw=1.1)[0],f'ecdf_{col}_{event}')
        aa.set_xlim(0,1);aa.set_ylim(0,1.02);aa.set_xticks([0,.5,1]);aa.set_yticks([0,1],['','1']);aa.set_ylabel('Area CDF',labelpad=1);aa.set_xlabel(label,labelpad=1)
    for i,(event,r) in enumerate(groups.iterrows()):
        yy=133+i*3;f.text(101,yy,'E'+str(i+1) if i<4 else 'C',color=GC[i],bold=True);f.text(122,yy,f'{int(r.tiles):,}',ha='right');f.text(151,yy,f'{r.valid_area_km2:,.0f}',ha='right');f.text(176,yy,f'{100*r.zero_flood_tile_fraction:.1f}%',ha='right')
    for x,s in [(122,'Tiles'),(151,'Valid km²'),(176,'Zero-flood')]:f.text(x,129,s,bold=True,ha='right')
    f.title(2,156,'e','Geographic non-overlap selection');q=f.read('displayed_overlap_footprints.csv');a=f.ax(11,166,39,27);bounds=[q.left.min(),q.right.max(),q.bottom.min(),q.top.max()]
    for r in q.itertuples():
        f.gid(a.add_patch(Rectangle((r.left,r.bottom),r.right-r.left,r.top-r.bottom,facecolor=to_rgba(C['value'],.42) if r.retained else 'none',edgecolor=C['value'] if r.retained else C['error'],lw=.35)),f'footprint_{r.sample_id}')
    a.set_xlim(bounds[:2]);a.set_ylim(bounds[2:]);a.set_aspect(1/np.cos(np.deg2rad(np.mean(bounds[2:]))));a.tick_params(labelsize=8);a.set_xticks(np.linspace(bounds[0],bounds[1],4)[1:-1]);a.set_yticks(np.linspace(bounds[2],bounds[3],4)[1:-1]);a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}'));a.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}'));a.set_xlabel('Longitude (°E)',labelpad=0);a.set_ylabel('Latitude (°N)',labelpad=0)
    xx0=bounds[0]+.08*(bounds[1]-bounds[0]);yy0=bounds[2]+.08*(bounds[3]-bounds[2]);xx1,_,_=GEOD.fwd(xx0,yy0,90,5000);a.plot([xx0,xx1],[yy0,yy0],c='black',lw=1);a.text(xx0,yy0,'5 km',fontsize=8,va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=.1))
    degree=f.read('overlap_degree.csv');aa=f.ax(63,169,29,20);v=degree.degree.value_counts().sort_index();f.gid(aa.bar(v.index,v.values,color=C['neutral'],edgecolor='black',lw=.3),'degree_distribution');aa.set_yscale('log');aa.set_xticks([0,2,4,6]);aa.set_xlabel('Overlap degree',labelpad=1);aa.set_ylabel('Tiles (log)',labelpad=1);f.text(55,163,'3,386 original pairs');f.text(53,197,'0 retained–retained conflicts')
    f.title(98,156,'f','Released-reference audit');f.text(100,163,'Threshold / change → manual + model refinement');f.text(100,168,'Label values (million pixels)',bold=True);lv=f.read('label_values.csv');pops=lv.population.unique();a=f.ax(117,177,30,16);a.set_xlim(0,3);a.set_ylim(3,0)
    for i,pop in enumerate(pops):
        for j,v in enumerate([0,1,255]):
            value=lv[lv.population.eq(pop)&lv.label_value.eq(v)].pixels.iloc[0];a.add_patch(Rectangle((j,i),1,1,facecolor=to_rgba(C['sar'],.06 if v==255 else .16),edgecolor='black',lw=.35));a.text(j+.5,i+.5,f'{value/1e6:.2f}' if value else '0',ha='center',va='center')
    a.set_xticks([.5,1.5,2.5],['0','1','255']);a.set_yticks([.5,1.5,2.5],['Raw','Retained','Common']);f.text(164,174,'Reference edge',ha='center');image_axes(153,177,24,18,xx[2:5].transpose(1,2,0)/255,boundary=True,scale=True);f.text(100,198,'Reference labels unchanged; no new adjudication')
    f.finish()
if __name__=='__main__':main()
