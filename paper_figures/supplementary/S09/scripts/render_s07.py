from plot_helpers import *
from matplotlib.colors import to_rgb
ST=['stable_detection','recovered','newly_missed','persistent_omission'];SN=['Stable detection','Recovered','Newly missed','Persistent omission'];SC=['#A6BAC6',C['value'],C['error'],C['fusion']]
def main():
 f=Canvas(7,190,args());atlas=np.load(f.sd/'geographic_atlas.npz');specs=f.js('geographic_atlas_manifest.json');m=f.read('atlas_marginals.csv');t=f.read('transition_areas.csv');components=f.read('components.csv.gz');bd=f.read('boundary_distance.csv')
 f.title(3,4,'a','Full-region flood-state anatomy');legend(f,5,11,[(s,c,'-') for s,c in zip(SN,SC)],43)
 for z,region in enumerate(REG):
  x=17+z*89;a=f.ax(x,25,58,48);top=f.ax(x,20,58,4);side=f.ax(x+61,25,5,48);f.text(x+29,17,RN[z],ha='center');allx=[];ally=[]
  for spec in [q for q in specs if q['region']==region]:
   gid=spec['grid_id'];bins=atlas[gid];area=bins.sum(2);fl=bins[:,:,1:].sum(2);mix=bins[:,:,1:]@np.array([to_rgb(c) for c in SC]);mix=np.divide(mix,fl[:,:,None],out=np.ones_like(mix),where=fl[:,:,None]>0);alpha=np.sqrt(np.divide(fl,area,out=np.zeros_like(fl),where=area>0));rgb=np.ones((*area.shape,3));rgb[area>0]=.94;rgb=rgb*(1-alpha[:,:,None])+mix*alpha[:,:,None];rgba=np.dstack([rgb,(area>0).astype(float)]);ext=atlas[gid+'_extent'];allx.extend(ext[:2]);ally.extend(ext[2:]);f.gid(a.imshow(rgba,extent=ext,interpolation='nearest'),f'a_native_{gid}')
   for axis,ax in [('longitude',top),('latitude',side)]:
    q=m[(m.grid_id==gid)&(m.axis==axis)].sort_values('lower');coord=(q.lower+q.upper)/2;vals=[q[s+'_km2'].to_numpy() for s in ST]
    if axis=='longitude':ax.stackplot(coord,*vals,colors=SC,lw=0)
    else:
     cum=np.zeros(len(q))
     for v,c in zip(vals,SC):ax.fill_betweenx(coord,cum,cum+v,color=c,lw=0);cum+=v
  a.set(xlim=(min(allx),max(allx)),ylim=(min(ally),max(ally)),xlabel='Longitude (°E)',ylabel='Latitude (°N)');a.set_aspect(1/np.cos(np.deg2rad(np.mean(ally))));a.apply_aspect();pos=a.get_position();top.set_position([pos.x0,top.get_position().y0,pos.width,top.get_position().height]);side.set_position([pos.x1+2/180,pos.y0,side.get_position().width,pos.height]);a.locator_params(nbins=3);top.set_xlim(a.get_xlim());side.set_ylim(a.get_ylim());top.set_axis_off();side.set_axis_off()
  # Geographical scale bar at the actual map latitude, width computed from WGS84.
  from pyproj import Geod
  xx=a.get_xlim()[0]+.08*(a.get_xlim()[1]-a.get_xlim()[0]);yy=a.get_ylim()[0]+.08*(a.get_ylim()[1]-a.get_ylim()[0]);xx2,_,_=Geod(ellps='WGS84').fwd(xx,yy,90,20000);a.plot([xx,xx2],[yy,yy],color='black',lw=1.5);a.text((xx+xx2)/2,yy+.05,'20 km',ha='center',va='bottom',fontsize=7)
 f.text(3,85,'Margins: state area by coordinate; white: unobserved; map colours mix states within 32 × 32-pixel cells.')
 f.title(3,92,'b','Conserved transition areas');f.title(96,92,'c','Component-area tails')
 for z,region in enumerate(REG):
  q=t[t.region==region].set_index('state');x=17+z*40;a=f.ax(x,105,30,22);vals=np.array([[q.loc[ST[0],'area_m2'],q.loc[ST[2],'area_m2']],[q.loc[ST[1],'area_m2'],q.loc[ST[3],'area_m2']]])/1e6;colour=np.array([[to_rgb(SC[0]),to_rgb(SC[2])],[to_rgb(SC[1]),to_rgb(SC[3])]]);a.set(xlim=(-.5,1.5),ylim=(1.5,-.5),xticks=[0,1],xticklabels=['Detected','Missed'],yticks=[0,1],yticklabels=['Detected','Missed'] if z==0 else ['', '']);a.tick_params(length=0);f.text(x+15,99,RN[z],ha='center')
  for i in range(2):
   for j in range(2):
    f.gid(a.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=colour[i,j],edgecolor='none')),f'b_state_{z}_{i}_{j}');a.text(j,i-.13,f'{vals[i,j]:.1f}',ha='center',va='center',color='white' if i==1 or j==1 else 'black',fontsize=7);a.text(j,i+.23,f'{100*vals[i,j]/vals.sum():.1f}%',ha='center',va='center',color='white' if i==1 or j==1 else 'black',fontsize=7)
  a.set_xlabel('Fusion');
  if z==0:a.set_ylabel('SAR')
  net=(q.loc['recovered','area_m2']-q.loc['newly_missed','area_m2'])/1e6;f.text(x+15,140,f'Recovered − new: {net:+.2f} km²',ha='center')
  x=107+z*38;a=f.ax(x,105,29,26);f.text(x+14.5,99,RN[z],ha='center')
  for state,col in zip(ST[1:],SC[1:]):
   for conn,ls in [(8,'-'),(4,'--')]:
    b=components[(components.region==region)&(components.state==state)&(components.connectivity==conn)].sort_values('area_m2');areas=b.area_m2.to_numpy()/1e6;ccdf=np.cumsum(areas[::-1])[::-1]/areas.sum();f.gid(a.step(areas,ccdf,color=col,ls=ls,lw=.9,where='post')[0],f'c_{z}_{state}_{conn}')
  a.set(xscale='log',xlim=(.00004,40),ylim=(0,1.02),xticks=[.0001,.01,1],yticks=[0,.5,1],xlabel='Component area (km²)');
  if z==0:a.set_ylabel('Area-weighted CCDF')
  else:a.set_yticklabels([])
 f.text(105,143,'Solid: 8-neighbour; dashed: 4-neighbour')
 f.title(3,147,'d','Distance to observed flood boundaries')
 for z,region in enumerate(REG):
  x=19+z*89;a=f.ax(x,158,55,20);f.text(x+27.5,153,RN[z],ha='center')
  for state,col in zip(ST,SC):
   q=bd[(bd.region==region)&(bd.state==state)&(~bd.coverage_censored)].sort_values('bin');q=q[np.isfinite(q.right_m)];total=bd[(bd.region==region)&(bd.state==state)].area_m2.sum();f.gid(a.plot(q.right_m,np.cumsum(q.area_m2)/total,color=col,lw=1)[0],f'd_{z}_{state}');v=bd[(bd.region==region)&(bd.state==state)&bd.coverage_censored].area_m2.sum()/total;f.text(x+58,158+ST.index(state)*4,f'{v*100:.1f}%',color=col)
  a.set(xscale='log',xlim=(5,5000),ylim=(0,1.02),xticks=[10,100,1000],yticks=[0,.5,1],xlabel='Boundary distance (m)')
  if z==0:a.set_ylabel('Resolved area / state area')
  else:a.set_yticklabels([])
  f.text(x+57,153,'Censored')
 for p,b in zip('abcd',[(0,0,180,88),(0,90,94,53),(94,90,86,53),(0,145,180,45)]):tag(f,p,*b)
 f.finish()
if __name__=='__main__':main()
