"""Redraw original manuscript figures from deposited numeric sources, without training."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def compare_tiff(reference,actual):
    from PIL import Image
    import numpy as np
    Image.MAX_IMAGE_PIXELS=200_000_000
    a=Image.open(reference);b=Image.open(actual)
    result=dict(reference_sha256=sha(reference),actual_sha256=sha(actual),dimensions=list(b.size),
                mode=b.mode,dpi=[float(v) for v in b.info.get('dpi',[])],same_dimensions=a.size==b.size,
                identical_pixels=True,different_channels=0,maximum_channel_difference=0)
    if a.size!=b.size:
        result['identical_pixels']=False
        return result
    for y in range(0,a.height,128):
        box=(0,y,a.width,min(y+128,a.height))
        x=np.asarray(a.crop(box).convert('RGB'),dtype=np.int16);z=np.asarray(b.crop(box).convert('RGB'),dtype=np.int16)
        delta=np.abs(x-z);result['different_channels']+=int(np.count_nonzero(delta))
        result['maximum_channel_difference']=max(result['maximum_channel_difference'],int(delta.max()))
    a.close();b.close();result['identical_pixels']=result['different_channels']==0
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',default='all');p.add_argument('--font',default='Times New Roman',choices=['Times New Roman','Arial'])
    p.add_argument('--final',action='store_true',help='Produce original 1200-dpi TIFF and vector masters')
    p.add_argument('--verify-reference',action='store_true',help='Compare regenerated TIFF RGB pixels with deposited manuscript masters')
    args=p.parse_args();index=json.loads((args.data/'FIGURE_INDEX.json').read_text())
    names={f['figure'] for f in index} if args.figures=='all' else set(args.figures.split(','))
    if not names.issubset({f['figure'] for f in index}):raise ValueError('Unknown figure ID')
    args.output.mkdir(parents=True,exist_ok=True);results=[]
    for item in index:
        name=item['figure']
        if name not in names:continue
        source=ROOT/'paper_figures'/item['group']/name/'scripts';out=args.output/'figures'/item['group']/name
        out.mkdir(parents=True,exist_ok=True);(out/'assets').mkdir(exist_ok=True)
        shutil.copytree(source,out/'scripts',dirs_exist_ok=True)
        shutil.copytree(args.data/'figure_data'/name/'source_data',out/'source_data',dirs_exist_ok=True)
        script=out/'scripts'/('render_figure.py' if item['group']=='main' else 'render_'+name.lower()+'.py')
        cmd=[sys.executable,'-X','utf8','-s',str(script),'--font',args.font,'--round','999']
        if args.final:cmd.append('--final')
        start=time.monotonic();print('Rendering '+name,flush=True)
        with (out/'render.log').open('w',encoding='utf-8') as stream:
            subprocess.run(cmd,cwd=args.output,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=dict(figure=name,rendered=True,final=args.final,font=args.font,elapsed_seconds=time.monotonic()-start,
                    renderer_sha256=sha(script),source_release_sha256=item.get('source_release_sha256','093a72b5ad047aa885d05b382e697192849305165ab63b226f33c1d9160ff242'))
        if args.verify_reference:
            if not args.final:raise ValueError('--verify-reference requires --final')
            suffix='_Arial' if args.font=='Arial' else ''
            reference=args.data/'reference_figures'/(name+'_main'+suffix+'.tiff')
            if not reference.exists():raise FileNotFoundError('Reference figure missing: '+reference.name)
            result['reference_comparison']=compare_tiff(reference,out/(name+'_main'+suffix+'.tiff'))
        results.append(result)
        (args.output/'figure_reproduction.json').write_text(json.dumps(dict(results=results,complete=False),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True)
    passed=all(r.get('reference_comparison',{}).get('identical_pixels',True) for r in results)
    report=dict(passed=passed,complete=True,figures=len(results),results=results,
                scope='Exact original renderers and deposited numerical sources; no fitted model or statistics altered')
    (args.output/'figure_reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not passed:raise ValueError('One or more regenerated figures differ from the manuscript TIFF pixels; see report')

if __name__=='__main__':main()
