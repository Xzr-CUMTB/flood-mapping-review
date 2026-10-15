"""Final source, preservation, typography, vector-object and lossless-raster audit."""
import os,sys,json,csv,re,hashlib,argparse,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configs/core_release_s17.json').exists());BASE=ROOT/'figures/supplementary';NS='{http://www.w3.org/2000/svg}';Image.MAX_IMAGE_PIXELS=190_000_000;CACHE={}
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 p=Path(p).resolve();key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
 if key in CACHE:return CACHE[key]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 CACHE[key]=h.hexdigest();return CACHE[key]
def clashes(bounds):
 return [[a['text'],b['text']] for i,a in enumerate(bounds) for b in bounds[i+1:] if min(a['x1_mm'],b['x1_mm'])-max(a['x0_mm'],b['x0_mm'])>.2 and min(a['y1_mm'],b['y1_mm'])-max(a['y0_mm'],b['y0_mm'])>.2]
def validate(n):
 out=BASE/f'S{n:02d}';sd=out/'source_data';assets=out/'assets';audit=js(sd/'scientific_audit.json');release=audit['release_sha256'];check={};details={};check['scientific_derivation_passed']=audit['passed'];numeric=js(sd/'numeric_recheck.json');check.update(numeric['checks']);check['release_unchanged']=sha(ROOT/'configs/core_release_s17.json')==release;check['all_compact_sources_hash_match']=all(sha(sd/k)==v for k,v in audit['source_files'].items());check['all_bound_original_inputs_hash_match']=all(sha(ROOT/x['path'])==x['sha256'] for x in js(sd/'input_bindings.json'))
 for j in range(1,7):
  prior=ROOT/f'figures/main/Fig{j:02d}';v=js(prior/f'Fig{j:02d}_validation.json');check[f'main_Fig{j:02d}_preserved']=v['passed'] and all(sha(prior/x['path'])==x['sha256'] for x in v['sha256_manifest'])
 for j in range(1,6):
  prior=BASE/f'S{j:02d}';v=js(prior/f'S{j:02d}_validation.json');check[f'previous_S{j:02d}_preserved']=v['passed'] and all(sha(prior/x['path'])==x['sha256'] for x in v['sha256_manifest'])
 for suffix in ['','_Arial']:
  stem=f'S{n:02d}_main'+suffix;meta=js(assets/f'render_metadata{suffix}.json');h=meta['height_mm'];pdf=PdfReader(out/(stem+'.pdf'));page=pdf.pages[0];objs=[x.get_object() for x in page['/Resources']['/Font'].values()];fonts=[str(x.get('/BaseFont')) for x in objs];text=page.extract_text();check[stem+'_physical_page']=len(pdf.pages)==1 and abs(float(page.mediabox.width)*25.4/72-180)<.01 and abs(float(page.mediabox.height)*25.4/72-h)<.01;check[stem+'_font_family']=all(('Arial' if suffix else 'TimesNewRoman') in x for x in fonts);check[stem+'_embedded_fonts']=all(bool((x['/DescendantFonts'][0].get_object() if x.get('/DescendantFonts') else x)['/FontDescriptor'].get_object().get('/FontFile2')) for x in objs);check[stem+'_English_selectable_text']=len(text)>500 and not re.search('[\u4e00-\u9fff]',text)
  tree=ET.parse(out/(stem+'.svg'));texts=tree.findall('.//'+NS+'text');groups={g.attrib.get('id',''):g for g in tree.findall('.//'+NS+'g')};imgs=tree.findall('.//'+NS+'image');image_ids={im.attrib.get('id') for im in imgs};sizes=[float(x) for t in texts for x in re.findall(r'([\d.]+)px',t.attrib.get('style',''))];check[stem+'_editable_text']=len(texts)>60 and min(sizes)>=7;check[stem+'_raster_layers_only_native_geography']=len(imgs)==(3 if n==7 else 0);check[stem+'_all_semantic_objects']=all((x in image_ids if n==7 and x.startswith('a_native_') else x in groups) for x in meta['roles']);check[stem+'_panel_count']=len(meta['panels'])==(6 if n in [8,9] else 4)
  if n==6:check[stem+'_18_actual_operating_points']=sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('d_'))==18
  if n==7:check[stem+'_8_state_cells_12_component_curves']=sum(x.startswith('b_state_') for x in groups)==8 and sum(x.startswith('c_') for x in meta['roles'])==12
  if n==8:check[stem+'_12_primary_risk_effects']=sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('e_'))==12
  if n==9:check[stem+'_complete_candidate_scatter']=sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('c_candidates_'))==424
  if n==10:check[stem+'_24_primary_and24_sensitivity_points']=sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('b_'))==24 and sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('c_'))==24
  bounds=js(assets/f'text_bounds{suffix}.json');outside=[x for x in bounds if x['x0_mm']<-.01 or x['x1_mm']>180.01 or x['y0_mm']<-.01 or x['y1_mm']>h+.01];collide=clashes(bounds);check[stem+'_all_text_inside_page']=not outside;check[stem+'_no_text_collisions']=not collide;check[stem+'_all_scatter_points_visible']=not any(c['outside'] for a in meta['axes'] for c in a['collections'])
  with Image.open(out/(stem+'.png')) as png,Image.open(out/(stem+'.tiff')) as tif:
   check[stem+'_1200dpi']=all(abs(float(v)-1200)<.02 for v in (*png.info['dpi'],*tif.info['dpi']));check[stem+'_raster_dimensions']=png.size==tif.size and all(abs(x-y)<1.1 for x,y in zip(png.size,[180/25.4*1200,h/25.4*1200]));equal=True
   for y in range(0,png.height,512):
    box=(0,y,png.width,min(y+512,png.height))
    if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
   check[stem+'_lossless_png_tiff']=equal and tif.info.get('compression')=='tiff_lzw'
  doc=pdfium.PdfDocument(str(out/(stem+'.pdf')));pg=doc[0];pg.render(scale=120/72).to_pil().save(assets/(stem+'_print_size_check.png'));pg.close();doc.close();details[stem]=dict(fonts=fonts,svg_text_objects=len(texts),raster_layers=len(imgs),minimum_font_pt=min(sizes),outside_text=outside,text_collisions=collide)
 # Bind each displayed file without changing the scientific records.
 manifest_path=out/f'S{n:02d}_panel_data_manifest.csv'
 with manifest_path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 for row in rows:row['source_sha256']=sha(sd/row['source_file'])
 with manifest_path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=['panel','source_file','release_sha256','source_sha256']);w.writeheader();w.writerows(rows)
 check['caption_English']=not re.search('[\u4e00-\u9fff]',(out/f'S{n:02d}_caption_en.md').read_text(encoding='utf-8'));check['three_review_rounds_recorded']=(assets/'visual_qa.md').is_file() and all((assets/f'S{n:02d}_round{r}.pdf').is_file() for r in [1,2,3]);manifest=[]
 for p in sorted(out.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.zip','.log'] or p.name in [f'S{n:02d}_validation.json',f'S{n:02d}_delivery.sha256']:continue
  manifest.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
 check={k:bool(v) for k,v in check.items()};result=dict(passed=all(check.values()),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=release,checks=check,details=details,sha256_manifest=manifest);(out/f'S{n:02d}_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(f'S{n:02d}',result['passed'],[k for k,v in check.items() if not v],flush=True);return result['passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--figure',type=int);a=p.parse_args();res=[validate(n) for n in ([a.figure] if a.figure else range(6,11))];sys.exit(0 if all(res) else 1)
