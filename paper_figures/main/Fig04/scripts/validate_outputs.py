"""Validate numerical handoff, proportionate vector flow geometry and actual export files."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,csv,json,re,hashlib,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
OUT=Path(os.path.abspath(__file__)).parents[1];ROOT=OUT.parents[2];SD=OUT/'source_data';AS=OUT/'assets'
Image.MAX_IMAGE_PIXELS=160_000_000
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(n):
    with (SD/n).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    audit=json.loads((SD/'scientific_audit.json').read_text());checks=dict(scientific_audit=audit['passed']);details={};release=audit['release_sha256']
    checks['release_unchanged']=sha(ROOT/'configs/core_release_s17.json')==release
    binding=json.loads((SD/'input_bindings.json').read_text());small=[r for r in binding if r['bytes']<2_000_000]
    checks['bound_small_inputs_unchanged']=all(sha(ROOT/r['path'])==r['sha256'] for r in small)
    b=rows('verified_review_budget_results.csv');checks['32_model_region_budget_rows']=len(b)==32
    checks['fn_fp_area_conservation']=all(abs(float(r[f'initial_{e}_km2'])-float(r[f'corrected_{e}_km2'])-float(r[f'residual_{e}_km2']))<1e-9 for r in b for e in ['fn','fp'])
    c=rows('panel_c_decomposition.csv');checks['signed_waterfall_conservation']=all(abs(float(r['delta_initial_fn_km2'])+float(r['negative_delta_corrected_fn_km2'])-float(r['delta_residual_fn_km2']))<1e-9 for r in c)
    geometry=rows('panel_d_rendered_flow_geometry.csv');checks['shared_exact_ribbon_scale']=len(geometry)==8 and all(abs(float(r['width_mm'])/float(r['area_km2'])-.072)<1e-12 for r in geometry)
    groups={}
    for r in geometry:groups.setdefault((r['region'],r['model']),[]).append(r)
    checks['rendered_flow_width_conservation']=all(len(v)==2 and abs(sum(float(r['width_mm']) for r in v)-float(v[0]['initial_width_mm']))<1e-10 for v in groups.values())
    inv=rows('panel_b_inversion_comparisons.csv');checks['both_focal_inversions_verified']=len(inv)==2 and all(r['inversion']=='True' and float(r['delta_iou'])>0 and float(r['delta_residual_fn_km2'])>0 for r in inv)
    a=rows('panel_a_budget_response.csv');checks['observed_budget_points']=len(a)==40 and set(float(r['budget_fraction']) for r in a)=={0,.01,.05,.1,.2}
    inter=rows('panel_f_paired_intervals.csv');checks['correct_simultaneous_family']=len(inter)==4 and all(float(r['budget_fraction'])==.1 and int(r['family_estimates'])==12 and int(r['block_size_m'])==10000 for r in inter)
    saved=json.loads((SD/'interval_reproduction.json').read_text());checks['full_family_recomputed']=len(saved['checks'])==24 and max(x['max_abs_error'] for x in saved['checks'])<1e-9
    cases=json.loads((SD/'panel_e_case_selection.json').read_text())['cases'];displays=json.loads((AS/'geographic_display_parameters.json').read_text());checks['geographic_aspect_corrected']=all(abs(d['longitude_latitude_aspect']-1/np.cos(np.deg2rad((r['extent_lonlat'][2]+r['extent_lonlat'][3])/2)))<1e-12 for r,d in zip(cases,displays))
    for r in cases:
        z=np.load(SD/f"panel_e_{r['region']}_window.npz");valid=z['valid'];checks[r['region']+'_native_case_shape']=valid.shape==(512,1280) and len(r['prediction_indices'])==10
        for m in ['fusion','value_gate']:
            fn=valid&(z['label']==1)&(z[m]<.5);sel=z['selected_'+m].repeat(64,0).repeat(64,1);w=z['area_m2'];st=r['statistics'][m]
            checks[r['region']+'_'+m+'_native_masks']=all(abs(float(v)-st[k])<1e-10 for k,v in [('initial_fn_km2',w[fn].sum()/1e6),('corrected_fn_km2',w[fn&sel].sum()/1e6),('residual_fn_km2',w[fn&~sel].sum()/1e6)])
    specs=[('a','panel_a_budget_response.csv;panel_a_paired_differences.csv','common-valid area; regional aggregates','0% initial map and observed 1/5/10/20% budgets'),('b','panel_b_rankings.csv;panel_b_inversion_comparisons.csv','four fixed models per region','higher IoU and lower residual FN define rank 1; same mean-entropy policy'),('c','panel_c_decomposition.csv','paired regional FN area differences','Value minus Fusion; delta R = delta F minus delta Q'),('d','panel_d_conservation_flows.csv;panel_d_rendered_flow_geometry.csv','within-model conserved FN areas','uniform 0.072 mm per km2; no cross-model transition'),('e','panel_e_case_selection.json;panel_e_all_candidate_windows.csv;panel_e_shared_zoom_selection.csv;panel_e_geographic_review_units.csv.gz;panel_e_Beijing_Baoding_window.npz;panel_e_Harbin_window.npz;panel_e_Beijing_Baoding_units.csv;panel_e_Harbin_units.csv','native georeferenced pixels; whole 64x64 review units','deterministic illustrative windows; fixed regional selections; identical paired image transformations'),('f','panel_f_endpoints_and_costs.csv;panel_f_paired_intervals.csv;panel_f_all_simultaneous_intervals.csv;interval_reproduction.json','paired 10km within-region spatial blocks','2000 bootstrap draws; 12-estimate simultaneous family per region at 10%')]
    with (OUT/'Fig04_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','source_files','statistical_unit','rule','release_sha256','sha256']);writer.writeheader()
        for panel,names,unit,rule in specs:writer.writerow(dict(panel=panel,source_files=names,statistical_unit=unit,rule=rule,release_sha256=release,sha256=json.dumps({n:sha(SD/n) for n in names.split(';')})))
    for suffix in ['','_Arial']:
        stem='Fig04_main'+suffix;reader=PdfReader(OUT/(stem+'.pdf'));p=reader.pages[0];fontnames=[str(v.get_object().get('/BaseFont')) for v in p['/Resources']['/Font'].values()]
        checks[stem+'_page_size']=len(reader.pages)==1 and abs(float(p.mediabox.width)*25.4/72-180)<.01 and abs(float(p.mediabox.height)*25.4/72-225)<.01
        checks[stem+'_embedded_font']=all(('Arial' if suffix else 'TimesNewRoman') in v for v in fontnames)
        checks[stem+'_selectable_pdf_text']=len(p.extract_text())>1400
        tree=ET.parse(OUT/(stem+'.svg'));texts=tree.findall('.//{http://www.w3.org/2000/svg}text');labels=[''.join(t.itertext()) for t in texts];sizes=[float(m.group(1)) for t in texts for m in [re.search(r'([\d.]+)px',t.attrib.get('style',''))] if m]
        paths=tree.findall('.//{http://www.w3.org/2000/svg}path');images=tree.findall('.//{http://www.w3.org/2000/svg}image')
        vector_roles={prefix:[g for g in tree.findall('.//{http://www.w3.org/2000/svg}g') if g.attrib.get('id','').startswith(prefix)] for prefix in ['fn_flow_','waterfall_','rank_link_','reference_context_','review_unit_']}
        checks[stem+'_editable_text']=len(texts)>150 and all(x in labels for x in 'abcdef');checks[stem+'_native_vectors']=all(len(vector_roles[k])==n for k,n in [('fn_flow_',8),('waterfall_',6),('rank_link_',8),('reference_context_',4),('review_unit_',45)]) and all(g.findall('.//{http://www.w3.org/2000/svg}path') for groups in vector_roles.values() for g in groups) and len(images)==8;checks[stem+'_minimum_body_size']=bool(sizes) and min(sizes)>=7
        bounds=json.loads((AS/('text_bounds'+suffix+'.json')).read_text());checks[stem+'_figure_labels_inside_page']=all(r['x0_mm']>=0 and r['x1_mm']<=180 and r['y0_mm']>=0 and r['y1_mm']<=225 for r in bounds)
        collisions=[]
        for i,u in enumerate(bounds):
            for v in bounds[i+1:]:
                if min(u['x1_mm'],v['x1_mm'])-max(u['x0_mm'],v['x0_mm'])>.2 and min(u['y1_mm'],v['y1_mm'])-max(u['y0_mm'],v['y0_mm'])>.2:collisions.append([u['text'],v['text']])
        checks[stem+'_figure_labels_no_collision']=not collisions
        with Image.open(OUT/(stem+'.png')) as png,Image.open(OUT/(stem+'.tiff')) as tif:
            dims=png.size;dpi=(*png.info['dpi'],*tif.info['dpi']);checks[stem+'_1200dpi']=all(abs(float(d)-1200)<.02 for d in dpi);checks[stem+'_raster_dimensions']=dims==tif.size and all(abs(x-y)<1.1 for x,y in zip(dims,[180/25.4*1200,225/25.4*1200]));equal=True
            for y in range(0,png.height,512):
                box=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_png_tiff_match']=equal;checks[stem+'_tiff_lzw']=tif.info.get('compression')=='tiff_lzw'
        doc=pdfium.PdfDocument(OUT/(stem+'.pdf'));preview=doc[0].render(scale=3).to_pil();preview.save(AS/(stem+'_pdf_preview.png'))
        for name,(x0,y0,x1,y1) in {'top':(0,0,180,80),'middle':(0,80,180,151),'geography':(0,151,106,225),'uncertainty':(107,151,180,225)}.items():
            big=doc[0].render(scale=5).to_pil();bb=(int(x0/180*big.width),int(y0/225*big.height),int(x1/180*big.width),int(y1/225*big.height));big.crop(bb).save(AS/(stem+'_'+name+'_pdf.png'))
        doc.close();details[stem]=dict(fonts=fontnames,svg_texts=len(texts),svg_paths=len(paths),vector_role_counts={k:len(v) for k,v in vector_roles.items()},raster_layers=len(images),body_sizes=sorted(set(sizes)),pixels=dims,label_collisions=collisions)
    caption=(OUT/'Fig04_caption_en.md').read_text(encoding='utf-8');details['caption_words']=len(caption.split());checks['caption_word_limit']=details['caption_words']<=350;checks['caption_english']=not re.search('[\u4e00-\u9fff]',caption)
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['Fig04_validation.json','Fig04_delivery.zip','Fig04_delivery.sha256'] or p.suffix=='.log':continue
        files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=all(checks.values()),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=release,checks=checks,details=details,sha256_manifest=files,visual_qa='Three design rounds plus actual PDF inspection in both fonts; see assets/visual_qa.md')
    (OUT/'Fig04_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v],details=details),indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
