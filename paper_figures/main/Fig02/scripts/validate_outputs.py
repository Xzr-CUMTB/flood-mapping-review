"""Verify final Fig. 2 exports and bind every delivered source; no experiment writes."""
from pathlib import Path
import csv,json,hashlib,os,re,shutil,sys,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
OUT=Path(os.path.abspath(__file__)).parents[1];ROOT=OUT.parents[2];SD=OUT/'source_data'
Image.MAX_IMAGE_PIXELS=150_000_000
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    checks={};detail={};audit=json.loads((SD/'scientific_audit.json').read_text());pro=json.loads((SD/'derivation_provenance.json').read_text());checks['scientific_audit_passed']=audit['passed'];checks['release_matches']=audit['release_sha256']==pro['release_sha256'];checks['frozen_source_and_fig1_integrity']=audit['checks']['all_142_frozen_files_preserved'] and audit['checks']['figure_1_unchanged']
    specs=[('a','panel_a_geographic_window.npz;panel_a_marginals.csv;panel_a_selection.json','SAR;fusion','BB','georeferenced common-valid pixel; fixed 32-pixel coordinate bins','m2 / km2','same geographic window; exact row-area weights'),('b','panel_b_components.csv.gz;panel_b_grid_summary.csv;component_cross_tile_audit.csv','fusion','BB;HB','observed 8-neighbour component; 4-neighbour sensitivity','m2 / ha','entire observed confident-omission area; unknown-edge pieces flagged'),('c','panel_c_transitions.csv;panel_c_patch_transitions.csv.gz','SAR;fusion','BB;HB','paired reference-flood pixel','m2; fraction','regional reference-flood area; frozen p>=0.5'),('d','panel_d_transect.csv','SAR;fusion','BB','native-grid sample along one true horizontal path','km; DN; probability','same actual locations; explicit missing-data gaps'),('e','panel_e_boundary_distance.csv;boundary_segments_Beijing_Baoding.npz;boundary_segments_Harbin.npz','SAR;fusion','BB;HB','reference-flood pixel; fixed projected-distance bin','m; m2; fraction','within-bin flood area; unresolved support-edge cases separately retained'),('f','panel_f_conditions.csv;panel_f_patch_conditions.csv.gz','SAR;fusion','BB;HB','reference-flood pixel in fixed quantized HH bin','DN; m2; percentage points','within-bin reference-flood area')]
    with (OUT/'Fig02_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','source_files','models','regions','statistical_unit','units','denominator_and_rule','script','release_sha256','file_sha256']);writer.writeheader()
        for panel,files,models,regions,unit,units,rule in specs:
            hashes={name:sha(SD/name) for name in files.split(';')};writer.writerow(dict(panel=panel,source_files=files,models=models,regions=regions,statistical_unit=unit,units=units,denominator_and_rule=rule,script='scripts/derive_spatial.py; scripts/render_figure.py',release_sha256=pro['release_sha256'],file_sha256=json.dumps(hashes)))
    for suffix in ['', '_Arial']:
        stem='Fig02_main'+suffix;reader=PdfReader(OUT/(stem+'.pdf'));p=reader.pages[0];fonts=[str(v.get_object().get('/BaseFont')) for v in p['/Resources']['/Font'].values()]
        checks[stem+'_vector_pdf']=len(reader.pages)==1 and abs(float(p.mediabox.width)*25.4/72-180)<.01 and abs(float(p.mediabox.height)*25.4/72-195)<.01 and len(p.extract_text())>900
        checks[stem+'_font_identity']=all(('Arial' if suffix else 'TimesNewRoman') in f for f in fonts)
        tree=ET.parse(OUT/(stem+'.svg'));texts=tree.findall('.//{http://www.w3.org/2000/svg}text');labels=[''.join(t.itertext()) for t in texts];sizes=[float(m.group(1)) for t in texts for m in [re.search(r'([\d.]+)px',t.attrib.get('style',''))] if m]
        checks[stem+'_editable_text']=len(texts)>100 and all(k in labels for k in 'abcdef');checks[stem+'_body_font_minimum']=bool(sizes) and min(sizes)>=7
        checks[stem+'_no_jpeg_image_layers']=all('image/jpeg' not in str(t.attrib) for t in tree.iter())
        with Image.open(OUT/(stem+'.png')) as png,Image.open(OUT/(stem+'.tiff')) as tif:
            dpi=tuple(map(float,png.info['dpi']));tdpi=tuple(map(float,tif.info['dpi']));dims=png.size
            checks[stem+'_raster_size']=dims==tif.size and all(abs(a-b)<1.1 for a,b in zip(dims,[180/25.4*1200,195/25.4*1200]));checks[stem+'_1200dpi']=all(abs(v-1200)<.02 for v in (*dpi,*tdpi));equal=True
            for y in range(0,png.height,512):
                box=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_pixel_equality']=equal
        doc=pdfium.PdfDocument(OUT/(stem+'.pdf'));doc[0].render(scale=3).to_pil().save(OUT/'assets'/(stem+'_pdf_preview.png'));doc.close()
        detail[stem]=dict(fonts=fonts,pixels=dims,png_dpi=dpi,tiff_dpi=tdpi,svg_text_objects=len(texts),body_sizes_pt=sorted(set(sizes)))
    caption=(OUT/'Fig02_caption_en.md').read_text(encoding='utf-8');detail['caption_words']=len(caption.split());checks['caption_length']=detail['caption_words']<=350
    detail['project_bytes_at_validation']=sum((Path(d)/n).stat().st_size for d,_,files in os.walk(ROOT) for n in files);detail['free_disk_bytes']=shutil.disk_usage(ROOT).free;detail['space_budget']='No user-imposed cap as of 2026-10-14'
    files=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or 'native_grids' in p.parts or '__pycache__' in p.parts or p.name in ['Fig02_validation.json','Fig02_delivery.zip','Fig02_delivery.sha256'] or p.suffix=='.log':continue
        if p.name.startswith('Fig02_round') and 'roundfinal' not in p.name:continue
        files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=all(checks.values()),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=pro['release_sha256'],checks=checks,details=detail,scientific_audit_sha256=sha(SD/'scientific_audit.json'),visual_qa='Three actual rendering rounds plus final PDF renders; details in assets/visual_qa.md',sha256_manifest=files)
    (OUT/'Fig02_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v],details=detail),indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
