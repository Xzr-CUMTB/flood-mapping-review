"""Validate delivered figure files and bind the reproducible source package."""
from pathlib import Path
import os,json,hashlib,csv,re,sys,zipfile,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
OUT=Path(os.path.abspath(__file__)).parents[1];ROOT=OUT.parents[2];SD=OUT/'source_data';AS=OUT/'assets'
Image.MAX_IMAGE_PIXELS=150_000_000
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    audit=json.loads((SD/'scientific_audit.json').read_text());verify=json.loads((ROOT/'reports/completed_batch_verification.json').read_text());checks=dict(scientific_audit=audit['passed']);details={}
    checks['core_csv_matches_completion_sha256']=sha(ROOT/'runs/core_s17_evaluation/map_metrics.csv')==verify['csv_sha256']['map_metrics.csv'];checks['identical_release']=audit['release_sha256']==verify['release_sha256']==sha(ROOT/'configs/core_release_s17.json')
    specs=[('a','panel_a_crossfit_provenance.json;panel_a_model_training.csv','actual nested training records','four development groups','20 expert epochs;10 head epochs;seed17'),('b','panel_b_joint_bins.csv.gz;panel_b_calibration.csv;panel_bc_event_summary.csv;oof_prediction_index.csv.gz','common-valid pixels, area-weighted; event-wise correlations','four outer held-out events','all valid pixels; no sampling; unequal value bin widths'),('c','panel_c_joint_bins.csv.gz;panel_bc_event_summary.csv','common-valid pixels, area-weighted','four outer held-out events','Brier(Fusion) minus Brier(Gate); mean weighted by area in bin'),('d','panel_de_event_metrics.csv;panel_de_paired_effects.csv','held-out disaster group','four outer held-out events','paired same-domain models; raw probability'),('e','panel_de_paired_effects.csv','held-out disaster group','four outer held-out events','absolute FN and FP area deltas in km2'),('f','panel_f_frozen_metrics.csv','frozen test region','Beijing_Baoding;Harbin','same release; raw p>=0.5; absolute metrics')]
    with (OUT/'Fig03_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['panel','source_files','statistical_unit','events','rule','status','release_sha256','sha256']);w.writeheader()
        for panel,files,unit,ev,rule in specs:w.writerow(dict(panel=panel,source_files=files,statistical_unit=unit,events=ev,rule=rule,status='complete real sources; read-only derivation',release_sha256=audit['release_sha256'],sha256=json.dumps({n:sha(SD/n) for n in files.split(';')})))
    for suffix in ['','_Arial']:
        stem='Fig03_main'+suffix;r=PdfReader(OUT/(stem+'.pdf'));p=r.pages[0];fonts=[str(v.get_object().get('/BaseFont')) for v in p['/Resources']['/Font'].values()]
        checks[stem+'_page']=len(r.pages)==1 and abs(float(p.mediabox.width)*25.4/72-180)<.01 and abs(float(p.mediabox.height)*25.4/72-200)<.01
        checks[stem+'_vector_text']=len(p.extract_text())>1200;checks[stem+'_font']=all(('Arial' if suffix else 'TimesNewRoman') in f for f in fonts)
        tree=ET.parse(OUT/(stem+'.svg'));nodes=tree.findall('.//{http://www.w3.org/2000/svg}text');labels=[''.join(t.itertext()) for t in nodes];sizes=[float(m.group(1)) for t in nodes for m in [re.search(r'([\d.]+)px',t.attrib.get('style',''))] if m]
        checks[stem+'_editable_svg']=len(nodes)>150 and all(k in labels for k in 'abcdef');checks[stem+'_body_min_7pt']=bool(sizes) and min(sizes)>=7
        with Image.open(OUT/(stem+'.png')) as png,Image.open(OUT/(stem+'.tiff')) as tif:
            dims=png.size;dpi=(*png.info['dpi'],*tif.info['dpi']);checks[stem+'_1200dpi']=all(abs(float(d)-1200)<.02 for d in dpi);checks[stem+'_size']=dims==tif.size and all(abs(x-y)<1.1 for x,y in zip(dims,[180/25.4*1200,200/25.4*1200]));equal=True
            for y in range(0,png.height,512):
                box=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_equality']=equal;checks[stem+'_lossless_tiff']=tif.info.get('compression')=='tiff_lzw'
        d=pdfium.PdfDocument(OUT/(stem+'.pdf'));d[0].render(scale=3).to_pil().save(AS/(stem+'_pdf_preview.png'));d.close();details[stem]=dict(fonts=fonts,svg_texts=len(nodes),body_sizes=sizes and sorted(set(sizes)),pixels=dims)
    with zipfile.ZipFile(OUT/'Fig03_panel_a_editable.pptx') as z:
        checks['pptx_entirely_native']=not any(n.startswith('ppt/media/') for n in z.namelist());xml=ET.fromstring(z.read('ppt/slides/slide1.xml'));texts=xml.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t');checks['pptx_native_text']=len(texts)>35;checks['pptx_no_placeholder']=not any('lorem' in (t.text or '').lower() for t in texts)
    r=PdfReader(AS/'pptx_render/Fig03_panel_a_editable.pdf');checks['pptx_actual_render']=len(r.pages)==1 and 'Nested cross-event' in r.pages[0].extract_text()
    details['caption_words']=len((OUT/'Fig03_caption_en.md').read_text(encoding='utf-8').split());checks['caption_350_words']=details['caption_words']<=350
    files=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or any(k in p.parts for k in ['lo_profile','__pycache__']) or p.name in ['Fig03_validation.json','Fig03_delivery.zip','Fig03_delivery.sha256'] or p.suffix=='.log':continue
        files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=all(checks.values()),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=audit['release_sha256'],checks=checks,details=details,sha256_manifest=files,visual_qa='Four development renders, independent subagent inspection, final PDFs and native PPT rendering; see assets/visual_qa.md')
    (OUT/'Fig03_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v],details=details),indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
