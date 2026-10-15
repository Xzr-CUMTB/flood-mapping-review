"""Independent compact-source, actual-vector and lossless-raster handoff validation."""
import os,sys,json,csv,gzip,re,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
OUT=Path(os.path.abspath(__file__)).parents[1];ROOT=OUT.parents[2];SD=OUT/'source_data';AS=OUT/'assets';Image.MAX_IMAGE_PIXELS=160_000_000
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(name):
    opener=gzip.open if name.endswith('.gz') else open
    with opener(SD/name,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    audit=read(SD/'scientific_audit.json');release=audit['release_sha256'];checks=dict(scientific_audit=audit['passed'],all_pixels_reproduced=max(audit['pixel_unit_max_abs_errors'])==0,all_core_review_units=audit['positive_cost_units']==129347);details={}
    checks['release_unchanged']=sha(ROOT/'configs/core_release_s17.json')==release
    binding=read(SD/'input_bindings.json');checks['bound_small_inputs_unchanged']=all(sha(ROOT/r['path'])==r['sha256'] for r in binding if r['bytes']<2_000_000)
    for name in ['Fig01','Fig02','Fig03','Fig04']:
        prior=ROOT/'figures/main'/name;v=read(prior/(name+'_validation.json'));checks[name+'_unchanged']=v['passed'] and all(sha(prior/r['path'])==r['sha256'] for r in v['sha256_manifest'])
    units=rows('review_units.csv.gz');checks['unique_unit_identifiers']=len({r['unit_id'] for r in units})==len(units)==129347
    checks['unit_release_binding']=all(r['release_sha256']==release for r in units)
    costs=np.array([float(r['cost_m2']) for r in units]);fn=np.array([float(r['fn_m2']) for r in units]);density=np.array([float(r['D']) for r in units]);checks['unit_area_and_density']=bool((costs>0).all() and np.allclose(density,fn/costs,rtol=0,atol=1e-15))
    for c in 'HLP':checks['FN_conservation_'+c]=all(abs(float(r['captured_FN_'+c+'_km2'])+float(r['residual_FN_'+c+'_km2'])-float(r['fn_m2'])/1e6)<1e-12 for r in units)
    b=rows('panel_b_budget_capture.csv');checks['observed_budget_records']=len(b)==70 and set(float(r['budget_fraction']) for r in b)=={0,.01,.05,.1,.2}
    checks['capture_conservation']=all(abs(float(r['initial_fn_km2'])-float(r['captured_fn_km2'])-float(r['residual_fn_km2']))<1e-9 and abs(float(r['capture_fraction'])-float(r['captured_fn_km2'])/float(r['initial_fn_km2']))<1e-14 for r in b)
    checks['whole_unit_budgets_respected']=all(float(r['actual_area_fraction'])<=float(r['budget_fraction'])+1e-12 for r in b)
    packed=np.load(SD/'review_selections_packed.npz');checks['all_56_actual_masks']=len(packed.files)==56
    group_stats=rows('panel_a_distribution_summary.csv');cells=rows('panel_d_disjoint_intersections.csv');npy=np.load(SD/'density_statistics.npz')
    for g in group_stats:
        region=g['region'];q=[r for r in units if r['region']==region];d=[r for r in cells if r['region']==region];w=np.array([float(r['cost_m2'])/1e6 for r in q]);f=np.array([float(r['fn_m2'])/1e6 for r in q]);membership=np.array([int(float(r['membership_HLP'])) for r in q]);zero=f==0
        checks[region+'_zero_FN_mass']=abs(w[zero].sum()/w.sum()-float(g['zero_FN_area_fraction']))<1e-12
        checks[region+'_intersections']=all(abs(w[membership==int(r['intersection_id'])].sum()-float(r['valid_area_km2']))<1e-9 and abs(f[membership==int(r['intersection_id'])].sum()-float(r['fn_area_km2']))<1e-9 for r in d)
        checks[region+'_histogram_area_conservation']=all(abs(npy[region+kind].sum()-expected)<1e-8 for kind,expected in [('_joint_H',w[~zero].sum()),('_copula',w.sum()),('_joint_L',w.sum())])
        for c,policy in [('H','mean_entropy'),('L','learned_omission_density'),('P','fn_probability_density')]:
            mask=np.array([r['selected_'+c]=='True' for r in q]);saved=np.unpackbits(packed[f'{region}|{policy}|0.1'])[:len(q)].astype(bool);checks[region+'_actual_selection_'+c]=bool(np.array_equal(mask,saved))
    checks['density_scale_covers_every_nonempty_bin']=all(float(npy[k][npy[k]>0].min())>=.0001 and float(npy[k].max())<=10000 for k in npy if 'joint' in k or 'copula' in k)
    qu=rows('panel_c_conditional_quantiles.csv');checks['all_positive_FN_quartiles_visible']=all(0<=float(r['positive_D_q25'])<=float(r['positive_D_median'])<=float(r['positive_D_q75'])<=.052 for r in qu)
    risk=read(SD/'risk_head_provenance.json');checks['risk_head_OOF_train_vs_test_distinction']=risk['core_model_and_risk_weights_fixed_before_test'] and not risk['risk_head_development_oof_predictions_available'] and sum(x['risk_training_patches'] for x in risk['system_oof_sources'])==risk['training_patches']
    bands=rows('panel_f_simultaneous_intervals.csv');original=rows('all_fixed_map_review_results.csv');checks['eight_real_frozen_effects']=len(bands)==8
    for r in bands:
        one=next(x for x in original if x['region']==r['region'] and x['policy']==r['first_policy'] and float(x['budget_fraction'])==.1);two=next(x for x in original if x['region']==r['region'] and x['policy']==r['second_policy'] and float(x['budget_fraction'])==.1)
        checks[r['region']+'_'+r['second_policy']+'_'+r['endpoint']]=abs(float(one[r['endpoint']])-float(two[r['endpoint']])-float(r['difference_first_minus_second']))<1e-9 and int(r['family_estimates'])==12 and int(r['block_size_m'])==10000
    inter=read(SD/'interval_provenance.json');checks['full_interval_family_unchanged']=inter['all_family_reproduction_record_sha256']==sha(ROOT/'figures/main/Fig04/source_data/interval_reproduction.json')
    cases=read(SD/'panel_e_case_selection.json')['cases'];displays=read(AS/'geographic_display_parameters.json')
    for r in cases:
        key=r['case_id'];z=np.load(SD/f'panel_e_{key}.npz');valid=z['valid'];w=z['area_m2'];fn_mask=valid&(z['label']==1)&(z['probability']<.5);h=z['selected_H'].repeat(64,0).repeat(64,1);l=z['selected_L'].repeat(64,0).repeat(64,1);casesummary=r['FN_by_selection_km2']
        checks[key+'_native_shape']=valid.shape==(256,1024) and len(r['prediction_indices'])==4
        checks[key+'_exact_FN_partition']=all(abs(w[fn_mask&mask].sum()/1e6-casesummary[name])<1e-12 for name,mask in [('H_only',h&~l),('L_only',l&~h),('both',h&l),('neither',~h&~l)])
        checks[key+'_recovery_conservation']=all(abs(r['initial_FN_km2']-r[c+'_corrected_FN_km2']-r[c+'_residual_FN_km2'])<1e-12 for c in 'HL')
        dp=next(x for x in displays if x['case_id']==key);ext=z['extent'];aspect=1/np.cos(np.deg2rad((ext[2]+ext[3])/2));checks[key+'_geographic_aspect']=abs(dp['longitude_latitude_aspect']-aspect)<1e-12 and abs(dp['rendered_height_mm']-dp['rendered_width_mm']*(ext[3]-ext[2])/(ext[1]-ext[0])*aspect)<1e-10
    # Full panel-level source map; shared unit table is an explicit input to a/c/d/e.
    manifests=[('a',['review_units.csv.gz','panel_a_distribution_summary.csv','density_statistics.npz','panel_ac_density_bins.csv.gz','analysis_rules.json'],'area-weighted units; positive-FN histograms and all-unit weighted midranks'),('b',['panel_b_budget_capture.csv','review_selections_packed.npz','review_selection_encoding.json','all_fixed_map_review_results.csv'],'actual whole-unit area-budget selections; 0/1/5/10/20%'),('c',['review_units.csv.gz','panel_c_score_bin_means.csv','panel_c_conditional_quantiles.csv','risk_head_provenance.json','density_statistics.npz'],'frozen-test unit scores; weighted means and positive-FN empirical quartiles'),('d',['review_units.csv.gz','panel_d_disjoint_intersections.csv','panel_d_recovered_set_totals.csv'],'exact disjoint unit-ID intersections; valid and contained-FN area'),('e',['panel_e_case_selection.json','panel_e_all_candidate_windows.csv','panel_e_patch_statistics.csv']+[p.name for p in sorted(SD.glob('panel_e_*_favouring*'))],'outcome-selected contiguous native-grid windows; frozen regional 10% selection'),('f',['panel_f_simultaneous_intervals.csv','interval_provenance.json'],'eight saved paired effects from 12-estimate simultaneous families per region')]
    with (OUT/'Fig05_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','source_files','statistical_unit_and_rule','release_sha256','source_sha256']);writer.writeheader()
        for panel,names,rule in manifests:writer.writerow(dict(panel=panel,source_files=';'.join(names),statistical_unit_and_rule=rule,release_sha256=release,source_sha256=json.dumps({n:sha(SD/n) for n in names})))
    for suffix in ['','_Arial']:
        stem='Fig05_main'+suffix;pdf=PdfReader(OUT/(stem+'.pdf'));page=pdf.pages[0];fonts=[str(v.get_object().get('/BaseFont')) for v in page['/Resources']['/Font'].values()]
        checks[stem+'_page_size']=len(pdf.pages)==1 and abs(float(page.mediabox.width)*25.4/72-180)<.01 and abs(float(page.mediabox.height)*25.4/72-205)<.01
        checks[stem+'_font_family']=all(('Arial' if suffix else 'TimesNewRoman') in f for f in fonts);checks[stem+'_selectable_pdf_text']=len(page.extract_text())>1500
        tree=ET.parse(OUT/(stem+'.svg'));texts=tree.findall('.//{http://www.w3.org/2000/svg}text');labels=[''.join(t.itertext()) for t in texts];sizes=[float(m.group(1)) for t in texts for m in [re.search(r'([\d.]+)px',t.attrib.get('style',''))] if m];paths=tree.findall('.//{http://www.w3.org/2000/svg}path');images=tree.findall('.//{http://www.w3.org/2000/svg}image');roles={}
        expected=dict(capture_=14,risk_bin_mean_=2,risk_conditional_=2,upset_area_=14,upset_fn_=14,upset_membership_=14,review_cell_=256,reference_flood_=4,paired_interval_=8)
        for prefix,n in expected.items():
            items=[g for g in tree.findall('.//{http://www.w3.org/2000/svg}g') if g.attrib.get('id','').startswith(prefix)];roles[prefix]=len(items);checks[stem+'_'+prefix+'vectors']=len(items)==n and all(g.findall('.//{http://www.w3.org/2000/svg}path') or g.findall('.//{http://www.w3.org/2000/svg}use') for g in items)
        # Verify the actual intended labels, rather than an arbitrary text-element count.
        expected_labels={r['text'] for r in read(AS/('text_bounds'+suffix+'.json'))}
        expected_labels.update(['Observed FN density','Learned score L','Residual area (km²)','Mean entropy (nats)','Risk decile'])
        checks[stem+'_editable_text']=expected_labels.issubset(set(labels)) and all(c in labels for c in 'abcdef');checks[stem+'_minimum_body_size']=bool(sizes) and min(sizes)>=7
        bounds=read(AS/('text_bounds'+suffix+'.json'));checks[stem+'_labels_inside_page']=all(r['x0_mm']>=0 and r['x1_mm']<=180 and r['y0_mm']>=0 and r['y1_mm']<=205 for r in bounds);collisions=[]
        for i,u in enumerate(bounds):
            for v in bounds[i+1:]:
                if min(u['x1_mm'],v['x1_mm'])-max(u['x0_mm'],v['x0_mm'])>.2 and min(u['y1_mm'],v['y1_mm'])-max(u['y0_mm'],v['y0_mm'])>.2:collisions.append([u['text'],v['text']])
        checks[stem+'_figure_labels_no_collision']=not collisions
        with Image.open(OUT/(stem+'.png')) as png,Image.open(OUT/(stem+'.tiff')) as tif:
            dims=png.size;checks[stem+'_1200dpi']=all(abs(float(d)-1200)<.02 for d in (*png.info['dpi'],*tif.info['dpi']));checks[stem+'_raster_dimensions']=dims==tif.size and all(abs(x-y)<1.1 for x,y in zip(dims,[180/25.4*1200,205/25.4*1200]));equal=True
            for y in range(0,png.height,512):
                box=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_png_tiff_match']=equal;checks[stem+'_tiff_lzw']=tif.info.get('compression')=='tiff_lzw'
        doc=pdfium.PdfDocument(OUT/(stem+'.pdf'));preview=doc[0].render(scale=3).to_pil();preview.save(AS/(stem+'_pdf_preview.png'));large=doc[0].render(scale=5).to_pil()
        for name,(x0,y0,x1,y1) in {'top':(0,0,180,80),'middle':(0,79,180,145),'geography':(0,143,113,205),'effects':(114,143,180,205)}.items():large.crop((round(x0/180*large.width),round(y0/205*large.height),round(x1/180*large.width),round(y1/205*large.height))).save(AS/(stem+'_'+name+'_pdf.png'))
        doc.close();details[stem]=dict(fonts=fonts,svg_texts=len(texts),svg_paths=len(paths),vector_role_counts=roles,raster_layers=len(images),body_sizes=sorted(set(sizes)),pixels=dims,figure_label_collisions=collisions)
    caption=(OUT/'Fig05_caption_en.md').read_text(encoding='utf-8');details['caption_words']=len(caption.split());checks['caption_english_and_under_300_words']=details['caption_words']<=300 and not re.search('[\u4e00-\u9fff]',caption)
    checks['no_PPTX_requested_or_generated']=not list(OUT.glob('*.pptx'))
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['Fig05_validation.json','Fig05_delivery.zip','Fig05_delivery.sha256'] or p.suffix=='.log':continue
        files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=bool(all(checks.values())),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=release,checks={k:bool(v) for k,v in checks.items()},details=details,sha256_manifest=files,visual_qa='Source audit, layout refinement, actual-PDF review and Arial glyph correction; see assets/visual_qa.md')
    (OUT/'Fig05_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v],details=details),indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
