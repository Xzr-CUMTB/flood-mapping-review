"""Audit compact numeric sources, physical vector geometry and lossless final exports."""
import os,sys,json,csv,re,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
OUT=Path(os.path.abspath(__file__)).parents[1];ROOT=OUT.parents[2];SD=OUT/'source_data';AS=OUT/'assets'
MODELS=['sar','fusion','ordinary_gate','value_gate'];REGIONS=['Beijing_Baoding','Harbin'];BUDGETS=[.01,.05,.1,.2];WEIGHTS=[.25,1.,4.]
NS='{http://www.w3.org/2000/svg}';Image.MAX_IMAGE_PIXELS=160_000_000
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(name):
    with (SD/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def f(r,k):return float(r[k])
def near(a,b,tol=1e-9):return abs(a-b)<tol
def coords(path):return np.array([(float(x),float(y)) for x,y in re.findall(r'[ML]\s+([-+\d.eE]+)\s+([-+\d.eE]+)',path.attrib['d'])])
def main():
    audit=read(SD/'scientific_audit.json');release=audit['release_sha256'];checks={'source_audit_passed':audit['passed']};details={}
    checks['release_unchanged']=sha(ROOT/'configs/core_release_s17.json')==release
    checks['all_bound_inputs_unchanged']=all(sha(ROOT/r['path'])==r['sha256'] for r in read(SD/'input_bindings.json'))
    for name in ['Fig01','Fig02','Fig03','Fig04','Fig05']:
        prior=ROOT/'figures/main'/name;v=read(prior/(name+'_validation.json'));checks[name+'_unchanged']=v['passed'] and all(sha(prior/r['path'])==r['sha256'] for r in v['sha256_manifest'])
    base=rows('verified_core_review_results.csv');candidates=rows('all_budget_cost_candidates.csv');atlas=rows('panel_a_decision_atlas.csv');params=rows('panel_b_analytical_parameters.csv');curves=rows('panel_b_analytical_lines.csv');marks=rows('panel_b_preset_weight_markers.csv');regrets=rows('panel_c_primary_regret.csv');scenarios=rows('panel_d_all_primary_scenarios.csv');ledger=rows('panel_d_displayed_scenarios.csv');geom=rows('rendered_quantitative_geometry.csv')
    checks['all_compact_data_release_bound']=all(r['release_sha256']==release for collection in [base,candidates,atlas,params,curves,marks,regrets,scenarios,ledger] for r in collection)
    checks['32_distinct_review_records']=len(base)==32 and {(r['region'],r['model'],f(r,'budget_fraction')) for r in base}=={(r,m,b) for r in REGIONS for m in MODELS for b in BUDGETS}
    with (ROOT/'runs/core_s17_evaluation/review_budget_results.csv').open(encoding='utf-8',newline='') as ff:original=list(csv.DictReader(ff))
    source_ok=True
    for r in base:
        q=next(x for x in original if x['region']==r['region'] and x['model']==r['model'] and x['policy']==r['policy'] and f(x,'budget_fraction')==f(r,'budget_fraction'))
        source_ok &= all(near(f(r,k),f(q,k)) for k in ['actual_area_fraction','selected_units','initial_fn_km2','initial_fp_km2','residual_fn_km2','residual_fp_km2','area_loss_lambda_0.25','area_loss_lambda_1.0','area_loss_lambda_4.0'])
    checks['compact_review_values_match_frozen_csv']=source_ok
    checks['96_distinct_candidates']=len(candidates)==96 and len({(r['region'],r['model'],f(r,'budget_fraction'),f(r,'lambda_fp')) for r in candidates})==96
    checks['all_96_losses_reproduced']=all(near(f(r,'weighted_loss_km2eq'),f(r,'residual_fn_km2')+f(r,'lambda_fp')*f(r,'residual_fp_km2')) for r in candidates)
    checks['whole_unit_budgets_respected']=all(0<=f(r,'actual_area_fraction')<=f(r,'budget_fraction')+1e-12 and near(f(r,'reviewed_valid_area_km2'),f(r,'actual_area_fraction')*f(r,'total_valid_area_km2')) for r in candidates)
    atlas_ok=len(atlas)==24;regret_ok=True;rank_ok=True
    for r in atlas:
        same=[q for q in candidates if q['region']==r['region'] and f(q,'budget_fraction')==f(r,'budget_fraction') and f(q,'lambda_fp')==f(r,'lambda_fp')];s=sorted(same,key=lambda q:(f(q,'weighted_loss_km2eq'),q['model']));low=f(s[0],'weighted_loss_km2eq');next_low=f(s[1],'weighted_loss_km2eq')
        atlas_ok &= len(s)==4 and {x['model'] for x in s}==set(MODELS) and r['best_model']==s[0]['model'] and r['second_model']==s[1]['model'] and near(f(r,'best_loss_km2eq'),low) and near(f(r,'second_loss_km2eq'),next_low) and near(f(r,'margin_km2eq'),next_low-low) and r['tied_best']=='False'
        for q in s:
            regret_ok &= f(q,'regret_km2eq')>=0 and near(f(q,'regret_km2eq'),f(q,'weighted_loss_km2eq')-low)
            rank_ok &= int(q['rank'])==1+sum(f(z,'weighted_loss_km2eq')<f(q,'weighted_loss_km2eq') for z in s)
    checks['24_complete_candidate_minima_and_margins']=atlas_ok;checks['96_nonnegative_correct_regrets']=regret_ok;checks['full_precision_ranks']=rank_ok
    checks['24_primary_regrets_match_complete_table']=len(regrets)==24 and all(r in candidates and f(r,'budget_fraction')==.1 for r in regrets)
    pair_ok=True
    for p in params:
        vals={r['model']:r for r in base if r['region']==p['region'] and f(r,'budget_fraction')==.1};dn=f(vals['value_gate'],'residual_fn_km2')-f(vals['fusion'],'residual_fn_km2');dp=f(vals['value_gate'],'residual_fp_km2')-f(vals['fusion'],'residual_fp_km2');root=-dn/dp
        pair_ok &= near(dn,f(p,'delta_FN_km2')) and near(dp,f(p,'delta_FP_km2')) and near(root,f(p,'algebraic_root_lambda'))
        pair_ok &= all(near(f(q,'delta_weighted_loss_km2eq'),dn+f(q,'lambda_fp')*dp) for q in curves+marks if q['region']==p['region'])
    checks['fixed_error_analytical_lines_and_markers']=pair_ok and len(marks)==6
    bb,hb=params;root=f(hb,'positive_break_even_lambda');checks['root_positive_Harbin_absent_nonnegative_BB']=bb['positive_break_even_lambda']=='' and f(bb,'delta_FN_km2')>0 and f(bb,'delta_FP_km2')>0 and near(root,1.882868947827029) and near(f(hb,'delta_FN_km2')+root*f(hb,'delta_FP_km2'),0)
    checks['pairwise_separate_from_candidate_minimum']=all(r['best_model']=='ordinary_gate' for r in atlas if r['region']=='Harbin' and f(r,'budget_fraction')==.1)
    ledger_ok=len(ledger)==4 and len(scenarios)==6 and {(r['region'],f(r,'lambda_fp')) for r in ledger}=={(region,w) for region in REGIONS for w in [.25,4.]}
    for r in ledger:
        q=next(x for x in candidates if x['region']==r['region'] and x['model']==r['best_model'] and f(x,'lambda_fp')==f(r,'lambda_fp') and f(x,'budget_fraction')==.1)
        ledger_ok &= all(near(f(r,a),f(q,b)) for a,b in [('residual_FN_km2','residual_fn_km2'),('residual_FP_km2','residual_fp_km2'),('best_loss_km2eq','weighted_loss_km2eq'),('reviewed_valid_area_km2','reviewed_valid_area_km2')])
    checks['scenario_selection_and_actual_quantities']=ledger_ok
    checks['quantitative_geometry_proportional']=len(geom)==32 and all(near(f(r,'length_mm'),f(r,'value')*f(r,'scale_mm_per_unit')) for r in geom)
    checks['uncertainty_scope_point_estimates']='no weighted-loss or root confidence intervals' in read(SD/'uncertainty_scope.json')['display']
    manifest=[('a',['panel_a_decision_atlas.csv','all_budget_cost_candidates.csv','analysis_rules.json'],'24 evaluated conditions; all four candidates; full-precision ranks and best-second gaps'),('b',['panel_b_analytical_parameters.csv','panel_b_analytical_lines.csv','panel_b_preset_weight_markers.csv','verified_core_review_results.csv','uncertainty_scope.json'],'fixed 10% errors; analytical reweighting over 0<=lambda<=4; point-estimate root'),('c',['panel_c_primary_regret.csv','all_budget_cost_candidates.csv'],'candidate loss minus four-candidate minimum; primary 10% budget'),('d',['panel_d_displayed_scenarios.csv','panel_d_all_primary_scenarios.csv','rendered_quantitative_geometry.csv'],'both regions at preset weight endpoints; actual reviewed and residual areas')]
    with (OUT/'Fig06_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as ff:
        writer=csv.DictWriter(ff,fieldnames=['panel','source_files','statistical_unit_and_rule','release_sha256','source_sha256']);writer.writeheader()
        for panel,names,rule in manifest:writer.writerow(dict(panel=panel,source_files=';'.join(names),statistical_unit_and_rule=rule,release_sha256=release,source_sha256=json.dumps({n:sha(SD/n) for n in names})))
    for suffix in ['','_Arial']:
        stem='Fig06_main'+suffix;reader=PdfReader(OUT/(stem+'.pdf'));page=reader.pages[0];fonts=[str(v.get_object().get('/BaseFont')) for v in page['/Resources']['/Font'].values()]
        checks[stem+'_page_size']=len(reader.pages)==1 and near(float(page.mediabox.width)*25.4/72,180,.01) and near(float(page.mediabox.height)*25.4/72,170,.01)
        checks[stem+'_font_family']=all(('Arial' if suffix else 'TimesNewRoman') in x for x in fonts)
        embedded=[]
        for item in page['/Resources']['/Font'].values():
            obj=item.get_object();descendant=obj.get('/DescendantFonts');fd=(descendant[0].get_object() if descendant else obj).get('/FontDescriptor');embedded.append(bool(fd and fd.get_object().get('/FontFile2')))
        checks[stem+'_fonts_embedded']=all(embedded)
        checks[stem+'_selectable_pdf_text']=all(x in page.extract_text() for x in ['Budget','Analytical','Candidate regret','Retrospective'])
        tree=ET.parse(OUT/(stem+'.svg'));texts=tree.findall('.//'+NS+'text');labels=[''.join(t.itertext()) for t in texts];paths=tree.findall('.//'+NS+'path');images=tree.findall('.//'+NS+'image');groups={g.attrib.get('id',''):g for g in tree.findall('.//'+NS+'g')};sizes=[float(m.group(1)) for t in texts for m in [re.search(r'([\d.]+)px',t.attrib.get('style',''))] if m];counts={}
        expected=dict(decision_cell_=24,margin_bar_=24,primary_budget_=2,preference_strip_=2,pairwise_zero=1,analytical_curve_=2,preset_markers_=2,break_even_point=1,regret_cell_=24,minimum_symbol_=6,ledger_area_=8)
        for prefix,n in expected.items():
            items=[g for key,g in groups.items() if key.startswith(prefix)];counts[prefix]=len(items);checks[stem+'_'+prefix+'vectors']=len(items)==n and all(g.findall('.//'+NS+'path') or g.findall('.//'+NS+'use') for g in items)
        checks[stem+'_no_raster_layers']=len(images)==0
        checks[stem+'_editable_text']=set(r['text'] for r in read(AS/('text_bounds'+suffix+'.json'))).issubset(set(labels)) and all(x in labels for x in 'abcd')
        checks[stem+'_minimum_text_size']=bool(sizes) and min(sizes)>=7
        all_bounds=read(AS/('all_text_bounds'+suffix+'.json'));outside=[r for r in all_bounds if r['x0_mm']<0 or r['x1_mm']>180 or r['y0_mm']<0 or r['y1_mm']>170];checks[stem+'_all_text_inside_page']=not outside
        bounds=read(AS/('text_bounds'+suffix+'.json'));collisions=[]
        for i,u in enumerate(bounds):
            for v in bounds[i+1:]:
                if min(u['x1_mm'],v['x1_mm'])-max(u['x0_mm'],v['x0_mm'])>.2 and min(u['y1_mm'],v['y1_mm'])-max(u['y0_mm'],v['y0_mm'])>.2:collisions.append([u['text'],v['text']])
        checks[stem+'_figure_labels_no_collision']=not collisions
        vector_errors=[]
        for r in geom:
            if r['role']=='margin_bar':key=f"margin_bar_{r['region']}_{BUDGETS.index(f(r,'budget_fraction'))}_{WEIGHTS.index(f(r,'lambda_fp'))}"
            else:key=f"ledger_area_{r['region']}_{f(r,'lambda_fp')}_{r['endpoint']}"
            xy=coords(groups[key].find('.//'+NS+'path'));vector_errors.append(abs((xy[:,0].max()-xy[:,0].min())*25.4/72-f(r,'length_mm')))
            checks[stem+'_'+key+'_inside_page']=bool((xy[:,0]>=0).all() and (xy[:,0]<=180*72/25.4).all() and (xy[:,1]>=0).all() and (xy[:,1]<=170*72/25.4).all())
        checks[stem+'_actual_bar_geometry']=max(vector_errors)<1e-6
        strip=coords(groups['preference_strip_fusion'].find('.//'+NS+'path'));point=groups['break_even_point'].find('.//'+NS+'use');checks[stem+'_actual_strip_root_alignment']=near(strip[:,0].max(),float(point.attrib['x']),1e-5)
        curve_errors=[]
        for region in REGIONS:
            q=[r for r in curves if r['region']==region];actual=coords(groups['analytical_curve_'+region].find('.//'+NS+'path'));wanted=np.array([[(14+f(r,'lambda_fp')/4.12*71)*72/25.4,(96+(36-f(r,'delta_weighted_loss_km2eq'))/44*24)*72/25.4] for r in q]);assert actual.shape==wanted.shape;curve_errors.append(float(np.abs(actual-wanted).max()))
        checks[stem+'_actual_curve_vertices']=max(curve_errors)<1e-5
        with Image.open(OUT/(stem+'.png')) as png,Image.open(OUT/(stem+'.tiff')) as tif:
            dims=png.size;checks[stem+'_1200dpi']=all(abs(float(d)-1200)<.02 for d in (*png.info['dpi'],*tif.info['dpi']));checks[stem+'_raster_dimensions']=dims==tif.size and all(abs(x-y)<1.1 for x,y in zip(dims,[180/25.4*1200,170/25.4*1200]));equal=True
            for y in range(0,png.height,512):
                rect=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(rect).convert('RGB'),tif.crop(rect).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_png_tiff_match']=equal;checks[stem+'_tiff_lzw']=tif.info.get('compression')=='tiff_lzw'
        doc=pdfium.PdfDocument(OUT/(stem+'.pdf'));preview=doc[0].render(scale=3).to_pil();preview.save(AS/(stem+'_pdf_preview.png'));doc[0].render(scale=120/72).to_pil().save(AS/(stem+'_size_check.png'));large=doc[0].render(scale=5).to_pil()
        for name,(x0,y0,x1,y1) in {'a':(0,0,180,73),'b':(0,74,94,134),'c':(94,74,180,134),'d':(0,134,180,170)}.items():large.crop((round(x0/180*large.width),round(y0/170*large.height),round(x1/180*large.width),round(y1/170*large.height))).save(AS/(stem+'_'+name+'_pdf.png'))
        doc.close();details[stem]=dict(fonts=fonts,svg_texts=len(texts),svg_paths=len(paths),vector_role_counts=counts,raster_layers=len(images),text_sizes=sorted(set(sizes)),pixels=dims,figure_label_collisions=collisions,outside_text=outside,max_bar_length_error_mm=max(vector_errors),max_curve_vertex_error_pt=max(curve_errors))
    caption=(OUT/'Fig06_caption_en.md').read_text(encoding='utf-8');details['caption_words']=len(caption.split());checks['caption_English_under_300_words']=details['caption_words']<=300 and not re.search('[\u4e00-\u9fff]',caption)
    checks['no_PPTX']=not list(OUT.glob('*.pptx'))
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['Fig06_validation.json','Fig06_delivery.zip','Fig06_delivery.sha256'] or p.suffix=='.log':continue
        files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=bool(all(checks.values())),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=release,checks={k:bool(v) for k,v in checks.items()},details=details,sha256_manifest=files,visual_qa='Three rounds; actual PDF panel inspection in both fonts; see assets/visual_qa.md')
    (OUT/'Fig06_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],failed=[k for k,v in checks.items() if not v],details=details),indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
