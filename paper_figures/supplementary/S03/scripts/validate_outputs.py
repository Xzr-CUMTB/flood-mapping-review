"""Independent numeric, provenance, vector and raster checks for S01--S05.

Uses CSV arithmetic and SVG/PDF inspection; does not train or alter frozen data.
Run with the bundled document Python (numpy, Pillow, pypdf, pypdfium2).
"""
import os,sys,json,csv,gzip,re,hashlib,argparse,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageChops
from pypdf import PdfReader
import pypdfium2 as pdfium
ROOT=next(p for p in Path(os.path.abspath(__file__)).parents if (p/'configs/core_release_s17.json').is_file());BASE=ROOT/'figures/supplementary'
EVENTS=['2020_flood_season','2021_Henan','2024_April','2024_June'];GROUPS=EVENTS+['2023_holdout'];NS='{http://www.w3.org/2000/svg}';Image.MAX_IMAGE_PIXELS=190_000_000
CACHE={}
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    p=p.resolve();key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if key in CACHE:return CACHE[key]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    CACHE[key]=h.hexdigest();return CACHE[key]
def rows(p):
    with (gzip.open(p,'rt',encoding='utf-8',newline='') if p.suffix=='.gz' else p.open(encoding='utf-8',newline='')) as f:return list(csv.DictReader(f))
def num(r,k):return float(r[k])
def true(r,k):return r[k]=='True'
def near(a,b):return bool(np.isclose(a,b,atol=1e-10,rtol=1e-10))
def text_clashes(bounds):
    out=[]
    for i,a in enumerate(bounds):
        for b in bounds[i+1:]:
            if min(a['x1_mm'],b['x1_mm'])-max(a['x0_mm'],b['x0_mm'])>.2 and min(a['y1_mm'],b['y1_mm'])-max(a['y0_mm'],b['y0_mm'])>.2:out.append([a['text'],b['text']])
    return out
PANELS={
1:{'a':['site_dates.csv'],'b':['event_roles.csv'],'c':['site_dates.csv'],'d':['event_composition.csv','retained_footprints.csv.gz']},
2:{'a':['inclusion_membership.csv.gz','inclusion_stages.csv'],'b':['tile_audit.csv.gz'],'c':['displayed_scene.npz','scene_provenance.json'],'d':['tile_audit.csv.gz','event_composition.csv'],'e':['overlap_pairs.csv','overlap_degree.csv','displayed_overlap_footprints.csv'],'f':['displayed_scene.npz','label_values.csv','label_semantics.json']},
3:{'a':['crossfit_provenance.csv','training_prediction_boolean_matrix.csv','oof_product_index.csv.gz'],'b':['omission_supervision.json','crossfit_provenance.csv'],'c':['integrity_counts.csv','training_rechecks.csv','frozen_files_verified.json','prediction_hash_recheck.json','completed_verification_record.json']},
4:{'a':['all_oof_metrics.csv','effects_vs_fusion.csv'],'b':['effects_vs_fusion.csv'],'c':['effects_vs_fusion.csv'],'d':['all_oof_metrics.csv','event_composition.csv'],'e':['core_model_ranks.csv'],'f':['effects_vs_fusion.csv']},
5:{'a':['all_core_map_metrics.csv','benchmark_cells.csv','model_provenance.csv'],'b':['paired_controls.csv'],'c':['capacity_comparison.csv','model_provenance.csv']}}
def validate(n):
    out=BASE/f'S{n:02d}';sd=out/'source_data';assets=out/'assets';checks={};details={};audit=js(sd/'scientific_audit.json');release=audit['release_sha256'];read=lambda name:rows(sd/name)
    checks['source_audit_passed']=audit['passed'];checks['release_unchanged']=sha(ROOT/'configs/core_release_s17.json')==release
    checks['compact_sources_unchanged']=all(sha(sd/name)==v for name,v in audit['source_files'].items())
    checks['bound_inputs_unchanged']=all(sha(ROOT/r['path'])==r['sha256'] for r in js(sd/'input_bindings.json'))
    for i in range(1,7):
        prior=ROOT/f'figures/main/Fig{i:02d}';v=js(prior/f'Fig{i:02d}_validation.json');checks[f'main_figure_{i}_preserved']=v['passed'] and all(sha(prior/r['path'])==r['sha256'] for r in v['sha256_manifest'])
    if n==1:
        d=read('site_dates.csv');g=read('event_composition.csv');r=read('event_roles.csv');p=read('retained_footprints.csv.gz')
        checks['nine_distinct_sites']=len(d)==len({x['site'] for x in d})==9
        checks['date_intervals_recomputed']=all((datetime.fromisoformat(x['sar_date'])-datetime.fromisoformat(x['optical_date'])).days==num(x,'optical_age_days') and (datetime.fromisoformat(x['sar_date'])-datetime.fromisoformat(x['paper_flood_date'])).days==num(x,'sar_delay_days') for x in d)
        checks['two_zero_delays_and_215_day_age']=sum(num(x,'sar_delay_days')==0 for x in d)==2 and max(num(x,'optical_age_days') for x in d)==215
        checks['all_17019_retained_ids']=len(p)==len({x['sample_id'] for x in p})==17019 and sum(num(x,'retained_tiles') for x in d)==17019
        checks['five_actual_event_counts']=all(sum(x['event_group']==z['event_group'] for x in p)==int(z['tiles']) for z in g)
        roles={(x['event_group'],x['stage']):x['role'] for x in r};checks['core_and_outer_exclusion_roles']=len(roles)==30 and all(roles[(e,f'Outer {j+1}')]==('unused' if e=='2023_holdout' else 'held out' if e==EVENTS[j] else 'training') for e in GROUPS for j in range(4)) and all(roles[(e,'Final fit')]==('unused' if e=='2023_holdout' else 'training') for e in GROUPS) and all(roles[(e,'Core test')]==('evaluation' if e=='2023_holdout' else 'unused') for e in GROUPS)
    if n==2:
        d=read('tile_audit.csv.gz');m=read('inclusion_membership.csv.gz');st=read('inclusion_stages.csv');pairs=read('overlap_pairs.csv');kept={x['sample_id'] for x in m if true(x,'nonoverlap_retained')}
        checks['exact_id_stage_counts']=all(sum(true(x,r['stage']) for x in m)==int(r['count']) for r in st) and [int(x['count']) for x in st]==[21483,18775,18431,17019,8786,8233]
        checks['development_core_partition']=all(int(true(x,'development'))+int(true(x,'core'))==int(true(x,'nonoverlap_retained')) for x in m)
        checks['all_overlap_pairs_retained_conflict_free']=len(pairs)==3386 and not any(x['sample_a'] in kept and x['sample_b'] in kept for x in pairs)
        checks['retained_max_offset']=near(max(num(x,'offset_px') for x in d if true(x,'retained')),.9658286031061382)
        degree=read('overlap_degree.csv');checks['degree_sum_equals_twice_pair_count']=sum(int(x['degree']) for x in degree)==2*len(pairs)
        checks['retained_geodetic_summaries']=all(near(sum(num(x,'valid_area_km2') for x in d if true(x,'retained') and x['event_group']==g['event_group']),num(g,'valid_area_km2')) and near(sum(num(x,'flood_area_km2')==0 for x in d if true(x,'retained') and x['event_group']==g['event_group'])/int(g['tiles']),num(g,'zero_flood_tile_fraction')) for g in read('event_composition.csv'))
        lv=read('label_values.csv');checks['actual_label_values_0_1_no_255']=all(num(x,'pixels')==0 for x in lv if x['label_value']=='255') and sum(num(x,'pixels') for x in lv if x['population']=='All original tiles')==21483*65536
        provenance=js(sd/'scene_provenance.json');z=np.load(sd/'displayed_scene.npz');idx=provenance['cache_index'];cache=ROOT/'data/processed/sen2gf3'
        checks['displayed_window_matches_cache']=all(np.array_equal(z[a],np.load(cache/b,mmap_mode='r')[idx]) for a,b in [('x','x_uint8.npy'),('label','labels_uint8.npy'),('support','support_uint8.npy')])
        checks['raw_display_scene_hashes']=all(sha(ROOT/x['path'])==x['sha256'] for x in provenance['raw_files'])
        checks['median_reference_only_rule']=provenance['sample_id']==13550 and provenance['candidate_count']==4726
    if n==3:
        p=read('crossfit_provenance.csv');b=read('training_prediction_boolean_matrix.csv');index=read('oof_product_index.csv.gz');jobs=read('training_rechecks.csv');frozen=js(sd/'frozen_files_verified.json');fresh=js(sd/'prediction_hash_recheck.json')
        checks['32_expert_dependencies_exclude_targets']=len(p)==32 and all(x['target_event'] not in x['training_groups'].split(';') and x['outer_event'] not in x['training_groups'].split(';') and true(x,'exclusion_verified') for x in p)
        checks['128_boolean_relations_no_leakage']=len(b)==128 and all(not(true(x,'trained_on_group') and (true(x,'predicted_group') or true(x,'outer_excluded'))) for x in b)
        checks['8786_unique_oof_tiles']=len(index)==len({x['sample_id'] for x in index})==8786
        checks['38_jobs_rechecked']=len(jobs)==38 and len({x['run_id'] for x in jobs})==38 and all(int(x['checkpoint_tensors'])>0 for x in jobs)
        checks['142_frozen_file_sha_matches']=len(frozen)==142 and all(sha(ROOT/x['path'])==x['sha256'] for x in frozen)
        checks['13_whole_prediction_file_hashes_rechecked']=fresh['passed'] and len(fresh['files'])==13 and fresh['release_sha256']==release and all(x['passed'] for x in fresh['files'])
        checks['risk_has_8786_development_records']=js(sd/'omission_supervision.json')['config']['samples']==8786 and not js(sd/'omission_supervision.json')['core_labels_used_in_training']
        checks['all_integrity_counts_match']=all(x['expected']==x['observed'] for x in read('integrity_counts.csv'))
    if n==4:
        data=read('all_oof_metrics.csv');effects=read('effects_vs_fusion.csv');ranks=read('core_model_ranks.csv');keys={(x['event_group'],x['model']):x for x in data}
        originals={}
        for x in data:
            path=x['source_csv']
            if path not in originals:originals[path]=rows(ROOT/path)
        checks['24_complete_oof_outputs']=len(data)==len(keys)==24 and set(x['event_group'] for x in data)==set(EVENTS)
        checks['oof_values_match_actual_csv']=all(all(near(num(x,k),num(next(r for r in originals[x['source_csv']] if r['model']==x['model']),k)) for k in ['flood_iou','brier','fn_km2','fp_km2']) for x in data)
        checks['96_effects_recomputed']=len(effects)==96 and all(near(num(x,'value'),num(keys[(x['event_group'],x['model'])],x['metric'])) and near(num(x,'difference'),num(x,'value')-num(keys[(x['event_group'],'fusion')],x['metric'])) and near(num(x,'beneficial_effect'),num(x,'difference')*(1 if x['metric']=='flood_iou' else -1)) and near(num(x,'colour_value'),num(x,'beneficial_effect')/num(x,'colour_scale')) for x in effects)
        checks['per_metric_max_absolute_scale']=all(near(num(x,'colour_scale'),max(abs(num(z,'beneficial_effect')) for z in effects if z['metric']==x['metric'])) for x in effects)
        checks['16_full_precision_core_ranks']=len(ranks)==16 and all(int(x['rank_IoU'])==1+sum(num(z,'flood_iou')>num(x,'flood_iou') for z in ranks if z['event_group']==x['event_group']) for x in ranks)
        checks['all_value_iou_signs_retained']=[np.sign(num(keys[(e,'sigmoid_value')],'flood_iou')-num(keys[(e,'fusion')],'flood_iou')) for e in EVENTS]==[-1,-1,1,-1]
        checks['exploratory_controls_distinguished']=all(true(x,'exploratory')==(x['model'] in ['analytic_value','ridge_value_001']) for x in data)
    if n==5:
        data=read('all_core_map_metrics.csv');cells=read('benchmark_cells.csv');meta=read('model_provenance.csv');pairs=read('paired_controls.csv');capacity=read('capacity_comparison.csv');keys={(x['region'],x['model']):x for x in data};original={(x['region'],x['model']):x for x in rows(ROOT/'runs/core_s17_evaluation/map_metrics.csv')};metrics=['flood_iou','fn_km2','fp_km2','brier','confident_omission_01']
        checks['all_32_actual_outputs']=len(data)==len(keys)==32 and len(meta)==16 and set(keys)==set(original)
        checks['160_raw_values_match_actual_csv']=len(cells)==160 and all(near(num(x,'value'),num(original[(x['region'],x['model'])],x['metric'])) for x in cells)
        checks['160_colour_effects_recomputed']=all(near(num(x,'beneficial_effect'),(num(x,'value')-num(keys[(x['region'],'fusion')],x['metric']))*(1 if x['metric']=='flood_iou' else -1)) and near(num(x,'colour_scale'),max(abs(num(z,'beneficial_effect')) for z in cells if z['metric']==x['metric'])) and near(num(x,'colour_value'),num(x,'beneficial_effect')/num(x,'colour_scale')) for x in cells)
        checks['temperature_binary_metrics_invariant']=all(near(num(keys[(region,model)],metric),num(keys[(region,model+'_temperature')],metric)) for region in ['Beijing_Baoding','Harbin'] for model in ['sar','fusion','value_gate'] for metric in ['flood_iou','fn_km2','fp_km2'])
        checks['paired_controls_actual']=len(pairs)==16 and all(near(num(x,'first_value'),num(keys[(x['region'],x['first_model'])],x['metric'])) and near(num(x,'second_value'),num(keys[(x['region'],x['second_model'])],x['metric'])) and near(num(x,'difference'),num(x,'first_value')-num(x,'second_value')) for x in pairs)
        checks['12_single_network_capacity_points']=len(capacity)==12 and all(x['model'] in [m['model'] for m in meta if m['output_type']=='trained network'] and near(num(x,'flood_iou'),num(keys[(x['region'],x['model'])],'flood_iou')) and int(float(x['parameters'])) in [1964529,5873617] for x in capacity)
        checks['ensemble_members_17_29_43']=next(x['ensemble_seeds'] for x in meta if x['model']=='fusion_ensemble')=='17;29;43'
    # Independently inspect each delivered vector and lossless raster pair.
    expected_roles={1:{'chronology_':9,'role_':30,'count_bar_':5},2:{'coverage_points_':2,'ecdf_':10,'inclusion_bar_':4},3:{'outer_matrix_':16,'inner_matrix_':36,'integrity_row_':8},4:{'performance_cell_':96,'paired_effect_':2,'joint_effect_':4,'rank_braid_':4,'synthesis_':20},5:{'benchmark_cell_':160,'paired_control_':4,'capacity_':12}}[n]
    for suffix in ['','_Arial']:
        stem=f'S{n:02d}_main'+suffix;meta=js(assets/('render_metadata'+suffix+'.json'));h=meta['height_mm'];pdf=PdfReader(out/(stem+'.pdf'));page=pdf.pages[0];font_objs=[x.get_object() for x in page['/Resources']['/Font'].values()];fonts=[str(x.get('/BaseFont')) for x in font_objs]
        checks[stem+'_physical_page']=len(pdf.pages)==1 and abs(float(page.mediabox.width)*25.4/72-180)<.01 and abs(float(page.mediabox.height)*25.4/72-h)<.01
        checks[stem+'_font_family']=all(('Arial' if suffix else 'TimesNewRoman') in x for x in fonts)
        checks[stem+'_embedded_fonts']=all(bool((x['/DescendantFonts'][0].get_object() if x.get('/DescendantFonts') else x)['/FontDescriptor'].get_object().get('/FontFile2')) for x in font_objs)
        extracted=page.extract_text();checks[stem+'_selectable_English_text']=len(extracted)>500 and not re.search('[\u4e00-\u9fff]',extracted)
        tree=ET.parse(out/(stem+'.svg'));texts=tree.findall('.//'+NS+'text');groups={g.attrib.get('id',''):g for g in tree.findall('.//'+NS+'g')};images=tree.findall('.//'+NS+'image');sizes=[float(z) for t in texts for z in re.findall(r'([\d.]+)px',t.attrib.get('style',''))]
        checks[stem+'_editable_text']=len(texts)>60 and min(sizes)>=7
        checks[stem+'_raster_layers_expected']=len(images)==(7 if n==2 else 0)
        checks[stem+'_all_semantic_vector_objects']=all(name in groups for name in meta['roles'])
        for prefix,count in expected_roles.items():checks[stem+'_'+prefix+'count']=sum(key.startswith(prefix) for key in groups)==count
        bounds=js(assets/('text_bounds'+suffix+'.json'));outside=[x for x in bounds if x['x0_mm']<-.01 or x['x1_mm']>180.01 or x['y0_mm']<-.01 or x['y1_mm']>h+.01];clashes=text_clashes(bounds)
        checks[stem+'_all_text_inside_page']=not outside;checks[stem+'_no_text_collisions']=not clashes
        checks[stem+'_all_scatter_points_visible']=not any(c['outside'] for a in meta['axes'] for c in a['collections'])
        if n==2:
            ims=[im for a in meta['axes'] for im in a['images']];checks[stem+'_seven_true_geo_extents']=len(ims)==7 and all(im['extent']==ims[0]['extent'] for im in ims);checks[stem+'_seven_500m_labels']=sum(x['text']=='500 m' for x in bounds)==7
        if n==5:checks[stem+'_all_capacity_points_visible']=sum(c['points'] for a in meta['axes'] for c in a['collections'] if (c['role'] or '').startswith('capacity_'))==12
        with Image.open(out/(stem+'.png')) as png,Image.open(out/(stem+'.tiff')) as tif:
            checks[stem+'_1200dpi']=all(abs(float(v)-1200)<.02 for v in (*png.info['dpi'],*tif.info['dpi']))
            checks[stem+'_raster_dimensions']=png.size==tif.size and all(abs(x-y)<1.1 for x,y in zip(png.size,[180/25.4*1200,h/25.4*1200]))
            equal=True
            for y in range(0,png.height,512):
                box=(0,y,png.width,min(y+512,png.height))
                if ImageChops.difference(png.crop(box).convert('RGB'),tif.crop(box).convert('RGB')).getbbox():equal=False;break
            checks[stem+'_lossless_png_tiff']=equal and tif.info.get('compression')=='tiff_lzw'
        doc=pdfium.PdfDocument(out/(stem+'.pdf'));pg=doc[0];pg.render(scale=120/72).to_pil().save(assets/(stem+'_print_size_check.png'));pg.close();doc.close()
        details[stem]=dict(fonts=fonts,svg_text_objects=len(texts),raster_layers=len(images),minimum_font_pt=min(sizes),outside_text=outside,text_collisions=clashes)
    with (out/f'S{n:02d}_panel_data_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['panel','source_files','source_sha256','release_sha256']);w.writeheader()
        for panel,names in PANELS[n].items():w.writerow(dict(panel=panel,source_files=';'.join(names),source_sha256=json.dumps({name:sha(sd/name) for name in names}),release_sha256=release))
    caption=(out/f'S{n:02d}_caption_en.md').read_text(encoding='utf-8');checks['caption_English']=not re.search('[\u4e00-\u9fff]',caption);checks['three_review_rounds_recorded']=(assets/'visual_qa.md').is_file() and all((assets/f'S{n:02d}_round{r}.pdf').is_file() for r in [1,2,3])
    manifest=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.zip','.log'] or p.name in [f'S{n:02d}_validation.json',f'S{n:02d}_delivery.sha256']:continue
        manifest.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(passed=bool(all(checks.values())),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=release,checks={k:bool(v) for k,v in checks.items()},details=details,sha256_manifest=manifest)
    (out/f'S{n:02d}_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(f'S{n:02d}',result['passed'],[k for k,v in checks.items() if not v],flush=True);return result['passed']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figure',type=int);a=parser.parse_args();parent=Path(os.path.abspath(__file__)).parents[1].name;default=int(parent[1:]) if re.fullmatch('S0[1-5]',parent) else None;selected=[a.figure or default] if a.figure or default else range(1,6);status=[validate(n) for n in selected];sys.exit(0 if all(status) else 1)
