"""Load tensor-only archived states with strict architecture and file-hash checks."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def load_model(data,run_id,device='cpu'):
 import torch
 from models.flood_unet import FloodUNet
 from models.review_heads import SpatialHead
 data=Path(data);records=json.loads((data/'model_states/MODEL_STATE_MANIFEST.json').read_text())
 matches=[r for r in records if r['run_id']==run_id]
 if len(matches)!=1:raise ValueError('Unknown or duplicate archived run ID')
 record=matches[0];path=data/'model_states'/record['file']
 if sha(path)!=record['sha256']:raise ValueError('Published model-state file changed')
 config=json.loads((data/'training_records'/run_id/'config.json').read_text())
 if 'terramind' in run_id:
  from models.terramind_flood import TerraMindFlood
  model=TerraMindFlood(pretrained=False)
 elif config.get('kind') in ['value','ordinary_gate','stacker','risk'] or run_id.endswith('fusion_risk'):
  model=SpatialHead(16 if run_id.endswith(('_risk','fusion_risk')) else 13)
 else:model=FloodUNet(config['modality'],config['width'])
 state=torch.load(path,map_location='cpu',weights_only=True)
 if set(state)!= {'model'}:raise ValueError('Expected model-only state, without optimizer')
 model.load_state_dict(state['model'],strict=True)
 return model.to(device).eval()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
 p.add_argument('--run-id');p.add_argument('--family',choices=['main','foundation'],default='main');p.add_argument('--output',type=Path)
 args=p.parse_args();records=json.loads((args.data/'model_states/MODEL_STATE_MANIFEST.json').read_text())
 names=[args.run_id] if args.run_id else [r['run_id'] for r in records if ('terramind' in r['run_id'])==(args.family=='foundation')]
 results=[]
 for name in names:
  model=load_model(args.data,name);results.append(dict(run_id=name,strict_load_passed=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
  del model
 report=dict(passed=True,models=results,scope='CPU architecture/state loading; no predictions, training or empirical data generated')
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
