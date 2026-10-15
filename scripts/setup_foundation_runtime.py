"""Install the fixed foundation-model runtime in the project-local Python 3.11 venv."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/foundation_setup'
PYTHON=ROOT/'.venv-foundation/Scripts/python.exe'


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    (OUT/'status.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)


def main(check_only=False):
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'setup.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        marker=ROOT/'vendor/wheels/torch_wheel_verification.json';start=time.monotonic()
        while not marker.exists():
            if time.monotonic()-start>3600:raise TimeoutError('Verified PyTorch wheel not ready')
            status('waiting_for_verified_torch_wheel');time.sleep(20)
        proof=json.loads(marker.read_text());expected=json.loads((ROOT/'configs/torch_foundation_wheel.json').read_text())
        if proof.get('status')!='verified' or proof.get('actual_sha256')!=expected['sha256']:raise ValueError('Unverified wheel')
        wheel=ROOT/'vendor/wheels/torch-2.7.1+cu128-cp311-cp311-win_amd64.whl'
        if wheel.stat().st_size!=expected['expected_bytes']:raise ValueError('Wheel size changed')
        env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
            HF_HOME=str(ROOT/'models/pretrained/hf_cache'),TORCH_HOME=str(ROOT/'models/pretrained/torch_cache'))
        installs=[('installing_pytorch',[str(wheel)]),
                  ('installing_torchvision',['torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu128']),
                  ('installing_terramind_runtime',['terratorch==1.2.6','numpy==2.2.6','-c',str(ROOT/'configs/foundation_constraints.txt')])]
        for stage,packages in ([] if check_only else installs):
            status(stage)
            with (OUT/(stage+'.log')).open('a',encoding='utf-8') as stream:
                subprocess.run([str(PYTHON),'-s','-m','pip','install','--no-cache-dir',*packages],cwd=ROOT,env=env,
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
        status('checking_imports')
        code="from experiments.foundation_environment import configure; configure(); import torch,terratorch,numpy,json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'numpy':numpy.__version__,'terratorch':m.version('terratorch')}))"
        result=subprocess.run([str(PYTHON),'-s','-c',code],cwd=ROOT,env=env,text=True,capture_output=True)
        (OUT/'imports_stdout.log').write_text(result.stdout);(OUT/'imports_stderr.log').write_text(result.stderr)
        result.check_returncode()
        freeze=subprocess.run([str(PYTHON),'-s','-m','pip','freeze'],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
        (ROOT/'configs/environment_foundation_freeze.txt').write_text(freeze.stdout)
        metadata=json.loads(result.stdout.strip().splitlines()[-1]);(ROOT/'reports/environment_foundation.json').write_text(json.dumps(metadata,indent=2))
        status('completed',environment=metadata)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    main(parser.parse_args().check_only)
