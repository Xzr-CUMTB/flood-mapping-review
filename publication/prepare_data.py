"""Verify transport archives and extract an independent reproduction dataset.

No credentials or network access are used. Archives may be obtained locally or
downloaded from the future public deposit using its actual file links.
"""
import argparse,csv,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def prepare(archives,manifest,output):
 archives=Path(archives);output=Path(output).resolve()
 if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty; existing research files are never overwritten')
 rows=list(csv.DictReader(Path(manifest).open(encoding='utf-8-sig',newline='')))
 if not rows:raise ValueError('Empty transport manifest')
 verified=[];names=set()
 for row in rows:
  name=row['filename']
  if Path(name).name!=name or name in names:raise ValueError('Unsafe or duplicate archive filename')
  names.add(name);path=archives/name
  if path.stat().st_size!=int(row['bytes']) or sha(path)!=row['sha256']:raise ValueError('Archive hash/size mismatch: '+name)
  verified.append(path)
 output.mkdir(parents=True,exist_ok=True);members=[]
 for path in verified:
  with tarfile.open(path,'r:gz') as tar:
   for member in tar:
    parts=PurePosixPath(member.name)
    if parts.is_absolute() or '..' in parts.parts or '\\' in member.name or ':' in member.name or not member.isfile():raise ValueError('Unsafe archive member: '+member.name)
    target=output.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(output) or target.exists():raise ValueError('Repeated or unsafe extraction target: '+member.name)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination,8*1024*1024)
    if target.stat().st_size!=member.size:raise ValueError('Truncated member: '+member.name)
    members.append(member.name)
 from publication.verify_payload import verify
 report=verify(output,output.parent/(output.name+'_verification'))
 result=dict(passed=report['passed'],archives=len(verified),members=len(members),source='Local transport archives; not a test of a Harvard Dataverse download')
 print(json.dumps(result),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archives',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();prepare(a.archives,a.manifest,a.output)
