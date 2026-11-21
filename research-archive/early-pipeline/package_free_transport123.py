from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
H=Path(__file__).parent;C=H.parent/'shuolee-coordination'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
root=C/'handoffs/free-clothing-transport-123-20261029'
root.mkdir(exist_ok=False)
files=[(P/'src/shuolee_portrait/free_clothing_transport.py','src/shuolee_portrait/free_clothing_transport.py'),
       (P/'tests/test_free_clothing_transport.py','tests/test_free_clothing_transport.py'),
       (H/'free-clothing-transport123-handoff.md','docs/自由衣物高斯逐帧运输.md')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=dict(created=datetime.now(timezone.utc).isoformat(),base_commit='8d54497c0395a0a8a8f378a33d4895f5ffcd56be',
    task='00 deploy reviewed CPU transport on an independent branch, run bounded CPU tests and CI, report actual commit and evidence back.',
    files=[],new_large_files=False,real_input_included=False,training_admitted=False,
    source_policy='Only generic module, synthetic tests and deployment doc may be committed. No private evidence/archive paths or original person files in public Git.',
    scope='18 CPU controls; Torch CPU transport. GPU research bridge incomplete after two retained environment attempts. No server or real training claim.',
    relative_reconstruction='A complete per-frame transport may cancel common camera scale and even E; unknown physical metric alone is not a rejection of all relative reconstruction.',
    no_training_or_GPU_authorized_in_this_increment=True)
for source,name in files:
    target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest['files'].append(dict(path=name,sha256=sha(target),bytes=target.stat().st_size))
archive=root/'free-clothing-transport-123-source.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for item in manifest['files']:z.write(root/'source'/item['path'],item['path'])
manifest.update(archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
save(root/'manifest.json',manifest)
p=C/'local-research-status.json';data=json.loads(p.read_text(encoding='utf-8-sig'))
data['free_clothing_transport_handoff']=dict(updated=manifest['created'],path=str(root),archive_sha256=manifest['archive_sha256'],state='prepared_not_sent',target_thread_id='01a17c8c-e70f-7822-ad39-440290b3d1da',training_admitted=False)
save(p,data)
print(json.dumps(dict(handoff=str(root),SHA256=manifest['archive_sha256'],bytes=manifest['archive_bytes']),ensure_ascii=False))
