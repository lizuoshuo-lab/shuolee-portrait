from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, tarfile

lab = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
stage = Path(r'C:\Users\Administrator\Downloads\04_mini_splatting2_work')
stage.mkdir(exist_ok=True)
out = lab / 'outputs/mini-splatting2-computer00-001'
shutil.copy2(Path(r'C:\Users\Administrator\Downloads\03_mini_splatting2_auth\server-environment-memory.txt'), out / 'server-environment.txt')
repo = lab / 'repos/mini-splatting2'
dataset = out / 'hover-sfm-001/dataset'
archive = stage / 'payload.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(repo.rglob('*')):
        rel = p.relative_to(repo)
        if p.is_file() and not any(x in {'.git', '__pycache__', 'assets'} for x in rel.parts):
            tar.add(p, arcname='repo/' + rel.as_posix(), recursive=False)
    for sub in ['images', 'sparse/0']:
        for p in sorted((dataset/sub).rglob('*')):
            if p.is_file():
                tar.add(p, arcname='dataset/' + p.relative_to(dataset).as_posix(), recursive=False)
    tar.add(out/'hover-sfm-001/state.json', arcname='provenance/local-sfm.json')
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(stage/'payload.sha256').write_text(digest+'  payload.tar.gz\n', encoding='utf-8')
manifest_path=lab/'manifests/mini-splatting2-computer00.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
manifest.update({
    'connection_observation':'Authenticated SSH through computer00 succeeded; read-only Linux inspection exited 0. Authentication input is held in process memory only.',
    'remote_environment_modified': True,
    'linux_ssh_authenticated': True,
    'remote_modification_scope':'Inspection artifacts on Windows and SSH known-host entry; no Linux package installs or training yet.',
    'feedback_destination':'QQ contact 小润, explicitly requested by user; send milestone and blocker updates.',
    'last_recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'server':{'hostname':'f6f30e205403','os':'Ubuntu 22.04.3 LTS','gpu':'NVIDIA RTX 3090','vram_mib':24576,'driver':'570.133.20','python':'3.10.12','torch':'2.1.0+cu121','torch_cuda':'12.1','cuda_available':True,'nvcc':'not on default PATH; inspecting toolkits','home_available_gib_approx':636,'report':str(out/'server-environment.txt')},
    'local_sfm_result':{'registered':59,'sparse_points':19662,'mean_reprojection_px':0.6153988717555579,'train_images':51,'test_images':8,'all_images_used_for_sfm':True},
    'remote_payload':{'path':str(archive),'sha256':digest,'bytes':archive.stat().st_size},
})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps({'stage':str(stage),'bytes':archive.stat().st_size,'sha256':digest}))
