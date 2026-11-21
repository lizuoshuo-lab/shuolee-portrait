"""Actual four Creator camera-byte edge check; no model decoding or GPU."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;ROOT=D.parents[3]
BASE=ROOT/'outputs/large-scene-gs-001';LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
OUT=BASE/'independent-evaluation/small-city006-posebytes-v2-actual-archives-CPU-001'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
source=D/'audit_city_hierarchy_lineage_006_posebytes_v2.py';old=D/'audit_city_hierarchy_lineage_006_posebytes.py'
assert sha(source)=='aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0'
assert sha(old)=='0148a71ae3ccf9678a6824e06f83cf1cc73c909db3001695ed09a509736ed1a2'
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(source)
assert a.keys()==b.keys()
changed=[k for k in a if a[k]!=b[k]];assert changed==['leaf_changes']
spec=importlib.util.spec_from_file_location('actual_archives_v2',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
config=read(LOCK/'configuration.json');names=read(BASE/'official-inputs/small-city-training-002/manifest.json')
protected={str(p):sha(p) for p in [old,source,D/'audit_city_hierarchy_lineage_006.py',LOCK/'configuration.json',BASE/'official-inputs/small-city-training-002/manifest.json']}
records=[]
for chunk in ['0_0','0_1','1_0','1_1']:
    archive=LOCK/'creator-camera-archives'/chunk
    receiptpath=LOCK/'trained_chunks'/chunk/'creator-outcome.json';receipt=read(receiptpath);ar=read(archive/'manifest.json')
    assert receipt['status']=='completed' and receipt['exit_code']==receipt['actual_child_poll_exit_code']==0
    assert ar==receipt['creator_train_camera_archive']
    source_sparse=BASE/'official-inputs'/config['input']/'chunks'/chunk/'sparse/0'
    source_poses=m.pose_blocks(source_sparse/'images.bin');archive_poses=m.pose_blocks(archive/'sparse/0/images.bin')
    split=next(c for c in names['chunks'] if c['name']==chunk)
    assert set(archive_poses)==set(split['train_names'])
    assert not set(archive_poses)&set(names['test_names'])
    assert all(raw==source_poses[name][0] and nobs==0 for name,(raw,nobs) in archive_poses.items())
    assert (source_sparse/'cameras.bin').read_bytes()==(archive/'sparse/0/cameras.bin').read_bytes()
    assert sha(source_sparse/'images.bin')==ar['source_images_sha256'] and sha(source_sparse/'cameras.bin')==ar['source_cameras_sha256']
    for row in ar['files']:assert sha(row['path'])==row['sha256']
    for p in [receiptpath,archive/'manifest.json',source_sparse/'images.bin',source_sparse/'cameras.bin',archive/'sparse/0/images.bin',archive/'sparse/0/cameras.bin']:protected[str(p)]=sha(p)
    records.append({'chunk':chunk,'actual_source_pose_count':len(source_poses),'actual_archive_pose_count':len(archive_poses),'actual_train_count':len(split['train_names']),'actual_global_test_overlap':0,'raw64_pose_blocks_exact':True,'archive_observation_count_all_zero':True,'camera_bytes_exact':True,'archive_manifest_equals_actual_creator_receipt':True,'creator_receipt_sha256':sha(receiptpath),'archive_manifest_sha256':sha(archive/'manifest.json')})
assert all(sha(p)==d for p,d in protected.items())
report={'status':'passed_actual_four_Creator_train_archive_names_pose64_K_bytes_CPU_check_only','records':records,'changed_AST_functions_relative_0148':changed,'protected_sources_sha256':protected,'sources_unchanged':True,'CUDA_called':False,'Torch_imported':'torch' in sys.modules,'actual_formal_posts_complete':False,'actual_1404_complete':False}
assert not report['Torch_imported']
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report_path':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json')}))
