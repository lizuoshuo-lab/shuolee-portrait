"""Prepare/run fixed official Creator -> train_post -> Merger, one GPU stage at a time.

Run through run_local.py. plan is CPU-only and accepts future chunk run names.
Other stages must only run under the parent task's GPU authorization/scheduling.
This wrapper preserves official algorithms, keeps independent stage artifacts,
and never turns an unoptimized hierarchy into an apparent hier_opt result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np

from read_hierarchy import read_hierarchy, read_anchors

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'outputs/large-scene-gs-001'
REPO = ROOT/'repos/hierarchical-3d-gaussians'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'
DRIVER_VERSION = 4
BUILD = OUTPUT/'hierarchy-build'
EXES = ROOT/'repos/gaussian-hierarchy-large-gs/build-local/Release'
TARGET = ['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def identifier(value):
    if not value.replace('-','').isalnum():
        raise ValueError(f'Invalid run identifier: {value}')
    return value

def source_audit():
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip() != COMMIT:
        raise RuntimeError('Official repository HEAD changed')
    records=[]
    for name in ['train_post.py','scene/__init__.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','arguments/__init__.py','scripts/full_train.py']:
        path=REPO/name
        canonical=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMIT}:{name}'])
        if path.read_bytes().replace(b'\r\n',b'\n') != canonical.replace(b'\r\n',b'\n'):
            raise RuntimeError(f'Official runtime source changed: {name}')
        records.append(dict(path=str(path),sha256=digest(path)))
    native=read(BUILD/'build-manifest.json')
    for artifact in native['artifacts']:
        if digest(artifact['path']) != artifact['sha256']:
            raise RuntimeError('Official native executable identity changed')
    if native['hierarchy_repository']['commit'] != '677c8553dc64dfd62c272eca94a291a277733113':
        raise RuntimeError('Official hierarchy acquisition commit changed')
    variant=read(BUILD/'merger-unoptimized/manifest.json')
    for path,key in [(variant['source_path'],'official_source_sha256'),(variant['derived_source_path'],'derived_source_sha256'),(variant['patch_path'],'patch_sha256')]:
        if digest(path)!=variant[key]:
            raise RuntimeError('Unoptimized input-filename interface source identity changed')
    extensions=read(BUILD/'python-extensions-manifest.json')
    for module in extensions['smoke']['modules']:
        if digest(module['path'])!=module['sha256']:
            raise RuntimeError('Official Python extension identity changed')
    for name in ['hierarchy','rasterizer','simple_knn']:
        archive_info=extensions['source_audit'][name]
        archive=Path(archive_info['archive'])
        if digest(archive)!=archive_info['archive_sha256']:
            raise RuntimeError(f'Official source archive changed: {name}')
        permitted={item['path']:item['current_sha256'] for item in archive_info['changes']}
        with zipfile.ZipFile(archive) as original:
            for entry in original.infolist():
                if entry.is_dir():
                    continue
                relative=entry.filename.split('/',1)[1]
                current=Path(archive_info['source_path'])/relative
                actual=digest(current)
                expected=permitted.get(relative,hashlib.sha256(original.read(entry)).hexdigest())
                if actual!=expected:
                    raise RuntimeError(f'Official compiled runtime source changed: {name}/{relative}')
                records.append(dict(path=str(current),sha256=actual))
    return records

def completed_training(run_name, stage, chunk=None):
    path=OUTPUT/'official-training'/identifier(run_name)
    outcome=read(path/'outcome.json')
    config=read(path/'configuration.json')
    if outcome['status'] != 'completed' or config['stage'] != stage or config.get('chunk') != chunk:
        raise RuntimeError(f'Training dependency incomplete/mismatched: {run_name}')
    saved=Path(outcome['saved_model_directory'])
    if digest(saved/'point_cloud.ply') != outcome['point_cloud_sha256']:
        raise RuntimeError('Saved training PLY identity mismatch')
    return saved,outcome,config

def creator_ply_contract(path, skybox):
    with path.open('rb') as f:
        if f.readline()!=b'ply\n' or f.readline()!=b'format binary_little_endian 1.0\n':
            raise RuntimeError('Creator requires LF-only binary little-endian PLY')
        parts=f.readline().decode('ascii').split()
        if parts[:2] != ['element','vertex'] or len(parts)!=3:
            raise RuntimeError('Creator requires vertex count on third line')
        count=int(parts[2])
        properties=[]
        for _ in range(200):
            line=f.readline()
            if line==b'end_header\n':
                break
            p=line.decode('ascii').split()
            if len(p)!=3 or p[:2]!=['property','float']:
                raise RuntimeError('Creator requires only scalar float vertex properties')
            properties.append(p[2])
        else:
            raise RuntimeError('Missing PLY end_header')
        offset=f.tell()
    if properties!=TARGET or path.stat().st_size-offset != count*248 or not 0<=skybox<count:
        raise RuntimeError('Canonical SH3 Creator PLY layout/count mismatch')
    array=np.memmap(path,dtype='<f4',mode='r',offset=offset,shape=(count,62))
    for begin in range(0,count,65536):
        if not np.isfinite(array[begin:begin+65536]).all():
            raise RuntimeError('Nonfinite PLY values')
    return dict(vertices=count,skybox_prefix_skipped=skybox,record_bytes=248,sha256=digest(path))

def pose_archive(source, target, train_names, test_names):
    """Keep original PINHOLE intrinsics/W2C; remove all heldout poses for filtering."""
    sys.path.insert(0,str(REPO/'preprocess'))
    import read_write_model as rw
    cameras=rw.read_cameras_binary(str(source/'sparse/0/cameras.bin'))
    images=rw.read_images_binary(str(source/'sparse/0/images.bin'))
    selected={i:v for i,v in images.items() if v.name in set(train_names)}
    if {v.name for v in selected.values()} != set(train_names) or set(train_names)&set(test_names):
        raise RuntimeError('Creator camera train/test identity mismatch')
    used={v.camera_id for v in selected.values()}
    cameras={i:v for i,v in cameras.items() if i in used}
    if any(v.model!='PINHOLE' for v in cameras.values()):
        raise RuntimeError('Creator AppearanceFilter requires four-parameter PINHOLE')
    poses={i:rw.Image(v.id,v.qvec,v.tvec,v.camera_id,v.name,np.empty((0,2)),np.empty(0,dtype=np.int64)) for i,v in selected.items()}
    sparse=target/'sparse/0'
    sparse.mkdir(parents=True,exist_ok=False)
    rw.write_model(cameras,poses,{},str(sparse),ext='.bin')
    rw.write_model(cameras,poses,{},str(sparse),ext='.txt')
    return dict(train_only_pose_count=len(poses),heldout_pose_count=0,source_images_sha256=digest(source/'sparse/0/images.bin'),
                archive_images_sha256=digest(sparse/'images.bin'),scope='Pose-only Creator filter archive; empty observations and points; not a training SfM reconstruction.')

def scaffold_prefix_audit(scaffold, chunk_input, chunk_ply=None):
    """Exact create_from_pcd ring predicate; shared sky prefix must remain frozen."""
    from plyfile import PlyData
    data=PlyData.read(scaffold/'point_cloud.ply')['vertex'].data
    skybox=int((scaffold/'pc_info.txt').read_text())
    center=np.loadtxt(chunk_input/'center.txt',dtype=np.float32)
    extent=np.loadtxt(chunk_input/'extent.txt',dtype=np.float32)
    xy=np.stack([data['x'],data['y']],axis=1)
    distance=np.max(np.abs(xy-center[:2]),axis=1)
    selected=(distance > np.float32(.5)*extent[0]) & (distance < np.float32(1.5)*extent[0])
    selected[:skybox]=True
    result=dict(skybox_prefix=skybox,full_scaffold_prefix=int(selected.sum()),non_sky_scaffold_ring=int(selected.sum())-skybox,
        predicate='max(abs(x-center.x),abs(y-center.y)) > .5*extent.x and <1.5*extent.x, plus all skybox prefix; strict comparisons and float32.',
        note='Only skybox prefix is frozen by skybox_locked; whole scaffold prefix is excluded from densification, and non-sky scaffold parameters can train.')
    if chunk_ply is not None:
        block=PlyData.read(chunk_ply)['vertex'].data
        for field in ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']:
            if not np.array_equal(data[field][:skybox],block[field][:skybox]):
                raise RuntimeError(f'Locked shared skybox changed in chunk PLY: {field}')
        for channel in range(3):
            for basis in range(15):
                actual=block[f'f_rest_{channel*15+basis}'][:skybox]
                expected=data[f'f_rest_{channel*3+basis}'][:skybox] if basis<3 else np.zeros(skybox,dtype=np.float32)
                if not np.array_equal(actual,expected):
                    raise RuntimeError('Chunk shared skybox SH1-to-SH3 padding mismatch')
        result['chunk_skybox_all_raw_fields_exact']=True
    return result

def hierarchy_numeric_audit(h):
    for field in ['positions','rotations','log_scales','opacities','sh']:
        array=getattr(h,field)
        for begin in range(0,len(array),65536):
            if not np.isfinite(array[begin:begin+65536]).all():
                raise RuntimeError(f'Nonfinite hierarchy Gaussian field: {field}')
    return dict(all_Gaussian_fields_finite=True,storage_Gaussians=len(h.positions),unowned_tail=h.unowned_tail)

def post_freeze_audit(before, after, anchors, scaffold, skybox):
    """Compare native-decoded anchors and one SH1 scaffold sky tail in float32.

    Native Creator half storage is expanded to float32 by the loader; post stores
    float32. Compare expanded values, not file bytes or a re-quantized hierarchy.
    The pinned writer defaults to compressed=True even for .hier_opt, quantizing
    appended coarse float32 fields to half (positions remain float32). Sky alpha
    is sigmoid(Ply logit); two float32 ULPs are allowed before storage conversion.
    Frozen fields must match the expected storage roundtrip, not raw PLY bytes.
    """
    from plyfile import PlyData
    if len(after.positions)!=len(before.positions)+skybox or not np.array_equal(before.nodes,after.nodes):
        raise RuntimeError('Official post tree/skybox-tail identity mismatch')
    if not np.array_equal(np.asarray(before.boxes,dtype=np.float32),after.boxes):
        raise RuntimeError('Official post changed stored hierarchy boxes')
    anchor_fields={}
    for field in ['positions','rotations','log_scales','opacities','sh']:
        reference=np.asarray(getattr(before,field)[anchors],dtype=np.float32)
        actual=np.asarray(getattr(after,field)[anchors],dtype=np.float32)
        equal=bool(np.array_equal(reference,actual))
        anchor_fields[field]=dict(exact=equal,max_abs_delta=float(np.max(np.abs(actual-reference))) if reference.size else 0.0)
        if not equal:
            raise RuntimeError(f'Official post changed frozen anchor field: {field}')
    data=PlyData.read(scaffold/'point_cloud.ply')['vertex'].data[:skybox]
    tail=slice(len(before.positions),len(after.positions))
    expected={
        'positions':np.stack([data[k] for k in ['x','y','z']],axis=1),
        'rotations':np.stack([data[f'rot_{i}'] for i in range(4)],axis=1),
        'log_scales':np.stack([data[f'scale_{i}'] for i in range(3)],axis=1),
        'sh':np.zeros((skybox,16,3),dtype=np.float32)}
    expected['sh'][:,0,:]=np.stack([data[f'f_dc_{i}'] for i in range(3)],axis=1)
    for channel in range(3):
        for basis in range(3):
            expected['sh'][:,basis+1,channel]=data[f'f_rest_{channel*3+basis}']
    logits=np.asarray(data['opacity'],dtype=np.float64)
    expected['opacities']=np.asarray(1/(1+np.exp(-logits)),dtype=np.float32)
    sky_fields={}
    for field,reference in expected.items():
        actual=np.asarray(getattr(after,field)[tail],dtype=np.float32)
        stored_dtype=np.float32 if field=='positions' or not after.compressed else np.float16
        stored_reference=np.asarray(reference,dtype=stored_dtype).astype(np.float32)
        delta=np.abs(actual-stored_reference)
        if field=='opacities':
            tolerance=2*np.abs(np.spacing(reference))
            if after.compressed:
                low=np.asarray(reference-tolerance,dtype=stored_dtype).astype(np.float32)
                high=np.asarray(reference+tolerance,dtype=stored_dtype).astype(np.float32)
                matches=bool(np.all((actual==stored_reference)|(actual==low)|(actual==high)))
            else:
                matches=bool(np.all(delta<=tolerance))
        else:
            matches=bool(np.array_equal(actual,stored_reference))
        sky_fields[field]=dict(exact=bool(np.array_equal(actual,stored_reference)),within_allowed_tolerance=matches,
            max_abs_delta=float(np.max(delta)) if delta.size else 0.0,
            max_abs_delta_from_source_PLY_float32=float(np.max(np.abs(actual-reference))) if delta.size else 0.0,
            stored_dtype=np.dtype(stored_dtype).name,
            tolerance='Two float32 ULPs for CPU/CUDA sigmoid before expected storage quantization' if field=='opacities' else 'exact expected storage roundtrip equality')
        if not matches:
            raise RuntimeError(f'Official post changed shared frozen skybox field: {field}')
    return dict(status='frozen_anchor_and_skybox_fields_verified',anchor_count=len(anchors),anchor_fields=anchor_fields,
        skybox_tail=skybox,skybox_fields=sky_fields,post_storage_compressed=after.compressed,nodes_exact=True,stored_boxes_exact_after_half_expansion=True,
        scope='Raw frozen fields and original stored topology; no quality improvement claim.')

def execute(command, cwd, directory, result_path):
    if result_path.exists():
        raise RuntimeError(f'Stage receipt exists; choose a new run: {result_path}')
    started=time.perf_counter()
    receipt=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command_argv=command,cwd=str(cwd),log=str(directory/'stage.log'))
    write(result_path,receipt)
    print(json.dumps(receipt),flush=True)
    with (directory/'stage.log').open('wb') as log:
        code=subprocess.call(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    receipt.update(exit_code=code,elapsed_seconds=time.perf_counter()-started,status='process_exited_zero' if code==0 else 'failed')
    write(result_path,receipt)
    if code:
        raise SystemExit(code)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','creator','post','audit-post','merge-unoptimized','merge'],required=True)
    p.add_argument('--run',required=True,type=identifier)
    p.add_argument('--input',default='truck-four-chunks-001',type=identifier)
    p.add_argument('--scaffold-run',default='truck-coarse-001',type=identifier)
    p.add_argument('--chunk-run',action='append',required=True,help='Repeat CHUNK=RUN for each manifest chunk')
    p.add_argument('--chunk',help='One chunk for creator/post; merge uses all configured chunks')
    p.add_argument('--post-iterations',type=int,default=15000)
    p.add_argument('--audit-only-config-update',action='store_true',help='CPU plan only: preserve prior configuration and revise only wrapper identity after an audit correction')
    a=p.parse_args()
    if a.post_iterations<2:
        raise ValueError('At least 2 post iterations required')
    mappings={}
    for pair in a.chunk_run:
        chunk,run=pair.split('=',1)
        if chunk in mappings:
            raise ValueError('Duplicate chunk mapping')
        mappings[chunk]=identifier(run)
    inputs=OUTPUT/'official-inputs'/a.input
    manifest=read(inputs/'manifest.json')
    chunks={c['name']:c for c in manifest['chunks']}
    if set(mappings)!=set(chunks) or manifest['status']!='ready_for_official_scaffold_and_chunk_training':
        raise RuntimeError('Exactly one training run mapping required for each ready input chunk')
    if a.stage in ['creator','post','audit-post'] and a.chunk not in chunks:
        raise ValueError('Select one known chunk')
    root=OUTPUT/'official-hierarchy'/a.run
    variant=read(BUILD/'merger-unoptimized/manifest.json')
    if digest(variant['executable']['path'])!=variant['executable']['sha256']:
        raise RuntimeError('Unoptimized filename-only Merger variant identity changed')
    configuration=dict(driver_version=DRIVER_VERSION,driver_sha256=digest(Path(__file__)),
        main_commit=COMMIT,hierarchy_commit='677c8553dc64dfd62c272eca94a291a277733113',
        input=a.input,input_manifest_sha256=digest(inputs/'manifest.json'),scaffold_run=a.scaffold_run,chunk_runs=mappings,
        post_iterations=a.post_iterations,expected_post_optimizer_calls=a.post_iterations-1,
        coordinate_frame=manifest['coordinate_frame'],baseline_geometry_sha256=manifest['baseline_geometry_sha256'],
        source_records=source_audit(),protocol='Fixed heldout identities; train-only Creator filtering; unchanged official post renderer and CLI; shared global frame; depth/exposure disabled and explicit skybox locking.',
        unoptimized_merger_variant_manifest_sha256=digest(BUILD/'merger-unoptimized/manifest.json'),
        consolidation='Official nearest-center 5% opacity falloff; not alpha-conserving deduplication.',
        final_background='Merged hierarchy excludes each post skybox tail; append one shared coarse skybox at rendering time.')
    config_path=root/'configuration.json'
    if config_path.exists():
        if read(config_path)!=configuration:
            existing=read(config_path)
            allowed_revision=a.stage=='plan' and a.audit_only_config_update
            if allowed_revision:
                previous_stable={k:v for k,v in existing.items() if k not in ['driver_version','driver_sha256']}
                current_stable={k:v for k,v in configuration.items() if k not in ['driver_version','driver_sha256']}
                if previous_stable!=current_stable:
                    raise RuntimeError('Audit-only revision cannot change any protocol/source/training identity')
            if a.stage!='plan' or (list(root.rglob('*-outcome.json')) and not allowed_revision):
                raise RuntimeError('Existing run configuration differs; only an unexecuted CPU plan may be revised')
            previous_sha=digest(config_path)
            backup=root/f'configuration-plan-before-v{DRIVER_VERSION}-{previous_sha[:12]}.json'
            if not backup.exists():
                shutil.copy2(config_path,backup)
            if (root/'plan.json').exists():
                shutil.copy2(root/'plan.json',root/f'plan-before-v{DRIVER_VERSION}-{previous_sha[:12]}.json')
            write(config_path,configuration)
            write(root/f'plan-update-v{DRIVER_VERSION}-{previous_sha[:12]}.json',dict(status='CPU_audit_only_wrapper_revision_preserving_prior_stage_receipts' if allowed_revision else 'CPU_plan_revised_before_any_stage_execution',
                previous_configuration_sha256=previous_sha,current_configuration_sha256=digest(config_path),
                previous_configuration=str(backup),driver_version=DRIVER_VERSION,driver_sha256=configuration['driver_sha256']))
    else:
        root.mkdir(parents=True,exist_ok=False)
        write(config_path,configuration)
    if a.stage=='plan':
        plan=dict(status='prepared_only_no_GPU_execution',configuration=str(config_path),
            dependencies={name:read(OUTPUT/'official-training'/run/'outcome.json')['status'] if (OUTPUT/'official-training'/run/'outcome.json').exists() else 'not_created' for name,run in mappings.items()},
            stages=['creator CHUNK after completed official single training','post CHUNK after verified Creator hierarchy and independent preflight',
                    'merge-unoptimized from every real Creator hierarchy.hier, explicit source_kind unoptimized','merge from every real post hierarchy.hier_opt, explicit source_kind post_optimized'],
            paired_merge=dict(unoptimized='merged-unoptimized.hier / merge-unoptimized-outcome.json',post_optimized='merged.hier / merge-outcome.json',
                ownership='Same fixed chunk centers/order/frame and official 5% falloff; filename-only variant selects real unoptimized .hier without fake .hier_opt'),
            post_iterations=a.post_iterations,expected_post_optimizer_calls=a.post_iterations-1,
            post_full_train_defaults=dict(feature_lr=.0005,opacity_lr=.01,scaling_lr=.001,limit_log_uniform=[.005,.1]),
            training_boundary='Final requested iteration does forward/backward and saves before return; no final optimizer update.',
            estimated_post_skybox_tail='Exact count read later from scaffold pc_info.txt; never inferred from hardcoded prefix.',
            post_frozen_field_audit='Exact decoded anchor position/rotation/log-scale/alpha/SH; one coarse skybox tail with exact expected writer storage roundtrip and two-float32-ULP sigmoid allowance before half quantization; nodes and original stored boxes unchanged.',
            global_test_names=manifest['test_names'])
        write(root/'plan.json',plan)
        print(json.dumps(plan))
        return
    scaffold,coarse_outcome,coarse_config=completed_training(a.scaffold_run,'coarse')
    skybox=int((scaffold/'pc_info.txt').read_text())
    if coarse_config['input_manifest_sha256']!=configuration['input_manifest_sha256']:
        raise RuntimeError('Scaffold coordinate/input identity mismatch')
    directory=root/'trained_chunks'/str(a.chunk)
    if a.stage=='creator':
        saved,training_outcome,training_config=completed_training(mappings[a.chunk],'chunk',a.chunk)
        if training_config['input_manifest_sha256']!=configuration['input_manifest_sha256']:
            raise RuntimeError('Chunk coordinate/input identity mismatch')
        contract=creator_ply_contract(saved/'point_cloud.ply',skybox)
        scaffold_prefix=scaffold_prefix_audit(scaffold,inputs/'chunks'/a.chunk,saved/'point_cloud.ply')
        if int((saved/'pc_info.txt').read_text())!=skybox:
            raise RuntimeError('Chunk/scaffold skybox prefix counts differ')
        directory.mkdir(parents=True,exist_ok=False)
        archive=pose_archive(inputs/'chunks'/a.chunk,directory/'creator-train-pose-archive',chunks[a.chunk]['train_names'],manifest['test_names'])
        exposure=OUTPUT/'official-training'/mappings[a.chunk]/'model/exposure.json'
        shutil.copy2(exposure,directory/'exposure.json')
        exposures=read(exposure)
        if any(not np.array_equal(np.asarray(v),np.eye(3,4)) for v in exposures.values()):
            raise RuntimeError('Expected frozen identity exposure matrices')
        command=[str(EXES/'GaussianHierarchyCreator.exe'),str(saved/'point_cloud.ply'),str(directory/'creator-train-pose-archive'),str(directory),str(scaffold)]
        receipt=execute(command,REPO,directory,directory/'creator-outcome.json')
        h=read_hierarchy(directory/'hierarchy.hier')
        anchors=read_anchors(directory/'anchors.bin',len(h.positions))
        receipt.update(status='completed',hierarchy=h.summary(),hierarchy_sha256=digest(h.path),anchor_count=len(anchors),anchor_sha256=digest(directory/'anchors.bin'),
            input_PLY=contract,creator_pose_archive=archive,scaffold_prefix=scaffold_prefix,source_training_outcome=training_outcome,scaffold_PLY_sha256=coarse_outcome['point_cloud_sha256'],
            numeric_audit=hierarchy_numeric_audit(h),source_kind='unoptimized',configuration_sha256=digest(config_path))
        write(directory/'creator-outcome.json',receipt)
    elif a.stage in ['post','audit-post']:
        creator=read(directory/'creator-outcome.json')
        if creator['status']!='completed' or digest(directory/'hierarchy.hier')!=creator['hierarchy_sha256'] or (a.stage=='post' and (directory/'hierarchy.hier_opt').exists()):
            raise RuntimeError('Creator dependency incomplete, changed, or already optimized')
        before=read_hierarchy(directory/'hierarchy.hier')
        if digest(directory/'anchors.bin')!=creator['anchor_sha256']:
            raise RuntimeError('Creator anchors identity changed')
        anchors=read_anchors(directory/'anchors.bin',len(before.positions))
        if len(anchors)!=creator['anchor_count']:
            raise RuntimeError('Creator anchor count changed')
        stage_dir=directory/'post-stage'
        if a.stage=='post':
            stage_dir.mkdir(exist_ok=False)
            write(stage_dir/'post-input-audit.json',dict(status='CPU_inputs_verified_before_GPU_post',hierarchy=before.summary(),
            hierarchy_sha256=creator['hierarchy_sha256'],anchors_sha256=creator['anchor_sha256'],anchor_count=len(anchors),
            scaffold_PLY_sha256=coarse_outcome['point_cloud_sha256'],expected_unowned_skybox_tail=skybox,
            numeric_audit=hierarchy_numeric_audit(before),requested_iterations=a.post_iterations,
            expected_optimizer_update_calls=a.post_iterations-1,configuration_sha256=digest(config_path)))
        command=[sys.executable,'-u',str(REPO/'train_post.py'),'-s',str(inputs/'chunks'/a.chunk),'-i',manifest['images_absolute'],
            '-m',str(stage_dir/'model'),'-r','1','--data_device','cpu','--eval','--iterations',str(a.post_iterations),'--save_iterations','-1',
            '--feature_lr','0.0005','--opacity_lr','0.01','--scaling_lr','0.001','--hierarchy',str(directory/'hierarchy.hier'),
            '--scaffold_file',str(scaffold),'--skybox_locked','--disable_viewer','--exposure_lr_init','0','--exposure_lr_final','0',
            '--depth_l1_weight_init','0','--depth_l1_weight_final','0']
        if a.stage=='audit-post':
            receipt=read(directory/'post-outcome.json')
            if receipt['status']!='process_exited_zero' or receipt['exit_code']!=0 or receipt['command_argv']!=command:
                raise RuntimeError('CPU post re-audit requires an actual matching successful process receipt')
            receipt['CPU_validation_recovery']=dict(driver_version=DRIVER_VERSION,driver_sha256=digest(Path(__file__)),
                GPU_reexecuted=False,reason='Pinned default compressed writer requires half-storage roundtrip comparison of coarse sky fields')
        else:
            receipt=execute(command,REPO,stage_dir,directory/'post-outcome.json')
        after=read_hierarchy(directory/'hierarchy.hier_opt',expected_unowned_tail=skybox)
        if digest(directory/'hierarchy.hier')!=creator['hierarchy_sha256']:
            raise RuntimeError('Post modified its original Creator hierarchy')
        freeze_audit=post_freeze_audit(before,after,anchors,scaffold,skybox)
        write(stage_dir/'post-freeze-audit.json',freeze_audit)
        receipt.update(status='completed',hierarchy=after.summary(),hierarchy_sha256=digest(after.path),
            requested_iterations=a.post_iterations,optimizer_update_calls=a.post_iterations-1,
            snapshot_boundary='Save after final backward, before return; final optimizer step not performed.',
            skybox_tail=skybox,selected_training_names=chunks[a.chunk]['train_names'],heldout_training_overlap=0,
            fixed_limit_sampling='Pinned train_post uses log-uniform dimensionless threshold in [0.005,0.1]; not pixel tau.',
            numeric_audit=hierarchy_numeric_audit(after),frozen_field_audit=freeze_audit,
            source_kind='post_optimized',configuration_sha256=digest(config_path))
        write(directory/'post-outcome.json',receipt)
    else:
        unoptimized=a.stage=='merge-unoptimized'
        source_kind='unoptimized' if unoptimized else 'post_optimized'
        filename='hierarchy.hier' if unoptimized else 'hierarchy.hier_opt'
        receipt_name='creator-outcome.json' if unoptimized else 'post-outcome.json'
        input_hierarchies=[]
        for chunk in chunks:
            receipt=read(root/'trained_chunks'/chunk/receipt_name)
            source=root/'trained_chunks'/chunk/filename
            if receipt['status']!='completed' or digest(source)!=receipt['hierarchy_sha256']:
                raise RuntimeError(f'Merge requires every real completed {source_kind} input')
            input_hierarchies.append(dict(chunk=chunk,source_kind=source_kind,path=str(source),sha256=receipt['hierarchy_sha256']))
        stem='merge-unoptimized' if unoptimized else 'merge'
        merged_filename='merged-unoptimized.hier' if unoptimized else 'merged.hier'
        stage_dir=root/f'{stem}-stage'
        stage_dir.mkdir(exist_ok=False)
        executable=variant['executable']['path'] if unoptimized else str(EXES/'GaussianHierarchyMerger.exe')
        command=[str(executable),str(root/'trained_chunks'),'0',str(inputs/'chunks'),str(root/merged_filename)]+list(chunks)
        receipt=execute(command,REPO,stage_dir,root/f'{stem}-outcome.json')
        h=read_hierarchy(root/merged_filename)
        receipt.update(status='completed',hierarchy=h.summary(),hierarchy_sha256=digest(h.path),skybox_in_tree=0,
            source_kind=source_kind,input_hierarchies=input_hierarchies,chunk_order=list(chunks),
            chunk_centers={name:chunks[name]['center'] for name in chunks},
            merger_executable_sha256=digest(Path(executable)),
            input_filename_interface_patch=variant['patch_path'] if unoptimized else None,
            shared_background_skybox=scaffold.as_posix(),shared_background_skybox_count=skybox,
            merge_algorithm='Unchanged official HierarchyExplicitLoader 5% nearest-center falloff; root summary moved to chunk center; per-block tails omitted.',
            numeric_audit=hierarchy_numeric_audit(h),configuration_sha256=digest(config_path))
        write(root/f'{stem}-outcome.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    main()
