"""One bounded, fixed-camera PixSfM KA runtime check; never a training launcher."""
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import time
import traceback

ROOT = Path('/opt/gs-slam-lab/shuolee-pixsfm116')
BASE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
SOURCE = BASE/'current/full-scene-localization-074'
OUT = BASE/'current/portrait-alignment-open-source-116/pixsfm'
RUN = OUT/'ka-vgg-smoke'
WEIGHT = Path('/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
WEIGHT_SHA = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
SOURCE_SHA = 'e0f19bb14e418e4fe44235d7cd7b5f6cd11cc3eafaee29a0094dfbdbe29626e2'
INPUT_SHA = 'a40d0cb5cd678b414fb0adc40425cb2709229952f900c1a12da0b2a1d7b1c37b'
FEATURE_VARIANT = 'Official PixSfM VGGNet, ImageNet1K V1, conv2_2 (128 channels)'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, value):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def db_read(p):
    return sqlite3.connect(p.as_uri()+'?mode=ro', uri=True)

def summary(values):
    import numpy as np
    a = np.asarray(values)
    return {'n': int(a.size), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'max': float(a.max())} if a.size else {'n': 0}

def progress(stage, **extra):
    save(RUN/'progress.json', {'updated': now(), 'state': 'running', 'stage': stage,
         'pid': os.getpid(), 'started': STARTED, 'deadline': DEADLINE,
         'feature_variant': FEATURE_VARIANT, 'cameras_optimized': False,
         'gaussian_training_started': False, **extra})

def deadline_hit(*_):
    raise TimeoutError('Pre-registered 30-minute KA smoke budget reached.')

def main():
    global STARTED, DEADLINE
    lock = (ROOT/'ka-smoke.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(OUT/'runtime-check.json')['imports_passed'] is True
    assert read(OUT/'setup-state.json')['state'] == 'ready_for_feature_weight_check'
    reg = read(OUT/'smoke-preregistration.json')
    assert reg['runtime_config']['dense_features']['model']['hypercolumn_layers'] == ['conv2_2']
    assert not RUN.exists(), 'Never overwrite a previous KA attempt.'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True)
    assert not gpu.stdout.strip(), 'Another GPU task is present; do not start this smoke.'
    RUN.mkdir()
    STARTED = now()
    DEADLINE = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=30)).isoformat()
    signal.signal(signal.SIGALRM, deadline_hit)
    signal.alarm(1800)
    started_clock = time.monotonic()
    progress('verify_inputs_and_official_weight')
    try:
        assert sha(SOURCE/'database.db') == SOURCE_SHA
        assert sha(OUT/'smoke-input.db') == INPUT_SHA
        assert sha(WEIGHT) == WEIGHT_SHA
        assert WEIGHT.stat().st_size == 553433881
        camera_hash = sha(SOURCE/'cameras.json')
        import numpy as np
        from PIL import Image
        import torch
        import torchvision
        from pixsfm.refine_colmap import PixSfM
        from pixsfm.util.colmap import read_keypoints_from_db, read_matches_from_db
        from omegaconf import OmegaConf

        def no_download(*args, **kwargs):
            raise RuntimeError('Network weight downloads are disabled for this smoke.')
        torch.hub.set_dir('/root/.cache/torch/hub')
        torch.hub.download_url_to_file = no_download
        expected_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        assert torchvision.models.VGG16_Weights.IMAGENET1K_V1.url == expected_url
        torch.set_num_threads(4)
        torch.manual_seed(116)
        sfm = PixSfM(reg['runtime_config'])
        state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
        encoder = sfm.extractor.model.encoder.state_dict()
        verified = []
        for key, value in encoder.items():
            reference = state['features.'+key]
            assert value.shape == reference.shape and torch.equal(value, reference), key
            verified.append(key)
        assert len(verified) == 8 and sfm.extractor.model.output_dims == [128]
        del state
        save(RUN/'weight-provenance.json', {
            'verified': now(), 'file': str(WEIGHT), 'sha256': WEIGHT_SHA,
            'official_url': expected_url, 'torchvision_enum': 'VGG16_Weights.IMAGENET1K_V1',
            'feature_variant': FEATURE_VARIANT, 'encoder_tensors_exactly_equal': verified,
            'random_weights_used': False, 'new_weights_downloaded': False,
            'official_default_s2dnet': False,
            'scope': 'Public official alternative feature backend; not an S2DNet reproduction.'})
        save(RUN/'resolved-config.json', OmegaConf.to_container(sfm.conf, resolve=True))
        source_db = db_read(OUT/'smoke-input.db')
        images = source_db.execute('SELECT image_id,name FROM images ORDER BY image_id').fetchall()
        assert sorted(int(Path(n).stem.split('_')[-1]) for _, n in images) == reg['frames']
        image_hashes = {}
        scales = {}
        for iid, name in images:
            path = SOURCE/'images'/name
            with Image.open(path) as image:
                assert image.size == (1520,2704)
                resized = sfm.extractor.get_scaled_image_size(image)
                scales[name] = np.array([resized[0]//2/1520, resized[1]//2/2704])
            image_hashes[name] = sha(path)
        before = read_keypoints_from_db(OUT/'smoke-input.db', as_cpp_map=False)
        pairs, matches, _ = read_matches_from_db(OUT/'smoke-input.db')
        matched = {name: set() for _,name in images}
        for (left,right), pair_matches in zip(pairs,matches):
            matched[left].update(map(int,pair_matches[:,0]))
            matched[right].update(map(int,pair_matches[:,1]))
        save(RUN/'input-manifest.json', {'created':now(), 'source_database_sha256':SOURCE_SHA,
            'smoke_database_sha256':INPUT_SHA, 'camera_sha256':camera_hash,
            'raw_image_hashes':image_hashes, 'frames':reg['frames'],
            'pixel_coordinates':'Unchanged COLMAP database coordinates on original raw images.',
            'matching_pairs':len(pairs), 'matched_keypoints':sum(map(len,matched.values()))})
        progress('fixed_camera_keypoint_adjustment')
        refined, solver_data, feature_manager = sfm.refine_keypoints_from_db(
            RUN/'refined.db', OUT/'smoke-input.db', SOURCE/'images',
            cache_path=RUN/'featuremaps.h5')
        torch.cuda.synchronize()
        progress('verify_outputs_and_input_preservation')
        delta_arrays = {}
        displacement_all = []
        displacement_matched = []
        region_changes = {name: [] for name in reg['region_names']}
        regions = np.load(OUT/'database-regions.npz')
        feature_bound_max = 0.0
        frame_records = []
        changed_matched = 0
        for iid,name in images:
            after = np.asarray(refined[name])
            assert after.shape == before[name].shape and np.isfinite(after).all()
            delta = after-before[name]
            ids = np.array(sorted(matched[name]), dtype=int)
            displacement = np.linalg.norm(delta,axis=1)
            feature_bound_max = max(feature_bound_max, float(np.abs(delta*scales[name]).max()))
            assert feature_bound_max <= reg['runtime_config']['KA']['optimizer']['bound']+0.001
            unmatched = np.ones(len(delta),dtype=bool);unmatched[ids]=False
            assert np.array_equal(after[unmatched],before[name][unmatched])
            changed_matched += int((displacement[ids] > 1e-5).sum())
            displacement_all.extend(displacement)
            displacement_matched.extend(displacement[ids])
            labels = regions[str(iid)]
            for region_index,region_name in enumerate(reg['region_names']):
                region_changes[region_name].extend(displacement[ids[labels[ids]==region_index]])
            delta_arrays[name] = delta
            frame_records.append({'name':name,'all':summary(displacement),
                                  'matched':summary(displacement[ids])})
        np.savez_compressed(RUN/'keypoint-displacements.npz',**delta_arrays)
        # Camera records are never passed to an optimizer; DB camera rows must remain byte-identical.
        output_db = db_read(RUN/'refined.db')
        assert source_db.execute('SELECT * FROM cameras ORDER BY camera_id').fetchall() == output_db.execute('SELECT * FROM cameras ORDER BY camera_id').fetchall()
        source_db.close();output_db.close()
        assert sha(SOURCE/'database.db') == SOURCE_SHA
        assert sha(OUT/'smoke-input.db') == INPUT_SHA
        assert sha(SOURCE/'cameras.json') == camera_hash
        assert image_hashes == {n:sha(SOURCE/'images'/n) for n in image_hashes}
        summaries = solver_data.get('summary',[])
        solver_records=[]
        for s in summaries:
            record={}
            for key in ('initial_cost','final_cost','num_successful_steps','num_unsuccessful_steps','termination_type'):
                value=getattr(s,key,None)
                record[key]=value if isinstance(value,(int,float,str,bool,type(None))) else str(value)
            for method in ('brief_report','BriefReport'):
                if hasattr(s,method):
                    record['brief_report']=getattr(s,method)();break
            solver_records.append(record)
        result={'completed':now(),'elapsed_seconds':time.monotonic()-started_clock,
            'state':'completed_runtime_smoke_only','runtime_smoke_pass':changed_matched>0,
            'feature_variant':FEATURE_VARIANT,'official_default_s2dnet':False,
            'frames':reg['frames'],'cameras_optimized':False,'original_inputs_unchanged':True,
            'finite_output':True,'matched_points_changed':changed_matched,
            'max_axis_displacement_feature_pixels':feature_bound_max,
            'displacement_original_pixels':{'all':summary(displacement_all),
                'matched':summary(displacement_matched),'matched_by_region':{k:summary(v) for k,v in region_changes.items()}},
            'frame_records':frame_records,'solver':solver_records,
            'gaussian_training_started':False,'alignment_quality_accepted':False,
            'physical_camera_calibration_verified':False,
            'interpretation':'Runtime and bounded keypoint movement only. Six-view graph has only one 3+-view lower-face component. No independent accuracy or ghosting improvement is claimed.'}
        save(RUN/'result.json',result)
        save(RUN/'progress.json',{'updated':now(),'state':'completed','stage':'runtime_smoke_only',
             'pid':os.getpid(),'started':STARTED,'deadline':DEADLINE,'quality_accepted':False})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        save(RUN/'result.json',{'updated':now(),'state':'failed','exception':str(exc),
             'traceback':traceback.format_exc(),'gaussian_training_started':False,
             'alignment_quality_accepted':False})
        save(RUN/'progress.json',{'updated':now(),'state':'failed','pid':os.getpid(),
             'started':STARTED,'deadline':DEADLINE,'exception':str(exc)})
        raise
    finally:
        signal.alarm(0)

if __name__ == '__main__':
    main()
