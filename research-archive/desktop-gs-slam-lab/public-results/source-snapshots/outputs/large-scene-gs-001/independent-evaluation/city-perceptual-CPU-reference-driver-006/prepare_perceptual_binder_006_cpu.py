"""CPU-only versioned wrapper; fixed CPU metric implementation stays unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
HERE=Path(__file__).resolve().parent
OLD=OUT/'city-perceptual-CPU-reference-driver-001/recompute_city_perceptual_cpu.py'
BOUNDARY=OUT/'city-boundary-image-binder-CPU-006/audit_city_boundary_images_006.py'
DEST=HERE/'recompute_city_perceptual_cpu_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD)=='1ec28a295b93869c7da414f830e2d70f8c1c4a02ef1f763de36ac8cfeeb6f9fc'
assert sha(BOUNDARY)=='f16ccb274707515a29cf3466be9c294b206b08a2cd1404f05321a99c88669379'
assert not DEST.exists()
t=OLD.read_text(encoding='utf-8')
t=t.replace("BOUNDARY_BINDER=ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py'", "BOUNDARY_BINDER=OUTROOT/'city-boundary-image-binder-CPU-006/audit_city_boundary_images_006.py'")
t=t.replace('15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f','f16ccb274707515a29cf3466be9c294b206b08a2cd1404f05321a99c88669379')
t=t.replace('actual_full_bef0_posterior_source_input_output_identity_before_CPU_metric','actual_full006_posterior_source_input_output_identity_before_CPU_metric')
t=t.replace("            save(output/'future1404-dispatch-binding.json'", """            # Boundary binder has a deliberately selected4-setting image scope.
            # The full CPU metric wrapper additionally seals all12 per-image
            # score lists to the completed posterior's actual metrics hashes.
            summaries={(r['model'],r['setting']):r for r in prior['summaries']}
            require(len(summaries)==12,'Actual006 posterior12 setting proofs missing')
            canonical=read(COARSE/'protocol.json')['test_names']
            for model in ('merged-unoptimized','merged-post'):
                for setting in ('leaves','tau1','tau3','tau5','tau6','tau15'):
                    folder=OUTROOT/a.evaluation_run/model/setting;metric_path=folder/'metrics.json';per_path=folder/'per-image-metrics.json'
                    require(digest(metric_path)==summaries[(model,setting)]['actualmetrics_sha256'],'Actual006 CPU dispatch metric changed since posterior')
                    metric,per=read(metric_path),read(per_path)
                    require(metric['per_image']==per and [r['name'] for r in per]==canonical and len(per)==117,'Actual006 per-image scores detached from full setting proof')
                    protected[str(metric_path)]=digest(metric_path);protected[str(per_path)]=digest(per_path)
            save(output/'future1404-dispatch-binding.json'""")
ast.parse(t);DEST.write_text(t,encoding='utf-8');shutil.copyfile(OLD,HERE/'original005-perceptual-wrapper.py')
(HERE/'generation-manifest.json').write_text(json.dumps({'status':'CPU006_perceptual_binding_prepared_not_actual1404','old_wrapper_sha256':sha(OLD),'new_wrapper_sha256':sha(DEST),'boundary006_sha256':sha(BOUNDARY),'optional_numeric_source_sha256':'4b318c7362c35cce5087605ac7c1902fb0697c67185c039e3150676cfd3a3181','CPU_threads_default':4,'interop_threads':1,'batch_size':1,'CUDA_called':False,'actual1404_completed':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(DEST),'sha256':sha(DEST),'status':'prepared_actual1404_missing'}))
