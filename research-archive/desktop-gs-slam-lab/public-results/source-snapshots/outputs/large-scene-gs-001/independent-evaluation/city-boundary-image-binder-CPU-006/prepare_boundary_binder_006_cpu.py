"""CPU source versioning only; preserve the immutable005 boundary binder."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py'
DEST=HERE/'audit_city_boundary_images_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD)=='15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f'
assert not DEST.exists()
t=OLD.read_text(encoding='utf-8').replace('parents[2]','parents[4]')
t=t.replace('city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py','city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py').replace('bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133','5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c')
t=t.replace('city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py','city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py').replace('c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610','260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3')
t=t.replace('small-city-hierarchy-005','small-city-hierarchy-006').replace('55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea','562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47').replace('0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84','a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6').replace('source_lock005_configuration_SHA','source_lock006_configuration_SHA')
t=t.replace("MODELS = ('merged-unoptimized', 'merged-post')", "BINDING=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'\nBINDING_SHA='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'\nNATIVE_LOCK=ADAPTER.parent/'native-workflow-lock.json'\nNATIVE_LOCK_SHA='0fac8a7e861fb05f5b67d3edab7851dacf4d6af99f211d16f1cf78c9b3eebea8'\nMODELS = ('merged-unoptimized', 'merged-post')")
t=t.replace("    protected = {str(audit_path):digest(audit_path)}", """    require(posterior['training_lineage_binding_sha256']==BINDING_SHA and posterior['native_workflow_lock_sha256']==NATIVE_LOCK_SHA and posterior['formal_observation_reclassification'] is False,'006 actual posterior origin/seal differs')
    protected = {str(audit_path):digest(audit_path)}
    bound_file(BINDING,BINDING_SHA,protected);bound_file(NATIVE_LOCK,NATIVE_LOCK_SHA,protected)
    bound_file(evaluation/'006-completed-training-lineage-binding.json',BINDING_SHA,protected)
    bound_file(evaluation/'006-native-workflow-lock.json',NATIVE_LOCK_SHA,protected)
    bound_file(LOCK/'driver-source.py',WORKFLOW_SHA,protected)
    mixed=read(BINDING);require(mixed['status']=='actual_complete_mixed_local_training_lineage_CPU_bound' and mixed['formal_observation_reclassification'] is False,'006 completed mixed training origin missing')""")
t=t.replace("    canonical = read(CANON/'protocol.json'); local = read(evaluation/'protocol.json')", """    require(source_outcome['training_lineage_binding_sha256']==overall['training_lineage_binding_sha256']==BINDING_SHA and source_outcome['native_workflow_lock_sha256']==overall['native_workflow_lock_sha256']==NATIVE_LOCK_SHA and source_outcome['formal_observation_reclassification'] is False and overall['formal_observation_reclassification'] is False,'006 actual outcome/metrics lineage differs')
    canonical = read(CANON/'protocol.json'); local = read(evaluation/'protocol.json')
    require(local['training_lineage_binding_sha256']==BINDING_SHA and local['native_workflow_lock_sha256']==NATIVE_LOCK_SHA and local['formal_observation_reclassification'] is False,'006 actual protocol lineage differs')""")
t=t.replace("selected_actual_models=list(MODELS),selected_settings=['leaves','tau3'])", "selected_actual_models=list(MODELS),selected_settings=['leaves','tau3'],training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=NATIVE_LOCK_SHA,formal_observation_reclassification=False,training_lineage_kind=mixed['training_lineage_kind'])")
ast.parse(t);DEST.write_text(t,encoding='utf-8');shutil.copyfile(OLD,HERE/'original005-boundary-source.py')
(HERE/'generation-manifest.json').write_text(json.dumps({'status':'CPU006_boundary_source_prepared_no_actual_dense_results','original_sha256':sha(OLD),'new_sha256':sha(DEST),'posterior_sha256':'5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c','adapter_sha256':'260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3','source_lock_sha256':'0fac8a7e861fb05f5b67d3edab7851dacf4d6af99f211d16f1cf78c9b3eebea8','Torch_imported':False,'CUDA_called':False,'actual_dense_completed':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(DEST),'sha256':sha(DEST),'status':'prepared_not_actual_dense'}))
