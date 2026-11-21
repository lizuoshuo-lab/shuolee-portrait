"""Create independent006 post64 adapter, preserving001 pilot and all formal sources."""
from pathlib import Path
import hashlib
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
old=ROOT/'scripts/large_scene_gs/run_small_city_post_preflight_001.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='11a68bc78bdf58a521934bf941f17bd0790b60a58f3354b2c458ee0c11b43a75'
s=old.read_text(encoding='utf-8')
s=s.replace('ROOT = Path(__file__).resolve().parents[2]',"ROOT = Path(r'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab')")
s=s.replace('small-city-hierarchy-005','small-city-hierarchy-006').replace('small-city-post00-i64-r4-001','small-city-post00-i64-r4-006')
s=s.replace('0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84','a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6')
s=s.replace('55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea','562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47')
s=s.replace('fixed005','fixed006').replace('Fixed005','Fixed006').replace('formal005','formal006')
s=s.replace("def training_binding(driver, config, manifest, training):\n", "def training_binding(driver, config, manifest, training):\n    _, codec, _training, _ = driver.frozen_helpers(LOCK)\n    driver.lineage_gate(config, manifest, training, codec)\n")
s=s.replace("    creator = read(source / 'creator-outcome.json')", "    creator = driver.creator_gate('0_0', config, manifest, training, codec, helper, read(LOCK / 'plan.json')['camera_archives'])")
s=s.replace("            or creator['input_model']['sha256'] != chunk_receipt['model_parameter_sha256']):", "            or creator['input_model']['sha256'] != chunk_receipt['model_parameter_sha256']\n            or creator['training_lineage_binding_sha256'] != driver.BINDING_SHA\n            or creator['training_lineage_binding_path'] != str(driver.BINDING_PATH)):")
start=s.index('def no_live_gpu_stage():');end=s.index('\n\ndef audit_post(',start)
s=s[:start]+'''def no_live_gpu_stage():
    driver, _, _, _, _, _ = modules()
    return driver.no_conflicting_gpu_stage()
'''+s[end:]
s=s.replace("                   formal_source_configuration_sha256=CONFIG_SHA, copied_input_hashes=binding['source_file_sha256'],", "                   formal_source_configuration_sha256=CONFIG_SHA, copied_input_hashes=binding['source_file_sha256'],\n                   training_lineage_binding_path=str(driver.BINDING_PATH), training_lineage_binding_sha256=driver.BINDING_SHA,\n                   training_lineage_kind=config['training_lineage_kind'], formal_observation_reclassification=False,")
s=s.replace("    static = {'wrapper_source_sha256': digest(__file__),", "    static = {'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,\n              'training_lineage_kind': formal['training_lineage_kind'], 'formal_observation_reclassification': False,\n              'wrapper_source_sha256': digest(__file__),")
s=s.replace("              'requested_iterations': 64, 'expected_optimizer_update_calls': 63, 'cwd': str(driver.REPO),", "              'requested_iterations': 64, 'expected_optimizer_update_calls': 63, 'cwd': str(driver.REPO),\n               'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,")
s=s.replace("        receipt['child_pid'] = child.pid", "        receipt['child_pid'] = child.pid\n        receipt['actual_child_birth'] = driver.process_birth(child.pid, owned_handle=child._handle)")
s=s.replace("                   elapsed_seconds=time.perf_counter()-start, process_ended_utc=now())", "                   elapsed_seconds=time.perf_counter()-start, process_ended_utc=now(), actual_child_poll_exit_code=child.poll())")
s=s.replace("    return driver, codec, helper, training, config, manifest", "    driver.execution_source_gate()\n    driver.fixed_sources()\n    return driver, codec, helper, training, config, manifest")
s=s.replace("    paths += [ROOT / 'scripts/large_scene_gs/run_small_city_stage.py', ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py']", "    paths += [ROOT / 'scripts/large_scene_gs/run_small_city_stage.py', ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py']\n    paths += [Path(config['training_lineage_binding_path'])]")
s=s.replace("            'raw_shared100k_sky': shared_sky, 'numeric_audit': numeric, 'GPU_started': False}", "            'raw_shared100k_sky': shared_sky, 'numeric_audit': numeric, 'GPU_started': False,\n            'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA}")
s=s.replace("    if args.stage == 'post' and not args.execute:", "    if args.run != DEFAULT_RUN:\n        raise RuntimeError('006 pilot namespace is fixed; independent64 cannot write another or formal namespace')\n    if args.stage == 'post' and not args.execute:")
s=s.replace("'preGPU_policy': 'Parent exclusive scheduling only after four actual trainers terminal;", "'preGPU_policy': 'Parent exclusive scheduling only after mixed-source all5 full models terminal; exact006 active ownership chain;")
s=s.replace("'prior_Truck_method_reference':", "'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,\n              'prior_Truck_method_reference':")
target=ROOT/'scripts/large_scene_gs/run_small_city_post_preflight_006.py'
assert not target.exists(),'Preserve existing006 source'
compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')
print(target);print(hashlib.sha256(target.read_bytes()).hexdigest())
