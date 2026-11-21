"""Generate isolated formal entries from reviewed originals; no old-source edit."""
from pathlib import Path
H=Path(__file__).parent
for name in ['prepare_native_person122_v3.py','train_native_person122_v2.py']:
    if (H/name).exists():raise FileExistsError(name)
s=(H/'prepare_native_person122_v2.py').read_text(encoding='utf-8')
s=s.replace('from native_surface122_v2 import initialize_surface','from native_surface122_v2 import initialize_surface\nfrom native_pipeline_binding122 import pipeline_sources,verify_sources,verify_preparation\nfrom native_training_policy122 import digest,validate_policy')
s=s.replace("assert_snapshot(audit,path);build_fit_inputs(m,path.parent,out)","assert_snapshot(audit,path);validate_policy(m['training'].get('sampling_policy'))\n    source_refs=pipeline_sources();save(out/'pipeline-source-snapshot.json',source_refs)\n    build_fit_inputs(m,path.parent,out)")
s=s.replace("fit_source=Path(__file__).parent/'fitting_output_residual122.py'","verify_sources(source_refs)\n    fit_source=Path(__file__).parent/'fitting_output_residual122.py'")
s=s.replace("from_scratch=True,old_individual_geometry_loaded=False,executed_command=sys.argv,exit_code=0,","from_scratch=True,old_individual_geometry_loaded=False,executed_command=sys.argv,exit_code=0,pipeline_sources_sha256=digest(source_refs),")
s=s.replace("receipt=read(out/'fresh-fit-receipt.json')","source_refs=read(out/'pipeline-source-snapshot.json');verify_sources(source_refs)\n    receipt=read(out/'fresh-fit-receipt.json')")
s=s.replace("assert_snapshot(audit,path)\n    initial=dict(schema='shuolee-fresh-clothing-initialization-v1'", """assert_snapshot(audit,path);verify_sources(source_refs)
    binding=dict(run_id=m['run_id'],source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],
        frame_plan_sha256=m['frame_plan']['sha256'],frames_sha256=digest(m['frames']),observations_sha256=m['observations']['sha256'],
        material_evidence_sha256=m['physical_evidence']['sha256'],source_manifest=ref(path),pipeline_sources=source_refs,
        fit_review=ref(out/'fit-review.json'),geometry_gate=ref(out/'fit/geometry-gate.json'),surface_result=ref(out/'surface/result.json'))
    save(out/'pipeline-binding.json',binding)
    initial=dict(schema='shuolee-fresh-clothing-initialization-v2-output-residual',pipeline_binding=ref(out/'pipeline-binding.json')""")
a=s.index('def dispatch_training(output):');b=s.index('\n\ndef main():',a)
s=s[:a]+'''def dispatch_training(output):
    out=Path(output).resolve(strict=True);state=read(out/'preparation-status.json')
    if state['state']!='actual_surface_review_pending':raise RuntimeError('Only this completed fresh surface may dispatch')
    origin=read(out/'input-reference.json');path=Path(origin['path'])
    assert_snapshot(read(out/'admission.json'),path)
    training=deepcopy(read(path));training['fresh_initialization']=ref(out/'initialization.json')
    training['surface_review']=ref(out/'surface-review.json')
    verify_preparation(training,path.parent)
    prepared=out/'prepared-training-input.json'
    if prepared.exists():raise FileExistsError('No second dispatch or cross-run optimizer')
    save(prepared,training)
    command=[sys.executable,str(Path(__file__).parent/'train_native_person122_v2.py'),'train','--input',str(prepared),'--output',str(out/'native-training')]
    with (out/'native-training.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status(out,state='native_training_entry_returned',exit_code=result.returncode,command=command,
        training_state=read(out/'native-training/status.json') if (out/'native-training/status.json').exists() else None)
    return result.returncode
''' + s[b:]
s=s.replace('The old fitter is invoked from its pinned local project with fresh outputs.','The versioned full-output-camera fitter uses the existing pinned priors and fresh outputs.')
(H/'prepare_native_person122_v3.py').write_text(s,encoding='utf-8')
s=(H/'train_native_person122.py').read_text(encoding='utf-8')
s=s.replace('native v2 optimizer','complete-output residual optimizer')
s=s.replace('from native_training_admission122 import','from native_training_admission122_v2 import')
s=s.replace('def now():', 'from native_pipeline_binding122 import verify_preparation,sampling_for_input\nfrom native_training_policy122 import digest\n\ndef now():')
s=s.replace('from native_training122 import NativeTrainingView','from native_training122_v4 import create_view')
s=s.replace('self.value=NativeTrainingView.create(camera,**values,device=self.device)','self.value=create_view(camera,**values,device=self.device)')
s=s.replace('if self.current!=frame[\'frame_id\']:',"key=(frame['frame_id'],digest(camera),digest(frame))\n        if self.current!=key:")
s=s.replace("self.current=frame['frame_id']","self.current=key")
s=s.replace('self.frame=frame;self.camera=camera;self.cache=cache','self.frame=frame;self.camera=json.loads(json.dumps(camera));self.cache=cache\n        self._recorded_camera_digest=digest(self.camera)')
s=s.replace("schema='shuolee-fresh-clothing-initialization-v1'","schema='shuolee-fresh-clothing-initialization-v2-output-residual'")
s=s.replace("r=read(checked_asset(base,manifest['fresh_initialization']))","preparation=verify_preparation(manifest,base)\n    r=read(checked_asset(base,manifest['fresh_initialization']))")
s=s.replace("input_manifest_sha256=manifest_sha)","input_manifest_sha256=manifest_sha,preparation=preparation)")
s=s.replace('from native_training122_v2 import FreshNativeGaussians,optimize_fresh,export_native_checkpoint','from native_training122_v5 import FreshNativeGaussians,optimize_fresh,export_native_checkpoint,render_view')
s=s.replace('from native_portrait_renderer122_v2 import NativeCameraContractV2','from native_residual_camera122 import ResidualRenderContract')
s=s.replace('from native_scene_package122_v2 import runtime_record,validate_runtime','from native_scene_package122_v5 import runtime_record,validate_runtime')
s=s.replace("contract=NativeCameraContractV2('pinhole','rectified')","contract=ResidualRenderContract('processed_output_pixels')")
s=s.replace("training=manifest['training'];steps=training['steps']","training=manifest['training'];steps=training['steps']\n        policy,graph=sampling_for_input(manifest,path.parent,cameras)")
a=s.index('        def snapshot(step,m):');b=s.index('        def review(step,snapshot_receipt):',a)
s=s[:a]+'''        def snapshot(step,m):
            assert_snapshot(report,path)
            verify_preparation(manifest,path.parent)
            scene=export_native_checkpoint(output/f'step-{step:06d}',m,views,contract,scope='unadmitted-candidate')
            files=[]
            # Review all predeclared original views. No resize, crop or image text.
            with torch.no_grad():
                for v in views:
                    rgb,alpha,_=render_view(m,v,contract)
                    for name,value in [('rgb',rgb),('alpha',alpha)]:
                        data=value.detach().cpu().numpy()
                        if name=='alpha':data=data[...,0]
                        file=scene.parent/f'frame-{v.frame["frame_id"]:06d}-{name}.png'
                        Image.fromarray(np.rint(np.clip(data,0,1)*255).astype(np.uint8)).save(file)
                        files.append(dict(path=str(file),sha256=sha(file)))
            origin=dict(step=step,scene_sha256=sha(scene),input_manifest_sha256=report['manifest_sha256'],
                initialization=init_receipt,complete_cameras_sha256=digest(cameras),
                original_frame_assets=manifest['frames'],review_renders=files,physical_quality_accepted=False)
            save(scene.parent/'training-origin.json',origin)
            return dict(scene=str(scene),scene_sha256=sha(scene),point_cloud_sha256=sha(scene.parent/'point_cloud.ply'),
                origin=dict(path=str(scene.parent/'training-origin.json'),sha256=sha(scene.parent/'training-origin.json')))
''' + s[b:]
s=s.replace("review_steps=[1000],review=review,checkpoint=snapshot)","review_steps=[1000],review=review,checkpoint=snapshot,policy=policy,graph=graph)")
s=s.replace("final=snapshot(result['steps'],model) if result['steps']!=1000 else status['snapshot']","final=result['snapshots'][result['steps']]")
(H/'train_native_person122_v2.py').write_text(s,encoding='utf-8')
print('R7_ENTRIES_WRITTEN')
