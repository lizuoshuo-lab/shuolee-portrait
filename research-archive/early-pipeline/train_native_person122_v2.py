"""Run the complete-output residual optimizer only after actual input and fresh-fit checks.

Prepared native frames/cameras/masks and a newly executed fit/surface stage are
required. This command does NOT implement raw-video calibration or fitting.
No bypass flag, checkpoint resume, old-body initialization or implicit K update.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, sys, time, traceback
import numpy as np
from PIL import Image
from native_training_admission122_v2 import audit_input, assert_snapshot, checked_asset, native_camera, read, save, sha

from native_pipeline_binding122 import verify_preparation,sampling_for_input
from native_training_policy122 import digest

def now():return datetime.now(timezone.utc).isoformat()

class PixelCache:
    """One native image/mask set on GPU; all other views retain only cameras."""
    def __init__(self,base,device):self.base=base;self.device=device;self.current=None;self.value=None
    def get(self,frame,camera):
        cache_key=(frame['frame_id'],digest(camera),digest(frame))
        if self.current!=cache_key:
            from native_training122_v4 import create_view
            self.value=None
            values={}
            for channel in ['rgb','alpha','rgb_valid','observation_valid']:
                with Image.open(checked_asset(self.base,frame[channel])) as im:a=np.array(im)
                if channel=='rgb':values[channel]=a.astype(np.float32)/255.
                elif channel=='alpha':values[channel]=a[...,None].astype(np.float32)/255.
                else:values[channel]=(a==255)[...,None]
            self.value=create_view(camera,**values,device=self.device)
            self.current=cache_key
        return self.value

class DiskView:
    def __init__(self,frame,camera,cache,device):
        import torch
        self.frame=frame;self.camera=json.loads(json.dumps(camera));self.cache=cache
        self._recorded_camera_digest=digest(self.camera)
        self.K=torch.tensor(camera['K'],dtype=torch.float32,device=device)[None]
        self.w2c=torch.tensor(camera['w2c'],dtype=torch.float32,device=device)[None]
    @property
    def rgb(self):return self.cache.get(self.frame,self.camera).rgb
    @property
    def alpha(self):return self.cache.get(self.frame,self.camera).alpha
    @property
    def rgb_valid(self):return self.cache.get(self.frame,self.camera).rgb_valid
    @property
    def observation_valid(self):return self.cache.get(self.frame,self.camera).observation_valid

def load_fresh_initialization(manifest,base,manifest_sha):
    """Bind a zero-step surface to this run, exact input and newly executed fit.

    Provenance is necessary, not a proof against maliciously forged records.
    The upstream fit/surface executor remains separately responsible for the
    actual new fit. This entry never relabels a loaded checkpoint as fresh.
    """
    preparation=verify_preparation(manifest,base)
    r=read(checked_asset(base,manifest['fresh_initialization']))
    expected=dict(schema='shuolee-fresh-clothing-initialization-v2-output-residual',run_id=manifest['run_id'],
        source_sha256=manifest['source']['asset']['sha256'],cameras_sha256=manifest['cameras']['sha256'],
        frame_plan_sha256=manifest['frame_plan']['sha256'],representation='free_world_clothing_surface',
        world_frame='right_handed_world_y_up',old_individual_geometry_loaded=False,old_gaussians_loaded=False,
        old_optimizer_loaded=False,training_steps=0,initial_points=200000)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Fresh initializer provenance disagrees with this admitted run')
    fit=read(checked_asset(base,r['fresh_fit_receipt']))
    for key in ['run_id','source_sha256','cameras_sha256','frame_plan_sha256']:
        if fit.get(key)!=expected[key]:raise ValueError('Fresh fit input/run mismatch')
    if fit.get('from_scratch') is not True or fit.get('old_individual_geometry_loaded') is not False or not fit.get('executed_command') or fit.get('exit_code')!=0:
        raise ValueError('Actual completed from-scratch fit receipt required')
    if fit.get('fit_arrays_sha256')!=r['fit_arrays']['sha256'] or fit.get('fit_source_sha256')!=r['fit_source']['sha256']:
        raise ValueError('Fresh fit receipt does not bind its actual output and source')
    if not r.get('executed_command') or r.get('exit_code')!=0:raise ValueError('Actual initializer execution receipt required')
    for key in ['fit_arrays','fit_source','initializer_source']:
        checked_asset(base,r[key])
    arrays_path=checked_asset(base,r['arrays'])
    with np.load(arrays_path,allow_pickle=False) as raw:
        if set(raw.files)!={'means','quats','scales','opacities','sh'}:raise ValueError('Initialization arrays only, never a checkpoint')
        arrays={k:raw[k].copy() for k in raw.files}
    shapes={'means':(200000,3),'quats':(200000,4),'scales':(200000,3),'opacities':(200000,),'sh':(200000,16,3)}
    if any(arrays[k].shape!=s or not np.isfinite(arrays[k]).all() for k,s in shapes.items()):raise ValueError('Exactly200000 finite complete Gaussian attributes required')
    if (arrays['scales']<=0).any() or ((arrays['opacities']<=0)|(arrays['opacities']>=1)).any() or (np.linalg.norm(arrays['quats'],axis=1)<1e-8).any():raise ValueError('Invalid initialization covariance/opacity')
    return arrays,dict(receipt_sha256=sha(checked_asset(base,manifest['fresh_initialization'])),arrays_sha256=sha(arrays_path),input_manifest_sha256=manifest_sha,preparation=preparation)

def execute(path,output,stage):
    path=Path(path).resolve(strict=True);output=Path(output)
    if output.exists():raise FileExistsError('New run output required; no resume: '+str(output))
    output.mkdir(parents=True,exist_ok=False)
    report,manifest=audit_input(path)
    save(output/'admission.json',report)
    status=dict(updated=now(),run_id=manifest.get('run_id'),stage='admission',input_admitted=report['training_admitted'],
        state='input_audited' if report['training_admitted'] else 'input_rejected',training_started=False,
        GPU_imported='torch' in sys.modules,initializer_loaded=False,optimizer_constructed=False,person_quality_accepted=False)
    save(output/'status.json',status)
    if not report['training_admitted']:return 3
    if stage=='audit':return 0
    if manifest['scope']!='person-reconstruction':raise ValueError('Synthetic fixture may audit but cannot enter the person training command')
    assert_snapshot(report,path)
    arrays,init_receipt=load_fresh_initialization(manifest,path.parent,report['manifest_sha256'])
    status.update(stage='fresh_initialization_verified',initializer_loaded=True);save(output/'status.json',status)
    if os.name!='posix':raise RuntimeError('Native person training requires the existing local Linux CUDA environment')
    import fcntl,subprocess
    with open('/tmp/shuolee-native122-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('Another GPU process is active; do not duplicate')
        import torch
        from native_training122_v5 import FreshNativeGaussians,optimize_fresh,export_native_checkpoint,render_view
        from native_residual_camera122 import ResidualRenderContract
        from native_scene_package122_v5 import runtime_record,validate_runtime
        assert_snapshot(report,path)
        validate_runtime(runtime_record())
        # No model/optimizer is created until both gates and the single-GPU lock.
        contract=ResidualRenderContract('processed_output_pixels')
        contract.validate()
        camera_doc=read(checked_asset(path.parent,manifest['cameras']))
        byframe={c['frame_id']:c for c in camera_doc['cameras']};cache=PixelCache(path.parent,'cuda')
        views=[];cameras=[];bridges=[]
        for frame in manifest['frames']:
            camera,bridge=native_camera(byframe[frame['frame_id']],camera_doc,manifest['source']['width'],manifest['source']['height'],Path(frame['rgb']['path']).name)
            views.append(DiskView(frame,camera,cache,'cuda'));cameras.append(camera);bridges.append(bridge)
        training=manifest['training'];steps=training['steps']
        policy,graph=sampling_for_input(manifest,path.parent,cameras)
        if type(steps)is not int or not 1000<=steps<=18000:raise ValueError('Person run budget1000..18000 with actual1000review required')
        model=FreshNativeGaussians(arrays,device='cuda')
        save(output/'pixel-origin-bridges.json',bridges);save(output/'fresh-init-binding.json',init_receipt)
        status.update(updated=now(),stage='native_gaussian_training',state='running',pid=os.getpid(),training_started=True,GPU_imported=True)
        save(output/'status.json',status)
        def snapshot(step,m):
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
        def review(step,snapshot_receipt):
            target=output/f'review-{step:06d}.json'
            status.update(updated=now(),stage='actual_review_pending',step=step,snapshot=snapshot_receipt)
            save(output/'status.json',status)
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                if target.exists():
                    decision=read(target)
                    if decision.get('snapshot')!=snapshot_receipt or not decision.get('actual_same_camera_face_body_review') or not decision.get('actual_center_distribution_review'):
                        raise ValueError('Review must bind actual snapshot and full person/center inspection')
                    checked_asset(output,decision['review_evidence'])
                    return decision.get('continue_same_run') is True
                time.sleep(5)
            return False
        result,_=optimize_fresh(model,views,contract,steps=steps,learning_rates=training['learning_rates'],seed=training['seed'],
            output=output/'gaussian-run',review_steps=[1000],review=review,checkpoint=snapshot,policy=policy,graph=graph)
        final=result['snapshots'][result['steps']]
        assert_snapshot(report,path)
        status.update(updated=now(),stage='quality_review_pending',state=result['state'],steps=result['steps'],final=final,
            optimizer_constructed=True,person_quality_accepted=False)
        save(output/'status.json',status)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','train']);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    existed_before=Path(args.output).exists()
    try:code=execute(args.input,args.output,args.stage)
    except Exception as e:
        out=Path(args.output)
        if not existed_before and out.exists() and not (out/'failure.json').exists():
            save(out/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
            if (out/'status.json').exists():
                status=read(out/'status.json');status.update(state='failed',updated=now(),error=repr(e));save(out/'status.json',status)
        raise
    print(json.dumps(dict(exit_code=code,status=str(Path(args.output)/'status.json'))))
    return code
if __name__=='__main__':sys.exit(main())
