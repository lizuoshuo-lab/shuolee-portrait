"""Graywoman12 input adapter and post-update/offline-review lifecycle only.

Reuses the existing SH0/ROI2x/paired-background preparation and native training
function. No fitting, solver/loss/sampling strategy, LR or density changes.
"""
import argparse
import ast
import copy
import difflib
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import train_football_half12_sh0126 as base
import train_indoor_person_zoom124 as zoom
from train_indoor_circle16_sh0124 import reused
import cross_subject_sh0124 as gs

SUBJECT = 'graywoman-half12-paired18k128'
OUT = gs.ROOT/SUBJECT
FIT = gs.ROOT/'graywoman-half12-pose128'
HANDOFF = FIT/'gaussian-native-handoff'
N = 12
TOTAL = 18000
COMPUTE_BUDGET = 2400
UPSTREAM_TRAIN_SHA = 'db308103af3b7cd7a6718fb17e168bea43725bdf4cb0be07899964afc3f9230a'
LIFECYCLE_SHA = 'd809fa52b92c126534e4d605194359d7e641b097ebf840a512cefa550e9ce224'
PROFILE_KEYS = ('options','upstream_options','schedule_overrides','shared_laplacian_weight',
                'goal_points_approx','goal_point_band','max_points_safety_stop')


def adapted_prepare(reference, frames):
    edits = [
        ('        ("row[\'wh\']==[1520,2704]", "row[\'wh\']==[2704,1520]", 1),\n', ''),
        ('("record_id=\'001\'", "record_id=\'011\'", 2)', '("record_id=\'001\'", "record_id=\'015\'", 2)'),
        ("code = code.replace('winterhat', 'football').replace('Winterhat', 'Football').replace('冬帽', '足球场新')",
         "code = code.replace('winterhat', 'graywoman').replace('Winterhat', 'Graywoman').replace('冬帽', '灰毛衣新')"),
    ]
    function, diff = reused(base.adapted_prepare, edits, dict(N=N,OUT=OUT,FIT=FIT,
        HANDOFF=HANDOFF,SUBJECT=SUBJECT,__file__=__file__), 'graywoman12-input-only')
    prepared, before, changed = function(reference, frames)
    assert "row['wh']==[1520,2704]" in changed
    return prepared, before, changed, diff


def adapter_from_prepare(changed, reference):
    # Evaluate just the already-existing adapter-string generation, never the
    # mesh handoff, mkdir, source decoding, fitting or GPU path.
    function = ast.parse(changed).body[0]
    start = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='before' for t in node.targets)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=='read_text')
    end = next(i for i,node in enumerate(function.body[start:],start) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='ast'
        and node.value.func.attr=='parse')
    scope = dict(base.previous.prepare.__globals__,reference=reference)
    exec(compile(ast.Module(body=function.body[start:end],type_ignores=[]),'<same-profile-string-generation>','exec'),scope)
    return base.add_alpha_export(scope['code'])


def lifecycle_adapter(original):
    """Patch only adapter lifecycle; never edit the AST of train.py::training."""
    tree = ast.parse(original)
    function = tree.body[0]
    assert isinstance(function,ast.FunctionDef) and function.name=='train'
    before_body = next(n for n in function.body if isinstance(n,ast.ClassDef) and n.name=='Body')
    before_body_dump = ast.dump(before_body,include_attributes=False)
    edits = dict(snapshot_schedule=0,run_mkdir=0,started=0,saved=0,scene_cache=0,
                 wait_removed=0,namespace=0,native_call=0,stop_handler=0)
    new_body = []
    for node in function.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='SNAPSHOTS':
                assert ast.literal_eval(node.value)==[1000,10000,18000]
                node.value=ast.parse('[10000,18000]',mode='eval').body;edits['snapshot_schedule']+=1
            elif name=='started':
                node.value=ast.Name(id='_GRAY_PROCESS_STARTED',ctx=ast.Load());edits['started']+=1
            elif name=='saved':
                assert ast.literal_eval(node.value)==[]
                node.value=ast.parse("[1000] if runtime_phase['mode']=='resume' else []",mode='eval').body;edits['saved']+=1
            elif name=='namespace':
                new_body.extend(ast.parse("life=_make_lifecycle(locals(),runtime_phase)\nfrom staged_training_lifecycle128 import ReviewStop as StageReviewStop").body)
                assert isinstance(node.value,ast.Call)
                for keyword in node.value.keywords:
                    if keyword.arg=='torch': keyword.value=ast.parse('life.torch_proxy(torch)',mode='eval').body
                    if keyword.arg=='DataLoader': keyword.value=ast.parse('life.camera_loader',mode='eval').body
                edits['namespace']+=1
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='run.mkdir':
            assert ast.unparse(node.value)=='run.mkdir(exist_ok=False)'
            new_body.extend(ast.parse("runtime_phase=_GRAY_RUNTIME_PHASE\nif runtime_phase['mode']!='resume':\n    run.mkdir(exist_ok=False)\nelse:\n    assert run.is_dir()").body)
            edits['run_mkdir']+=1
            continue
        if isinstance(node,ast.ClassDef) and node.name=='Scene':
            new_body.extend(ast.parse('scene_holder={}').body)
            init=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
            init.body.insert(0,ast.parse("scene_holder['scene']=self").body[0]);edits['scene_cache']+=1
            save_fn=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='save')
            filtered=[]
            for statement in save_fn.body:
                if isinstance(statement,ast.If) and ast.unparse(statement.test)=="step == 1000 and (not getattr(self, 'geometry_failed', False))":
                    assert 'Review timeout' in ast.unparse(statement)
                    edits['wait_removed']+=1
                else: filtered.append(statement)
            save_fn.body=filtered
        if isinstance(node,ast.FunctionDef) and node.name=='report':
            node.body[:0]=ast.parse("if runtime_phase['mode']=='resume' and step==1001:\n    assert life.last_resume_receipt['next_iteration']==1001\n    save(run/'resume-state-verification.json',dict(life.last_resume_receipt,actual_first_resumed_iteration=step,prior_total_compute_seconds=runtime_phase['prior_compute_seconds'],offline_review_wall_seconds=runtime_phase['offline_review_wall_seconds']))").body
        if isinstance(node,ast.Try) and any(isinstance(h.type,ast.Name) and h.type.id=='ControlComplete' for h in node.handlers):
            call=node.body[0].value
            assert isinstance(call,ast.Call) and ast.unparse(call.func)=="namespace['training']"
            assert len(call.args)==8 and ast.literal_eval(call.args[5])==[] and ast.literal_eval(call.args[6]) is None
            call.args[5]=ast.parse('[1000]',mode='eval').body
            call.args[6]=ast.parse("str(run/'chkpnt1000.pth') if runtime_phase['mode']=='resume' else None",mode='eval').body
            node.handlers.insert(0,ast.ExceptHandler(type=ast.Name(id='StageReviewStop',ctx=ast.Load()),name='boundary',
                body=ast.parse("_stage_preview_after_checkpoint(boundary,scene_holder['scene'],life,locals(),runtime_phase)\nreturn").body))
            edits['native_call']+=1;edits['stop_handler']+=1
        new_body.append(node)
    function.body=new_body
    # Existing progress/logging seconds now include earlier computation phases;
    # offline-review wall time is separately recorded and never consumes RNG.
    class Elapsed(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node)=='time.monotonic() - started':
                return ast.copy_location(ast.parse('_compute_elapsed(runtime_phase,started)',mode='eval').body,node)
            return node
    tree=Elapsed().visit(tree);ast.fix_missing_locations(tree)
    # Keep first-phase metadata intact on resume. These fields describe this
    # same experiment; the native checkpoint is not an old person's weights.
    class Metadata(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='save' and node.args and ast.unparse(node.args[0])=="run / 'provenance.json'":
                node.args[0]=ast.parse("run/('resume-provenance.json' if runtime_phase['mode']=='resume' else 'provenance.json')",mode='eval').body
                for keyword in node.args[1].keywords:
                    if keyword.arg=='old_checkpoint_loaded':
                        keyword.value=ast.parse("runtime_phase['mode']=='resume'",mode='eval').body
                node.args[1].keywords.extend([
                    ast.keyword(arg='same_experiment_resume',value=ast.parse("runtime_phase['mode']=='resume'",mode='eval').body),
                    ast.keyword(arg='offline_review_wall_seconds',value=ast.parse("runtime_phase['offline_review_wall_seconds']",mode='eval').body)])
            if ast.unparse(node.func)=='status' and len(node.args)==2 and ast.literal_eval(node.args[0])=='training_starting':
                node.args[1]=ast.parse("1000 if runtime_phase['mode']=='resume' else 0",mode='eval').body
            if ast.unparse(node.func)=='save' and len(node.args)>=2:
                first=ast.unparse(node.args[0])
                if first in ("run / 'offset-initialization.json'", "run / 'training-cameras.json'", "run / 'effective-optimization.json'"):
                    node.func=ast.Name(id='_save_frozen_metadata',ctx=ast.Load())
                if first=="run / 'result.json'" and isinstance(node.args[1],ast.Call):
                    keywords={k.arg:k for k in node.args[1].keywords}
                    if 'steps' in keywords and 'optimizer_updates' not in keywords:
                        node.args[1].keywords.append(ast.keyword(arg='optimizer_updates',
                            value=ast.parse('opt.iterations-1',mode='eval').body))
            if ast.unparse(node.func)=="(run / 'upstream-training-function.py').write_text":
                node=ast.Call(func=ast.Name(id='_write_frozen_text',ctx=ast.Load()),
                    args=[ast.parse("run/'upstream-training-function.py'",mode='eval').body,*node.args],keywords=node.keywords)
            return node
    tree=Metadata().visit(tree);ast.fix_missing_locations(tree)
    assert all(v==1 for v in edits.values()),edits
    after_body=next(n for n in function.body if isinstance(n,ast.ClassDef) and n.name=='Body')
    assert ast.dump(after_body,include_attributes=False)==before_body_dump
    result=ast.unparse(tree)+'\n'
    assert 'Review timeout' not in result and '+ 600' not in result
    assert "'lr': 1e-06" in result
    assert "namespace['training'](dataset, opt, pipe, [], SNAPSHOTS, [1000]," in result
    assert 'compile(ast.Module(body=[fn]' in result
    compile(result,'<graywoman-adapter-with-staged-lifecycle>','exec')
    return result,dict(edits, original_Body_AST_exact=True, original_train_function_AST_not_edited=True,
                      original_total_clock=18000, snapshot1000_phase='after_density_adam_zero_grad')


def _compute_elapsed(phase, started):
    return float(phase['prior_compute_seconds'])+time.monotonic()-started


def _save_frozen_metadata(path,value):
    if Path(path).exists():
        assert gs.read(path)==value,'Reconstructed immutable metadata changed: '+str(path)
    else:
        gs.save(path,value)


def _write_frozen_text(path,value):
    if Path(path).exists():
        assert Path(path).read_text()==value,'Original training function source changed'
    else:
        Path(path).write_text(value)


def _verify_decision():
    request_path=OUT/'review-1000-request.json';decision_path=OUT/'review-1000-decision.json'
    request=gs.read(request_path);decision=gs.read(decision_path)
    assert decision['actual_visual_review'] is True and decision['decision']=='continue'
    assert decision['request_sha256']==gs.sha(request_path)
    assert decision['compute_phase_sha256']==gs.sha(OUT/'compute-phase-train.json')
    for key in ('world_ply_sha256','snapshot_result_sha256','native_checkpoint_sha256',
                'lifecycle_sidecar_sha256','lifecycle_manifest_sha256'):
        assert decision[key]==request[key],key
    for key,path in [('world_ply_sha256',OUT/'gaussian-run/step-001000/world.ply'),
                     ('snapshot_result_sha256',OUT/'gaussian-run/step-001000/result.json'),
                     ('native_checkpoint_sha256',OUT/'gaussian-run/chkpnt1000.pth'),
                     ('lifecycle_sidecar_sha256',OUT/'gaussian-run/chkpnt1000.pth.lifecycle.pt'),
                     ('lifecycle_manifest_sha256',OUT/'gaussian-run/chkpnt1000.pth.lifecycle.json')]:
        assert gs.sha(path)==request[key],path
    evidence=Path(decision['review_evidence_path'])
    assert gs.sha(evidence)==decision['review_evidence_sha256']
    review=gs.read(evidence)
    assert review['actual_visual_review'] is True and review['step']==1000
    assert review.get('request_sha256')==gs.sha(request_path) or (
        review.get('native_checkpoint_sha256')==request['native_checkpoint_sha256'] and
        review.get('world_ply_sha256')==request['world_ply_sha256']), 'Review evidence is not bound to this model'
    assert request['optimizer_updates']==1000 and request['checkpoint_phase']=='after_density_adam_zero_grad'
    return request,decision


def _begin_phase(mode):
    assert mode in ('train','resume')
    if mode=='train':
        assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
        return dict(mode=mode,prior_compute_seconds=0.,offline_review_wall_seconds=0.)
    assert gs.read(OUT/'process-exit.json')['exit_code']==0
    accounting=gs.read(OUT/'compute-phase-train.json')
    assert accounting['successful'] is True and accounting['state']=='checkpoint_ready_for_offline_review'
    assert accounting['mode']=='train' and accounting['prior_compute_seconds']==0.
    request,decision=_verify_decision()
    prior=float(accounting['cumulative_compute_seconds'])
    manifest=gs.read(OUT/'gaussian-run/chkpnt1000.pth.lifecycle.json')
    assert prior>=float(manifest['cumulative_compute_seconds'])
    assert 0<prior<COMPUTE_BUDGET
    offline=max(0.,time.time()-float(accounting['finished_unix_seconds']))
    return dict(mode=mode,prior_compute_seconds=prior,offline_review_wall_seconds=offline,
        review_request_sha256=gs.sha(OUT/'review-1000-request.json'),
        review_decision_sha256=gs.sha(OUT/'review-1000-decision.json'),
        snapshot_result_sha256=request['snapshot_result_sha256'])


def _actual_training_sources(local):
    paths={Path(__file__),Path(base.__file__),gs.HERE/'staged_training_lifecycle128.py',
        gs.UP/'train.py',gs.HERE/'native_target_cache124.py',gs.HERE/'paired_training_background124.py',
        gs.HERE/'smplify_direct_surface123.py',gs.HERE/'facial_geometry_boundary123.py',
        gs.HERE/'body_root_sh_colors123.py',gs.HERE/'derived_camera_contract124.py'}
    for name in ('official','arguments','rt'): paths.add(Path(local[name].__file__))
    functions=[local['camera'],local['named_pose_layout'],local['original_render'],
        local['l1_loss'],local['ssim'],local['SMPLXGaussianModel'].densify_and_split,
        local['SMPLXGaussianModel'].select_mesh_by_timestep,local['render_root_frame'],
        local['enable_clothing'],local['clothing_vertices'],local['clothing_geometry']]
    for fn in functions: paths.add(Path(inspect.getsourcefile(fn) or inspect.getfile(fn)))
    rasterizer=local['original_render'].__globals__['GaussianRasterizer']
    raster=sys.modules[rasterizer.__module__]
    paths.add(Path(raster.__file__));paths.add(Path(raster._C.__file__))
    for obj in vars(local['official']).values():
        if inspect.isfunction(obj) and obj.__module__.startswith('utils.'):
            paths.add(Path(inspect.getsourcefile(obj) or inspect.getfile(obj)))
    return sorted({p.resolve() for p in paths},key=str)


def _make_lifecycle(local,phase):
    from staged_training_lifecycle128 import StagedTrainingLifecycle
    assert gs.sha(gs.HERE/'staged_training_lifecycle128.py')==LIFECYCLE_SHA
    assert gs.sha(gs.UP/'train.py')==UPSTREAM_TRAIN_SHA
    counts=local['counts']
    def restore_counters(state):
        values=np.asarray(state['counts'],dtype=counts.dtype)
        assert values.shape==counts.shape
        counts[:]=values
    hashes=dict(local['plan']['source_hashes'])
    hashes.update({str(p):gs.sha(p) for p in _actual_training_sources(local)})
    identity=dict(schema='graywoman128-staged-original-18k-v1',subject=SUBJECT,
        plan_sha256=gs.sha(OUT/'plan.json'),fit_sha256=local['receipt']['fit_sha256'],
        training_inputs_sha256=local['receipt']['training_inputs_sha256'],
        reference_frame=local['layout']['reference_frame_id'],total_compute_budget_seconds=COMPUTE_BUDGET,
        full_schedule_steps=TOTAL,scientific_training_profile_changed=False)
    life=StagedTrainingLifecycle(model=local['model'],paired_background=local['paired_background'],
        camera_ids=[local['rows'][i]['frame'] for i in local['training_slots']],
        identity=identity,source_hashes=hashes,total_iterations=TOTAL,stop_iterations=[1000],optimization=local['opt'],
        counters_getter=lambda:dict(counts=counts.tolist()),counters_setter=restore_counters,
        compute_seconds_getter=lambda:time.monotonic()-local['started'])
    life._verify_sources()
    return life


def _stage_preview_after_checkpoint(boundary,scene,life,local,phase):
    import torch
    from staged_training_lifecycle128 import capture_rng_state,_equal,_snapshot_tree
    checkpoint=Path(boundary.checkpoint)
    assert checkpoint==OUT/'gaussian-run/chkpnt1000.pth'
    manifest_path=Path(boundary.manifest);manifest=gs.read(manifest_path)
    assert manifest['completed_iteration']==1000 and manifest['checkpoint_phase']=='after_density_adam_zero_grad'
    paths=[checkpoint,Path(str(checkpoint)+'.lifecycle.pt'),manifest_path]
    immutable={str(p):gs.sha(p) for p in paths}
    rng=capture_rng_state();capture=_snapshot_tree(local['model'].capture())
    paired_rng=life.paired.generator.get_state().clone()
    sampler=_snapshot_tree(life.loader.state_dict())
    counts=local['counts'].copy();paired_calls=life.paired.calls
    with torch.no_grad(): scene.save(1000)
    assert _equal(rng,capture_rng_state())
    assert _equal(paired_rng,life.paired.generator.get_state()) and _equal(sampler,life.loader.state_dict())
    life._validate_model(capture)
    assert np.array_equal(counts,local['counts']) and life.paired.calls==paired_calls==1000
    assert immutable=={str(p):gs.sha(p) for p in paths}
    snapshot=OUT/'gaussian-run/step-001000';result=gs.read(snapshot/'result.json')
    result.update(checkpoint_phase='after_density_adam_zero_grad',optimizer_updates=1000,
        global_iteration=1000,configured_schedule_steps=TOTAL,resumable_native_checkpoint=str(checkpoint),
        native_checkpoint_sha256=immutable[str(checkpoint)],offline_review=True,quality_accepted=False)
    gs.save(snapshot/'result.json',result)
    request=dict(updated=gs.now(),step=1000,optimizer_updates=1000,
        checkpoint_phase='after_density_adam_zero_grad',snapshot=str(snapshot),
        world_ply_sha256=gs.sha(snapshot/'world.ply'),snapshot_result_sha256=gs.sha(snapshot/'result.json'),
        native_checkpoint_sha256=immutable[str(checkpoint)],
        lifecycle_sidecar_sha256=immutable[str(paths[1])],lifecycle_manifest_sha256=immutable[str(manifest_path)],
        actual_visual_review=False,offline_review_no_live_optimizer=True,
        decision_required_fields=['actual_visual_review','decision','request_sha256','world_ply_sha256',
            'snapshot_result_sha256','native_checkpoint_sha256','lifecycle_sidecar_sha256',
            'lifecycle_manifest_sha256','compute_phase_sha256','review_evidence_path','review_evidence_sha256'],
        review_evidence_must_bind='request_sha256 OR both native_checkpoint_sha256 and world_ply_sha256',
        quality_accepted=False)
    assert not (OUT/'review-1000-request.json').exists()
    gs.save(OUT/'review-1000-request.json',request)
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state='checkpoint_ready_for_offline_review',
        pid=None,step=1000,optimizer_updates=1000,optimizer_waiting=False,training_running=False,
        resumable_checkpoint=str(checkpoint),seconds=_compute_elapsed(phase,local['started']),
        offline_review_wall_seconds=0.,quality_accepted=False))


def _adapter_namespace(phase,started):
    return dict(gs.__dict__,OUT=OUT,LIMIT=TOTAL,SNAPSHOTS=[10000,18000],suspended_baseline=lambda:None,
        _GRAY_RUNTIME_PHASE=phase,_GRAY_PROCESS_STARTED=started,
        _compute_elapsed=_compute_elapsed,_make_lifecycle=_make_lifecycle,
        _save_frozen_metadata=_save_frozen_metadata,_write_frozen_text=_write_frozen_text,
        _stage_preview_after_checkpoint=_stage_preview_after_checkpoint)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    plan=gs.read(FIT/'plan.json');review=gs.read(FIT/'conclusion.json')
    assert review['actual_visual_review'] and review['exploratory_gaussian_admitted']
    rows=plan['rows'];frames=[r['frame'] for r in rows];reference=plan['reference_frame']
    assert len(rows)==len(set(frames))==N and reference in frames and reference!=frames[0]
    assert all(r['wh']==[1520,2704] and np.array_equal(r['w2c'],np.eye(4)) for r in rows)
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        assert z['frame_ids'].astype(str).tolist()==frames and z['betas'].shape==(1,10)
        assert z['K_integer_centers'].shape==(3,3)
        assert all(np.array_equal(z['K_integer_centers'],np.asarray(r['K_integer_centers'])) for r in rows)
    fn,before,changed,connector_diff=adapted_prepare(reference,frames)
    expected=lifecycle_adapter(adapter_from_prepare(changed,reference))[0]
    baseline=gs.read(base.previous.BASELINE/'plan.json')
    fn()
    adapter=OUT/'adapter.py';original=base.add_alpha_export(adapter.read_text())
    code,receipt=lifecycle_adapter(original)
    assert code==expected
    adapter.write_text(code)
    (OUT/'graywoman-input-connectors.diff').write_text(connector_diff)
    (OUT/'staged-lifecycle-interface.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),code.splitlines(True),fromfile='unchanged-existing-profile',tofile='offline-review-lifecycle-only')))
    current=gs.read(OUT/'plan.json')
    assert all(current[k]==baseline[k] for k in PROFILE_KEYS)
    current.update(updated=gs.now(),subject=SUBJECT,label='灰毛衣新人物 · 半圈12帧 SH0',record_id='015',
        reference_frame=reference,initial_admission_max_step=1000,automatic_18k_continuation=False,
        fixed='Original SH0/ROI2x/paired Uniform/18k clock/20908 one per triangle/offset1e-6 and protected face-head-hands-feet.',
        only_parameter_change='New subject/input/reference connectors; training science unchanged. Engineering-only complete post-update checkpoint and offline review lifecycle.',
        budget_note='2400 seconds cumulative computation across phases; offline review wall time recorded separately. No600-second live wait.',
        lifecycle=dict(receipt,helper_sha256=LIFECYCLE_SHA,review_offline=True,first_checkpoint_optimizer_updates=1000,
            first_resume_iteration=1001,original_terminal18000_optimizer_updates=17999,total_compute_budget_seconds=COMPUTE_BUDGET),
        quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False)
    for path in list(current['source_hashes']):
        if Path(path).is_relative_to(OUT): current['source_hashes'][path]=gs.sha(path)
    for path in [Path(__file__),Path(base.__file__),gs.HERE/'staged_training_lifecycle128.py',gs.HERE/'body_root_sh_colors123.py']:
        current['source_hashes'][str(path)]=gs.sha(path)
    gs.save(OUT/'plan.json',current)
    assert all(current[k]==baseline[k] for k in PROFILE_KEYS)
    print('GRAYWOMAN12_LIFECYCLE_PREPARED',reference,flush=True)


def preflight_function():
    edits=[
        ('namespace=dict(gs.__dict__,OUT=OUT,LIMIT=LIMIT,suspended_baseline=lambda:None)',
         "namespace=_adapter_namespace(dict(mode='preflight',prior_compute_seconds=0.,offline_review_wall_seconds=0.),time.monotonic())"),
        ('dataset=ctx[\'Images\']();results=[];torch.cuda.reset_peak_memory_stats()',
         "dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()\n    from paired_training_background124 import PairedTrainingBackground\n    paired=PairedTrainingBackground(ctx['render'],ctx['masks'],[r['frame'] for r in ctx['rows']],ctx['target_cache'])"),
        ("output=ctx['render'](cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image",
         "output=paired(cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image"),
        ('assert not model.optimizer.state','assert paired.calls==12\n    assert not model.optimizer.state')]
    return reused(zoom.preflight,edits,dict(OUT=OUT,LIMIT=TOTAL,_adapter_namespace=_adapter_namespace),
                  'graywoman12-unchanged-GPU-preflight')


def preflight():
    function,diff=preflight_function()
    (OUT/'preflight-interface.diff').write_text(diff)
    function('graywoman-preflight')


def train(mode='train'):
    process_started=time.monotonic()
    phase=_begin_phase(mode)
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed']
    assert gs.read(OUT/'validity-review.json')['training_admitted']
    launch=OUT/('launch.json' if mode=='train' else 'resume-launch.json')
    assert not launch.exists(), 'No automatic retry or launch overwrite'
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remaining=COMPUTE_BUDGET-_compute_elapsed(phase,process_started)
    assert remaining>0
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Cumulative2400compute-seconds exhausted')))
    signal.alarm(math.ceil(remaining))
    gs.save(launch,dict(updated=gs.now(),pid=os.getpid(),command=sys.argv,
        proc_start_identity=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],single_gpu_lock=True,
        lifecycle_mode=mode,remaining_compute_seconds=remaining,phase=phase))
    namespace=_adapter_namespace(phase,process_started)
    successful=False
    try:
        exec(compile((OUT/'adapter.py').read_text(),str(OUT/'adapter.py'),'exec'),namespace)
        namespace['train']()
        successful=True
    finally:
        signal.alarm(0)
        progress=gs.read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
        accounting=OUT/f'compute-phase-{mode}.json'
        assert not accounting.exists()
        gs.save(accounting,dict(updated=gs.now(),mode=mode,successful=successful,
            state=progress.get('state'),phase_compute_seconds=time.monotonic()-process_started,
            prior_compute_seconds=phase['prior_compute_seconds'],
            cumulative_compute_seconds=_compute_elapsed(phase,process_started),
            offline_review_wall_seconds=phase['offline_review_wall_seconds'],finished_unix_seconds=time.time(),
            budget_seconds=COMPUTE_BUDGET,offline_review_excluded_from_compute=True,quality_accepted=False))
        lock.close()


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    frames=[f'frame_{i:06d}' for i in range(1,N+1)];reference=frames[5]
    _,_,changed,_=adapted_prepare(reference,frames)
    original=adapter_from_prepare(changed,reference)
    adapter,receipt=lifecycle_adapter(original)
    assert "receipt['frames'] == 12" in adapter and 'counts = np.zeros(12, dtype=int)' in adapter
    assert "row['wh']==[1520,2704]" in changed
    assert 'shared_laplacian_weight' in changed and 'shared_laplacian_wtwelve' not in changed
    preflight_function()
    baseline=gs.read(base.previous.BASELINE/'plan.json')
    assert all(k in baseline for k in PROFILE_KEYS)
    assert gs.sha(gs.UP/'train.py')==UPSTREAM_TRAIN_SHA
    assert gs.sha(gs.HERE/'staged_training_lifecycle128.py')==LIFECYCLE_SHA
    result=dict(updated=gs.now(),state='static_graywoman12_lifecycle_adapter_passed',helper_sha256=gs.sha(__file__),
        frames=N,native_wh=[1520,2704],record_id='015',reference_from_future_fit_plan=True,
        lifecycle=receipt,profile_copy_keys=list(PROFILE_KEYS),original_profile_AST_preserved=True,
        no_live_review_wait=True,post_update1000_updates=1000,terminal18000_global_updates=17999,
        cumulative_compute_budget_seconds=COMPUTE_BUDGET,offline_review_separate=True,
        preflight_function_compiled_without_execution=True,prepare_executed=False,fit_read=False,
        GPU_started=False,experiment_created=False,quality_accepted=False)
    gs.save(Path(__file__).with_suffix('.audit.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','prepare','preflight','train','resume'])
    mode=parser.parse_args().mode
    try:
        if mode in ('train','resume'):
            train(mode)
            path=OUT/('process-exit.json' if mode=='train' else 'resume-process-exit.json')
            assert not path.exists()
            gs.save(path,dict(updated=gs.now(),exit_code=0,state=gs.read(OUT/'progress.json').get('state')))
        else: globals()[mode]()
    except BaseException as error:
        if OUT.exists():
            path=OUT/(mode+'-failure.json')
            if not path.exists(): gs.save(path,dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise
