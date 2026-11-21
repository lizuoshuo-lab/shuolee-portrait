"""Stage additional named official fits through the existing unified Gaussian entry."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_016_006 as base


common=base.common
COMMIT=base.COMMIT
need,read,sha,pin=base.need,base.read,base.sha,base.pin
DRIVER='7aa4d97afa7177f5bd326a9ed24bbd8612d4b9af45495c2d712de35ee8f96a57'
HELPER='5195801e98475198d74a17e3cfe0bcf76b5d2d2520d4d8f75378553fe766e93c'
SUBJECTS={
    '001': dict(parent='shuolee-smpl-coverage-001',arc='half',frames=24,reference='frame_000056',
        result='f02291d953b2dc3df627c613c2e80a7a968330d9c41cce674015776068815e67',
        video='32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c',
        camera='28c838033eb699c2497ba292793516596f1dfaba7fb8d4ae64fc9a11316a65b5',
        holdouts=('frame_000013','frame_000033','frame_000049','frame_000065')),
    '005': dict(parent='shuolee-smpl-sixworker-ffprobe-005',arc='half',frames=23,reference='frame_000020',
        result='05656f2e9084ff7b32261571a309d25c7a6e296d3f382895617b12f40124d53a',
        video='846558b3a023650b3dccd1fa55da4dc3cfde596ea87de09ab43ab494f5306e14',
        camera='65bc4ccfbf6d53f57cf7f47cc782ffe793d9000cab3580d7aa896c00d764fdd4',
        holdouts=('frame_000011','frame_000027','frame_000042','frame_000055')),
    '008': dict(parent='shuolee-smpl-visibility-008',arc='full',frames=19,reference='frame_000004',
        result='4f2605fca0b011798308fa414de604c3c3dc00dfcb5d4779d31d5338dc89aa84',
        video='871d55a34424c942dab7c51755dc24bc1cf8eb0a649df2b422bbdabc800cd79b',
        camera='b4e7a76076da210b57371ccc30d338655ddda2732b53b8823d607e6ab7385741',
        holdouts=('frame_000015','frame_000043','frame_000071','frame_000098')),
}
ORDER=tuple(SUBJECTS)


def verify_driver():
    pin(Path(base.__file__).resolve(),DRIVER)
    pin(Path(common.__file__).resolve(),HELPER)


def replay_selector(old,receipt,cfg,selection,expected_candidates):
    import numpy as np
    seq,selector,_,_=base.imports()
    data=seq.load_inputs(old/'data/inputs.json',old/'data/observations.npz',
        camera_length_contract=cfg['camera_length_contract'],reference_frame=selection['reference_frame'],
        capture_layout=cfg['capture_layout'])
    fit_ids=data['scope']['fit_frame_ids']
    holds=data['scope']['retained_holdouts']
    need(len(fit_ids)==expected_candidates and holds==selection['retained_holdouts']
         and not set(fit_ids)&set(holds) and data['confidence']==receipt['source']['confidence']
         and all(not data['targets'][data['ids'].index(f)].any() for f in holds),
         'Frozen source-side visibility, fitting roster, or heldout exclusion differs.')
    historical=receipt['source']['input_files']
    visibility=Path(cfg['capture_layout']['visibility_annotations']['path'])
    for path in (old/'data/inputs.json',old/'data/observations.npz',visibility):
        digest=historical[str(path)]
        pin(path,digest)
        need(data['bound'][str(path)]==digest,'Retained input provenance changed.')
    with np.load(old/'fit/fitted.npz',allow_pickle=False) as archive:
        beta=archive['shared_betas'].copy()
    winners,reports,pins={}, {}, {}
    for frame in fit_ids:
        folder=old/'fit/frames'/frame
        winner,report=folder/'winner-camera-space.npz',folder/'result.json'
        with np.load(winner,allow_pickle=False) as archive:
            winners[frame]=dict(archive)
        reports[frame]=read(report)
        seq.check_result(reports[frame],data['rows'][data['ids'].index(frame)])
        pins.update({str(winner):sha(winner),str(report):sha(report)})
    replay=selector.select_frames(data,winners,scope=data['scope'],reports=reports,shared_betas=beta)
    need(replay==selection,'The stock selector changed the 008 archive, so exact fit reuse is blocked.')
    return pins


def prepare(subject):
    verify_driver()
    common.git_state(COMMIT)
    seq,selector,profile,verify=base.imports()
    spec=SUBJECTS[subject]
    old=common.LAB/spec['parent']/'run'
    result_path=old/'result.json'
    pin(result_path,spec['result'])
    result=read(result_path)
    receipt=read(old/'fit/official-fit-result.json')
    selection=read(old/'fit/selection.json')
    metadata=read(old/'run.json')
    cfg=read(old/'config.json')
    checked=verify(old)
    named=read(old/'data/training-inputs.json')['records']
    selected=selection['selected_frames']
    holds=selection['retained_holdouts']
    ref=selection['reference_frame']
    need(result['state']=='completed' and result['format_passed'] is True
         and result['from_scratch'] is True and receipt['new_fitting_started'] is True
         and receipt['selection']==selection and checked['fit_sha256']==receipt['fit_sha256']
         and metadata['assets']==str(common.ASSETS)
         and metadata['video_sha256']==spec['video'] and metadata['camera_config_sha256']==spec['camera']
         and cfg['capture_layout']['video_sha256']==spec['video']
         and selection['arc']==spec['arc'] and tuple(holds)==spec['holdouts']
         and ref==result['reference_frame_id']==spec['reference']
         and len(set(selected))==result['frames']==spec['frames']
         and [ref]+[f for f in selected if f!=ref]==[row['frame'] for row in named]
         and not set(selected)&set(holds) and selection['coverage_basis']==selector.SOURCE_BIN_COVERAGE_BASIS,
         'The exact archived fit, source camera, or protected named roster differs.')
    pins={str(old/n):sha(old/n) for n in ('run.json','config.json','result.json',*base.COPY)}
    src_video=Path(metadata['video'])
    src_camera=old.parent/'camera.json'
    for path,digest in ((src_video,spec['video']),(src_camera,spec['camera'])):
        pin(path,digest)
        pins[str(path)]=digest
    selector_source=[(p,h) for p,h in receipt['source']['input_files'].items()
                     if Path(p).name=='stable_selection.py']
    [old_selector]=selector_source
    pin(Path(old_selector[0]),old_selector[1])
    need(sha(Path(selector.__file__).resolve())==old_selector[1],
         'The unified selector differs from the original archived selector.')
    pins.update(receipt['source']['input_files'])
    if subject=='008':
        pins.update(replay_selector(old,receipt,cfg,selection,60))
    for path,digest in pins.items():
        pin(Path(path),digest)
    effective=profile()
    need(effective['opacity_reset_steps']==[1800,3600,5400,7200]
         and effective['iterations']==10000 and effective['sh_degree']==0
         and effective['sampling_policy']=='seeded_equal_occupied_source_bin_cycles',
         'The integrated Gaussian profile changed.')
    plan=dict(schema='shuolee-unified-reused-fit-v1',subject=subject,candidate_commit=COMMIT,
        baseline_run=str(old),baseline_result_sha256=spec['result'],
        source_video_sha256=spec['video'],camera_config_sha256=spec['camera'],
        selector_replayed_from_original_candidates=subject=='008',
        selector_source_identity_verified=True,selected_frames=selected,retained_holdouts=holds,
        reference_frame=ref,coverage=selection['coverage'],full_bin_coverage=selection['full_bin_coverage'],
        copied_fitting_input_from=str(old),fit_reused=True,fit_calls=0,cold_video_clock=False,
        fresh_gaussian_optimizer=True,gaussian_from_scratch=True,
        code_sha256={n:sha(common.PROJECT/n) for n in common.SOURCE_FILES},
        input_sha256=pins,asset_sha256=common.official_pins(),runtime=common.cpu_runtime(common.clean_env()),
        script_sha256=sha(Path(__file__).resolve()),launcher_helper_sha256=HELPER,
        reused_driver_sha256=DRIVER)
    return plan,None,(old,receipt,selection,metadata,cfg),effective


def launch(subject):
    verify_driver()
    index=ORDER.index(subject)
    if index:
        prior=common.root_for(ORDER[index-1])
        done=read(prior/'wrapper-terminal.json')
        need(done['state']=='completed' and done['candidate_commit']==COMMIT,
             'The preceding unified queue item must complete delivery first.')
        pin(prior/'run/result.json',done['result_sha256'])
    # Retain the exact launch implementation, changing only its worker script path.
    invoke=FunctionType(base.execute.__code__,dict(vars(base),__file__=__file__))
    return invoke(subject)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch','worker'))
    parser.add_argument('subject',choices=SUBJECTS)
    args=parser.parse_args()
    verify_driver()
    if args.action=='worker':
        base.worker(args.subject)
    elif args.action=='launch':
        raise SystemExit(launch(args.subject))
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        state=prepare(args.subject)
        if args.action=='stage':
            base.stage(args.subject,state)
        else:
            plan=state[0]
            print(json.dumps(dict(state='fit_reuse_verified',subject=args.subject,
                selected_frames=plan['selected_frames'],coverage=plan['coverage'],
                selector_replayed=plan['selector_replayed_from_original_candidates'])),flush=True)


if __name__=='__main__':
    main()
