"""Fresh original and interpolated2x MSV2 training using the unchanged4x mask loss."""
from pathlib import Path
import argparse
import importlib.util
import subprocess
import sys

BASE=Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104')
OUT=Path('/mnt/d/00/记录/2026-11-04_MSV2_mask原始_插值2倍_4倍对比')
RUNS={1:Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-native-20261104'),
      2:Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-interpolate2x-20261104')}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def helper():
    h=load('same_mask_resolution',Path(__file__).with_name('transparent_resolution_comparison.py'))
    h.BASE=BASE;h.OUT=OUT;h.RUNS=RUNS;return h

def create(scale):
    h=helper();h.create(scale)
    run=RUNS[scale];ex=load('resolution_contract',run/'experiment.py')
    inputs=ex.read(run/'training-inputs.json')
    for row in inputs['records']:
        row['upsampling_scale']=scale
        row['input_linear_scale']=scale
        assert row['wh']==[476*scale,960*scale]
    ex.save(run/'training-inputs.json',inputs)
    contract=ex.read(run/'input-contract.json')
    contract.update(boundary_radius_native_pixels=8,boundary_radius_training_pixels=8*scale,
        boundary_radius_scales_with_input=True,source_masks_unchanged=True)
    ex.save(run/'input-contract.json',contract)
    base_plan=ex.read(BASE/'plan.json');plan=ex.read(run/'plan.json')
    plan.update(human_direction='mask发小润，然后训练原版和二倍，二倍用插值',
        loss=base_plan['loss'],boundary_band=f'8 native pixels = {8*scale} training pixels; same source labels',
        fixed=['MSV2 model and optimizer source','59 source frames and camera extrinsics',
          'untrained200000-point initialization and seed6060','foreground RGB loss',
          'balanced alpha BCE0.2 plus boundary-negative BCE0.1','MSV2 schedule and18000 labels'],
        comparability='Only input sampling and matching K change. The4x reference uses SwinIR;2x uses Lanczos. Masks use bilinear enlargement. No new loss coefficient or model change.',
        single_main_variable='Original input vs Lanczos2x vs existing SwinIR4x under the same mask penalty',
        model_architecture_changed=False,msv2_schedule_changed=False,postprocessing_pruning=False,
        source_masks_changed=False,learned_gaussians_reused=False,quality_accepted=False,
        user_selected_visible_improvement=False,user_requested_same_mask_configuration=True,
        pretraining_evidence=str(run/'rgba-probe.json'),
        final_gate='Review all59 original-resolution cameras and12 orbit views; compare face exterior noise, person/face coverage and source-image error. Complete18000 labels; no automatic quality promotion.')
    plan.pop('documentation_clarification',None)
    ex.save(run/'plan.json',plan)
    parity=ex.read(run/'configuration-parity.json')
    parity.update(model_source_unchanged=True,optimizer_and_schedule_unchanged=True,
        mask_loss_byte_identical=True,boundary_radius_native_pixels=8,
        only_changes=['input sampling and matching K','resolution metadata and output paths'])
    assert ex.sha(run/'transparent_hooks.py')==ex.sha(BASE/'transparent_hooks.py')
    assert ex.sha(run/'source/msv2/train.py')==ex.sha(BASE/'source/msv2/train.py')
    assert ex.sha(run/'source/scene/gaussian_model.py')==ex.sha(BASE/'source/scene/gaussian_model.py')
    ex.save(run/'configuration-parity.json',parity)
    ex.save(OUT/'comparison-plan.json',dict(created=ex.now(),requested=True,
        runs={str(k):str(v) for k,v in RUNS.items()}|{'4':str(BASE)},
        methods=['original','Lanczos2x','SwinIR4x'],model_unchanged=True,
        mask_penalty_identical=True,boundary_width_native_pixels=8,from_scratch=True,
        initial_points=200000,seed=6060,iteration_labels=18000,quality_accepted=False,
        comparison='All59 native-reference cameras and12 orbit cameras;4x also changes SR method',
        previous_unpenalized_runs_are_comparators_only=True))
    print('SAME_MASK_INPUT_CONTRACT_PASSED',scale,8*scale,flush=True)

def probe(scale):
    helper().probe(scale)

def run(scale):
    root=RUNS[scale]
    for command,name in [([str(Path(__file__).with_name('audit_demo_original_pixels.py'))],'original-pixel-audit.log'),
                         ([str(root/'experiment.py'),'preflight'],'preflight.log'),
                         ([str(Path(__file__)),'probe',str(scale)],'rgba-probe.log'),
                         ([str(root/'experiment.py'),'train'],'train.log')]:
        print('START',scale,name,flush=True)
        with (root/name).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',scale,name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe','run']);p.add_argument('scale',type=int,choices=[1,2])
    a=p.parse_args();globals()[a.action](a.scale)
