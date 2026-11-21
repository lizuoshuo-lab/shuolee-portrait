"""Read-only temporal renders after the current training process has finished."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB / 'outputs/dynamic-human-street-temporal-012'
BODY = LAB / 'outputs/dynamic-human-street-smpl-joint-011/articulated'
CONTROL = LAB / 'outputs/dynamic-human-street-joint-009/resume-10000-expandable'


def status(value, **fields):
    result = dict(status=value, pid=os.getpid(), updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    temporary = OUT / 'state.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(OUT / 'state.json')
    print(json.dumps(result), flush=True)


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    args = parser.parse_args()
    assert not (OUT / 'state.json').exists(), 'Existing process or attempt: inspect instead of duplicating'
    assert (OUT / 'protocol.json').exists()
    expected = b'/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_street_smpl.py'
    process = Path(f'/proc/{args.after_pid}/cmdline')
    if process.exists():
        assert expected in process.read_bytes().split(b'\0'), 'Process identity changed'
    status('waiting_for_verified_training_process_exit', training_pid=args.after_pid)
    deadline = time.monotonic() + 10800
    while process.exists() and expected in process.read_bytes().split(b'\0'):
        if time.monotonic() > deadline:
            raise TimeoutError('Training still active; no overlapping GPU job started')
        time.sleep(15)
    for run in (CONTROL, BODY):
        receipt = json.loads((run / 'completion.json').read_text())
        assert receipt['optimizer_updates'] == 30000 and len(receipt['views']) == 105
    # Heavy imports and CUDA activity start only after completed training.
    import gc
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    import train_street_deformable as helper
    from resume_street_deformable import cuda_tree, equal_tree
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    helper.install_capture(helper.ArgoVerseCameraData)
    helper.install_training_only(helper.DrivingDataset)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert len(protocol['images']) == 80
    helper.write_json(OUT / 'render-protocol.json', dict(status='sealed_before_temporal_rendering',
        script_sha256=file_sha(Path(__file__)), windows_sha256=file_sha(OUT / 'protocol.json'),
        checkpoints={name: file_sha(run / 'checkpoint-30000.pt') for name, run in [('deformable', CONTROL), ('articulated', BODY)]},
        optimized_steps=0, dataset='Shared original baseline dataset; body state loaded from its checkpoint, no Gaussian initialization repeated.',
        verification='Every saved model tensor restored exactly; overlapping heldout renders must match prior final renders within one8bit level.'))
    status('loading_shared_original_dataset')
    dataset = helper.DrivingDataset(OmegaConf.load(CONTROL / 'config.yaml').data)
    completed = []
    for name, run in [('deformable', CONTROL), ('articulated', BODY)]:
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        assert not (directory / 'completion.json').exists()
        status('restoring_model', branch=name)
        cfg = OmegaConf.load(run / 'config.yaml')
        trainer = helper.MultiTrainer(**cfg.trainer, num_timesteps=156,
            model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)), num_train_images=987,
            num_full_images=1092, test_set_indices=dataset.test_timesteps,
            scene_aabb=dataset.aabb.reshape(2, 3), device=torch.device('cuda'))
        saved = torch.load(run / 'checkpoint-30000.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['step'] == 29999
        trainer.load_state_dict(cuda_tree(saved), load_only_model=True, strict=True)
        equal_tree(saved, trainer.state_dict(only_model=True), f'{name}.all_model_tensors')
        del saved
        trainer.completed_updates = 30000
        trainer.render_each_class = False
        trainer.set_eval()
        gc.collect()
        torch.cuda.empty_cache()
        rows, comparisons, temporal = [], [], []
        window_map = {(r['frame'], r['camera']): w for w in protocol['windows'] for r in w['images']}
        previous = {}
        status('rendering_sequence', branch=name, images=80)
        with torch.no_grad():
            for item in protocol['images']:
                frame, camera = item['frame'], item['camera']
                stem = f'frame{frame:03d}-camera{camera}'
                info, cam = dataset.full_image_set.get_image(frame * 7 + camera, 1.)
                info, cam = helper.to_cuda(info), helper.to_cuda(cam)
                result = trainer(info, cam, novel_view=True)
                pred = result['rgb'].clamp(0, 1).cpu().numpy()
                target = info['pixels'].cpu().numpy()
                expected_target = np.asarray(Image.open(OUT / 'ground-truth' / (stem + '.png')))
                actual_target = np.uint8(np.clip(target, 0, 1) * 255 + .5)
                assert np.array_equal(expected_target, actual_target), stem
                helper.save_rgb(directory / (stem + '-prediction.png'), pred)
                window = window_map[(frame, camera)]
                x0, y0, x1, y1 = window['fixed_person_box_union']
                rows.append(dict(frame=frame, camera=camera, heldout=item['heldout'], window=window['name'],
                    full_image=helper.score(target, pred),
                    fixed_union_region=helper.score(target[y0:y1, x0:x1], pred[y0:y1, x0:x1])))
                if item['heldout']:
                    old = np.asarray(Image.open(run / 'evaluation/final' / (stem + '-prediction.png')), dtype=np.int16)
                    new = np.uint8(np.clip(pred, 0, 1) * 255 + .5).astype(np.int16)
                    difference = np.abs(old - new)
                    assert difference.max() <= 1, (name, stem, int(difference.max()))
                    comparisons.append(dict(frame=frame, camera=camera, max_8bit_difference=int(difference.max()),
                        mean_8bit_difference=float(difference.mean())))
                if window['name'] in previous:
                    prev_frame, prev_target, prev_pred = previous[window['name']]
                    assert prev_frame + 1 == frame
                    residual = np.abs((pred - prev_pred) - (target - prev_target))
                    temporal.append(dict(window=window['name'], previous_frame=prev_frame, frame=frame, camera=camera,
                        heldout_transition=item['heldout'] or prev_frame % 10 == 0,
                        full_image_temporal_difference_residual=float(residual.mean()),
                        fixed_union_temporal_difference_residual=float(residual[y0:y1, x0:x1].mean())))
                previous[window['name']] = frame, target, pred
                del result, info, cam
        assert len(rows) == 80 and len(temporal) == 76
        assert len(comparisons) == sum(r['heldout'] for r in protocol['images'])
        helper.write_json(directory / 'completion.json', dict(status='all80_temporal_views_rendered_reload_agreement_passed',
            branch=name, optimized_steps=0, rows=rows, reload_comparisons=comparisons,
            temporal_differences=temporal, temporal_definition=protocol['measurements'][1],
            boundaries=protocol['boundaries'], visual_review_pending=True))
        completed.append(name)
        del trainer, previous
        gc.collect()
        torch.cuda.empty_cache()
    status('both_models_rendered_independent_image_and_video_review_pending', completed=completed)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_needs_inspection', error=repr(error))
        raise
