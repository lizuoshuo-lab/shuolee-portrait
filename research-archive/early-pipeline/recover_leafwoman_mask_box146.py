"""Re-prompt the existing SAM2 on isolated vertical box truncation frames."""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from official_smplifyx_entry123 import sha
from replay_leafwoman_arm138 import OUT as ARM, mounted
from run_leafwoman_gvhmr143 import CONTACTS, SOURCE_OBS, WSL
from update_leafwoman129_state import atomic, read


OUT = ARM.with_name('smplify-front24-145')


def main():
    assert WSL
    root = OUT/'source-leg-visibility146'
    out = root/'mask-recovery'
    assert not out.exists()
    receipts = read(root/'source-audit.json')
    observations = read(root/'raw-lower-observations.json')
    assert sha(CONTACTS) == receipts['contacts_sha256'] == observations['contacts_sha256']
    assert sha(SOURCE_OBS) == observations['source_obs_sha256']
    source = read(CONTACTS)['records']
    width, height = receipts['full_native_rasters']
    box_y2 = np.asarray([row['initial_detector_box'][3] for row in source])
    slots = [i for i in range(1, len(source)-1)
             if box_y2[i-1] == box_y2[i+1] == height and box_y2[i] < height-32]
    assert [source[i]['frame'] for i in slots] == ['frame_000052', 'frame_000055']
    project = mounted('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
    assets = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
    weight = assets/'sam2/sam2.1_hiera_tiny.pt'
    source_assets = read(project/'configs/server-assets.json')['files']
    manifest = next(x for x in source_assets if x['path']=='sam2/sam2.1_hiera_tiny.pt')
    assert sha(weight) == manifest['sha256'] and weight.stat().st_size == manifest['bytes']
    native_fit = read(OUT.parent/'fit-plan.json')
    fit_ids = [r['frame'] for r in native_fit['front_arc']]
    assert not ({source[i]['frame'] for i in slots} & (set(fit_ids) | set(native_fit['retained_holdouts'])))
    atomic(out/'plan.json', dict(runner_sha256=sha(__file__), source_contacts_sha256=sha(CONTACTS),
        source_observations_sha256=sha(SOURCE_OBS), sam2_checkpoint_sha256=manifest['sha256'],
        trigger='An isolated box bottom is more than the existing 32px crop margin above the raster bottom while both neighbors reach it.',
        change='Set only box bottom to the raster height. Same RGB, original torso prompt, original SAM2 model and settings.',
        frames=[source[i]['frame'] for i in slots], front_fit_or_holdouts_changed=False,
        no_detector_or_fitter_started=True, quality_accepted=False))
    sys.path.insert(0, str(project/'third_party/sam2'))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 4*2**30
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(weight)
    model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(weight), device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    with np.load(SOURCE_OBS, allow_pickle=False) as z:
        assert z['frame_ids'].tolist() == [r['frame'] for r in source]
        keypoints, boxes = z['keypoints'], z['boxes']
    started = time.monotonic()
    results = []
    with torch.inference_mode():
        for i in slots:
            row = source[i]
            image, previous_mask = Path(row['source_image']), Path(row['source_mask'])
            assert sha(image) == row['source_sha256'] and sha(previous_mask) == row['source_mask_sha256']
            with Image.open(image) as im:
                rgb = np.array(im.convert('RGB'))
            with Image.open(previous_mask) as im:
                before = np.array(im.convert('L')) >= 128
            assert rgb.shape == (height,width,3) and before.shape == (height,width)
            box = boxes[i].copy()
            assert np.array_equal(box, row['initial_detector_box'])
            box[3] = height
            prompt = keypoints[i, [5,6,11,12]].mean(0)[None]
            predictor.set_image(rgb)
            masks, score, _ = predictor.predict(box=box, point_coords=prompt, point_labels=np.ones(1,dtype=np.int32),
                                                multimask_output=False)
            after = masks[0].astype(bool)
            assert after.shape == before.shape and .005 < after.mean() < .95 and score[0] >= .5
            save = out/(row['frame']+'-candidate-mask.png')
            Image.fromarray(after.astype(np.uint8)*255).save(save)
            bottom_before = int(np.nonzero(before)[0].max()+1)
            bottom_after = int(np.nonzero(after)[0].max()+1)
            cutoff = round(float(boxes[i,3]))
            results.append(dict(frame=row['frame'], source_index=row['source_frame_index'], mask=save.name,
                sha256=sha(save), old_box=boxes[i].tolist(), candidate_box=box.tolist(), mask_bottom_old=bottom_before,
                mask_bottom_candidate=bottom_after, old_fg_pixels=int(before.sum()), candidate_fg_pixels=int(after.sum()),
                newly_covered_below_old_box=int((after[cutoff:] & ~before[cutoff:]).sum()),
                common_old_upper_pixels=int((before[:cutoff] & after[:cutoff]).sum()),
                original_inputs_unchanged=True, quality_accepted=False))
            print('MASK_RECOVERY022', row['frame'], bottom_before, bottom_after, flush=True)
    atomic(out/'result.json', dict(plan_sha256=sha(out/'plan.json'), results=results, seconds=time.monotonic()-started,
        actual_sam2_used=True, candidate_visual_review_pending=True, fitting_or_training_started=False, quality_accepted=False))


if __name__ == '__main__':
    main()
