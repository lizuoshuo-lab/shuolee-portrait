"""Independent saved-image and sealed-protocol audit for interval 31--60."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[2]
OUT = LAB/'outputs/dynamic-human-temporal-validation-006'
DATA = LAB/'data/dynamic-human/dynamic3dgaussians/data/basketball'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol = json.loads((OUT/'evaluation-protocol.json').read_text())
    results = json.loads((OUT/'validation-metrics.json').read_text())
    assert results['protocol_sha256'] == sha(OUT/'evaluation-protocol.json')
    for item in protocol['implementation_files']:
        assert sha(LAB/item['path']) == item['sha256'], item['path']
    metadata = json.loads((DATA/'test_meta.json').read_text())
    rows = results['per_view']
    indexed = {(r['mode'], r['frame'], r['camera']):r for r in rows}
    assert len(indexed) == len(rows) == 240
    audited, diffs, pairs = [], [], []
    for t in protocol['frames']:
        files = {int(fn.split('/')[0]):fn for fn in metadata['fn'][t]}
        for c in protocol['test_cameras']:
            x0,y0,x1,y1 = protocol['fixed_human_rois_xyxy'][str(c)]
            gt = np.array(Image.open(DATA/'ims'/files[c]).convert('RGB'), dtype=np.float64)/255
            for mode in protocol['methods']:
                path = OUT/f'renders/{mode}/{c}/{t:04d}.png'
                with Image.open(path) as image:
                    assert image.size == (640,360) and image.mode == 'RGB'
                    pred = np.array(image, dtype=np.float64)/255
                mse = np.mean((pred[y0:y1,x0:x1]-gt[y0:y1,x0:x1])**2)
                psnr = -10*math.log10(max(float(mse),1e-12))
                diff = abs(psnr-indexed[(mode,t,c)]['person_region']['psnr'])
                assert diff < .02, (mode,t,c,diff)
                diffs.append(diff)
                audited.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),png_psnr=psnr))
            a = indexed[('aliased',t,c)]['person_region']
            b = indexed[('frozen',t,c)]['person_region']
            pairs.append(dict(frame=t,camera=c,psnr_delta=b['psnr']-a['psnr'],ssim_delta=b['ssim']-a['ssim']))
    camera_summary = []
    for c in protocol['test_cameras']:
        subset = [r for r in pairs if r['camera']==c]
        camera_summary.append(dict(camera=c,paired_views=len(subset),
            mean_psnr_delta=float(np.mean([r['psnr_delta'] for r in subset])),
            mean_ssim_delta=float(np.mean([r['ssim_delta'] for r in subset])),
            psnr_improved=sum(r['psnr_delta']>0 for r in subset)))
    weights = []
    for mode in protocol['methods']:
        completion = json.loads((OUT/mode/'completion.json').read_text())
        for t in protocol['frames']:
            path = OUT/mode/f'frame-{t:04d}.pt'
            assert path.is_file()
            weights.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    review = dict(status='saved_images_and_sealed_sources_verified_visual_samples_reviewed',
        paired_views=120, pngs_verified=len(audited),
        png_vs_float_psnr_max_difference_db=max(diffs),
        protocol_sha256=sha(OUT/'evaluation-protocol.json'),
        metrics_sha256=sha(OUT/'validation-metrics.json'),
        script_sha256=sha(Path(__file__)),per_camera=camera_summary,
        all_view_psnr_improved=sum(r['psnr_delta']>0 for r in pairs),
        all_view_ssim_improved=sum(r['ssim_delta']>0 for r in pairs),
        worst_delta=min(pairs,key=lambda r:r['psnr_delta']),
        lowest_frozen_quality=min([r for r in rows if r['mode']=='frozen'],key=lambda r:r['person_region']['psnr']),
        visual_samples=[f'assets/camera{c}-frames031-045-060.png' for c in protocol['test_cameras']],
        visual_findings={
            'camera0':'Overall pose and limbs retained; frozen reference removes some colored contamination. Hands and clothing are still softer than observation.',
            'camera10':'Basketball shape and torso improve modestly; fine hands/face and shirt details remain soft.',
            'camera15':'Severe near-person shoulder/arm/shirt colored fragments persist at frame31 in both methods. More distant person is recognizable, but clothing remains noisy.',
            'camera30':'Pose and silhouette retained; sleeves and torso remain mottled. Frozen color reference does not restore garment texture.'},
        interpretation='Temporal color-reference repair transfers to this later interval, but does not solve near-person appearance.',
        limitations=['Same scene and camera identities; no unseen-scene or motion-forecast claim',
            'ROIs include background and multiple people in camera15; not segmentation ground truth',
            'Still-image visual sample covers 12 frame-camera pairs, not every motion transition',
            'Temporal image-change residual is not motion-compensated flicker'],
        per_pair=pairs,images=audited,checkpoints=weights)
    (OUT/'independent-audit.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:review[k] for k in ['status','pngs_verified','png_vs_float_psnr_max_difference_db','per_camera','worst_delta','all_view_psnr_improved']},indent=2))

if __name__=='__main__':
    main()
