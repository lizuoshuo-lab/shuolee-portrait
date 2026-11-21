"""One observation-derived hand-proximity abstention rule, not device segmentation."""
from pathlib import Path
import argparse
import os
import time
import traceback
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from train_demo_body_upstream123 import DEMO

BASE = DEMO / 'body-gaussian-smplify-eight-dc-reviewedfit'
OUT = BASE / 'hand-proximity-uncertainty-control'
OBS = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')


def prepare():
    assert not OUT.exists(), 'Do not repeat or replace this bounded rule.'
    selection = read(BASE / 'selection.json')['selected_frames']
    assert len(selection) == 8
    OUT.mkdir()
    files = [Path(__file__).resolve(), OBS, BASE/'selection.json', BASE/'data/training-inputs.json',
             BASE/'eight-view-surface-audit/cpu-result.json']
    save(OUT/'plan.json', dict(updated=now(), state='registered_before_execution', frames=selection,
        hypothesis='A reproducible uncertainty region around both hands may remove contradictory device/sleeve supervision without inferring device identity or treating missing body as background.',
        rule='For each anatomical side, form the native-pixel bounding box of the COCO wrist plus all existing RTMW hand points with confidence>=0.5; expand each side by ceil(projected elbow-to-wrist distance), then clip to native image bounds. Both hands use the same rule. No extra scale factor, morphology or fitted-mesh/render-error input.',
        arm_pairs=[[7,9],[8,10]], hand_ranges=[[91,112],[112,133]], confidence_min=0.5,
        missing_observation='Reject the candidate if elbow or wrist confidence is below0.5, coordinates are nonfinite or projected forearm length is below1pixel. No fallback or enlargement.',
        purpose='This is an abstention/unknown region, not a claim about equipment segmentation or occlusion depth. Original binary person masks remain unchanged.',
        evaluation='All fixed8 frames reviewed. Report loss of original foreground pixels, face ROI pixels and confident face landmark observations. Compare raw images with uncertainty overlays. Keep all original regions in later full-image quality evaluation; no scoring only on the easier retained region.',
        admission='Reject if visible device pixels previously labeled person remain outside uncertainty, if a complete hand/body region is mislabeled as reliable evidence, or face supervision becomes unusable. No automatic training or mask adoption from this probe.',
        risk='May remove useful hands, sleeves or face evidence, especially foreshortened forearms; may miss extended devices. 2D keypoint confidence is not physical truth. No threshold sweep to fit this scene.',
        budget=dict(device='CPU', frames=8, seconds=120, candidates=1, optimization_updates=0),
        source_hashes={str(p):sha(p) for p in files}, training_started=False, quality_accepted=False))
    print('HAND_UNCERTAINTY_REGISTERED', flush=True)


def run():
    assert not (OUT/'result.json').exists(), 'Already completed; do not repeat.'
    import cv2
    import signal
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU120s budget')))
    signal.alarm(120)
    plan = read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():
        assert sha(p) == h, p
    z = dict(np.load(OBS, allow_pickle=False))
    assert z['keypoints'].shape[1:] == (133,2)
    lookup = {str(f):i for i,f in enumerate(z['frame_ids'])}
    rows = {r['frame']:r for r in read(BASE/'data/training-inputs.json')['records']}
    crop_by_frame = {r['frame']:r['box'] for r in read(BASE/'eight-view-surface-audit/cpu-result.json')['records']}
    (OUT/'validity').mkdir()
    records = []
    start = time.monotonic()
    for frame in plan['frames']:
        row = rows[frame]; idx = lookup[frame]
        assert sha(row['image']) == row['source_sha256'] and sha(row['mask']) == row['mask_sha256']
        rgb = np.array(Image.open(row['image']).convert('RGB'))
        old = np.array(Image.open(row['mask']).convert('L'))
        h,w = old.shape
        assert [w,h] == row['wh'] and np.isin(old,[0,255]).all()
        person = old == 255
        kp, scores = z['keypoints'][idx], z['scores'][idx]
        unknown = np.zeros((h,w), dtype=bool)
        hands = []
        for (elbow,wrist),(lo,hi) in zip(plan['arm_pairs'],plan['hand_ranges']):
            assert min(scores[elbow],scores[wrist]) >= plan['confidence_min'], (frame,elbow,wrist,'low confidence')
            assert np.isfinite(kp[[elbow,wrist]]).all()
            distance = float(np.linalg.norm(kp[elbow]-kp[wrist]))
            assert distance >= 1., (frame,'forearm projection degenerate')
            ids = [j for j in range(lo,hi) if scores[j] >= plan['confidence_min'] and np.isfinite(kp[j]).all()]
            pts = kp[[wrist]+ids]
            margin = int(np.ceil(distance))
            low = np.floor(pts.min(0)-margin).astype(int)
            high = np.ceil(pts.max(0)+margin).astype(int)+1
            x0,y0 = np.maximum(low,[0,0]); x1,y1 = np.minimum(high,[w,h])
            assert 0<=x0<x1<=w and 0<=y0<y1<=h
            unknown[y0:y1,x0:x1] = True
            hands.append(dict(elbow=elbow,wrist=wrist,projected_forearm_px=distance,
                confidence=[float(scores[elbow]),float(scores[wrist])],hand_ids=ids,
                unknown_rectangle_xyxy=[int(x0),int(y0),int(x1),int(y1)],margin_px=margin))
        valid = ~unknown
        ssim_valid = ~cv2.dilate(unknown.astype('uint8'), np.ones((11,11),np.uint8)).astype(bool)
        face = np.zeros((h,w),dtype=bool)
        fbox = np.rint(row['face_box_xyxy']).astype(int)
        face[fbox[1]:fbox[3],fbox[0]:fbox[2]] = True
        face_ids = [j for j in range(23,91) if scores[j]>=plan['confidence_min'] and np.isfinite(kp[j]).all()]
        lost_ids=[]
        for j in face_ids:
            x,y=np.rint(kp[j]).astype(int)
            if not(0<=x<w and 0<=y<h) or unknown[y,x]: lost_ids.append(j)
        dst=OUT/'validity'/f'{frame}.png'
        Image.fromarray(valid.astype('uint8')*255).save(dst)
        overlay=rgb.copy()
        overlay[unknown]=(rgb[unknown].astype(float)*.45+np.array([255,170,0])*.55).astype('uint8')
        labels=np.zeros_like(old); labels[person&valid]=255; labels[unknown]=128
        known_person=rgb*(person&valid)[:,:,None]
        crop=crop_by_frame[frame]
        panels=[Image.fromarray(v).convert('RGB').crop(crop) for v in [rgb,old,overlay,labels,known_person]]
        cw,ch=panels[0].size; sheet=Image.new('RGB',(5*cw,ch))
        for col,img in enumerate(panels):sheet.paste(img,(col*cw,0))
        picture=Path('/mnt/d/00/记录/demo')/f'mask_手邻域未知区_{frame}_原图_原mask_未知区_标签_有效人像.png'
        sheet.save(picture)
        records.append(dict(frame=frame,source=row['image'],source_sha256=row['source_sha256'],
            original_mask_sha256=row['mask_sha256'],validity=str(dst),validity_sha256=sha(dst),
            native_wh=[w,h],hands=hands,unknown_pixels=int(unknown.sum()),
            original_person_pixels=int(person.sum()),person_pixels_ignored=int((person&unknown).sum()),
            person_ignored_fraction=float((person&unknown).sum()/person.sum()),
            face_ROI_pixels=int(face.sum()),face_ROI_ignored_fraction=float((face&unknown).sum()/max(1,face.sum())),
            confident_face_landmarks=len(face_ids),ignored_face_landmarks=lost_ids,
            ssim_valid_windows=int(ssim_valid.sum()),
            picture=str(picture),picture_sha256=sha(picture),native_crop=crop,
            resampling=False,text_added=False))
        save(OUT/'progress.json',dict(updated=now(),state='CPU_running',pid=os.getpid(),done=len(records)))
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    result=dict(updated=now(),state='completed_pending_fixed8_review',seconds=time.monotonic()-start,
        records=records,real_occlusion_semantics_proven=False,original_masks_modified=False,
        training_started=False,adopted=False,quality_accepted=False)
    save(OUT/'result.json',result)
    save(OUT/'progress.json',dict(updated=now(),state=result['state'],pid=None,done=len(records)))
    signal.alarm(0)
    print('HAND_UNCERTAINTY_COMPLETE',result['seconds'],flush=True)
    for r in records:print(r['frame'],r['person_ignored_fraction'],r['face_ROI_ignored_fraction'],len(r['ignored_face_landmarks']),r['confident_face_landmarks'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    try:
        prepare() if args.mode=='prepare' else run()
    except BaseException as error:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),retry=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None))
        raise
