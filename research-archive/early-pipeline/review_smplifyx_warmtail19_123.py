"""Native saved-result comparison for the one registered curriculum control."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
from review_full_sequence123 import mesh_renderer, joined
from review_smplifyx_orientations123 import metrics
import control_smplifyx_warmtail19_123_v2 as control


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    b=control.b; p=b.verify_plan(); result=b.read(b.OUT/'result.json')
    assert result['unchanged_camera_shape_topology']
    out=b.OUT/'review';assert not out.exists();out.mkdir()
    old=b.load_npz(p['candidate_path']);new=b.load_npz(b.OUT/'traced-winner.npz')
    target=b.load_npz(b.OUT/'trajectory.npz')['observed_native_keypoints'][0]
    for key in ['betas','K_integer_centers','camera_translation','faces']:assert np.array_equal(old[key],new[key]),key
    before,after=metrics(old,target),metrics(new,target)
    protections={name:dict(delta_px=after['groups'][name]['median']-before['groups'][name]['median'],limit_px=limit,
        passed=after['groups'][name]['median']-before['groups'][name]['median']<=limit)
        for name,limit in [('body17',1),('feet6',1),('face51',.5)]}
    elbow=after['left_elbow_px']<=.5*before['left_elbow_px']
    row=next(r for r in b.read(b.common.BASE/'data/training-inputs.json')['records'] if r['frame']==b.FRAME)
    assert b.sha(row['image'])==row['source_sha256']
    raw=Image.open(row['image']).convert('RGB');crops=b.read(b.common.BASE/'review'/(b.FRAME+'_saved-review.json'))
    mesh,source=mesh_renderer();record=Path('/mnt/d/00/记录/demo');files=[]
    for region in ['body','face','left_hand','right_hand','feet']:
        crop=next(r['crop_integer_xyxy'] for r in crops['files'] if r['region']==region)
        im=raw.crop(crop); panels=[im]+[mesh(im,np.array(crop),fit) for fit in [old,new]]
        path=out/f'frame19_{region}_raw_five_stages_original_tail.png';joined(panels).save(path)
        dest=record/f'SMPL-X_已有姿态原版精修_frame_000019_{region}_原图_五阶段_末两阶段.png'
        assert not dest.exists();dest.write_bytes(path.read_bytes());assert b.sha(dest)==b.sha(path)
        files.append(dict(path=str(path),record_path=str(dest),sha256=b.sha(path),region=region,crop_integer_xyxy=crop,
            no_resampling=True,no_text_added=True,panels=['raw','original_five_stages','original_final_two_stages']))
    b.save(out/'comparison.json',dict(updated=b.now(),state='numeric_review_complete_visual_pending',before=before,after=after,
        protections=protections,left_elbow_halved=elbow,numeric_gate_passed=elbow and all(v['passed'] for v in protections.values()),
        files=files,candidate_sha256=b.sha(b.OUT/'traced-winner.npz'),baseline_sha256=b.sha(p['candidate_path']),
        renderer_sha256=b.sha(source),actual_visual_review=False,quality_accepted=False,goal_complete=False))
    print('WARMTAIL19_ELBOW',before['left_elbow_px'],after['left_elbow_px'],'PROTECTIONS',protections,flush=True)


if __name__=='__main__':main()
