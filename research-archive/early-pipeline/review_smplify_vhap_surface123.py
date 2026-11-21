"""CPU-only native crops for every frame of the SMPLify-X -> VHAP trial."""
from pathlib import Path
import os,time,json
import numpy as np
from PIL import Image
from review_full_sequence123 import mesh_renderer,joined,read,sha
from review_demo_hand_data59 import save,stats,load

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
RUN=ROOT/'smplify-vhap-surface'
SOURCE=ROOT/'fullbody-smplifyx/full-sequence'
DATA=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
RECORD=Path('/mnt/d/00/记录/demo')
RECORD_PREFIX='SMPL-X_VHAP外扩'
CANDIDATE_LABEL='VHAP_shared_offset'
CONTROL_RUN=None
CONTROL_LABEL=None
RECORD_SUFFIX=None
CHECKPOINT_EPOCH=30
CONTROL_EPOCH=30
COMPARISON_DIR='comparison'
REQUIRE_COMPLETE=True


def validate_update_counts(final, current):
    """Reject partial or miscounted runs before creating any review artifacts."""
    assert final['state']==('complete_pending_visual_review' if REQUIRE_COMPLETE else 'saved_checkpoint_only')
    assert final['frames']==current['frames']==59
    counts={
        'offset_initialization':final.get('offset_initialization_updates',0),
        'sequential_refinement':final.get('sequential_refinement_updates',0),
        'global_refinement':final.get('global_refinement_updates',59*CHECKPOINT_EPOCH),
    }
    assert counts['offset_initialization'] in (0,500)
    assert counts['sequential_refinement'] in (0,59*50)
    assert counts['global_refinement']==59*CHECKPOINT_EPOCH
    assert final['steps']==sum(counts.values()),'Run update counts do not match completed stages.'
    assert current['steps']==final['steps'],'Final mesh was not saved after all stages.'
    return counts


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    initial=read(RUN/'initial/result.json');current=read(RUN/f'epoch-{CHECKPOINT_EPOCH:02d}/result.json')
    if REQUIRE_COMPLETE:
        final=read(RUN/'result.json')
    else:
        progress=read(RUN/'progress.json')
        init=read(RUN/'offset-initialization.json')
        final=dict(state='saved_checkpoint_only',frames=current['frames'],steps=current['steps'],
            offset_initialization_updates=init['steps'],global_refinement_updates=59*CHECKPOINT_EPOCH,
            elapsed_seconds=progress.get('seconds',progress.get('elapsed_seconds')))
    update_counts=validate_update_counts(final,current)
    out=RUN/COMPARISON_DIR;out.mkdir(exist_ok=True)
    mesh=load(RUN/f'epoch-{CHECKPOINT_EPOCH:02d}/tracked-smplx-params.npz')
    rows=read(DATA/'data/inputs.json')['records']
    assert list(mesh['frame_ids'])==[r['frame'] for r in rows]
    control=None
    if CONTROL_RUN is not None:
        assert CONTROL_LABEL and CONTROL_RUN != RUN
        control=load(CONTROL_RUN/f'epoch-{CONTROL_EPOCH:02d}/tracked-smplx-params.npz')
        assert np.array_equal(control['frame_ids'],mesh['frame_ids'])
        assert np.array_equal(control['faces'],mesh['faces'])
        assert np.array_equal(control['K_integer_centers'],mesh['K_integer_centers'])
    render,source=mesh_renderer();began=time.monotonic();results=[]
    for i,row in enumerate(rows):
        name=row['frame'];receipt=out/(name+'.json')
        if receipt.exists():
            old=read(receipt)
            assert all(sha(x['record_path'])==x['sha256'] for x in old['images'])
            results.append(old);continue
        prior=read(SOURCE/'review'/(name+'_saved-review.json'))
        baseline=load(SOURCE/'frames'/name/'winner-camera-space.npz')
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB')
        candidate=dict(vertices_camera=mesh['vertices_camera'][i:i+1],faces=mesh['faces'],K_integer_centers=mesh['K_integer_centers'][i])
        models=[baseline]
        labels=['raw','SMPLify-X']
        if control is not None:
            models.append(dict(vertices_camera=control['vertices_camera'][i:i+1],faces=control['faces'],K_integer_centers=control['K_integer_centers'][i]))
            labels.append(CONTROL_LABEL)
        models.append(candidate);labels.append(CANDIDATE_LABEL)
        overlays=[render(raw,np.array([0,0,*raw.size]),x) for x in models]
        images=[]
        for region in ('body','face','left_hand','right_hand','feet'):
            item=next(x for x in prior['files'] if x['region']==region);box=item['crop_integer_xyxy']
            panels=[im.crop(box) for im in [raw,*overlays]]
            suffix=RECORD_SUFFIX or ('原图_SMPLify初始化_VHAP' if control is None else '原图_SMPLify初始化_原版平滑控制_逐帧精修')
            path=RECORD/f'{RECORD_PREFIX}_{name}_{region}_{suffix}.png'
            assert not path.exists(),str(path)
            joined(panels).save(path)
            with Image.open(path) as check:
                assert np.array_equal(np.array(check)[:,:panels[0].width],np.array(panels[0]))
            images.append(dict(region=region,record_path=str(path),sha256=sha(path),crop_integer_xyxy=box,
                panel_wh=list(panels[0].size),panels=labels,resampling=False,text_added=False))
        result=dict(frame_id=name,initial=initial['records'][i],candidate=current['records'][i],images=images,
            vertices_max_move_m=float(np.linalg.norm(candidate['vertices_camera'][0]-baseline['vertices_camera'][0],axis=-1).max()),
            renderer_sha256=sha(source),quality_accepted=False)
        assert initial['records'][i]['frame']==current['records'][i]['frame']==name
        save(receipt,result);results.append(result)
        print(json.dumps(dict(frame=name,completed=len(results))),flush=True)
    summary={}
    for metric in ('mask_iou','target_uncovered','mesh_outside','elbow_px'):
        a=np.array([r['initial'][metric] for r in results]);b=np.array([r['candidate'][metric] for r in results])
        summary[metric]=dict(initial=stats(a),candidate=stats(b),increased=int((b>a).sum()),decreased=int((b<a).sum()))
    for region in ('body','face','left_hand','right_hand','feet'):
        a=np.array([r['initial']['groups'][region]['median'] for r in results]);b=np.array([r['candidate']['groups'][region]['median'] for r in results])
        summary[region]=dict(initial=stats(a),candidate=stats(b),improved=int((b<a).sum()),worsened=int((b>a).sum()))
    result=dict(state='complete_pending_visual_review' if REQUIRE_COMPLETE else 'checkpoint_review_pending',frames=59,images=295,per_frame=results,summary=summary,
        offset_mm=current['offset_mm'],fit_seconds=final['elapsed_seconds'],review_seconds=time.monotonic()-began,
        actual_visual_review=False,quality_accepted=False,gaussian_training_started=False,
        checkpoint_epoch=CHECKPOINT_EPOCH,partial_checkpoint=not REQUIRE_COMPLETE,
        checkpoint_sha256=sha(RUN/f'epoch-{CHECKPOINT_EPOCH:02d}/tracked-smplx-params.npz'),
        stage_update_counts=update_counts,
        matched_control_checkpoint_sha256=sha(CONTROL_RUN/f'epoch-{CONTROL_EPOCH:02d}/tracked-smplx-params.npz') if CONTROL_RUN else None,
        limits=['Vertex displacement includes every enabled parameter group; it is not necessarily only the shared offset.',
                'Detector residuals and silhouette overlap are not independent physical shape truth.',
                'All frames are included; no result filtering.'])
    save(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('state','frames','summary','offset_mm')}),flush=True)


if __name__=='__main__':main()
