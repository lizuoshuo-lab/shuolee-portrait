"""022 source roster and frozen-fit diagnostic. No optimizer or CUDA import."""
import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from update_leafwoman129_state import C, FIT, INPUT, NATIVE, R, ROOT, S, T, V, atomic, read, sha

OUT = ROOT / 'leafwoman-fit40-132'
VIDEO = Path('D:/00/工作/数据集/video22.mp4')
VIDEO_SHA = 'c5aac80e574dba892137d59186ff17713ab0403ec1a969d788b054a364f68cd0'
ARCHIVE = FIT / 'selected-fit.npz'
SOURCE = NATIVE / 'source-pose-review/source-contacts.json'
POSE = NATIVE / 'source-pose-review/pose-groups.json'


def hash_stream(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _sha_chunks(stream)


def _sha_chunks(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def region(error, confidence, indices):
    e, c = error[:, indices], confidence[:, indices]
    observed = c > 0
    return dict(positive=int(observed.sum()), total=observed.size,
        median_px=float(np.median(e[observed])) if observed.any() else None,
        max_px=float(e[observed].max()) if observed.any() else None,
        rms_weighted_px=float(np.sqrt((c*c*e*e).sum()/(c*c).sum())) if observed.any() else None)


def prepare():
    source, pose, split = read(SOURCE), read(POSE), read(INPUT / 'plan.json')
    assert pose['actual_all57_source_visual_review'] is True
    assert source['video_sha256'] == pose['source_video_sha256'] == hash_stream(VIDEO) == VIDEO_SHA
    records = {row['frame']: row for row in source['records']}
    assert len(records) == 57
    with np.load(ARCHIVE, allow_pickle=False) as z:
        frames = z['frame_ids'].astype(str).tolist()
        assert frames == split['training_frames'] and len(set(frames)) == len(frames) == 24
        assert not set(frames) & set(split['heldout_frames'])
        assert z['body_pose_axis_angle'].shape == (1, 63) and z['betas'].shape == (1, 10)
        assert z['global_orient'].shape == (24, 3)
        targets = z['observed_native_keypoints']
        confidence = targets[..., 2]
        error = np.linalg.norm(z['joints_projected_native'] - targets[..., :2], axis=-1)
        assert np.isfinite(error).all() and np.isfinite(confidence).all()
        metrics = {name:region(error, confidence, ids) for name,ids in dict(
            inner_face=list(range(67,118)), ears=[17,18], shoulders=[2,5], elbows=[3,6],
            wrists=[4,7], upper_body=[2,3,4,5,6,7], knees=[10,13],
            feet=[11,14,19,20,21,22,23,24]).items()}
        assert metrics['ears']['positive']==48 and metrics['knees']['positive']==metrics['feet']['positive']==0
        key_examples={name:float(error[frames.index(f'frame_{i:06d}'),j]) for name,i,j in (
            ('frame27_right_ear',27,17),('frame39_right_elbow',39,3),('frame20_left_wrist',20,7))}
        K=z['K_integer_centers'].tolist()
        vertices,joints=z['vertices_camera'],z['joints_camera']
        ear_vertices={}
        for side,j in [('right',17),('left',18)]:
            distance=np.linalg.norm(vertices-joints[:,j:j+1,:],axis=-1)
            closest=np.argmin(distance,axis=1)
            ear_vertices[side]=dict(nearest_vertex_ids=np.unique(closest).tolist(),
                max_distance_in_saved_model_units=float(distance[np.arange(len(frames)),closest].max()))
    fit_result = read(FIT/'result.json')
    assert fit_result['external_world_camera_used'] is False and fit_result['single_shared_local_body_pose'] is True
    front = [dict(frame=f, pts_seconds=records[f]['time_seconds'], source_frame_index=records[f]['source_frame_index'],
                  source_sha256=records[f]['source_sha256'], mask_sha256=records[f]['source_mask_sha256']) for f in frames]
    assert all(a['pts_seconds'] < b['pts_seconds'] for a,b in zip(front,front[1:]))
    stamp=datetime.now(timezone.utc).isoformat()
    evidence=dict(updated=stamp,archive_sha256=sha(ARCHIVE),source_video_sha256=VIDEO_SHA,
        metrics=metrics,examples_native_px=key_examples,detector_residuals_not_anatomical_GT=True,
        constraints=dict(ear_mapping={'source_left_right':[3,4], 'body25_left_right':[18,17]},
            one_ear_point_per_side=True, per_ear_visibility_previously_used=False,
            saved_mesh_ear_anchor_replay=ear_vertices,
            use_face_contour=False,inner_face_point_count=51,body25_fixed_loss_zero=[1,9,12],
            one_shared_local_pose_across24=True,per_view_root_and_placement=True,
            external_scene_camera_used=False,fixed_conditional_K=K),
        fit_quality_accepted=False,actual_new_fit_started=False)
    plan=dict(updated=stamp,subject='022 / video22',state='front24_ready_rear16_missing',
        quotas={'front_arc':24,'rear_arc':16},front_arc=front,rear_arc=[],missing_rear_count=16,
        front_arc_includes_oblique_profiles=True, rear_means_visible_posterior_torso_not_late_timestamps=True,
        source_covers_rear=False,source_coverage_review='All57 reviewed; tree occludes back, late poses change; no verified posterior torso view.',
        retained_holdouts=split['heldout_frames'], source_video=str(VIDEO), source_video_sha256=VIDEO_SHA,
        previous_fit_sha256=sha(ARCHIVE),source_pose_review_sha256=sha(POSE),source_contacts_sha256=sha(SOURCE),
        input_handling=dict(ear='Gate each ear independently before confidence normalization. Use verified visible same-anatomy points; hair and occluded ears remain unobserved.',
            face_shape='Check side jaw/cheek/ear contour with reliable semantics and visibility before adding observations. Do not let ear error alone change the whole face.',
            body='Share updateable shape, preserve original anatomical priors. Separate demonstrably different arm poses; check shoulder/elbow/wrist projections. Cropped or occluded knees/feet stay unknown.',
            rear='Select across actual visible rear views of the same person and outfit, with original time/pixel mapping and separate camera provenance. No mirrored/generated replacement.'),
        full_roster_admitted=False,fit_started=False,gaussian_admitted=False,quality_accepted=False)
    atomic(OUT/'existing-fit-audit.json',evidence)
    atomic(OUT/'fit-plan.json',plan)
    print(json.dumps({'state':plan['state'],'front':len(front),'rear':0,'source_frames':len(records),
        'ear_median_px':metrics['ears']['median_px'],'body_median_px':metrics['upper_body']['median_px'],
        'output':str(OUT)}, ensure_ascii=False))


def publish():
    prepare()
    plan_path=OUT/'fit-plan.json'
    plan=read(plan_path); evidence=read(OUT/'existing-fit-audit.json')
    ear_path=OUT/'ear-audit/ear-visibility-review.json'; ear=read(ear_path)
    assert ear['actual_all24_native_panels_viewed'] is True
    assert [r['frame'] for r in ear['rows']]==[r['frame'] for r in plan['front_arc']]
    sides=('right','left')
    visibility=np.asarray([[r['ears'][s]['visibility']=='visible_external_ear' for s in sides] for r in ear['rows']])
    assert visibility.shape==(24,2) and visibility.sum()==19
    with np.load(ARCHIVE,allow_pickle=False) as z:
        source_targets=z['observed_native_keypoints']
        preview=source_targets.copy()
        preview[:,17:19,2]*=visibility
        assert np.array_equal(preview[...,:2],source_targets[...,:2])
        assert np.array_equal(preview[:,np.r_[0:17,19:118],2],source_targets[:,np.r_[0:17,19:118],2])
        assert int(np.count_nonzero(preview[:,17:19,2]))==19
        np.savez_compressed(OUT/'front24-ear-visibility-preview.npz',frame_ids=z['frame_ids'],
            observed_native_keypoints=preview,ear_candidate_visibility=visibility)
    plan.update(ear_review_sha256=sha(ear_path),visible_ear_candidates=19,unsupported_ear_observations_removed_in_preview=29,
        preview_sha256=sha(OUT/'front24-ear-visibility-preview.npz'),
        preview_is_old_normalized_scores_not_new_fit_input=True,independent_ear_point_semantics_verified=False,
        last_observed_windows_commit_available_gib=2.57)
    atomic(plan_path,plan)
    reports=(R/'video10-global-initializer/index.html',V/'diagnostics/video10-global-initializer/index.html',S/'diagnostics/video10-global-initializer/index.html')
    old=[p.read_text(encoding='utf-8-sig') for p in reports];assert len(set(old))==1
    begin='<!-- BEGIN VIDEO22 FIT40 -->';end='<!-- END VIDEO22 FIT40 -->'
    paragraphs=(
        '022 已登记前半弧24帧、背侧16帧的拟合配额。现有57张原生抽帧已实审；前半弧24帧含斜侧视角，背侧有效0帧。后段树干遮背且有抬手动作，不能按时间切出16张冒充背面。原留出帧不挪入拟合。',
        '当前只用51个内部脸点和每侧1个粗耳点，没有侧脸/下颌轮廓；旧24帧的48个耳点都有权重。耳点对检测目标的误差中位14.95 px，内部脸点3.20 px。逐耳核查得到19个可见候选，其余29个在预览中置零；19个仍需确认检测点与模型的解剖语义。第27帧近右耳相差30.94 px。',
        '肩肘腕正权重观测的残差中位18.77 px，第39帧右肘72.7 px、第20帧左腕54.8 px。现有24帧共享一组身体姿态，髋部没有进入损失，膝脚无可用观测。侧视网格倾斜还受相机坐标与真实重力方向未对齐的影响；拍摄视角的肘腕错位本身已可见。下一轮共享可更新体型，按真实动作分组约束肢体，不把裙下遮挡当作空白。',
        '仅完成输入与旧拟合核验，新拟合和高斯未启动。等待同人同服装、后背无遮挡的视角；原始输入、相机、旧拟合和模型保留。已清理228个可重建缓存或逐字节重复文件，约84 MB。复测Windows提交内存仅余2.57 GiB，磁盘清理不等于解决提交上限。')
    assets=[(OUT/'ear-audit'/f'ear-contact-{i:02d}.png',f'ear-contact-{i:02d}.png','原尺寸同帧：左原图、右耳点；粉色检测、蓝色投影，灰色不作为可见观测。') for i in range(1,5)]
    assets += [(FIT/'review/page-06.png','body-fit-review-06.png','旧拟合：原图、关键点、SMPL-X叠加与侧视网格。'),
               (NATIVE/'source-pose-review/full-context-14.jpg','late-coverage-14.jpg','视频末段原图与mask：人物后背仍被树干遮住。')]
    record=Path('D:/00/记录/022');assert record.is_dir()
    for src,name,_ in assets:
        for report in reports:
            dest=report.parent/'video22-fit40'/name;dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(src,dest);assert sha(src)==sha(dest)
        if name.startswith('ear-contact-'):
            assert ear['panel_sha256'][name]==sha(src)
            dest=record/('特征点_022_耳点核验_'+name)
            shutil.copyfile(src,dest);assert sha(src)==sha(dest)
    block=begin+'<section id="video22-fit40"><h2>022 · 前侧24＋背侧16拟合准备</h2>'
    block+=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    block+=''.join(f'<figure><figcaption>{html.escape(caption)}</figcaption><img loading="lazy" style="max-width:100%;height:auto" src="video22-fit40/{name}" alt="022观测核验"></figure>' for _,name,caption in assets)
    block+='</section>'+end
    original=old[0]
    if begin in original:
        start=original.index(begin);stop=original.index(end,start)+len(end)
        fresh=original[:start]+block+original[stop:]
    else:
        anchor='<!-- END DCBOX131-CONTINUATION -->';assert original.count(anchor)==1
        fresh=original.replace(anchor,anchor+block,1)
    for report in reports:atomic(report,fresh)
    assert len({sha(p) for p in reports})==1
    stamp=datetime.now(timezone.utc).isoformat()
    state=dict(updated=stamp,state=plan['state'],fit_quota_front=24,fit_quota_rear=16,
        actual_front_candidates=24,actual_rear_candidates=0,fit_plan_sha256=sha(plan_path),
        fit_plan_path=str(plan_path),ear_review_path=str(ear_path),ear_review_sha256=sha(ear_path),
        visibility_preview_only=True,training_running=False,worker_pid=None,quality_accepted=False,
        cleanup_bytes=84035082,cleanup_file_count=228,last_observed_commit_available_gib=2.57,
        report_url='diagnostics/video10-global-initializer/index.html#video22-fit40',
        next='Verify ear anchor semantics and visible body observations; add true rear footage before admitting 24+16. Do not run high-memory fit while Windows commit is near limit.')
    for path in (R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',C/'local-research-status.json'):
        value=read(path);value['video22_fit40']=state
        value['next']=state['next']
        if path.name in ('queue-state.json','local-research-status.json'):
            value.update(updated_at=stamp,state=state['state'],next_action=state['next'],
                training_running=False,worker_pid=None,pid=None,goal_status='active',execution_mode='goal')
        atomic(path,value)
    for folder in (V,S):
        path=folder/'latest-result.json';value=read(path);other={k:v for k,v in value.items() if k!='latestDiagnostic'}
        value['latestDiagnostic']=dict(updated=stamp,round='132',title='022 · 耳部和身体拟合核验 · 背侧缺16帧',
            summary=' '.join(paragraphs[:3]),reportUrl=state['report_url'],status=state['state'],qualityAccepted=False,goalComplete=False)
        atomic(path,value);assert {k:v for k,v in read(path).items() if k!='latestDiagnostic'}==other
    long_plan=T/'shuolee/long-term-plan.md'
    note=begin+'\n'+'\n'.join(paragraphs)+'\n'+end
    text=long_plan.read_text(encoding='utf-8-sig')
    if begin in text:text=text[:text.index(begin)]+note+text[text.index(end)+len(end):]
    else:text=text.rstrip()+'\n\n'+note+'\n'
    atomic(long_plan,text)
    atomic(OUT/'publication.json',dict(**state,report_sha256=sha(reports[0]),only_images_added_to_record=True))
    print('VIDEO22_FIT40_INPUT_AUDIT_PUBLISHED',sha(reports[0]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',nargs='?',choices=('prepare','publish'),default='prepare')
    {'prepare':prepare,'publish':publish}[parser.parse_args().mode]()
