"""Read-only held projections after all110nodes have been visually reviewed."""
from datetime import datetime,timezone
import time,json
import numpy as np
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,stats

OUT=REC/'grayT-near-scene-support-review'
NOTES={
 '5b73f914d652':('static_road_candidate','同一粗粒路面纹理区域，24节点可见相似局部结构；未发现跨物体跳接，亚像素中心精度未独立测量。'),
 '79137d8223ae':('static_road_candidate','路面裂缝与邻近浅色斑块结构在21节点中可辨，点位保持相近裂缝区域；物理测量真值未建立。'),
 '8036d2b7f840':('static_road_candidate','此前同帧地面对照，同一粗粒沥青区域；9节点已再次完整展示，不重复计算三维点。'),
 '6faeb26b4c3f':('vegetation_not_static_certified','10节点落在树枝叶片交界，风动/遮挡未排除；估计深度远，不能作为近人物静止参照。'),
 '249e00abab3a':('static_road_candidate','32节点在小深色裂纹与邻近棕色碎屑的路面区域。周围纹理相似，未观察跨物体跳接；碎屑自身静止性不能仅凭点名保证。'),
 '0552c9c0a50d':('static_road_candidate','14节点为粗粒路面纹理，较弱局部图案仍随视角连续；唯一材料身份和亚像素精度不具测量保证。')}

def main():
    started=time.monotonic()
    if (OUT/'result.json').exists():raise RuntimeError('Completed audit must not rerun.')
    selection=read(OUT/'selection.json');index=read(OUT/'index.json')
    # Actual review of all6sheets/all110nodes is recorded before accessing errors.
    review=dict(created=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,all110_nodes_actually_reviewed=True,
        classifications=[dict(id=im['frame'],category=NOTES[im['frame'][:12]][0],note=NOTES[im['frame'][:12]][1],
            observations=im['all_original_nodes'],gross_cross_object_jump_observed=False,subpixel_truth_certified=False) for im in index['images']],
        selection_sha256=sha(OUT/'selection.json'),index_sha256=sha(OUT/'index.json'),
        reviewed_before_error_read=True,no_observations_deleted=True,quality_accepted=False)
    save(OUT/'actual-review.json',review)
    files=[ROOT/'camera/held-evaluation.json',ROOT/'camera/solver-output.npz',ROOT/'data/training-inputs.json']
    hashes=[sha(p) for p in files];held={r['id']:r for r in read(files[0])['records']};z=np.load(files[1]);K=z['K'];E=z['w2c'];rows=read(files[2])['records']
    fs={};results=[];alltargets=[];static_targets=[]
    for pick in selection['selected']:
        tr=held[pick['id']];records=[];category=NOTES[tr['id'][:12]][0]
        assert tr['test_frames']==[i for i,j in tr['nodes'][1::2]]
        for (i,j),expected in zip(tr['nodes'][1::2],tr['errors']):
            if i not in fs:fs[i]=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy']
            uv=fs[i][j].astype(float);xyz=E[i,:3,:3]@np.asarray(tr['point'])+E[i,:3,3];p=K@xyz;pred=p[:2]/p[2]
            err=float(np.linalg.norm(pred-uv));assert abs(err-expected)<1e-7
            target=i+1 in [58,60,62,65]
            d=dict(frame=rows[i]['frame'],frame_number=i+1,xy=uv.tolist(),projected=pred.tolist(),
                residual=(pred-uv).tolist(),euclidean_error_px=err,depth=float(xyz[2]),same_frame_target=target)
            records.append(d)
            if target:
                alltargets.append(dict(track_id=tr['id'],category=category,**d))
                if category=='static_road_candidate':static_targets.append(dict(track_id=tr['id'],**d))
        results.append(dict(id=tr['id'],category=category,checks=records,all_original_checks=stats([d['euclidean_error_px'] for d in records]),
            target_checks=stats([d['euclidean_error_px'] for d in records if d['same_frame_target']]),
            source_point_unchanged=True,depth_ratios=pick['depth_ratio_by_frame']))
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,tracks=results,
        total_target_checks=stats([d['euclidean_error_px'] for d in alltargets]),
        road_candidate_target_checks=stats([d['euclidean_error_px'] for d in static_targets]),
        same_frame_road_errors={str(i):[dict(id=d['track_id'],error=d['euclidean_error_px'],xy=d['xy'],depth=d['depth']) for d in static_targets if d['frame_number']==i] for i in [58,60,62,65]},
        near_depth_track='249e00abab3a',near_depth_ratio=[1.1378101331401798,1.1612667550988895,1.1768886737825364],
        selection_uses_errors=False,no_error_based_deletion=True,all110nodes_visually_reviewed=True,
        physical_ground_truth_certified=False,camera_precision_certified=False,quality_accepted=False,
        sources=[dict(path=str(p),sha256=h) for p,h in zip(files,hashes)])
    assert result['seconds']<120 and all(sha(p)==h for p,h in zip(files,hashes))
    save(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['tracks','same_frame_road_errors','sources']},ensure_ascii=False))
    print(json.dumps([dict(id=t['id'][:12],category=t['category'],all=t['all_original_checks'],target=t['target_checks']) for t in results]))

if __name__=='__main__':main()
