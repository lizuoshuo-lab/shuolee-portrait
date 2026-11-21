"""Deliver a real pose51 export without relabeling it as improved training."""
from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime, timezone
from review_full_sequence123 import read,sha
from inspect_limb_reference123 import OUT,save

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
V=T/'viewer';S=Path('D:/00/工作/查看器');D=OUT/'selected-reference'
OLD='local-surface-123-grass-official-smplifyx-native1520-s01000'
ID=OLD+'-pose51'

def main(promote):
    result=read(D/'result.json');validation=read(D/'validation.json');digest=sha(D/'point_cloud.ply')
    assert result['seeded_binding_replay_verified'] and result['training_steps']==0
    assert validation['format_passed'] and validation['sha256']==digest
    model=read(V/f'model-{OLD}.json')
    model.update(label='草地白衣人物 · 睁眼候选原帧250动作 · 原生1000步',
        model=f'models/{ID}/point_cloud.ply',sha256=digest,
        referenceFrameId='frame_000051',initialView=50,
        selectedSourceFrameIndex=250,selectedSourceTimeSeconds=result['source_time_seconds'],
        poseSelectionReviewed=True,reposedNotRetrained=True,newTrainingSteps=0,
        qualityAccepted=False,deliveryAccepted=promote,
        cameraNote='点云已实际烘焙到第51张训练图（视频帧250）的动作，默认相机也对应这一帧。完整94相机保留；其他动作的原图不是当前静态点云的同姿态真值。',
        report='diagnostics/video10-global-initializer/index.html#official-reference-selection',
        status='已按拟合、代表性姿态和睁眼原图选第51帧，实际重导出；没有重训。高斯中心仍偏裸身、衣物扩散和肢体质量未通过。')
    assert model['cameras'][50]['img_name']=='frame_000051'
    if promote:
        ui=read(D/'viewer-ui-review.json');q=read(D/'quality-review.json');package=read(D/'package-check.json')
        assert ui['file_opened_through_UI'] and set(ui['modes'])=={'Gaussian','points'}
        assert ui['observed_count']==200000 and ui['camera_count']==94 and ui['source_sha256']==digest
        assert q['actual_same_camera_review'] and q['continuous_novel_views_reviewed']==36 and q['source_sha256']==digest
        assert package['model_sha256']==digest and package['cameras']==94
        for x in ui['screenshots']:assert sha(Path(x['path']))==x['sha256']
    for base in [V,S]:
        dest=base/'models'/ID;dest.mkdir(parents=True,exist_ok=True)
        for name in ['point_cloud.ply']:
            if not (dest/name).exists():shutil.copy2(D/name,dest/name)
            assert sha(dest/name)==digest
        save(dest/'cameras.json',model['cameras']);shutil.copy2(D/'cameras.json',dest/'camera-records.json')
        save(dest/'model-context.json',model);save(base/f'model-{ID}.json',model)
        if promote:
            js=base/'viewer.js';s=js.read_text(encoding='utf-8')
            if json.dumps(ID) not in s:
                assert s.count('const BUILTINS = [')==1
                js.write_text(s.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(ID)+', ',1),encoding='utf-8')
            latest=read(base/'latest-result.json')
            latest.update(round=ID,modelId=ID,sha256=digest,revision=digest,qualityAccepted=False,
                updated=datetime.now(timezone.utc).isoformat(),download_url=f'models/{ID}/scene.zip?download=1',
                report=model['report'],status=model['status'],referenceFrameId='frame_000051',reposedNotRetrained=True)
            save(base/'latest-result.json',latest)
    receipt=dict(modelId=ID,source_sha256=digest,points=200000,camera_count=94,
        ply=str(V/'models'/ID/'point_cloud.ply'),camera_file=str(V/'models'/ID/'cameras.json'),
        context=str(V/'models'/ID/'model-context.json'),zip=str(V/'models'/ID/'scene.zip'),
        viewer_url=f'http://127.0.0.1:8870/?model={ID}',reference_frame_id='frame_000051',
        initial_view=50,quality_accepted=False,local_UI_verified=promote,company00_verified=False)
    save(D/'staged-delivery.json',receipt);print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--promote-after-ui',action='store_true');main(p.parse_args().promote_after_ui)
