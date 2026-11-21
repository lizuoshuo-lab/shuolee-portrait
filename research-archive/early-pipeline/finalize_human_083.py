"""Finalize only already inspected images, standalone UI and canonical report."""
import json,hashlib,shutil,re,html
from pathlib import Path
from datetime import datetime
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V=B.parent/'viewer';W=Path(__file__).parent;R=B/'current/person-static-human-surface-init-fresh-083'
P=V/'body-human-surface-current';ID='whole-body-static-human-init'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def main():
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    d=read(R/'review/visual-decision.json');ui=read(R/'review/viewer-interaction-review.json')
    result=read(R/'gaussian-run/result.json');pair=read(R/'review/paired-transfer-comparison.json')
    receipt=read(V/f'models/{ID}/delivery.json');meta=read(V/f'model-{ID}.json')
    assert result['steps']==18000 and result['frames_used']==175 and result['from_scratch']
    assert d['actually_inspected'] and ui['manual_open_file'] and ui['gaussian_mode_seen'] and ui['point_mode_seen']
    assert ui['points_displayed']==result['points']==receipt['points']
    assert sha(Path(receipt['download_path']))==receipt['sha256']==meta['sha256']
    assert receipt['compatibility_passed'] and read(V/'latest-result.json')['round']=='083'
    T=np.asarray(receipt['world_to_viewer']['rotation']);origin=np.asarray(receipt['world_to_viewer']['origin_world'])
    rows=read(R/'data/training-inputs.json')['records'];rng=np.random.default_rng(83);maximum=0.
    assert len(rows)==len(meta['cameras'])==175
    for row,cam in zip(rows,meta['cameras']):
        E,K=np.asarray(row['w2c']),np.asarray(row['K'])
        assert (row['source_frame_index'],row['frame'],row['camera_kind'])==(cam['raw_source_frame'],cam['img_name'],cam['camera_kind'])
        assert row['camera_kind']=='fresh_full_scene_sfm'
        Kv=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
        np.testing.assert_allclose(K,Kv,rtol=0,atol=1e-10)
        xc=rng.uniform(-1,1,(64,3));xc[:,2]=rng.uniform(1,10,64)
        xw=(xc-E[:3,3])@E[:3,:3];xv=(xw-origin)@T.T
        nv=(xv-np.asarray(cam['position']))@np.asarray(cam['rotation'])
        a,b=xc@K.T,nv@Kv.T;maximum=max(maximum,float(abs(a[:,:2]/a[:,2:]-b[:,:2]/b[:,2:]).max()))
    assert maximum<1e-8
    save(R/'gaussian-run/viewer-camera-roundtrip.json',dict(cameras=175,rays_per_camera=64,max_projection_difference_px=maximum,passed=True,claim_boundary='Export transform consistency only; no independent camera GT'))
    screenshots=[R/'review'/n for n in ['viewer-preset.png','viewer-standalone-gaussians.png','viewer-standalone-points.png']]
    assert all(p.is_file() and p.stat().st_size>10000 for p in screenshots)
    receipt.update(local_standalone_open_verified=True,local_preset_open_verified=True,
        local_viewer_proof=[dict(path=str(p),sha256=sha(p)) for p in screenshots],quality_accepted=d['quality_accepted'],pc00_opened_this_round=False)
    save(V/f'models/{ID}/delivery.json',receipt)
    registry=read(B/'gaussian-deliveries.json');registry['rounds']['083'].update(local_standalone_open_verified=True,
        local_gaussian_and_centers_verified=True,quality_accepted=d['quality_accepted'],local_viewer_proof=receipt['local_viewer_proof']);registry['updated']=now;save(B/'gaussian-deliveries.json',registry)
    names=['head-033.jpg','head-069.jpg','head-099.jpg','body-099.jpg','novel-above25-bg0.25.jpg','novel-below25-bg0.25.jpg',
        'continuous-same-camera.mp4','same-camera-result.json','paired-transfer-comparison.json','visual-decision.json']
    for n in names:shutil.copy2(R/'review'/n,P/n)
    shutil.copy2(R/'review/depth/result.json',P/'depth-result.json')
    shutil.copy2(R/'gaussian-run/review-current.jpg',P/'training-review.jpg')
    progress=read(R/'gaussian-run/progress.json');progress.update(round='083',updated=now,state='reviewed',running=False,new_model_published=True,quality_accepted=d['quality_accepted'],review_step=18000);save(P/'progress.json',progress)
    rel='body-human-surface-current/';esc=html.escape
    fig=lambda n,c:f'<figure><a href="{rel}{n}"><img src="{rel}{n}" alt="{esc(c)}"></a><figcaption>{esc(c)}</figcaption></figure>'
    low=pair['groups']['face_lower']['metrics']['rgb_center'];up=pair['groups']['face_upper']['metrics']['rgb_center']
    block='<h3>本轮训练效果与验收</h3>'
    block+=f'<p><a href="./?model={ID}">打开083高斯</a> · <a href="models/{ID}/{Path(receipt["download_path"]).name}?download=1">下载本轮标准PLY</a> · <a href="./?model=whole-body-static-correspondence">077对照</a></p>'
    block+=fig('head-099.jpg','同一原视频第100帧与原像素裁片：左实拍，中077，右083。没有额外二维对齐。')
    block+=fig('body-099.jpg','同相机完整人物，比较衣服、手脚与轮廓。')
    block+=f'<p>{esc(d["observed_conclusion"])}</p><p>固定共同对应的下半脸框投影误差中位数：{low["before_median"]:.3f} → {low["after_median"]:.3f}像素；上半脸框：{up["before_median"]:.3f} → {up["after_median"]:.3f}像素。采用完整去畸变画幅像素，同一特征可能参与多个点对；这些自动区域含下颌或邻近衣领，不是完整的鼻口真值。</p>'
    block+=fig('novel-below25-bg0.25.jpg','同一下方25度检查相机：左077，右083。此方向没有实拍真值。')
    block+=f'<p>{esc(d["novel_conclusion"])}</p><p>{esc(d["mechanism_conclusion"])}</p><video controls preload="metadata" style="width:100%;max-width:1040px" src="{rel}continuous-same-camera.mp4"></video>'
    block+=f'<p><strong>下一步：</strong>{esc(d["next_step"])}</p><p>标准PLY的位置、协方差与全部球谐系数共同转换到+Y向上，已经通过原生／重载对照及本地“打开文件”的高斯、点云两种模式。格式兼容不代表重建质量通过；本轮没有在00部署。</p>'
    page=V/'body-smplx.html';body=page.read_text(encoding='utf8')
    start,end='<!-- HUMAN_SURFACE_083_BEGIN -->','<!-- HUMAN_SURFACE_083_END -->'
    a,b=body.index(start),body.index(end);part=body[a:b]
    part=re.sub(r'<p class="notice">.*?</p>',f'<p class="notice"><strong>083已完成从头18,000步，初始200,000点，最终{result["points"]:,}点。</strong>{esc(d["viewer_status"])}</p>',part,count=1,flags=re.S)
    assert '<h3>本轮训练效果与验收</h3>' not in part
    part=part.replace('<h3>把人体先验落实到初始点的位置</h3>',block+'<h3>把人体先验落实到初始点的位置</h3>',1)
    part=part.replace('完成后仍须检查脸部重复结构、衣物与手脚、固定俯仰路径，并通过标准PLY重新读取和文件导入检查。','本轮已检查六个实拍方向、上下25度方向与连续路径的六个固定位置，以及标准PLY重新读取和文件导入。未逐帧审阅完整150帧路径。')
    part=re.sub(r'<script>\(\(\)=>\{let previousImage=-1;.*?</script>','',part,flags=re.S)
    part=part.replace('<p id="human083-progress">读取本地训练状态……</p>',f'<p id="human083-progress">083：18,000／18,000步，最终{result["points"]:,}点。训练与格式核验完成；质量结论见上。</p>')
    part=part.replace('value="0"></progress>','value="18000"></progress>')
    part=part.replace('id="human083-review" hidden','id="human083-review"')
    part=part.replace('<img id="human083-review-image" alt=', '<img id="human083-review-image" src="body-human-surface-current/training-review.jpg?step=18000" alt=')
    part=part.replace('<figcaption id="human083-review-caption"></figcaption>','<figcaption id="human083-review-caption">18,000步原生实拍相机输入／渲染；不等于未知视角质量通过。</figcaption>')
    body=body[:a]+part+body[b:];page.write_text(body,encoding='utf8')
    pub=read(R/'publication.json');pub.update(updated=now,published_gaussian_round='083',new_ply_published=True,browser_verified=False,quality_accepted=d['quality_accepted'],report_url='http://127.0.0.1:8870/body-smplx.html#human-surface-init');save(R/'publication.json',pub)
    status=read(B/'work-status.json');status.update(updated=now,current_phase='083 completed and reviewed',current_decision=d['summary'],local_training_running=False)
    status['latest_local_execution'].update(round='083',state='completed_reviewed',running=False,steps=18000)
    status['latest_completed_static_delivery']=dict(round='083',run=str(R),steps=18000,points=result['points'],download_url=receipt['download_url'],viewer_url=receipt['viewer_url'],local_ply=receipt['download_path'],quality_accepted=d['quality_accepted'],compatibility_passed=True,local_viewer_passed=True,pc00_opened_this_round=False,training_cameras='175 whole-image SfM+BA,0 interpolated',review=str(R/'review/visual-decision.json'))
    status['next_bounded_work']=dict(type='diagnostic before any new training',scope=d['next_step'],same_static_goal=True,new_training_authorized_by_gate=False,gaussian_optimizer_updates=0,final_goal_complete=False,latest_visual_evidence=str(R/'review/visual-decision.json'))
    status['iteration_policy'].update(new_training_ready=False,required_next_protocol=d['next_step']);save(B/'work-status.json',status)
    plan=B/'long-term-plan.md';s=plan.read_text(encoding='utf8');marker='<!-- CAMERA_REGISTRATION_CURRENT_END -->';assert marker in s
    if '<!-- HUMAN_083_RESULT -->' not in s:s=s.replace(marker,'<!-- HUMAN_083_RESULT -->\n'+d['summary']+'\n'+d['next_step']+'\n'+marker,1)
    plan.write_text(s,encoding='utf8')
    completion=dict(updated=now,round='083',workflow_complete=True,training_steps=18000,initial_points=200000,final_points=result['points'],from_scratch=True,whole_scene_camera_frames=175,interpolated_camera_frames=0,representation='shared_static_world',quality_accepted=d['quality_accepted'],root_cause_established=False,all_numeric_gates_passed=pair['all_numeric_gates_passed'],standard_ply=receipt['download_path'],ply_sha256=receipt['sha256'],compatibility_passed=True,local_standalone_viewer_passed=True,pc00_deployed=False,camera_roundtrip_px=maximum,report_url=pub['report_url'],deck=pub['deck'],deck_sha256=pub['deck_sha256'],goal_complete=False,goal_turn_classification='progress',repeated_blocking_condition_count=0,next_step=d['next_step'])
    save(R/'completion.json',completion)
    for n in ['finalize_human_083.py','compare_transfer_083.py','review_person_083.sh','publish_human_init_083.py','commit_human_deck_083.py']:shutil.copy2(W/n,R/'source-final'/n)
    save(B/'current/human-head-surface-initialization-082/completion.json',dict(updated=now,round='082',workflow_complete=True,type='geometry diagnostic',individual_shape_accepted=False,new_gaussian=False,decision=str(B/'current/human-head-surface-initialization-082/decision.json'),followed_by='083 fresh neutral initialization experiment',goal_complete=False))
    print(json.dumps(dict(finalized=True,quality_accepted=d['quality_accepted'],points=result['points'],camera_roundtrip_px=maximum)))
if __name__=='__main__':main()
