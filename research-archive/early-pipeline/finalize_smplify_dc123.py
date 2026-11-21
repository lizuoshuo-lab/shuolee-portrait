"""Record completed human visual review and matched SH0 evidence; no training."""
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
import audit_gaussianavatars_params123 as a
from publish_import_teacher123 import win
from publish_smplify_dc123 import OUT,BASE

def main():
    stamp=datetime.now(timezone.utc).isoformat()
    snap=OUT/'gaussian-run/step-018000'
    other=BASE/'gaussian-run/step-018000'
    info=a.read(snap/'review/result.json')
    orbit=a.read(snap/'orbit-comparison/result.json')
    metrics=a.read(snap/'matched-metrics.json')
    receipt=a.read(snap/'result.json')
    assert receipt['active_sh_degree']==0 and metrics['gaussians']['DC']['high_order_fields']==0
    assert len(info['records'])==59 and len(info['pages'])==30 and orbit['views']==36
    for p in info['pages']+orbit['pages']:
        assert a.sha(win(p['path']))==p['sha256']
    notes={
        'body':'All59 source/current-mesh/Gaussian views reviewed. Training views are comparatively coherent. Frames16/18-23 retain a fragmented or missing held arm;40/51 also show pose-associated breakup. Remaining side views retain irregular cloth, hem and silhouette. No claimed geometry acceptance.',
        'face':'All59 native face crops reviewed. Eye/nose/mouth and beard retain black-white granular artifacts away from the8 training views. Protected mesh is unchanged, not newly constrained by independent3D observations.',
        'left_hand':'All59 held-hand crops reviewed. Fingers remain blurred or fragmented and displaced in several excluded views. SH0 does not restore missing fingers or device occlusion.',
        'right_hand':'All59 fixed crops reviewed, but the pocket hand is mostly occluded. Some historical detector ROIs include sleeve/background. These crop errors are not independent visible-hand geometry evidence.',
        'feet':'All59 crops reviewed. Shoe outlines/laces remain soft, and later side views retain shoe-coloured edge fragments. RGB MAE improvement on some frames does not establish correct foot geometry.',
        'novel':'All36 paired fixed-camera renders reviewed in4 pages. SH0 has no per-Gaussian directional colour term; side/back holes and strong spatial speckle remain, sometimes with harsher light-dark contrast. Neither model is accepted on unseen sides. These are synthetic cameras without ground-truth images.'}
    review=dict(updated=stamp,step=18000,actual_visual_review=True,all59_reviewed=True,
        regions=['body','face','left_hand','right_hand','feet'],pages=info['pages'],
        native_same_frame_crops=True,contact_sheet_body_uniformly_resized=True,
        text_added_to_images=False,notes=notes,quality_accepted=False)
    a.save(snap/'visual-review.json',review)
    a.save(snap/'orbit-comparison/visual-review.json',dict(updated=stamp,actual_visual_review=True,
        views=36,pages=orbit['pages'],same_native_rasterizer=True,same_camera_matrices=True,
        notes=notes['novel'],ground_truth_available=False,quality_accepted=False))
    record=Path('D:/00/记录/demo')
    publication=[]
    for p in orbit['pages']:
        name=Path(p['path']).name
        dst=record/('分析_SH0与SH3同为18000步_'+name)
        a.lp(dst).write_bytes(a.lp(win(p['path'])).read_bytes())
        publication.append(dict(path=str(dst),name='matched-'+name,
            caption='同为18000步、同相机；每对左=三阶球谐训练，右=从头零阶训练。侧背面仍有缺块和斑驳，未通过人物质量验收。'))
    for frame,region in [('frame_000018','body'),('frame_000033','body'),('frame_000058','body'),('frame_000018','face'),('frame_000033','face')]:
        row=next(r for r in info['records'] if r['frame']==frame)
        item=next(x for x in row['images'] if x['region']==region)
        x0,y0,x1,y1=item['crop_integer_xyxy'];w,h=x1-x0,y1-y0
        src=Image.open(a.lp(win(item['path']))).convert('RGB')
        assert src.size==(3*w,h)
        old=Image.open(a.lp(other/(frame+'_gaussian.png'))).convert('RGB').crop((x0,y0,x1,y1))
        new=Image.new('RGB',(4*w,h))
        for i,im in enumerate([src.crop((0,0,w,h)),src.crop((w,0,2*w,h)),old,src.crop((2*w,0,3*w,h))]):
            new.paste(im,(i*w,0))
        name=f'高斯_SH颜色18000步对照_{frame}_{region}_原图_当前mesh_SH3_SH0.png'
        dst=record/name;new.save(a.lp(dst))
        publication.append(dict(path=str(dst),name=f'matched-{frame}-{region}.png',
            caption=f'{frame} · {region}：左起原图、零阶训练当前mesh、三阶高斯、零阶高斯。统一原生整数裁片，无变形或文字；18和33帧未参与本轮高斯训练，58帧参与训练。'))
    summary=('已从头完成18000步零阶球谐训练：25088个高斯，高阶颜色参数和导出字段均为0。'
             '实际审阅59帧及36个固定环绕视角后，方向性颜色自由度已消除，但空间斑驳、侧背面缺块与部分手臂破碎仍在，人物质量未通过。'
             '未参与高斯训练的51帧中，逐帧身体/面部masked MAE相对变化的中位数为+2.07%/+3.85%，没有整体改善。')
    a.save(snap/'comparison-review.json',dict(updated=stamp,summary=summary,
        next='The requested SH restriction is implemented and verified as a separate SH0 candidate. Do not repeat this completed18k control or promote it as quality accepted. Next geometry work must inspect the remaining pose/coverage/anisotropic-shape compensation separately, while keeping face guidance excluded. No new parameter sweep registered.',
        all59_visual_review=True,novel36_visual_review=True,notes=notes,publication_images=publication,
        groups=metrics['groups'],offsets=metrics['offsets'],gaussians=metrics['gaussians'],
        same_step=18000,sole_training_option_change='max_sh_degree3->0',
        changed_point_count_is_downstream_of_optimization_not_an_added_hyperparameter=True,
        native_ply_parity=receipt['native_ply_parity'],
        temporal_split_scope='51 frames excluded from Gaussian optimization; all59 were used previously in SMPLify-X fitting, so not fully independent pipeline test data.',
        quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False,
        geometry_conclusion='Clothing offset median0.08599->0.11068mm, max0.36624->0.38133mm. Not clothing-surface fitting. Gaussian scale-ratio P90 226.6->488.1 but longest-axis P99 remains about15.8mm: ratio alone does not prove elongated absolute footprints or unique cause.',
        requested_colour_restriction_implemented=True,recommended_as_final_high_quality_model=False,
        baseline_default_viewer_preserved=True))
    print('SH0_MATCHED_REVIEW_RECORDED',flush=True)

if __name__=='__main__':main()
