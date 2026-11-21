"""Independent CPU audit of original/refined street-person renderings."""
from pathlib import Path
import argparse
import json
import statistics
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_street_joint_images import check_quantization, sha

LAB = Path(__file__).resolve().parents[2]
SOURCE = LAB / 'outputs/dynamic-human-street-smpl-joint-011/articulated'
RUN = LAB / 'outputs/dynamic-human-street-pose-refinement-014'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'


def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--long',action='store_true')
    args=parser.parse_args()
    if args.long:args.batch=True
    if args.batch:RUN=LAB/'outputs/dynamic-human-street-pose-batch-015'
    if args.long:RUN=LAB/'outputs/dynamic-human-street-pose-convergence-016'
    if not (RUN/'completion.json').exists():
        print(json.dumps(dict(status='not_ready_no_quality_claim')))
        return
    completion=json.loads((RUN/'completion.json').read_text())
    assert completion['source_updates']==30000
    if args.batch:assert completion['extra_lbfgs_iterations']<=(120 if args.long else 20)
    else:assert completion['additional_pose_updates']==3000
    expected={(f,c) for f in range(10,151,10) for c in range(7)}
    regions=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    assert len(regions)==435
    roi_lookup={(r['frame'],r['camera'],r['instance']):r for r in regions}
    summaries, checks, region_checks = {}, [], []
    for name,run,updates in [('source',SOURCE,30000),('pose_refined',RUN,completion['optimizer_updates'])]:
        metrics=json.loads((run/'evaluation/final/metrics.json').read_text())
        assert metrics['optimizer_updates']==updates
        assert len(metrics['views'])==105 and {(r['frame'],r['camera']) for r in metrics['views']}==expected
        rows={(r['frame'],r['camera'],r['instance']):r for r in metrics['person_box_regions']}
        assert len(rows)==len(metrics['person_box_regions'])==435 and rows.keys()==roi_lookup.keys()
        for item in metrics['views']:
            frame,camera=item['frame'],item['camera']
            stem=f'frame{frame:03d}-camera{camera}'
            path=run/'evaluation/final'/(stem+'-prediction.png')
            prediction=np.asarray(Image.open(path).convert('RGB'))
            original=Image.open(SCENE/'images'/f'{frame:03d}_{camera}.jpg').convert('RGB')
            target=np.asarray(original.resize((prediction.shape[1],prediction.shape[0]),Image.Resampling.BILINEAR))
            assert np.array_equal(target,np.asarray(Image.open(run/'ground-truth'/(stem+'.png'))))
            score,diff=check_quantization(target,prediction,item['psnr_db'])
            checks.append(dict(branch=name,frame=frame,camera=camera,png_psnr_db=score,quantization_difference_db=diff,sha256=sha(path)))
            for key,roi in roi_lookup.items():
                if key[:2]!=(frame,camera):continue
                row=rows[key]
                assert roi['box']==row['box'] and roi['image_size']==row['image_size']
                x0,y0,x1,y1=roi['box']
                score,diff=check_quantization(target[y0:y1,x0:x1],prediction[y0:y1,x0:x1],row['psnr_db'])
                if key[2]==79:
                    assert np.array_equal(prediction[y0:y1,x0:x1],np.asarray(Image.open(run/'evaluation/final'/(stem+'-person79.png'))))
                    assert np.array_equal(target[y0:y1,x0:x1],np.asarray(Image.open(run/'ground-truth'/(stem+'-person79.png'))))
                region_checks.append(dict(branch=name,frame=frame,camera=camera,instance=key[2],psnr_db=row['psnr_db'],ssim=row['ssim'],png_psnr_db=score,quantization_difference_db=diff))
        summaries[name]=dict(full_mean_psnr_db=statistics.mean(r['psnr_db'] for r in metrics['views']),
            target13_mean_psnr_db=statistics.mean(r['psnr_db'] for r in rows.values() if r['instance']==79),
            target13_mean_ssim=statistics.mean(r['ssim'] for r in rows.values() if r['instance']==79))
    out=RUN/'independent-audit'
    assert not (out/'independent-audit.json').exists(),'Preserve completed audit'
    out.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    figures=[]
    for frame,camera,instance in sorted(roi_lookup):
        if instance!=79:continue
        stem=f'frame{frame:03d}-camera{camera}'
        target=Image.open(SOURCE/'ground-truth'/(stem+'-person79.png')).convert('RGB')
        panels=[target]+[Image.open(run/'evaluation/final'/(stem+'-person79.png')).convert('RGB') for run in (SOURCE,RUN)]
        assert all(panel.size==target.size for panel in panels)
        scale=min(3.,260/target.width,650/target.height)
        size=tuple(round(x*scale) for x in target.size)
        canvas=Image.new('RGB',(840,size[1]+90),'white');draw=ImageDraw.Draw(canvas)
        for col,(label,panel) in enumerate(zip(['原始观测','原人体关节模型','仅姿态再优化'],panels)):
            draw.text((280*col+12,8),label,font=font,fill='#182d39')
            canvas.paste(panel.resize(size,Image.Resampling.NEAREST),(280*col+(280-size[0])//2,42))
        budget='第三列按全部训练区域优化姿态' if args.batch else '第三列额外 3,000 次姿态更新'
        draw.text((12,size[1]+52),f'帧 {frame} / 相机 {camera} · 同区域；{budget}',font=font,fill='#182d39')
        file=out/(stem+'-person79-comparison.png');canvas.save(file)
        figures.append(dict(file=file.name,sha256=sha(file),frame=frame,camera=camera))
    training=[]
    for label in ('before','after'):
        metrics=json.loads((RUN/'training-evaluation'/label/'metrics.json').read_text())
        verified=0
        for frame,camera in [(129,4),(131,4),(149,6),(151,6)]:
            row=next(r for r in metrics['rows'] if (r['frame'],r['camera'])==(frame,camera))
            stem=f'frame{frame:03d}-camera{camera}'
            pred=np.asarray(Image.open(RUN/'training-evaluation'/label/(stem+'-prediction.png')))
            original=Image.open(SCENE/'images'/f'{frame:03d}_{camera}.jpg').convert('RGB')
            target=np.asarray(original.resize((pred.shape[1],pred.shape[0]),Image.Resampling.BILINEAR))
            assert np.array_equal(target,np.asarray(Image.open(RUN/'training-ground-truth'/(stem+'.png'))))
            x0,y0,x1,y1=row['box']
            check_quantization(target[y0:y1,x0:x1],pred[y0:y1,x0:x1],row['psnr_db'])
            assert np.array_equal(pred[y0:y1,x0:x1],np.asarray(Image.open(RUN/'training-evaluation'/label/(stem+'-person79.png'))))
            verified+=1
        training.append(dict(label=label,total_training_regions=metrics['count'],mean_training_psnr_db=metrics['mean_person_psnr_db'],
            mean_training_mae=metrics['mean_mae'],independently_verified_saved_views=verified))
    report=dict(status='all210_heldout_images_and870_regions_verified_pose_visual_acceptance_pending',
        source_updates=30000,extra_pose_updates=completion.get('additional_pose_updates'),extra_lbfgs_iterations=completion.get('extra_lbfgs_iterations'),
        closure_evaluations=completion.get('closure_evaluations'),summary=summaries,training=training,images=checks,regions=region_checks,figures=figures,
        quality_accepted=False,boundaries=['Exploratory diagnostic after failure inspection, not an untouched benchmark.',
            'Only pose changed; additional optimization budget differs from original source.',
            '13target boxes contain background/occlusion; no silhouette or joint ground truth.',
            'All final PNG scores independently checked; only four saved training views per stage independently checked.',
            'No sharpening, image synthesis repair or model/pose weights published.'])
    (out/'independent-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],summary=summaries,training=training),indent=2))


if __name__=='__main__':main()
