"""Independent heldout PNG audits and native-pixel matched surface comparisons."""
from pathlib import Path
import argparse
import json
import statistics
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_street_joint_images import check_quantization,sha

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-surface-refit-018'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['appearance_only','surface_and_appearance']);args=parser.parse_args()
    regions=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    lookup={(r['frame'],r['camera'],r['instance']):r for r in regions}
    assert len(lookup)==435
    branches=[args.branch] if args.branch else ['appearance_only','surface_and_appearance']
    for branch in branches:
        run=OUT/branch if branch=='appearance_only' else OUT/'surface-retry-v2'/branch
        assert (run/'completion.json').exists()
        assert not (run/'independent-image-audit.json').exists(),'Preserve completed audit'
        metrics=json.loads((run/'evaluation/final/metrics.json').read_text())
        assert len(metrics['views'])==105 and {(r['frame'],r['camera']) for r in metrics['views']}=={(f,c) for f in range(10,151,10) for c in range(7)}
        scores={(r['frame'],r['camera'],r['instance']):r for r in metrics['person_box_regions']}
        assert len(scores)==435 and scores.keys()==lookup.keys()
        image_checks=[];roi_checks=[]
        for view in metrics['views']:
            f,c=view['frame'],view['camera'];stem=f'frame{f:03d}-camera{c}'
            path=run/'evaluation/final'/(stem+'-prediction.png')
            prediction=np.asarray(Image.open(path).convert('RGB'))
            target=np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((prediction.shape[1],prediction.shape[0]),Image.Resampling.BILINEAR))
            assert np.array_equal(target,np.asarray(Image.open(run/'ground-truth'/(stem+'.png'))))
            value,diff=check_quantization(target,prediction,view['psnr_db'])
            image_checks.append(dict(frame=f,camera=c,sha256=sha(path),png_psnr_db=value,quantization_difference_db=diff))
            for key,roi in lookup.items():
                if key[:2]!=(f,c):continue
                record=scores[key];assert roi['box']==record['box'] and roi['image_size']==record['image_size']
                x0,y0,x1,y1=roi['box']
                value,diff=check_quantization(target[y0:y1,x0:x1],prediction[y0:y1,x0:x1],record['psnr_db'])
                if key[2]==79:
                    assert np.array_equal(prediction[y0:y1,x0:x1],np.asarray(Image.open(run/'evaluation/final'/(stem+'-person79.png'))))
                    assert np.array_equal(target[y0:y1,x0:x1],np.asarray(Image.open(run/'ground-truth'/(stem+'-person79.png'))))
                roi_checks.append(dict(frame=f,camera=c,instance=key[2],psnr_db=record['psnr_db'],ssim=record['ssim'],png_psnr_db=value,quantization_difference_db=diff))
        training=[]
        for label in ('before','after'):
            m=json.loads((run/'training-evaluation'/label/'metrics.json').read_text())
            assert m['count']==len(m['rows'])==101
            for f,c in [(129,4),(131,4),(149,6),(151,6)]:
                r=next(r for r in m['rows'] if (r['frame'],r['camera'])==(f,c));stem=f'frame{f:03d}-camera{c}'
                pred=np.asarray(Image.open(run/'training-evaluation'/label/(stem+'-prediction.png')))
                gt=np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((pred.shape[1],pred.shape[0]),Image.Resampling.BILINEAR))
                assert np.array_equal(gt,np.asarray(Image.open(run/'training-ground-truth'/(stem+'.png'))))
                x0,y0,x1,y1=r['box'];check_quantization(gt[y0:y1,x0:x1],pred[y0:y1,x0:x1],r['psnr_db'])
            training.append(dict(label=label,mean_psnr_db=m['mean_person_psnr_db'],mean_mae=m['mean_mae'],saved_views_checked=4))
        target_scores=[r for r in roi_checks if r['instance']==79];assert len(target_scores)==13
        result=dict(status='all105_heldout_PNGs435regions_and_saved_training_views_independently_verified',branch=branch,
            summary=dict(full_mean_psnr_db=statistics.mean(r['psnr_db'] for r in metrics['views']),target13_mean_psnr_db=statistics.mean(r['psnr_db'] for r in target_scores),target13_mean_ssim=statistics.mean(r['ssim'] for r in target_scores)),
            images=image_checks,regions=roi_checks,training=training,quality_accepted=False,
            boundaries=['PSNR independently recomputed from PNG with deterministic quantization bounds; SSIM is the original float evaluator score.',
                'Fixed person boxes include background and occlusion; no ground-truth person segmentation.',
                'Visual and temporal acceptance remain separate from image-score verification.'])
        (run/'independent-image-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(branch=branch,summary=result['summary'],training=training)),flush=True)
    # Only build four-column figures after both independently audited branches exist.
    final_runs=[OUT/'appearance_only',OUT/'surface-retry-v2/surface_and_appearance']
    if all((run/'independent-image-audit.json').exists() for run in final_runs):
        figures=OUT/'paired-audit';figures.mkdir(exist_ok=True)
        assert not (figures/'figures.json').exists(),'Preserve paired figures'
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);records=[]
        for f,c,identity in sorted(lookup):
            if identity!=79:continue
            stem=f'frame{f:03d}-camera{c}'
            target=Image.open(SOURCE/'ground-truth'/(stem+'-person79.png')).convert('RGB')
            panels=[target]+[Image.open(p/'evaluation/final'/(stem+'-person79.png')).convert('RGB') for p in [SOURCE,*final_runs]]
            scale=min(3.,250/target.width,620/target.height);size=tuple(round(x*scale) for x in target.size)
            image=Image.new('RGB',(1080,size[1]+84),'white');draw=ImageDraw.Draw(image)
            for col,(label,panel) in enumerate(zip(['原始观测','姿态优化后的起点','仅优化颜色','表面与颜色联合优化'],panels)):
                assert panel.size==target.size
                draw.text((col*270+9,8),label,font=font,fill='#182d39')
                image.paste(panel.resize(size,Image.Resampling.NEAREST),(col*270+(270-size[0])//2,38))
            draw.text((10,size[1]+48),f'帧 {f} / 相机 {c} · 两支各 100 次全区域更新；同区域，未锐化',font=font,fill='#182d39')
            file=figures/(stem+'-person79-comparison.png');image.save(file)
            records.append(dict(file=file.name,sha256=sha(file),frame=f,camera=c))
        (figures/'figures.json').write_text(json.dumps(dict(figures=records,visual_review_pending=True),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
