"""Independent saved-image checks for the untrained compositing interventions."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from audit_street_joint_images import check_quantization,sha

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'


def main():
    assert (OUT/'completion.json').exists()
    assert not (OUT/'independent-image-audit.json').exists(),'Preserve completed audit'
    report=json.loads((OUT/'completion.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert report['optimized_steps']==0 and report['all_model_tensors_unchanged']
    assert [(r['frame'],r['camera']) for r in report['rows']]==[(r['frame'],r['camera']) for r in protocol['views']]
    checks=[]
    for row in report['rows']:
        f,c=row['frame'],row['camera'];stem=f'frame{f:03d}-camera{c}'
        names=['observed','full','without-target','visible-alpha','target-contribution','other-contribution','half-scale','quarter-scale']
        paths={key:OUT/'renders'/(stem+'-'+key+'.png') for key in names}
        images={key:np.asarray(Image.open(path).convert('RGB')) for key,path in paths.items()}
        observed=images['observed'];h,w=observed.shape[:2]
        original=Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((w,h),Image.Resampling.BILINEAR)
        assert np.array_equal(observed,np.asarray(original))
        assert all(a.shape==observed.shape for a in images.values())
        original_full=SOURCE/('evaluation/final' if row['heldout'] else 'training-evaluation/after')/(stem+'-prediction.png')
        delta=np.abs(images['full'].astype(np.int16)-np.asarray(Image.open(original_full),dtype=np.int16))
        assert delta.max()<=1
        weight=images['visible-alpha'];assert np.array_equal(weight[:,:,0],weight[:,:,1]) and np.array_equal(weight[:,:,0],weight[:,:,2])
        linear_path=OUT/'renders'/(stem+'-linear-components.npz')
        with np.load(linear_path) as linear:
            assert all(np.isfinite(linear[key]).all() and linear[key].shape==observed.shape for key in linear.files)
            reconstructed=np.clip(np.minimum(linear['target']+linear['other'],1)+linear['sky'],0,1)
            additive=np.abs(reconstructed-linear['full'])
            assert additive.max()<2e-5
            quantized=np.uint8(linear['full']*255+.5)
            assert np.array_equal(quantized,images['full'])
            for key,raw in [('target-contribution',linear['target']),('other-contribution',linear['other']+linear['sky'])]:
                assert np.array_equal(np.uint8(np.clip(raw,0,1)*255+.5),images[key])
        x0,y0,x1,y1=row['box']
        scores={}
        for key,item in row['variants'].items():
            score,diff=check_quantization(observed[y0:y1,x0:x1],images[key][y0:y1,x0:x1],item['psnr_db'])
            scores[key]=dict(png_psnr_db=score,quantization_difference_db=diff)
        checks.append(dict(frame=f,camera=c,heldout=row['heldout'],source_target_exact=True,full_reload_max8bit=int(delta.max()),
            linear_components_sha256=sha(linear_path),author_clamp_order_max_float_error=float(additive.max()),scores=scores,
            files=[dict(file=str(p.relative_to(OUT)),sha256=sha(p)) for p in paths.values()]))
    result=dict(status='six_views_48_saved_images_verified_no_training_no_quality_acceptance',views=checks,
        quality_accepted=False,optimized_steps=0,
        boundaries=['Full image reload and observed pixels independently checked; pre-clamp color contributions reproduce the original clamp-before-sky render.',
            'Published contribution pictures are individually clipped for display and need not add exactly in saturated pixels.',
            'Visible weight is model compositing attribution, not human segmentation ground truth.',
            'Scale modifications are untrained interventions and may expose background errors or holes.',
            'Renderer footprint statistics are diagnostic metadata, not independently verified silhouette metrics.'])
    (OUT/'independent-image-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],views=len(checks),images=len(checks)*8)))


if __name__=='__main__':main()
