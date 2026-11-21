"""One CPU-only hash/raw-alpha audit of the existing129 step1000 snapshot."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageFilter

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
G=ROOT/'leafwoman-half12-paired18k129'
SNAP=G/'gaussian-run/step-001000'
OUT=SNAP/'alpha-summary129.json'
REGIONS=('mask','mask_eroded_5_derived_pixels')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stats(values,probs):
    assert values.size and np.isfinite(values).all()
    return dict(pixels=int(values.size),mean=float(values.mean(dtype=np.float64)),
        median=float(np.median(values)),quantiles=[float(v) for v in np.quantile(values,probs)],
        count_below_0_5=int((values<.5).sum()),count_below_0_95=int((values<.95).sum()),
        fraction_below_0_5=float((values<.5).mean()),fraction_below_0_95=float((values<.95).mean()))


def main():
    assert not OUT.exists(),'Preserve the existing summary; do not repeat or overwrite.'
    paths=[G/'plan.json',G/'data/training-inputs.json',G/'pixel-mapping.json',G/'adapter.py',
           G/'process-exit.json',SNAP/'result.json']
    hashes={str(p):sha(p) for p in paths}
    plan,inputs,mapping=read(paths[0]),read(paths[1]),read(paths[2])
    result=read(SNAP/'result.json');assert result['step']==result['optimizer_updates']==1000
    assert result['points']==33767 and result['frames_rendered']==12
    assert read(G/'process-exit.json')['exit_code']==0
    rows=inputs['records'];assert len(rows)==12 and len({r['frame'] for r in rows})==12
    maps={r['frame']:r for r in mapping['records']};assert set(maps)=={r['frame'] for r in rows}
    pooled={r:[] for r in REGIONS};records=[];probs=None
    for row in rows:
        frame=row['frame'];meta_path=SNAP/(frame+'_cumulative-alpha.json')
        npy=SNAP/(frame+'_cumulative-alpha.npy');png=SNAP/(frame+'_cumulative-alpha.png')
        mask_path=Path(row['mask']);meta=read(meta_path)
        for p in (meta_path,npy,png,mask_path):hashes[str(p)]=sha(p)
        assert meta['frame']==frame and hashes[str(npy)]==meta['alpha_npy_sha256']
        assert hashes[str(png)]==meta['display_png_sha256']
        assert hashes[str(mask_path)]==meta['mask_sha256']==row['mask_sha256']
        alpha=np.load(npy,allow_pickle=False)
        assert alpha.dtype==np.float32 and alpha.ndim==2 and np.isfinite(alpha).all()
        assert alpha.shape==tuple(row['wh'][::-1]) and row['wh']==maps[frame]['derived_wh']
        assert maps[frame]['source']['wh']==[2704,1520] and maps[frame]['scale']==2
        crop=maps[frame]['crop_xyxy'];assert row['wh']==[2*(crop[2]-crop[0]),2*(crop[3]-crop[1])]
        display=np.asarray(Image.open(png).convert('L'))
        assert np.array_equal(display,np.rint(np.clip(alpha,0,1)*255).astype(np.uint8))
        assert float(alpha.min())==meta['original_min'] and float(alpha.max())==meta['original_max']
        if probs is None:probs=meta['quantile_probabilities']
        assert probs==meta['quantile_probabilities']
        mask_image=Image.open(mask_path).convert('L');assert mask_image.size==tuple(row['wh'])
        masks={'mask':np.asarray(mask_image)>0,
               'mask_eroded_5_derived_pixels':np.asarray(mask_image.filter(ImageFilter.MinFilter(11)))>0}
        measured={}
        for region,mask in masks.items():
            values=alpha[mask];s=stats(values,probs);reported=meta['regions'][region]
            assert s['pixels']==reported['pixels']
            assert np.allclose(s['quantiles'],reported['quantiles'],rtol=0,atol=1e-12)
            assert s['fraction_below_0_5']==reported['fraction_below_0_5']
            assert s['fraction_below_0_95']==reported['fraction_below_0_95']
            measured[region]=s;pooled[region].append(values)
        records.append(dict(frame=frame,derived_wh=row['wh'],alpha_dtype=str(alpha.dtype),
            alpha_json_sha256=hashes[str(meta_path)],alpha_npy_sha256=hashes[str(npy)],
            alpha_png_sha256=hashes[str(png)],derived_mask_sha256=hashes[str(mask_path)],
            raw_float32_and_display_verified=True,reported_statistics_reproduced=True,regions=measured))
    aggregate={k:stats(np.concatenate(v),probs) for k,v in pooled.items()}
    for region,s in aggregate.items():
        assert s['pixels']==sum(r['regions'][region]['pixels'] for r in records)
        for label in ('0_5','0_95'):
            count=sum(r['regions'][region]['count_below_'+label] for r in records)
            assert count==s['count_below_'+label]
            assert s['fraction_below_'+label]==count/s['pixels']
    adapter=(G/'adapter.py').read_text(encoding='utf-8-sig')
    assert 'white_background=False' in adapter
    interval=plan['options']['opacity_reset_interval'];assert interval==1800
    assert all(sha(p)==digest for p,digest in hashes.items())
    viewed=[]
    for n in ('frame_000018_gaussian.png','frame_000018_cumulative-alpha.png'):
        p=SNAP/n;viewed.append(dict(path=str(p),sha256=sha(p),actual_visual_review=True))
    summary=dict(updated=datetime.now(timezone.utc).isoformat(),subject='leafwoman',step=1000,
        optimizer_updates=1000,points=33767,frames=12,helper_sha256=sha(__file__),source_hashes=hashes,
        region_names=list(REGIONS),quantile_probabilities=probs,records=records,area_weighted_aggregate=aggregate,
        aggregation='Each region is aggregated separately by concatenating its exact raw float32 pixel values from12 views. Fractions use pooled below-threshold pixel count divided by pooled region area. Quantiles/median are actual pooled quantiles, not averaged per-frame quantiles. All ROI scales2, so area factors are common; no cross-region pooling.',
        checks=dict(alpha_npy_sha_checks=12,alpha_png_sha_checks=12,derived_mask_sha_checks=12,
            raw_dtype_float32=12,PNG_is_exact_round_clip_float32=True,all24_region_JSON_statistics_reproduced=True,
            original_inputs_unchanged_after_read=True,pixel_mapping_and_derived_mask_contract_reused=True),
        measurement='Cumulative all-layer camera-ray alpha1-T from native zero-color/white-background render; includes sorting/truncation. Not per-point opacity, front-surface-only opacity, true material opacity or physical geometry truth.',
        diagnostic_erosion='Exactly original PIL MinFilter(11),5 derived pixels per edge, used only for this interior diagnostic. Source/training masks were not eroded or changed. At ROI2x this is not5 native pixels.',
        source_hole_caveat='Original SAM masks sometimes include/exclude leaf prop and finger/stem gaps or scarf/hair boundary. Pixels outside a source mask hole are absent from mask/interior metrics; metrics cannot prove missing anatomy or uniform body transparency. Native skirt/feet bottom truncation remains unknown.',
        entire_surface_too_transparent_conclusion='Not established by these cumulative all-layer pixel statistics; spatial holes and view/region differences need actual image review and do not measure front-surface-only response.',
        reset_context=dict(configured_periodic_interval=interval,snapshot_global_iteration=1000,
            periodic_reset_multiples_reached=0,dataset_white_background=False,
            white_background_start_reset_branch_applies=False,
            conclusion='No1800-period reset has occurred before this1000 snapshot; these artifacts cannot establish periodic opacity-reset causation.'),
        actual_viewed_images=viewed,
        frame18_visual_note='Actual Gaussian+alpha PNG inspected: lower-edge pale/bright detached-looking pieces and internal silhouette gaps are visible. Bright alpha is accumulated coverage, not proof these pieces are the correct skirt/feet or front surface. No pixels/points removed based on this observation.',
        GPU_started=False,rerendered=False,training_or_loss_modified=False,training_masks_changed=False,
        quality_accepted=False)
    with OUT.open('x',encoding='utf-8') as stream:stream.write(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),frames=12,aggregate=aggregate),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
