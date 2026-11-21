"""Score complete native renders and build compact, unenhanced visual comparisons."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import torch
import lpips
from shuolee_pgsr_evaluate import OUT, ROOT, SOURCE, Video, dump, read, sha


def distribution(a):
    a=np.asarray(a,dtype=float)
    if not len(a):return {'count':0}
    return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),mean=float(a.mean()))


def stats(p,g,mask):
    assert mask.shape==p.shape[:2] and mask.any()
    err=p-g;mse=float((err[mask]**2).mean())
    _,smap=structural_similarity(p,g,channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,
        use_sample_covariance=False,win_size=11,full=True)
    if smap.ndim==3:smap=smap.mean(-1)
    interior=cv2.erode(mask.astype('uint8'),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    ps=cv2.cvtColor(p,cv2.COLOR_RGB2GRAY);gs=cv2.cvtColor(g,cv2.COLOR_RGB2GRAY)
    gradient=lambda a:np.stack([cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)],-1)
    pg,gg=gradient(ps),gradient(gs)
    edges=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return dict(psnr=float(-10*np.log10(max(mse,1e-12))),mae=float(np.abs(err[mask]).mean()),
        ssim=float(smap[interior].mean()),valid_fraction=float(mask.mean()),
        gradient_error=float(np.abs(pg-gg)[edges].mean()/(np.abs(gg)[edges].mean()+1e-8)),
        gradient_magnitude_ratio=float(np.linalg.norm(pg,axis=-1)[edges].mean()/(np.linalg.norm(gg,axis=-1)[edges].mean()+1e-8)))


def title_tile(arr,title,size=(520,650)):
    # Aspect ratio retained, same pixels and crop for every method.
    im=Image.fromarray(arr);im.thumbnail((size[0]-16,size[1]-48),Image.Resampling.LANCZOS)
    out=Image.new('RGB',size,'white');out.paste(im,((size[0]-im.width)//2,42))
    fonts=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']
    font=next((ImageFont.truetype(p,20) for p in fonts if Path(p).exists()),ImageFont.load_default())
    ImageDraw.Draw(out).text((12,10),title,font=font,fill='#18333f')
    return out


@torch.no_grad()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    protocol=read(OUT/'protocol.json')
    assert all(read(OUT/arm/'manifest.json')['status']=='complete' for arm in ['baseline','pgsr'])
    review=OUT/'review';review.mkdir(exist_ok=True)
    assert not any(review.iterdir()),'Do not overwrite completed comparison evidence'
    mask_folder=OUT/'development-masks'
    mask_protocol=read(mask_folder/'protocol.json')
    assert mask_protocol['status']=='source_only_annotation_reviewed'
    assert mask_protocol['original_protocol_sha256']==sha(OUT/'protocol.json')
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    rows=[]
    for c in protocol['development_cameras']:
        n=c['img_name'];source=SOURCE/'images'/(n+'.png');assert sha(source)==protocol['images_sha256'][n]
        gt=np.asarray(Image.open(source).convert('RGB')).astype('float32')/255
        h,w=gt.shape[:2]
        mask_path=mask_folder/(n+'.png')
        assert sha(mask_path)==mask_protocol['rows'][n]['sha256']
        motion=np.asarray(Image.open(mask_path).convert('L'))
        valid=cv2.resize(motion,(w,h),interpolation=cv2.INTER_NEAREST)==0
        boxes=protocol['regions'][n];person=np.zeros((h,w),bool)
        x0,y0,x1,y1=boxes['person'];person[y0:y1,x0:x1]=True
        regions={'full':[0,0,w,h],**boxes,'outside_person_box':[0,0,w,h]}
        for arm in ['baseline','pgsr']:
            pred=np.asarray(Image.open(OUT/arm/(n+'.png')).convert('RGB')).astype('float32')/255
            assert pred.shape==gt.shape
            result={}
            for region,(a,b,d,e) in regions.items():
                mask=valid[b:e,a:d].copy()
                if region=='outside_person_box':mask&=~person
                pp=pred[b:e,a:d];gg=gt[b:e,a:d]
                result[region]=stats(pp,gg,mask)
                if region in ['head','person']:
                    # Unmasked rectangle: excluded fraction is explicitly reported.
                    p=torch.from_numpy(pp.copy()).permute(2,0,1).cuda()[None]*2-1
                    g=torch.from_numpy(gg.copy()).permute(2,0,1).cuda()[None]*2-1
                    result[region]['lpips_unmasked_rectangle']=float(perceptual(p,g))
                    del p,g
            result['full_unmasked']=stats(pred,gt,np.ones((h,w),bool))
            rows.append(dict(frame=n,arm=arm,metrics=result))
            print(arm,n,'head',result['head'],flush=True)
    aggregate={arm:{r:{m:float(np.mean([x['metrics'][r][m] for x in rows if x['arm']==arm]))
        for m in rows[0]['metrics'][r]} for r in rows[0]['metrics']} for arm in ['baseline','pgsr']}
    difference={r:{m:aggregate['pgsr'][r][m]-aggregate['baseline'][r][m] for m in aggregate['baseline'][r]} for r in aggregate['baseline']}
    geometry={};paired_depth={}
    raw_depth={arm:read(OUT/arm/'depth-observations.json') for arm in ['baseline','pgsr']}
    for arm,observations in raw_depth.items():
        geometry[arm]={}
        for region in ['head','outside_head']:
            obs=[x for x in observations if x['head']==(region=='head')]
            accepted=[x for x in obs if x['valid']]
            geometry[arm][region]=dict(eligible_observations=len(obs),valid_fraction=len(accepted)/max(len(obs),1),
                relative_error=distribution([x['relative_error'] for x in accepted]),
                unique_points=len({x['point_id'] for x in accepted}))
    base={(x['frame'],x['point_id']):x for x in raw_depth['baseline']}
    for region in ['head','outside_head']:
        pairs=[(base[(x['frame'],x['point_id'])],x) for x in raw_depth['pgsr'] if x['head']==(region=='head')
            and x['valid'] and base[(x['frame'],x['point_id'])]['valid']]
        paired_depth[region]={arm:distribution([p[i]['relative_error'] for p in pairs]) for i,arm in enumerate(['baseline','pgsr'])}
    # Frozen examples; do not choose only favorable views after scoring.
    canvas=Image.new('RGB',(1560,1300),'white')
    for row,n in enumerate(['000016','000040']):
        box=protocol['regions'][n]['head']
        files=[SOURCE/'images'/(n+'.png'),OUT/'baseline'/(n+'.png'),OUT/'pgsr'/(n+'.png')]
        for col,(path,label) in enumerate(zip(files,['Source photograph','Mini-Splatting2 baseline','PGSR surface candidate'])):
            arr=np.asarray(Image.open(path).convert('RGB').crop(box))
            canvas.paste(title_tile(arr,f'{label} / {n}'),(col*520,row*650))
    canvas.save(review/'real-head-comparison.png')
    canvas=Image.new('RGB',(1040,1950),'white')
    for row,name in enumerate(['elevation-015','elevation-045','distance-045']):
        for col,arm in enumerate(['baseline','pgsr']):
            arr=np.asarray(Image.open(OUT/arm/(name+'.png')).convert('RGB'))
            canvas.paste(title_tile(arr,f'{arm} / {name}'),(col*520,row*650))
    canvas.save(review/'novel-view-comparison.png')
    for kind in ['distance','elevation']:
        caps=[cv2.VideoCapture(str(OUT/arm/(kind+'.mp4'))) for arm in ['baseline','pgsr']]
        assert all(c.isOpened() for c in caps)
        video=Video(review/(kind+'.mp4'),1152,1072);count=0
        try:
            while True:
                samples=[c.read() for c in caps]
                assert samples[0][0]==samples[1][0]
                if not samples[0][0]:break
                a=np.concatenate([s[1][...,::-1] for s in samples],axis=1)
                header=Image.new('RGB',(1152,48),'white');d=ImageDraw.Draw(header)
                d.text((20,16),'Mini-Splatting2 baseline',fill='black');d.text((596,16),'PGSR surface candidate',fill='black')
                video.add(np.concatenate([np.asarray(header),a],axis=0));count+=1
        finally:
            video.close()
            for c in caps:c.release()
        assert count==60
    result=dict(status='complete_pending_visual_review',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        metric_implementation='skimage0.20 Gaussian-window SSIM, static-mask erosion11; native-resolution LPIPS AlexNet rectangle',
        protocol_sha256=sha(OUT/'protocol.json'),evaluation_amendment_sha256=sha(mask_folder/'protocol.json'),
        masking_scope=mask_protocol['scope'],rows=rows,aggregate=aggregate,delta_candidate_minus_baseline=difference,
        geometry=geometry,geometry_common_support=paired_depth,
        geometry_is_independent_GT=False,method_comparison_not_causal_ablation=True,
        novel_views_reviewed=False,human_reviews=0,accepted_as_best=False)
    dump(review/'metrics.json',result)
    print(json.dumps(dict(status=result['status'],delta=difference,geometry=geometry),indent=2),flush=True)


if __name__=='__main__':main()
