"""Full native frame evaluation with no target RGB/matte passed into the model."""
from pathlib import Path
import json,hashlib,random,torch,cv2,numpy as np,os
from PIL import Image,ImageDraw
from easyvolcap.engine import cfg,MODELS,DATASETS
from easyvolcap.scripts.main import preflight
from easyvolcap.utils.data_utils import default_collate,to_cuda,add_iter
from audit_enerf_input import read_cameras,digest
OUT=Path(os.environ.get('FOURKD_RUN_DIR','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-official-035'))
ROOT=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
EVAL=OUT/'native-evaluation';EVAL.mkdir(exist_ok=True)
assert (OUT/'training-completion.json').is_file()
assert not (EVAL/'metrics.json').exists()
completion=json.loads((OUT/'training-completion.json').read_text())
history_updates=completion.get('history_updates',completion['completed_updates_this_process'])
expected_epoch=completion.get('epoch',19)
torch.set_num_threads(8);cv2.setNumThreads(8);random.seed(0);torch.manual_seed(0)
checkpoint=OUT/'model/latest.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert cp['epoch']==expected_epoch
cfg.model_cfg.sampler_cfg.skip_loading_points=True
model=MODELS.build(cfg.model_cfg).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
source_gate=os.environ.get('FOURKD_SOURCE_FOREGROUND_GATE','')=='1'
if source_gate:
    from fourkd_source_support import enable_source_foreground_support
    enable_source_foreground_support(model.sampler)
del cp;torch.cuda.empty_cache()
dataset=DATASETS.build(cfg.val_dataloader_cfg.dataset_cfg)
cameras=read_cameras(ROOT/'optimized');records=[]
projection_path=os.environ.get('FOURKD_PROJECTION_OFFSETS','')
projection_offsets=None
if projection_path:
    from fourkd_projection_offsets import apply_projection_offsets
    projection_offsets={r['camera']:r['shift_xy_px'] for r in json.loads(Path(projection_path).read_text())['rows']}
protocol=dict(status='frozen_before_full_native_evaluation',cameras=['00','06','12','01','08','17'],frame=750,
    history_updates=history_updates,
    checkpoint_sha256=digest(checkpoint),resolution=[1920,1080],renderer='Official PyTorch3D path; native K restored from released calibration; full image without target crop',
    appearance_sources='8nearest allowed training cameras; target camera excluded even for training-view diagnostics',
    target_data='Target RGB/matte/weights removed from forward batch; separate CPU reference/matte used only after rendering',
    references='Original RGB and released soft matte, rectified at native K, gray0.5background; matte is not semantic human GT',
    regions='Full valid pixels, matte>0.5, foreground union bbox, upper60percent of union bbox, lower40percent; no identity assumption through overlap',
    scope='Single-frame foreground point-based neural rendering, not temporal, 3DGS, or fullscene completion')
protocol['source_foreground_support_gate']=source_gate
(EVAL/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
if projection_offsets is not None:
    protocol['projection_offset_file']=projection_path
    protocol['projection_offsets']='Frozen037empirical per-camera pixeloffset after originalnative targetK restoration; sourceoffsetafter originalcrop. No targetimagesusedinforward.'
    (EVAL/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for name in protocol['cameras']:
    c=int(name);dataset.append_gt_prob=1. if c in [0,6,12] else 0.
    item=dataset[c]
    source_cameras=[int(dataset.view_inds[int(i)]) for i in item.src_inds]
    assert c not in source_cameras and not set(source_cameras)&{0,6,12}
    cam=cameras[name];K=np.asarray(cam['K']);w2c=np.asarray(cam['w2c'])
    assert np.allclose(item.R.numpy(),w2c[:3,:3],atol=1e-6)
    assert np.allclose(item.T.numpy().reshape(3),w2c[:3,3],atol=1e-6)
    item.K=torch.tensor(K,dtype=torch.float32);item.meta.K=item.K
    item.tar_ixt=item.K
    item.H=1080;item.W=1920;item.meta.H=1080;item.meta.W=1920
    for key in ['rgb','msk','wet','dpt','bkg','norm','ray_o','ray_d','near','far','coords']:
        item.pop(key,None);item.meta.pop(key,None)
    batch=add_iter(default_collate([item]),history_updates,history_updates)
    if projection_offsets is not None:apply_projection_offsets(batch,c,source_cameras,projection_offsets)
    batch=to_cuda(batch)
    with torch.no_grad():output=model(batch)
    raw_rgb=output.rgb_map.detach().cpu().reshape(1080,1920,3).numpy()
    alpha=output.acc_map.detach().cpu().reshape(1080,1920).numpy()
    assert np.isfinite(raw_rgb).all() and np.isfinite(alpha).all()
    pred=np.clip(raw_rgb+.5*(1-alpha[...,None]),0,1)
    rp=ROOT/'images'/name/'000750.jpg';mp=ROOT/'bgmtv2'/name/'000750.jpg'
    maps=cv2.initUndistortRectifyMap(K,np.asarray(cam['D']),None,K,(1920,1080),cv2.CV_32FC1)
    src=cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB)
    source=cv2.remap(src,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    matte=cv2.remap(cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=(maps[0]>=0)&(maps[0]<1919)&(maps[1]>=0)&(maps[1]<1079)
    ref=source.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None])
    ys,xs=np.where((matte>.5)&valid);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    x0,y0,x1,y1=bbox;ym=y0+round(.6*(y1-y0))
    masks={'full_valid':valid,'foreground_matte_gt0_5':valid&(matte>.5)}
    for label,box in [('foreground_union_bbox',bbox),('upper_union',[x0,y0,x1,ym]),('lower_union',[x0,ym,x1,y1])]:
        m=np.zeros_like(valid);m[box[1]:box[3],box[0]:box[2]]=True;masks[label]=m&valid
    metrics={}
    for label,m in masks.items():
        diff=pred.astype(np.float64)[m]-ref.astype(np.float64)[m]
        metrics[label]=dict(pixels=int(m.sum()),psnr_db=float(-10*np.log10(np.mean(diff**2))),mae=float(np.abs(diff).mean()))
    prefix=f'camera{name}-frame000750'
    def u8(x):return np.rint(np.clip(x,0,1)*255).astype('uint8')
    for suffix,arr in [('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(alpha)),('source',source)]:Image.fromarray(arr).save(EVAL/f'{prefix}-{suffix}.png')
    np.savez_compressed(EVAL/f'{prefix}-raw.npz',rgb=raw_rgb,alpha=alpha,source=source,matte=matte,valid=valid,K=K,w2c=w2c)
    for label,box in [('full-person',bbox),('upper',[x0,y0,x1,ym])]:
        xx0,yy0,xx1,yy1=box;ww,hh=xx1-xx0,yy1-yy0
        canvas=Image.new('RGB',(2*ww,hh+38),'white');draw=ImageDraw.Draw(canvas)
        for k,(title,arr) in enumerate([('Reference (released matte)',ref),(f'Official4K4D {history_updates}updates',pred)]):
            canvas.paste(Image.fromarray(u8(arr[yy0:yy1,xx0:xx1])),(k*ww,38));draw.text((k*ww+8,10),title,fill='black')
        canvas.save(EVAL/f'{prefix}-{label}-comparison.png')
    records.append(dict(camera=name,split='heldout' if c in [0,6,12] else 'training_diagnostic_target_excluded_from_sources',source_cameras=source_cameras,
        source_rgb_sha256=digest(rp),source_matte_sha256=digest(mp),bbox=bbox,metrics=metrics,alpha_range=[float(alpha.min()),float(alpha.max())]))
    print('NATIVE_EVALUATION',json.dumps(records[-1]),flush=True)
    del batch,item,output;torch.cuda.empty_cache()
(EVAL/'metrics.json').write_text(json.dumps(dict(status='six_native_renders_complete_audit_and_visual_review_pending',records=records),indent=2)+'\n')
