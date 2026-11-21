"""Matched training-only color refinement, with a spatial foreground prior.

This is an exploratory, fixed-geometry intervention on known cameras. It is
not an unseen-scene result, a postprocessing filter, or a full dynamic model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont
from adapter import LAB, DATA, load_official, dataset
from evaluate_benchmark import metric, png

OUT = LAB/'outputs/dynamic-human-appearance-refinement-008'
SOURCE = LAB/'outputs/dynamic-human-reconstruction-003/dynamic/frame-0000.pt'
MODES = ['photometric', 'spatial_color']
STEPS = 2000

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf8')

def prepare():
    OUT.mkdir(exist_ok=True)
    protocol = dict(source=str(SOURCE.relative_to(LAB)),source_sha256=sha(SOURCE),
        implementation_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(__file__).with_name('adapter.py')),
        stage='known-scene first-frame appearance intervention',train_frame=0,seed=73,
        training_steps_per_branch=STEPS,train_camera_count=27,modes=MODES,
        trainable_parameters=['rgb_colors'],fixed='positions, rotations, scale, opacity, labels and camera gain/bias',
        optimizer=dict(name='Adam',lr=.0025,eps=1e-15,initial_moments='zero for both branches'),
        photometric_loss='0.8 L1 + 0.2 (1-SSIM), training-camera exposure from the source checkpoint',
        spatial_prior=dict(weight=.01,neighbors=8,foreground_only=True,
            graph='same-foreground 8-nearest neighbors of fixed source geometry; remove self explicitly',
            kernel='exp(-squared_distance/(2*per_point_8th_neighbor_distance_squared)); normalize weights per point',
            penalty='mean weighted sqrt(channel_color_difference^2+0.001^2)-0.001',
            selection='one predeclared untuned coefficient; no heldout loss or per-test-camera fitting'),
        heldout_cameras=[0,10,15,30],evaluation_frame=0,
        evaluation_rois=json.loads((LAB/'outputs/dynamic-human-reconstruction-003/evaluation-protocol.json').read_text())['fixed_human_rois_xyxy'],
        known_shirt_roi=dict(camera=15,xyxy=[275,200,365,310]),
        scope='Diagnostic intervention on previously inspected cameras; not independent generalization or dynamic validation',
        limitations=['Foreground includes basketball as in released labels',
            'Euclidean neighbors can connect nearby distinct surfaces; color regularization can oversmooth true texture',
            'Fixed geometry isolates appearance and cannot repair depth, shape, motion or visibility errors'])
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        save(OUT/'protocol.json',protocol)
    return protocol

def train(mode):
    protocol = prepare()
    folder = OUT/mode
    folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists(), 'Do not overwrite completed refinement'
    assert not (folder/'resume.pt').exists(), 'Existing partial refinement requires explicit resume review'
    torch.set_num_threads(4)
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    tr,h = load_official()
    source = torch.load(SOURCE,map_location='cpu',weights_only=True)
    p = {k:v.cuda() for k,v in source.items()}
    p['rgb_colors'] = torch.nn.Parameter(p['rgb_colors'])
    optimizer = torch.optim.Adam([p['rgb_colors']],lr=.0025,eps=1e-15)
    fg = np.flatnonzero(source['seg_colors'][:,0].numpy()>.5)
    points = source['means3D'].numpy()[fg]
    distances, indices = cKDTree(points).query(points,k=9,workers=4)
    # Duplicate split centers can put self anywhere among zero-distance ties.
    filtered = [(d[ii!=n][:8],ii[ii!=n][:8]) for n,(d,ii) in enumerate(zip(distances,indices))]
    distances = np.stack([x[0] for x in filtered])
    indices = np.stack([x[1] for x in filtered])
    weights = np.exp(-distances**2/(2*np.maximum(distances[:,-1:]**2,1e-12)))
    weights /= weights.sum(axis=1,keepdims=True)
    fg_index = torch.as_tensor(fg,device='cuda')
    neighbors = torch.as_tensor(fg[indices],device='cuda')
    graph_weight = torch.as_tensor(weights,dtype=torch.float32,device='cuda')
    graph_hash = hashlib.sha256(fg.tobytes()+indices.tobytes()+weights.tobytes()).hexdigest()
    views = dataset(0,json.loads((DATA/'basketball/train_meta.json').read_text()))
    history=[]
    camera_sequence=[]
    started=time.perf_counter()
    for i in range(STEPS):
        index=random.randrange(len(views))
        camera_sequence.append(index)
        batch=views[index]
        raw,_,_=tr.Renderer(batch['cam'])(**h.params2rendervar(p))
        fitted=raw*p['cam_m'][index].exp()[:,None,None]+p['cam_c'][index][:,None,None]
        photo=.8*h.l1_loss_v1(fitted,batch['im'])+.2*(1-tr.calc_ssim(fitted,batch['im']))
        colors=p['rgb_colors']
        delta=colors[fg_index,None,:]-colors[neighbors]
        prior=((torch.sqrt(delta**2+1e-6)-.001).mean(-1)*graph_weight).sum(-1).mean()
        loss=photo+(.01*prior if mode=='spatial_color' else 0)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(colors.grad).all()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i%250==0 or i==STEPS-1:
            row=dict(step=i+1,photometric_loss=float(photo.detach()),spatial_color_penalty=float(prior.detach()),
                elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            save(folder/'training-log.json',history)
            print('APPEARANCE',mode,i+1,round(row['photometric_loss'],6),round(row['spatial_color_penalty'],6),flush=True)
        if (i+1)%1000==0:
            checkpoint=dict(params={k:v.detach().cpu() for k,v in p.items()},step=i+1,
                optimizer=optimizer.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            tmp=folder/'resume.partial'
            torch.save(checkpoint,tmp)
            tmp.replace(folder/'resume.pt')
    result={k:v.detach().cpu() for k,v in p.items()}
    for key in source:
        if key!='rgb_colors':
            assert torch.equal(source[key],result[key]),key
    torch.save(result,folder/'frame-0000.pt')
    save(folder/'completion.json',dict(status='trained_evaluation_pending',mode=mode,steps=STEPS,
        seconds=time.perf_counter()-started,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'protocol.json'),
        fixed_parameters_byte_equal=True,points=len(p['means3D']),graph_sha256=graph_hash,
        camera_sequence_sha256=hashlib.sha256(np.array(camera_sequence,dtype=np.int32).tobytes()).hexdigest(),
        checkpoint_sha256=sha(folder/'frame-0000.pt'),max_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))

def evaluate():
    protocol=prepare()
    assert not (OUT/'evaluation.json').exists()
    completions=[json.loads((OUT/m/'completion.json').read_text()) for m in MODES]
    assert completions[0]['camera_sequence_sha256']==completions[1]['camera_sequence_sha256']
    assert completions[0]['graph_sha256']==completions[1]['graph_sha256']
    tr,h=load_official()
    rows=[]
    assets=OUT/'assets'
    assets.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    all_models=[('source',SOURCE)]+[(m,OUT/m/'frame-0000.pt') for m in MODES]
    params={m:{k:v.cuda() for k,v in torch.load(path,map_location='cpu',weights_only=True).items()} for m,path in all_models}
    for view in dataset(0,json.loads((DATA/'basketball/test_meta.json').read_text())):
        camera=int(view['filename'].split('/')[0])
        gt=view['im'].permute(1,2,0).cpu().numpy()
        images=[gt]
        x0,y0,x1,y1=protocol['evaluation_rois'][str(camera)]
        region=(slice(y0,y1),slice(x0,x1))
        alpha_source=None
        for name,p in params.items():
            rgb,_,alpha=tr.Renderer(view['cam'])(**h.params2rendervar(p))
            if alpha_source is None:alpha_source=alpha
            else:assert torch.equal(alpha_source,alpha), 'Geometry/opacity changed'
            pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy()
            images.append(pred)
            row=dict(mode=name,camera=camera,full_image=metric(pred,gt),person_region=metric(pred[region],gt[region]))
            if camera==15:row['known_shirt_interior']=metric(pred[200:310,275:365],gt[200:310,275:365])
            rows.append(row)
            dest=OUT/f'renders/{name}'
            dest.mkdir(parents=True,exist_ok=True)
            png(pred).save(dest/f'camera{camera}.png')
        labels=['原始观测','原始首帧模型','仅图像损失再训练','增加空间颜色约束']
        # Native pixel size, arranged 2x2 instead of narrowing four images in one row.
        for full in [False,True]:
            chosen=images if full else [im[region] for im in images]
            hh,ww=chosen[0].shape[:2]
            panel=Image.new('RGB',(ww*2,(hh+32)*2),'white')
            draw=ImageDraw.Draw(panel)
            for j,(label,im) in enumerate(zip(labels,chosen)):
                xx=(j%2)*ww; yy=(j//2)*(hh+32)
                panel.paste(png(im),(xx,yy+32))
                draw.text((xx+ww/2,yy+4),label,font=font,anchor='mt',fill='#183747')
            panel.save(assets/f"camera{camera}-{'full' if full else 'person'}.png")
    summary={name:{key:float(np.mean([r['person_region'][key] for r in rows if r['mode']==name])) for key in ['psnr','ssim','reference_gradient_mae']} for name,_ in all_models}
    save(OUT/'evaluation.json',dict(status='evaluated_visual_review_pending',protocol_sha256=sha(OUT/'protocol.json'),
        heldout_cameras=4,frame=0,summary=summary,per_view=rows,alpha_exactly_equal_all_branches=True,
        role=protocol['scope'],limitations=protocol['limitations']))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['train','evaluate'],required=True)
    parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args()
    if args.phase=='train':
        assert args.mode
        train(args.mode)
    else:
        with torch.no_grad():evaluate()
