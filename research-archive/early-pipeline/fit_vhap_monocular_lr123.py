"""Use upstream monocular LR values with the existing direct-body caller.

Only root rotation differs from the completed direct-body control. No change
to the upstream optimizer, RGB energy, mask term, input, or fitting stages.
"""
import types
import numpy as np
import torch
from PIL import Image
import fit_smplify_vhap_direct_body123 as trial
import fit_smplify_vhap_surface123 as bridge
from run_official_gaussians123 import read, save, sha, now

ROOT = trial.ROOT
OUT = ROOT / 'smplify-vhap-monocular-lr'
PLAN = ROOT / 'smplify-vhap-monocular-lr-plan.json'
PROFILE = dict(base=5e-3, translation=1e-3, rotation=5e-3, expr=5e-2,
               static_offset=5e-4, dynamic_offset=5e-4, camera=5e-3, light=5e-3)
original_make_tracker = trial.base.make_tracker
original_checkpoint = bridge.checkpoint


def make_tracker(out):
    t, rows = original_make_tracker(out)
    before = {k:getattr(t.cfg.lr,k) for k in PROFILE}
    differences = [k for k,v in PROFILE.items() if before[k] != v]
    assert differences == ['rotation'] and before['rotation'] == 1e-3
    for k,v in PROFILE.items(): setattr(t.cfg.lr,k,v)
    configure = t.configure_optimizer
    def checked_optimizer(self, params, lr_scale=1.):
        optimizer = configure(params,lr_scale=lr_scale)
        receipt = []
        for group in optimizer.param_groups:
            members = []
            for name in bridge.PARAMS:
                value = getattr(self,name)
                if any(value is p for p in group['params']): members.append(name)
            if 'global_orient' in members: assert group['lr'] == PROFILE['rotation']*lr_scale
            if 'static_offset' in members: assert group['lr'] == PROFILE['static_offset']*lr_scale
            receipt.append(dict(members=members,lr=group['lr']))
        save(out/('optimizer-scale-'+str(lr_scale)+'.json'),dict(updated=now(),lr_scale=lr_scale,groups=receipt))
        return optimizer
    t.configure_optimizer = types.MethodType(checked_optimizer,t)
    save(out/'monocular-profile.json',dict(updated=now(),profile=PROFILE,before=before,
        changed_keys=differences,batch_size=1,one_view_per_timestep=True,
        source='Original ShenhanQian/VHAP doc/monocular.md -> track.py -> BaseTrackingConfig; original rotation falls in base optimizer group.',
        scope='A full-body adaptation of original head-tracker LR, not a published original full-body monocular preset.',
        global_stage_scale=.1,epoch_decay=.9,mask_weight_unchanged=30,RGB_weight_unchanged=30,
        physical_camera_unchanged=True,quality_accepted=False))
    return t, rows


def checkpoint(t,rows,epoch,frozen):
    original_checkpoint(t,rows,epoch,frozen)
    if epoch not in (0,30): return
    plan=read(PLAN)
    folder=OUT/('rgb-initial' if epoch==0 else 'rgb-final')
    folder.mkdir(exist_ok=False)
    images=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            if row['frame'] not in plan['rgb_visual_roster']: continue
            t.clear_cache()
            total,logs,*rest=t.compute_energy(t.dataset.one(i),stage=None)
            output=rest[-1]
            rgb=output['pred_rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            alpha=output['pred_alpha'][0,0].clamp(0,1).cpu().numpy()
            raw=Image.open(row['image']).convert('RGB')
            gt=np.asarray(raw)/255.
            mask=np.asarray(Image.open(row['mask']).convert('L'))/255.
            masked=np.uint8(np.clip(gt*mask[...,None]+1-mask[...,None],0,1)*255)
            x0,y0,x1,y1=[int(v) for v in row['initial_box_xyxy']]
            box=(max(0,x0-48),max(0,y0-48),min(raw.width,x1+48),min(raw.height,y1+48))
            tiles=[im.crop(box) for im in (raw,Image.fromarray(masked),Image.fromarray(np.uint8(rgb*255)),Image.fromarray(np.uint8(alpha*255)).convert('RGB'))]
            panel=Image.new('RGB',(tiles[0].width*4,tiles[0].height))
            for j,im in enumerate(tiles):panel.paste(im,(j*im.width,0))
            path=folder/(row['frame']+'_raw_masked_target_rendered_rgb_alpha.png')
            panel.save(path)
            images.append(dict(frame=row['frame'],path=str(path),sha256=sha(path),crop=list(box),
                photo_loss=float(logs['photo']),RGB_rendered=True,disturbance_enabled=False))
    t.clear_cache()
    save(folder/'result.json',dict(updated=now(),epoch=epoch,images=images,quality_accepted=False,
        no_optimizer_updates_from_export=True,scope='Actual textured/shaded RGB and alpha, not flat-colored mesh overlay.'))


def main():
    plan=read(PLAN)
    assert plan['wrapper_sha256']==sha(__file__) and plan['changed_keys']==['lr.rotation']
    trial.OUT=OUT
    trial.PLAN=PLAN
    trial.base.make_tracker=make_tracker
    bridge.checkpoint=checkpoint
    trial.main()
    for name in ('result.json','progress.json'):
        d=read(OUT/name)
        d.update(updated=now(),trial='monocular_LR_root_rotation_only',changed_keys=['lr.rotation'],
            original_monocular_lr_values=True,official_fullbody_monocular_reproduction=False,
            fullbody_prior_and_hand_representation_unchanged=True)
        save(OUT/name,d)


if __name__=='__main__':main()
