"""Make reviewed dataset/mask adaptations in the isolated PGSR runtime only."""
import difflib
import json
from pathlib import Path
import shutil
from shuolee_prepare_pgsr import SOURCE,RUNTIME,OUT,HOME,sha,dump


def main():
    assert json.loads((OUT/'runtime.json').read_text())['status']=='imports_verified'
    repo=RUNTIME/'repo'
    before=(repo/'train.py').read_text();text=before
    a=text.index('    # backup main code');b=text.index('    gaussians = GaussianModel',a)
    text=text[:a]+'    # Code and patch are recorded once under the shuolee runtime.\n\n'+text[b:]
    edits=[
        ('import torch.nn.functional as F','import torch.nn.functional as F\nfrom shuolee_pgsr_support import valid_mask, multiview_valid, patch_static_valid, audit, verify_scene'),
        ('    gaussians.training_setup(opt)','    verify_scene(scene)\n    gaussians.training_setup(opt)'),
        ('torch.load(checkpoint)','torch.load(checkpoint, weights_only=False)'),
        ('        ssim_loss = (1.0 - ssim(image, gt_image))','        static_valid = valid_mask(viewpoint_cam)\n        ssim_loss = (1.0 - ssim(image*static_valid+gt_image*(1-static_valid), gt_image))'),
        ('            Ll1 = l1_loss(app_image, gt_image)','            Ll1 = ((app_image-gt_image).abs()*static_valid).sum()/(3*static_valid.sum().clamp_min(1))'),
        ('            Ll1 = l1_loss(image, gt_image)','            Ll1 = ((image-gt_image).abs()*static_valid).sum()/(3*static_valid.sum().clamp_min(1))'),
        ('                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()',
         '                normal_loss = weight * (image_weight * (depth_normal-normal).abs().sum(0) * static_valid[0]).sum()/static_valid.sum().clamp_min(1)'),
        ('                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()',
         '                normal_loss = weight * ((depth_normal-normal).abs().sum(0)*static_valid[0]).sum()/static_valid.sum().clamp_min(1)'),
        ('                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])',
         '                d_mask = d_mask & multiview_valid(viewpoint_cam, nearest_cam, pts_in_nearest_cam, patch_size)\n                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])'),
        ('                if iteration % 200 == 0:', '                if False:  # Avoid accumulating repetitive debug montages.'),
        ('                        mask = ncc_mask.reshape(-1)',
         '                        mask = ncc_mask.reshape(-1) & patch_static_valid(nearest_cam, grid)'),
        ('        loss.backward()', '        assert torch.isfinite(loss), "Nonfinite PGSR loss"\n        loss.backward()'),
        ('            # Log and save','            audit(scene, iteration, image_loss, normal_loss, geo_loss, ncc_loss)\n            # Log and save'),
        ('scene.model_path + "/chkpnt" + str(iteration) + ".pth"','scene.model_path + "/latest.pth"'),
        ('                app_model.save_weights(scene.model_path, iteration)','                app_model.save_weights(scene.model_path, 0)'),
        ('    app_model.save_weights(scene.model_path, opt.iterations)','    app_model.save_weights(scene.model_path, 0)'),
        ('    TENSORBOARD_FOUND = True','    TENSORBOARD_FOUND = False  # Compact JSON evidence replaces repeated TensorBoard images.'),
    ]
    for old,new in edits:
        assert text.count(old)==1,(old,text.count(old))
        text=text.replace(old,new)
    (repo/'train.py').write_text(text)
    diff=''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='official/train.py',tofile='shuolee/train.py'))
    (OUT/'adaptation.patch').write_text(diff)
    shutil.copy2(Path(__file__).with_name('shuolee_pgsr_support.py'),repo/'shuolee_pgsr_support.py')
    # Staging copies only the small calibration/point files. Images are linked,
    # and PGSR's unconditional points3D.ply export cannot touch source SfM files.
    dataset=RUNTIME/'dataset';(dataset/'sparse').mkdir(parents=True)
    source=HOME.parent/'source-resolution-rectified-002/dataset'
    (dataset/'images').symlink_to(source/'images',target_is_directory=True)
    for name in ['cameras.bin','images.bin','points3D.bin']:
        shutil.copy2(source/'sparse/0'/name,dataset/'sparse'/name)
    names=sorted(p.stem for p in (source/'images').glob('*.png'))
    assert len(names)==59
    dump(dataset/'split.json',dict(train=[n for n in names if int(n)%8],test=[n for n in names if int(n)%8==0]))
    dump(OUT/'adaptation.json',dict(status='prepared',official_commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
        changes=['Motion masks applied to RGB, single-view normal, both sides of multi-view geometric and NCC supervision',
                 'One reusable checkpoint; compact progress evidence; no repeated code/debug backups','Local staging for original calibration, linked unmodified photos'],
        unchanged=['Official CUDA renderer','Plane depth/normal definitions','Densification and pruning algorithms','Camera poses and source images'],
        dataset=str(dataset),mask_source=str(HOME.parent/'motion-masks-001'),
        training_sha256=sha(repo/'train.py'),support_sha256=sha(repo/'shuolee_pgsr_support.py'),
        data_source=str(source),source_calibration_sha256={n:sha(source/'sparse/0'/n) for n in ['cameras.bin','images.bin','points3D.bin']}))
    print('PGSR adaptation prepared',flush=True)


if __name__=='__main__':main()
