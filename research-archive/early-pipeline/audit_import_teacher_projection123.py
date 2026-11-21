"""Separate pixel discrepancy from depth discrepancy of the saved teacher."""
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,OUT as AUDIT,verify
from probe_import_teacher_contract123 import weighted_quantile

CONTRACT=AUDIT/'original-import-teacher-contract'
OUT=CONTRACT/'pixel-depth-audit'
FRAMES=['frame_000033','frame_000015','frame_000044','frame_000054']


def main():
    verify();started=time.monotonic();OUT.mkdir(exist_ok=False)
    cf=CONTRACT/'original-import-lbs-cache.npz';tf=CONTRACT/'teacher-support-diagnostic-only.npz'
    renders=io.read(AUDIT/'mesh-initialization-boundary/render-result.json')
    io.save(OUT/'plan.json',dict(created=io.now(),cpu_only=True,optimizer_updates=0,frames=59,max_seconds=90,
        hashes={str(p):io.sha(p) for p in [Path(__file__),cf,tf,CONTRACT/'result.json']},
        hypothesis='A small image-plane discrepancy need not imply agreement in3D. Measure the existing point-to-bound-triangle displacement along and transverse to its camera ray without changing selection or geometry.',
        no_new_threshold_or_point_selection=True,no_claim_of_depth_ground_truth=True,
        figures='Same raw frame / previous native VHAP import projection / diagnostic footpoint-to-Gaussian projection. Cyan=footpoint;red=Gaussian center. Lines are not measured physical tracks. No text or resampling.'))
    with np.load(cf,allow_pickle=False) as z:cache=dict(z)
    with np.load(tf,allow_pickle=False) as z:teacher=dict(z)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];assert np.array_equal(teacher['frame_ids'],[r['frame'] for r in rows])
    faces=cache['faces'];indices=faces[teacher['binding']];records=[];images=[]
    allpixels=[];alllength=[];allray=[];alltangent=[];allweights=[]
    folder=OUT/'review';folder.mkdir();dest=Path('/mnt/d/00/记录/demo')
    for i,row in enumerate(rows):
        assert time.monotonic()-started<90
        weight=teacher['weights'][i];use=weight>0
        target=teacher['targets'][i];foot=(cache['base_vertices'][i][indices]*teacher['barycentric_per_frame'][i,:,:,None]).sum(1)
        E,K=np.asarray(row['w2c']),np.asarray(row['K'])
        tc=target@E[:3,:3].T+E[:3,3];fc=foot@E[:3,:3].T+E[:3,3]
        ray=tc/np.linalg.norm(tc,axis=-1,keepdims=True).clip(1e-20)
        diff=tc-fc;along=(diff*ray).sum(-1);tangent=np.linalg.norm(diff-along[:,None]*ray,axis=-1)
        pt=tc@K.T;pf=fc@K.T;ut=pt[:,:2]/pt[:,2:3].clip(1e-20);uf=pf[:,:2]/pf[:,2:3].clip(1e-20)
        pixels=np.linalg.norm(ut-uf,axis=-1);length=np.linalg.norm(diff,axis=-1)
        records.append(dict(frame=row['frame'],teacher_to_footpoint_pixels=weighted_quantile(pixels,weight),
            distance3D_mm=weighted_quantile(length*1000,weight),along_view_ray_abs_mm=weighted_quantile(np.abs(along)*1000,weight),
            transverse_mm=weighted_quantile(tangent*1000,weight),
            along_view_ray_energy_fraction=float((weight*along**2).sum()/(weight*length**2).sum())))
        allpixels.append(pixels);alllength.append(length);allray.append(np.abs(along));alltangent.append(tangent);allweights.append(weight)
        if row['frame'] not in FRAMES:continue
        old=next(x for x in renders['images'] if x['frame']==row['frame']);assert io.sha(old['record_path'])==old['sha256']
        raw=Image.open(row['image']).convert('RGB');mask=np.array(Image.open(row['mask']).convert('L'))>127
        overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        for j in np.argsort(weight[use]):
            idx=np.flatnonzero(use)[j];x,y=ut[idx];a,b=uf[idx]
            draw.line((float(a),float(b),float(x),float(y)),fill=(255,165,0),width=1)
            draw.ellipse((float(a-1),float(b-1),float(a+1),float(b+1)),fill=(0,220,230))
            draw.ellipse((float(x-1),float(y-1),float(x+1),float(y+1)),fill=(250,60,60))
        box=old['crop_integer_xyxy'];w=box[2]-box[0];h=box[3]-box[1]
        oldsheet=Image.open(old['record_path']).convert('RGB');assert oldsheet.size==(w*3,h)
        panels=[raw.crop(box),oldsheet.crop((w,0,w*2,h)),overlay.crop(box)]
        sheet=Image.new('RGB',(w*3,h))
        for k,im in enumerate(panels):sheet.paste(im,(w*k,0))
        name=f'分析_原网格教师对应_{row["frame"]}_原图_VHAP网格_青色网格红色高斯.png'
        path=folder/name;sheet.save(path);record=dest/name;assert not record.exists();record.write_bytes(path.read_bytes())
        images.append(dict(frame=row['frame'],path=str(path),record_path=str(record),sha256=io.sha(path),
            no_text_added=True,no_resampling=True,occlusion_scope='Target points contributed to old rendered foreground; connectingfootpoints on the import mesh have no independently validated visibility or material identity.'))
    W=np.array(allweights)
    result=dict(state='completed_pixel_depth_discrepancy_audit',updated=io.now(),frames=59,images=images,frames_result=records,
        teacher_to_footpoint_pixels=weighted_quantile(np.array(allpixels),W),
        teacher_to_footpoint_distance3D_mm=weighted_quantile(np.array(alllength)*1000,W),
        along_view_ray_abs_mm=weighted_quantile(np.array(allray)*1000,W),transverse_mm=weighted_quantile(np.array(alltangent)*1000,W),
        along_view_ray_energy_fraction=float((W*np.array(allray)**2).sum()/(W*np.array(alllength)**2).sum()),
        physical_surface_or_3D_feature_truth=False,independent_validation=False,
        no_new_training_or_Gaussian=True,optimizer_updates=0,elapsed_seconds=time.monotonic()-started,quality_accepted=False)
    io.save(OUT/'result.json',result)
    print('TEACHER_PIXEL_DEPTH',result['teacher_to_footpoint_pixels'],result['teacher_to_footpoint_distance3D_mm'],result['along_view_ray_energy_fraction'],flush=True)


if __name__=='__main__':main()
