"""Read-only native-pixel audit of the four remaining selected-view failures."""
import ast,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
import audit_gaussianavatars_params123 as a
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster

ROOT=a.REC/'cross-subject-sh0-124'
FIT=ROOT/'grass-screened-seven124'
OUT=ROOT/'selected-body-point-contract124'
OFFICIAL=a.REC/'smplifyx-reference-control/official'
COCO=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
NAMES={2:'right_shoulder',3:'right_elbow',4:'right_wrist',5:'left_shoulder',6:'left_elbow',7:'left_wrist',9:'right_hip',10:'right_knee',11:'right_ankle',12:'left_hip',13:'left_knee',14:'left_ankle'}


def local(p):
    p=str(p)
    if p.startswith('/opt/'): return Path('//wsl.localhost/Ubuntu-20.04'+p)
    if p.startswith('/mnt/'): return Path(p[5].upper()+':/'+p[7:])
    return Path(p)


def main():
    assert not OUT.exists(), 'One read-only audit; inspect saved evidence instead of repeating.'
    started=time.monotonic();stamp=datetime.now(timezone.utc).isoformat()
    frames=['frame_000004','frame_000012','frame_000058','frame_000070']
    paths=[Path(__file__),FIT/'selected-fit.npz',FIT/'selection-reviewed.json',ROOT/'grass-input/data/observations.npz',
           ROOT/'grass-input/data/inputs.json',OFFICIAL/'smplifyx/utils.py',OFFICIAL/'cfg_files/fit_smplx.yaml']
    hashes={str(p):a.sha(p) for p in paths}
    a.save(OUT/'plan.json',dict(updated=stamp,frames=frames,source_hashes=hashes,
        hypothesis='An existing screen rejection can reflect an ignored/occluded joint rather than a demonstrated wrong limb. Inspect native source and exact mapping before changing a gate or fit.',
        budget=dict(CPU_seconds=120,detector_calls=0,fit_calls=0),
        constraints=['No coordinate/confidence edit','No gate change','No inference of a hidden joint as physical truth','No model promotion from a mapping audit']))
    with np.load(FIT/'selected-fit.npz') as z:
        ids=z['frame_ids'].astype(str).tolist();target=z['observed_native_keypoints'].copy();proj=z['joints_projected_native'].copy()
    rows=a.read(ROOT/'grass-input/data/inputs.json')['records'];rowmap={r['frame']:r for r in rows}
    with np.load(ROOT/'grass-input/data/observations.npz') as z: raw=dict(z)
    assert raw['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    confidence,_=relative_confidence(raw['scores'],raw['boxes'],raw['flip_disagreement'])
    dst=np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    rebuilt=np.zeros((len(rows),118,3),np.float32)
    rebuilt[:,dst,:2]=raw['keypoints'][:,src];rebuilt[:,dst,2]=confidence[:,src]
    rebuilt=np.concatenate([mask_outside_raster(rebuilt[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
    slots=[raw['frame_ids'].astype(str).tolist().index(f) for f in ids]
    assert np.array_equal(target,rebuilt[slots])
    # Evaluate only the pinned, pure mapping function; no torch/model/optimizer.
    tree=ast.parse((OFFICIAL/'smplifyx/utils.py').read_text(encoding='utf-8'))
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='smpl_to_openpose')
    scope={'np':np};exec(compile(ast.Module(body=[func],type_ignores=[]),'<pinned-joint-map>','exec'),scope)
    mapping=scope['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    expected={2:17,3:19,4:21,5:16,6:18,7:20,9:2,12:1}
    assert all(int(mapping[i])==j for i,j in expected.items())
    records=[]
    selected={r['frame']:r for r in a.read(FIT/'selection-reviewed.json')['records']}
    for f in frames:
        i=ids.index(f);row=rowmap[f];impath=local(row['image'])
        assert hashlib.sha256(impath.read_bytes()).hexdigest()==row['source_sha256']
        im=Image.open(impath).convert('RGB');b=row['initial_box_xyxy']
        crop=[max(0,int(np.floor(b[0]))-20),max(0,int(np.floor(b[1]))-10),min(im.width,int(np.ceil(b[2]))+20),min(im.height,int(np.ceil(b[3]))+10)]
        rawim=im.crop(crop);panels=[rawim.copy(),rawim.copy(),rawim.copy()]
        for panel,points in [(panels[1],target[i,:,:2]),(panels[2],proj[i])]:
            draw=ImageDraw.Draw(panel)
            for chain,color in [([5,6,7],(255,70,40)),([2,3,4],(0,170,255)),([5,12,13,14],(255,70,40)),([2,9,10,11],(0,170,255)),([5,2],(255,240,80)),([12,9],(255,240,80))]:
                xy=[tuple(points[k]-np.asarray(crop[:2])) for k in chain]
                draw.line(xy,fill=color,width=2)
                for x,y in xy:draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        canvas=Image.new('RGB',(3*rawim.width,rawim.height))
        for j,panel in enumerate(panels):canvas.paste(panel,(j*rawim.width,0))
        dest=Path('D:/00/记录/005')/f'特征点_草地肩肘髋核验124_{f}_原图_2D红左蓝右_3D投影.png'
        assert not dest.exists();canvas.save(dest)
        joints=[]
        for failure in selected[f]['body_gate_violations']:
            j=failure['index'];raw_index=int(np.flatnonzero(dst==j)[0]);raw_index=int(src[raw_index])
            joints.append(dict(**failure,raw_RTMWholeBody_index=raw_index,SMPLX_joint_index=int(mapping[j]),
                detector_xy=target[i,j,:2].tolist(),model_xy=proj[i,j].tolist(),
                relative_confidence=float(target[i,j,2]),loss_weight_squared=float(target[i,j,2]**2),
                original_joint_weight=0 if j in (1,9,12) else 1,
                source_visibility_reviewed=False))
        records.append(dict(frame=f,native_source_sha256=row['source_sha256'],image=str(dest),image_sha256=a.sha(dest),native_crop=crop,
            no_resize=True,text_on_image=False,failures=joints))
    assert time.monotonic()-started<120 and all(a.sha(p)==h for p,h in hashes.items())
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='native_body_contract_images_ready',seconds=time.monotonic()-started,
        records=records,all_seven_observations_bit_identical=True,original_joint_mapping_verified=expected,
        ignored_original_fit_joints=[1,9,12],source_hashes_unchanged=True,thresholds_changed=False,
        actual_visual_review=False,quality_accepted=False,training_admission=False,new_fit_calls=0)
    a.save(OUT/'result.json',result)
    print({'state':result['state'],'records':records})


if __name__=='__main__':main()
