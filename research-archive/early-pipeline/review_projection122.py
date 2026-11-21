"""Preserve and compare all frozen N5/N6 errors; native unwarped diagnostic crops."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
H=Path(__file__).parent;T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';B=R/'video-update-20261027/background-geometry';G=B/'shared-path';N=B/'residual-radial';C=N/'serialization-repair'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
base=read(G/'solver-repair/result.json');candidate=read(C/'result.json');old=read(G/'result.json')
assert not (B/'comparison-review.json').exists()
assert candidate['optimizer']['history'][-1]['cost']==read(N/'progress.json')['last']['cost']
assert candidate['k1']==read(N/'progress.json')['last']['k1']
save(N/'serialization-recovery.json',dict(updated=stamp,original_failure_retained=True,same_final_cost_and_k1_exact=True,
    original_progress_sha256=sha(N/'progress.json'),recovered_result_sha256=sha(C/'result.json'),new_model_or_threshold=False,
    first_attempt='Computation completed, final JSON serialization failed on NumPybool; final vector was not saved.',
    recovery='Deterministic identical computation for serialization recovery, not a second selected candidate.'))
# The historical stale running marker must no longer imply a live process.
save(N/'progress.json',dict(state='failed_serialization_recovered',pid=None,updated=stamp,
    original_final_state=read(N/'serialization-repair-plan.json')['original_final_progress'],result='serialization-repair/result.json'))
models={'stalled_pinhole':old['validation']['shared'],'solved_pinhole':base['validation'],'residual_radial':candidate['validation']}
per=[]
for name,groups in models.items():
    for group,v in groups.items():
        for frame in base['actual_registered_frames']:
            rows=[o for o in v['observations'] if o['target']==frame];errors=[o['error_px'] for o in rows if o['valid_projection']]
            per.append(dict(model=name,group=group,frame=frame,n=len(rows),valid=len(errors),median=float(np.median(errors)),max=float(max(errors)),lt1=sum(e<1 for e in errors)))
save(B/'comparison-per-view.json',dict(rows=per,all_validity_retained=True,physical_face_ground_truth=False))
frames=base['actual_registered_frames'];size=320;original=Image.new('RGB',(size*7,size*3),(255,255,255));overlay=original.copy();manifest=[]
obs=candidate['validation']['face_candidates']['observations']
for col,fi in enumerate(frames):
    rows=[o for o in obs if o['target']==fi];center=np.mean([o['observed'] for o in rows],axis=0);left,top=np.rint(center-size/2).astype(int)
    im=Image.open(B/f'frame_{fi:06d}.png').convert('RGB');box=(int(left),int(top),int(left+size),int(top+size));crop=im.crop(box)
    for row,model in enumerate(['stalled_pinhole','solved_pinhole','residual_radial']):
        original.paste(crop,(col*size,row*size));tile=crop.copy();draw=ImageDraw.Draw(tile)
        targets=[o for o in models[model]['face_candidates']['observations'] if o['target']==fi]
        for o in targets:
            x,y=np.array(o['observed'])-[left,top];draw.ellipse((x-2,y-2,x+2,y+2),outline=(0,255,80),width=1)
            if o['valid_projection']:
                px,py=np.array(o['predicted'])-[left,top];draw.line((x,y,px,py),fill=(255,160,0),width=1)
                draw.line((px-3,py,px+3,py),fill=(255,40,180),width=1);draw.line((px,py-3,px,py+3),fill=(255,40,180),width=1)
        overlay.paste(tile,(col*size,row*size))
    manifest.append(dict(frame=fi,source=f'frame_{fi:06d}.png',crop=box,native_pixels=True,resized=False,warped=False))
original.save(B/'face_native_originals.png');overlay.save(B/'face_projection_comparison.png')
for i in range(3):overlay.crop((0,i*size,size*7,(i+1)*size)).save(B/f'face_projection_row{i}.png')
for i in range(7):overlay.crop((i*size,0,(i+1)*size,size*3)).save(B/f'face_frame_{frames[i]:06d}_three_models.png')
save(B/'comparison-image-index.json',dict(crops=manifest,row_order=['stalled_initial_pinhole','numerically_solved_fixedK_pinhole','shared_oneparameter_radial_output_model'],
    green='observed numeric candidate',magenta='target-excluded3D prediction',orange='residual connection',text_on_images=False,observation_identity_certified=False))
review=dict(updated=stamp,state='numeric_review_pending_actual_image_inspection',
    source_hashes={str(p):sha(p) for p in [G/'result.json',G/'solver-repair/result.json',C/'result.json',B/'observations.json']},
    conclusions=[
      'OriginalN5c optimization did not advance: cost1289.530659,gradient155187.411. Correct Jacobian; replacing stopped numerical solve with sameHuber/gauge/data explicitSchur produced18descent steps,cost329.508078,gradient0.031787. This is a confirmed solver termination failure, not proof of optical distortion.',
      'Same conditionalK,backgroundFIT935 and frozenheldout: facecandidate median10.134->3.106px,max23.006->10.142px after successfulsolve. Still below no finalfaceadmission.',
      'Single shared residual radial coefficient k1=0.001493198 changes fixedray cornerpixels by up to2.523px. FITcost329.508->329.321; facecandidate median3.106->3.142px,max10.142->10.061px; backgroundmedian0.435->0.432px,max3.199->3.158px. No meaningful ghosting fix or modelpromotion.',
      'Weighted local radial derivative after pose/point nuisance removal retains only0.01204 of its norm, indicating substantial conditional coupling. Estimated residualfield is not physical distortion calibration.',
      'Hypothesis is only one time-invariant residual radial coefficient in alreadyrectified output domain. Its failure does not rule out decentering,tangential,asymmetric/spatial residualwarp,wrongK,crop,pixel timing,personmicro-motion orwrongmaterial.',
      'PairwiseSampson<=0.704px did not imply shared3D face consistency. All681background and57face validationobservations preserved; target excluded from owntriangulation. Face material/uncertainty notcertified anddata are not unseenexternaltest.'
    ],
    training_started=False,physical_camera_accepted=False,latest119_unchanged=True,
    next='Do not assume either exactpinhole orrawfisheye. Examine perframe andspatial residuals plus support/depth and physical correspondence before registering any richer model; no automatic polynomial sweep.')
save(B/'comparison-review.json',review)
print(json.dumps(dict(images=[str(B/'face_projection_comparison.png')],face_by_view=[p for p in per if p['group']=='face_candidates']),ensure_ascii=False))
