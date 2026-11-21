"""Independent detector comparison on native video pixels; never edits training observations."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',TORCH_HOME='/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-cache')
from pathlib import Path
import sys,json,subprocess,time,hashlib
sys.path.insert(0,'/opt/gs-slam-lab/smpl-x-alignment-20261029/face-diagnostic-deps')
import torch,numpy as np
torch.set_num_threads(4)
import face_alignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
OUT=Path('/mnt/d/Projects/Shuolee-analysis-20261028/face-local-detector');OUT.mkdir(exist_ok=False)
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records'];metadata=json.loads((ROOT/'run.json').read_text())
obs=dict(np.load(ROOT/'data/observations.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
net=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D,device='cpu',face_detector='folder',flip_input=True)
selected=[0,18,37,56,75,94];W,H=1520,2704;ratio=np.array([W,H])/np.array(rows[0]['wh'])
command=['ffmpeg','-nostdin','-hide_banner','-v','error','-i',metadata['video'],'-vf','select=not(mod(n\\,5))','-pix_fmt','rgb24','-vsync','0','-f','rawvideo','pipe:1']
process=subprocess.Popen(command,stdout=subprocess.PIPE);predictions=[];scores=[];rois=[];images=[];started=time.monotonic()
try:
    for i,row in enumerate(rows):
        raw=bytearray()
        while len(raw)<W*H*3:
            block=process.stdout.read(W*H*3-len(raw))
            if not block:raise RuntimeError(f'Native frame missing at {i}')
            raw.extend(block)
        im=np.frombuffer(raw,np.uint8).reshape(H,W,3)
        assert row['source_frame_index']==i*5
        box=(np.array(row['face_box_xyxy']).reshape(2,2)+.5)*ratio-.5
        face,score,_=net.get_landmarks_from_image(im,detected_faces=[box.ravel()],return_landmark_score=True)
        if face is None or len(face)!=1:raise RuntimeError(f'Face ROI model missing at {i}')
        predictions.append(face[0]);scores.append(score[0]);rois.append(box)
        if i in selected:images.append(im.copy())
        if i%20==0:print(json.dumps(dict(frames=i+1,total=len(rows),seconds=time.monotonic()-started)),flush=True)
    assert process.stdout.read(1)==b''
    if process.wait()!=0:raise RuntimeError('Native video decode failed')
finally:
    process.stdout.close()
    if process.poll() is None:process.terminate();process.wait()
native=np.asarray(predictions);decoded=(native+.5)/ratio-.5
np.savez_compressed(OUT/'observations.npz',native_face68=native,decoded_face68=decoded,scores=np.asarray(scores),rois_native=np.asarray(rois),frame_ids=obs['frame_ids'])
difference=np.linalg.norm(decoded[:,17:68]-obs['keypoints'][:,40:91],axis=-1);valid=weights['face']>0
report=dict(frames=len(rows),native_wh=[W,H],analysis_wh=rows[0]['wh'],new_detector='3DFAN-4 2.5D semantic landmarks, face_alignment 1.4.1',
            model_sha256=hashlib.sha256(Path(os.environ['TORCH_HOME']+'/hub/checkpoints/3DFAN4-4a694010b9.zip').read_bytes()).hexdigest(),
            input='Native RGB video; RTMW supplies ROI only',cpu_only=True,flip_averaging=True,
            comparison_median_decoded_px=float(np.median(difference[valid])),comparison_p90_decoded_px=float(np.quantile(difference[valid],.9)),
            observations_changed=False,training_started=False,physical_truth_provided=False,quality_accepted=False,
            limitation='Two learned detectors are not independent physical measurements; different outputs alone do not identify the correct one.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(6,3,figsize=(10,13.2),squeeze=False)
for line,(i,im) in enumerate(zip(selected,images)):
    old=(obs['keypoints'][i,40:91]+.5)*ratio-.5;new=native[i,17:68];keep=weights['face'][i]>0
    pts=np.concatenate([old[keep],new[keep]]);lo=pts.min(0)-20*ratio;hi=pts.max(0)+20*ratio
    for col,(p,title) in enumerate(zip([None,old,new],['Original native frame','Whole-body RTMW face','Face-crop 3DFAN'])):
        ax=axes[line,col];ax.imshow(im,interpolation='none')
        if p is not None:ax.scatter(p[keep,0],p[keep,1],s=15,facecolors='none',edgecolors='#00c584' if col==1 else '#008eff')
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        if line==0:ax.set_title(title,fontsize=10)
        if col==0:ax.set_ylabel(rows[i]['frame'],fontsize=9)
fig.suptitle('Native video pixels | detector comparison only, no training-input replacement',fontsize=12)
fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/'detector-comparison.png',dpi=160);plt.close(fig)
print(json.dumps(report),flush=True)
