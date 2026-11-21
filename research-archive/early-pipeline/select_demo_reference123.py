"""Rank existing observed frames for a visible, near-frontal initialization."""
import numpy as np
import cv2
from PIL import Image
from preprocess_demo_fps3_123 import DATA,ROOT
from run_official_gaussians123 import read,save,sha,now
from smplify_score_contract123 import relative_confidence
from review_demo_fps3_123 import bounded,join

def main():
    out=ROOT/'reference-selection';out.mkdir(exist_ok=False)
    rows=read(DATA/'data/inputs.json')['records']
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    values=[]
    for i,r in enumerate(rows):
        face=obs['keypoints'][i,23:91];le=face[36:42].mean(0);re=face[42:48].mean(0);nose=face[30]
        eyevec=re-le;eyedist=np.linalg.norm(eyevec)
        imbalance=abs(np.dot(nose-(le+re)/2,eyevec))/(eyedist**2+1e-6)
        def ear(p):return (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*np.linalg.norm(p[0]-p[3])+1e-6)
        raw=Image.open(r['image']).convert('RGB');crop=bounded(np.r_[face.min(0),face.max(0)],raw.size,16)
        blur=float(cv2.Laplacian(cv2.cvtColor(np.array(raw.crop(crop)),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
        values.append(dict(slot=i,frame=r['frame'],source_frame=r['source_frame_index'],
            source_sha256=r['source_sha256'],frontal_eye_nose_imbalance=float(imbalance),
            eye_aspect_min=float(min(ear(face[36:42]),ear(face[42:48]))),
            face_confidence_median=float(np.median(confidence[i,40:91])),
            body_confidence_median=float(np.median(confidence[i,:17])),face_laplacian_var=blur,
            face_crop=crop,frontal_rank_only_not_head_pose_or_quality_truth=True))
    ranked=sorted(values,key=lambda x:x['frontal_eye_nose_imbalance'])
    cells=[]
    for v in ranked[:10]:
        i=v['slot'];r=rows[i];raw=Image.open(r['image']).convert('RGB')
        body=raw.crop(bounded(obs['boxes'][i],raw.size,15));body.thumbnail((300,470))
        face=raw.crop(v['face_crop']);face.thumbnail((220,200))
        panel=Image.new('RGB',(520,480),'white');panel.paste(body,(0,0));panel.paste(face,(300,0))
        cells.append(panel)
        join([raw.crop(bounded(obs['boxes'][i],raw.size,15)),raw.crop(v['face_crop'])]).save(out/f'slot-{i:03d}-candidate.png')
    contact=Image.new('RGB',(1040,2400),'white')
    for k,cell in enumerate(cells):contact.paste(cell,((k%2)*520,(k//2)*480))
    contact.save(out/'candidates.jpg',quality=95)
    save(out/'ranking.json',dict(updated=now(),observations_sha256=sha(DATA/'data/observations.npz'),
        criteria='Rank near-frontal eye/nose symmetry, then actual visual review for eye visibility, blur, device occlusion and complete body; hands not a gate.',
        candidates=ranked,contact_slots=[v['slot'] for v in ranked[:10]],selected=None))
    print([(v['slot'],round(v['frontal_eye_nose_imbalance'],4),round(v['eye_aspect_min'],3)) for v in ranked[:10]])

if __name__=='__main__':main()
