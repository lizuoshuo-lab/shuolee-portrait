"""Read saved grass fits once; create a reproducible review shortlist, no GPU."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY, select, self_test

ROOT=a.REC/'cross-subject-sh0-124'
OUT=ROOT/'quality-first-eight124'
RUN=ROOT/'grass-temporal-refinement'


def local(p):
    p=str(p)
    if p.startswith('/mnt/'):
        p=p[5].upper()+':/'+p[7:]
    return Path(p)


def main():
    assert not a.lp(OUT/'grass-shortlist.json').exists(), 'Do not silently repeat or overwrite selection'
    stamp=datetime.now(timezone.utc).isoformat()
    fitpath=RUN/'official-fit/fit/fitted.npz'; digest=a.sha(fitpath)
    plan=dict(updated=stamp,policy=POLICY,source_fit=str(fitpath),source_fit_sha256=digest,
        actual_fit_count=94,new_fit_calls=0,gaussian_training_started=False,
        control_results=self_test(), selection_only=True, no_threshold_search=True)
    a.save(OUT/'plan.json',plan)
    rows=a.read(RUN/'official-fit/data/training-inputs.json')['records']; byid={r['frame']:r for r in rows}
    oldrows={r['frame']:r for r in a.read(ROOT/'grass/selection.json')['records']}
    ev=a.read(RUN/'review-094/result.json'); reviews={r['frame']:r for r in ev['records']}
    with np.load(a.lp(fitpath),allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist(); obs=z['observed_native_keypoints']; proj=z['joints_projected_native']
        rotations=z['w2c'][:,:3,:3]@z['root_rotation']; imgsha=z['image_sha256'].astype(str)
    chronological=np.argsort([byid[f]['time_seconds'] for f in ids],kind='stable')
    yaw0=np.arctan2(rotations[:,0,2],rotations[:,2,2]);yaw=np.empty(len(ids));yaw[chronological]=np.unwrap(yaw0[chronological])
    temporal={f:[] for f in ids}
    for t in ev['transitions']:
        speed=t['refinement']['max_local_joint_deg']/t['dt']
        if speed>POLICY['joint_speed_deg_per_second']:
            for f in (t['previous'],t['current']): temporal[f].append('body_pose_jump_over_existing135deg_s')
    records=[]; body=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    for i,f in enumerate(ids):
        r=byid[f]; old=oldrows[f];e=np.linalg.norm(proj[i]-obs[i,:,:2],axis=-1)
        assert imgsha[i]==r['source_sha256']
        h=r['initial_box_xyxy'][3]-r['initial_box_xyxy'][1];assert h>0
        flags=list(temporal[f]); metrics={}
        for name,slots in [('body',body),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w=obs[i,slots,2]**2
            assert w.sum()>0
            metrics[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*e[slots]**2)/w.sum()))
            # All supported source observations remain evaluated, including an occluded hand.
            maximum=float(e[slots][w>0].max())
            metrics[name+'_supported_max_over_height']=maximum/h
            if name in ('body','feet') and maximum/h>POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=f,time_seconds=r['time_seconds'],source_fit_sha256=digest,
            source_image_sha256=imgsha[i],yaw_proxy_deg=float(np.degrees(yaw[i])),
            face_sharpness=old['face_sharpness'],body_confidence=float(obs[i,body,2].mean()),
            screen_flags=flags,**metrics))
    # Record the user's concrete correction, never turn a universal 'hand at chest' rule on.
    decisions={f'frame_{i:06d}':dict(status='rejected',reason='User: hand should lie close to chest; side mesh extends it away.',
        evidence='grass-temporal-handidentity/review-013/page-01.png; arrays identical to this source') for i in range(26,30)}
    a.save(OUT/'grass-review-decisions.json',dict(updated=stamp,source_fit_sha256=digest,records=decisions))
    shortlist=select(records,decisions,admitted_only=False)
    shortlist.update(updated=stamp,source_fit=str(fitpath),source_fit_sha256=digest,policy=POLICY,
        inherited_sharpness_source=str(ROOT/'grass/selection.json'),policy_sha256=a.sha(OUT/'plan.json'))
    a.save(OUT/'grass-shortlist.json',shortlist)
    a.save(OUT/'grass-selection.json',select(records,decisions,admitted_only=True))
    render_shortlist()
    print(dict(shortlist=shortlist['selected_frames'],screened_out=sum(bool(r['screen_flags']) for r in records),
        missing_bins=shortlist['missing_bins'],training_admitted=0,output=str(OUT)))


def render_shortlist():
    shortlist=a.read(OUT/'grass-shortlist.json')
    reviews={r['frame']:r for r in a.read(RUN/'review-094/result.json')['records']}
    boards=[]
    for name in shortlist['selected_frames']:
        review=reviews[name];paths={k:local(v['path']) for k,v in review['images'].items()}
        for k,p in paths.items():assert a.sha(p)==review['images'][k]['sha256']
        # Existing evidence: raw + this one coherent fit; no best-per-frame mixtures.
        bodyim=Image.open(paths['body']).convert('RGB');w=bodyim.width//3
        pair=Image.new('RGB',(w*2,bodyim.height));pair.paste(bodyim.crop((0,0,w,bodyim.height)),(0,0));pair.paste(bodyim.crop((2*w,0,3*w,bodyim.height)),(w,0))
        sideim=Image.open(paths['sides']).convert('RGB');side=sideim.crop((340,480,680,960))
        scale=min(1.,720/pair.height);pair=pair.resize((round(pair.width*scale),round(pair.height*scale)),Image.Resampling.LANCZOS)
        side=side.resize((510,720),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(pair.width+510,max(pair.height,720)));canvas.paste(pair,(0,0));canvas.paste(side,(pair.width,0))
        path=Path('D:/00/记录/005')/f'分析_质量优先选帧124_{name}_原图_同帧拟合_侧面.png'
        assert not a.lp(path).exists();canvas.save(a.lp(path))
        boards.append(dict(frame=name,path=str(path),sha256=a.sha(path),overview_only=True,native_evidence=review['images']))
    a.save(OUT/'grass-review-images.json',boards)
    print('REVIEW_BOARDS_READY',len(boards))


if __name__=='__main__':
    import sys
    render_shortlist() if '--render' in sys.argv else main()
