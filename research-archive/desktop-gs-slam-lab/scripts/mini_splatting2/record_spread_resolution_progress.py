"""Keep current state factual while preserving all frozen trial analyses."""
import datetime
import json
from pathlib import Path
import shutil
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def main():
    path=ROOT/'native-spread-final-review-001.json'
    if not path.exists():
        data=dict(status='reviewed',created=time.time(),accepted=False,
            analysis=str(ROOT/'native-spread-candidate-review.json'),
            comparison='same-video seed0, against fresh zero-variance wrapper control',
            findings=['Center-depth dispersion and shared SfM agreement improved.',
                'Head/shoulder PSNR decreased0.156/0.313dB; gradient fidelity did not consistently improve.',
                'Opacity coverage decreased; common-mask depth statistics alone are insufficient.',
                'Fixed extrapolation images still show duplicated face/background details and floaters.'],
            reviewed_images=['elevation-00.png','elevation-15.png','elevation-45.png','distance-15.png','distance-45.png'],
            visual_scope='representative fixed-path frames, RGB head comparison and previous all8view003preview; not continuous-video geometric GT',
            disposition='Do not adopt as human-clarity solution or promote by depth-statistic alone.',
            next='Paired lossless resolution ablation with unchanged cameras, priors, masks and losses.',
            no_independent_high_pitch_or_depth_gt=True,
            browser=dict(report='http://127.0.0.1:8820/native-spread.html',
                controls=['weight','error/RGB','region','near/far grouping'],
                images_complete=True,console_errors=[],horizontal_overflow=False,
                path_videos_loaded=True,distance_video_playback_verified=True,
                three_new_models_loaded=True,proof=str(ROOT/'native-spread-browser-proof.png')))
        path.write_text(json.dumps(data,indent=2))
    register=ROOT/'viewer-spread-registration-001/summary.json'
    r=json.loads(register.read_text(encoding='utf-8'));r.update(status='browser_verified',browser_evidence=str(path));register.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    pub=ROOT/'native-spread-publication-queue.json';r=json.loads(pub.read_text());r.update(status='complete',browser_review=str(path));pub.write_text(json.dumps(r,indent=2))
    # Keep the input inventory byte-identical to the predeclared protocol hash.
    r=dict(status='validated_inputs_training_started',used_for_training=True,ready_for_gpu_training=True,
        numerical_validation=str(ROOT/'native-resolution-probe-001/summary.json'),
        training_queue=str(ROOT/'native-resolution-queue.json'),
        visual_review='rectification-review.jpg inspected: corrected full-resolution frames align with prior undistorted images',
        preparation_manifests_are_frozen=True)
    (ROOT/'source-resolution-rectified-002/training-status.json').write_text(json.dumps(r,indent=2))
    manifest=LAB/'manifests/mini-splatting2-optimization.json';m=json.loads(manifest.read_text(encoding='utf-8'))
    m['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['confirmed_user_method']='Mip-Splatting'
    m['local_viewer_models']=15
    queues={n:json.loads((ROOT/n).read_text()) for n in ['native-spread-queue.json','native-spread-acceptance-queue.json','native-spread-publication-queue.json','native-resolution-queue.json']}
    m.setdefault('native_queues',{}).update(queues)
    m['active_job']=[q.get('current',n) for n,q in queues.items() if q['status'] not in ['complete','failed']]
    m['native_spread_stage']=dict(status='reviewed',accepted=False,analysis=str(ROOT/'native-spread-candidate-review.json'),
        review=str(path),report='http://127.0.0.1:8820/native-spread.html',scope='No stable human clarity improvement from seed0variance intervention')
    m['resolution_stage']=dict(protocol=str(ROOT/'native-resolution-protocol-001.json'),
        input_validation=str(ROOT/'native-resolution-probe-001/summary.json'),
        queue=str(ROOT/'native-resolution-queue.json'),training_started=True,accepted=False)
    receipt=ROOT/'spread-package-receipt.json'
    if receipt.exists():m['spread_package']=json.loads(receipt.read_text())
    m['next']=['Finish paired low/fullresolution18000-step runs and render both at identical target resolutions.',
        'Compare real head/shoulder detail and fixed view paths before any adoption; repeat useful candidate.',
        'Keep highpitch/independent capture verification open; no synthetic image can certify unseen geometry.']
    manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint=dict(created=m['updated_at'],goal='active',queues=queues,models=15,
        spread_report=m['native_spread_stage'],resolution_stage=m['resolution_stage'],
        current_training_session=64583,unrelated_windows_asr_pid=18064,
        note='Do not rerun existing queues or replace original models. Resolution stage uses matching lossless controls; no head attention/variance/diffusion added.')
    (ROOT/('progress-checkpoint-'+stamp+'.json')).write_text(json.dumps(checkpoint,indent=2))
    print(json.dumps(dict(active_job=m['active_job'],models=15,spread_accepted=False)))


if __name__=='__main__':main()
