"""Reproduce root-energy accounting from the immutable GPU audit, CPU only."""
from pathlib import Path
import hashlib
import numpy as np
import audit_gaussianavatars_params123 as a
from smplify_score_contract123 import relative_confidence


def main():
    root=a.REC/'demo-vhap-initialization'
    out=root/'smplify-vhap-hand-capacity-audit'
    checkpoint=root/'smplify-vhap-normalized-laplacian/epoch-30/tracked-smplx-params.npz'
    observations=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')
    result=a.read(out/'result.json')
    plan=a.read(root/'smplify-vhap-hand-capacity-plan.json')
    wsl_checkpoint='/mnt/c/'+str(checkpoint).replace('\\','/')[3:]
    assert a.sha(checkpoint)==plan['input_hashes'][wsl_checkpoint]
    obs_hash=hashlib.sha256(observations.read_bytes()).hexdigest()
    assert obs_hash==plan['input_hashes']['/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz']
    obs=dict(np.load(observations,allow_pickle=False))
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    with np.load(a.lp(checkpoint),allow_pickle=False) as z:
        assert list(z['frame_ids'])==list(obs['frame_ids'])==[r['frame'] for r in result['records']]
        records=[]
        for i,item in enumerate(result['records']):
            row=dict(frame=item['frame'],hands={})
            for side,model_start,obs_start in [('left',25,91),('right',46,112)]:
                residual=z['joints_projected_native'][i,model_start:model_start+21]-obs['keypoints'][i,obs_start:obs_start+21]
                energy=np.sum((residual*np.array([2/2704,2/1520]))**2,axis=1)*confidence[i,obs_start:obs_start+21]
                floor=float(energy[0]/max(energy.sum(),1e-24))
                assert 0<=floor<1
                row['hands'][side]=dict(root_energy_fraction=floor,root_residual_px=float(np.linalg.norm(residual[0])),
                    fingers_median_px=float(np.median(np.linalg.norm(residual[1:],axis=1))),
                    pca12_finger_only_unexplained_fraction=float((item['hands'][side]['local_span']['pca12']['residual_energy_fraction']-floor)/(1-floor)),
                    full45_finger_only_unexplained_fraction=float((item['hands'][side]['local_span']['full45']['residual_energy_fraction']-floor)/(1-floor)))
            records.append(row)
    summary={s:{k:float(np.median([r['hands'][s][k] for r in records])) for k in records[0]['hands'][s]} for s in ('left','right')}
    prior=a.read(out/'root-energy-decomposition.json')
    assert summary==prior['summary'] and records==prior['records']
    prior.update(source_sha256=a.sha(__file__),checkpoint_sha256=a.sha(checkpoint),observations_sha256=obs_hash,
                 gpu_result_sha256=a.sha(out/'result.json'),cpu_reproduction_exact=True)
    a.save(out/'root-energy-decomposition.json',prior)
    print('CPU decomposition reproduced exactly for all59x2 hands; source/input hashes retained.')


if __name__=='__main__':main()
