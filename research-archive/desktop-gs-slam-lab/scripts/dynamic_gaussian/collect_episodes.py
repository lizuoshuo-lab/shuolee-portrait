"""Predeclared whole-episode splits; resumable collection, no model/test tuning."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
from PIL import Image
from ficus_env import FicusEnv,LAB
OUT=LAB/'outputs/dynamic-world-model-002'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def protocol():
    return dict(version=1,frames=48,frame_dt=.04,train_seeds=list(range(4100,4116)),validation_seeds=list(range(5100,5104)),test_seeds=list(range(6100,6104)),action_rule='seeded independent piecewise-constant u; holds 2..7 frames, amplitude uniform[-0.9,0.9], every fifth segment zero',state_history=2,prediction_horizons=[1,5,10,25,48],selection_rule='Fit train only; choose ridge regularization by validation rollout error. Open test once after selection. No retuning from test.',baselines=['persistence','no-action autoregression','action-conditioned autoregression','training-set mean by time'],control='Frozen learned model; compare no action, open-loop plan, receding-horizon feedback under same initial state, target and injected velocity disturbance. Privileged compact state input; not a visual world model.')

def action_sequence(seed,n):
    rng=np.random.default_rng(seed);a=[];segment=0
    while len(a)<n:
        value=0. if segment%5==4 else float(rng.uniform(-.9,.9));a.extend([value]*int(rng.integers(2,8)));segment+=1
    return np.asarray(a[:n],np.float64)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke-only',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True)
    path=OUT/'protocol.json';spec=protocol()
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==spec
    else:save(path,spec)
    env=FicusEnv();save(OUT/'environment-contract.json',env.contract())
    # Save actual reset/replay tests, including action sign and invalid-action rejection.
    if not (OUT/'environment-checks.json').exists():
        zero=env.observe();image0=env.render();initial=env.snapshot();seq=[.7,.7,-.3,0.];states=[]
        for a in seq:s,_=env.step(a);states.append(s)
        end1=env.snapshot();reset=env.reset();image1=env.render();fresh=env.snapshot()
        assert np.array_equal(zero,reset) and np.array_equal(image0,image1)
        for k in initial:assert np.array_equal(initial[k],fresh[k]),k
        states2=[]
        for a in seq:s,_=env.step(a);states2.append(s)
        end2=env.snapshot();maxerr=float(np.max(abs(end1['particle_x']-end2['particle_x'])));assert maxerr<1e-5
        env.reset();positive,_=env.step(.9);env.reset();negative,_=env.step(-.9)
        assert env.observation_weights@positive>0 and env.observation_weights@negative<0
        rejected=0
        for bad in [float('nan'),2.]:
            try:env.step(bad)
            except ValueError:rejected+=1
        assert rejected==2
        save(OUT/'environment-checks.json',dict(status='passed',reset_state_exact=True,reset_rgb_exact=True,replay_max_particle_position_difference=maxerr,replay_tolerance=1e-5,positive_action_mean_dx=float(env.observation_weights@positive),negative_action_mean_dx=float(env.observation_weights@negative),invalid_actions_rejected=rejected,scope='4-step same-action replay; finite output-frame state checks; does not assert long-run bitwise determinism'))
    if args.smoke_only:return
    for split in ['train','validation','test']:
        for seed in spec[split+'_seeds']:
            folder=OUT/'episodes'/f'{split}-{seed}';folder.mkdir(parents=True,exist_ok=True);receipt=folder/'receipt.json'
            if receipt.exists():
                r=json.loads(receipt.read_text());assert r['status']=='completed' and hashfile(folder/'trajectory.npz')==r['trajectory_sha256'];continue
            frames=folder/'rgb';frames.mkdir(exist_ok=True);actions=action_sequence(seed,spec['frames']);states=[env.reset()];infos=[];rgb_hashes=[];started=time.perf_counter()
            for i in range(spec['frames']+1):
                rgb=env.render();f=frames/f'{i:04d}.png';Image.fromarray(rgb).save(f);rgb_hashes.append(hashfile(f))
                if i==spec['frames']:break
                obs,info=env.step(actions[i]);states.append(obs);infos.append(info)
            np.savez_compressed(folder/'trajectory.npz',states=np.stack(states),actions=actions,times=np.arange(spec['frames']+1)*env.frame_dt,foliage_readout=env.observation_weights)
            save(folder/'step-info.json',infos)
            save(receipt,dict(status='completed',split=split,seed=seed,transitions=spec['frames'],observations=spec['frames']+1,trajectory_sha256=hashfile(folder/'trajectory.npz'),rgb_sha256=rgb_hashes,wall_seconds=time.perf_counter()-started))
            save(OUT/'progress.json',dict(stage='trajectory_collection',last_completed=f'{split}-{seed}',completed=len(list((OUT/'episodes').glob('*/receipt.json'))),total=24))
            print('EPISODE',split,seed,'seconds',round(time.perf_counter()-started,1),flush=True)
    save(OUT/'dataset-completion.json',dict(status='completed',episodes=24,transitions=24*48,images=24*49,protocol_sha256=hashfile(path),no_world_model_trained_by_collector=True))
    print('DATASET COMPLETE',flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
