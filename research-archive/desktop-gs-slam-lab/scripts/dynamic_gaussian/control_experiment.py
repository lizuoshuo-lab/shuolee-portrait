"""Frozen state-prediction model, matched open/closed-loop bounded MPC."""
from pathlib import Path
import json,time,hashlib,datetime
import numpy as np
from scipy.optimize import lsq_linear
from PIL import Image
from fit_dynamics import OUT,save,sha,predict

class Controller:
    def __init__(self,model,readout,horizon=15,error_scale=.03,action_penalty=.03,slew_penalty=.02):
        self.A=np.array(model['A']);self.B=np.array(model['B']);self.b=np.array(model['b']);self.w=np.array(readout)
        self.M=np.r_[self.A,np.c_[np.eye(8),np.zeros((8,8))]]
        self.U=np.r_[self.B,np.zeros(8)];self.c=np.r_[self.b,np.zeros(8)];self.W=np.r_[self.w,np.zeros(8)]
        self.horizon=horizon;self.error_scale=error_scale;self.slew_penalty=slew_penalty
        G=np.zeros((horizon,horizon))
        for i in range(horizon):
            for j in range(i+1):G[i,j]=self.W@np.linalg.matrix_power(self.M,i-j)@self.U
        self.G=G
        D=np.eye(horizon);D[np.arange(1,horizon),np.arange(horizon-1)]=-1
        self.design=np.r_[G/error_scale,np.sqrt(action_penalty)*np.eye(horizon),np.sqrt(slew_penalty)*D]
    def plan(self,history,targets,previous_action):
        state=history.copy();free=[]
        for _ in range(self.horizon):state=self.M@state+self.c;free.append(self.W@state)
        slew=np.zeros(self.horizon);slew[0]=previous_action
        rhs=np.r_[(targets-np.array(free))/self.error_scale,np.zeros(self.horizon),np.sqrt(self.slew_penalty)*slew]
        sol=lsq_linear(self.design,rhs,bounds=(-1.,1.),tol=1e-9,max_iter=200)
        if not sol.success:raise RuntimeError('Bounded controller failed: '+sol.message)
        return sol.x,dict(optimizer_cost=float(sol.cost),optimality=float(sol.optimality),iterations=sol.nit)

def main():
    from ficus_env import FicusEnv
    import torch
    model_file=OUT/'models/action.json';model=json.loads(model_file.read_text());contract=json.loads((OUT/'environment-contract.json').read_text());w=np.array(contract['foliage_readout'])
    spec=dict(version=1,frames=64,frame_dt=.04,target_onset_state_index=8,target_magnitude=.025,disturbance_step_index=28,disturbance_delta_v_magnitude=.20,disturbance_direction='opposite target sign',horizon=15,error_scale=.03,action_penalty=.03,slew_penalty=.02,action_bounds=[-1,1],policies=['zero','open_loop','closed_loop'],scenarios=['positive_nominal','positive_disturbed','negative_nominal','negative_disturbed'],model_sha256=sha(model_file),open_loop='same receding-horizon optimizer and target preview, driven by its own predicted history after initialization; never receives actual state',closed_loop='same optimizer, receives actual compact state history at each step',disturbance_information='neither controller receives disturbance time/magnitude; only closed loop receives its consequences through observations',evaluation='all-state-index tracking RMSE, target-active RMSE, post-disturbance RMSE indices 29..64; mean squared action and saturation fraction')
    path=OUT/'control-protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec,'Do not silently change frozen control protocol'
    else:save(path,spec)
    controller=Controller(model,w);env=FicusEnv();checks=[]
    # Independent reconstruction of lifted predictions from repeated one-step calls.
    history=np.linspace(-.01,.01,16);actions=np.linspace(-.7,.8,15);hh=history.copy();explicit=[]
    for a in actions:s=predict(model,hh,a);explicit.append(w@s);hh=np.r_[s,hh[:8]]
    hh=history.copy();free=[]
    for _ in range(15):hh=controller.M@hh+controller.c;free.append(controller.W@hh)
    lift_error=float(np.max(abs(np.array(explicit)-(free+controller.G@actions))))
    assert lift_error<1e-10;save(OUT/'controller-algebra-check.json',dict(status='passed',max_lifted_prediction_error=lift_error,tolerance=1e-10))
    n=spec['frames'];horizon=spec['horizon']
    for scenario in spec['scenarios']:
        sign=1 if scenario.startswith('positive') else -1;disturbed=scenario.endswith('disturbed')
        targets=np.zeros(n+horizon+1);targets[spec['target_onset_state_index']:]=sign*spec['target_magnitude']
        for policy in spec['policies']:
            folder=OUT/'control'/f'{scenario}-{policy}';folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/'receipt.json'
            if receipt.exists():assert sha(folder/'trajectory.npz')==json.loads(receipt.read_text())['trajectory_sha256'];continue
            rgbfolder=folder/'rgb';rgbfolder.mkdir(exist_ok=True);initial=env.reset();states=[initial];actions=[];infos=[];plans=[];history=np.r_[initial,initial];prior_action=0.;started=time.perf_counter()
            Image.fromarray(env.render()).save(rgbfolder/'0000.png')
            for k in range(n):
                if policy=='closed_loop':history=np.r_[states[-1],states[max(0,len(states)-2)]]
                if policy=='zero':action=0.;planning={}
                else:plan,planning=controller.plan(history,targets[k+1:k+1+horizon],prior_action);action=float(plan[0]);planning['sequence']=plan.tolist()
                delta_v=-sign*spec['disturbance_delta_v_magnitude'] if disturbed and k==spec['disturbance_step_index'] else 0.
                obs,info=env.step(action,disturbance_delta_v=delta_v);states.append(obs);actions.append(action);infos.append(info);plans.append(planning)
                Image.fromarray(env.render()).save(rgbfolder/f'{k+1:04d}.png')
                if policy=='open_loop':next_state=predict(model,history,action);history=np.r_[next_state,history[:8]]
                prior_action=action
            s=np.array(states);a=np.array(actions);reference=targets[:n+1];y=s@w;err=y-reference
            metrics=dict(tracking_rmse=float(np.sqrt(np.mean(err**2))),target_active_rmse=float(np.sqrt(np.mean(err[8:]**2))),post_disturbance_rmse=float(np.sqrt(np.mean(err[29:]**2))),final_error=float(err[-1]),mean_squared_action=float(np.mean(a*a)),saturation_fraction=float(np.mean(abs(a)>.999)),min_det_F=float(min(r['min_det_F'] for r in infos)),max_lower_displacement=float(max(r['lower_max_displacement'] for r in infos)))
            np.savez_compressed(folder/'trajectory.npz',states=s,actions=a,targets=reference,times=np.arange(n+1)*spec['frame_dt'],foliage_readout=w)
            save(folder/'step-info.json',infos);save(folder/'plans.json',plans)
            save(receipt,dict(status='completed',scenario=scenario,policy=policy,metrics=metrics,wall_seconds=time.perf_counter()-started,trajectory_sha256=sha(folder/'trajectory.npz'),model_sha256=sha(model_file),control_protocol_sha256=sha(path),rgb_sha256=[sha(f) for f in sorted(rgbfolder.glob('*.png'))]))
            save(OUT/'progress.json',dict(stage='matched_control',last_completed=folder.name,completed=len(list((OUT/'control').glob('*/receipt.json'))),total=12))
            print('CONTROL',folder.name,'RMSE',round(metrics['tracking_rmse'],6),'seconds',round(time.perf_counter()-started,1),flush=True)
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'control').glob('*/receipt.json'))]
    save(OUT/'control-evaluation.json',dict(status='completed',frozen_protocol_sha256=sha(path),trials=receipts,interpretation='4 predetermined scenarios, 3 policies each, single material/object; no statistical multi-seed generalization claim'))
    print('CONTROL COMPLETE',flush=True)

if __name__=='__main__':
    import torch
    with torch.no_grad():main()
