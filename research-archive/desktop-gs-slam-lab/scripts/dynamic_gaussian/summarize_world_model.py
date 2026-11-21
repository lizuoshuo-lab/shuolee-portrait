"""Independently audit saved episodes and control metrics, then plot actual data."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fit_dynamics import OUT,LAB,save,sha

def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    spec=read(OUT/'protocol.json');controlspec=read(OUT/'control-protocol.json');selection=read(OUT/'model-selection.json');evaluation=read(OUT/'test-evaluation.json');controls=read(OUT/'control-evaluation.json')
    seeds=[set(spec[k+'_seeds']) for k in ['train','validation','test']]
    assert all(not a&b for i,a in enumerate(seeds) for b in seeds[i+1:])
    assert selection['frozen_utc']<evaluation['evaluated_utc']
    checks=[];rgb_checked=0;initial_rgb=[]
    for group,n in [('episodes',48),('control',64)]:
        for folder in sorted((OUT/group).iterdir()):
            receipt=read(folder/'receipt.json');z=np.load(folder/'trajectory.npz');s=z['states'];a=z['actions'];t=z['times'];infos=read(folder/'step-info.json')
            assert sha(folder/'trajectory.npz')==receipt['trajectory_sha256']
            assert s.shape==(n+1,8) and a.shape==(n,) and np.isfinite(s).all() and np.isfinite(a).all() and np.max(abs(a))<=1+1e-10
            assert np.array_equal(s[0],np.zeros(8)) and np.allclose(t,np.arange(n+1)*.04,rtol=0,atol=1e-12)
            assert len(infos)==n and min(i['min_det_F'] for i in infos)>0
            assert all(abs(i['time']-i['solver_time'])<1e-8 for i in infos)
            assert np.allclose([i['action'] for i in infos],a,rtol=0,atol=0)
            assert np.allclose(s[1:]@z['foliage_readout'],[i['foliage_mean_dx'] for i in infos],atol=1e-12)
            files=sorted((folder/'rgb').glob('*.png'));assert len(files)==n+1
            assert [sha(f) for f in files]==receipt['rgb_sha256'];rgb_checked+=len(files);initial_rgb.append(sha(files[0]))
            if group=='control':
                error=s@z['foliage_readout']-z['targets'];assert abs(np.sqrt(np.mean(error**2))-receipt['metrics']['tracking_rmse'])<1e-12
                assert receipt['model_sha256']==sha(OUT/'models/action.json')
                disturb=[i['disturbance_delta_v'] for i in infos];nonzero=np.flatnonzero(disturb).tolist()
                assert nonzero==([28] if receipt['scenario'].endswith('disturbed') else [])
            checks.append(dict(group=group,episode=folder.name,transitions=n,trajectory_sha256=sha(folder/'trajectory.npz'),min_det_F=min(i['min_det_F'] for i in infos)))
    assert len(checks)==36 and len(set(initial_rgb))==1
    for name in ['action','no_action']:
        assert selection['model_hashes'][name]==sha(OUT/f'models/{name}.json')
        scores=selection['candidate_scores'][name];best=min(scores,key=lambda row:row['validation_normalized_mse']);assert best['alpha']==selection['selected'][name]['alpha']
    # Identical open-loop actions with and without an unannounced disturbance.
    action_pair_errors={}
    for sign in ['positive','negative']:
        a=np.load(OUT/f'control/{sign}_nominal-open_loop/trajectory.npz')['actions']
        b=np.load(OUT/f'control/{sign}_disturbed-open_loop/trajectory.npz')['actions']
        assert np.array_equal(a,b);action_pair_errors[sign]=float(np.max(abs(a-b)))
    by={r['scenario']+'-'+r['policy']:r['metrics'] for r in controls['trials']}
    improvement={}
    for scenario in controlspec['scenarios']:
        op=by[scenario+'-open_loop'];cl=by[scenario+'-closed_loop']
        improvement[scenario]=dict(tracking_rmse_reduction=1-cl['tracking_rmse']/op['tracking_rmse'],post_disturbance_rmse_reduction=1-cl['post_disturbance_rmse']/op['post_disturbance_rmse'],open_loop=op,closed_loop=cl,zero=by[scenario+'-zero'])
    metrics=dict(status='completed',training_episodes=16,validation_episodes=4,test_episodes=4,dataset_transitions=1152,dataset_rgb_frames=1176,control_trials=12,control_transitions=768,control_rgb_frames=780,particles=171553,action_model_alpha=selection['selected']['action']['alpha'],action_spectral_radius=selection['selected']['action']['augmented_spectral_radius'],test=evaluation['metrics'],control=improvement,units='normalized length and simulator time; no real-world calibration',visual_world_model_trained=False,dreamerv3_trained=False,closed_loop_uses_privileged_state=True)
    save(OUT/'metrics.json',metrics)
    save(OUT/'validation.json',dict(status='passed',episodes_checked=checks,rgb_hashes_checked=rgb_checked,all_initial_rgb_identical=True,split_seed_disjoint=True,selection_precedes_test=True,open_loop_actions_identical_across_disturbance=action_pair_errors,checks_scope='saved compact states, frame-level MPM diagnostics, all image/file hashes, control metric recomputation, frozen models and selection rule; not a guarantee at every internal physics substep'))
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    colors={'action':'#0072B2','no_action':'#D55E00','persistence':'#7A7A7A','time_mean':'#009E73'}
    labels={'action':'动作条件预测','no_action':'无动作预测','persistence':'状态不变','time_mean':'训练集时间均值'}
    z=np.load(OUT/'test-predictions.npz');w=np.array(read(OUT/'environment-contract.json')['foliage_readout']);t=np.arange(49)*.04
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    axes[0].plot(t,z['6100_truth']@w,color='#172F3D',lw=2.3,label='物理仿真真值')
    for name in ['action','no_action']:axes[0].plot(t,z[f'6100_{name}']@w,color=colors[name],lw=1.8,ls='--' if name=='no_action' else '-',label=labels[name])
    axes[0].set(title='固定首个留出回合：从初态连续预测',xlabel='仿真时间',ylabel='枝叶平均横向位移（归一化长度）');axes[0].legend(frameon=False,fontsize=9)
    for name in labels:axes[1].plot(np.array(spec['prediction_horizons'])*.04,[evaluation['metrics'][name][str(h)]['foliage_rmse'] for h in spec['prediction_horizons']],marker='o',lw=1.8,label=labels[name],color=colors[name])
    axes[1].set(title='全部 4 个留出回合：预测终点误差',xlabel='预测时长（仿真时间）',ylabel='位移 RMSE（归一化长度）');axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.17)
    fig.savefig(assets/'action-prediction.png',dpi=180);plt.close(fig)
    policylabels={'zero':'无动作','open_loop':'开环预测控制','closed_loop':'闭环预测控制'};policycolors={'zero':'#777777','open_loop':'#D55E00','closed_loop':'#0072B2'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for ax,scenario,title in zip(axes,['positive_disturbed','negative_disturbed'],['正向目标 + 反向扰动','负向目标 + 反向扰动']):
        for policy in policylabels:
            d=np.load(OUT/f'control/{scenario}-{policy}/trajectory.npz');ax.plot(d['times'],d['states']@w,color=policycolors[policy],lw=1.8,label=policylabels[policy])
        ax.step(d['times'],d['targets'],where='post',color='#172F3D',ls='--',lw=1.5,label='目标位移');ax.axvline(1.12,color='#8B5B99',ls=':',lw=1.7,label='施加速度扰动')
        ax.set(title=title,xlabel='仿真时间',ylabel='枝叶平均横向位移（归一化长度）');ax.grid(alpha=.17);ax.legend(frameon=False,fontsize=8.5,ncol=2)
    fig.savefig(assets/'closed-loop-response.png',dpi=180);plt.close(fig)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',19);small=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',16);frames=[]
    for i in range(65):
        canvas=Image.new('RGB',(960,372),'white');draw=ImageDraw.Draw(canvas)
        for j,policy in enumerate(policylabels):
            canvas.paste(Image.open(OUT/f'control/positive_disturbed-{policy}/rgb/{i:04d}.png').convert('RGB'),(320*j,34));draw.text((160+320*j,5),policylabels[policy],font=font,fill=policycolors[policy],anchor='mt')
        text=f't = {i*.04:.2f}   目标位移：{0 if i<8 else .025:.3f}';text+='   外部速度扰动已施加' if i>=29 else ''
        draw.text((14,351),text,font=small,fill='#173747');frames.append(canvas)
    frames[0].save(assets/'feedback-comparison.gif',save_all=True,append_images=frames[1:],duration=40,loop=0,optimize=False)
    source_hashes={str(p.relative_to(LAB)).replace('\\','/'):sha(p) for p in (LAB/'scripts/dynamic_gaussian').glob('*.py')}
    manifest=dict(status='state_dynamics_and_feedback_milestone_complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=source_hashes,protocol_sha256=sha(OUT/'protocol.json'),fit_protocol_sha256=sha(OUT/'fit-protocol.json'),control_protocol_sha256=sha(OUT/'control-protocol.json'),completed=['resettable Gaussian MPM environment','24 whole-episode trajectories','validation-selected action-conditioned and action-free state predictors','four held-out action episodes','12 matched feedback-control trials','independent saved-data audit'],not_completed=['image-to-state encoder','learned RGB future prediction','DreamerV3 actor/critic training','4DGS video reconstruction training','held-out material/object/camera generalization'],next_acceptance=['train a visual observation model using training episodes only','freeze encoder using validation data before a fresh independent evaluation','compare observed-state control against image-estimated-state control','report long-horizon and material-shift failures separately'])
    save(LAB/'manifests/dynamic-world-model-work.json',manifest)
    save(OUT/'completion.json',dict(status='local_research_complete_website_pending',metrics_sha256=sha(OUT/'metrics.json'),validation_sha256=sha(OUT/'validation.json'),source_hashes=source_hashes))
    print(json.dumps({'status':'passed','images_checked':rgb_checked,'control':{k:{m:round(v[m],4) for m in ['tracking_rmse_reduction','post_disturbance_rmse_reduction']} for k,v in improvement.items()}},indent=2))

if __name__=='__main__':main()
