"""Recheck saved physics states on CPU and publish measured visual comparisons."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-gaussian-001';ASSETS=OUT/'assets'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=['zero','half','full','soft','half_dt']
summaries={c:read(OUT/c/'summary.json') for c in cases}
rows={c:read(OUT/c/'measurements.json') for c in cases}
checks=[]
base=None
for c in cases:
    assert len(rows[c])==30 and len(list((OUT/c/'frames').glob('*.png')))==31
    assert summaries[c]['particles']==171553 and summaries[c]['status']=='completed'
    for p in sorted((OUT/c).glob('state-*.npz')):
        with np.load(p) as v:
            F=v['F'].astype(np.float64);cov0=v['initial_covariance'].astype(np.float64)
            expected=F@cov0@F.transpose(0,2,1)
            relative=float(np.linalg.norm(v['covariance']-expected)/np.linalg.norm(expected))
            assert relative<1e-6
            eig=np.linalg.eigvalsh(v['covariance'].astype(np.float64))
            relative_min=float((eig[:,0]/eig[:,-1]).min())
            # Very flat pretrained Gaussians lose strict definiteness when their
            # factorized covariance is materialized in float32. Report this; do
            # not relabel a roundoff-tolerant check as a strict SPD certificate.
            assert relative_min > -3e-7 and np.isfinite(v['position']).all() and np.linalg.det(F).min()>0
            if base is None:base={k:v[k].copy() for k in ['initial_position','initial_covariance','mass']}
            for k in base:assert np.array_equal(base[k],v[k]),(c,k)
            recalc=float(np.sqrt(((v['position'].astype(float)-v['initial_position'])**2).sum(1).mean()))
            row=rows[c][int(p.stem.split('-')[1])-1];assert abs(recalc-row['rms_displacement'])<1e-7
            checks.append(dict(case=c,state=p.name,sha256=digest(p),min_covariance_eigenvalue=float(eig.min()),nonpositive_covariance_count=int((eig[:,0]<=0).sum()),min_eigenvalue_relative_to_max=relative_min,covariance_formula_relative_error=relative,rms_displacement_recomputed=recalc))
# Independently derive occupied-cell volumes and the spatial density override.
x=base['initial_position'];cells=np.floor(x/.04).astype(int);_,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
volume=.04**3/counts[inv];density=np.full(len(x),200.);density[np.all(np.abs(x-np.array([1,1,1.5]))<np.array([1,1,.5]),axis=1)]=70.
assert np.allclose(base['mass'],density*volume,rtol=2e-6,atol=1e-10)
init_hashes=[digest(OUT/c/'frames/0000.png') for c in cases];assert len(set(init_hashes))==1
dt0=read(OUT/'full/config.json');dt1=read(OUT/'half_dt/config.json')
assert dt0['substep_dt']*dt0['boundary_conditions'][1]['num_dt']==dt1['substep_dt']*dt1['boundary_conditions'][1]['num_dt']
assert abs(dt0['grid_v_damping_scale']**400-dt1['grid_v_damping_scale']**800)<1e-10
with np.load(OUT/'full/state-0030.npz') as a,np.load(OUT/'half_dt/state-0030.npz') as b:
    dt_error=float(np.sqrt(((a['position'].astype(float)-b['position'])**2).sum(1).mean()))
full=np.array([r['foliage_mean_dx'] for r in rows['full']]);half=np.array([r['foliage_mean_dx'] for r in rows['half']]);soft=np.array([r['foliage_mean_dx'] for r in rows['soft']]);fine=np.array([r['foliage_mean_dx'] for r in rows['half_dt']])
metrics=dict(status='completed_with_reported_numerical_limits',cases=cases,particles=171553,frames_per_case=31,total_output_frames=155,total_physics_substeps=sum(s['substeps'] for s in summaries.values()),coordinate_units='normalized simulation length; no physical metre calibration',time_units='simulation time; not measured real material time calibration',peak_abs_mean_foliage_dx={c:max(abs(r['foliage_mean_dx']) for r in rows[c]) for c in cases},first_negative_peak_time={c:rows[c][int(np.argmin([r['foliage_mean_dx'] for r in rows[c]]))]['time'] for c in cases},half_force_peak_ratio=float(np.max(abs(half))/np.max(abs(full))),soft_material_peak_ratio=float(np.max(abs(soft))/np.max(abs(full))),half_dt_peak_relative_change=float(abs(np.max(abs(fine))-np.max(abs(full)))/np.max(abs(full))),half_dt_max_foliage_trajectory_difference=float(np.max(abs(fine-full))),half_dt_final_all_particle_position_rmse=dt_error,zero_max_displacement=max(r['max_displacement'] for r in rows['zero']),fixed_lower_region_max_mean_displacement=max(r['lower_mean_displacement'] for c in cases for r in rows[c]),all_initial_images_bitwise_equal=True,independent_state_snapshots=len(checks),all_saved_covariances_strictly_positive_definite=all(r['min_covariance_eigenvalue']>0 for r in checks),initial_covariance_nonpositive_count=int((np.linalg.eigvalsh(base['initial_covariance'].astype(float))[:,0]<=0).sum()),worst_relative_covariance_min_eigenvalue=min(r['min_eigenvalue_relative_to_max'] for r in checks),float32_covariance_roundoff_bound=3e-7,volume_and_mass_independently_matched=True,finite_and_positive_determinant_checks_at_output_frames_only=True)
save(OUT/'metrics.json',metrics);save(OUT/'validation.json',dict(checks=checks,checks_summary=metrics))
font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',23);small=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',19)
labels=['无外力','半强度外力','原强度外力','原外力 · 杨氏模量减半'];panels=[]
for index in range(31):
    canvas=Image.new('RGB',(1600,448),'white');draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(cases[:4]):
        im=Image.open(OUT/c/'frames'/f'{index:04d}.png').convert('RGB').resize((400,400),Image.Resampling.LANCZOS)
        canvas.paste(im,(400*i,48));draw.text((400*i+200,10),labels[i],font=font,fill='#173747',anchor='mt')
    draw.text((14,418),f't = {index*.04:.2f}（仿真时间）',font=small,fill='#385161')
    panels.append(canvas)
panels[0].save(ASSETS/'ficus-interventions.gif',save_all=True,append_images=panels[1:],duration=40,loop=0,optimize=False)
panels[8].save(ASSETS/'ficus-interventions-t032.png')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
colors=['#666666','#0072B2','#D55E00','#009E73'];names=['No force','Half force','Full force','Half stiffness']
for c,name,color in zip(cases[:4],names,colors):
    axes[0].plot([0]+[r['time'] for r in rows[c]],[0]+[r['foliage_mean_dx'] for r in rows[c]],label=name,c=color,lw=1.8)
axes[0].set(xlabel='Simulation time',ylabel='Mean foliage x displacement\n(normalized length)',title='Controlled force and stiffness edits');axes[0].legend(frameon=False,fontsize=8)
axes[1].plot([r['time'] for r in rows['full']],fine-full,color='#6B4C9A',lw=1.8)
axes[1].set(xlabel='Simulation time',ylabel='Half-step minus reference\n(normalized length)',title='Time-step sensitivity with equal impulse')
for ax in axes:ax.axhline(0,c='#aaaaaa',lw=.6);ax.grid(alpha=.15)
fig.savefig(ASSETS/'measured-response.png',dpi=190);plt.close(fig)
source_hashes={str(p.relative_to(LAB)).replace('\\','/'):digest(p) for p in sorted((LAB/'scripts/dynamic_gaussian').glob('*.py'))}
kernel_hashes={str(p.relative_to(LAB)).replace('\\','/'):digest(p) for p in sorted((LAB/'repos/PhysGaussian/mpm_solver_warp').glob('*.py'))}
manifest=dict(status='first_physics_milestone_complete',source_commit=zipfile.ZipFile(OUT/'physgaussian-source.zip').comment.decode(),model=read(OUT/'model-receipt.json'),runtime=dict(torch='2.7.1+cu128',warp='1.10.1',h5py='3.14.0',python='3.10.6'),compatibility=['read-only large-GS Torch environment with isolated .dynamic-deps overlay','zero-copy ownership via public wp.from_torch','occupied-cell volume via Torch rather than Taichi; independent NumPy mass/volume check','gsplat covariance renderer with packed=False; full SH color and inverse material rotation','no changes to official MPM kernel source'],preserved_failures=['Git HTTPS clone reset, recovered from official codeload ZIP with commit identifier','render-packed-failure.log: local gsplat packed background shape assertion','adapter-state-timing-failure.log: validation initially compared pre-G2P F; fixed to renderer F_trial; inverse rotation convention also corrected before published simulation'],source_hashes=source_hashes,kernel_hashes=kernel_hashes,completed=['5 controlled physics runs','7 full original paper figures','Gaussian rendering and independent saved-state verification'],not_completed=['4DGS video training','real material parameter identification','action-conditioned learned world model','closed-loop policy training'],next_acceptance=['Environment reset reproduces same Gaussian state and camera','Force action includes selected material region and duration; log observations, actions and state','Hold out entire action episodes and compare no-action/time-only prediction baselines','Closed-loop evaluation under matched targets, disturbances and action budgets'])
save(LAB/'manifests/dynamic-gaussian-work.json',manifest)
print(json.dumps(metrics,indent=2))
