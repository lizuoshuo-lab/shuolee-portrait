"""Reconcile desktop questions with measured evidence; never imply a new training run."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
HOME=ROOT/'shuolee'
OUT=HOME/'current/company-problem-audit'
PG=HOME/'current/pgsr-surface'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    now=datetime.now(timezone.utc).isoformat()
    m=read(PG/'evaluation/review/metrics.json')
    delta=m['delta_candidate_minus_baseline']
    assert delta is not None, list(m)
    review=dict(id='shuolee-pgsr-surface-20261019',updated=now,status='not_promoted_after_preliminary_review',
        training_completed=True,iterations=30000,gaussians=567187,baseline_gaussians=316044,
        head_delta=delta['head'],person_delta=delta['person'],
        geometry_common_support=m['geometry_common_support'],
        inspected_images=['real-head-comparison.png','novel-view-comparison.png'],
        finding='Fixed sampled views retain facial and eyeglass ghosting. PGSR is smoother in hair and clothing; no clear head improvement.',
        full_continuous_video_perceptual_review=False,human_ratings=0,
        accepted_as_best=False,goal_complete=False,
        caveats=['Different representations, renderers, schedules and budgets: not a count-only causal ablation.',
                 'Development cameras reused; SfM support participated in initialization, not independent geometric GT.',
                 'Coarse development motion masks were repaired from source-only images and applied to both frozen renders; full-unmasked metrics also reported.',
                 'Better common-support median depth agreement coexists with worse mean head error and lower outside-head coverage.'],
        model_deleted=False,next='Audit early density allocation and temporal head consistency before selecting one bounded intervention.',
        evidence='current/pgsr-surface/evaluation/review/metrics.json')
    write(PG/'evaluation/review/preliminary-decision.json',review)
    audit=dict(updated=now,status='research_and_experiment_design_complete',source=read(OUT/'extracted.json'),
        questions=[
            dict(question='Does insufficient person Gaussian count cause elevation-view ghosting?',
                 status='unproven',measured_total_counts=dict(old=149413,current=316044,late_trial=404128,pgsr=567187,remy=727737),
                 person_counts_in_source='Unmeasured estimates: ours30k andRemy300k. Do not use as verified semantic counts.',
                 evidence='More total Gaussians have not by themselves delivered stable face improvements; late trial and PGSR are not accepted.',
                 next='Measure head-contribution weights, projected footprints, opacity and retention through reinitialization/simplification. Compare one allocation rule at a time.'),
            dict(question='Does static fitting average facial movement into ghosting?',status='plausible_not_proven',
                 epipolar=read(ROOT/'head-epipolar-audit-001/summary.json')['aggregate'],
                 next='Estimate head rigid motion relative to fixed scene cameras, then inspect eyes/mouth residuals. Separate reflection, blur, rolling shutter and pose error.',
                 caution='Good corner epipolar agreement does not exclude along-epipolar/depth motion or expression in untracked skin regions.')],
        preferred_order=['MiniGS early head allocation audit','head rigid motion and expression diagnosis','one chosen density or motion intervention','generated detail only after consistency checks'],
        sources=read(OUT/'sources/manifest.json'),
        candidates=[
            dict(name='AbsGS',url='https://github.com/TY424/AbsGS',target='Gradient cancellation hides oversized Gaussians needing splitting',
                 implementation='Per-pixel absolute x/y position gradients before summation, with native rasterizer support. abs of summed gradients is not equivalent.'),
            dict(name='Pixel-GS',url='https://github.com/zhengzhang01/Pixel-GS',target='View-average gradients underweight important covered pixels',
                 implementation='Pixel-count-weighted gradient aggregation plus camera-distance scaling. Keep separate from AbsGS for first comparison.'),
            dict(name='VHAP and GaussianAvatars',url='https://github.com/ShenhanQian/GaussianAvatars',tracker='https://github.com/ShenhanQian/VHAP',
                 target='Canonical head with per-frame pose/expression and mesh-bound Gaussians',
                 requirements='Licensed FLAME2023 with jaw rotation and FLAME masks, quality head tracking, scene/head coordinate alignment',
                 license='CC-BY-NC-SA-4.0 plus explicit commercial-use permission requirement; reference research candidate, not deployment-approved'),
            dict(name='Deformable 3D Gaussians',url='https://github.com/ingra14m/Deformable-3D-Gaussians',
                 target='Canonical Gaussians and time-conditioned deformation without a FLAME body template',
                 implementation='Head-only deformation with static background is a proposed adaptation, not an official verified feature.',
                 caution='Can absorb camera error or overfit time; require temporal holdout and deformation regularization.')],
        no_new_training_started=True,no_large_download=True)
    write(OUT/'analysis.json',audit)
    protocol=dict(updated=now,status='design_pending_instrumentation',training_started=False,
        input='HOVER_VERSA_0102 fixed cameras,51train8development,1514x2694',
        schedule=dict(iterations=18000,densify_until=3000,depth_reinit=2000,simplifications=[3000,8000],aggressive_clone_every=250),
        audit_checkpoints=[1999,2001,2999,3001,7999,8001,18000],
        measurements=['Per-head contribution sum of transmittance-times-alpha, using training-only semantic masks',
                      'Contribution-weighted projected Gaussian major/minor axes and depth spread',
                      'Head contribution and primitive retention before/after reinitialization and simplification',
                      'Separate camera motion, rigid head motion, nonrigid mouth/eye residual and tracking coverage'],
        semantic_masks='Current rectangles include neck/background. They are diagnostic ROIs, not semantic person counts.',
        early_pilot=dict(stop_after=3300,seed=0,comparison='Unchanged baseline versus exactly one instrumented density rule',
                         gate='Neutral-forward/backward test, finite losses, correct mask projection, GPU memory fit and local footprint evidence'),
        full_pilot=dict(iterations=18000,seeds_initial=[0],repeat_only_after='Visual gain on real development and fixed continuous paths',
                        point_budget='Initial comparative peak/final budget within10percent where feasible; otherwise disclose unmatched cost and do a budget-matched follow-up'),
        control=['Identical source photos, cameras, resolution, random view order, RGB/depth/Mip settings, render exposure and SH',
                 'Do not combine AbsGS, Pixel-GS, head loss, deformation and diffusion in the first intervention',
                 'Record numerical backward-repeatability limits of the native renderer'],
        selection_bias='Development set reused. Final promotion requires reserved public upper cameras or independent sequence.',
        motion_gate='Use held-out landmark tracks and similar angle-coverage controls; fewer frames alone may reduce conflicting observations or simply reduce viewpoint coverage.',
        acceptance=['Direct raw Gaussian render improves face/eyeglass structure and subjective clarity',
                    'No new floaters, silhouette tearing or continuous-view flicker',
                    'Full person and background remain usable; same-camera evidence and cost reported'],
        resources='RTX5080 local first; large assets on00; no unverified company-server execution',
        source_report_sha256=digest(OUT/'问题.pptx'))
    write(OUT/'next-experiment.json',protocol)
    state=read(HOME/'work-status.json')
    state.update(updated=now,current_phase='Desktop questions researched; early density allocation and head-motion diagnosis designed',
        current_decision=review,local_training_running=False,
        next_bounded_experiment='Instrument MiniGS steps2000/3000/8000 head allocation and validate temporal head consistency before choosing one intervention',
        latest_problem_audit='current/company-problem-audit/analysis.json')
    state['current_experiment']['preliminary_decision']='current/pgsr-surface/evaluation/review/preliminary-decision.json'
    for x in ['AbsGS','Pixel-GS','VHAP','GaussianAvatars','Deformable-3D-Gaussians']:
        if x not in state['research_alternatives']:state['research_alternatives'].append(x)
    write(HOME/'work-status.json',state)
    ledger=read(HOME/'results-ledger.json')
    ledger['decisions']=[x for x in ledger['decisions'] if x.get('id')!=review['id']]+[review]
    write(HOME/'results-ledger.json',ledger)
    print(json.dumps(dict(status='saved',head_delta=delta['head'],new_training_started=False),ensure_ascii=False))

if __name__=='__main__':main()
