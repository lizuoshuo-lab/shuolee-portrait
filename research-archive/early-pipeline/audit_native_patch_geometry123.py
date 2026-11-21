"""Read-only missing-affine-direction audit; no observed coordinates are changed."""
from pathlib import Path
import json,signal,time
import numpy as np
from native_patch_control123 import R,read,save,sha,local,now,gray,crop_context,make_spline,TranslationPhotometry
D=R/'video10-correspondence-contrast/head-roi-contrast';O=D/'native-photometric-contrast'

def main():
    assert not (O/'geometry-audit-plan.json').exists()
    p=read(D/'plan.json');result=read(O/'result.json')
    plan=dict(registered=now(),source_sha256=sha(Path(__file__)),result_sha256=sha(O/'result.json'),
        hypothesis='Translation-only patch matching ignores scale, shear and rotation between viewpoints. Examine whether its remaining residual has a material first-order component along these four missing affine directions.',
        operation='For all144 saved patch fits reconstruct native31x31 residual/Jacobian at saved estimates. Project out existing translation/gain/bias, then least-squares project residual into4additional local affine derivatives. Do not apply any proposed step. Report all patches, including boundaries and failures.',
        prespecified_evidence='At least2 of the4 originally cycle-passing rows each have at least1 patch with >=25percent residual-energy reduction after existing directions are projected out, missing-affine corner displacement>0.5px, all4affine coefficients<0.25absolute and fullrank. This only supports a separately registered affine control, not a real-data success claim.',
        budget=dict(CPU_seconds=60,patches=144,real_parameter_updates=0,training=0),stop='One local linearization; no patch-size/threshold scan or camera changes.')
    save(O/'geometry-audit-plan.json',plan);start=time.monotonic();images={int(f):gray(local(c['source'])) for f,c in p['crops'].items()};grid=np.arange(-15,16);xx,yy=np.meshgrid(grid,grid);xx=xx.ravel();yy=yy.ravel();rows=[]
    def alarm(*a):raise TimeoutError('60second read-only geometry audit budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(60)
    try:
        for row in result['rows']:
            details=[]
            for edgeid,e in enumerate(row['refined']['edges']):
                for direction,q in enumerate(e['patches']):
                    if not q['attempted']:
                        details.append(dict(edge=edgeid,direction=direction,attempted=False,qualifies=False));continue
                    a,b=q['source'],q['target'];si,sc,_=crop_context(images[a],q['source_uv']);ti,tc,_=crop_context(images[b],q['seed_uv']);m=TranslationPhotometry(make_spline(si),make_spline(ti),sc,tc)
                    estimate=np.array(q['estimate']);r=m.fun(estimate);j=m.jac(estimate)
                    A=np.column_stack([j[:,0]*xx,j[:,0]*yy,j[:,1]*xx,j[:,1]*yy]);rp=r-j@np.linalg.lstsq(j,r,rcond=None)[0];Ap=A-j@np.linalg.lstsq(j,A,rcond=None)[0]
                    delta,_,rank,svals=np.linalg.lstsq(Ap,-rp,rcond=None);rr=rp+Ap@delta;den=float(rp@rp);fraction=float(1-(rr@rr)/den) if den>1e-20 else 0
                    corners=np.array([[-15,-15],[-15,15],[15,-15],[15,15]]);displacements=corners@delta.reshape(2,2).T;corner=float(np.linalg.norm(displacements,axis=-1).max())
                    qualifies=bool(rank==4 and fraction>=.25 and corner>.5 and np.abs(delta).max()<.25)
                    details.append(dict(edge=edgeid,direction=direction,attempted=True,source=a,target=b,rank=int(rank),missing_affine_delta=delta.tolist(),
                        projected_residual_energy_fraction=fraction,corner_displacement_px=corner,original_at_bound=q['at_bound'],original_success=q['success'],qualifies=qualifies))
            rows.append(dict(row=row['row'],family=row['family'],split=row['split'],baseline_pass=row['baseline']['eligible'],qualifying_patches=sum(q['qualifies'] for q in details),patches=details))
        support=sum(r['baseline_pass'] and r['qualifying_patches']>0 for r in rows)
        out=dict(updated=now(),state='complete',seconds=time.monotonic()-start,rows=rows,baseline_pass_rows=4,baseline_pass_rows_supporting_affine=support,preregistered_support=support>=2,
            no_steps_applied=True,real_camera_or_correspondence_changes=False,limitation='A local linear residual direction does not establish true viewpoint geometry or physical matching. No affine optimizer has run.')
        save(O/'geometry-audit.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
    finally:signal.alarm(0)

if __name__=='__main__':main()
