"""Locate thin triangles at the existing VHAP -> Gaussian boundary; CPU only."""
from pathlib import Path
import time
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT, SNAP, OUT as AUDIT, verify
from audit_teacher_base_mesh123 import normals, quality

OUT = AUDIT/'mesh-initialization-boundary'


def stats(x):
    return dict(min=float(x.min()), p01=float(np.percentile(x, 1)), median=float(np.median(x)))


def main():
    verify(); started=time.monotonic(); OUT.mkdir(exist_ok=False)
    source=ROOT/'fit/fitted.npz'; current=SNAP/'current-meshes.npz'
    cf=AUDIT/'frozen-skeleton-contract/frozen-lbs-cache.npz'
    trial=AUDIT/'teacher-total-offset-100/candidate-diagnostic-only.npz'
    files=[Path(__file__),source,current,cf,trial,SNAP/'binding.npz']
    io.save(OUT/'plan.json',dict(created=io.now(),cpu_only=True,optimizer_updates=0,max_seconds=90,
        hashes={str(p):io.sha(p) for p in files},
        hypothesis='Determine whether the locally fragile triangles predate Gaussian training. Also classify the total-offset trial normal changes relative to its frozen unexpanded reference without relaxing its guard.',
        diagnostic_quality_cutoff=.01,
        cutoff_scope='Descriptive triangle aspect quality only; not an acceptance gate, collision test, teacher filter or point selection.',
        no_Gaussian_training=True,no_model_export=True))
    with np.load(source,allow_pickle=False) as z:fit=dict(z)
    with np.load(current,allow_pickle=False) as z:saved=dict(z)
    with np.load(cf,allow_pickle=False) as z:cache=dict(z)
    with np.load(trial,allow_pickle=False) as z:candidate=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:current_offset=z['static_offset']
    rows=io.read(ROOT/'data/training-inputs.json')['records']
    fit_slots={str(f):i for i,f in enumerate(fit['frame_ids'])}
    saved_slots={str(f):i for i,f in enumerate(saved['frame_ids'])}
    initial=np.stack([fit['vertices_world'][fit_slots[r['frame']]] for r in rows])
    current=np.stack([saved['vertices_camera'][saved_slots[r['frame']]] for r in rows])
    faces=fit['faces']; assert np.array_equal(faces,saved['faces'])
    qinit,qcur=quality(initial,faces),quality(current,faces)
    ni,nc=normals(initial,faces),normals(current,faces)
    relative=(ni*nc).sum(-1)<=0
    records=[]
    for i,r in enumerate(rows):
        records.append(dict(frame=r['frame'],VHAP_import_quality=stats(qinit[i]),Gaussian120k_quality=stats(qcur[i]),
            VHAP_import_faces_below_diagnostic_cutoff=int((qinit[i]<.01).sum()),
            Gaussian120k_faces_below_diagnostic_cutoff=int((qcur[i]<.01).sum()),
            normal_opposed_to_import=int(relative[i].sum())))
    import_offset=np.asarray(fit['static_offset']).reshape(10475,3)
    offset_change=np.linalg.norm(current_offset.reshape(10475,3)-import_offset,axis=-1)
    audit=io.read(AUDIT/'teacher-base-mesh-audit/result.json')
    details=[]
    row_slots={r['frame']:i for i,r in enumerate(rows)}
    for item in audit['failed_face_details']:
        i=row_slots[item['frame']];j=item['triangle']
        details.append(dict(frame=item['frame'],triangle=j,
            VHAP_import_quality=float(qinit[i,j]),Gaussian120k_quality=float(qcur[i,j]),
            normal_dot_import=float(np.dot(ni[i,j],nc[i,j])),
            vertices=item['vertices']))
    slots={str(f):i for i,f in enumerate(cache['frame_ids'])}
    order=np.array([slots[str(f)] for f in candidate['frame_ids']])
    base,after=candidate['vertices_before'],candidate['vertices_after']
    ref=base-(cache['linear'][order]@cache['initial_offset'][None,:,:,None])[...,0]
    nb,na,nr=normals(base,faces),normals(after,faces),normals(ref,faces)
    changed=(nb*na).sum(-1)<=0
    oldneg=(nb*nr).sum(-1)<0;newneg=(na*nr).sum(-1)<0
    classification=dict(normal_changed_against_source=int(changed.sum()),
        reference_opposed_before=int(oldneg.sum()),reference_opposed_after=int(newneg.sum()),
        changed_source_opposed_reference_now_aligned=int((changed&oldneg&~newneg).sum()),
        changed_source_aligned_reference_now_opposed=int((changed&~oldneg&newneg).sum()),
        changed_both_opposed_to_reference=int((changed&oldneg&newneg).sum()),
        changed_both_aligned_to_reference=int((changed&~oldneg&~newneg).sum()),
        caution='Reference-normal changes can include straightening old folds and introducing new folds. These counts are not physical collision or clothing-quality proof. Candidate remains rejected under the original guard.')
    result=dict(state='VHAP_to_Gaussian_mesh_boundary_audited',updated=io.now(),frames=59,
        VHAP_import_quality=stats(qinit),Gaussian120k_quality=stats(qcur),
        VHAP_import_thin_observations=int((qinit<.01).sum()),Gaussian120k_thin_observations=int((qcur<.01).sum()),
        VHAP_import_unique_thin_faces=int((qinit<.01).any(0).sum()),Gaussian120k_unique_thin_faces=int((qcur<.01).any(0).sum()),
        normal_opposed_to_import_observations=int(relative.sum()),failed_face_boundary=details,
        Gaussian120k_offset_change_from_import_m=dict(median=float(np.median(offset_change)),p99=float(np.percentile(offset_change,99)),max=float(offset_change.max())),
        frames_result=records,total_reference_control_classification=classification,
        original_geometric_guard_unchanged=True,new_model_admitted=False,quality_accepted=False,
        optimizer_updates=0,elapsed_cpu_seconds=time.monotonic()-started)
    assert result['elapsed_cpu_seconds']<90
    io.save(OUT/'result.json',result)
    print('MESH_BOUNDARY',result['VHAP_import_thin_observations'],result['Gaussian120k_thin_observations'],details,classification,flush=True)


if __name__=='__main__':main()
