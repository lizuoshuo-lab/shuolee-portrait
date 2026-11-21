"""Choose equivalent axis-angle branches, anchored at an existing good frame.

No interpolation, fitting, smoothing or orientation changes. For a nonzero
principal vector theta*u, every (theta+2*pi*k)*u represents the same rotation.
The closest branch to an adjacent represented rotation is found analytically.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def unwrap_root_axis_angles(values, anchor):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('Expected finite [N,3] axis angles in radians')
    if not 0<=anchor<len(values):raise ValueError('Invalid existing reference index')
    principal=Rotation.from_rotvec(values).as_rotvec()
    result=np.empty_like(values);result[anchor]=values[anchor]
    for indices,delta in ((range(anchor+1,len(values)),-1),(range(anchor-1,-1,-1),1)):
        for i in indices:
            r=principal[i];theta=np.linalg.norm(r);prev=result[i+delta]
            if theta<1e-8:
                if np.linalg.norm(prev)>np.pi:
                    raise ValueError('Ambiguous identity-axis branch; preserve unknown instead of inventing an axis')
                result[i]=r;continue
            axis=r/theta
            nearest=int(np.rint((np.dot(prev,axis)-theta)/(2*np.pi)))
            # Neighboring integers protect a deterministic tie; this is not a parameter search.
            candidates=[(np.linalg.norm((theta+2*np.pi*k)*axis-prev),abs(k),k) for k in (nearest-1,nearest,nearest+1)]
            k=min(candidates)[2];result[i]=(theta+2*np.pi*k)*axis
    before=Rotation.from_rotvec(values).as_matrix();after=Rotation.from_rotvec(result).as_matrix()
    if np.max(np.abs(before-after))>1e-12:raise ValueError('Rotation changed')
    angle=np.linalg.norm(result,axis=1)
    conditioning=np.abs(np.sinc(angle/(2*np.pi)))
    if conditioning.min()<.2:raise ValueError('Selected branch is close to an exponential-map singularity')
    return result


def checks():
    angles=np.radians([170,175,179,-179,-175,-170])
    values=np.c_[np.zeros(6),angles,np.zeros(6)]
    out=unwrap_root_axis_angles(values,4)
    assert np.array_equal(out[4],values[4])
    assert np.max(np.linalg.norm(np.diff(out,axis=0),axis=1))<np.radians(6)
    assert np.allclose(unwrap_root_axis_angles(values[::-1],1)[::-1],out,atol=1e-12)
    assert np.max(np.abs(Rotation.from_rotvec(values).as_matrix()-Rotation.from_rotvec(out).as_matrix()))<1e-12
    try:
        unwrap_root_axis_angles([[0., 0., 0.], [0., np.radians(190), 0.]],1)
    except ValueError as e:
        assert 'Ambiguous identity-axis branch' in str(e)
    else:
        raise AssertionError('An unknown identity-axis branch must be rejected')
    return dict(branch_crossing_repaired_without_rotation_change=True,anchor_preserved=True,
                chronological_reverse_parity=True,identity_axis_ambiguity_rejected=True,
                note='Representation only; no claim of correct fitted anatomy or improved optimization.')
