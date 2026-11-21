"""Experimental observation-only identity guard; never a 3D truth classifier.

Finite, raster-checked OpenPose118 observations and real chronological times
are required. Coordinates are preserved. Only ambiguous wrist/hand confidence
can be withheld, until separated, internally consistent observations return.
The upstream SMPLify-X loss, optimizer and temporal term are untouched.
"""
import numpy as np

BODY = (7, 4)
HANDS = (slice(25, 46), slice(46, 67))
MARGIN = .5


def identity_guard(points, times):
    p = np.asarray(points)
    t = np.asarray(times, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (118, 3) or not np.isfinite(p).all():
        raise ValueError('Expected finite [frames,118,3] native observations.')
    if len(t) != len(p) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('One strictly increasing real timestamp per frame.')
    if np.any(p[..., 2] < 0) or np.any(p[..., 2] > 1):
        raise ValueError('Explicit normalized confidence is required.')
    q = p.copy()
    active = np.zeros(2, dtype=bool)
    records = []
    for i, row in enumerate(p):
        hands = [row[s] for s in HANDS]
        body = row[list(BODY)]
        roots = np.array([h[0, :2] for h in hands])
        known = bool(np.all(body[:, 2] > 0) and all(h[0, 2] > 0 for h in hands))
        usable = [h[h[:, 2] > 0, :2] for h in hands]
        known = known and all(len(h) >= 2 for h in usable)
        entered = np.zeros(2, dtype=bool)
        recovered = False
        details = None
        if known:
            spans = [np.linalg.norm(h[:, None]-h[None, :], axis=-1).max() for h in usable]
            scale = float(np.mean(spans))
            if scale > 0:
                centroids = np.array([h.mean(0) for h in usable])
                cluster = float(np.linalg.norm(centroids[0]-centroids[1]))
                body_sep = float(np.linalg.norm(body[0, :2]-body[1, :2]))
                own = np.linalg.norm(roots-body[:, :2], axis=-1)
                other = np.linalg.norm(roots-body[::-1, :2], axis=-1)
                # Recovery needs distinct clusters plus consistent identities.
                recovered = bool(cluster > scale and np.all(own < MARGIN*scale)
                                 and np.all(own+MARGIN*scale < other))
                if recovered:
                    active[:] = False
                for side in range(2):
                    opposite = 1-side
                    # Overlap alone never triggers. Separated body anchors and
                    # a distinctly better opposite association are necessary.
                    evidence = bool(body_sep > scale and cluster < MARGIN*scale
                                    and own[side] > other[side]+MARGIN*scale
                                    and own[opposite] < MARGIN*scale)
                    entered[side] = evidence and not active[side]
                    active[side] |= evidence
                details = dict(hand_span_px=scale, cluster_separation_px=cluster,
                               body_wrist_separation_px=body_sep,
                               own_root_distance_px=own.tolist(),
                               opposite_root_distance_px=other.tolist())
        for side in range(2):
            if active[side]:
                q[i, BODY[side], 2] = 0
                q[i, HANDS[side], 2] = 0
        records.append(dict(index=i, time_seconds=float(t[i]),
                            ambiguous=active.tolist(), entered=entered.tolist(),
                            recovery_evidence=recovered, metrics=details))
    assert np.array_equal(q[..., :2], p[..., :2])
    return q, records


def verify():
    # Positive, actual-contact negative, collapse-persistence and recovery
    # controls. No optimization or data-specific threshold adjustment.
    p = np.zeros((4, 118, 3), dtype=np.float64)
    p[..., 2] = 1
    palm = np.column_stack([np.linspace(0, 20, 21), 8*np.sin(np.linspace(0, np.pi, 21))])
    for row in p:
        row[25:46, :2] = palm
        row[46:67, :2] = palm+[100, 0]
        row[7, :2] = row[25, :2]
        row[4, :2] = row[46, :2]
    p[1:3, 25:46, :2] = p[1:3, 46:67, :2]
    p[2, 7, :2] = p[2, 4, :2]
    old = p.copy(); times = [0, .2, .43, .9]
    q, records = identity_guard(p, times)
    flags = [r['ambiguous'] for r in records]
    allowed = np.zeros(118, bool); allowed[7] = True; allowed[25:46] = True
    checks = dict(identity_enters_persists_recovers=flags==[[False, False],[True, False],[True, False],[False, False]],
                  raw_unchanged=np.array_equal(p, old), coordinates_unchanged=np.array_equal(q[..., :2], p[..., :2]),
                  no_other_confidence_change=np.array_equal(q[:, ~allowed], p[:, ~allowed]),
                  confidence_not_increased=bool(np.all(q[..., 2] <= p[..., 2])))
    contact = p[:1].copy()
    contact[:, 46:67, :2] = contact[:, 25:46, :2]+[8, 0]
    contact[:, 4, :2] = contact[:, 46, :2]
    cq, cr = identity_guard(contact, [0])
    checks['genuine_close_hand_negative'] = np.array_equal(cq, contact) and cr[0]['ambiguous']==[False, False]
    for factor in [.1, 3.7]:
        transformed = p.copy(); transformed[..., :2] = transformed[..., :2]*factor+[143, -57]
        _, rr = identity_guard(transformed, times)
        checks[f'scale_translation_{factor}'] = [r['ambiguous'] for r in rr]==flags
    mirror = p.copy(); mirror[..., 0] *= -1
    mirror[:, [7,4]] = mirror[:, [4,7]].copy()
    mirror[:, 25:67] = np.concatenate([mirror[:,46:67].copy(),mirror[:,25:46].copy()],axis=1)
    _, rr = identity_guard(mirror, times)
    checks['left_right_reflection_equivariance'] = [r['ambiguous'][::-1] for r in rr]==flags
    try:
        identity_guard(p, [0,.2,.2,.9])
    except ValueError:
        checks['duplicate_time_rejected'] = True
    assert all(checks.values()), checks
    return checks
