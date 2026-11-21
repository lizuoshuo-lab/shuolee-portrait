"""Single-frame admission and argument bridge for the pinned SMPLify-X entry.

The two original init inputs must be reviewed separately. Positive detector
scores are not evidence that a joint or a depth edge is physically supported.
No choice of substitute anchors, new loss, or sequence solver is made here.
"""
import ast
import inspect

import numpy as np

import official_smplifyx_entry123 as original


PIN = '9dd537a60bffb00263214c6e27a3a132136a8b588662596f83681c05569e213f'
ORIGINAL_JOINTS = (9, 12, 2, 5)
ORIGINAL_EDGES = ((5, 12), (2, 9))


def checked_arguments(keypoints, image_wh, *, init_joints, depth_edges,
                      reviewed_joints, reviewed_depth_edges):
    """Return only source-reviewed original init arguments for one frame."""
    xy = np.asarray(keypoints)
    if xy.shape != (1, 118, 3):
        raise ValueError('Expected one native-pixel 118-point input.')
    w, h = image_wh
    ids, edges = tuple(init_joints), tuple(tuple(e) for e in depth_edges)
    supported = set(reviewed_joints)
    valid_edges = {frozenset(e) for e in reviewed_depth_edges}
    if len(ids) < 4 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 25 for i in ids):
        raise ValueError('Camera init needs at least four distinct BODY25 anchors.')
    if not (set(ids) | {2, 5}) <= supported:
        raise ValueError('Unverified camera-init anchor or shoulder orientation input.')
    if len(edges) < 2 or len({frozenset(e) for e in edges}) != len(edges):
        raise ValueError('Two distinct source-reviewed depth edges are required.')
    for e in edges:
        if len(e) != 2 or not all(type(i) is int and i in ids for i in e) or e[0] == e[1] or frozenset(e) not in valid_edges:
            raise ValueError('Unverified or invalid depth edge; no fallback estimate.')
    selected = xy[0, sorted(set(ids) | {2, 5}), :2]
    if not np.isfinite(selected).all() or not ((selected >= 0) & (selected < (w, h))).all():
        raise ValueError('Camera-init anchors must be finite native-raster pixels.')
    if np.linalg.matrix_rank(selected - selected.mean(axis=0)) != 2:
        raise ValueError('Camera-init image anchors are collinear.')
    if any(np.array_equal(xy[0,a,:2], xy[0,b,:2]) for a,b in edges):
        raise ValueError('Zero-length depth edge.')
    return ids, edges


def check_seed_geometry(points, seed, ids, edges):
    """Reject a degenerate original seed before the camera optimizer exists."""
    p = np.asarray(points, dtype=np.float64)[0]
    if any(np.array_equal(p[a],p[b]) for a,b in edges):
        raise ValueError('Zero-length model depth edge.')
    q = p[list(ids)] + np.asarray(seed, dtype=np.float64)[0]
    if not np.isfinite(q).all() or not (q[:,2] > 0).all():
        raise ValueError('Original seed does not give finite positive-depth anchors.')
    x,y,z = q.T
    zero = np.zeros_like(z)
    proj = np.stack((np.stack((1/z,zero,-x/z**2),axis=1),
                     np.stack((zero,1/z,-y/z**2),axis=1)),axis=1)
    eye = np.broadcast_to(np.eye(3),(len(q),3,3))
    rotation = np.cross(eye, q[:,None,:]).transpose(0,2,1)
    # Shifting the rotation pivot changes only columns in the translation span.
    jac = (proj @ np.concatenate((eye,rotation),axis=2)).reshape(-1,6)
    if np.linalg.matrix_rank(jac) != 6:
        raise ValueError('Reviewed seed anchors do not constrain all local rigid degrees of freedom.')


def compiled_entry(ids, edges):
    """Bind just two kwargs at the existing fit_single_frame call site."""
    assert original.sha(original.__file__) == PIN
    tree = ast.parse(inspect.getsource(original.fit_one))
    hook, = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(k.arg == 'body_tri_idxs' for k in n.keywords)]
    depth, = [k for k in hook.keywords if k.arg == 'body_tri_idxs']
    assert ast.literal_eval(depth.value) == [list(e) for e in ORIGINAL_EDGES]
    assert hook.keywords[-1].arg is None and hook.keywords[-1].value.id == 'config'
    depth.value = ast.Name(id='_reviewed_depth_edges', ctx=ast.Load())
    hook.keywords.insert(-1, ast.keyword(arg='init_joints_idxs',
                                        value=ast.Name(id='_reviewed_init_joints', ctx=ast.Load())))
    def checked_loader(root):
        utils,prior,fitting,optimizer = original.load_official(root)
        guess = fitting.guess_init
        def checked_guess(model, xy, edge_indices, **kwargs):
            captured = []
            def capture(*args, **kw):
                value = model(*args, **kw)
                captured.append(value.joints.detach().cpu().numpy())
                return value
            seed = guess(capture, xy, edge_indices, **kwargs)
            assert len(captured) == 1 and tuple(map(tuple,edge_indices)) == edges
            check_seed_geometry(captured[0], seed.detach().cpu().numpy(), ids, edges)
            return seed
        fitting.guess_init = checked_guess
        return utils,prior,fitting,optimizer
    scope = dict(vars(original), _reviewed_init_joints=ids, _reviewed_depth_edges=edges,
                 load_official=checked_loader)
    exec(compile(ast.fix_missing_locations(tree), '<pinned-entry-reviewed-init-arguments>', 'exec'), scope)
    return scope['fit_one']


def fit_one_reviewed(*, init_joints, depth_edges, reviewed_joints, reviewed_depth_edges, **kwargs):
    """Call the original five-stage fit after explicit support admission."""
    h, w = np.asarray(kwargs['image']).shape[:2]
    ids, edges = checked_arguments(kwargs['keypoints'], (w, h), init_joints=init_joints,
                                  depth_edges=depth_edges, reviewed_joints=reviewed_joints,
                                  reviewed_depth_edges=reviewed_depth_edges)
    arrays, receipt = compiled_entry(ids, edges)(**kwargs)
    receipt['camera_init_explicitly_reviewed'] = dict(joint_ids=list(ids), depth_edges=[list(e) for e in edges],
        scope='Only original camera-init arguments are selected. Caller must bind their review to source images.')
    return arrays, receipt
