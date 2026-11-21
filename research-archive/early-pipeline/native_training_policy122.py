"""Explicit sampling policy for fixed complete-output cameras. CPU only."""
import copy,hashlib,json
import numpy as np
from native_residual_camera122 import validate_camera,base_camera
from native_covisibility122 import build_graph,select_cycle,validate_schedule

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_policy(policy):
    if not isinstance(policy,dict) or set(policy)!={'kind','density'}:
        raise ValueError('Explicit sampling kind and density policy required')
    if policy['kind'] not in ('uniform_single','covisible_pair') or policy['density']!='fixed_free_centers':
        raise ValueError('Only declared balanced sampling and fixed free centers are implemented')
    return copy.deepcopy(policy)

def complete_graph(cameras,observations,*,source_sha256,min_shared):
    for c in cameras:validate_camera(c)
    g=build_graph([base_camera(c) for c in cameras],observations,source_sha256=source_sha256,min_shared=min_shared)
    g.update(schema='native-cvo-fit-graph-v2-output-residual',cameras_sha256=digest(cameras))
    return g

def cycle_schedule(cameras,policy,seed,*,graph=None):
    validate_policy(policy)
    if type(seed)is not int:raise ValueError('Integer sampling seed required')
    for c in cameras:validate_camera(c)
    ids=[c['id'] for c in cameras]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Distinct recorded cameras required')
    if policy['kind']=='uniform_single':
        if graph is not None:raise ValueError('Unused CVO evidence must not silently change uniform sampling')
        return [[i] for i in np.random.default_rng(seed).permutation(ids).tolist()]
    if not graph or graph.get('schema')!='native-cvo-fit-graph-v2-output-residual' or graph.get('cameras_sha256')!=digest(cameras):
        raise ValueError('CVO graph must bind these exact complete K/E/D cameras')
    if set(graph['frames'])!=set(ids):raise ValueError('CVO graph cannot omit registered frames')
    schedule=select_cycle(graph,seed=seed,policy='covisible')
    validate_schedule(graph,schedule)
    return schedule

def validate_reviews(steps,review_steps,review,checkpoint):
    if type(steps)is not int or not 1<=steps<=18000:raise ValueError('Finite1..18000 step budget required')
    if len(review_steps)!=len(set(review_steps)):raise ValueError('Duplicate review step')
    if any(type(s)is not int or not 1<=s<=steps for s in review_steps):raise ValueError('Review outside run')
    if steps>=1000 and 1000 not in review_steps:raise ValueError('Actual1000step review required')
    if review_steps and (review is None or checkpoint is None):raise ValueError('Actual review and checkpoint callback required')

