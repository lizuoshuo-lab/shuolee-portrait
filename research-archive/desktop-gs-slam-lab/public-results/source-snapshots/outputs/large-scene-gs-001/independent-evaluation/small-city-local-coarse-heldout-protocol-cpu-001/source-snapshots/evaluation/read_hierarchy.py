"""Read pinned GraphDeco .hier storage and audit CPU translations of official cuts.

This is an independent CPU decoder/reference translation, NOT the official extension
or a renderer. Geometry/alpha interpolation must be matched separately in rendering.
Source contracts: gaussian-hierarchy 677c8553dc64dfd62c272eca94a291a277733113.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

NODE_FIELDS = ['depth','parent','start','count_leafs','count_merged','start_children','count_children']
SOURCE_COMMIT = '677c8553dc64dfd62c272eca94a291a277733113'

@dataclass
class Hierarchy:
    path: Path
    compressed: bool
    positions: np.ndarray
    rotations: np.ndarray
    log_scales: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    nodes: np.ndarray
    boxes: np.ndarray
    unowned_tail: int = 0

    def summary(self):
        return dict(path=str(self.path),compressed=self.compressed,gaussian_count=len(self.positions),node_count=len(self.nodes),
                    leaf_gaussian_count=int(self.nodes[:,3].sum()),merged_gaussian_count=int(self.nodes[:,4].sum()),
                    root_depth=int(self.nodes[0,0]),node_fields=NODE_FIELDS,source_commit=SOURCE_COMMIT,
                    unowned_tail=self.unowned_tail)

def read_hierarchy(path, expected_unowned_tail=0):
    """Memory-map float arrays; compressed nodes are normalized to official Nx7 int32."""
    path = Path(path).resolve()
    size = path.stat().st_size
    with path.open('rb') as f:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError('Missing signed Gaussian count')
        signed_count, = struct.unpack('<i',raw)
    p = abs(signed_count)
    compressed = signed_count < 0
    if not p or p > (size-8)//(124 if compressed else 236):
        raise ValueError('Gaussian count incompatible with file length')
    offset = 4
    def mapped(dtype, shape):
        nonlocal offset
        needed = np.dtype(dtype).itemsize * math.prod(shape)
        if offset+needed > size:
            raise ValueError('Truncated hierarchy payload')
        result = np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape)
        offset += needed
        return result
    positions = mapped('<f4',(p,3))
    dtype = '<f2' if compressed else '<f4'
    rotations = mapped(dtype,(p,4))
    log_scales = mapped(dtype,(p,3))
    opacities = mapped(dtype,(p,))
    sh = mapped(dtype,(p,16,3))
    with path.open('rb') as f:
        f.seek(offset)
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError('Missing node count')
    n, = struct.unpack('<i',raw)
    offset += 4
    if n <= 0 or n > (size-offset)//(36 if compressed else 60):
        raise ValueError('Node count incompatible with file length')
    if compressed:
        packed_dtype = np.dtype([('parent','<i4'),('start','<i4'),('start_children','<i4'),('dccc','<i2',(4,))],align=False)
        assert packed_dtype.itemsize == 20
        packed = mapped(packed_dtype,(n,))
        nodes = np.empty((n,7),dtype=np.int32)
        nodes[:,0] = packed['dccc'][:,0]
        nodes[:,1] = packed['parent']
        nodes[:,2] = packed['start']
        nodes[:,3] = packed['dccc'][:,2]
        nodes[:,4] = packed['dccc'][:,3]
        nodes[:,5] = packed['start_children']
        nodes[:,6] = packed['dccc'][:,1]
    else:
        nodes = mapped('<i4',(n,7))
    boxes = mapped(dtype,(n,2,4))
    if offset != size:
        raise ValueError(f'Unexpected trailing bytes: {size-offset}')
    if not isinstance(expected_unowned_tail,int) or not 0 <= expected_unowned_tail < p:
        raise ValueError('Expected unowned tail must be an explicit nonnegative count smaller than G')
    result = Hierarchy(path,compressed,positions,rotations,log_scales,opacities,sh,nodes,boxes,expected_unowned_tail)
    validate_tree(result)
    return result

def read_gaussian_bin(path):
    """Read loader.cpp::loadBin storage, which is NOT .hier despite same scalar types."""
    path = Path(path).resolve()
    with path.open('rb') as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError('Missing raw point-cloud count')
    p, = struct.unpack('<i',raw)
    if p <= 0 or path.stat().st_size != 4+236*p:
        raise ValueError('Raw point-cloud count/payload mismatch')
    offset = 4
    arrays = {}
    for name,shape in [('positions',(p,3)),('sh',(p,16,3)),('opacity_logits',(p,)),('log_scales',(p,3)),('rotations_wxyz',(p,4))]:
        arrays[name] = np.memmap(path,dtype='<f4',mode='r',offset=offset,shape=shape)
        offset += math.prod(shape)*4
    return arrays

def read_anchors(path, gaussian_count):
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError('Missing anchor count')
    n, = struct.unpack('<i',raw)
    if n < 0 or path.stat().st_size != 4+4*n:
        raise ValueError('Anchor count/payload mismatch')
    anchors = np.fromfile(path,dtype='<i4',count=n,offset=4)
    if np.any(anchors < 0) or np.any(anchors >= gaussian_count):
        raise ValueError('Anchor index out of range')
    return anchors

def validate_tree(h):
    """Validate root/parent/child ranges, exclusive storage and spatial AABBs."""
    n,p = len(h.nodes),len(h.positions)
    nodes = h.nodes
    if nodes[0,1] != -1 or np.any(nodes[:,[0,2,3,4,6]] < 0):
        raise ValueError('Invalid root, depth or counts')
    if np.any(nodes[1:,1] < 0) or np.any(nodes[1:,1] >= n):
        raise ValueError('Invalid parent index')
    if np.any(nodes[:,2].astype(np.int64)+nodes[:,3]+nodes[:,4] > p):
        raise ValueError('Gaussian range out of bounds')
    if np.any((nodes[:,6] > 0) & ((nodes[:,5] < 0) | (nodes[:,5].astype(np.int64)+nodes[:,6] > n))):
        raise ValueError('Missing/out-of-bounds children (streaming chunks unsupported)')
    if np.any((nodes[:,0] == 0) & (nodes[:,6] != 0)):
        raise ValueError('Depth-zero node has children')
    xyz = np.asarray(h.boxes[:,:,:3])
    if not np.isfinite(xyz).all() or np.any(xyz[:,0] > xyz[:,1]):
        raise ValueError('Invalid spatial AABB')
    if np.any(np.isnan(h.boxes[:,:,3])) or np.any(h.boxes[:,0,3] < 0):
        raise ValueError('Invalid size metadata')
    # A positive infinity fourth component is allowed: compressed 1e9 root sentinel.
    seen = np.zeros(n,dtype=np.uint8)
    storage = np.zeros(p,dtype=np.uint8)
    stack = [0]
    while stack:
        i = stack.pop()
        if seen[i]:
            raise ValueError('Cycle or multiply owned tree node')
        seen[i] = 1
        depth,parent,start,leaves,merged,child_start,children = map(int,nodes[i])
        if storage[start:start+leaves+merged].any():
            raise ValueError('Overlapping stored Gaussian ranges')
        storage[start:start+leaves+merged] = 1
        for child in range(child_start,child_start+children):
            if nodes[child,1] != i or nodes[child,0] >= depth:
                raise ValueError('Parent/child depth inconsistency')
            stack.append(child)
    tree_end = p-h.unowned_tail
    if not seen.all() or not storage[:tree_end].all() or storage[tree_end:].any():
        raise ValueError('Unreachable nodes or unexpected unowned Gaussian storage; explicit post skybox tail required')

def expand_to_target_cpu(h, target):
    """Direct translation of official CPU traversal.cpp::recExpand."""
    indices,nodes_for_render = [],[]
    stack = [0]
    while stack:
        i = stack.pop()
        depth,parent,start,leaves,merged,child_start,children = map(int,h.nodes[i])
        selected = leaves + (merged if depth <= target else 0)
        indices.extend(range(start,start+selected))
        nodes_for_render.extend([i]*selected)
        if depth > target:
            stack.extend(reversed(range(child_start,child_start+children)))
    return np.asarray(indices,dtype=np.int32),np.asarray(nodes_for_render,dtype=np.int32)

def official_size_proxy_cpu(h, camera):
    """Float32 reference to computeSizeGPU; view direction is unused by source."""
    camera = np.asarray(camera,dtype=np.float32)
    if camera.shape != (3,) or not np.isfinite(camera).all():
        raise ValueError('Finite global-frame camera center required')
    lower = np.asarray(h.boxes[:,0,:3],dtype=np.float32)
    upper = np.asarray(h.boxes[:,1,:3],dtype=np.float32)
    inside = ((camera >= lower) & (camera <= upper)).all(axis=1)
    closest = np.maximum(lower,np.minimum(upper,camera))
    diff = camera-closest
    distance = np.sqrt(np.sum(diff*diff,axis=1,dtype=np.float32))
    with np.errstate(divide='ignore',invalid='ignore'):
        sizes = np.asarray(h.boxes[:,0,3],dtype=np.float32)/distance
    sizes[inside] = np.finfo(np.float32).max
    return sizes

def tau_to_threshold(tau, fx):
    """render_hierarchy.py: theta=4(tau+.5)tan(FoVx/2)/width=2(tau+.5)/fx."""
    if tau < 0 or fx <= 0 or not math.isfinite(tau+fx):
        raise ValueError('tau>=0 and finite fx>0 required')
    return float(np.float32(2*(tau+0.5)/fx))

def expand_to_size_cpu(h, camera, threshold):
    """Direct predicate translation of official CUDA markNodesForSize.

    Not a general root-fallback cut: source can select nothing if its root is small.
    """
    threshold = np.float32(threshold)
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('Positive finite dimensionless threshold required')
    sizes = official_size_proxy_cpu(h,camera)
    indices,parents,node_ids = [],[],[]
    for i,row in enumerate(h.nodes):
        depth,parent,start,leaves,merged,child_start,children = map(int,row)
        count = leaves if sizes[i] >= threshold else 0
        if sizes[i] < threshold and parent != -1 and sizes[parent] >= threshold:
            count = leaves + (merged if depth != 0 else 0)
        indices.extend(range(start,start+count))
        parents.extend([int(h.nodes[parent,2]) if parent != -1 else -1]*count)
        node_ids.extend([i]*count)
    return np.asarray(indices,dtype=np.int32),np.asarray(parents,dtype=np.int32),np.asarray(node_ids,dtype=np.int32)

def interpolation_weights_cpu(h, node_ids, camera, threshold):
    """Reference to computeTsIndexed, returning child weight t and parent child count."""
    sizes = official_size_proxy_cpu(h,camera)
    weights,kids = [],[]
    threshold = np.float32(threshold)
    for i in node_ids:
        parent = int(h.nodes[i,1])
        t = np.float32(1)
        if parent != -1:
            parent_size = sizes[parent]
            if parent_size <= np.float32(2)*threshold:
                start = max(np.float32(0.5)*parent_size,sizes[i])
                diff = parent_size-start
                if diff > 0:
                    t = max(np.float32(1)-max(np.float32(0),threshold-start)/diff,np.float32(0))
        weights.append(t)
        kids.append(1 if parent == -1 else int(h.nodes[parent,6]))
    return np.asarray(weights,dtype=np.float32),np.asarray(kids,dtype=np.int32)

def coverage_audit(h, indices):
    """Every stored leaf must have exactly one selected local leaf or ancestor summary.

    Summaries of a node are a group. This verifies topology, not photometric equivalence.
    """
    indices = np.asarray(indices,dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(h.positions)):
        raise ValueError('Selected Gaussian index out of bounds')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Duplicate selected Gaussian index')
    chosen = np.zeros(len(h.positions),dtype=np.bool_)
    chosen[indices] = True
    path_counts = []
    selected_summary_nodes = []
    stack = [(0,0)]
    while stack:
        i,ancestors = stack.pop()
        depth,parent,start,leaves,merged,child_start,children = map(int,h.nodes[i])
        summary_count = int(chosen[start+leaves:start+leaves+merged].any())
        if summary_count:
            selected_summary_nodes.append(i)
        for leaf in range(start,start+leaves):
            path_counts.append(ancestors+summary_count+int(chosen[leaf]))
        for child in range(child_start,child_start+children):
            stack.append((child,ancestors+summary_count))
    histogram = {str(k):path_counts.count(k) for k in sorted(set(path_counts))}
    return dict(selected_gaussians=len(indices),stored_leaves=len(path_counts),root_to_leaf_coverage_histogram=histogram,
                every_leaf_exactly_once=bool(path_counts) and all(x == 1 for x in path_counts),selected_summary_nodes=selected_summary_nodes,
                scope='Topology coverage only; no image/rasterizer equivalence assertion')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('hierarchy',type=Path)
    p.add_argument('--camera',nargs=3,type=float)
    p.add_argument('--fx',type=float)
    p.add_argument('--taus',nargs='+',type=float,default=[0,3,6])
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    h = read_hierarchy(a.hierarchy)
    result = h.summary()
    result['sha256'] = hashlib.sha256(a.hierarchy.read_bytes()).hexdigest()
    if a.camera is not None:
        if a.fx is None:
            p.error('--camera requires --fx at the actual rendering resolution')
        result['cpu_reference_cuts'] = []
        for tau in a.taus:
            theta = tau_to_threshold(tau,a.fx)
            indices,parents,nodes = expand_to_size_cpu(h,a.camera,theta)
            t,kids = interpolation_weights_cpu(h,nodes,a.camera,theta)
            entry = dict(tau=tau,threshold=theta,**coverage_audit(h,indices),interpolation_min=float(t.min()) if len(t) else None,
                         interpolation_max=float(t.max()) if len(t) else None)
            result['cpu_reference_cuts'].append(entry)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
