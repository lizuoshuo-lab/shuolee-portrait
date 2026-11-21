"""Fixed-K global components. No real-camera admission is implied by this module."""
import numpy as np

class InitializationRejected(ValueError):
    def __init__(self,reason,diagnostic):
        super().__init__(reason);self.reason=reason;self.diagnostic=diagnostic

def angle_between_rotations(a,b):
    q=a@b.T
    sine=np.linalg.norm([q[2,1]-q[1,2],q[0,2]-q[2,0],q[1,0]-q[0,1]])/2
    cosine=(np.trace(q)-1)/2
    return float(np.arctan2(sine,cosine))

def connected_components(n,edges):
    adjacency=[set() for _ in range(n)]
    for a,b,*_ in edges:adjacency[a].add(b);adjacency[b].add(a)
    remaining=set(range(n));groups=[]
    while remaining:
        stack=[min(remaining)];seen=set()
        while stack:
            x=stack.pop()
            if x in seen:continue
            seen.add(x);stack.extend(adjacency[x]-seen)
        remaining-=seen;groups.append(sorted(seen))
    return groups

def average_rotations(n,relative_edges):
    """Input edge(a,b,Q) uses Q=R_b @ R_a.T; return world-to-camera R_i, R_0=I."""
    edges=list(relative_edges)
    for a,b,q in edges:
        if not(0<=a<n and 0<=b<n and a!=b):raise ValueError('invalid camera identity')
        q=np.asarray(q)
        if q.shape!=(3,3) or not np.isfinite(q).all():raise ValueError('invalid relative rotation')
        if np.max(np.abs(q.T@q-np.eye(3)))>1e-8 or abs(np.linalg.det(q)-1)>1e-8:raise ValueError('relative rotation is not SO3')
    groups=connected_components(n,edges)
    if len(groups)!=1:raise InitializationRejected('disconnected_rotation_graph',dict(components=groups))
    L=np.zeros((3*n,3*n))
    for a,b,q in edges:
        aa=slice(3*a,3*a+3);bb=slice(3*b,3*b+3)
        L[aa,aa]+=np.eye(3);L[bb,bb]+=np.eye(3);L[aa,bb]-=q.T;L[bb,aa]-=q
    values,vectors=np.linalg.eigh(L);Y=vectors[:,:3].reshape(n,3,3)
    # Remove the common O3 eigenbasis before individual SO3 projection.
    # Projecting each block first would create arbitrary reflections when det(basis)<0.
    R=[]
    for yi in Y:
        raw=yi@Y[0].T;u,s,vt=np.linalg.svd(raw)
        if s[-1]<1e-12:raise InitializationRejected('singular_rotation_gauge',dict(singular_values=s.tolist()))
        fix=np.diag([1.,1.,np.linalg.det(u@vt)]);R.append(u@fix@vt)
    R=np.asarray(R);R[0]=np.eye(3)
    residual=[angle_between_rotations(R[b]@R[a].T,q) for a,b,q in edges]
    return R,dict(method='connection-Laplacian, first-block gauge, SO3 projection',
        edges=len(edges),first_camera_fixed=True,eigenvalues=values.tolist(),edge_angle_max_rad=max(residual,default=0.))

def global_positions(K,R,observations,point_count,rank_relative_threshold=1e-9):
    """Solve all camera centers and world points simultaneously from fixed rotations.

    observations: [(camera_index, point_index, [u,v]), ...]. Integer-center native K.
    This exact homogeneous solver intentionally rejects extra null directions.
    It is not yet a robust estimator for noisy/outlier-contaminated real bearings.
    """
    K=np.asarray(K,float);R=np.asarray(R,float);n=len(R);inv=np.linalg.inv(K)
    obs=list(observations);cols=3*(n-1)+3*point_count;M=np.zeros((2*len(obs),cols))
    if not np.allclose(R[0],np.eye(3),atol=1e-10,rtol=0):raise ValueError('first rotation gauge must be identity')
    for k,(ci,pi,uv) in enumerate(obs):
        if not(0<=ci<n and 0<=pi<point_count):raise ValueError('invalid observation identity')
        uv=np.asarray(uv,float)
        if uv.shape!=(2,) or not np.isfinite(uv).all():raise ValueError('invalid native pixel')
        d=R[ci].T@(inv@np.r_[uv,1.]);d/=np.linalg.norm(d)
        axis=np.eye(3)[np.argmin(np.abs(d))];e1=np.cross(d,axis);e1/=np.linalg.norm(e1);e2=np.cross(d,e1)
        B=np.stack([e1,e2]);row=slice(2*k,2*k+2)
        if ci:M[row,3*(ci-1):3*ci]=-B
        M[row,3*(n-1)+3*pi:3*(n-1)+3*pi+3]=B
    norms=np.linalg.norm(M,axis=0)
    if np.any(norms<1e-15):raise InitializationRejected('unobserved_parameter',dict(columns=np.flatnonzero(norms<1e-15).tolist()))
    normalized=M/norms
    _,s,vt=np.linalg.svd(normalized,full_matrices=False)
    tolerance=float(s[0]*rank_relative_threshold);rank=int(np.sum(s>tolerance));nullity=cols-rank
    diagnostic=dict(matrix_shape=list(M.shape),rank=rank,nullity=nullity,singular_values=s.tolist(),
        rank_relative_threshold=rank_relative_threshold,rank_absolute_threshold=tolerance,
        method='Joint world bearing constraints, first center fixed, one homogeneous scale')
    if nullity!=1:raise InitializationRejected('position_geometry_not_unique_up_to_scale',diagnostic)
    v=vt[-1]/norms;C=np.vstack([np.zeros(3),v[:3*(n-1)].reshape(n-1,3)]);X=v[3*(n-1):].reshape(point_count,3)
    z=np.array([(R[ci]@(X[pi]-C[ci]))[2] for ci,pi,_ in obs])
    if np.count_nonzero(z>0)<np.count_nonzero(z<0):C=-C;X=-X;z=-z
    scale=float(np.median(np.linalg.norm(C[1:],axis=1)))
    if not np.isfinite(scale) or scale<1e-12:raise InitializationRejected('zero_camera_baseline',diagnostic)
    C/=scale;X/=scale;z/=scale
    if not np.all(z>1e-9):raise InitializationRejected('nonpositive_depth',dict(**diagnostic,nonpositive_count=int(np.sum(z<=1e-9))))
    diagnostic.update(scale_convention='Median camera-center distance from camera0 equals1; sign chosen by positive depth',
        depth_min=float(z.min()),world_scale=scale,linear_residual_max=float(np.max(np.abs(M@v))))
    return C,X,diagnostic

def project(K,R,C,X,observations):
    uv=[];depth=[]
    for ci,pi,*_ in observations:
        y=R[ci]@(X[pi]-C[ci]);p=K@y;uv.append(p[:2]/p[2]);depth.append(y[2])
    return np.asarray(uv),np.asarray(depth)
