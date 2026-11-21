"""Fixed synthetic fixture for a new framewise native training entry.

Known geometry only; intentionally does not load previous probe outputs, person
images, old poses or checkpoints. Camera/frame roster and pose order differ.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
from PIL import Image

FRAME_IDS=('moving:20','reference:10')
REFERENCE_ID='reference:10'
WH=(128,96)
GIDS=tuple('free-'+str(i) for i in range(6))
FACES=np.array([[0,1,2]],dtype=np.int64)
LENGTH=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='synthetic_known',
            evidence='Known synthetic coordinates, not real camera metric evidence.')
V=np.array([[-.9,-.7,3.],[.9,-.7,3.],[0.,1.,3.]])
XYZ=np.array([[-.50,-.35,3.16],[-.10,-.38,3.28],[.36,-.30,3.19],
              [-.42,.19,3.23],[.03,.28,3.14],[.46,.24,3.30]])
QUATS=np.array([[.97,.12,-.18,.05],[.95,-.20,.08,.19],[.96,.10,.19,-.12],
                [.94,-.18,-.13,.25],[.98,.08,-.10,-.14],[.93,.24,.18,.09]])
SCALES=np.array([[.095,.050,.035],[.070,.045,.028],[.085,.040,.031],
                 [.080,.055,.030],[.090,.042,.026],[.075,.050,.033]])
OPACITY=np.array([.92,.94,.93,.92,.95,.94])
SH=np.zeros((6,16,3))
SH[:,0,:]=np.array([[.08,-.12,.16],[-.20,.14,.04],[.12,.04,-.15],
                   [.20,-.10,-.08],[-.12,.15,.11],[.05,-.03,.20]])
SH[:,1:,:]=.025*np.sin(np.arange(6*15*3).reshape(6,15,3)*.37)

def rot(axis,a):
    c,s=math.cos(a),math.sin(a)
    if axis=='x':return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    if axis=='y':return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def frame_math(frame):
    if frame not in FRAME_IDS: raise ValueError('Unknown fixed synthetic frame')
    if frame==REFERENCE_ID:
        Q=np.eye(3); a=1.; c=V.mean(0)
        E=np.eye(4);E[:3,:3]=rot('y',-.04);E[:3,3]=[-.04,.02,.15]
        K=np.array([[125.,0.,60.2],[0.,117.,44.7],[0.,0.,1.]])
    else:
        Q=rot('z',.23)@rot('x',-.17);a=1.15;c=V.mean(0)+[.07,-.03,.08]
        E=np.eye(4);E[:3,:3]=rot('z',-.045)@rot('y',.08);E[:3,3]=[.06,-.04,.2]
        K=np.array([[121.,0.,57.4],[0.,119.,42.8],[0.,0.,1.]])
    return Q,a,c,E,K

def mesh(frame):
    Q,a,c,_,_=frame_math(frame)
    return (V-V.mean(0))@Q.T*a+c

def source_transport(count=6):
    from shuolee_portrait.free_clothing_transport import FreeClothingTransport
    ids=GIDS if count==6 else tuple('perf-'+str(i) for i in range(count))
    return FreeClothingTransport(frame_ids=FRAME_IDS,reference_frame_id=REFERENCE_ID,
        vertices_world=np.stack([mesh(f) for f in FRAME_IDS]),faces=FACES,
        motion_support_triangle_ids=np.zeros(count,dtype=np.int64),gaussian_ids=ids,
        translation_unit='meters',frame_source_sha256={f:hashlib.sha256(mesh(f).tobytes()).hexdigest() for f in FRAME_IDS})

def tensor_initial(torch, device, *, perturbed):
    def t(x):return torch.tensor(x,device=device,dtype=torch.float32)
    xyz=XYZ.copy(); sh=SH.copy(); logs=np.log(SCALES)
    if perturbed:
        xyz+=np.array([.045,-.025,.015])
        sh[:,0,:]+=[-.08,.06,-.05]
        logs+=.06
    return dict(xyz=t(xyz),quats=t(QUATS),log_scales=t(logs),
                opacity_logits=t(np.log(OPACITY/(1-OPACITY))),sh=t(sh))

def analytic_render_args(frame, torch, device, eval_sh, *, params=None):
    """Independent rigid-pose construction, never calls the transport class."""
    from scipy.spatial.transform import Rotation
    p=tensor_initial(torch,device,perturbed=False) if params is None else params
    t=lambda x:torch.tensor(x,device=device,dtype=torch.float32)
    Q,a,c,E,K=frame_math(frame)
    xyz=a*((p['xyz']-t(V.mean(0)))@t(Q).T)+t(c)
    q=p['quats']/p['quats'].norm(dim=-1,keepdim=True)
    qm=Rotation.from_matrix(Q).as_quat()[[3,0,1,2]]
    w,v=t(qm)[0],t(qm)[1:]
    # Independent Hamilton product using vector dot/cross.
    outq=torch.cat((w*q[:,:1]-(v*q[:,1:]).sum(-1,keepdim=True),
                    w*q[:,1:]+q[:,:1]*v+torch.linalg.cross(v.expand_as(q[:,1:]),q[:,1:],dim=-1)),dim=-1)
    center=-E[:3,:3].T@E[:3,3]
    direction=xyz-t(center);direction=direction/direction.norm(dim=-1,keepdim=True)
    direction=direction@t(Q)
    colors=(eval_sh(3,p['sh'].transpose(1,2),direction)+.5).clamp_min(0)
    corner=K.copy();corner[:2,2]+=.5
    return (xyz,outq,p['log_scales'].exp()*a,p['opacity_logits'].sigmoid(),colors,t(E[None]),t(corner[None]))

def write_bundle(root, images, masks):
    from shuolee_portrait.frame_pose_contract import SCHEMA,load_frame_pose_bundle
    root=Path(root);root.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    save=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for i,frame in enumerate(FRAME_IDS):
        image=root/f'image{i}.png'; mask=root/f'mask{i}.png'
        Image.fromarray(np.rint(np.clip(images[frame],0,1)*255).astype('uint8')).save(image)
        Image.fromarray(np.rint(np.clip(masks[frame],0,1)*255).astype('uint8')).save(mask)
        E,K=frame_math(frame)[3:]
        rows.append(dict(frame=frame,source_frame=20 if i==0 else 10,sequence_id='native-entry-fixture',video_sha256='e'*64,
            wh=list(WH),image=image.name,source_sha256=sha(image),mask=mask.name,mask_sha256=sha(mask),
            K_integer_centers=K.tolist(),w2c=E.tolist()))
    save(root/'frames.json',dict(records=rows))
    order=[1,0]
    np.savez(root/'poses.npz',frame_ids=np.array([FRAME_IDS[i] for i in order]),
        source_frames=np.array([rows[i]['source_frame'] for i in order]),image_sha256=np.array([rows[i]['source_sha256'] for i in order]),
        sequence_id=np.array('native-entry-fixture'),video_sha256=np.array('e'*64),
        vertices_world=np.stack([mesh(FRAME_IDS[i]) for i in order]),faces=FACES,shared_betas=np.zeros((1,10)),
        K_integer_centers=np.array([rows[i]['K_integer_centers'] for i in order]),w2c=np.array([rows[i]['w2c'] for i in order]))
    save(root/'manifest.json',dict(schema=SCHEMA,purpose='synthetic_contract_test',sequence_id='native-entry-fixture',video_sha256='e'*64,
        coordinate_system='opencv_world_meters',pixel_domain='rectified_pinhole',pixel_convention='integer_centers',reference_frame_id=REFERENCE_ID,
        frames=dict(path='frames.json',sha256=sha(root/'frames.json'),rows_key='records'),poses=dict(path='poses.npz',sha256=sha(root/'poses.npz'))))
    return load_frame_pose_bundle(root/'manifest.json')

def write_cube_bundle(root):
    """Six known silhouette views; file-loaded production contract, zero skew."""
    from shuolee_portrait.frame_pose_contract import SCHEMA,load_frame_pose_bundle
    root=Path(root);root.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    save=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    center=np.array([0.,0.,4.]);size=65
    names=('right','reference','left','above','below','back')
    cameras=np.array([[3,0,4],[0,0,0],[-3,0,4],[0,3,4],[0,-3,4],[0,0,8]],dtype=float)
    K=np.array([[45.,0.,30.2],[0.,45.,33.4],[0.,0.,1.]])
    rows=[];meshes=[]
    y,x=np.indices((size,size));pixel=np.stack((x,y,np.ones_like(x)),-1)
    for i,(name,C) in enumerate(zip(names,cameras)):
        direction=center-C;direction/=np.linalg.norm(direction)
        up=np.array([0.,1.,0.]) if abs(direction[1])<.9 else np.array([0.,0.,1.])
        right=np.cross(up,direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
        E=np.eye(4);E[:3,:3]=np.stack((right,up,direction));E[:3,3]=-E[:3,:3]@C
        shift=np.zeros(3) if name=='reference' else np.array([.6,.15,.1])
        rays=(pixel@np.linalg.inv(K).T)@E[:3,:3];origin=C-center-shift
        with np.errstate(divide='ignore',invalid='ignore'):
            a,b=(-.45-origin)/rays,(.45-origin)/rays
        hit=np.maximum(np.minimum(a,b).max(-1),0)<=np.maximum(a,b).min(-1)
        image=root/(name+'.png');mask=root/(name+'-mask.png')
        pixels=np.full((size,size,3),[30,50,60],dtype=np.uint8);pixels[hit]=[153,102,77]
        Image.fromarray(pixels).save(image);Image.fromarray((hit*255).astype('uint8')).save(mask)
        rows.append(dict(frame=name,source_frame=10+i*7,sequence_id='known-moving-cube-production',video_sha256='c'*64,
            wh=[size,size],image=image.name,source_sha256=sha(image),mask=mask.name,mask_sha256=sha(mask),K_integer_centers=K.tolist(),w2c=E.tolist()))
        meshes.append(np.array([[-.3,-.3,4.],[.4,-.2,4.],[.1,.4,4.]])+shift)
    save(root/'frames.json',dict(records=rows))
    order=[4,0,2,1,5,3]
    np.savez(root/'poses.npz',frame_ids=np.array([names[i] for i in order]),source_frames=np.array([rows[i]['source_frame'] for i in order]),
        image_sha256=np.array([rows[i]['source_sha256'] for i in order]),sequence_id=np.array('known-moving-cube-production'),video_sha256=np.array('c'*64),
        vertices_world=np.stack([meshes[i] for i in order]),faces=FACES,shared_betas=np.zeros((1,10)),
        K_integer_centers=np.array([rows[i]['K_integer_centers'] for i in order]),w2c=np.array([rows[i]['w2c'] for i in order]))
    save(root/'manifest.json',dict(schema=SCHEMA,purpose='synthetic_contract_test',sequence_id='known-moving-cube-production',video_sha256='c'*64,
        coordinate_system='opencv_world_meters',pixel_domain='rectified_pinhole',pixel_convention='integer_centers',reference_frame_id='reference',
        frames=dict(path='frames.json',sha256=sha(root/'frames.json'),rows_key='records'),poses=dict(path='poses.npz',sha256=sha(root/'poses.npz'))))
    return load_frame_pose_bundle(root/'manifest.json')
