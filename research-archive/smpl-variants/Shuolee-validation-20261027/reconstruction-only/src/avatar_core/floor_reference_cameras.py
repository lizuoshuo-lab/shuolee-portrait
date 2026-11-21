"""Read-only inspection cameras based on measured floor, not fitted body up."""
import numpy as np
from scipy.spatial.transform import Rotation
import shuolee_common as c


def make_cameras():
    source=c.BASE/'current/smplx-observation-support-014'
    floor=c.read(source/'floor-reference.json')
    with np.load(source/'surface-support.npz') as f: data={k:f[k].copy() for k in f.files}
    up=np.array(floor['floor_normal_world']);d=floor['offset'];xyz=data['xyz']
    height=float(np.quantile(xyz@up+d,.997)+.12)
    foot=xyz[data['regions']==4];base=np.median(foot,axis=0);base=base-(base@up+d)*up
    target=base+.5*height*up
    rows=c.read(c.ROOT/'data/inputs.json')['records'];E=np.array(rows[23]['w2c'])
    center=-E[:3,:3].T@E[:3,3]
    front=center-target;front-=front@up*up;front/=np.linalg.norm(front)
    cams=[]
    for name,yaw,pitch in [('front',0,0),('left45',-45,0),('right45',45,0),('left90',-90,0),('right90',90,0),('above25',0,25),('below25',0,-25)]:
        horizontal=Rotation.from_rotvec(up*np.deg2rad(yaw)).apply(front)
        offset=np.cos(np.deg2rad(pitch))*horizontal+np.sin(np.deg2rad(pitch))*up
        eye=target+1.65*height*offset;forward=target-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
        R=np.stack([right,down,forward]);E=np.eye(4);E[:3,:3]=R;E[:3,3]=-R@eye
        cams.append({'view':name,'yaw_degrees':yaw,'elevation_degrees':pitch,'w2c':E.tolist(),
                     'K':[[900.,0.,260.],[0.,900.,395.],[0.,0.,1.]],'width':520,'height':790})
    old=c.read(c.BASE/'current/smplx-whole-body-prior-fresh-006/gaussianavatar-comparison/route-comparison.json')['cameras']
    old_front=next(p for p in old if p['view']=='front');old_up=-np.array(old_front['w2c'])[1,:3]
    angle=float(np.rad2deg(np.arccos(np.clip(old_up@up,-1,1))))
    info={'created':c.stamp(),'floor':floor,'same_cameras_for_all_models':True,'geometry_transformed':False,
          'old_front_camera_up_vs_floor_normal_deg':angle,'target_world':target.tolist(),'support_height_world_units':height,
          'scope':'Inspection camera orientation only. Does not repair local joint pose or certify anatomy. Floor normal is not independent IMU gravity.',
          'cameras':cams,'floor_sha256':c.sha(source/'floor-reference.json'),'support_sha256':c.sha(source/'surface-support.npz')}
    return cams,info,data
