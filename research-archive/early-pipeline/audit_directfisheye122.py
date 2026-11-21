"""Bounded CPU source/analytic controls. No imported third-party training runtime."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,math,sys,os
from typing import NamedTuple
from types import SimpleNamespace
import numpy as np

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'directfisheye-research'
assert not (O/'code-audit.json').exists()
sources=json.loads((O/'sources.json').read_text())
for row in sources['files']:
    assert hashlib.sha256(Path(row['snapshot']).read_bytes()).hexdigest()==row['sha256']

def source(name):return (O/name.replace('/','__')).read_text(encoding='utf-8')
def extract(name,names,ns):
    tree=ast.parse(source(name)); chosen=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert {x.name for x in chosen}==set(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),name,'exec'),ns)
    return chosen

ns={'np':np}
extract('utils/find_camera_groups.py',['quaternion_to_rotation_matrix','compute_camera_direction','compute_angle_between_vectors'],ns)
def qmul(a,b):
    w,x,y,z=a;W,X,Y,Z=b
    return np.array([w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W])
def angle(a,b):return float(np.degrees(ns['compute_angle_between_vectors'](a,b)))
q1=np.array([1.,0,0,0]);q2=np.array([math.cos(math.radians(20)),0,math.sin(math.radians(20)),0])
inv_global=np.array([math.cos(math.pi/4),-math.sin(math.pi/4),0,0])
qa=[q1,q2];qb=[qmul(q,inv_global) for q in qa]
bad=[angle(*[ns['compute_camera_direction'](q) for q in qs]) for qs in [qa,qb]]
good=[angle(*[ns['quaternion_to_rotation_matrix'](q).T@np.array([0.,0,1.]) for q in qs]) for qs in [qa,qb]]
assert abs(bad[0]-40)<1e-8 and abs(bad[1])<1e-5 and max(abs(x-40) for x in good)<1e-8

ns.update(NamedTuple=NamedTuple,sys=sys,os=os,Image=SimpleNamespace(open=lambda p:None),qvec2rotmat=ns['quaternion_to_rotation_matrix'],focal2fov=lambda f,p:2*math.atan(p/(2*f)))
extract('scene/dataset_readers.py',['CameraInfo','readColmapCameras'],ns)
cam=json.loads((R/'native-capacity/portrait-native-capacity/package/cameras.json').read_text())[0]
w,h=cam['width'],cam['height'];K=np.asarray(cam['K'],float)
center=np.array([(w-1)/2,(h-1)/2]);principal=K[:2,2]
extr={1:SimpleNamespace(qvec=q1,tvec=[0.,0.,0.],camera_id=1,name='synthetic.png')}
def load(k):
    intr={1:SimpleNamespace(width=w,height=h,id=1,model='PINHOLE',params=[k[0,0],k[1,1],k[0,2],k[1,2]])}
    return ns['readColmapCameras'](extr,intr,'stub-no-file-access')[0]
out1=load(K);K2=K.copy();K2[:2,2]+=np.array([23.,-17.]);out2=load(K2)
assert 'cx' not in out1._fields and 'cy' not in out1._fields and out1.FovX==out2.FovX and out1.FovY==out2.FovY

training=ast.parse(source('training_core.py'))
fn=next(x for x in training.body if isinstance(x,ast.FunctionDef) and x.name=='train_loop')
viewloop=next(x for x in ast.walk(fn) if isinstance(x,ast.For) and ast.unparse(x.target)=='view_cam')
loopcalls=[x for x in ast.walk(viewloop) if isinstance(x,ast.Call)]
assert any(ast.unparse(x.func)=='loss.backward' for x in loopcalls)
assert not any(ast.unparse(x.func).endswith('optimizer.step') for x in loopcalls)
assert not any(ast.unparse(x.func).endswith('add_densification_stats') for x in loopcalls)
assert ast.unparse(viewloop.iter)=='reversed(cam_list)'
lossassign=next(x for x in viewloop.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in x.targets))
assert not any(isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div) for x in ast.walk(lossassign.value))
viewlocals={t.id for a in ast.walk(viewloop) if isinstance(a,ast.Assign) for t in a.targets if isinstance(t,ast.Name)}
assert {'viewspace_point_tensor','visibility_filter','radii'}.issubset(viewlocals)
opt=ast.parse(source('scene/gaussian_model.py'))
setup=next(x for x in ast.walk(opt) if isinstance(x,ast.FunctionDef) and x.name=='training_setup')
groups=next(x.value for x in setup.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='l' for t in x.targets))
group_names=[next(ast.literal_eval(v) for k,v in zip(d.keys,d.values) if isinstance(k,ast.Constant) and k.value=='name') for d in groups.elts]
assert group_names==['xyz','f_dc','f_rest','opacity','scaling','rotation']

nsl={'torch':SimpleNamespace(abs=np.abs)}
extract('utils/loss_utils.py',['l1_loss'],nsl)
phi=np.array([1.,3.]);err1=np.array([1.,0.]);err2=np.array([0.,1.])
scalar=[float(np.mean(phi*nsl['l1_loss'](e,np.zeros(2)))) for e in [err1,err2]]
pixel=[float(np.mean(phi*np.abs(e))) for e in [err1,err2]]
assert scalar==[1.,1.] and pixel==[.5,1.5]

result={
 'updated':datetime.now(timezone.utc).isoformat(),'commit':sources['commit'],
 'scope':'Official source snapshot plus isolated CPU helper/analytic controls; not an installed CUDA reproduction or person training',
 'all_21_source_hashes_verified':True,'no_repository_source_modified':True,'no_person_data_uploaded':True,
 'principal_point':{'CameraInfo_fields':out1._fields,'different_cx_cy_same_loaded_fov':True,'native_dimensions':[w,h],'conditional_K':K.tolist(),'public_default_pixel_center':center.tolist(),'lost_offset_xy_px':(principal-center).tolist(),'lost_offset_euclidean_px':float(np.linalg.norm(principal-center)),'caveat':'Potential offset when using this public centered loader at native resolution; not a diagnosis that current local pipeline has this error. Current K remains conditional.'},
 'CVO_loop':{'view_loop_lines':[viewloop.lineno,viewloop.end_lineno],'backward_each_view':True,'optimizer_step_after_views':True,'batch_loss_is_sum_not_mean':True,'densification_statistics_after_loop':True,'densification_view':'last rendered view = anchor under reversed(cam_list)','learned_gaussian_groups':group_names,'cameras_or_distortion_in_optimizer':False,'explicit_feature_reprojection_loss_in_this_loop':False,'source_derived_not_CUDA_test':True},
 'world_gauge_angle_control':{'official_direction_expression':'R @ [0,0,1]','correct_for_COLMAP_world_to_camera':'R.T @ [0,0,1]','official_pair_angle_before_after_world_gauge_deg':bad,'correct_pair_angle_before_after_world_gauge_deg':good,'applies':'Angle-based sorting and automatic skip statistics. Current overlap sort itself does not use angles.'},
 'edge_weight_control':{'public_scalar_L1_weighted_results':scalar,'per_pixel_weighted_results':pixel,'finding':'l1_loss already reduces to a scalar; multiplying phi afterwards rescales the image loss rather than weighting each edge residual.'},
 'mask_source_audit':{'source':'utils/camera_utils.py:42','public_channel_condition':'resized_image_rgb.shape[1] == 4','expected_layout':'C,H,W','example_RGBA_shape':[4,h,w],'condition_actual':h==4,'finding':'Checks height instead of channel count; ordinary RGBA alpha is not loaded. Separate foreground supervision requires explicit adaptation.'},
 'training_executed':False,'physical_quality_passed':False
}
(O/'code-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
