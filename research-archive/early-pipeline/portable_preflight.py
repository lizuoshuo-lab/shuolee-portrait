import os,json,sys,importlib.util,hashlib
from pathlib import Path
P=Path('/mnt/d/00/工作/人体重建')
R=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/portable-preflight-20261025')
spec=importlib.util.spec_from_file_location('launcher',P/'scripts/experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.configure(m.read(P/'configs/hover175.json'),R,'gaussianavatar')
import fit_avatar_pair_static_114,avatar_pair_114,torch,smplx
from shuolee_vposer import load_vposer
v,_,vm=load_vposer()
model=smplx.SMPLX(str(c.MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False).cuda()
with torch.no_grad():out=model();pose=v.decode(torch.zeros((1,32),device='cuda'))
assert tuple(out.vertices.shape)==(1,10475,3)
with __import__('numpy').load(c.MODEL) as f:assert 'vt' in f and 'ft' in f
from compatible_gaussian_export import export_and_verify
import export_viewer_upright
from floor_reference_cameras import make_cameras
cams,_,_=make_cameras();assert len(cams)>0
sys.path.insert(0,str(P/'third_party/repos/GaussianAvatar'))
from model.network import POP_no_unet
net=POP_no_unet(c_geom=64,geom_layer_type='conv',nf=32,hsize=128).cuda()
record={'status':'passed','cpu_input_audit':m.read(R/'input-audit.json'),'imports':True,'smplx_template_vertices':10475,'uv_atlas_available':True,'vposer_loaded':True,'GaussianAvatar_network_constructed':True,'export_helpers_imported':True,'review_cameras':len(cams),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'optimizer_updates':0,'training_started':False,'target_server_tested':False,'runtime':'Existing local WSL environment; build sources supplied but clean remote build not tested','known_limitation':'Coarse fitting masks 59/175; physical camera accuracy and ghosting not accepted'}
m.save(P/'provenance/portability-validation.json',record)
print(json.dumps(record,ensure_ascii=False))
