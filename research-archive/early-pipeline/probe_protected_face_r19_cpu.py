from pathlib import Path
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import numpy as np
import torch
torch.set_num_threads(4)
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-protected-face-20261111/src')
from shuolee_portrait.protected_face import optimize_protected_face
from shuolee_portrait.world_scale import scale_config
from utils.sh_utils import C0

N,H,W=2,24,24
yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
grid=torch.stack((xx,yy),-1).float()

class Model:
    def __init__(self):
        self._xyz=torch.nn.Parameter(torch.tensor([[9.,11.,1.],[17.,15.,1.]]))
        self._scaling=torch.nn.Parameter(torch.tensor([[5.,5.,1.],[3.,3.,1.]]).log())
        self._rotation=torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]*N))
        self._features_rest=torch.nn.Parameter(torch.empty(N,0,3))
        self._opacity=torch.nn.Parameter(torch.tensor([[3.],[3.]]))
        self._features_dc=torch.nn.Parameter(torch.zeros(N,1,3))
        self.active_sh_degree=self.max_sh_degree=0
        self.binding=torch.arange(N)
        self.binding_counter=torch.ones(N,dtype=torch.int)
        self.timestep=123
        self.face_center=torch.zeros(N,3)
        self.densification_ids=torch.arange(N)
        self.next_densification_id=N
        self.optimizer=torch.optim.Adam([dict(name=key,params=[getattr(self,value)],lr=0.)
            for key,value in {'xyz':'_xyz','scaling':'_scaling','rotation':'_rotation','f_rest':'_features_rest',
                              'opacity':'_opacity','f_dc':'_features_dc'}.items()])
        for group in self.optimizer.param_groups:
            group['params'][0].grad=torch.full_like(group['params'][0],.002)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    def select_mesh_by_timestep(self,i):
        self.timestep=i
        self.face_center=torch.tensor([[.02*(i%3),0,0],[0,0,0]])
    @property
    def get_xyz(self): return self._xyz+self.face_center
    @property
    def get_scaling(self): return self._scaling.exp()
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_opacity(self): return self._opacity.sigmoid()
    @property
    def get_features(self): return torch.cat((self._features_dc,self._features_rest),1)

def render(camera,scene,pipe,background,override_color=None):
    p=scene.get_xyz[:,None,None,:2]
    scale=scene.get_scaling[:,None,None,:2]
    a=scene.get_opacity[:,0,None,None]*(-.5*((grid-p)/scale).square().sum(-1)).exp()
    w=torch.stack((a[0],a[1]*(1-a[0])))
    color=override_color if override_color is not None else scene.get_features[:,0]*C0+.5
    return {'render':torch.einsum('nhw,nc->chw',w,color)}

def data():
    original=[]
    aux=[]
    rows=[]
    observed=[]
    for i in range(28):
        inside=((xx-9).square()+(yy-11).square()<=12) | ((xx-17).square()+(yy-15).square()<=5)
        face=((xx-9).square()+(yy-11).square()<=9)
        outside=((xx-9).square()+(yy-11).square()>=105)&((xx-17).square()+(yy-15).square()>=45)
        target=torch.ones(3,H,W)*(.53 if i<24 else .55)
        mask=inside.float()[None]
        if i<24:
            row={'frame':f'o{i:02d}','face_box_xyxy':[5,7,13,16]}
            rows.append(row)
            original.append(NS(box=(0,0,W,H),rgb=target,inside=inside,outside=outside,mask=mask,face=face.nonzero().T))
            points=np.zeros((118,3),np.float32)
            points[67:,:]=[9.,11.,.9]
            points[[17,18],:]=[9.,11.,.9]
            observed.append(points)
        else:
            row={'frame':f'a{i-24:02d}'}
            view=NS(box=(0,0,W,H),rgb=target,inside=face,outside=torch.zeros_like(outside),face=None)
            aux.append(NS(row=row,camera=i,slot=i,rgb=target,valid=face,alpha=mask[0],guard_view=view))
    config=scale_config()['render_support']
    return NS(views=original,cameras=list(range(24)),slots=list(range(24)),rows=rows,bins=[i//4 for i in range(24)],
              options=dict(foreground_target=.98,low_alpha=.65),schedule={'world_scale':{'render_support':config}}),aux,observed

def run(fail):
    model=Model()
    constraints,aux,observed=data()
    pose=model.face_center
    with TemporaryDirectory() as tmp:
        path=Path(tmp)
        argv=(model,constraints,aux,torch.tensor([True,False]),render,None,torch.zeros(3))
        if fail:
            with patch.object(torch,'save',side_effect=RuntimeError('synthetic artifact error')):
                r=optimize_protected_face(*argv,observed=observed,directory=path)
        else:
            r=optimize_protected_face(*argv,observed=observed,directory=path)
        assert len(r['baseline']['frames'])==len(r['final']['frames'])==28
        assert all(len(v['draw_order'])==28 for v in r['protected']['cycles'])
        assert r['rng_unchanged'] and model.timestep==123 and model.face_center is pose
        assert r['endpoint']['fixed_model']==r['live_final']['fixed_model']
        assert r['endpoint']['fixed_optimizer']==r['live_final']['fixed_optimizer']
        assert r['final']['all_guards_pass']
        if fail:
            assert r['state']=='rolled_back' and r['reason'].startswith('live_application_failed')
            assert r['live_final']==r['endpoint'] and r['committed_updates']==0
            assert not list(path.iterdir())
        else:
            assert r['state']=='committed',[(p['phase'],p['gains'],p.get('failed_frames')) for p in r['protected']['cycles']]
            assert r['committed_updates']==r['selected']['optimizer_updates']
            assert r['final_gains']['joint']>=r['config']['minimum_gain']
            receipt=torch.load(path/r['optimizer_state']['file'],weights_only=True)
            for name in receipt['groups']:
                assert int(receipt['groups'][name]['step'])==1
                assert int(receipt['local_groups'][name]['step'])==1+r['committed_updates']
                assert torch.equal(receipt['groups'][name]['parameter'],receipt['local_groups'][name]['parameter'])
        print(r['state'],r['attempted_updates'],r['committed_updates'],r['final_gains'])

run(False)
run(True)
