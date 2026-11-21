"""Zero-update synthetic viewport parity and pixel-allocation evidence."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--backend-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.package)
    sys.path.insert(0, args.repo)
    from msv2.roi import wrap_renderer, verify_backend, backend_path
    from msv2.io import sha256, save_json
    from gaussian_renderer import render_imp

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(813)
    torch.cuda.manual_seed_all(813)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.tensor([.15, .35, .6], device='cuda')
    settings = dict(model_updates=0, optimizer_updates=0,
        source_commit='5ec9202c2db7d900728bfc09733e85467cff885e',
        package=args.package, repo=args.repo,
        adapter_sha256=sha256(Path(args.package)/'msv2/roi.py'),
        patch_sha256=sha256(Path(args.package)/'msv2/roi_backend.patch'),
        backend=verify_backend(args.repo, args.backend_dir),
        backend_directory=str(backend_path(args.repo,args.backend_dir)),
        proof_scope='synthetic only; no training or real-model quality acceptance',
        gpu=torch.cuda.get_device_name(), cases=[])

    class PC:
        active_sh_degree = 1
        max_sh_degree = 1
        def __init__(self, empty=False):
            count = 37
            xyz = torch.rand(count,3,device='cuda')
            xyz[:,:2] = (xyz[:,:2] - .5)*1.7
            xyz[:,2] = 1.1 + xyz[:,2]*1.4
            if empty:
                xyz[:,:2] = .7
            self._xyz = xyz.requires_grad_()
            self._dc = (torch.rand(count,1,3,device='cuda')*.4).requires_grad_()
            self._rest = (torch.rand(count,3,3,device='cuda')*.2-.1).requires_grad_()
            self._opacity = (torch.rand(count,1,device='cuda')*1.5).requires_grad_()
            self._scaling = torch.full((count,3), -2.5 if not empty else -4.5,device='cuda',requires_grad=True)
            self._rotation = torch.randn(count,4,device='cuda',requires_grad=True)
        @property
        def get_xyz(self): return self._xyz
        @property
        def get_features_dc(self): return self._dc
        @property
        def get_features_rest(self): return self._rest
        @property
        def get_opacity(self): return torch.sigmoid(self._opacity)
        @property
        def get_scaling(self): return torch.exp(self._scaling)
        @property
        def get_rotation(self): return torch.nn.functional.normalize(self._rotation)
        def parameters(self): return [self._xyz,self._dc,self._rest,self._opacity,self._scaling,self._rotation]
        def clear(self):
            for parameter in self.parameters(): parameter.grad=None

    def camera(width,height,offaxis=True):
        fovx,fovy = 1.1,1.3
        proj = torch.zeros(4,4,device='cuda')
        proj[0,0]=1/math.tan(fovx/2)
        proj[1,1]=1/math.tan(fovy/2)
        proj[0,2]=.27 if offaxis else 0
        proj[1,2]=-.19 if offaxis else 0
        proj[2,2]=100/(100-.01)
        proj[2,3]=-100*.01/(100-.01)
        proj[3,2]=1
        return SimpleNamespace(image_width=width,image_height=height,FoVx=fovx,FoVy=fovy,
            world_view_transform=torch.eye(4,device='cuda'),full_proj_transform=proj.T.contiguous(),
            camera_center=torch.zeros(3,device='cuda'))

    for label,fullwh,origin,wh,allculled,empty in [
        ('offaxis_interior',(128,96),(32,16),(64,48),False,False),
        ('odd_right_bottom_padding',(131,99),(96,80),(48,32),False,False),
        ('warmup2',(262,198),(64,32),(96,80),False,False),
        ('full4',(524,396),(128,64),(192,160),False,False),
        ('full_viewport',(128,96),(0,0),(128,96),False,False),
        ('all_culled',(128,96),(32,16),(64,48),True,False),
        ('no_roi_overlap',(128,96),(0,0),(16,16),False,True),
    ]:
        pc=PC(empty)
        full=camera(*fullwh)
        child=SimpleNamespace(**vars(full))
        child.image_width,child.image_height=wh
        child._msv2_roi=dict(full_wh=list(fullwh),viewport_xy=list(origin),wh=list(wh))
        culling=torch.full((len(pc._xyz),),allculled,dtype=torch.bool,device='cuda')
        x,y=origin; w,h=wh; fw,fh=fullwh
        vw,vh=min(w,fw-x),min(h,fh-y)
        grad=torch.randn(3,h,w,device='cuda')
        grad[:,vh:,:]=0; grad[:,:,vw:]=0
        fullgrad=torch.zeros(3,fh,fw,device='cuda')
        fullgrad[:,y:y+vh,x:x+vw]=grad[:,:vh,:vw]
        native,roiruns=[],[]
        buffers={}

        def execute(render,camera_,roi_):
            pc.clear()
            saved=[]
            def pack(tensor):
                if tensor.dtype==torch.uint8 and tensor.is_cuda:
                    saved.append(tensor.numel())
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
                pkg=render(camera_,pc,pipe,bg,culling=culling)
            image=pkg['render']
            image.retain_grad()
            if allculled and not roi_:
                # The unchanged native backend launches zero backward blocks
                # for this case. Its analytic reference is the zero gradient.
                for parameter in pc.parameters(): parameter.grad=torch.zeros_like(parameter)
                pkg['viewspace_points'].grad=torch.zeros_like(pkg['viewspace_points'])
            elif roi_:
                image.backward(grad)
            else:
                image.backward(fullgrad)
            torch.cuda.synchronize()
            gradients=[p.grad.detach().clone() for p in pc.parameters()]
            dummy=pkg['viewspace_points'].grad.detach().clone()
            if roi_:
                expected_padding=bg[:,None,None].expand_as(image)
                pad=torch.ones(h,w,dtype=torch.bool,device='cuda');pad[:vh,:vw]=False
                assert torch.equal(image[:,pad],expected_padding[:,pad])
            return dict(image=image.detach().clone(),gradients=gradients,dummy=dummy,
                area=pkg['area_max'].detach().clone(),buffers=saved,
                visibility=pkg['visibility_filter'].detach().clone(),radii=pkg['radii'].detach().clone())

        wrapped=wrap_renderer(render_imp,args.repo,args.backend_dir)
        for _ in range(3): native.append(execute(render_imp,full,False))
        for _ in range(3): roiruns.append(execute(wrapped,child,True))
        expected=bg[:,None,None].expand(3,h,w).clone()
        expected[:,:vh,:vw]=native[0]['image'][:,y:y+vh,x:x+vw]
        error=float((roiruns[0]['image']-expected).abs().max())
        assert error==0,(label,'RGB',error)
        comparisons=[]
        names=['xyz','dc','sh','opacity','scaling','rotation','screen_ndc']
        for index,name in enumerate(names):
            def tensor(run): return run['dummy'] if index==6 else run['gradients'][index]
            ref=tensor(native[0])
            native_spread=max(float((tensor(a)-tensor(b)).abs().max()) for a in native for b in native)
            delta=max(float((tensor(a)-tensor(b)).abs().max()) for a in native for b in roiruns)
            maxref=float(ref.abs().max())
            tolerance=max(1e-6,1e-5*maxref,2*native_spread)
            assert delta<=tolerance,(label,name,delta,tolerance,native_spread,maxref)
            comparisons.append(dict(name=name,max_error=delta,reference_max=maxref,
                native_repeat_max_error=native_spread,tolerance=tolerance,passed=True))
        # Prove that a provided mother dummy is passed by identity with no 4x scaling.
        pc.clear()
        mother=render_imp(camera(max(1,fw//4),max(1,fh//4)),pc,pipe,bg,culling=culling)
        child._msv2_shared_viewspace_points=mother['viewspace_points']
        shared=wrapped(child,pc,pipe,bg,culling=culling)
        assert shared['viewspace_points'] is mother['viewspace_points']
        shared['render'].backward(grad)
        shared_delta=float((mother['viewspace_points'].grad-roiruns[0]['dummy']).abs().max())
        assert shared_delta<=comparisons[-1]['tolerance']
        del shared,mother
        pc.clear()
        assert len(native[0]['buffers'])==len(roiruns[0]['buffers'])==4
        if wh!=fullwh:
            assert roiruns[0]['buffers'][2]<native[0]['buffers'][2]
        case=dict(name=label,full_wh=fullwh,viewport_xy=origin,roi_wh=wh,
            full_pixels=fw*fh,roi_pixels=w*h,rgb_max_error=error,
            gradients=comparisons,shared_dummy_identity=True,shared_dummy_max_error=shared_delta,
            full_saved_uint8_bytes=native[0]['buffers'],roi_saved_uint8_bytes=roiruns[0]['buffers'],
            image_state_allocated_for_roi_only=True,
            roi_empty=bool((roiruns[0]['radii']==0).all()))
        settings['cases'].append(case)
        save_json(output/'proof.json',dict(settings,state='running'))
        print(json.dumps(dict(name=label,rgb_max_error=error,max_gradient_error=max(x['max_error'] for x in comparisons),
            buffers=roiruns[0]['buffers'])),flush=True)
        del pc,native,roiruns
        torch.cuda.empty_cache()
    settings['state']='passed'
    settings['quality_accepted']=False
    save_json(output/'proof.json',settings)
    print('MSV2_ROI_SYNTHETIC_PASSED')


if __name__=='__main__':
    main()
