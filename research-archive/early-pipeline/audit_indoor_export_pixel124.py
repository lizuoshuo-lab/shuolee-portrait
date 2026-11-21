"""One native-buffer read-only diagnosis of the observed PLY mismatch pixel."""
import inspect
from pathlib import Path
import diagnose_indoor_circle16_export124 as prior


def buffer_fields(tensor, spec):
    raw=tensor.detach().cpu().numpy()
    assert raw.dtype.name=='uint8'
    offset=0;result={}
    for name,dtype,count,width in spec:
        offset=(offset+127)//128*128
        result[name]=prior.np.frombuffer(raw,dtype=dtype,count=count*width,offset=offset).reshape(count,width)
        offset+=result[name].nbytes
    return result


def inspect_pixel(torch, model, plain, cam, render, pipe, bg, native, portable, out, snap):
    import math
    import diff_gaussian_rasterization as dgr
    import numpy as np
    n=len(model.get_xyz);h,w=native.shape[1:]
    def raw_render(pc):
        blank=torch.Tensor([])
        args=(bg,pc.get_xyz,blank,pc.get_opacity,pc.get_scaling,pc.get_rotation,1.,blank,
              cam.world_view_transform.cuda(),cam.full_proj_transform.cuda(),math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),
              h,w,pc.get_features,pc.active_sh_degree,cam.camera_center.cuda(),False,False)
        count,rgb,radii,geom,bins,img=dgr._C.rasterize_gaussians(*args)
        gs=buffer_fields(geom,[('depth','<f4',n,1),('clamped','?',n,3),('radius','<i4',n,1),
            ('xy','<f4',n,2),('cov3d','<f4',n,6),('conic','<f4',n,4),('rgb','<f4',n,3)])
        bs=buffer_fields(bins,[('ids','<u4',count,1)])
        ims=buffer_fields(img,[('T','<f4',h*w,1),('n','<u4',h*w,1),('ranges','<u4',h*w,2)])
        # The actual caller provides an external radii tensor. The allocated
        # GeometryState.internal_radii is unused, hence not valid evidence.
        gs['radius']=radii.detach().cpu().numpy()[:,None]
        assert np.isfinite(gs['xy'][radii.cpu().numpy()>0]).all()
        return dict(rgb=rgb,g=gs,b=bs,i=ims,tile_instances=count)
    a,b=raw_render(model),raw_render(plain)
    assert torch.equal(a['rgb'],native) and torch.equal(b['rgb'],portable),'Raw buffer call differs from production render'
    p=(native-portable).abs().max(0).values
    pixels=torch.nonzero(p>=.001).cpu().numpy()
    all_records=[]
    for y,x in pixels:
        index=int(y*w+x);tile=int((y//16)*((w+15)//16)+(x//16))
        details=[];id_lists=[]
        for tag,r in [('checkpoint',a),('ply',b)]:
            start,end=r['i']['ranges'][tile]
            ids=r['b']['ids'][int(start):int(end),0].astype(np.int64)
            id_lists.append(ids)
            g=r['g'];xy=g['xy'][ids].astype(np.float64)
            dx,dy=(xy-np.array([x,y])).T
            q=g['conic'][ids].astype(np.float64)
            power=-.5*(q[:,0]*dx*dx+q[:,2]*dy*dy)-q[:,1]*dx*dy
            alpha=np.minimum(.99,q[:,3]*np.exp(np.minimum(power,0)))
            alpha[(power>0)|(alpha<1/255)]=0
            T=1.;color=np.zeros(3);contributions=[]
            for j,(pid,opacity) in enumerate(zip(ids,alpha)):
                nt=T*(1-opacity)
                if nt<.0001:break
                c=g['rgb'][pid].astype(float)*opacity*T
                color+=c
                if opacity>0:contributions.append((int(pid),float(opacity),float(T),c.tolist(),float(power[j])))
                T=nt
            details.append(dict(tag=tag,cuda_rgb=r['rgb'][:,y,x].cpu().tolist(),float64_from_native_conic_rgb=color.tolist(),
                float64_replay_error=float(np.max(np.abs(color-r['rgb'][:,y,x].cpu().numpy()))),
                cuda_transmittance=float(r['i']['T'][index,0]),cuda_last_contributor=int(r['i']['n'][index,0]),
                candidate_count=len(ids),used_ids=[v[0] for v in contributions],
                contributions=contributions))
        same=np.array_equal(*id_lists)
        all_records.append(dict(x=int(x),y=int(y),max_rgb_difference=float(p[y,x]),same_ordered_tile_candidates=same,
            candidate_only_checkpoint=np.setdiff1d(id_lists[0],id_lists[1]).tolist(),
            candidate_only_ply=np.setdiff1d(id_lists[1],id_lists[0]).tolist(),details=details))
    visible=(a['g']['radius'][:,0]>0)|(b['g']['radius'][:,0]>0)
    result=dict(updated=prior.now(),source_checkpoint_sha256=prior.sha(snap/'checkpoint.pt'),
        source_ply_sha256=prior.sha(snap/'world.ply'),binary=str(dgr._C.__file__),binary_sha256=prior.sha(dgr._C.__file__),
        wrapper=str(dgr.__file__),raw_call_matches_training_renderer=True,
        diagnostic_pixels_threshold=.001,threshold_not_acceptance=True,records=all_records,
        visible_property_max_abs={k:float(np.max(np.abs(a['g'][k][visible].astype(float)-b['g'][k][visible].astype(float))))
                                  for k in ('xy','depth','cov3d','conic','rgb','radius')},
        optimizer_updates=0,training_resumed=False,quality_accepted=False)
    prior.save(out/'result.json',result)
    # Decisive summary only; individual contributors remain in the JSON.
    prior.save(out/'summary.json',dict(updated=result['updated'],pixels=[dict(x=r['x'],y=r['y'],delta=r['max_rgb_difference'],
        same_candidates=r['same_ordered_tile_candidates'],cuda_T=[d['cuda_transmittance'] for d in r['details']],
        cpu_replay_error=[d['float64_replay_error'] for d in r['details']],
        changed_included_ids=sorted(set(r['details'][0]['used_ids'])^set(r['details'][1]['used_ids']))) for r in all_records],
        visible_property_max_abs=result['visible_property_max_abs']))
    print('EXPORTED_PIXEL_BUFFER_AUDIT_READY',len(all_records),flush=True)


def main():
    out=prior.ROOT/'export-pixel-buffer-audit124'
    if out.exists():
        assert (out/'driver-input-repair.json').exists() and not (out/'result.json').exists()
    else:out.mkdir()
    source_root=prior.PROJECT/'src/avatar_core/submodules/diff-gaussian-rasterization'
    planfile=out/('effective-plan.json' if (out/'plan.json').exists() else 'plan.json')
    prior.save(planfile,dict(updated=prior.now(),hypothesis='FP32 scale rounding changes native conic evaluation or discrete raster inclusion at a few pixels; not a position/SH mismatch.',
        source_hashes={str(p):prior.sha(p) for p in [Path(__file__),prior.ROOT/'gaussian-run/step-018000/checkpoint.pt',
            prior.ROOT/'gaussian-run/step-018000/world.ply',source_root/'cuda_rasterizer/forward.cu',source_root/'cuda_rasterizer/rasterizer_impl.cu']},
        method='Read actual CUDA geometry/bin/image buffers using unchanged native forward. Validate raw replay bit-exact. Inspect pixels with RGB difference>=0.001, original threshold0.005 remains unchanged.',
        budget_seconds=120,optimizer_updates=0,no_parameter_mutation=True,no_kernel_changes=True))
    code=inspect.getsource(prior.main)
    code=code[:code.index('        delta=(native-portable).abs()')]
    code=code.replace("out=ROOT/'export-failure-diagnostic'","out=ROOT/'export-pixel-buffer-audit124'")
    code=code.replace('    out.mkdir(exist_ok=False)\n','')
    code+='''        inspect_pixel(torch,model,plain,cam,render,pipe,bg,native,portable,out,snap)
        for path,digest in hashes.items():assert sha(path)==digest
'''
    ns=dict(prior.main.__globals__,inspect_pixel=inspect_pixel)
    exec(compile(code,__file__+':immutable-checkpoint-load','exec'),ns)
    ns['main']()

if __name__=='__main__':main()
