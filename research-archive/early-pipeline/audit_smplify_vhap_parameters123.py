"""CPU evidence for the zero-offset -> refinement stage mismatch."""
from datetime import datetime,timezone
import numpy as np
import audit_gaussianavatars_params123 as a

def stats(v):
    return dict(minimum=float(np.min(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(np.max(v)))

def main():
    root=a.REC/'demo-vhap-initialization';out=root/'smplify-vhap-parameter-audit'
    a.lp(out).mkdir(exist_ok=True)
    model='D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz'
    with np.load(a.lp(model),allow_pickle=False) as z:
        template=z['v_template'].copy();dirs=z['shapedirs'][...,:10].copy();faces=z['f'].copy()
        dominant_joint=np.argmax(z['weights'],axis=-1)
    runs={}
    for name in ('surface','joint','mask','offset-init','normalized-laplacian','single-landmark-weight'):
        folder=root/('smplify-vhap-'+name);path=folder/'epoch-30/tracked-smplx-params.npz'
        if not a.lp(path).exists():continue
        with np.load(a.lp(path),allow_pickle=False) as z:
            offset=z['static_offset'][0];beta=z['betas'];frame_ids=z['frame_ids']
        cano=template+np.einsum('vck,k->vc',dirs,beta)
        with np.load(a.lp(folder/'initial/tracked-smplx-params.npz'),allow_pickle=False) as z:
            initial_beta=z['betas'].copy()
        shape_only=np.einsum('vck,k->vc',dirs,beta-initial_beta)*1000
        tri=cano[faces];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(cano)
        for col in range(3):np.add.at(normals,faces[:,col],fn)
        normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
        component=(offset*normals).sum(-1)*1000;length=np.linalg.norm(offset,axis=-1)*1000
        tangent=np.sqrt(np.maximum(0,length**2-component**2))
        cfg=a.read(folder/'config.json');cpath=folder/'comparison/result.json';cmp=a.read(cpath) if a.lp(cpath).exists() else None
        runs[name]=dict(checkpoint=str(path),sha256=a.sha(path),frames=len(frame_ids),offset_norm_mm=stats(length),
            normal_signed_mm=stats(component),tangent_mm=stats(tangent),canonical_extent_m=np.ptp(cano,axis=0).tolist(),
            signed_mesh_volume_m3=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6),
            outward_over_1mm=int((component>1).sum()),inward_beyond_1mm=int((component< -1).sum()),
            sub_1mm_offset=int((length<1).sum()),total_vertices=len(offset),
            saved_static_offset_lr=cfg['lr']['static_offset'],reg_offset=cfg['w']['reg_offset'],reg_offset_lap=cfg['w']['reg_offset_lap'],
            review_checkpoint_hash_matches=cmp['checkpoint_sha256']==a.sha(path) if cmp else None)
        # Fixed SMPL-X dominant skinning groups, not error-selected vertices.
        regions={'torso':[0,3,6,9], 'arms':[13,14,16,17,18,19],
            'legs_and_feet':[1,2,4,5,7,8,10,11], 'head_neck':[12,15,22,23,24],
            'hands':[20,21]+list(range(25,55))}
        runs[name]['regions_by_dominant_skinning_joint']={}
        for region,ids in regions.items():
            selected=np.isin(dominant_joint,ids)
            runs[name]['regions_by_dominant_skinning_joint'][region]=dict(vertices=int(selected.sum()),
                displacement_mm=stats(length[selected]),normal_signed_mm=stats(component[selected]),
                outward_over_1mm=int((component[selected]>1).sum()),
                shape_only_displacement_mm=stats(np.linalg.norm(shape_only[selected],axis=-1)))
    repo=a.REPO if hasattr(a,'REPO') else None
    first=5e-4*.1;last=first*.9**29
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='complete_read_only',optimizer_updates=0,runs=runs,
        primary_finding='The completed surface/joint/mask controls initialized static_offset to zero but executed only rgb_global_tracking at lr_scale0.1; the original rgb_init_offset500-step stage at scale1 was omitted. The separately registered offset-init control restores that stage.',
        rates=dict(original_offset_init=5e-4,actual_global_first=first,actual_global_last_epoch=last,
            nominal_step_size_sum_m=59*sum(first*.9**e for e in range(30)),
            interpretation='Learning-rate sum is a schedule comparison, not an Adam displacement bound.'),
        distinct_findings=['No global/global second scaling: configure_optimizer multiplies by0.1 once; scheduler applies0.9 per epoch.',
            'The original base reg_offset300 and Laplacian1e6 were retained, not accidentally zeroed or changed.',
            'A separate exact-operator audit found the inherited SMPL-X D-A differs from original VHAP/FLAME A/degree-I. Equal numeric weights do not mean equal smoothing strength; see offset-init/regional-gradient-audit/operator-result.json.',
            'static_offset is an unconstrained3D vector; its magnitude alone does not establish outward-normal expansion.',
            'The source photometric objective comments out alpha/silhouette loss. The separately registered mask control adds it explicitly.',
            'Saved canonical body dimensions are in meter-scale units; no1000-fold unit mismatch is evident.',
            'The rendered review checkpoint SHA is checked against the actual saved mesh, not an old SMPL initialization.'],
        limitations=['A skipped initialization stage is a confirmed integration choice, not yet proof that restoring it alone yields good clothing geometry.',
            'Remaining pose/hand errors and camera calibration are separate from this parameter audit.'])
    a.save(out/'result.json',result);print(__import__('json').dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
