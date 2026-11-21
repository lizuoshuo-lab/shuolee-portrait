"""Export an immutable head snapshot; reuse the existing rigid PLY exporter."""
import argparse,json,sys
from pathlib import Path
import numpy as np

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.export import rotate_ply

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/head-gaussian-first-cycle')
RUNS=('head-gaussian-first-cycle','head-gaussian-long300k')

def main(step,run=RUNS[0]):
    if run not in RUNS:raise ValueError('Unknown head run: '+str(run))
    root=ROOT.parent/run
    snap=root/'gaussian-run'/f'step-{step:06d}'
    result=rt.read(snap/'result.json')
    assert result['step']==step and result['native_ply_parity']['passed']
    assert result['world_ply_sha256']==rt.sha(snap/'world.ply')
    out=root/'delivery'/f'step-{step:06d}'
    out.mkdir(parents=True,exist_ok=False)
    R=np.diag([1.,-1.,-1.]);origin=np.zeros(3)
    v=rotate_ply(snap/'world.ply',out/'point_cloud.ply',R,origin)
    v.update(coordinate_space='Reference head camera frame rotated to +Y preview; no measured gravity or metric calibration',
             source_sha256=result['world_ply_sha256'],reference_frame=result['reference_frame'],
             step=step,active_sh_degree=result['active_sh_degree'],native_source_ply_parity=result['native_ply_parity'],
             native_rotated_render_parity='pending',viewer_ui_review='pending',quality_accepted=False)
    rows=rt.read(root/'inputs.json')['records'];r=rows[result['reference_slot']]
    assert r['frame']==result['reference_frame']
    E=np.array(r['w2c']);newE=E.copy();newE[:3,:3]=E[:3,:3]@R.T
    newE[:3,3]=E[:3,:3]@origin+E[:3,3]
    c2w=np.linalg.inv(newE);K=np.array(r['K'])
    camera=dict(id=0,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],
                position=c2w[:3,3].tolist(),rotation=c2w[:3,:3].tolist(),
                fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),
                pixelConvention='pixel_corners',projectionModel='pinhole_export_assumption')
    # Fixed-frame cloud has only one directly corresponding recorded camera.
    # Other training frames include nonrigid FLAME pose/expression changes.
    rt.save(out/'cameras.json',[camera])
    rt.save(out/'validation.json',v)
    rt.save(out/'context.json',dict(label=f'demo face+hat {step} steps / reference frame33',steps=step,
        referenceFrameId=r['frame'],referenceFrameIsPoseSnapshot=True,initialView=0,
        model=dict(sha256=v['sha256']),world=v['coordinate_space'],
        rendererContract=dict(schema='shuolee-renderer-contract-v1',cameraProjection='pinhole',
            gaussianEvaluation='screen-space-2d',previewBackend='spark',scope='preview-only',
            trainingRenderer='GaussianAvatars diff-gaussian-rasterization full-K local adapter',
            nativeImageEquivalence='source PLY roundtrip passed; Spark equivalence not established')))
    print(json.dumps(v,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True)
    p.add_argument('--run',choices=RUNS,default=RUNS[0])
    args=p.parse_args();main(args.step,args.run)
