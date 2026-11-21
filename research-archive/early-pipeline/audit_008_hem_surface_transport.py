"""Trace the archived hem gap through the terminal 008 training poses on CPU."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from native_residual_fit122 import TriangleBVH
from review_bound_alpha_ab import pin, read, require, sha


LAB = Path('/opt/gs-slam-lab')
RUN = LAB/'shuolee-smpl-head-appearance-008/run'
SNAPSHOT = RUN/'gaussian-run/step-010000-head-000600'
REPLAY = SNAPSHOT/'posed-replay'
ALPHA_PANEL = LAB/'shuolee-smpl-unified-008-matched-views/late-new-native.png'
OUT = LAB/'shuolee-smpl-008-hem-surface-transport'
ANCHOR = 'frame_000058'
HOLDOUTS = ['frame_000015','frame_000043','frame_000071','frame_000098']
PANELS = ['frame_000047','frame_000048','frame_000050','frame_000053','frame_000054','frame_000057',ANCHOR]
PINS = {
    RUN/'result.json':'e3e5738bdaaaad0e666b5dc63f0feee559a43f804e0abfa3785e8cf38d70d0af',
    RUN/'data/training-inputs.json':'e485e3657b57d4de800623ee5f0f60da3fce8a50820afdd5f456bedf3e4443d8',
    RUN/'fit/selection.json':'44e85e52a01642cda7c294f9afbb8ef4520f3e31b53cf218fadca0dcf5c0950d',
    RUN/'gaussian-run/training-cameras.json':'aa261c3522abd441d816fa21dc97a7c9fa54055be46501a7d9013f1a5327e396',
    SNAPSHOT/'bound-replay.json':'2fb89dd3179303b31c88330a1ee7ccda298876b1e1e755a6934decb2dc0aac19',
    REPLAY/'result.json':'f0cfda522bd03307bc56ae759873f176b885e92f8635c1e1d717cbc701fbd0ef',
    REPLAY/'meshes.npz':'73c1945a4f04a3bb928c7d542a454a827b234634d9c030843ed18462ca542928',
    ALPHA_PANEL:'ee62e9673362406d73e0d3153d983ca0a4e5e9b79f6f3633ececbe41b98be942',
}
ROI_SHA = 'd4640409df7cf3756d599ab75c9f489bc31a8f41a6219dfa56db8c1d8be53225'
OWNER_HITS = {4047:8,4198:312,4199:203,4237:167,5313:39,14579:1,14730:218,14731:237,15276:177,15842:1}
DEPTH_TOLERANCE_M = 1e-5
PANEL_WH = 204,120


def array_sha(values):
    return hashlib.sha256(values.tobytes(order='C')).hexdigest()


def anchor_pixels():
    with Image.open(ALPHA_PANEL) as page:
        matte = np.asarray(page.crop((834,21,1668,1199)).convert('L'))
        alpha = np.asarray(page.crop((0,1220,834,2398)).convert('L'))
    y = np.arange(342,1520)[:,None]
    groups,_ = label((matte>=253) & (alpha<128) & (y>=850) & (y<=1250),np.ones((3,3)))
    component = groups[1209-342,1305-940]
    require(component>0,'Archived hem seed does not identify a component.')
    yy,xx = np.nonzero(groups==component)
    full = np.zeros((1520,2704),dtype=bool)
    full[yy+342,xx+940] = True
    require(len(xx)==1363 and array_sha(full)==ROI_SHA,'Archived hem region changed.')
    return np.column_stack((xx+940,yy+342))


def anchor_surface(vertices,faces,camera,pixels):
    K,E = np.asarray(camera['K']),np.asarray(camera['w2c'])
    R,t = E[:3,:3],E[:3,3]
    center = -R.T@t
    rays_camera = np.c_[pixels+.5,np.ones(len(pixels))] @ np.linalg.inv(K).T
    rays = rays_camera @ R
    distance,owners = TriangleBVH(vertices,faces).trace(center,rays)
    ids,counts = np.unique(owners,return_counts=True)
    require(dict(zip(ids.tolist(),counts.tolist(),strict=True))==OWNER_HITS,'Archived nearest owner votes changed.')
    world = center + rays/np.linalg.norm(rays,axis=1)[:,None]*distance[:,None]
    triangles = vertices[faces[owners]].astype(np.float64)
    first = triangles[:,1]-triangles[:,0]
    second = triangles[:,2]-triangles[:,0]
    offset = world-triangles[:,0]
    dot = lambda a,b:np.einsum('ij,ij->i',a,b)
    aa,ab,bb = dot(first,first),dot(first,second),dot(second,second)
    ap,bp = dot(first,offset),dot(second,offset)
    den = aa*bb-ab**2
    u,v = (ap*bb-bp*ab)/den,(bp*aa-ap*ab)/den
    bary = np.column_stack((1-u-v,u,v))
    require(np.isfinite(bary).all() and bary.min()>=-1e-8
            and np.max(abs(np.einsum('ij,ijk->ik',bary,triangles)-world))<1e-8,'Barycentric anchor did not replay.')
    return owners,bary


def surface_projection(vertices,faces,camera,owners,bary):
    K,E = np.asarray(camera['K']),np.asarray(camera['w2c'])
    width,height = camera['wh']
    center = -E[:3,:3].T@E[:3,3]
    world = np.einsum('ij,ijk->ik',bary,vertices[faces[owners]])
    camera_points = world @ E[:3,:3].T + E[:3,3]
    require((camera_points[:,2]>.2).all(),'Transport crosses the rasterizer depth cutoff.')
    q = camera_points@K.T
    projected = q[:,:2]/q[:,2:3]-.5
    native = np.rint(projected).astype(np.int64)
    distance,hit = TriangleBVH(vertices,faces).trace(center,world-center)
    inside = ((native>=0)&(native<[width,height])).all(axis=1)
    visible = (hit==owners)&(abs(distance-np.linalg.norm(world-center,axis=1))<DEPTH_TOLERANCE_M)&inside
    return projected,visible,np.unique(native[visible],axis=0)


def support_panel(source,matte,render,pixels):
    width,height = PANEL_WH
    center = (pixels.min(axis=0)+pixels.max(axis=0))//2
    x,y = center-[width//2,height//2]
    crop = (int(x),int(y),int(x+width),int(y+height))
    xy = pixels-[x,y]
    require(((xy>=35)&(xy<[width-35,height-35])).all(),'Fixed native crop margins changed.')
    xx,yy = xy.T
    source_tile = source[y:y+height,x:x+width]
    matte_tile = matte[y:y+height,x:x+width]
    shown = source_tile.copy()
    observed = matte_tile[yy,xx]>=253
    for group,tint in ((observed,[20,255,35]),(~observed,[255,30,20])):
        shown[yy[group],xx[group]] = (.57*shown[yy[group],xx[group]]+.43*np.array(tint)).astype(np.uint8)
    return crop,[source_tile,shown,matte_tile,render[y:y+height,x:x+width]]


def audit():
    pins = {str(path):pin(path,digest) for path,digest in PINS.items()}
    terminal, replay = read(RUN/'result.json'),read(REPLAY/'result.json')
    selection = read(RUN/'fit/selection.json')
    rows = read(RUN/'data/training-inputs.json')['records']
    training_cameras = read(RUN/'gaussian-run/training-cameras.json')
    cameras = training_cameras['records']
    frames = [row['frame'] for row in rows]
    require(terminal['state']=='completed' and terminal['bound_replay_sha256']==pins[str(REPLAY/'result.json')]
            and terminal['head_appearance']['state']=='completed'
            and replay['mesh_sha256']==pins[str(REPLAY/'meshes.npz')]
            and replay['bound_replay_sha256']==pins[str(SNAPSHOT/'bound-replay.json')]
            and replay['training_inputs_sha256']==training_cameras['source_sha256']==pins[str(RUN/'data/training-inputs.json')]
            and replay['training_camera_sha256']==pins[str(RUN/'gaussian-run/training-cameras.json')]
            and replay['retained_holdouts']==selection['retained_holdouts']==HOLDOUTS
            and len(frames)==19 and set(frames)==set(selection['selected_frames']) and set(frames).isdisjoint(HOLDOUTS),
            'Terminal training, mesh, camera or holdout lineage changed.')
    with np.load(REPLAY/'meshes.npz',allow_pickle=False) as z:
        require(z['frame_ids'].tolist()==frames and z['pose_indices'].tolist()==list(range(len(frames)))
                and z['source_frames'].tolist()==[row['source_frame_index'] for row in rows]
                and z['image_sha256'].tolist()==[row['source_sha256'] for row in rows]
                and z['mask_sha256'].tolist()==[row['mask_sha256'] for row in rows],
                'Saved posed meshes no longer match the training roster.')
        vertices,faces = z['vertices_world'],z['faces']
    pixels = anchor_pixels()
    anchor = frames.index(ANCHOR)
    owners,bary = anchor_surface(vertices[anchor],faces,cameras[anchor],pixels)
    records,projected_all,visible_all,panels = [],[],[],{}
    for i,(row,camera,rendered,mesh) in enumerate(zip(rows,cameras,replay['records'],vertices,strict=True)):
        frame = row['frame']
        require(camera==rendered['camera'] and camera['frame']==rendered['frame']==frame
                and camera['pose_index']==i and camera['pixel_convention']=='pixel_corners'
                and all(row[key]==camera[key] for key in ('K','w2c','wh'))
                and all(row[key]==rendered[key] for key in ('source_frame_index','source_sha256','mask_sha256')),
                'Native per-view mesh, image or camera changed.')
        files = {Path(row['image']):row['source_sha256'],Path(row['mask']):row['mask_sha256'],
                 REPLAY/rendered['image']:rendered['image_sha256']}
        pins.update({str(path):pin(path,digest) for path,digest in files.items()})
        projected,visible,observed = surface_projection(mesh,faces,camera,owners,bary)
        if frame==ANCHOR:
            require(visible.all() and np.max(abs(projected-pixels))<1e-7,'Native anchor pixel replay changed.')
        projected_all.append(projected)
        visible_all.append(visible)
        reason = selection['reasons'][frame]
        record = dict(frame=frame,source_frame_index=row['source_frame_index'],side=reason['side'],source_bin=reason['source_bin'],
            camera=camera,posed_vertices_sha256=array_sha(mesh),projected_anchor_xy_sha256=array_sha(projected),
            frontmost_anchor_mask_sha256=array_sha(visible),source_sha256=row['source_sha256'],matte_sha256=row['mask_sha256'],
            replay_rgb_sha256=rendered['image_sha256'],frontmost_samples=int(visible.sum()),
            distinct_frontmost_pixels=len(observed),positive_matte_pixels=0,zero_matte_pixels=0,
            fractional_matte_pixels=0,positive_mask_black_rgb_mae_uint8=None,
            positive_mask_pixels_rgb_error_over_80_uint8=0)
        if len(observed):
            with Image.open(row['image']) as im,Image.open(row['mask']) as matte_im,Image.open(REPLAY/rendered['image']) as rgb:
                require(list(im.size)==list(matte_im.size)==list(rgb.size)==row['wh'],'Native support raster changed.')
                source = np.asarray(im.convert('RGB'))
                matte = np.asarray(matte_im.convert('L'))
                render = np.asarray(rgb.convert('RGB'))
            x,y = observed.T
            positive = matte[y,x]>=253
            zeros = matte[y,x]==0
            error = abs(source[y,x].astype(np.int16)-render[y,x]).mean(axis=1)
            record.update(positive_matte_pixels=int(positive.sum()),zero_matte_pixels=int(zeros.sum()),
                fractional_matte_pixels=int((~positive&~zeros).sum()),
                positive_mask_black_rgb_mae_uint8=float(error[positive].mean()) if positive.any() else None,
                positive_mask_pixels_rgb_error_over_80_uint8=int(((error>80)&positive).sum()))
            if frame in PANELS:
                crop,panels[frame] = support_panel(source,matte,render,observed)
                record['proof_crop_xyxy'] = crop
        records.append(record)
    require(set(panels)==set(PANELS),'Missing representative crop.')
    helper = Path(inspect.getsourcefile(TriangleBVH)).resolve()
    pins[str(helper)] = sha(helper)
    pins[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    evidence = dict(state='train_pose_surface_support_measured',subject='008',anchor=ANCHOR,anchor_roi_sha256=ROI_SHA,
        anchor_roi_pixels=len(pixels),anchor_owner_counts=OWNER_HITS,anchor_barycentric_sha256=array_sha(bary),
        shared_topology_sha256=array_sha(faces),depth_tolerance_m=DEPTH_TOLERANCE_M,records=records,source_pins=pins,
        protected_frames=HOLDOUTS,cohort='Exact 19 fitted Gaussian training views of the delivered terminal.',
        support_scope='Frontmost frozen mesh and original matte agree per native sample. Mesh transport is a surface proxy.',
        color_scope='Archived black-composited RGB pixel difference. Current per-view Alpha is unresolved.',
        proof_rows=PANELS,proof_columns=['original RGB','native support overlay','training matte','terminal Gaussian RGB'],
        proof_overlay='Green means matte >=253. Red means lower matte. Original RGB remains in the adjacent tile.',
        formal_evaluation=False,gpu_calls=0,optimizer_updates=0,quality_accepted=False)
    arrays = dict(frames=np.asarray(frames),anchor_pixels=pixels,anchor_owners=owners,anchor_barycentric=bary,
                  native_projected_xy=np.stack(projected_all),frontmost_same_surface=np.stack(visible_all))
    return evidence,arrays,panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','record'))
    args = parser.parse_args()
    evidence,arrays,panels = audit()
    if args.action=='record':
        OUT.mkdir(exist_ok=False)
        data = OUT/'surface-transport.npz'
        np.savez_compressed(data,**arrays)
        width,height = PANEL_WH
        page = Image.new('RGB',(4*width,len(PANELS)*height))
        for i,frame in enumerate(PANELS):
            for j,tile in enumerate(panels[frame]):
                page.paste(Image.fromarray(tile).convert('RGB'),(j*width,i*height))
        proof = OUT/'source-matte-render.png'
        page.save(proof)
        result = OUT/'result.json'
        evidence.update(transport_array_sha256=sha(data),proof_panel_sha256=sha(proof))
        result.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(result=str(result),sha256=sha(result),proof=str(proof))))
    else:
        print(json.dumps(dict(state=evidence['state'],anchor_owner_counts=evidence['anchor_owner_counts'],
            views=[(row['frame'],row['positive_matte_pixels'],row['zero_matte_pixels']) for row in evidence['records']],gpu_calls=0)))


if __name__=='__main__':
    main()
