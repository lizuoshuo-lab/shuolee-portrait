"""Read-only published COLMAP feature scale evidence, no Torch/GPU or control.

The original record byte subsets are inspection evidence only, not a training
reconstruction. Decode those bytes with the unchanged author's COLMAP reader.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,mmap,struct,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
SOURCE=BASE/'official-inputs/small-city-training-002/aligned/sparse/0'
OLD=BASE/'independent-evaluation/small-city-boundary-selection-CPU-001'
CANON=BASE/'independent-evaluation/small-city-local-coarse-003-flat-heldout-001'
OUT=BASE/'independent-evaluation/city-boundary-feature-coordinate-audit-CPU-001'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)

def main():
    OUT.mkdir(exist_ok=False);old=read(OLD/'selection.json');report=read(OLD/'report.json');canonical=read(CANON/'protocol.json')
    required=[SOURCE/'cameras.bin',SOURCE/'images.bin',SOURCE/'points3D.bin',OLD/'selection.json',OLD/'selected-sparse-observations.npz',CANON/'protocol.json',Path(canonical['evaluated_input_npz_path']),ROOT/'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py']
    before={str(p):digest(p) for p in required}
    for p in (SOURCE/'cameras.bin',SOURCE/'images.bin',SOURCE/'points3D.bin'):require(before[str(p)]==report['source_hashes'][str(p)],'Original data changed since selector001')
    require(before[str(OLD/'selection.json')]==report['selection_sha256'] and before[str(OLD/'selected-sparse-observations.npz')]==report['coordinate_npz_sha256'],'001 artifacts changed')
    require(before[str(Path(canonical['evaluated_input_npz_path']))]==canonical['evaluated_input_npz_sha256'],'Frozen117 arrays changed')
    selected=old['selected_names'];wanted=set(canonical['test_names']);z=np.load(OLD/'selected-sparse-observations.npz');camera_data=np.load(canonical['evaluated_input_npz_path']);lookup={n:i for i,n in enumerate(camera_data['names'].tolist())}
    ranges=[];subset=[];feature_total=0;outside_metadata=0;minimum=np.array([np.inf,np.inf]);maximum=-minimum;payloads={};headers={}
    dtype=np.dtype([('x','<f8'),('y','<f8'),('id','<i8')])
    with (SOURCE/'images.bin').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
        count=struct.unpack_from('<Q',b,0)[0];offset=8
        for _ in range(count):
            begin=offset;head=struct.unpack_from('<idddddddi',b,offset);offset+=64
            end=b.find(b'\0',offset);require(end>=offset,'Missing name');name=b[offset:end].decode('utf-8');offset=end+1
            n=struct.unpack_from('<Q',b,offset)[0];offset+=8;end=offset+24*n;require(end<=len(b),'Truncated original image')
            if name in wanted:
                a=np.frombuffer(b,dtype=dtype,count=n,offset=offset).copy();a=a[a['id']>=0];xy=np.column_stack([a['x'],a['y']]);require(np.isfinite(xy).all(),'Nonfinite raw feature pixels')
                lo=xy.min(0);hi=xy.max(0);outside=int(np.any((xy<0)|(xy>=[1024,690]),axis=1).sum());feature_total+=len(xy);outside_metadata+=outside;minimum=np.minimum(minimum,lo);maximum=np.maximum(maximum,hi)
                ranges.append(dict(name=name,registered_observations=len(xy),raw_feature_min=lo.tolist(),raw_feature_max=hi.tolist(),outside_published1024x690=int(outside)))
                if name in selected:subset.append(bytes(b[begin:end]));payloads[name]=a;headers[name]=head
            offset=end
        require(offset==len(b) and count==5822 and len(ranges)==117,'Whole image stream count/bounds wrong')
    target=OUT/'actual-image-record-byte-subset.bin'
    with target.open('wb') as f:f.write(struct.pack('<Q',len(subset)));f.writelines(subset)
    # Choose real IDs actually used by 001, retaining original point record and
    # all original track bytes. No rewritten poses, coordinates, errors or IDs.
    requested=set()
    for j in range(8):requested.update(map(int,z[f'view_{j:02d}_ids'][:4]))
    points=[]
    with (SOURCE/'points3D.bin').open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
        count=struct.unpack_from('<Q',b,0)[0];offset=8
        for _ in range(count):
            point_id=struct.unpack_from('<Q',b,offset)[0];tracks=struct.unpack_from('<Q',b,offset+43)[0];end=offset+51+8*tracks;require(end<=len(b),'Truncated original point')
            if point_id in requested:points.append(bytes(b[offset:end]))
            offset=end
        require(offset==len(b) and count==2390040 and len(points)==len(requested),'Point stream/count/sample missing')
    target_points=OUT/'actual-point-record-byte-subset.bin'
    with target_points.open('wb') as f:f.write(struct.pack('<Q',len(points)));f.writelines(points)
    reader_path=ROOT/'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py';spec=importlib.util.spec_from_file_location('actual_COLMAP_record_subset_reader',reader_path);reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
    cameras=reader.read_cameras_binary(str(SOURCE/'cameras.bin'));images=reader.read_images_binary(str(target));sample_points=reader.read_points3D_binary(str(target_points));official={im.name:im for im in images.values()}
    camera=cameras[1];require(camera.model=='PINHOLE' and camera.width==1024 and camera.height==690,'Unexpected camera model/size')
    controls=[];all_xy=[];all_projected=[];point_controls=[]
    for j,name in enumerate(selected):
        im=official[name];original=payloads[name];valid=im.point3D_ids>=0
        require(np.array_equal(im.xys[valid],np.column_stack([original['x'],original['y']])) and np.array_equal(im.point3D_ids[valid],original['id']),'Official decoder differs on actual byte-subset image data')
        k=lookup[name];K=camera_data['K'][k];c2w=camera_data['c2w'][k];w2c=np.linalg.inv(c2w);R=reader.qvec2rotmat(im.qvec)
        pose_error=max(float(np.max(np.abs(w2c[:3,:3]-R))),float(np.max(np.abs(w2c[:3,3]-im.tvec))));require(pose_error<1e-9,'Frozen camera pose does not match official raw pose')
        world=z[f'view_{j:02d}_world_xyz'];camera_xyz=world@R.T+im.tvec;hom=camera_xyz@K.T;projected=hom[:,:2]/hom[:,2:3];xy=z[f'view_{j:02d}_xy'];saved=z[f'view_{j:02d}_projected_xy']
        require(np.allclose(projected,saved,rtol=0,atol=2e-11),'Independent official-pose projection differs from001savedprojection')
        legacy=np.linalg.norm(projected-xy,axis=1);half=np.linalg.norm(projected-.5*xy,axis=1);design=np.column_stack([xy,np.ones(len(xy))]);aff=np.linalg.lstsq(design,projected,rcond=None)[0];aff_err=np.linalg.norm(design@aff-projected,axis=1)
        controls.append(dict(name=name,actual_observations=len(xy),pose_error=pose_error,original_legacy_median=float(np.median(legacy)),diagnostic_half_feature_median=float(np.median(half)),diagnostic_half_feature_p95=float(np.percentile(half,95)),affine_coefficients=aff.tolist(),affine_median=float(np.median(aff_err)),affine_p95=float(np.percentile(aff_err,95))))
        all_xy.append(xy);all_projected.append(projected)
        for i,pid in enumerate(z[f'view_{j:02d}_ids']):
            if int(pid) not in sample_points:continue
            p=sample_points[int(pid)];require(np.array_equal(p.xyz,world[i]),'Official original point XYZ differs from001 actual storedworldXYZ');point_controls.append(dict(name=name,point_id=int(pid),official_XYZ_exact=True,original_track_length=len(p.image_ids)))
    xy=np.concatenate(all_xy);p=np.concatenate(all_projected);a=np.column_stack([xy,np.ones(len(xy))]);coef=np.linalg.lstsq(a,p,rcond=None)[0];err=np.linalg.norm(a@coef-p,axis=1);half=np.linalg.norm(p-.5*xy,axis=1)
    require('torch' not in sys.modules,'This source audit must not import Torch')
    for path,sha in before.items():require(digest(path)==sha,'Read-only input changed')
    result=dict(status='confirmed_published_feature_xy_approximately2x_canonical_calibration_001_spatial_ROI_invalid',observed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=before,protected_originals_and001_unchanged=True,
        camera=dict(id=camera.id,model=camera.model,width=camera.width,height=camera.height,params=camera.params.tolist()),registered117_feature_count=feature_total,raw_feature_range=[minimum.tolist(),maximum.tolist()],outside_published1024x690_count=outside_metadata,outside_published1024x690_fraction=outside_metadata/feature_total,
        selected8_affine_feature_to_canonical=coef.tolist(),selected8_affine_error_percentile_50_95_99=np.percentile(err,[50,95,99]).tolist(),diagnostic_half_scaled_feature_error_percentile_50_95_99=np.percentile(half,[50,95,99]).tolist(),views=controls,
        actual_official_reader_original_byte_subset_images=8,actual_official_reader_point_subset_count=len(sample_points),official_point_controls=point_controls,
        evidence='Publishedcamera1024x690/K describes releasedrectified calibration; rawfeaturexy range isapproximately2048x1380. Withoriginalposes/worldXYZ andfrozen117K, canonicalprojection isapproximately0.5 of001scaledfeaturexy, notapixel-center offset. Exactrelease/resamplinghistory notprovided; diagnosticfactorisnotappliedtoanyinput.',
        conclusion='001rawfeaturexy crop/frame/tile coordinates invalid forcanonical256x172 ROI. 002shouldselect/project onlyworldXYZ+canonicalK/C2W andfront/framebounds;registeredpointIDs arecorrespondenceproxy, oldxy onlyresidual evidence. NoSfM/model/training/input/protocol correction.',
        limitations=['This doesnotidentifywhetherpublisherresized images/intrinsics withoutrevisingfeatures orperformedotherprocessing; sampleaffine2x evidence isdirect.','Residualafterhalfdiagnostic includesactualSfMerror; do notsilentlyrecover/modifyfeaturecoordinates ortrainingsplit.','Byte-subsetfiles retainoriginalrecords forindependentread only; nofake completedtraining/reconstruction claim.'],Torch_imported=False,CUDA_called=False)
    save(OUT/'audit.json',result);(OUT/'source-snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(status=result['status'],registered_features=feature_total,raw_range=result['raw_feature_range'],outside_metadata_fraction=result['outside_published1024x690_fraction'],affine_error=result['selected8_affine_error_percentile_50_95_99'],half_error=result['diagnostic_half_scaled_feature_error_percentile_50_95_99'],Torch_imported=False)))
if __name__=='__main__':main()
