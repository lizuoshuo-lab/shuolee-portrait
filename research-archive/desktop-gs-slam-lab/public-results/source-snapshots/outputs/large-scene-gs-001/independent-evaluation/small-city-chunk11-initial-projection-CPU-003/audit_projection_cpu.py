"""Read-only initial geometry proxy for real failed11 scaffold row, all1470train.

NumPy CPU only, no rendering/gradients, RGB scores, model modification or CUDA.
float64 shadow and NumPyfloat32 are not bitwise CUDA/FMA/activation reproductions.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,struct,sys,time,traceback
import numpy as np
from plyfile import PlyData
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';OUT=Path(__file__).parent
TRAIN=BASE/'official-training/small-city-chunk11-i30000-r4-003'
COARSE=BASE/'official-training/small-city-coarse-i30000-r4-003/model/point_cloud/iteration_30000'
INPUT=BASE/'official-inputs/small-city-training-002';LOCK=BASE/'official-hierarchy/small-city-hierarchy-005'
LINEAGE=BASE/'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-lineage-CPU-002/lineage-activation.json'
SOURCE_ROW=1778538;SCAFFOLD_ROW=1377084;W,H=256,172
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def safe(x):
    x=float(x)
    return x if math.isfinite(x) else 'NaN' if math.isnan(x) else '+Inf' if x>0 else '-Inf'
def rot(q,dtype=np.float64):
    w,x,y,z=np.asarray(q,dtype=dtype);one=dtype(1);two=dtype(2)
    return np.array([[one-two*(y*y+z*z),two*(x*y-w*z),two*(x*z+w*y)],
                     [two*(x*y+w*z),one-two*(x*x+z*z),two*(y*z-w*x)],
                     [two*(x*z-w*y),two*(y*z+w*x),one-two*(x*x+y*y)]],dtype=dtype)
def camera_records(path):
    rows={}
    with Path(path).open('rb') as f:
        count,=struct.unpack('<Q',f.read(8))
        for _ in range(count):
            raw=f.read(64);values=struct.unpack('<i7di',raw);name=bytearray()
            while True:
                b=f.read(1);require(b,'Truncatedimage name')
                if b==b'\0':break
                name.extend(b)
            n,=struct.unpack('<Q',f.read(8));f.seek(24*n,1);name=name.decode('utf-8')
            require(name not in rows,'Duplicate registeredcamera name')
            rows[name]=dict(image_id=values[0],qvec=np.array(values[1:5]),tvec=np.array(values[5:8]),camera_id=values[8],raw64sha=hashlib.sha256(raw).hexdigest())
        require(f.tell()==Path(path).stat().st_size,'Unexpectedimage payload bytes')
    return rows
def scalar_stats(a,valid):
    values=np.asarray(a,dtype=np.float64)[valid];values=values[np.isfinite(values)]
    if not len(values):return None
    return dict(count=len(values),min=float(values.min()),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),max=float(values.max()))
def footprint_rect(p,radius):
    # Source getRect float→int truncates towardzero. ext supports both the
    # actualnonnullrects axis extents and explicitly diagnosticcircle radius.
    ext=np.broadcast_to(np.asarray(radius,np.float64),(2,))
    if not np.isfinite(p).all() or not np.isfinite(ext).all() or np.any(ext>2**30):return None
    grid=(math.ceil(W/16),math.ceil(H/16))
    lower=[min(grid[d],max(0,int((float(p[d])-ext[d])/16))) for d in (0,1)]
    upper=[min(grid[d],max(0,int((float(p[d])+ext[d]+15)/16))) for d in (0,1)]
    return dict(min_tile=lower,max_tile=upper,nonempty=(upper[0]-lower[0])*(upper[1]-lower[1])>0)
def project_cov(muc,Rc,fx,fy,tanx,tany,Sigma,dtype):
    t=np.asarray(muc,dtype=dtype);zero=dtype(0)
    rx=t[0]/t[2];ry=t[1]/t[2];lx=dtype(1.3)*dtype(tanx);ly=dtype(1.3)*dtype(tany)
    clampx=np.clip(rx,-lx,lx);clampy=np.clip(ry,-ly,ly);xp=clampx*t[2];yp=clampy*t[2]
    J=np.array([[dtype(fx)/t[2],zero,-(dtype(fx)*xp)/(t[2]*t[2])],[zero,dtype(fy)/t[2],-(dtype(fy)*yp)/(t[2]*t[2])]],dtype=dtype)
    B=J@np.asarray(Rc,dtype=dtype);C=B@np.asarray(Sigma,dtype=dtype)@B.T
    # glm cov[0][1] isrow1/column0. Native exportsoneoffdiagonal entry;
    # symmetrize thisentry forour scalar/eigen probe ratherthan average it.
    C[0,1]=C[1,0];a,b,c=C[0,0],C[1,0],C[1,1];det=a*c-b*b
    Cp=C.copy();Cp[0,0]+=dtype(.3);Cp[1,1]+=dtype(.3);ap,cp=Cp[0,0],Cp[1,1];detp=ap*cp-b*b
    ratio=det/detp if detp!=0 else dtype(np.nan)
    gain=np.sqrt(max(dtype(.000025),ratio)) if np.isfinite(ratio) else dtype(np.nan)
    mid=dtype(.5)*(ap+cp);lambda_hi=mid+np.sqrt(max(dtype(.1),mid*mid-detp));radius=np.ceil(dtype(3)*np.sqrt(lambda_hi))
    return J,C,Cp,float(det),float(detp),float(gain),float(radius),bool(rx!=clampx or ry!=clampy)
def main():
    require(not (OUT/'outcome.json').exists(),'Existingdiagnostic refuseoverwrite')
    start=time.perf_counter();write(OUT/'outcome.json',dict(status='running_readonly_CPU_initial_projection',CUDA_called=False,gradients_executed=False))
    try:
        protected={}
        def bound(path,expected=None):
            path=Path(path);sha=digest(path)
            require(expected is None or sha==expected,'Actual source changed: '+str(path));protected[str(path)]=sha;return sha
        bound(LINEAGE,'68bc09e49af01f0a2afe46c3ff0a41e25365b848dd25142da189aafb6dd6d8df');lineage=read(LINEAGE)
        bound(COARSE/'point_cloud.ply','abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31');sky=int((COARSE/'pc_info.txt').read_text());bound(COARSE/'pc_info.txt');require(sky==100000,'Wrongsky prefix')
        bound(INPUT/'manifest.json','2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5');manifest=read(INPUT/'manifest.json')
        bound(TRAIN/'configuration.json');config=read(TRAIN/'configuration.json');bound(TRAIN/'outcome.json');require(read(TRAIN/'outcome.json')['status']=='failed','Do not relabel actualfailed11')
        require(config['chunk']=='1_1' and len(config['training_names'])==1470 and not set(config['training_names'])&set(manifest['test_names']),'Actualtrain1470 names/split invalid')
        for row in config['official_source_files']:bound(row['path'],row['sha256'])
        native=ROOT/'repos/hierarchy-rasterizer-large-gs'
        for name in ('cuda_rasterizer/forward.cu','cuda_rasterizer/auxiliary.h','cuda_rasterizer/config.h','cuda_rasterizer/rasterizer_impl.cu','rasterize_points.cu','diff_gaussian_rasterization/__init__.py'):bound(native/name)
        vertices=PlyData.read(COARSE/'point_cloud.ply',mmap='r')['vertex'].data;v=vertices[SOURCE_ROW]
        xyz=np.array([v[n] for n in ('x','y','z')],np.float32);logs=np.array([v[f'scale_{i}'] for i in range(3)],np.float32);rawq=np.array([v[f'rot_{i}'] for i in range(4)],np.float32)
        require(np.isfinite(np.r_[xyz,logs,rawq,float(v['opacity'])]).all(),'Initialfields nonfinite')
        expected=lineage['failed_rows'][0];require(expected['derived_original_coarse_PLY_row_index_zero_based']==SOURCE_ROW and expected['final_PLY_row_index_zero_based']==SCAFFOLD_ROW,'Wrongfailedrow family')
        for name in ('x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity'):require(float(v[name])==expected['original_coarse_raw_values'][name],'Initialraw mismatch')
        centerpath=INPUT/'chunks/1_1/center.txt';extentpath=INPUT/'chunks/1_1/extent.txt';bound(centerpath);bound(extentpath);center=np.loadtxt(centerpath,dtype=np.float32);extent=np.loadtxt(extentpath,dtype=np.float32)
        xy=np.stack([vertices['x'],vertices['y']],axis=1);distance=np.max(np.abs(xy-center[:2]),axis=1);selected=(distance>np.float32(.5)*extent[0])&(distance<np.float32(1.5)*extent[0]);selected[:sky]=True;indices=np.flatnonzero(selected)
        require(len(indices)==1835543 and indices[SCAFFOLD_ROW]==SOURCE_ROW and SOURCE_ROW>=sky,'Initialsource selection/prefix/nonSky mapping failed')
        scale32=np.exp(logs);norm32=np.sqrt(np.sum(rawq*rawq,dtype=np.float32));q32=rawq/max(norm32,np.float32(1e-12));Rq32=rot(q32,np.float32);B32=Rq32*scale32[None,:];Sigma32=B32@B32.T
        # Same nominal activatedf32endpoint, evaluatedinfloat64forconditioning.
        B64=Rq32.astype(np.float64)*scale32.astype(np.float64)[None,:];Sigma64=B64@B64.T
        alpha32=float(np.float32(1)/(np.float32(1)+np.exp(-np.float32(v['opacity']))))
        initial=dict(original_coarse_row=SOURCE_ROW,selected_scaffold_row=SCAFFOLD_ROW,classification='nonSky_selected_coarse_context_gaussian_not_newSfM_densified_row',sky_prefix_count=sky,
            initial_mu_f32=xyz.tolist(),initial_log_sigma_f32=logs.tolist(),initial_sigma_numpy_f32=scale32.tolist(),initial_raw_wxyz_f32=rawq.tolist(),initial_quaternion_norm_numpy_f32=float(norm32),normalized_wxyz_numpy_f32=q32.tolist(),raw_opacity_logit=float(v['opacity']),sigmoid_opacity_numpy_f32=alpha32,
            world_covariance_shadow_from_f32_endpoint=Sigma64.tolist(),world_covariance_eigenvalues=np.linalg.eigvalsh(Sigma64).tolist(),world_covariance_condition=float(np.linalg.cond(Sigma64)),scale_axis_ratio=float(scale32.max()/scale32.min()),
            selected_scaffold_count=len(indices),annulus_distance_xy_worldunits=float(distance[SOURCE_ROW]),center=center.tolist(),extent=extent.tolist(),activation_scope='exp logs andnormalize endpoint once per source; native q=rot nosecondnormalization. NumPyactivations notbitwiseGPU guarantee',length_units='PublishedalignedSfM/arbitraryworldunits; notGT/notmeters')
        archive=LOCK/'creator-camera-archives/1_1';am=read(archive/'manifest.json');bound(archive/'manifest.json')
        for row in am['files']:bound(row['path'],row['sha256'])
        records=camera_records(archive/'sparse/0/images.bin');original=camera_records(INPUT/'chunks/1_1/sparse/0/images.bin');bound(INPUT/'chunks/1_1/sparse/0/images.bin',am['source_images_sha256']);bound(INPUT/'chunks/1_1/sparse/0/cameras.bin',am['source_cameras_sha256'])
        names=config['training_names'];require(set(records)==set(names) and am['train_only_pose_count']==1470 and all(records[n]['raw64sha']==original[n]['raw64sha'] for n in names),'Actualtrain-onlypose archive mismatch')
        with (archive/'sparse/0/cameras.bin').open('rb') as f:
            require(struct.unpack('<Q',f.read(8))[0]==1,'Expectedonesharedcamera');cid,modelid,ow,oh,ofx,ofy,ocx,ocy=struct.unpack('<iiQQ4d',f.read(56))
        require(modelid==1 and (ow,oh)==(1024,690) and (round(ow/4),round(oh/4))==(W,H),'ActualPINHOLE/r4 sizes mismatch')
        tanx=ow/(2*ofx);tany=oh/(2*ofy);fx64=W/(2*tanx);fy64=H/(2*tany);fx32=np.float32(W)/(np.float32(2)*np.float32(tanx));fy32=np.float32(H)/(np.float32(2)*np.float32(tany));K=np.array([[fx64,0,ocx*W/ow],[0,fy64,ocy*H/oh],[0,0,1.]])
        proto=Path(manifest['protocol_source'])/'global-cameras.npz';bound(proto)
        with np.load(proto,allow_pickle=False) as z:global_names=z['names'].tolist();globalK=z['K'].copy();globalC2W=z['c2w'].copy()
        lookup={n:j for j,n in enumerate(global_names)};rows=[];saved={k:[] for k in ('W2C_f32','K','camera_mean_f32','Jacobian_clamped_shadow','projected_cov_shadow','projected_cov_AA_shadow','projected_cov_numpy_f32','projected_cov_AA_numpy_f32','native_axis_extents_px')}
        for name in names:
            rec=records[name];require(rec['camera_id']==cid,'WrongcameraID');Rt=np.eye(4);Rt[:3,:3]=rot(rec['qvec']);Rt[:3,3]=rec['tvec'];C2W=np.linalg.inv(Rt);W2C=np.linalg.inv(C2W).astype(np.float32)
            j=lookup[name];referenceK=globalK[j].copy();referenceK[0]*=W/ow;referenceK[1]*=H/oh
            require(np.max(np.abs(K-referenceK))<1e-10,'ActualtrainK not same publishedglobalintrinsics')
            pose_delta=float(np.max(np.abs(C2W-globalC2W[j])))
            # Actualpublishedchunkposes are bytepreserved, but not bitwise the
            # fullscene doublepose archive. Quantify, never replace actualinput.
            Rc=W2C[:3,:3];m32=Rc@xyz+W2C[:3,3];m64=Rc.astype(np.float64)@xyz.astype(np.float64)+W2C[:3,3].astype(np.float64)
            front=bool(m32[2]>np.float32(.2));center_projection=np.array([fx64*m64[0]/m64[2]+K[0,2]-.5,fy64*m64[1]/m64[2]+K[1,2]-.5])
            # Epsilon/projection f32 shadow: actualP defaults .01/100 retained;
            # no x/y clamp is appliedto trueGaussian center.
            P=np.zeros((4,4),np.float32);P[0,0]=np.float32(1/tanx);P[1,1]=np.float32(1/tany);P[0,2]=np.float32(2*ocx/ow-1);P[1,2]=np.float32(2*ocy/oh-1);P[2,2]=np.float32(100/99.99);P[2,3]=np.float32(-1/99.99);P[3,2]=1
            hom=(P@W2C)@np.r_[xyz,np.float32(1)];ndc=hom[:3]/(hom[3]+np.float32(1e-7));p32=(ndc[:2]+np.float32(1))*np.array([W,H],np.float32)*np.float32(.5)-np.float32(.5)
            frame=bool(front and 0<=p32[0]<W and 0<=p32[1]<H);fov=bool(front and np.all(np.abs(ndc[:2])<=1));extended=bool(front and np.all(np.abs(ndc[:2])<=1.3))
            A,C,Cp,d,dp,gain,radius,clamped=project_cov(m64,Rc,fx32,fy32,np.float32(tanx),np.float32(tany),Sigma64,np.float64)
            A32,C32,Cp32,d32,dp32,gain32,radius32,clamped32=project_cov(m32,Rc,fx32,fy32,np.float32(tanx),np.float32(tany),Sigma32,np.float32)
            eigen=np.linalg.eigvalsh(C);eigenp=np.linalg.eigvalsh(Cp);eigen32=np.linalg.eigvalsh(C32.astype(np.float64));eigenp32=np.linalg.eigvalsh(Cp32.astype(np.float64));cameraCov=Rc.astype(np.float64)@Sigma64@Rc.astype(np.float64).T;zstd=math.sqrt(cameraCov[2,2]);circle=footprint_rect(p32,radius32)
            axis=np.ceil(np.float32(3)*np.sqrt(np.diag(Cp32))).astype(np.int32);rect=footprint_rect(p32,axis)
            row=dict(name=name,train_ordinal=len(rows),camera_image_id=rec['image_id'],camera_mean_numpy_f32=m32.tolist(),camera_mean_shadow=m64.tolist(),depth_worldunits=float(m32[2]),front_z_gt_0_2=front,
                train_pose_vs_global_double_C2W_max_abs_delta=pose_delta,mean_projected_pixel_shadow=center_projection.tolist(),mean_projected_pixel_numpy_f32=p32.tolist(),mean_inside_image_frame=frame,mean_inside_NDC_frustum=fov,mean_inside_extendedNDC1_3=extended,
                xy_ratio_clamp_applied_numpy_f32=clamped32,Jacobian_spectral_norm=float(np.linalg.norm(A,ord=2)),
                projected_covariance_eigenvalues_shadow=eigen.tolist(),projected_covariance_AA_eigenvalues_shadow=eigenp.tolist(),projected_covariance_eigenvalues_from_numpy_f32=eigen32.tolist(),projected_covariance_AA_eigenvalues_from_numpy_f32=eigenp32.tolist(),
                raw_covariance_condition_shadow=safe(eigen[-1]/eigen[0]),AA_covariance_condition_shadow=safe(eigenp[-1]/eigenp[0]),det_cov_shadow=safe(d),det_cov_plus_0_3I_shadow=safe(dp),det_cov_numpy_f32=safe(d32),det_cov_plus_0_3I_numpy_f32=safe(dp32),
                AA_opacity_gain_shadow=safe(gain),AA_opacity_gain_numpy_f32=safe(gain32),peak_activated_alpha_AA_numpy_f32=safe(alpha32*gain32),source_radius_numpy_f32_px=safe(radius32),conservative_circle_3sigma_tile_rectangle_proxy=circle,
                native_axis_extents_numpy_f32_px=axis.tolist(),native_axis_3sigma_tile_rectangle_proxy=rect,conservative_circle_candidate_proxy=bool(front and dp32!=0 and circle is not None and circle['nonempty']),
                native_candidate_from_initial_geometry_proxy=bool(front and dp32!=0 and rect is not None and rect['nonempty']),camera_depth_sigma_shadow=zstd,depth_minus_3sigma_worldunits=float(m64[2]-3*zstd),nearplane_in_3sigma_depth_span_proxy=bool(m64[2]-3*zstd<=.2<=m64[2]+3*zstd),
                actual_runtime_visibility_or_gradient=False,point_fields_are_initial_not_runtime=True)
            rows.append(row)
            for key,value in [('W2C_f32',W2C),('K',K),('camera_mean_f32',m32),('Jacobian_clamped_shadow',A),('projected_cov_shadow',C),('projected_cov_AA_shadow',Cp),('projected_cov_numpy_f32',C32),('projected_cov_AA_numpy_f32',Cp32),('native_axis_extents_px',axis)]:saved[key].append(value)
        arrays=OUT/'all1470-initial-projection.npz';np.savez(arrays,names=np.array(names),**{k:np.stack(v) for k,v in saved.items()})
        write(OUT/'per-camera.json',rows);front=np.array([r['front_z_gt_0_2'] for r in rows]);candidate=np.array([r['native_candidate_from_initial_geometry_proxy'] for r in rows]);frame=np.array([r['mean_inside_image_frame'] for r in rows]);eligible=[r for r in rows if r['front_z_gt_0_2']]
        rankings={}
        for label,key,reverse in [('nearest_front_mean_depth','depth_worldunits',False),('largest_Jacobian_norm','Jacobian_spectral_norm',True),('largest_raw_covariance_condition','raw_covariance_condition_shadow',True),('smallest_AA_determinant','det_cov_plus_0_3I_shadow',False),('largest_source_radius','source_radius_numpy_f32_px',True)]:
            available=[r for r in eligible if isinstance(r[key],(int,float))];rankings[label]=sorted(available,key=lambda r:(-r[key] if reverse else r[key],r['name']))[:10]
            included=[r for r in available if r['native_candidate_from_initial_geometry_proxy']];rankings[label+'_actual_axis_candidate_only']=sorted(included,key=lambda r:(-r[key] if reverse else r[key],r['name']))[:10]
        summaries={key:dict(all_cameras=scalar_stats([r[key] for r in rows],np.ones(len(rows),bool)),front_only=scalar_stats([r[key] for r in rows],front),initial_native_candidate_proxy_only=scalar_stats([r[key] for r in rows],candidate)) for key in ('depth_worldunits','Jacobian_spectral_norm','raw_covariance_condition_shadow','AA_covariance_condition_shadow','det_cov_numpy_f32','det_cov_plus_0_3I_numpy_f32','source_radius_numpy_f32_px','camera_depth_sigma_shadow','train_pose_vs_global_double_C2W_max_abs_delta')}
        require(all(digest(path)==sha for path,sha in protected.items()),'Read-onlysource data changed duringCPUaudit');require('torch' not in sys.modules,'UnexpectedTorch import')
        report=dict(status='completed_actual1470train_initial_gaussian_projection_CPU_proxy_not_cause',initial=initial,
            cameras=dict(count=1470,width=W,height=H,K_r4=K.tolist(),native_focal_numpy_f32=[float(fx32),float(fy32)],original_K=[ofx,ofy,ocx,ocy],W2C_from_original_colmap_and_official_getWorld2View2_roundtrip=True,train_pose_bytes_exact_original=True,heldout_used=0),
            counts=dict(all_train=1470,front_z_gt_0_2=int(front.sum()),behind_or_nearclipped=int((~front).sum()),mean_inside_image_frame=int(frame.sum()),initial_native_axis_candidate_proxy=int(candidate.sum()),conservative_circle_candidate_proxy=sum(r['conservative_circle_candidate_proxy'] for r in rows),clamped_front=sum(r['front_z_gt_0_2'] and r['xy_ratio_clamp_applied_numpy_f32'] for r in rows),front_numpy_f32_det_AA_nonpositive=sum(r['front_z_gt_0_2'] and isinstance(r['det_cov_plus_0_3I_numpy_f32'],(int,float)) and r['det_cov_plus_0_3I_numpy_f32']<=0 for r in rows),front_depth3sigma_span_crosses_nearplane=sum(r['front_z_gt_0_2'] and r['nearplane_in_3sigma_depth_span_proxy'] for r in rows)),
            summaries=summaries,geometry_only_ranked_candidates=rankings,all1470_numeric_archive_path=str(arrays),all1470_numeric_archive_sha256=digest(arrays),all1470_camera_report_sha256=digest(OUT/'per-camera.json'),source_records=protected,
            covariance_contract='WorldSigma=R(normalizedwxyz)diag(exp(logsigma)^2)R.T; C=J Rc Sigma Rc.T J.T. SourceJ x/z,y/z clamped±1.3tanFoVonlyforcovariance; true mean projection unclamped; add0.3I pixel²,det-ratioAA sqrt(max2.5e-5,detC/detCplus).',
            visibility_limit='Actualrasterize_points.cu allocatesrects[P,2], soactivebranchaxisceil3sqrt(diagCplus) boundingtiles. Circularradius branch separatelydiagnostic. Nativeonly z>.2/worldAABB/nonempty3sigma tiles; NDCcenterfrustum exclusioncommented. Meanframe flags notvisibility; default+-FLT_MAXworldAABB doesnotfilterinitialmu.',
            precision_limit='f64shadow conditioning and NumPyf32 operation probes, no CUDA FMA/GLM/exp/normalize/radius exactness guarantee. GPU/backward/intermediateupdates notexecuted.',
            interpretation='FixedinitialfiniteGaussian/all1470train poses geometriccontext only; cannotprovefirstNaNiteration, optimizerstate, gradients, actualrastervisibility orcomputationalcause. Worldunitsnotmeters, publishedSfMnotGT.',
            elapsed_CPU_seconds=time.perf_counter()-start,CUDA_called=False,Torch_imported=False,gradients_executed=False,source_models_modified=False,failed11_repaired_or_reclassified_complete=False)
        write(OUT/'report.json',report);write(OUT/'outcome.json',dict(status=report['status'],exit_code=0,report_sha256=digest(OUT/'report.json'),source_sha256=digest(__file__),elapsed_CPU_seconds=report['elapsed_CPU_seconds'],CUDA_called=False,Torch_imported=False))
        print(json.dumps(dict(status=report['status'],counts=report['counts'],initial_sigma=scale32.tolist(),initial_world_condition=initial['world_covariance_condition'],elapsed=report['elapsed_CPU_seconds'])),flush=True)
    except Exception as error:
        write(OUT/'outcome.json',dict(status='failed_preserved_CPU_initial_projection',error=str(error),traceback=traceback.format_exc(),CUDA_called=False,Torch_imported='torch' in sys.modules));raise
if __name__=='__main__':main()
