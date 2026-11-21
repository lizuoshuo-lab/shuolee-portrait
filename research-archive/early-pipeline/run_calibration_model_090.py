"""Bounded calibration-model comparison with freshly triangulated structure."""
import os,sys,json,hashlib,sqlite3,shutil,subprocess,time
from pathlib import Path
from datetime import datetime
import numpy as np

L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee';S=B/'current/full-scene-localization-074'
O=B/'current/joint-calibration-model-audit-090'
T=Path('C:/Users/Administrator/colmap-x64-windows-cuda');EXE=T/'bin/colmap.exe'
sys.path.insert(0,str(L/'repos/hierarchical-3d-gaussians/preprocess'))
import read_write_model as rw
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.now().astimezone().isoformat(timespec='seconds')
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');q.replace(p)

def main():
    assert not O.exists();O.mkdir();(O/'logs').mkdir()
    c=read(S/'cameras.json');assert c['registered']==175 and not c['interpolation_used']
    sourcehash=sha(S/'cameras.json');dbhash=sha(S/'database.db')
    protocol=dict(created=stamp(),reason='089 per-frame focal correction with fixed3D failed. Fixed3D cannot exclude shared calibration/structure coupling. Test one richer global camera model with structure re-created in each arm.',
      arms=['SIMPLE_RADIAL','OPENCV'],changed='Global calibration model only: shared f,cx,cy,k1 versus shared fx,fy,cx,cy,k1,k2,p1,p2. Changes several intrinsic terms as one model class; cannot attribute an improvement to one term.',
      reused='Original full images, raw SIFT keypoints and verified match graph, and registered074camera poses/calibration as permitted calibrated input. No old3Dpoint, body fit or Gaussian.',
      fresh='For eacharm write empty3Dmodel, clear every observation pointID, re-triangulate from same image correspondences and fixed input poses; then joint BA releases all poses,3Dpoints andsharedintrinsics.',
      limits='Not fresh independent camera registration: both arms start from the same175registered input poses. Matchgraph is frozen from074. This is an intrinsicmodel diagnostic, not a newbodytraining or evidence of exactphysicalcalibration.',
      controls='Same images, matchgraph, initialprojection, randomseed0, CPUtriangulation, intrinsics fixed during triangulation, same max100globalBA iterations. No personmask/semanticregionused in fitting.',
      initial_projection='OpenCV initialized [f,f,cx,cy,k1,0,0,0], exactly equivalent to SIMPLE_RADIAL.',
      validation='Use frozen080 image-onlytracks and duplicateclusters. Freshly triangulate everyvalidationpoint for eacharm usingouterthirdobservations only; evaluate identical withheldmiddlethird queries. Report lowerface,upperface,body,background atsameframes, both invalid/negative depth andmissingdata.',
      gates='Compare richerarm to freshlyrecomputed SIMPLE_RADIAL: lowerface median<=0.8baseline andp90<=baseline; backgroundp90<=baseline+0.2rawpx;torsop90<=baseline+0.5rawpx; all175finiteposes retained; no asymmetricqueryexclusion. Sourcephoto reviewrequired. Passing nevermeansGaussianqualitypassed.',
      stop='At most two predefinedarms. Perarm900seconds triangulation and600secondsBA. Noadaptiveparameteror cameramodelscan. Do notoverwritecanonicalcameras.',
      source_camera_sha256=sourcehash,raw_database_sha256=dbhash,source_sha256=sha(__file__),camera_updates=0,gaussian_updates=0,quality_accepted=False)
    save(O/'protocol.json',protocol);shutil.copy2(__file__,O/Path(__file__).name)
    env=os.environ.copy();env['PATH']=str(T/'bin')+os.pathsep+env.get('PATH','');env['QT_PLUGIN_PATH']=str(T/'plugins');env['OMP_NUM_THREADS']='8'
    state=dict(created=stamp(),running=True,pid=os.getpid(),commands=[],completed_arms=[],quality_accepted=False)
    save(O/'state.json',state)
    start=time.monotonic()
    def run(label,cmd,limit):
        state.update(stage=label,updated=stamp());state['commands'].append(dict(label=label,cmd=list(map(str,cmd))));save(O/'state.json',state)
        print(json.dumps(dict(stage=label,started=stamp())),flush=True)
        t=time.monotonic()
        with (O/'logs'/f'{label}.log').open('w',encoding='utf8') as log:
            proc=subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;save(O/'state.json',state)
            try:rc=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise TimeoutError(label)
        state.pop('child_pid',None);state['commands'][-1].update(returncode=rc,seconds=time.monotonic()-t);save(O/'state.json',state)
        if rc:raise RuntimeError(f'{label}: exit{rc}')
    try:
        ca,im,_=rw.read_model(str(S/'refined_sparse'),ext='.bin');sourcecam=next(iter(ca.values()))
        assert len(im)==175 and len(ca)==1
        for name,modelid in [('SIMPLE_RADIAL',2),('OPENCV',4)]:
            root=O/name.lower();root.mkdir()
            for d in ['seed','triangulated','refined']:(root/d).mkdir()
            params=np.array(c['intrinsics'])
            if name=='OPENCV':params=np.array([params[0],params[0],params[1],params[2],params[3],0.,0.,0.])
            cc={sourcecam.id:sourcecam._replace(model=name,params=params)}
            ii={k:v._replace(point3D_ids=np.full_like(v.point3D_ids,-1)) for k,v in im.items()}
            rw.write_model(cc,ii,{},str(root/'seed'),ext='.bin')
            db=root/'database.db';shutil.copy2(S/'database.db',db)
            with sqlite3.connect(db) as conn:
                conn.execute('UPDATE cameras SET model=?, params=? WHERE camera_id=?',(modelid,np.asarray(params,dtype=np.float64).tobytes(),sourcecam.id));conn.commit()
                assert conn.execute('SELECT COUNT(*) FROM cameras').fetchone()[0]==1
            run(name.lower()+'_triangulate',[EXE,'point_triangulator','--database_path',db,'--image_path',S/'images','--input_path',root/'seed','--output_path',root/'triangulated','--clear_points','1','--refine_intrinsics','0','--Mapper.num_threads','8','--Mapper.ba_use_gpu','0','--random_seed','0'],900)
            tc,ti,tp=rw.read_model(str(root/'triangulated'),ext='.bin');assert len(ti)==175 and next(iter(tc.values())).model==name
            assert np.max(abs(next(iter(tc.values())).params-params))<1e-8
            run(name.lower()+'_joint_ba',[EXE,'bundle_adjuster','--input_path',root/'triangulated','--output_path',root/'refined','--BundleAdjustment.refine_focal_length','1','--BundleAdjustment.refine_principal_point','1','--BundleAdjustment.refine_extra_params','1','--BundleAdjustment.refine_extrinsics','1','--BundleAdjustment.max_num_iterations','100','--BundleAdjustment.use_gpu','0','--random_seed','0'],600)
            co,io,po=rw.read_model(str(root/'refined'),ext='.bin');cam=next(iter(co.values()));assert len(io)==175 and cam.model==name and np.isfinite(cam.params).all()
            indexed={r['name']:r for r in c['records']};rows=[]
            for v in sorted(io.values(),key=lambda v:v.name):
                E=np.eye(4);E[:3,:3]=rw.qvec2rotmat(v.qvec);E[:3,3]=v.tvec;assert np.isfinite(E).all() and abs(np.linalg.det(E[:3,:3])-1)<1e-8
                row=indexed[v.name].copy();row.update(w2c=E.tolist(),observation_points=int((v.point3D_ids>=0).sum()));rows.append(row)
            save(root/'cameras.json',dict(model=name,width=cam.width,height=cam.height,intrinsics=cam.params.tolist(),records=rows,registered=175,interpolation_used=False,accepted_for_training=False,source='090samecalibratedposeinitialization,fresh3Dtriangulation,jointglobalBA'))
            state['completed_arms'].append(dict(arm=name,points=len(po),intrinsics=cam.params.tolist(),camera_path=str(root/'cameras.json'),native_training_reprojection_mean=float(np.mean([v.error for v in po.values()]))));save(O/'state.json',state)
            print(json.dumps(dict(stage='arm_complete',**state['completed_arms'][-1])),flush=True)
        assert sha(S/'cameras.json')==sourcehash and sha(S/'database.db')==dbhash
        state['status']='two_arms_complete_pending_independent_query_evaluation'
    except Exception as e:state.update(status='not_accepted_error',error=repr(e));raise
    finally:state.update(running=False,updated=stamp(),elapsed_seconds=time.monotonic()-start);save(O/'state.json',state)

if __name__=='__main__':main()
