"""Check all175 exported-camera projections against each actual new PLY."""
from pathlib import Path
import json,numpy as np,hashlib,argparse
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');P=B/'current/static-ncc-paired-fresh-093';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def xyz(path):
    props=[];count=None
    with path.open('rb') as f:
        assert f.readline().strip()==b'ply';assert f.readline().strip()==b'format binary_little_endian 1.0'
        for line in iter(f.readline,b''):
            line=line.decode('ascii').strip()
            if line.startswith('element vertex'):count=int(line.split()[-1])
            elif line.startswith('property'):
                a=line.split();assert a[1]=='float';props.append((a[2],'<f4'))
            elif line=='end_header':break
        assert count and all(k in dict(props) for k in ['x','y','z']);d=np.fromfile(f,dtype=np.dtype(props),count=count)
    assert len(d)==count
    return np.stack([d[k] for k in ['x','y','z']],-1).astype(float)
def main(arm):
    suffix='control' if arm=='control' else 'off';meta_path=V/f'model-whole-body-static-ncc-{suffix}.json';meta=read(meta_path)
    rows=read(P/arm/'data/training-inputs.json')['records'];assert len(rows)==len(meta['cameras'])==175
    native=P/arm/'gaussian-run/point_cloud.ply';viewer=V/meta['model'];a,b=xyz(native),xyz(viewer);assert a.shape==b.shape
    idx=np.linspace(0,len(a)-1,128).astype(int);a,b=a[idx],b[idx];checks=[]
    for i,(row,cam) in enumerate(zip(rows,meta['cameras'])):
        assert cam['raw_source_frame']==i+1 and cam['width']==row['wh'][0] and cam['height']==row['wh'][1]
        E=np.array(row['w2c']);K=np.array(row['K']);x=a@E[:3,:3].T+E[:3,3]
        y=(b-np.array(cam['position']))@np.array(cam['rotation'])
        good=(x[:,2]>.01)&(y[:,2]>.01);assert good.any()
        uv=(x@K.T);uv=uv[:,:2]/uv[:,2:]-.5
        vv=y[:,:2]/y[:,2:]*[cam['fx'],cam['fy']]+[cam['cx']-.5,cam['cy']-.5]
        delta=np.linalg.norm(uv[good]-vv[good],axis=1);checks.append(dict(raw_frame=i+1,points=int(good.sum()),maximum_pixel_difference=float(delta.max())))
    mx=max(q['maximum_pixel_difference'] for q in checks);assert mx<.01,mx
    out=dict(arm=arm,cameras=175,point_probes_per_camera=128,max_pixel_difference=mx,projection_contract_passed=True,
        native_sha256=sha(native),viewer_sha256=sha(viewer),metadata_sha256=sha(meta_path),checks=checks,
        limitation='Verifies exported coordinate/camera equivalence, not physical camera calibration or person reconstruction accuracy.',quality_accepted=False)
    (P/arm/'viewer-camera-audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['control','no_ncc']);main(p.parse_args().arm)
