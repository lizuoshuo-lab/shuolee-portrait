"""Frozen template, same44 texture queries: first surface ray intersection.
No fitting or Gaussian edits; missing hits are retained in the denominator.
"""
import json,pathlib,numpy as np
B=pathlib.Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
O=B/'human-surface-attribution-084';R=B/'person-static-human-surface-init-fresh-083'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def quant(x):
 x=np.asarray(x,float)
 return {'n':len(x),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'mean':float(x.mean())} if len(x) else {'n':0}
def project(X,row):
 E=np.array(row['w2c']);v=E[:3,:3]@X+E[:3,3];p=np.array(row['K'])@v
 return p[:2]/p[2]-.5
def hit(origin,direction,tri):
 a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;p=np.cross(direction,e2);det=(e1*p).sum(-1)
 inv=np.divide(1,det,out=np.zeros_like(det),where=abs(det)>1e-12);v=origin-a;u=(v*p).sum(-1)*inv;q=np.cross(v,e1)
 b=(direction*q).sum(-1)*inv;t=(e2*q).sum(-1)*inv
 good=(abs(det)>1e-12)&(u>=0)&(b>=0)&(u+b<=1)&(t>1e-8)
 if not good.any():return None,None
 k=np.argmin(np.where(good,t,np.inf));return origin+direction*t[k],int(k)
def main():
 assert not (O/'template-transfer.json').exists()
 save(O/'template-transfer-protocol.json',dict(purpose='Check whether the original neutral surface is a reliable target for a future constraint.',
  queries='All44 original078 lower/upper held texture rays; same reference observation in original records, none selected by new error.',
  methods=['first positive ray intersection with full initial FLAME surface','083 contribution-weighted center depth on query ray','original triangulated feature from078'],
  caveats=['Photometric feature transfer is not measured3Dgroundtruth.','Template excludes hair and glasses.','No fitted parameters or thresholds are searched.'],
  gates='Do not authorize hard binding solely from off-surface distance if template cross-view correspondence remains materially inaccurate.',updates=0))
 old=read(B/'static-depth-mixture-audit-078/depth-result.json')['records'];rows=read(R/'data/training-inputs.json')['records']
 h=np.load(R/'head-fit/rigid_neutral.npz');tri=h['vertices'][h['faces']];a=np.load(O/'083-attribution.npz')
 eligible=np.load(R/'clothing-init/rejected-face-candidate.npz')['eligible_triangles'];lut={int(k):j for j,k in enumerate(a['contributor_ids'])};records=[]
 for rec in old:
  if rec['kind'] not in ['lower_texture','upper_texture']:continue
  i=rec['raw_frame']-1;row=rows[i];E=np.array(row['w2c']);K=np.array(row['K']);p=np.array(rec['pixel_full'])+.5
  dc=np.array([(p[0]-K[0,2])/K[0,0],(p[1]-K[1,2])/K[1,1],1.]);origin=-E[:3,:3].T@E[:3,3];direction=E[:3,:3].T@dc
  X,j=hit(origin,direction,tri);target=rows[rec['reference_frame']];obs=np.array(rec['reference_pixel_full'])
  out=dict(index=rec['index'],raw_frame=rec['raw_frame'],kind=rec['kind'],template_hit=X is not None,
   source_pixel_full=rec['pixel_full'],reference_frame=rec['reference_frame']+1,reference_pixel_full=obs.tolist(),
   triangulated_error=float(rec['transfer_error_full_px']['triangulated']))
  if X is not None:
   pred=project(X,target);out.update(template_xyz=X.tolist(),triangle=j,in_initial_patch=bool(j in eligible),template_prediction=pred.tolist(),template_error=float(np.linalg.norm(pred-obs)))
  lo,hi=a['offsets'][rec['index']:rec['index']+2];ind=a['ray_indices'][lo:hi];w=a['ray_weights'][lo:hi].astype(float)
  if w.sum()>1e-8:
   xyz=a['xyz'][[lut[int(k)] for k in ind]];z=(xyz@E[:3,:3].T+E[:3,3])[:,2];depth=float(np.sum(z*w)/w.sum());Q=origin+direction*depth
   pred=project(Q,target);out.update(gaussian_prediction=pred.tolist(),gaussian_center_ray_error=float(np.linalg.norm(pred-obs)))
  records.append(out)
 groups={}
 for kind in ['lower_texture','upper_texture']:
  r=[v for v in records if v['kind']==kind];common=[v for v in r if v['template_hit'] and 'gaussian_center_ray_error' in v]
  groups[kind]=dict(queries=len(r),template_hits=sum(v['template_hit'] for v in r),common=len(common),
   common_patch_hits=sum(v['in_initial_patch'] for v in common),template_error_full_px=quant([v['template_error'] for v in common]),
   gaussian_center_ray_error_full_px=quant([v['gaussian_center_ray_error'] for v in common]),triangulated_error_full_px=quant([v['triangulated_error'] for v in common]),
   template_better_than_gaussian=sum(v['template_error']<v['gaussian_center_ray_error'] for v in common),
   template_over5px=sum(v['template_error']>5 for v in common),template_over10px=sum(v['template_error']>10 for v in common))
 out=dict(groups=groups,records=records,updates=0,quality_accepted=False)
 save(O/'template-transfer.json',out);print(json.dumps(groups))
if __name__=='__main__':main()
