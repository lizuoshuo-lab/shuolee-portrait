from pathlib import Path
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-gradient-087';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(O/'summary.json');branches={}
for b in ['parent','fine_photo']:
    r=read(O/f'{b}-records.json');assert len(r['records'])==105;g=np.load(O/f'{b}-aggregate-gradients.npz');check=0.;allfamilies={}
    for k,m in r['aggregate'].items():
        stem=k.replace('.','_');a,c,d=[g[stem+'_'+q].ravel() for q in ['roi','rest','regularizer']];vals=[np.linalg.norm(x) for x in [a,c,d]]
        for name,v in zip(['roi_norm','rest_norm','regularizer_norm'],vals):check=max(check,abs(v-m[name]));assert abs(v-m[name])<1e-12
        rows=[rr['families'][k] for rr in r['records']];valid=[x for x in rows if x['roi_total_cosine'] is not None]
        allfamilies[k]=dict(aggregate=m,per_image_total_opposes_roi=sum(x['roi_total_cosine']<0 for x in valid),valid_contexts=len(valid),median_roi_rest_norm_ratio=float(np.median([x['roi_rest_norm_ratio'] for x in rows])),aggregate_coherence=m['roi_norm']/max(np.mean([x['roi_norm'] for x in rows]),1e-30))
    branches[b]=dict(families=allfamilies,mean_roi_area_fraction=float(np.mean([x['roi_area_fraction'] for x in r['records']])),aggregate_recompute_max_error=check,gradient_file_sha256=sha(O/f'{b}-aggregate-gradients.npz'))
out=dict(status='210context_gradients_independently_aggregated_no_training',branches=branches,decision='ROIphotogradients dominate selectedGaussian parameter gradients. Remainderpixels do not explainstall; increasingROIweight lacks currentevidence. Regularizeraffectsnodegradients but aggregatephotodirection remainsaligned. Next frozen-motion single-time versus multi-time localgeometry fit distinguishes temporalsharing from current surfacefit limitations; treatasdiagnostic,notcompletecontinuousscene.',limits=['160pxcontextincludeshandheldobject andshirt.','Fixedgray, notallrandombackgroundtrainingdraws.','Euclideanperfamilygradients,notAdamupdate; noabsoluteunitmixednorm.','No claim thatagradientcorrelationprovesblurcause.'],goal_complete=False);save(O/'final-summary.json',out);print(json.dumps({b:{'roi_area':r['mean_roi_area_fraction'],'means_coherence':r['families']['gaussian.means']['aggregate_coherence'],'means_context_conflicts':r['families']['gaussian.means']['per_image_total_opposes_roi']} for b,r in branches.items()}))
