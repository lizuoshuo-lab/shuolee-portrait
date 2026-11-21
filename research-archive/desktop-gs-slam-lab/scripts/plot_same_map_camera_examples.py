"""Display fixed first/middle/final RGB examples from already audited predictions.

No rendering, fitting, metric recalculation or data-dependent view selection.
Matplotlib standard RGB display range is used; original NPZ remains untouched.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tum-v4-estimated-camera-summary-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    audit_path=ROOT/'outputs/tum-online-v4-estimated-seen-map-evaluation-002/independent_estimated_seen_map_audit.json'
    summary_path=OUT/'summary.json'; protocol_path=ROOT/'manifests/tum-online-v4-estimated-seen-map-evaluation-protocol-002.json'
    assert sha(audit_path)=='c7c5e51ff64f1a85617f59890de19e7e6d8fc0e86f46870c14d7ac54091ba6c3'
    assert sha(summary_path)=='cd276e3ca50de935383088f8ed88c37b9c691d3d507f13248360609fbc2c2cd7'
    assert sha(protocol_path)=='124571c2835445a08e696e9a08f4f64495a7fb88aa35072f260404cefa8b2094'
    audit, summary, protocol=read(audit_path),read(summary_path),read(protocol_path)
    assert audit['status']=='passed_independent_estimated_seen_map_audit'
    destination=OUT/'public_diagnostics';destination.mkdir(exist_ok=False)
    watched={audit_path:sha(audit_path),summary_path:sha(summary_path),protocol_path:sha(protocol_path)};figures=[]
    for record in summary['maps']:
        scene=record['scene']; witness=protocol['datasets'][scene]; dataset=ROOT/witness['dataset']
        fig,axes=plt.subplots(3,3,figsize=(10.8,7.5),dpi=150)
        for column,index in enumerate((0,2,4)):
            view=record['views'][index];meta=witness['evaluation_sensor_metadata'][index]
            assert view['frame_id']==meta['frame_id']
            observed_path=dataset/meta['rgb_path'];key=observed_path.relative_to(ROOT).as_posix()
            assert sha(observed_path)==witness['input_file_sha256'][key];watched[observed_path]=sha(observed_path)
            with Image.open(observed_path) as image: observed=np.asarray(image.convert('RGB').resize((160,120),Image.Resampling.BILINEAR),dtype=np.float32)/np.float32(255.)
            values=[observed]
            for prediction in (view['same_map_GT_prediction'],view['prediction']):
                path=ROOT/prediction['path'];assert sha(path)==prediction['sha256'];watched[path]=sha(path)
                with np.load(path,allow_pickle=False) as saved: values.append(saved['rgb'].copy())
            for row,value in enumerate(values):
                assert value.shape==(120,160,3) and np.isfinite(value).all()
                axes[row,column].imshow(value,vmin=0.,vmax=1.,interpolation='nearest');axes[row,column].axis('off')
            axes[0,column].set_title(f"frame {view['frame_id']}",fontsize=11)
        for y,label in zip((.77,.49,.21),('Observed RGB','GT camera render','Saved camera render')):fig.text(.025,y,label,rotation=90,va='center',fontsize=10)
        fig.suptitle(record['run']+'\nSame frozen map: fixed first / middle / final examples',fontsize=11,y=.985)
        fig.subplots_adjust(left=.065,right=.985,top=.91,bottom=.035,hspace=.09,wspace=.035)
        target=destination/(record['run']+'.png');fig.savefig(target,dpi=150);plt.close(fig)
        assert target.stat().st_size<10*1024*1024
        figures.append({'run':record['run'],'path':target.relative_to(ROOT).as_posix(),'sha256':sha(target),'bytes':target.stat().st_size,'frame_ids':[record['views'][i]['frame_id'] for i in (0,2,4)]})
    assert all(sha(p)==h for p,h in watched.items())
    result={'status':'completed_fixed_audited_RGB_examples_only','source_sha256':sha(Path(__file__)),'audit_sha256':sha(audit_path),'summary_sha256':sha(summary_path),'figures':figures,'consumed_files_rehashed_unchanged':len(watched),'display':'StandardRGB[0,1], nearest display; raw NPZ/metrics unchanged; no enhancement or view selection','GPU_calls':0}
    with (OUT/'visual_examples.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'figures':len(figures),'GPU_calls':0}))


if __name__=='__main__':main()
