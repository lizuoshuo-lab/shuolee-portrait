"""Complete fixed primary/path/24frame outputs, leaving scientific acceptance to review."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

S=Path(__file__).parent;ROOT=S.parents[1]/'outputs/mini-splatting2-mip-depth-001'

def main():
    path=ROOT/'native-late-publication-queue.json';assert not path.exists()
    state=dict(status='waiting_for_training',started=time.time(),completed=[],accepted=False)
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(path)
    def execute(script,args=[]):
        state.update(status='running',current=script);save()
        with (ROOT/('late-'+Path(script).stem+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(S/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(script);save()
    save()
    try:
        protocol=json.loads((ROOT/'native-resolution-followup-protocol-001.json').read_text())
        protocol.update(created=time.time(),status='frozen_before_late_results',
            models={'low':'native-resolution-cache-full-18000-s0','full':'native-late-full-18000-s0'},
            meaning_of_low_full_keys='Original schedule versus late densification; both trained at1514x2694')
        for name,digest in protocol['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((S/'evaluate_resolution_additional_frames.py').read_bytes()).hexdigest()==protocol['script_sha256']
        (ROOT/'native-late-followup-protocol-001.json').write_text(json.dumps(protocol,indent=2))
        sources=json.loads((ROOT/'native-late-publication-sources-001.json').read_text())
        for name,digest in sources['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==digest
        while True:
            queue=json.loads((ROOT/'native-late-queue.json').read_text())
            if queue['status']=='failed':raise RuntimeError('Training failed: '+queue.get('error','inspect log'))
            if queue['status']=='complete_pending_review':break
            time.sleep(20)
        execute('render_native_late_paths.py')
        execute('build_late_report.py')
        args=['--output-name','additional-frame-late-006']
        for key,name in protocol['models'].items():args+=['--variant',key+'='+name]
        execute('evaluate_resolution_additional_frames.py',args)
        extra=json.loads((ROOT/'additional-frame-late-006/summary.json').read_text())
        groups=json.loads((ROOT/protocol['observed_groups']).read_text());values={}
        for key,result in extra['models'].items():
            values[key]={}
            for group,spec in groups['groups'].items():
                rows=[r for r in result['rows'] if r['divisor']==1 and r['frame'] in spec['frames']]
                assert len(rows)==spec['count']
                values[key][group]=dict(full_psnr=statistics.mean(r['full']['psnr'] for r in rows),
                    person_psnr=statistics.mean(r['subject_crop']['psnr'] for r in rows),
                    head_psnr=statistics.mean(r['head']['psnr'] for r in rows),
                    head_gradient_error=statistics.mean(r['head']['normalized_gradient_error'] for r in rows))
        result=dict(status='complete',created=time.time(),protocol=protocol,groups=groups,models=values,
            full_minus_low={g:{m:v-values['low'][g][m] for m,v in data.items()} for g,data in values['full'].items()},
            accepted=False,independent_extrapolation_gt=False)
        (ROOT/'native-late-observed-groups-001.json').write_text(json.dumps(result,indent=2))
        execute('append_late_observed_report.py')
        state.update(status='complete_pending_browser_and_visual_review',finished=time.time());state.pop('current',None);save()
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise

if __name__=='__main__':main()
