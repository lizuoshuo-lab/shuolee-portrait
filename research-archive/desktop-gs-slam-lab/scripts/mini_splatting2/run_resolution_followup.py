"""Secondary frozen24-frame check and viewer registration after primary GPU jobs."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001';SCRIPTS=Path(__file__).parent


def main():
    path=ROOT/'native-resolution-followup-queue.json';assert not path.exists()
    state=dict(status='waiting_for_primary_publication',started=time.time(),completed=[],accepted=False)
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
    save()
    try:
        protocol=json.loads((ROOT/'native-resolution-followup-protocol-001.json').read_text())
        assert hashlib.sha256((SCRIPTS/'evaluate_resolution_additional_frames.py').read_bytes()).hexdigest()==protocol['script_sha256']
        for name,digest in protocol['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        while True:
            queue=json.loads((ROOT/'native-resolution-cache-publication-queue.json').read_text())
            if queue['status']=='failed':raise RuntimeError('Primary queue failed: '+queue.get('error','inspect log'))
            if queue['status'] in ['complete','complete_pending_browser_and_visual_review']:break
            time.sleep(20)
        command=[sys.executable,str(SCRIPTS/'evaluate_resolution_additional_frames.py'),'--output-name','additional-frame-resolution-005']
        for key,name in protocol['models'].items():command+=['--variant',key+'='+name]
        state.update(status='evaluating_additional24');save()
        with (ROOT/'resolution-additional24.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append('additional24');save()
        extra=json.loads((ROOT/'additional-frame-resolution-005/summary.json').read_text())
        groups=json.loads((ROOT/protocol['observed_groups']).read_text())
        values={}
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
            full_minus_low={group:{metric:value-values['low'][group][metric] for metric,value in data.items()} for group,data in values['full'].items()},
            accepted=False,independent_extrapolation_gt=False)
        (ROOT/'native-resolution-observed-groups-001.json').write_text(json.dumps(result,indent=2))
        state.update(status='registering_viewer');save()
        with (ROOT/'resolution-viewer-registration.log').open('x') as log:
            subprocess.run([sys.executable,str(SCRIPTS/'register_resolution_viewer.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed']+=['observed_groups','viewer_registration']
        state.update(status='complete_pending_browser_and_visual_review',finished=time.time());save();print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()
