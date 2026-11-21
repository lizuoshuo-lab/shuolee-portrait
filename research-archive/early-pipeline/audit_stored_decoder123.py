"""Read-only measurement of an existing SimCC decoder, without new inference."""
from pathlib import Path
import argparse,json,hashlib
from datetime import datetime,timezone
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--coverage',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.coverage);out=Path(args.output)
assert not out.exists(), 'Inspect existing result rather than rerun.'
metadata=json.loads((root/'observations.json').read_text(encoding='utf-8-sig'))
rows=[]
for r in metadata['rows']:
    f=root/'responses'/f"{r['slot']:03d}.npz"
    with np.load(f) as z:
        sx,sy=z['simcc_x'],z['simcc_y'];scale=z['scale'];center=z['center']
        assert sx.shape[-1]==576 and sy.shape[-1]==768
        step=scale/np.array([576.,768.])
        reconstructed=np.stack((sx.argmax(-1),sy.argmax(-1)),-1)[0]*step+center-scale/2
        parity=float(np.abs(reconstructed-z['keypoints']).max())
        assert parity<.001
        rows.append(dict(slot=r['slot'],source_frame=r['source_frame_index'],native_grid_step_xy_px=step.tolist(),
                         ideal_nearest_grid_quantization_radius_px=float(np.linalg.norm(step/2)),
                         reconstruction_parity_max_px=parity,
                         face_response_score_range=[float(z['scores'][23:91].min()),float(z['scores'][23:91].max())],
                         response_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
steps=np.array([r['native_grid_step_xy_px'] for r in rows])
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='stored_decoder_measured',frames=len(rows),
            decoder='Existing integer SimCC argmax, ratio2; scale maps predictions back to native frame.',
            native_grid_step_xy_px_min=steps.min(0).tolist(),native_grid_step_xy_px_median=np.median(steps,0).tolist(),native_grid_step_xy_px_max=steps.max(0).tolist(),
            ideal_nearest_grid_quantization_radius_px_range=[min(r['ideal_nearest_grid_quantization_radius_px'] for r in rows),max(r['ideal_nearest_grid_quantization_radius_px'] for r in rows)],
            face_response_score_range=[min(r['face_response_score_range'][0] for r in rows),max(r['face_response_score_range'][1] for r in rows)],
            output_is_calibrated_probability=False,face_ground_truth_error_measured=False,
            interpretation='Floating point native coordinates retain the integer argmax grid. Grid spacing is not a measured error or a strict lower bound. Response maxima are not calibrated visibility or localization probabilities. No decoder or thresholds changed.',
            new_inference=False,new_fit=False,rows=rows)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
