"""Historical stage costs, not a claimed new eight-frame end-to-end run."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

def main():
    demo=a.REC/'demo-vhap-initialization';out=demo/'body-gaussian-smplify-eight-laplacian'
    fit=demo/'fullbody-smplifyx/full-sequence/fit/official-fit-result.json';r=a.read(fit)
    chosen=set(a.read(out/'selection.json')['selected_frames'])
    rows=[dict(frame=x['frame_id'],seconds=x['elapsed_seconds']) for x in r['frame_results'] if x['frame_id'] in chosen]
    assert len(rows)==8
    total=sum(x['seconds'] for x in rows)
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='historical_timing_audit_only',
        existing_all59_fit_seconds=r['elapsed_seconds'],selected8_historical_fit_seconds_sum=total,
        selected_frames=rows,ratio_to_recorded59_wall_time=total/r['elapsed_seconds'],
        omitted_costs=['New video decode and deterministic preselection','Masks and2D observations',
            'New shared body-shape reference fit and its possible correction','Training, export and native/UI verification'],
        causal_limit='These eight times were measured inside the old59-frame run using an already fitted reference33 body shape. Summation is neither a measured8-frame wall time nor a new-video benchmark.',
        current_selection_dependency='Current8-frame rule uses already-computed59-frame SMPLify quality/yaw. It cannot be used unchanged before fitting.',
        minimum_future_interface='Use a reproducible cheap pixel/2D-observation preselection before SMPLify, then pass --frame-ids through the existing original sequence entry. Reference identity must be fitted from the selected set and validated; do not silently import old subject shape or fit59 then discard51.',
        solver_rewrite_required=False,new_fit_started=False,source_sha256=a.sha(fit),
        quality_accepted=False,end_to_end_15min_accepted=False)
    a.save(out/'pipeline-timing-audit.json',result)
    print(result)

if __name__=='__main__':main()
