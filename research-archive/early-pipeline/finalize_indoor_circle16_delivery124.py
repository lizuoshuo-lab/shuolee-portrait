"""Apply the existing verified delivery receipt to the new16-view artifact."""
import inspect
import finalize_indoor_clock_delivery124 as existing
from publish_indoor_circle16_124 import main as report


def main():
    source = inspect.getsource(existing.main)
    old = 'indoor-zoom-clock18k124'
    assert source.count(old) == 3
    source = source.replace(old, 'indoor-circle16-paired18k124')
    assert source.count('training_frames=6') == 1
    source = source.replace('training_frames=6', 'training_frames=16')
    old_assert="    assert a.read(root/'process-exit.json')['exit_code']==0"
    new_assert="""    assert a.read(root/'process-exit.json')['exit_code']==1
    assert a.read(root/'progress.json')['state']=='training_budget_reached_snapshot_export_failed'
    assert v['diagnostic_preview_only'] and not v['native_source_ply_parity']['passed']
    assert not v['native_rotated_render_parity']['passed']
    assert v['native_source_roundoff_diagnosis']['passed'] and v['native_rotated_roundoff_diagnosis']['passed']"""
    assert old_assert in source;source=source.replace(old_assert,new_assert)
    source = source.replace('INDOOR_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN',
                            'INDOOR_CIRCLE16_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN')
    namespace = dict(existing.main.__globals__, report=report)
    exec(compile(source, __file__+':existing-delivery-checks', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
