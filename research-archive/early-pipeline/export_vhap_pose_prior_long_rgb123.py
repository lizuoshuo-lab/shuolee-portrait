"""Read-only white RGB replay of the completed120-epoch long fit."""
import export_vhap_body_continuity_rgb123 as replay

if __name__=='__main__':
    replay.RUN=replay.ROOT/'smplify-vhap-pose-prior-long'
    replay.EXPECTED_STEPS=7580
    replay.CHECKPOINT_EPOCH=120
    replay.main()
