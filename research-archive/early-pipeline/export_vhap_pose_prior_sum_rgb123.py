"""Use the existing pure RGB replayer for the completed prior-only control."""
import export_vhap_body_continuity_rgb123 as replay

if __name__ == '__main__':
    replay.RUN = replay.ROOT / 'smplify-vhap-pose-prior-sum'
    replay.EXPECTED_STEPS = 2270
    replay.main()
