from pathlib import Path
P=Path('D:/00/工作/人体重建')
source=P/'src/avatar_core/fit_avatar_pair_static_114.py'
s=source.read_text('utf-8')
s=s.replace('from fresh_sequence_pipeline import c, REPO, RUN','from .runtime import c, ASSETS\nREPO=ASSETS/"gvhmr"')
start=s.index('    # Original pretrained hand predictions')
end=s.index('    # Scores are arbitrary detector outputs.',start)
s=s[:start]+'''    # New-video route: shared mean hand pose initializes finger variables.
    # All 42 freshly detected hand landmarks remain observations in the fit.
    # No HaMeR output, old hand fit or previous subject file is read.
    left0=torch.zeros((n,45),device='cuda')
    right0=torch.zeros((n,45),device='cuda')
'''+s[end:]
s=s.replace('s0=float(sol.x[3]);assert .05<s0<20','s0=float(sol.x[3]);assert np.isfinite(s0) and s0>1e-6, "Unobservable or non-positive body/world scale"')
s=s.replace("        original_hand_observations_sha256=c.sha(hand_src),source_sha256=c.sha(__file__),template_sha256=c.sha(c.MODEL),","        hand_initialization='SMPL-X mean; this-video 2D hand landmarks',source_sha256=c.sha(__file__),template_sha256=c.sha(c.MODEL),")
s=s.replace("        non_original_sfm_observation_weight=.5,","        non_original_sfm_observation_weight=1.0,")
s=s.replace("        weight_policy='Original SfM frames1.0; additional frames0.5, held unchanged for the camera-input comparison.',","        weight_policy='Every directly registered new-video frame has camera weight1.0; detector uncertainty is separate.',")
s=s[:s.index("if __name__=='__main__':")]
s=s.replace('# Shared-static adapter for the two repository comparison. Derived from C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\repos\\GVHMR-shuolee\\shuolee\\fit_fresh_sequence.py','# Static new-video fitter derived from the recorded local shared-static adapter.\n# New inputs, all available masks, mean hand initialization, fixed SfM cameras.')
target=P/'src/shuolee_portrait/fitting.py';target.write_text(s,encoding='utf-8')
compile(s,str(target),'exec')
print('Server fitter written; no optimizer executed')
