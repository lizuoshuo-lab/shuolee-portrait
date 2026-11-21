"""Register the user's renewed, bounded static-scene calibration diagnosis."""
from pathlib import Path
import datetime as dt
import hashlib
import json

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O = B / 'current/portrait-causal-diagnosis-117'
PREV = B / 'current/portrait-alignment-open-source-116'

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    assert not O.exists(), 'Do not overwrite an existing protocol or start a duplicate experiment.'
    O.mkdir()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    sources = [B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json',
               PREV/'hsfm/joint-diagnostic-reserved/frozen-validation.npz',
               PREV/'hsfm/joint-diagnostic-reserved/frozen-validation.json',
               PREV/'hsfm/joint-diagnostic-reserved/validation-visual-review.json']
    protocol = dict(
        created=now, round='117', status='diagnostic_only',
        authorization='User renewed long-term causal diagnosis and explicitly chose fixed scene points around the person to constrain cameras.',
        objective='Separate camera calibration, material correspondence and person micromotion before any fresh Gaussian training.',
        primary_camera_constraint='One shared 3D point per static scene material feature. One shared intrinsic model conditional on capture stability, separate R/t per actual frame. After held validation, freeze cameras for Gaussian training.',
        fixed_inputs=dict(frame_ids=list(range(1,176)), baseline='074 immutable estimated COLMAP calibration',
            subject='One shared static clothed subject; no dynamic reconstruction branch',
            image_space='Original rectified1514x2694, integer pixel centers; K principal point subtract0.5 exactly once',
            old_training='114 completed and failed quality; do not restart or reuse learned subject or Gaussians'),
        source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        stages=[
            dict(id='D0', variable='None: read-only baseline reanalysis',
                 observations='Existing231 three-view material tracks;65 manually reviewed static-object tracks. This is reused validation, not new independent GT.',
                 outputs=['Endpoint triangulation and held middle projection', 'Sampson residual for each of three pairs, with track-level summaries',
                          'Static-anchor frame/spatial coverage, relative depth and parallax, source overlay'],
                 stop='One deterministic CPU analysis. Do not optimize a camera or infer a unique cause from these numbers.'),
            dict(id='D1', variable='Static scene calibration only',
                 preparation='Build fresh175-frame static-object track inventory from original images; remove visible moving people, reflections, screens and occlusions by observation review before any candidate residual. Separate fit and validation tracks by material/time blocks.',
                 parameters='Candidate1 fixedK optimizes per-frameR/t and staticXYZ; candidate2 changes only the shared intrinsic-refinement setting with exactly the same tracks, initialization, budget and gauge. No person landmarks in either fit.',
                 gauge='Fix first camera and one nonzero baseline length in the original arbitrary scale. No absolute metric claim.',
                 admission='Connected175-camera track graph; every frame has >=20 unique verified static tracks in >=4 cells of a4x4 image grid; at least3 spatially separated object patches per frame; check ray angles, depth variation and local sensitivity. If weak/planar geometry cannot constrain K, keep Kfixed and label it unobservable.',
                 stop='At most one track-extraction pass plus one implementation repair, and at most2 calibration fits (<=45min each). No threshold sweep. Inadequate coverage is a result, not permission to interpolate or relax gates.'),
            dict(id='D2', variable='Material correspondence with cameras fixed',
                 method='Inspect forward/backward and3-view cycle plus original-image patches. Compare stable face/clothing texture against semantic landmark labels on the same frames. Never equate contour/occluded/reflected points solely by detector index.',
                 stop='One preregistered pair/cycle audit, <=60min. Preserve all rejected observation reasons; do not select by lower camera error.'),
            dict(id='D3', variable='Small relative subject/head motion with background camera fixed',
                 method='Compare shared staticXYZ to low-degree rigid residual motion on disjoint fitting/held material tracks, with static-object negative controls and camera-uncertainty sensitivity. Head and torso distinguished. Diagnostic only; no per-frame Gaussian deformation or image warping.',
                 admission='Reliable static camera/anchor coverage and material-identity gates first; otherwise report motion not identifiable.',
                 stop='One fixed static-versus-rigid diagnostic, <=45min; report uncertainty if motion cannot be separated.'),
            dict(id='T', variable='Two confirmed avatar repositories, same accepted frozen cameras',
                 admission='All alignment checks, source overlays and independent face/body/background protection pass. A causal correction must have evidence beyond fitting residual.',
                 training='aipixel/GaussianAvatar and jmaraval/GaussianAvatars_SMPLX local missing-entry adaptation; each fresh subject fit,200000 initial points, empty optimizer,18000steps, same175images.',
                 stop='At most one fresh pair; no automatic next experiment.')],
        provisional_gates=dict(
            note='Freeze exact unseen validation split and statistical protocol before D1 fitting. The reused116bank remains a secondary regression check, not an untouched test.',
            finite_positive_depth=True, complete_175_real_frames=True,
            background_p90_increase_max_px=0.2, face_lower_median_relative_reduction_min=0.2,
            face_lower_and_upper_p90_must_not_increase=True, torso_and_legs_p90_increase_max_px=0.5,
            units='Original rectified pixels, exact unchanged observations',
            quality='Numerical gates alone insufficient; same-view original overlays, wholebody, face and continuous novel-view render review after training required.'),
        unresolved=['Original sensor K/Rt unavailable; tdsp/SDK later from user.',
                    'Sequential monocular images may not uniquely identify physical calibration and subject motion.',
                    'Staticity and exact material correspondence remain measurements to validate, not assumed labels.'],
        rules=['No rerun of closed116PixSfM/RoMa/HSfM sweeps.', 'Only one GPU task; verifyPID,command,log before launch.',
               'No edits to running source, no new weight scans/downloads, no upload of subject data.',
               'No 2D warping of training images to conceal mismatch.', 'Do not silently change074, historical models or tab15.',
               'Diagnostic rounds update body-smplx.html#alignment117 and latestDiagnostic only; do not fabricate a newPLY.',
               'Trained results require full covariance andSH world/+YPLY, native readback, actual OpenFile Gaussian/points UI review.',
               'No00deployment claim without current remote validation. Update only the existing52pagePPT lastpage for substantive results.'],
        close_when='Complete bounded diagnosis plus accepted fresh-pair quality review, or diagnostic budget/observability gates exhausted. Stop heartbeat and report remaining evidence; do not indefinitely append methods.',
        root_cause_resolved=False, training_started=False, quality_accepted=False)
    save(O/'protocol.json',protocol)
    save(O/'queue-state.json',dict(updated=now,state='registered_D0_ready',stage='D0',active_pid=None,
        training_started=False,quality_accepted=False,root_cause_resolved=False,automation_status='NOT_CREATED',
        completed=[],next_action='Run the deterministic read-only D0 baseline audit, inspect source overlays, then prepare fresh static-object anchors.'))
    save(O/'evidence-ledger.json',dict(updated=now,entries=[dict(id='user_scope',
        observation='User chose stable points in the scene around the person. Prior116closed with no accepted camera.',
        implication='Newbounded117scope; scene anchors estimate cameras, face/body initially independent checks.',
        evidence=['protocol.json',str(PREV/'closure.json')])]))
    print(json.dumps(dict(root=str(O),state='preregistered_before_D0_measurement'),ensure_ascii=False))

if __name__ == '__main__':
    main()
