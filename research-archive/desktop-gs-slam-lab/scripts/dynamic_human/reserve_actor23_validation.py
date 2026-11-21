"""Reserve an independent-scene evaluation split before fitting or viewing results."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106'
D=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor2_3')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(O/'source-inventory.json')['heldout_RGB_opened'] is False
assert not (O/'validation-reservation.json').exists()
frames=list(range(600,625));held=['00','06','12'];train=[f'{c:02d}' for c in range(18) if f'{c:02d}' not in held]
for c in train+held:
    for f in frames:
        for folder in ['images','bgmtv2']:
            assert (D/folder/c/f'{f:06d}.jpg').is_file(),(folder,c,f)
save(O/'validation-reservation.json',dict(status='split_reserved_training_not_started',recorded_utc=datetime.now(timezone.utc).isoformat(),dataset='actor2_3',frame_window=frames,reference_frame=612,train_cameras=train,train_frames=frames[::2],heldout_cameras=held,heldout_times=frames[1::2],source_selection='Camera04 six evenly spaced source thumbnails only. Window600..624 selected for bending and lifted-leg motion visible in the training source at600; no candidate model rendered or heldout RGB inspected.',training_image_count=195,evaluation_splits={'known_camera_unseen_time':180,'unseen_camera_unseen_time':36,'unseen_camera_known_time':39},evaluation='Original native1920x1080 RGB; full image, whole-person, clothing, limbs, hand/object context and temporal continuity. Fixed image windows must be recorded before candidate renders. Foreground masks are uncertain observations, not exact semantic truth.',training_policy='Do not import actor5_6 geometry, estimated clock offsets or hand poses. Train-only geometry initialization; released surfs are excluded because they may use reserved cameras. Freeze a complete static/dynamic protocol after training-only calibration and synchronization qualification, before independent evaluation.',required_preconditions=['Read native training images and masks; apply actual calibration and verify common remapping.','Qualify geometric reprojection and time consistency from training cameras; do not assert hardware synchronization from frame names.','Specify unchanged static/dynamic loss, iteration budget, initialization, motion capacity and native-resolution rendering.','Create fixed full-person and detail review plan before rendering candidates.'],not_yet_done=['native training preprocessing','training-only geometry and timing qualification','static/dynamic training','heldout rendering','full scene background and OmniRe integration'],heldout_RGB_opened=False,source_survey_visually_reviewed=True,source_survey_sha256=sha(O/'training-source-survey.png'),goal_complete=False))
print('Reserved actor2_3 frames600..624:195training photos,255heldout contexts; no training or heldout viewing.')
