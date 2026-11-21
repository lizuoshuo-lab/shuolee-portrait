from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

root=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-portrait-server-check-008')
rows=json.loads((root/'data/inputs.json').read_text())['records']
frames=['frame_000001','frame_000005','frame_000009','frame_000014','frame_000018','frame_000023']
images=[]
for frame in frames:
    row=next(r for r in rows if r['frame']==frame)
    for kind in ['input','mesh_overlay']:
        file=root/'reviews/fit'/f'{frame}_{kind}.png'
        images.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                           source_frame_index=row['source_frame_index'],kind=kind))
review=dict(updated=datetime.now(timezone.utc).isoformat(),actual_images_reviewed=True,
    reviewer='direct image inspection in current task',images=images,observations=[
        'All six paired inputs and mesh overlays were visually inspected. The subject is in the broadly correct image region across the sampled sides and front.',
        'Side-view head, hands, calves and feet do not coincide exactly with the visible boundaries; front shoulder/sleeve and shoe boundaries also differ.',
        'The overlay is an unclothed body template, not reconstructed hair, glasses, jacket, trousers or shoes. Some silhouette differences are expected; they cannot establish camera error or physical geometry accuracy by themselves.',
        'Background bystanders are present and change between frames. This image review does not classify all scene features as static.',
        'The coarse joint gate uses fitted observations, not independent truth. No improvement in Gaussian ghosting is established by this integration run.'
    ],images_are_undistorted_inputs_with_isotropic_display_resize=True,
    two_dimensional_subject_warp_used=False,feature_observations_independent_truth=False,
    physical_geometry_accepted=False,quality_accepted=False,gaussian_training_admitted=False,
    new_gaussian_or_ply=False,novel_view_quality_tested=False)
(root/'visual-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Reviewed 12 actual images; physical quality remains unaccepted.')
