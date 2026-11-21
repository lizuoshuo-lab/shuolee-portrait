from pathlib import Path
import datetime,json
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-surface-consistency-fresh-079')
data=dict(checked=datetime.datetime.now().astimezone().isoformat(),purpose='Compare next open-source surface representations without changing the active079 training.',
    current=dict(method='PGSR with observed correspondence and local plane dispersion',
        distinction='Retains standard3D Gaussian RGB rendering. The new residual is point-to-plane at a detached blend; it is not official2DGS ray distortion.',
        evidence='078 original forward output parity and old-gradient numerical parity passed. New trained export parity still required.'),
    references=[dict(name='2D Gaussian Splatting',repo='https://github.com/hbb1/2d-gaussian-splatting',
        source='https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py',
        verified='Official oriented-disk surfels and perspective-correct rasterization; render_dist, surface depth and normals are exposed by diff_surfel_rasterization.',
        inference='A representation-level comparison could reduce volume layers; this video outcome is unknown.',
        prerequisite='Fresh native fit/initialization and camera/pixel contract, then an explicit viewer/PLY rendering-equivalence check. Existing3DGS loader compatibility must not be assumed.',
        locally_trained=False),
    dict(name='Gaussian Opacity Fields',repo='https://github.com/autonomousvision/gaussian-opacity-fields',
        sources=['https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py',
            'https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py'],
        verified='Official implementation uses a rasterized depth-distortion map, depth-normal consistency, filtered scales/opacities, and an opacity-field integration path for surface extraction.',
        inference='A3D-Gaussian surface alternative, but complete native rendering and export compatibility need testing; it does not by itself establish accurate camera estimates.',
        locally_trained=False)],
    decision='Finish the preregistered079 and inspect actual output first. Do not launch either alternative or a weight sweep during this round.')
(B/'open-source-comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Saved verified primary-source comparison; no additional training launched.')
