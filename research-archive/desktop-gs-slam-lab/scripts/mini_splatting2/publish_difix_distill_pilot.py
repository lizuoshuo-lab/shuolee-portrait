"""Publish paired trained 3D models and their complete paths, preserving earlier evidence."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from build_difix_novel_report import encode

S = Path(__file__).parent
ROOT = S.parents[1] / 'outputs/mini-splatting2-mip-depth-001'
RUN = ROOT / 'difix-distill-pilot-001'
VIEW = ROOT / 'viewer'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((RUN / 'summary.json').read_text())
    assert summary['status'] == 'complete_pending_visual_review'
    audit = RUN / 'publication'
    audit.mkdir(exist_ok=False)
    for name in ['index.html', 'viewer.js', 'novel-views.html']:
        shutil.copy2(VIEW / name, audit / ('before-' + name))
    before = {p.name: digest(p) for p in VIEW.glob('model-*.json')}
    template = json.loads((VIEW / 'model-native-mip-resolution-full.json').read_text(encoding='utf-8'))
    models = []
    for arm, label in [('control', '扩散三维对照 · 只用真实照片'), ('candidate', '扩散三维试验 · 真实照片 + 生成俯仰视角')]:
        key = 'difix-distill-' + arm
        source = RUN / arm / 'model.ply'
        assert digest(source) == summary['models'][arm]['model_sha256']
        destination = VIEW / 'models' / key / 'point_cloud.ply'
        destination.parent.mkdir(exist_ok=False)
        shutil.copy2(source, destination)
        meta = copy.deepcopy(template)
        for old in ['resolution_ablation', 'archiveSha256']:
            meta.pop(old, None)
        initial_depth = meta.pop('depth_guidance', None)
        meta.update(label=label, status='1,200步固定高斯数量微调 · 探索实验，尚未验收',
                    iterations=1200, refinement_steps=1200, pretrained_iterations=18000,
                    initialization_depth_guidance=initial_depth,
                    current_depth_guidance=dict(coefficient=0, used=False),
                    gaussianCount=316044, model=f'models/{key}/point_cloud.ply',
                    modelBytes=destination.stat().st_size, sha256=digest(destination),
                    provenance=f'difix-distill-pilot-001/{arm}', report='novel-views.html',
                    scope='同起点、同真实照片顺序；实验组额外使用12个生成俯仰视角。gsplat微调，不是MiniGS2原生增密训练。',
                    bounds_note='View framing bounds inherited from the common initial model; not recomputed geometry bounds.',
                    comparability_group='difix-distill-1200-seed17',
                    diffusion_distillation=dict(arm=arm, steps=1200, seed=17, fixed_topology=True,
                        pseudo_weight=0.25 if arm=='candidate' else 0, accepted=False,
                        renderer='gsplat', mip3d='Baked initial scales; no additional filter refresh'))
        (VIEW / f'model-{key}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        models.append(dict(key=key, label=label, sha256=meta['sha256']))
    html = (VIEW / 'index.html').read_text(encoding='utf-8')
    match = re.search(r'(<select id="model-select">)(.*?)(</select>)', html, re.S)
    assert match
    group = '<optgroup label="第五组：扩散三维微调 · 探索实验">' + ''.join(
        '<option value="' + m['key'] + '">' + m['label'] + '</option>' for m in models) + '</optgroup>'
    html = html[:match.end(2)] + group + html[match.end(2):]
    (VIEW / 'index.html').write_text(html, encoding='utf-8')
    js = (VIEW / 'viewer.js').read_text(encoding='utf-8')
    match = re.search(r'const variants=(\[.*?\]);', js)
    assert match
    keys = json.loads(match[1]); keys.extend(m['key'] for m in models)
    js = js[:match.start(1)] + json.dumps(keys) + js[match.end(1):]
    (VIEW / 'viewer.js').write_text(js, encoding='utf-8')
    assert all(digest(VIEW / name) == sha for name, sha in before.items())

    assets = VIEW / 'distill-view-assets'; assets.mkdir(exist_ok=False)
    assignments = {'elevation': {'A': 'candidate', 'B': 'control'}, 'distance': {'A': 'control', 'B': 'candidate'}}
    for arm in ['control', 'candidate']:
        shutil.copytree(RUN / arm / 'paths', assets / arm)
        shutil.copytree(RUN / arm / 'evaluation', assets / (arm + '-observed'))
    for kind, assignment in assignments.items():
        frames = [[RUN / assignment[a] / 'paths' / f'{kind}-{i:03d}.png' for a in ['A', 'B']] for i in range(60)]
        encode(frames, assets / f'{kind}-ab.mp4', ['A', 'B'])
    data = json.loads((VIEW / 'novel-view-assets/summary.json').read_text())
    data['distillation'] = dict(summary=summary, assignments=assignments, models=models,
        base='distill-view-assets/', protocol='difix-distill-pilot-001')
    (assets / 'summary.json').write_text(json.dumps(data['distillation'], indent=2))
    template_html = (S / 'novel_view_report.html').read_text(encoding='utf-8')
    (VIEW / 'novel-views.html').write_text(template_html.replace('__DATA__', json.dumps(data).replace('</', '<\\/')), encoding='utf-8')
    state = dict(status='published_pending_browser_review', created=time.time(), models=models,
        total_models=len(keys), original_metadata_unchanged=True, source_preserved=summary['source_preserved'],
        url='http://127.0.0.1:8820/novel-views.html', subjective_ratings_fabricated=False,
        accepted=False, videos={p.name:digest(p) for p in assets.glob('*.mp4')})
    (audit / 'summary.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
