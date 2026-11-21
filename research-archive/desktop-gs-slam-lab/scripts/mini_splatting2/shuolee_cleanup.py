"""Prepare a bounded cleanup; deletion is performed by the guarded PowerShell runner."""
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
ROOT = LAB / 'outputs/mini-splatting2-mip-depth-001'
DOWNLOADS = Path.home() / 'Downloads/05_mini_splatting2_viewer'
OUT = ROOT / 'shuolee'
VIEW = ROOT / 'viewer'
KEEP = ['original', 'native-motion-mip-depthmask', 'native-mip-depth003', 'native-mip-resolution-full']

RETIRED = '''portable-depth-003 portable-native-002 portable-repeat-audit-004 portable-resolution-006 portable-spread-005
classic-refine-s0 classic-refine-s0-r1 classic-refine-s1 classic-refine-s2
depth-refine-s0 depth-refine-s1 depth-refine-s2 mip-depth-refine-s0 mip-depth-refine-s1 mip-depth-refine-s2 mip-refine-s0 mip-refine-s1 mip-refine-s2
depth-spread-candidates-002 depth-spread-evaluation-001 depth-spread-research-001
difix-appearance-test difix-appearance-test-r1 difix-distill-pilot-001 difix-novel-paths-001 difix-training-support-gating-001
free-space-audit-001 geometry-evaluation
native-baseline-00032-s0 native-baseline-18000-s1 native-baseline-18000-s2
native-depth-smoke-600-s0 native-depthspread-smoke-600-s0
native-depthguide003-18000-s1 native-depthguide003-18000-s2 native-depthguide010-18000-s0 native-depthguide010-18000-s1 native-depthguide010-18000-s2
native-depthspread000-18000-s0 native-depthspread003-18000-s0 native-depthspread010-18000-s0
native-head-full-18000-s0 native-head-full-smoke-3300-s0 native-head-observed-publication-001 native-head-paths-001 native-head-review-001 native-head-visual-review-001
native-late-full-18000-s0 native-late-full-smoke-5100-s0 native-late-observed-publication-001 native-late-paths-001 native-late-review-001 native-late-visual-review-001 native-late-extra-visual-review-001
native-mip-32-s0 native-motion-mip-depthmask-18000-s1 native-motion-mip-depthmask-18000-s2 native-motion-rgb-18000-s1 native-motion-rgb-18000-s2
native-resolution-cache-full-smoke-3300-s0 native-resolution-cache-low-18000-s0 native-resolution-cache-paths-001 native-resolution-cache-review-001
native-resolution-full-18000-s0 native-resolution-full-smoke-2200-s0 native-resolution-low-18000-s0 native-resolution-extra-visual-review-001 native-resolution-visual-review-001
native-spread-paths-001 native-spread003-head-preview-001 native-spread-observed-groups-001
additional-frame-head-007 additional-frame-late-006 additional-frame-resolution-005 additional-frame-spread-004
render-only-aa render-only-classic render-only-mip renderer-probe remote-video-check'''.split()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'cleanup-plan.json').exists():
        raise SystemExit('A plan already exists. Review it; do not overwrite it during deletion.')
    protected = [LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4',
                 ROOT/'viewextrapolator-pilot-001/pitch_up.npz', ROOT/'viewextrapolator-pilot-001/pitch_down.npz']
    for variant in KEEP:
        meta = json.loads((VIEW/f'model-{variant}.json').read_text(encoding='utf-8'))
        p = VIEW/meta['model']
        assert p.is_file() and sha(p) == meta['sha256'], variant
        protected.append(p)
    for run in ['native-depthguide003-18000-s0', 'native-motion-mip-depthmask-18000-s0', 'native-resolution-cache-full-18000-s0']:
        protected.extend((ROOT/run).rglob('*.ply'))
    # The prepared clips are lossless and self-contained. Keep the common camera path too.
    camera_source = ROOT/'difix-novel-paths-001/cameras.json'
    camera_dest = ROOT/'viewextrapolator-pilot-001/reference-camera-paths.json'
    shutil.copy2(camera_source, camera_dest)
    protected.append(camera_dest)

    targets = [ROOT/name for name in RETIRED]
    targets += [ROOT/'sources/difix_ref']  # Retired single-image teacher; keep its provenance below.
    targets += [p for p in (VIEW/'models').iterdir() if p.is_dir() and p.name not in KEEP]
    targets += [p for p in VIEW.glob('*-assets') if p.is_dir()]
    targets += [p for p in VIEW.glob('model-*.json') if p.stem[6:] not in KEEP]
    targets += [DOWNLOADS/name for name in ['MiniGS2_Depth_20261019.zip','MiniGS2_Native_20261019.zip',
        'MiniGS2_Optimization_20261018.zip','MiniGS2_RepeatAudit_20261019.zip','MiniGS2_Resolution_20261019.zip','MiniGS2_Spread_20261019.zip']]
    targets = [p.resolve() for p in targets if p.exists()]
    allowed = [ROOT.resolve(), DOWNLOADS.resolve()]
    for p in targets:
        assert any(p.is_relative_to(a) and p != a for a in allowed), p
        assert not p.is_symlink(), p
        for q in protected:
            assert not q.resolve().is_relative_to(p), (q, p)

    ledger = {'created':datetime.now().isoformat(), 'owner':'shuolee',
        'policy':'Delete rejected experimental models, renders, caches and obsolete delivery packages. Preserve inputs, source, useful baselines and active work. Small metrics and decision records remain here.',
        'retained_models':KEEP, 'final_goal_accepted':False,
        'decisions':[
            {'method':'motion mask + Mip + depth 0.003','decision':'Retain as useful reference; wide-pitch human clarity is unresolved.'},
            {'method':'full resolution 1514x2694','decision':'Retain only as active video-diffusion initialization, not as an accepted breakthrough.'},
            {'method':'depth spread, late densification, head weighting','decision':'Delete model and media artifacts; no decisive human-clarity improvement.'},
            {'method':'independent Difix + weak 3D distillation','decision':'Delete models, generated media and cached teacher weights; appearance/continuity concerns and no decisive 3D clarity gain.'}],
        'evidence':{}, 'protected_files':[]}
    evidence = [ROOT/name for name in ['native-depth-matched-repeats.json','native-spread-final-review-001.json',
        'native-resolution-final-review-001.json','native-late-final-review-001.json','native-depth-summary.json']]
    for p in targets:
        if p.is_dir() and p.parent == ROOT:
            evidence.extend(q for q in p.rglob('*.json') if q.name in ['summary.json','evaluation.json','protocol.json','manifest.json','visual-review.json'] and q.stat().st_size < 300000)
    evidence.append(ROOT/'sources/difix-download-receipt.json')
    for p in sorted(set(evidence)):
        if p.is_file():
            try: ledger['evidence'][str(p.relative_to(ROOT))] = json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError): pass
    for p in protected:
        ledger['protected_files'].append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(OUT/'results-ledger.json', ledger)

    # Shrink the existing viewer in place so old entry URLs remain useful.
    html = (VIEW/'index.html').read_text(encoding='utf-8')
    labels = ['原始 Mini-Splatting2 基线','运动遮罩＋Mip 对照','Mip＋深度 0.003 · 保留参考','全分辨率模型 · 下一轮初始化']
    options = ''.join(f'<option value="{v}">{label}</option>' for v,label in zip(KEEP,labels))
    html = re.sub(r'(<select id="model-select">).*?(</select>)',lambda m:m[1]+options+m[2],html,flags=re.S)
    html = re.sub(r'<a class="report-link".*?</a>', '', html, flags=re.S)
    html = re.sub(r'<details><summary>实验口径与模型信息</summary>.*?</details>',
        '<details><summary>模型保留说明</summary><p>仅保留原始基线、已有收益的组合和下一轮必需的初始化模型。无明显收益的试验模型、重复渲染和旧打包文件已按要求清理。</p><p>当前尚未解决大幅俯仰下的人物模糊、外观变形和漂浮物。全分辨率模型用于继续验证，不代表已达到验收标准。</p><a id="metadata" href="model-original.json" target="_blank">当前模型参数与校验值</a></details>', html, flags=re.S)
    html = html.replace('MINI-SPLATTING2 · LOCAL ABLATIONS','SHUOLEE · MINI-SPLATTING2')
    (VIEW/'index.html').write_text(html,encoding='utf-8')
    js=(VIEW/'viewer.js').read_text(encoding='utf-8')
    js=re.sub(r'const variants=\[.*?\];','const variants='+json.dumps(KEEP)+';',js,count=1)
    (VIEW/'viewer.js').write_text(js,encoding='utf-8')
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>shuolee · 当前训练结果</title><style>body{max-width:900px;margin:60px auto;padding:0 24px;background:#101719;color:#e5edeb;font:18px/1.8 system-ui}a{color:#acd9bc}h1{font-size:30px}li{margin:14px 0}.muted{color:#a6b5b1}</style><h1>当前训练结果与下一步</h1><p><a href="./">进入整体场景查看器 →</a></p><p>目前保留四个模型：原始 Mini-Splatting2、运动遮罩＋Mip 对照、增加 0.003 深度指导的参考模型，以及视频扩散下一轮使用的全分辨率初始化模型。</p><p>运动遮罩减轻背景运动残影；Mip 改善缩小观察时的混叠；深度指导带来有限且经重复验证的改善。大幅俯仰下的人物清晰度、外观一致性和漂浮物问题仍未通过验收。</p><p>深度分散约束、后期增密、头部加权、独立 Difix 与弱三维蒸馏未带来明确突破。其模型、重复媒体和旧安装包按要求删除，保留一份精简结果台账。</p><p>下一轮重点：联合视频扩散的一致性约束。连续视角的清晰度、人物外观、结构和稳定性作为主要观察项，真实视图误差作为辅助证据。生成视角不作为真实几何。</p><p class="muted">旧实验页面已合并到这里；不会把已清理的试验显示成正在运行或已验收。</p></html>'''
    for p in VIEW.glob('*.html'):
        if p.name!='index.html': p.write_text(report,encoding='utf-8')
    for v in KEEP:
        p=VIEW/f'model-{v}.json';meta=json.loads(p.read_text(encoding='utf-8'))
        meta['retention_owner']='shuolee';meta['report']='report.html'
        if v=='native-mip-depth003':
            meta['scope']='运动遮罩、Mip 与 0.003 深度指导；三次匹配重复已完成，保留 seed0 代表模型。大幅俯仰清晰度仍未验收。'
        if v=='native-mip-resolution-full': meta['status']='下一轮视频扩散初始化 · 不是已验收改进'
        dump(p,meta)

    rows=[]
    for p in targets:
        files=[p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()]
        rows.append({'path':str(p),'bytes':sum(q.stat().st_size for q in files),'files':len(files),
            'reason':'retired experiment or superseded delivery copy'})
    plan={'owner':'shuolee','created':datetime.now().isoformat(),'allowed_roots':list(map(str,allowed)),
        'protected_files':ledger['protected_files'],'ledger':str(OUT/'results-ledger.json'),
        'targets':rows,'total_bytes':sum(r['bytes'] for r in rows)}
    dump(OUT/'cleanup-plan.json',plan)
    print(json.dumps({'targets':len(rows),'GiB':round(plan['total_bytes']/2**30,3),'ledger_bytes':(OUT/'results-ledger.json').stat().st_size,'retained_models':KEEP}))


if __name__=='__main__': main()
