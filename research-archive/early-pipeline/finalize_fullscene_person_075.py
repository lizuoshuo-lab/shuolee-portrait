"""Publish the requested full-scene-camera -> fresh human Gaussian experiment."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V = B.parent / 'viewer'
R = B / 'current/person-static-fullscene-camera-fresh-075'
O = V / 'body-fullscene-person-current'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    tmp.replace(p)


def image(name, caption):
    return (f'<figure><a href="body-fullscene-person-current/{name}">'
            f'<img src="body-fullscene-person-current/{name}" alt="{caption}"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def main(running):
    now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    progress = read(R / 'gaussian-run/progress.json')
    O.mkdir(exist_ok=True)
    for name in ['analysis-before-training.json', 'preparation.json', 'initialization-review.json']:
        shutil.copy2(R / name, O / name)
    shutil.copy2(R / 'gaussian-run/progress.json', O / 'progress.json')
    intro = '''<p>本轮按要求把两个环节接起来：<strong>整幅图求相机 → 人体粗配准 → 衣物轮廓初始化 → 人物高斯训练与渲染。</strong>相机估计使用人物和环境中的匹配点，通过几何检验与全局束调整求解；人体模板只提供人物的大致位置、尺度和初始化范围，最终点可落在衣服表面。</p>
<p>175帧均有本轮SfM相机，没有时间插值相机。新相机的内参、外参和径向畸变共同用于重新去畸变图像、掩码与特征坐标。重新进行1,200步SMPL-X粗拟合，再根据175帧轮廓与765个多视图衣物支撑点初始化200,000个点，训练共享静态高斯18,000步。未加载上轮拟合结果、高斯或优化器。</p>
<p>074的独立面部纹理检查没有通过改善门槛，这项记录保留。075是依照要求进行的完整流程对照；是否改善鼻口重影，必须看实际渲染。这里比较的是两套相机输入经过重新初始化后的完整流程，不是只替换外参的单变量实验。</p>'''
    if running:
        active = bool(progress.get('running', False))
        state = (f'<p class="notice"><strong>正在训练：{progress["step"]:,}／18,000步。</strong>20万点初始化与输入一致性检查已完成；当前预览是中间结果，尚未交付新的最终PLY。</p>' if active else
                 '<p class="notice"><strong>18,000步训练已完成。</strong>正在核对PLY导出与175帧同相机效果，最终交付结果即将更新。</p>')
        shutil.copy2(R / 'gaussian-run/review-current.jpg', O / 'training-preview.jpg')
        body = image('training-preview.jpg', '本轮原图与人物渲染，最终质量以同相机和新视角核验为准。' if not active else '本轮训练中的原图与人物渲染。6,000和9,000步的保存恰在透明度重置后，因此中间图偏淡；不以这些预览判定最终质量。')
        links = '<p><a href="./?model=latest">查看上一轮完整高斯</a> · <a href="#full-scene-localization">整图定位的原始核验</a></p>'
        decision_text = '075正在执行：整图相机、重新人体拟合、20万点初始化、从头18,000步静态人物训练。'
    else:
        result = read(R / 'gaussian-run/result.json')
        review = read(R / 'review/same-camera-result.json')
        decision = read(R / 'review/visual-decision.json')
        assert result['steps'] == 18000
        for p in (R / 'review').iterdir():
            if p.suffix in ['.jpg', '.png', '.mp4', '.json']:
                shutil.copy2(p, O / p.name)
        a, b = review['summary']['072']['all'], review['summary']['075']['all']
        state = '<p class="notice"><strong>075已完成：175帧、20万点初始化、从头18,000步。</strong>' + decision['viewer_status'] + '</p>'
        body = '<h3>相同相机、相同像素下的实际效果</h3>'
        body += '<p>' + decision['observed_view_conclusion'] + '</p>'
        body += image('head-033.jpg', '原视频34帧：左为实拍，中为072，右为075。两份高斯使用同一075相机和相同裁片；没有额外二维对齐。')
        body += image('head-099.jpg', '原视频100帧的相同对照。训练视角效果不代表新视角的真实几何。')
        body += image('body-099.jpg', '完整人物与衣物轮廓对照。模板只限定初始化范围，没有把高斯固定在裸人体表面。')
        body += (f'<p>175帧平均轮廓距离：{a["contour_mean_px"]:.3f} → {b["contour_mean_px"]:.3f}像素；'
                 f'边界透明度绝对误差：{a["boundary_alpha_mae"]:.4f} → {b["boundary_alpha_mae"]:.4f}；'
                 f'人物内部RGB绝对误差：{a["inside_rgb_mae"]:.4f} → {b["inside_rgb_mae"]:.4f}。'
                 '像素单位对应1536像素高的评测画幅；掩码是观测，不是独立几何真值。</p>')
        body += '<h3>未训练方向与连续观察</h3><p>' + decision['novel_view_conclusion'] + '</p>'
        body += image('novel-below25-bg0.25.jpg', '统一的下方25度相机：左072，右075。没有该方向实拍真值，作为外观和结构诊断。')
        body += '<video controls preload="metadata" style="width:100%;max-width:1040px" src="body-fullscene-person-current/continuous-same-camera.mp4"></video><p class="small">同一连续相机路径，左072、右075；同时改变方位和俯仰。未将生成或自洽图像当成真值。</p>'
        body += '<p><strong>结论与下一步：</strong>' + decision['next_step'] + '</p>'
        links = '''<p><a href="./?model=whole-body-static-fullscene">打开075人物高斯</a> · <a href="models/whole-body-static-fullscene/shuolee_075_s18000_f0100.ply?download=1">下载适配shuolee查看器的PLY</a> · <a href="body-fullscene-person-current/same-camera-result.json">同相机核验数据</a> · <a href="#full-scene-localization">相机估计核验</a></p>
<p>PLY为统一+Y向上的世界坐标，已核对完整协方差和球谐系数变换。00的shuolee查看器使用“打开文件”直接载入；不再加载人体骨架或蒙皮。文件名f0100仅表示参考查看相机，整个模型是共享静态几何。本轮是否在00实际打开，以交付凭据为准。</p>'''
        decision_text = decision['viewer_status']
    block = '<!-- FULL_SCENE_PERSON_075_BEGIN -->\n<section id="full-scene-person"><h2>075：整图相机定位后，重新初始化人物高斯</h2>' + state + intro + links + body + '</section>\n<!-- FULL_SCENE_PERSON_075_END -->'
    page = V / 'body-smplx.html'
    html = page.read_text(encoding='utf8')
    if '<!-- FULL_SCENE_PERSON_075_BEGIN -->' in html:
        i = html.index('<!-- FULL_SCENE_PERSON_075_BEGIN -->')
        j = html.index('<!-- FULL_SCENE_PERSON_075_END -->', i) + len('<!-- FULL_SCENE_PERSON_075_END -->')
        html = html[:i] + block + html[j:]
    else:
        marker = '<!-- FULL_SCENE_074_BEGIN -->'
        assert marker in html
        html = html.replace(marker, block + '\n' + marker, 1)
    html = html.replace('因此暂不采用这套相机开启下一轮训练。', '这项独立检查未通过；根据随后明确的完整流程要求，075已将这套相机作为试验输入，继续人体初始化与高斯训练，见上节。')
    html = html.replace('当前高斯对照仍为072（质量未通过）', '当前最新人物高斯（质量状态见075）')
    page.write_text(html, encoding='utf8')
    s = read(B / 'work-status.json')
    s.update(updated=now, current_experiment='075 full-scene camera / fresh human pipeline',
             current_phase='075 fresh static Gaussian training' if running else '075 training and local delivery completed; see visual decision',
             local_training_running=bool(progress.get('running', False)), current_decision=decision_text)
    s['latest_local_execution'] = dict(running=running, state='training' if running else 'complete',
        directory=str(R), training_steps=progress['step'], steps=18000,
        process_session=25384 if running else None, wsl_pid=386 if running else None,
        scope='full-scene074 cameras -> fresh person fit -> clothing initialization -> static Gaussian')
    s['full_scene_person_075'] = dict(directory=str(R), running=running, camera_ground_truth=False,
        camera_source='074', all_frames_registered=True, from_scratch=True,
        point_initialization='fresh human bbox plus 175 silhouette hull and 765 clothing tracks',
        initial_points=200000, steps=progress['step'], quality_accepted=False,
        report='http://127.0.0.1:8870/body-smplx.html#full-scene-person')
    if not running:
        s['full_scene_person_075'].update(visual_review=str(R / 'review/visual-decision.json'),
            viewer='http://127.0.0.1:8870/?model=whole-body-static-fullscene',
            compatible_ply=True, remote_00_deployed=False)
    save(B / 'work-status.json', s)
    mp = V / 'model-full-scene-localization-074.json'
    m = read(mp)
    m.update(status='074整图定位结果；075已使用这套相机重新进行人体初始化与静态人物训练，质量以075对照为准。',
             report='body-smplx.html#full-scene-person', experimentalUseRound='075')
    save(mp, m)
    print(json.dumps(dict(running=running, steps=progress['step'], report=str(page)), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--running', action='store_true')
    main(p.parse_args().running)
