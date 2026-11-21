"""Plot saved local Bright Screen runs, with explicit acceptance and provenance gates.

Never computes ATE, aligns trajectories, compares accuracy, or uses ROS odometry
headers. A failed-run layout preview requires the explicit --preview-failed flag.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
RUN_PATTERN = re.compile(r'^bright_screen_wall_(livo|lio)_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$')
BRIGHT_SHA = '03b0487ab3727e379e53b9fed63a82fb984ebdd6bf15168c2291815daf2b301d'
COLORS = {'livo': '#0f766e', 'lio': '#53687e', 'update': '#c76b2c', 'ink': '#183247'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(LAB)).replace('\\', '/')


def load_tum(path):
    timestamps, positions, norm_errors = [], [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values = line.split()
        require(len(values) == 8, 'Not an eight-column TUM row')
        timestamps.append(int(Decimal(values[0]) * Decimal(1000000000)))
        numeric = np.array([float(v) for v in values[1:]], dtype=np.float64)
        require(np.isfinite(numeric).all(), 'Non-finite TUM pose')
        positions.append(numeric[:3])
        norm_errors.append(abs(float(np.linalg.norm(numeric[3:])) - 1.0))
    timestamps, positions = np.array(timestamps, dtype=np.int64), np.array(positions)
    require(len(timestamps) >= 2 and (np.diff(timestamps) > 0).all(), 'TUM timestamps must be strictly increasing')
    require(max(norm_errors) <= 1e-3, 'TUM quaternion is not unit length')
    return timestamps, positions


def choose_font(explicit):
    candidates = ([explicit] if explicit else []) + [
        Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for path in candidates:
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return str(path)
    raise ValueError('Chinese chart font unavailable; provide --font-file')


def style(ax):
    ax.set_facecolor('#fbfcfd')
    ax.grid(color='#dae3e9', linewidth=0.7, alpha=0.65)
    ax.tick_params(colors='#466174', labelsize=10)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#bccbd5')
    ax.xaxis.label.set_color(COLORS['ink'])
    ax.yaxis.label.set_color(COLORS['ink'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--runs', nargs=2, required=True, metavar=('LIVO_RUN_NAME','LIO_RUN_NAME'))
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--paired-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-failed', action='store_true')
    parser.add_argument('--font-file', type=Path)
    args = parser.parse_args()
    summary_file, audit_file, csv_file = [p.resolve() for p in [args.summary,args.audit,args.paired_csv]]
    summary, audit = [json.loads(p.read_text(encoding='utf-8')) for p in [summary_file,audit_file]]
    group = summary_file.parent.name
    require(group.startswith('fast-livo2-local-replay-'), 'Summary is not a local replay group')
    require(not summary.get('ground_truth_available', False), 'This no-GT chart requires the correct no-GT summary')
    require(not audit.get('ATE_computed', False), 'Unexpected accuracy-derived audit inputs')
    selected = {}
    for name in args.runs:
        match = RUN_PATTERN.fullmatch(name)
        require(match is not None, '--runs accepts bare validated run names, not paths or latest selectors')
        mode = match.group(1)
        matches = [row for row in summary['runs'] if row['run_name'] == name]
        require(len(matches) == 1 and matches[0]['mode'] == mode, 'Explicit run does not uniquely match the summary')
        row = matches[0]
        require(row['group'] == group, 'Mixed replay groups are not allowed')
        run = summary_file.parent / 'runs' / name
        require(run.resolve().parent == (summary_file.parent / 'runs').resolve(), 'Unsafe run location')
        require(row['rate'] == 0.5, 'Two selected runs must use the fixed half-speed protocol')
        require(row['bag']['bytes'] == 377748402 and row['bag']['sha256_before'] == BRIGHT_SHA
            and row['bag']['sha256_after'] == BRIGHT_SHA, 'Not the fixed complete Bright Screen sensor input')
        audited = audit['modes'][mode]
        require(audited['run_name'] == name, 'Audit belongs to another run')
        if not args.preview_failed:
            require(row['accepted'] is True, 'Final chart requires accepted selected runs')
            require(row['child_exits']['all_required_children_exit_zero'] is True, 'Final chart requires actual child exit-zero evidence')
            require(audited['clean_mapper_shutdown_proven'] is True, 'Independent audit has not confirmed a clean mapper exit')
            require(row['bag']['unchanged'] is True, 'Final chart requires unchanged complete sensor input')
        trajectory = run / 'official-log-after/result' / (name + '.txt')
        require(sha(trajectory) == audited['saved_after_lidar_before_vio_trajectory']['sha256'], 'Trajectory differs from independent audit')
        timestamps, positions = load_tum(trajectory)
        require(len(timestamps) == audited['saved_after_lidar_before_vio_trajectory']['rows'], 'TUM row count differs from audit')
        selected[mode] = {'row': row, 'run': run, 'trajectory': trajectory,
            'times': timestamps, 'positions': positions, 'audited': audited}
    require(set(selected) == {'livo','lio'}, 'Need exactly one LIVO and one LIO run')
    if args.preview_failed:
        require(not all(value['row']['accepted'] for value in selected.values()), 'Failed preview flag must not hide accepted final results')
    require(csv_file.name == selected['livo']['run'].name + '-paired-events.csv', 'VIO paired CSV must belong to the explicit LIVO run')
    pairs = []
    with csv_file.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['kind'] == 'vio' and row['complete'].lower() == 'true':
                pairs.append([float(Decimal(row['relative_time_s'])), float(row['printed_position_delta_m'])])
    pairs = np.asarray(pairs, dtype=np.float64)
    require(len(pairs) > 0 and np.isfinite(pairs).all() and (pairs[:,1] >= 0).all(), 'Invalid paired internal VIO updates')
    pairing_audit = selected['livo']['audited']['debug_event_pairing']
    expected_pairs = pairing_audit['vio_printed_state_changes']['paired_events']
    require(len(pairs) == expected_pairs, 'VIO paired CSV row count differs from audit')
    require(np.isclose(pairs[:,1].sum(), pairing_audit['vio_printed_state_changes']['position']['sum'], rtol=1e-10, atol=1e-12), 'VIO change sum differs from independent audit')
    require(pairing_audit['all_event_pre_out_timestamps_identical'] and pairing_audit['all_vio_pre_equals_immediately_preceding_lio_out_first19'], 'Paired event phase/time validation missing')
    # Summary is a mutable navigation/index file that may later reference this PNG.
    # Bind immutable selected run/status/TUM and audit inputs, avoiding a hash cycle.
    inputs = [audit_file,csv_file] + [path for value in selected.values()
        for path in [value['trajectory'], value['run']/'status.json']]
    before = {relative(path): sha(path) for path in inputs}
    font = choose_font(args.font_file)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(12.4,8.2))
    grid = fig.add_gridspec(2,2, width_ratios=[1.17,1], height_ratios=[0.86,1],
        left=0.075, right=0.96, bottom=0.22, top=0.78, wspace=0.31, hspace=0.43)
    xy = fig.add_subplot(grid[:,0])
    z = fig.add_subplot(grid[0,1])
    update = fig.add_subplot(grid[1,1])
    for ax in [xy,z,update]:
        style(ax)
    for mode in ['lio','livo']:
        value = selected[mode]
        p, color = value['positions'], COLORS[mode]
        label = 'LIO · LiDAR 后' if mode == 'lio' else 'LIVO · 同轮 VIO 前'
        xy.plot(p[:,0],p[:,1],color=color,lw=1.8,ls='--' if mode=='lio' else '-',label=label)
        xy.scatter(p[0,0],p[0,1],color=color,s=42,marker='o',zorder=4)
        xy.scatter(p[-1,0],p[-1,1],color=color,s=38,marker='s',zorder=4)
    xy.set_aspect('equal', adjustable='datalim')
    xy.margins(0.15)
    xy.set_title('估计坐标中的 XY 轨迹',loc='left',pad=12,color=COLORS['ink'])
    xy.set_xlabel('X (m)')
    xy.set_ylabel('Y (m)')
    xy.legend(loc='upper left',fontsize=10,frameon=True,facecolor='white',edgecolor='#e3e9ee')
    xy.text(0.03,0.03,'● 起点    ■ 终点\n原始估计坐标；未做对齐或缩放',transform=xy.transAxes,
        fontsize=9,color='#516a7d',va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=0.88,pad=5))
    origin = min(value['times'][0] for value in selected.values())
    for mode in ['lio','livo']:
        value = selected[mode]
        elapsed = (value['times']-origin).astype(np.float64)/1e9
        z.plot(elapsed,value['positions'][:,2],lw=1.8,color=COLORS[mode],ls='--' if mode=='lio' else '-')
    z.set_title('Z 位置随传感器时间变化',loc='left',pad=11,color=COLORS['ink'])
    z.set_xlabel('相对首个 TUM 输出时间 (s)')
    z.set_ylabel('Z (m)')
    update.plot(pairs[:,0],pairs[:,1]*1000,lw=0.85,color=COLORS['update'],alpha=0.75)
    update.scatter(pairs[:,0],pairs[:,1]*1000,s=5,color=COLORS['update'],alpha=0.6,rasterized=True)
    update.set_title('VIO 前后内部位置改变量',loc='left',pad=11,color=COLORS['ink'])
    update.set_xlabel('源码内部相对时间 (s)')
    update.set_ylabel('配对位置改变量 (mm)')
    update.set_ylim(bottom=0)
    update.text(0.97,0.94,'%d 对完成事件\n改变量不是误差' % len(pairs),transform=update.transAxes,
        fontsize=9,ha='right',va='top',color='#96511f',bbox=dict(facecolor='white',edgecolor='none',alpha=0.88,pad=4))
    title = 'FAST-LIVO2 · Bright Screen 本机功能回放'
    fig.text(0.075,0.95,title,fontsize=19,weight='bold',color=COLORS['ink'],va='top')
    fig.text(0.075,0.895,'完整 66.7 s 公共数据  /  Ubuntu + ROS Noetic  /  两次均 0.5×  /  无已知 GT',
        fontsize=10.5,color='#557185')
    group_number = group.rsplit('-',1)[-1]
    banner = (group_number + ' 布局预览：两轮收尾未通过；不是验收成功图') if args.preview_failed else (group_number + ' 两轮功能验收通过：包含实际 mapper 与图像桥退出码 0')
    fig.text(0.075,0.843,banner,fontsize=11,weight='bold',
        color='#a33838' if args.preview_failed else '#0f766e')
    counts = 'LIO：%d 个传感器时间位姿    LIVO：%d 个传感器时间位姿' % tuple(len(selected[m]['times']) for m in ['lio','livo'])
    fig.text(0.075,0.12,counts,fontsize=10.5,color=COLORS['ink'])
    fig.text(0.075,0.083,'范围：图左及 Z 曲线均为 LiDAR 更新后、同轮 VIO 前的 IMU-body 位姿；右下为有限精度日志重建的 VIO 内部变化。',fontsize=9,color='#557185')
    patch = summary.get('source_local_patch')
    fig.text(0.075,0.049,('本机编译修复后回放；' if patch and not args.preview_failed else '') +
        '不计算 ATE／漂移精度，不据曲线差异比较优劣；观察者收齐数据不证明估计器无丢帧。',fontsize=9,color='#557185')
    after = {relative(path): sha(path) for path in inputs}
    require(before == after, 'Chart inputs changed during rendering; no final chart saved')
    output = args.output.resolve()
    try:
        output.relative_to(LAB/'outputs')
    except ValueError:
        raise ValueError('Chart output must remain inside lab/outputs')
    output.mkdir(parents=True,exist_ok=True)
    png, metrics_file = output/'trajectory-and-evidence.png', output/'chart-metrics.json'
    require(not png.exists() and not metrics_file.exists(), 'Refusing to overwrite an existing chart or metric receipt')
    fig.savefig(str(png),dpi=160)
    plt.close(fig)
    updates_mm = pairs[:,1]*1000
    metrics = {'status': 'failed_run_layout_preview_only' if args.preview_failed else 'accepted_local_run_chart_generated',
        'group': group, 'selected_runs': {mode: value['row']['run_name'] for mode,value in selected.items()},
        'summary_selection': {'path': relative(summary_file), 'sha256_bound': False,
            'reason': 'Mutable index may reference this chart; immutable selected run/status/TUM and audit files are bound instead.',
            'accepted_at_render': {mode: value['row']['accepted'] for mode,value in selected.items()}},
        'script_sha256': sha(__file__), 'chart_sha256': sha(png), 'chart_bytes': png.stat().st_size,
        'input_sha256_before': before, 'input_sha256_after': after, 'inputs_unchanged': before==after,
        'font_file_name': Path(font).name, 'pose_input': 'TUM integer sensor nanoseconds; ROS odometry is not read',
        'pose_phase': 'IMU body after LiDAR, before same-cycle VIO; no alignment, scaling or origin normalization',
        'pose_rows': {mode: len(value['times']) for mode,value in selected.items()},
        'sensor_time_origin_ns_for_z_panel': int(origin),
        'vio_internal_printed_position_change': {'pairs': len(pairs), 'units': 'mm',
            'positive_pairs': int(np.sum(updates_mm>0)), 'minimum': float(np.min(updates_mm)),
            'median': float(np.median(updates_mm)), 'p95': float(np.percentile(updates_mm,95)),
            'maximum': float(np.max(updates_mm)),
            'time_scope': 'Source debug internal relative time, not ROS clock or TUM epoch',
            'meaning': 'Rounded pre/post internal state changes, not error, residual decrease or accepted-step count'},
        'GT_available': False, 'ATE_computed': False, 'drift_accuracy_computed': False,
        'best_method_claimed': False, 'estimator_zero_drop_claimed': False}
    metrics_file.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':metrics['status'],'png':str(png),'chart_sha256':metrics['chart_sha256']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
