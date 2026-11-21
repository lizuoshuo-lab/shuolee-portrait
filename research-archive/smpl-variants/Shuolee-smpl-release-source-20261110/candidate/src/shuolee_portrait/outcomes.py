"""User-facing terminal outcomes, independent of CUDA and process exit status."""
import json
from pathlib import Path


def read_optional(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def failure_feedback(error='', *, stage=None, exit_code=None, interrupted=False):
    """Classify observed evidence; a killed process alone does not prove OOM."""
    text = str(error).lower()
    outcome, code = 'system_error', 'PROCESSING_ERROR'
    message = '处理未能完成，原视频和已有计算结果已保留。'
    actions = ['根据任务日志排查处理程序后重试。']
    if interrupted:
        outcome, code = 'interrupted', 'PROCESS_INTERRUPTED'
        message = '处理已中断，尚未生成完整模型。'
        actions = ['恢复处理服务后由运维重启任务；原视频已保留。']
    elif exit_code is not None and exit_code < 0:
        code = 'PROCESS_TERMINATED'
        message = f'处理进程被信号 {-exit_code} 终止，尚未生成完整模型。'
        actions = ['检查服务器系统日志和资源使用情况；仅凭退出信号不能确定是否内存不足。']
    elif 'out of memory' in text:
        code = 'MEMORY_EXHAUSTED'
        message = '服务器可用内存或显存不足，本次处理未完成。'
        actions = ['释放服务器资源后重试。']
    elif any(x in text for x in ('time budget', 'declared time budget')):
        code = 'CAMERA_TIME_LIMIT'
        message = '相机求解超时，本次未生成完整模型。'
        actions = ['检查服务器负载和相机诊断；可缩短视频、减少重复片段后重试。']
    elif any(x in text for x in ('missing real camera', 'camera coverage', 'verified fitting observations',
                                'seed pair', 'seed cheirality', 'essential matrix', 'holdouts',
                                'fresh texture', 'fresh raw texture', 'focal solve', 'focal is locally',
                                'focal estimation requires')):
        outcome, code = 'needs_recapture', 'CAMERA_NOT_RECOVERABLE'
        message = '现有画面未能恢复足够连续的相机位姿，尚未生成三维模型。'
        actions = ['让人物保持姿势，相机缓慢绕拍，保留有纹理的静态背景。',
                   '避免模糊、长时间遮挡和突然跳转；固定相机拍人物转身不适用于当前静态重建流程。']
    elif any(x in text for x in ('no person detected', 'multiple similarly sized people',
                                'person track ambiguous', 'person mask failed',
                                'reliable body landmarks', 'body fit failed', 'common silhouette volume')):
        outcome, code = 'needs_recapture', 'SUBJECT_NOT_RECOVERABLE'
        message = '未能得到稳定的人物观测或人体对齐，本次未生成完整模型。'
        actions = ['画面中保留一个完整人物，保持姿势，改善光照并减少遮挡。']
    elif any(x in text for x in ('hdr', 'rotation metadata', 'video dimensions', 'video timestamps',
                                'selected frames outside', 'expected exactly one selected video',
                                'changes image dimensions', 'decoded frame identity')):
        outcome, code = 'needs_recapture', 'VIDEO_INPUT_UNSUPPORTED'
        message = '视频格式、时序或有效长度不符合当前重建要求。'
        actions = ['使用方向已写入画面的普通 SDR 视频，保持固定分辨率和连续时间轴。']
    elif any(x in text for x in ('asset check failed', 'checkpoint checksum', 'cuda gpu unavailable',
                                'modulenotfounderror', 'no module named', 'missing ffmpeg', 'missing ffprobe')):
        code = 'ENVIRONMENT_NOT_READY'
        message = '服务器依赖或模型尚未准备好，本次处理未完成。'
        actions = ['重新运行部署命令，核对服务就绪后重试。']
    elif any(x in text for x in ('ffprobe', 'ffmpeg', 'unreadable new frame')):
        outcome, code = 'needs_recapture', 'VIDEO_DECODE_ERROR'
        message = '未能完整读取视频画面。'
        actions = ['检查视频能否从头播放到尾，重新导出完整视频后上传。']
    elif any(x in text for x in ('nonfinite', 'non-finite', 'point budget', 'no visible points',
                                'proposal boundary', 'render parity', 'native geometry', 'incomplete standard')):
        code = 'RESULT_VALIDATION_ERROR'
        message = '重建的数值或导出一致性检查未通过，尚无可交付模型。'
        actions = ['保留运行数据，检查拟合、训练或导出诊断后重试。']
    elif exit_code == 0:
        code = 'INCOMPLETE_RESULT'
        message = '处理进程已退出，但未产生通过完整性检查的重建结果。'
        actions = ['检查训练步数和导出日志后重试。']
    return dict(outcome=outcome, code=code, message=message, actions=actions,
                model_available=False, stage=stage)


def job_outcome(job, completed=None):
    root = Path(job['run'])
    status = read_optional(root / 'status.json')
    preparation = read_optional(root / 'preparation.json')
    if completed is not None:
        feedback = dict(outcome='completed', code='MODEL_READY', message='重建完成，模型已导出。',
                        actions=[], model_available=True, stage='export',
                        model=str(root / 'export/point_cloud.ply'))
        selection = read_optional(root / 'camera/frame-selection.json')
        dropped = selection.get('dropped', [])
        feedback['skipped_sampled_frames'] = len(dropped)
        if dropped:
            feedback['message'] = f'已跳过 {len(dropped)} 个不可用采样帧，重建完成，模型已导出。'
    else:
        error = str(status.get('error') or preparation.get('error') or job.get('error') or '')
        # Native crashes may have no run status; read only a bounded log tail.
        try:
            with Path(job.get('log') or root / 'logs/job.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16384))
                error += '\n' + stream.read().decode('utf-8', errors='replace')
        except OSError:
            pass
        feedback = failure_feedback(error, stage=status.get('stage', 'prepare'),
            exit_code=job.get('exit_code'), interrupted=job.get('state') == 'interrupted')
    return dict(schema='smpl-x-outcome-v1', job_id=job['id'], video=job.get('video', job.get('input')),
                run=job['run'], log=job.get('log'), exit_code=job.get('exit_code'),
                quality_accepted=False, **feedback)
