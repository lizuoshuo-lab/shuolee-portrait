"""Read-only decode verification for the two newly available source videos."""
from pathlib import Path
import datetime,hashlib,json,subprocess,time
base=Path('/mnt/d/Projects/Shuolee-analysis-20261028');dataset=Path('/mnt/d/00/工作/数据集')
records=[]
for name in ['video9.mp4','video10.mp4']:
    path=dataset/name;before=hashlib.sha256(path.read_bytes()).hexdigest()
    probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],check=True,capture_output=True,text=True)
    started=time.monotonic()
    decode=subprocess.run(['ffmpeg','-hide_banner','-v','error','-xerror','-threads','4','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
    after=hashlib.sha256(path.read_bytes()).hexdigest();assert before==after
    item=dict(name=name,path=str(path),sha256=before,probe=json.loads(probe.stdout),decode_exit_code=decode.returncode,
        decode_errors=decode.stderr,decode_seconds=time.monotonic()-started,source_unchanged=True,training_started=False)
    records.append(item);print(json.dumps({k:item[k] for k in ['name','sha256','decode_exit_code','decode_errors','decode_seconds']}),flush=True)
report=dict(checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),videos=records,all_full_decodes_passed=all(r['decode_exit_code']==0 for r in records),quality_accepted=False)
(base/'arrived-videos-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
manifest_path=Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图/清单.json')
manifest=json.loads(manifest_path.read_text());current=dataset/'video3.mp4'
assert hashlib.sha256(current.read_bytes()).hexdigest()==manifest['source_video_sha256']
manifest['source_video_current_path']='D:/00/工作/数据集/video3.mp4'
manifest['source_video_path_note']='Source was renamed in the shared dataset after extraction; bytes verified unchanged.'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
