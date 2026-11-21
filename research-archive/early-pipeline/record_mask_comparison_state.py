"""Record task-owned status and the verified QQ delivery."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

LOCAL=Path('C:/Users/Administrator/Documents/New project')
ROOT=Path('D:/00/记录/2026-11-04_MSV2_mask原始_插值2倍_4倍对比')
FOUR=Path('D:/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,data):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

now=datetime.now(timezone.utc).isoformat()
bundle=FOUR/'mask_penalty/demo_mask惩罚_MSV2_18000步_世界坐标点云与相机.zip'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='51bd3b0e32625b58b6ea8f2f7c6baa1e52245c7afe48fe66d1100ee0057e12b5'
save(FOUR/'delivery-to-xiaorun.json',dict(verified_at=now,recipient='小润',channel='QQ',
    file=str(bundle),sha256='51bd3b0e32625b58b6ea8f2f7c6baa1e52245c7afe48fe66d1100ee0057e12b5',
    bytes=bundle.stat().st_size,sent_local='2026-11-04 17:40 Asia/Shanghai',
    recipient_receipt_local='2026-11-04 17:42 Asia/Shanghai',
    receiver_confirmation_observed=True,
    receipt_text='对方已成功接收文件“demo_mask惩罚_MSV2_18000步_世界坐标点云与相机.zip”',
    experiment_limitations_message_sent=True,original_chat_draft_restored=True))
status=read(LOCAL/'shuolee-coordination/company00-status.json')
audit=read(ROOT/'原视频到训练像素核验.json');assert audit['passed']
result={}
for label,run in [('original','msv2-demo-boundary-mask-native-20261104'),('interpolation2x','msv2-demo-boundary-mask-interpolate2x-20261104')]:
    p=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab')/run
    result[label]=read(p/'execution.json');assert result[label]['state']=='training_completed'
status['msv2_mask_1x2x4x_comparison']=dict(updated=now,state='training_complete_review_in_progress',
    record=str(ROOT),same_mask_penalty=True,model_source_unchanged=True,original_pixels_verified=True,
    downsample_before_enlargement=False,training_completed=result,own_gpu_training_jobs_running=False,
    review_in_progress=True,quality_accepted=False,mask4x_sent_to_xiaorun=True,
    original_video_wh=[2704,1520],native_integer_crop_wh=[476,960],
    training_sizes=[[476,960],[952,1920],[1904,3840]])
save(LOCAL/'shuolee-coordination/company00-status.json',status)
print('MASK_DELIVERY_AND_TRAINING_STATUS_RECORDED')
