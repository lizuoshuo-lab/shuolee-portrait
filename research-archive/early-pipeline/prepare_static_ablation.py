"""Fresh static ablation inputs; no previously fitted subject model is reused."""
from pathlib import Path
import json, shutil, hashlib, datetime, argparse
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/person-clothing-pipeline-fresh-060'
NAMES={'065':'person-static-no-scale-fresh-065','066':'person-static-no-multiview-fresh-066'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(number):
    R=B/'current'/NAMES[number];assert not R.exists(),R
    R.mkdir();sources=[]
    for rel in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
        src=S/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        sources.append(dict(file=rel,sha256=sha(src),source=str(src),role='Raw observation or official pretrained detector output'))
    stamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    save(R/'source-inputs.json',dict(created=stamp,inputs=sources,from_scratch=True,
        old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
        real_frames=175,initial_points=200000,steps=18000,representation='shared_static_world',
        body_refit_usage='Fresh coarse bounds and scale only. No skeleton, skinning or attraction during Gaussian training.',
        controlled_change='scale weight 100 to 0' if number=='065' else 'multi-view geometry/NCC weights to0; single-view normals unchanged',
        reference_round='064',license_scope='Local PGSR research only; no commercial deployment',quality_accepted=False))
    frozen=R/'source-final';frozen.mkdir()
    for name in ['shuolee_static.py']:
        p=B.parents[2]/'repos/PGSR-shuolee'/name
        shutil.copy2(p,frozen/name)
    save(frozen/'manifest.json',dict(created=stamp,files={p.name:sha(p) for p in frozen.iterdir() if p.is_file()}))
    st=read(B/'work-status.json');st['updated']=stamp;st['local_training_running']=False
    st['current_phase']='static_surface_controlled_ablation'
    st['current_decision']='064的袖口衣摆局部改善，但鞋底和仰视未通过；继续静态单项约束对照。'
    st['latest_local_execution']=dict(kind='static_ablation_'+number,state='preparing_fresh_inputs',running=False,directory=str(R),
        from_scratch=True,representation='shared_static_world',quality_accepted=False,previous_completed_delivery='064')
    save(B/'work-status.json',st)
    p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8-sig')
    marker='<!-- END_GVHMR_059_ACTIVE -->'
    text=f'\n当前{number}受控对照已开始准备：'+('只将最小尺度压缩权重100改为0。' if number=='065' else '只将多视图几何及NCC权重改为0，单视图法线项保持。')+'其余175帧、相机、1536高度、增密方式、学习率、随机种子和18000步一致。所有先验拟合及20万点初始化重算；保留064用于只读对照。两项被关闭时仍执行相同采样代码，避免意外改变帧顺序。未完成和未通过质量前不晋级。\n\n'
    t=t.replace(marker,text+marker,1);p.write_text(t,encoding='utf8')
    print(str(R))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('number',choices=NAMES);main(p.parse_args().number)
