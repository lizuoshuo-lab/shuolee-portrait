"""Read only known pretrained assets on CPU. No subject predictions or download."""
from pathlib import Path
import ast, datetime as dt, gc, hashlib, importlib.util, json, os, sys, time, zipfile
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
H=BASE/'current/portrait-alignment-open-source-116/hsfm'
REPO=LAB/'repos/GVHMR-shuolee'
PRIVATE=BASE/'current/novel-view-face/head-structure/private-runtime/python-deps'
sys.path[:0]=[str(REPO),str(PRIVATE),str(PRIVATE/'smplx-official')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir(REPO)
def save(name,data):
    p=H/name; p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not (H/'asset-compatibility.json').exists()
    start=time.monotonic();result=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
        gpu=False,weight_download=False,prior_subject_loaded=False,subject_inference=False,findings={})
    import torch
    torch.set_num_threads(2)
    from hmr4d.network.hmr2 import HMR2
    from hmr4d.network.hmr2.configs import get_config
    weight=Path('/mnt/d/GS-SLAM-Data/models/gvhmr/hmr2/epoch=10-step=25000.ckpt')
    assert sha(weight)=='2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'
    cfgpath=REPO/'hmr4d/network/hmr2/configs/model_config.yaml'
    cfg=get_config(str(cfgpath));cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256];cfg.freeze()
    # The official ViT constructor calls .item() on its drop-path schedule;
    # ordinary CPU construction is required, with no upstream monkey-patching.
    model=HMR2(cfg)
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    state={k:v for k,v in checkpoint['state_dict'].items() if k.split('.')[0] in ['backbone','smpl_head']}
    expected=model.state_dict()
    assert set(expected)==set(state)
    assert all(expected[k].shape==v.shape for k,v in state.items())
    loaded=model.load_state_dict(state,strict=True,assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    result['findings']['hmr2']=dict(state='strict_network_schema_passed_cpu_only',path=str(weight),
        sha256='2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e',
        matched_network_tensors=len(state),params=sum(v.numel() for v in state.values()),
        config_sha256=sha(cfgpath),implementation='Existing GVHMR full-head HMR2 adaptation',
        difference='HMR2.0a epoch10 local checkpoint; HSfM default is epoch35 HMR2.0b. Must label initializer substitution.',
        forward_required='Full head feat_mode=False; do not use feature-only token output',
        output_conversion='axis-angle from adapted model -> rotation matrices expected by HSfM',
        shape_pose_accuracy_accepted=False)
    del checkpoint,state,expected,model;gc.collect()
    deps={x:bool(importlib.util.find_spec(x)) for x in ['onnxruntime','rtmlib']}
    result['findings']['rtmw_deps']=deps
    if all(deps.values()):
        import onnxruntime as ort
        from rtmlib.visualization.skeleton.coco133 import coco133
        path=Path('/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        inp=[dict(name=x.name,shape=x.shape,type=x.type) for x in sess.get_inputs()]
        out=[dict(name=x.name,shape=x.shape,type=x.type) for x in sess.get_outputs()]
        assert inp[0]['shape'][-3:]==[3,384,288]
        assert [r['shape'][-2:] for r in out]==[[133,576],[133,768]]
        names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
        tree=ast.parse((H/'source/joint_names.py').read_text()); source_names={}
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
                for target in node.targets:
                    if isinstance(target,ast.Name):source_names[target.id]=ast.literal_eval(node.value)
        assert names[:23]==source_names['COCO_WHOLEBODY_KEYPOINTS'][:23]
        smpl_names=source_names['SMPL_45_KEYPOINTS']
        assert all(n in smpl_names for n in names[:23])
        with zipfile.ZipFile(path.with_suffix('.zip')) as z: pipeline=json.loads(z.read('pipeline.json'))
        transforms=pipeline['pipeline']['tasks'][0]['transforms']
        result['findings']['rtmw']=dict(state='cpu_graph_and_body23_schema_verified',path=str(path),sha256=sha(path),
            providers=sess.get_providers(),inputs=inp,outputs=out,body23_names=names[:23],
            normalization=[v for v in transforms if v['type']=='Normalize'],
            detector_substitution=True,official_hsfm_detector='ViTPose wholebody',
            fresh_detections_pending=True,confidence_is_calibrated_probability=False,
            full_face133_mapping_verified=False)
    else:
        result['findings']['rtmw']=dict(state='known_model_present_runtime_unavailable_in_current_env')
    result.update(state='known_assets_reviewed_no_subject_inference',seconds=time.monotonic()-start,
                  new_camera=False,gaussian_training=False)
    save('asset-compatibility.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
