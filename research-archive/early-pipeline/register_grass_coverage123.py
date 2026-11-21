"""Record actual source review and bind the existing coverage algorithm to a manifest."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'grass-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();r=read(PRE/'result.json');assert r['selected_frames']==94
assert not (PRE/'coverage-plan.json').exists()
notes=[
 '0000：左向侧脸，头发与自遮挡限制远侧脸部观察。全身与鞋在画内。背景包含树木、灯杆、路面和其他人，不能把全部背景标成静止。',
 '0115：近正面偏侧，近侧眼鼻嘴可见，一部分面部受头发遮挡。双脚鞋在画内。远景有人，草叶树叶和重复树干需分别核验。',
 '0235：近正面眼鼻嘴有图像信息，领口与持发手互相遮挡。双脚和鞋可见，裙摆遮挡上腿。没有独立相机或三维真值。',
 '0350：右向近侧脸可见，远侧脸自遮挡。手和衣袖、头发相交；脚在画内但鞋的背面不可见。',
 '0465：侧后方，头发遮住面部，不能生成有效眼鼻嘴真值。人物背部与侧身可见；仅这些视角不能证明人体或背景静止。']
save(PRE/'visual-review.json',dict(updated=stamp,actual_full_frames_reviewed=5,original_wh=[2704,1520],tool_display_wh=[2048,1151],
    native_pixel_precision_review=False,rows=[dict(**im,note=note) for im,note in zip(r['images'],notes)],
    physical_landmark_labels_provided=False,full94_visibility_not_yet_reviewed=True,physical_staticity_certified=False,training=False,quality_accepted=False))
save(PRE/'decoder-policy.json',dict(updated=stamp,shared_frames=5,PTS_match=True,all_pixel_exact=False,
    mean_channel_difference_range=[min(x['mean_channel_difference'] for x in r['shared_historical_frame_checks']),max(x['mean_channel_difference'] for x in r['shared_historical_frame_checks'])],
    action='Use only current FFmpeg RGB rasters for new observations. Do not import old121 feature coordinates across different RGB decodes. No image orK transform applied.',
    physical_distortion_or_translation_not_inferred=True))
src=H/'indoor_coverage123.py';s=src.read_text(encoding='utf-8');original=s
start=s.index("TOP=Path(");end=s.index("sys.path.insert(0,str(PROJECT/'src'))")
s=s[:start]+'''import argparse
parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
PLAN_PATH=Path(args.plan)
SETTINGS=json.loads(PLAN_PATH.read_text(encoding='utf-8-sig'))
PRE=PLAN_PATH.parent;OUT=PRE/'coverage';ROOT=Path(SETTINGS['run_root'])
PROJECT=Path(SETTINGS['project']);ASSETS=Path(SETTINGS['assets'])
N=SETTINGS['selected_frame_count'];WH=SETTINGS['native_wh']
'''+s[end:]
s=s.replace('177','N').replace("f'coverage {slot+1}/N selection=", "f'coverage {slot+1}/{N} selection=")
s=s.replace("plan=read(PRE/'coverage-plan.json')","plan=read(PLAN_PATH)")
s=s.replace("r['wh']==[2704,1520]","r['wh']==WH")
s=s.replace("previous_goal_turn='progress: indoorNnativeframes prepared; five native originals reviewed; four historical decoder pairs checked without changing pixels'", "previous_goal_turn=SETTINGS['previous_goal_turn']")
dest=H/'coverage_registered_native123.py';assert not dest.exists();ast.parse(s);dest.write_text(s,encoding='utf-8',newline='\n')
# Exact main-body proof after normalizing only roster/path/display metadata.
newmain=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='main')
expected=original.replace('177','N').replace("f'coverage {slot+1}/N selection=", "f'coverage {slot+1}/{N} selection=")
expected=expected.replace("plan=read(PRE/'coverage-plan.json')","plan=read(PLAN_PATH)").replace("r['wh']==[2704,1520]","r['wh']==WH")
expected=expected.replace("previous_goal_turn='progress: indoorNnativeframes prepared; five native originals reviewed; four historical decoder pairs checked without changing pixels'", "previous_goal_turn=SETTINGS['previous_goal_turn']")
oldmain=next(x for x in ast.parse(expected).body if isinstance(x,ast.FunctionDef) and x.name=='main')
assert ast.dump(newmain,include_attributes=False)==ast.dump(oldmain,include_attributes=False)
plan=dict(registered=stamp,subject='grass',input_sha256=r['source_sha256'],manifest_sha256=sha(PRE/'decoded.json'),
    source_frame_count=470,selected_frame_count=94,selected_source_indices=list(range(0,470,5)),native_wh=[2704,1520],
    run_root=r['run_root'],project='/opt/gs-slam-lab/shuolee-local-surface-123/runtime',assets='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',
    previous_goal_turn='progress: full94grassnativeframes decoded, fixedfive originals reviewed; old24observations not reused across decoderRGBdifference',
    purpose='First complete registered grass94 region/landmark inventory; close missing seventh-person observation stream before camera or fitting decisions.',
    new_information='89selected source frames missing from historical24. No old camera or oldindividualgeometry reused.',
    source_review=str(PRE/'visual-review.json'),method='Same YOLOv8n/RTMW-L model, thresholds, subject selection and inverseROI mapping as indoor177; only manifestpaths/raster/framecount differ.',
    script=str(dest),script_sha256=sha(dest),reused_algorithm=str(src),reused_algorithm_sha256=sha(src),
    main_AST_unchanged_after_only_roster_parameterization=True,
    budget=dict(frames=94,passes=1,max_wall_seconds=1200,GPU_jobs=1,new_downloads=0,feature_matching=0,cameras=0,SMPLX=0,training=0),
    review_slots=[0,23,47,70,93],
    stop='No repeated extraction. Source/model/raster/mapping failure or budgetexhaustion stops. Ambiguousperson stays explicit. Confidence never establishes visibility or subpixeltruth.',
    missing_flip_measurements_remain_unknown=True,training_allowed=False)
save(PRE/'coverage-plan.json',plan)
print(json.dumps(dict(registered=True,frames=94,algorithm_parameters_unchanged=True,script=str(dest)),ensure_ascii=False))
