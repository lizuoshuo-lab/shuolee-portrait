"""Freeze a single processed-video residual-distortion test before extracting lines."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'residual-distortion-direction'; C=R/'native-continuity'
DATA=Path('D:/GS-SLAM-Data/robust122-whiteT-native-supervision')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'plan.json').exists()
r=read(C/'result.json'); assert r['state']=='complete_pending_material_review'
idx=read(C/'review-inputs/index.json')
save(C/'review.json',dict(updated=stamp,actual_review_complete=True,contact_pages=13,track_rows=87,
  face_anchor_strips=6,largest_disagreement_strips=8,all_images_native_unannotated=True,
  findings_zh=[
  '已审阅87条轨迹的13页原生裁片、6组面部旧检测/逐帧跟踪对照、8组三列最大分歧裁片。原生裁片仅用于材料审阅，不以目视代替亚像素测量。',
  '第264帧短裤family974：旧SIFT落在另一块印花，逐帧跟踪仍在种子附近的原印花区域；两者相差168.793px。保留两个原始结果，不依据误差大小重新分配拟合/留出。',
  '同一帧两种观测落在不同印花，不可能单靠一组共享镜头畸变参数使两者成为同一物理点。该例支持对应关系存在问题，不排除同时存在残余畸变。',
  '人脸6组在粗解剖区域上连续，但眼睑、胡须与头发局部变化及定位差异仍不能证明每个观测的物理误差小于1像素。',
  '场景地面纹理与身体边缘发生接近/遮挡，楼顶重复结构及画面边缘也存在支持限制。数值双向一致不自动等于物理身份通过。',
  '464帧每帧至少10个场景拟合候选通过数值检查；这不是全部464帧相机注册或训练准入。'],
  same_frame_material_contradiction=dict(family=974,frame=264,old_sift_uv=[869.8606567382812,1891.531982421875],
    propagated_uv=[727.8739013671875,1800.2581787109375],disagreement_px=168.79320467564318,
    evidence='review-inputs/disagreement_00_family_974_frame_000264.png',
    geometry_fit_used=False),
  physical_admission=False,training=False,source_splits_unchanged=True))

lines=[]
def line(name,frame,group,split,a,b,description,exclude=None):
 lines.append(dict(id=name,frame=frame,physical_group=group,split=split,a=a,b=b,
                   description=description,exclude_fraction=exclude or []))
# The full architecture and native crops were inspected before these seeds were frozen.
# Groups A/C are fitting structures; B/D are wholly held-out buildings, not nearby duplicate lines.
line('A1',0,'near_apartment_A','fit',[828,10],[811,631],'Opaque facade side against narrow dark recess')
line('A2',0,'near_apartment_A','fit',[1466,12],[1413,551],'Right opaque facade border, avoid pipe lower junction')
line('A3',0,'near_apartment_A','fit',[850,351],[1406,365],'Continuous opaque upper window-frame border; exclude opening windows',[[.19,.47]])
line('A4',0,'near_apartment_A','fit',[853,568],[1390,575],'Opaque lower window-frame border, not reflected content')
line('A5',0,'near_apartment_A','fit',[370,717],[1390,733],'Canopy upper rigid edge; head overlap excluded before measurement',[[.34,.58]])
line('C1',463,'left_tower_C','fit',[489,60],[507,526],'Outer tower-to-sky rigid edge above lower adjoining building')
line('C2',463,'left_tower_C','fit',[551,18],[576,818],'Continuous gray/red facade junction; thin panel crossings retained')
line('C3',463,'left_tower_C','fit',[644,20],[658,740],'Continuous cream pillar right border, above person/tree overlap')
line('B1',232,'office_B','heldout',[1022,367],[1501,208],'Outer roof-to-sky straight edge, leaves at left excluded',[[0,.08]])
line('B2',232,'office_B','heldout',[1080,395],[1383,307],'Upper rigid balcony edge before facade step')
line('B3',232,'office_B','heldout',[974,493],[962,787],'Left opaque building-to-sky edge below branches')
line('D1',463,'right_tower_D','heldout',[1163,12],[1128,707],'Outer tower-to-sky edge above trees')
line('D2',463,'right_tower_D','heldout',[1223,15],[1174,710],'Continuous gray/red facade junction above trees')

plan=dict(stage='D1_residual_lines',updated=stamp,user_authorization='怎么没有改善，就延这个方向试一下',
 hypothesis='A common small nonlinear residual mapping remains in the already rectified/cropped MP4. Test with physical straight architecture, independently of pose and person tracks.',
 previous_negative='N6 fitted one radial coefficient with 7-view geometry on winter-hat clip; it does not exhaust residual models or this white-T clip.',
 controlled_change='One common inverse Brown4 processed-output mapping (k1,k2,p1,p2). K and all old camera/3D/track states remain immutable. No image resampling.',
 model=dict(name='processed_output_inverse_brown4',input='observed native MP4 integer-center normalized coordinates x=(u-cx)/fx,y=(v-cy)/fy',
   formula=['xu=x*(1+k1*r2+k2*r2*r2)+2*p1*x*y+p2*(r2+2*x*x)',
            'yu=y*(1+k1*r2+k2*r2*r2)+p1*(r2+2*y*y)+2*p2*x*y'],
   output='ideal ray [xu,yu,1]; forward rendering must invert this exact mapping, never treat these coefficients as raw FISHEYE_8 or OpenCV forward Brown.',
   K=[[1494.4425475491018,0,761.3168340952368],[0,1493.0941798544786,1357.2026900985354],[0,0,1]],
   K_source='unchanged primary first-YAML4000x3000 conditional resize/centercrop',width=1520,height=2704),
 extraction=dict(frames=[0,232,463],native_gray='cv2 IMREAD_COLOR then BGR2GRAY',
   samples_per_line=120,normal_search_half_width_px=8,normal_step_px=.25,
   tangent_average_offsets_px=[-2,-1,0,1,2],method='signed grayscale normal derivative; dominant sign per line; quadratic sub-sample peak',
   weak_gradient_limit_gray_per_px=3,finite_and_margin_check=True,
   ambiguity='Reject profile if a separated secondary same-sign peak >=0.9 times primary; do not use curvature/model residual for selection.',
   no_after_fit_point_removal=True,min_valid_fraction=.65,min_samples=50),
 fit=dict(max_nfev=150,initial=[0,0,0,0],bounds=[.05,.05,.02,.02],loss='soft_l1',f_scale_px=.5,
   line_nuisance='TLS straight line recomputed per line, equal total weight per physical line',
   CPU_only=True,max_seconds=180),
 acceptance=dict(min_fit_lines=6,min_heldout_lines=4,min_fit_groups=2,min_heldout_groups=2,
   scaled_jacobian_rank=4,max_scaled_condition=1000,
   heldout_rms_improvement_fraction=.2,heldout_max_rms_worsening_px=.15,
   max_full_frame_displacement_px=32,min_jacobian_determinant=.5,
   face='Separate independent physical face <1px validation remains mandatory; line fit alone never admits training.',
   coverage='Architectural evidence is upper-image only. Do not claim feet/lower-image correction or per-frame time-varying warp from it.'),
 stop_conditions=['No coefficient sweep, alternate models or new feature extraction under this registration.',
 'Insufficient reviewed line support or nonidentifiability: save evidence and do not promote.',
 'Any held-out/model-domain guard fails: keep candidate diagnostic; never overwrite old K or camera/preview.'],
 line_seeds=lines,
 input_hashes={f'frame_{f:06d}.png':sha(DATA/'rgb'/f'frame_{f:06d}.png') for f in [0,232,463]},
 no_person_data_upload=True,no_GPU=True,no_training=True)
save(O/'plan.json',plan)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='D1_residual_line_measurement_pre_registered',running=None,
 next='Execute one registered native straight-edge measurement and four-parameter residual-output fit; inspect held-out structures and field identifiability. T6 complete, never rerun.',
 training_admitted=False,training_started=False,quality_accepted=False)
q['completed'].append('T6 once-only all464 adjacent-native-frame diagnostic complete;87families/13pages/6face strips/8disagreement strips actuallyreviewed. Same-frame shorts family974 physical mismatch documented; no new camera or training.')
save(R/'queue-state.json',q)
p=read(R/'protocol.json');p['latest_user_steering']=dict(updated=stamp,statement='Continue residual-distortion compensation direction after previous one-parameter failure; new physical straight-line diagnostic preregistered.',plan=str(O/'plan.json'))
p['next']=q['next'];save(R/'protocol.json',p)
print(json.dumps(dict(plan=str(O/'plan.json'),lines=len(lines),fit=8,heldout=5),ensure_ascii=False))
