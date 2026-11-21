from pathlib import Path
import json,shutil,hashlib,datetime
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-static-calibration-037'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
audit=read(OUT/'independent-calibration-audit.json');adapter=read(OUT/'projection-adapter-audit.json')
assert len(audit['calibration_checks'])==54 and len(audit['nearby_checks'])==45 and len(adapter['records'])==15
save(OUT/'final-summary.json',dict(status='static_projection_candidate_validated_not_trained',
    source_fit='All18cameras static background correspondences atframe700; spatial checks at700, otherimages710/720; no frame750appearanceusedtofit.',
    validation=audit['aggregate'],nearby_validation=audit['nearby_median_of15_camera_medians_px'],
    contract='15realtrainingbatches: decodedtarget/sourceRGBandmatteunchanged, originalR/Tunchanged, onlyprojectionprincipalpointoffsetonceafterundistortion/cropping.',
    extra_data_qualification='00/06/12calibrationphotos700/710/720accessed onlyforstaticregistration; they remainexcludedfromforegroundtexturetraining.',
    decision='Keep035baselineunchanged. Afterfinal10000visualreview, usecandidateinmatchedtrainingonlyifclaritystillfails. Do notclaimphysicalcalibrationorhumanqualityfixed.',goal_complete=False))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
names=['calibrate_enerf_static_pixel_offsets.py','audit_enerf_static_offsets.py','fourkd_projection_offsets.py','probe_4k4d_projection_offsets.py','run_4k4d_projection_offsets_probe.sh','finalize_enerf_static_calibration.py']
entries=[]
for n in names:
    shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n);entries.append(dict(file=n,sha256=hashlib.sha256((snap/n).read_bytes()).hexdigest()))
save(OUT/'source-snapshot.json',dict(files=entries))
near=LAB/'outputs/dynamic-human-enerf-static-camera-036'
shutil.copy2(LAB/'scripts/dynamic_human/audit_enerf_static_cameras.py',near/'audit_enerf_static_cameras.py')
page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
anchor='### 官方 4K4D 的原生单帧人物验证'
addition='''进一步检查全部 15 个训练相机，固定第 750 帧的部分空间区域拟合、其余区域检查，并在第 740、760 帧验证。第 750 帧各相机检查误差中位数的中位数为 2.17 像素，PnP 候选降到 0.40 像素；仅拟合每个相机的二维像素平移，也能降到 0.49 像素。因此静态纹理错位具有可重复性，但它还不能唯一确定为物理相机位姿变化。[全部训练相机的几何检查](../outputs/dynamic-human-enerf-static-camera-036/independent-audit-and-offset-comparison.json)。

随后使用另一段时间的静态特征拟合更简单的投影偏移：仅在第 700 帧拟合，第 710、720 帧检查。18 个相机的检查误差中位数汇总分别由 1.94、1.89 像素降到约 0.40 像素；将同一组偏移直接用于原有第 740、750、760 帧的静态对应，15 个训练相机的三维重投影误差汇总由约 2.15～2.24 像素降到 0.65～0.67 像素。这次额外读取了外观留出相机 00、06、12 在第 700、710、720 帧的静态标定照片，不能再称为完全没有访问这些相机的 RGB；它们没有进入人体外观训练，也没有用第 750 帧的人物误差拟合偏移。候选只在原图去畸变和裁剪后调整投影主点，15 个真实训练批次的图像、透明度与外参保持逐元素不变。**当前模型仍使用原相机；人物清晰度是否改善，需要下一组配对训练验证。** [时间与数据划分](../outputs/dynamic-human-enerf-static-calibration-037/protocol.json)、[独立复算](../outputs/dynamic-human-enerf-static-calibration-037/independent-calibration-audit.json)、[投影接口检查](../outputs/dynamic-human-enerf-static-calibration-037/projection-adapter-audit.json)。

'''
if '全部训练相机的几何检查' not in text:
    assert text.count(anchor)==1
    page.write_text(text.replace(anchor,addition+anchor),encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
if 'outputs/dynamic-human-enerf-static-calibration-037/${name}' not in text:
    lines=text.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'outputs/dynamic-human-4k4d-official-035/${name}' in line)+1
    lines.insert(i,"  ['outputs/dynamic-human-enerf-static-camera-036/independent-audit-and-offset-comparison.json', '15训练相机静态投影检查；未修改模型'],\n  ...['protocol.json', 'independent-calibration-audit.json', 'projection-adapter-audit.json'].map(name => [`outputs/dynamic-human-enerf-static-calibration-037/${name}`, '另段时间静态标定与投影接口核验；尚未用于人体训练']),\n")
    sync.write_text(''.join(lines),encoding='utf8')
mf=LAB/'manifests/dynamic-human-work.json';m=read(mf)
m['outdoor_static_camera_diagnosis']='outputs/dynamic-human-enerf-static-camera-036/independent-audit-and-offset-comparison.json'
m['outdoor_static_projection_candidate']='outputs/dynamic-human-enerf-static-calibration-037/final-summary.json'
m['goal_turn_classification']='progress:036all15trainingcameras45staticchecks;037separate-time18cameracalibration54checksplus45nearbychecksverified;15actualdecodedbatchesprojection-onlyinterventionpassed;035trainingliveandunchanged.'
m['next_actions']=[
    'Observe82968and20504already-runningtraining/evaluationqueue; do not startcompetingGPUtraining or restartfromstatefilesalone.',
    'After10000historyfinalrendering, inspect6pairedpersonpanelsand3fullframes. Do notacceptbasedonlyonPSNR.',
    'Ifsingle-frameclaritystillfails, conductsame-parent/same-budget original-vs037projectionoffsettrainingcontrol, explicitlyreportextraall18camera staticcalibrationphotographs. OriginaldecodeK/D/pixelsremainunchanged.',
    'Onlyafterclearfit, extendwalking/occlusionandtemporalevaluation;completebackground/personcompositionandOmniReinterface. Pixelregistrationdoesnotprovesynchronization.'
]
line='036/037:All15traincameras staticmisregistrationreproduced;18cameraoffsetsfitat700andcheckedat710/720,45nearbychecksverified.15realtrainingbatches pixelsandposesunchangedbyprojection-onlyadapter. Candidateunapplied.'
if line not in m['completed']:m['completed'].append(line)
live=read(LAB/'outputs/dynamic-human-4k4d-official-035/continuation-10000/training-live.json')
m['current_runs'][0]['last_observed_new_updates']=live['completed_updates_this_process'];m['current_runs'][0]['last_observed_history_updates']=live['history_updates']
m['last_recorded_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(mf,m)
print(json.dumps(dict(calibration_status='validated_not_applied',training_history=live['history_updates'],new_public_resources=4)))
