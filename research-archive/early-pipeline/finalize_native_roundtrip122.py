"""Record actual permitted browser checks and register the next readonly diagnostic."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, re, hashlib, shutil

T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'covariance-sh-ply';R121=T/'shuolee/current/raw-domain-121'
V=T/'viewer';S=Path(r'D:\00\工作\查看器');L=Path(r'D:\00\工作日志\2026-10-27\内参文件解析')
C=Path(__file__).parent.parent/'shuolee-coordination'
stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['report-ui.png','viewer-preview-ui.png','native-guard-ui.png']:assert (O/n).is_file()
save(R/'viewer-ui-review.json',{
    'updated':stamp,'version':'2026.10.27.5','browser':'Existing Codex in-app browser',
    'report_tab':'40','preview_tab':'31','historical_tab15_untouched':True,
    'report_updated_and_actually_viewed':True,'report_url':'http://127.0.0.1:8870/body-smplx.html#robust122-ply',
    'preview_loaded':{'model':'new-capture-119-sofa-clothed','points':204649,'cameras':89,'SH_degree':3,'quality_accepted':False},
    'native_contract_openfile_negative_control':{
        'path':'covariance-sh-ply/expected-reject-native-ray3d.zip','synthetic_points':6,
        'method':'Actual existing OpenFile button and browser file chooser; local ZIP parsing only',
        'observed_message':'导入未完成，保留原模型：模型声明的相机或高斯求值方式与当前 Spark 预览不一致，请使用模型指定的原生渲染器',
        'passed':True,'original_model_preserved':True},
    'camera_adapter_note':'Synthetic diagnostic cameras contained redundant w2c plus c2w fields; portable parser rejected ambiguity. A separate portable-cameras.json emits only standard rotation/position after equality verification. Original record unchanged. Not an observed defect in historical person cameras.',
    'screenshots':['covariance-sh-ply/report-ui.png','covariance-sh-ply/viewer-preview-ui.png','covariance-sh-ply/native-guard-ui.png'],
    'screenshot_limit':'The guard error was verified in the accessibility state; the screenshot records preservation of the prior model after the transient message.',
    'not_verified':['Native interactive renderer UI','Official SIBR or Spark companion GUI','Company00','WPS cloud'],
    'person_training_started':False})

plan={
    'registered':stamp,'stage':'C0a','state':'registered_not_executed',
    'title':'Local observability of effective rectified K with fixed existing background observations',
    'new_causal_variable':'For each intrinsic direction separately, quantify its pixel sensitivity remaining after projection onto the pose/point nuisance Jacobian. This is a local linear diagnostic, not another SfM/BA candidate.',
    'hypothesis':'Background-only consistency may weakly constrain output K when near-planar support permits pose/structure compensation; then low ground residual cannot certify face-height projection.',
    'subjects':['sofa','portrait'],
    'other_subjects':'grass and indoor have no accepted camera graph; report unsupported, do not invent poses or rerun extraction',
    'inputs':{
        'root':'/opt/gs-slam-lab/shuolee-raw-domain-121/<subject>',
        'observations':'roi-supplement/observations.json',
        'frozen_fit_ids':'global-fixedk/input-contract.json',
        'readonly_linearization_cameras':'global-fixedk/cameras-diagnostic.json',
        'readonly_linearization_structure':'global-fixedk/background-structure.npz',
        'scope':'These failed121 poses/points are evaluated only at their saved values; they are not an initialization for any new camera fit or person training.',
        'hash_all_inputs_before_and_after':True},
    'fixed':[
        'Use only saved triangulated FIT material identities, never heldout/quarantine for estimating a nuisance subspace',
        'Keep all original observations of those identities in registered cameras; do not prune by residual',
        'No new feature extraction, ROI supplement, pose solve, nonlinear optimization, changed K, image warp, camera export or person training',
        'First saved camera fixed; one saved seed-center scale component fixed as in121 gauge',
        'No residual lens correction or vendor FISHEYE_8 applied to rectified pixels'],
    'method':[
        'Hash and verify frame/track/split/K identity, gauge and finite positive-depth input; stop if unsupported',
        'Build analytic projection Jacobians for local SO3 left increments, camera-center coordinates and independent 3D point coordinates',
        'Check Jacobian sign/scale with a single central-difference fixture at registered epsilon, no search',
        'Test log(fx), log(fy), cx, cy separately with scales[0.001,0.001,1,1]; do not update any parameter',
        'Project out per-point Jacobians by rank-revealing SVD, then camera nuisance columns; report norms/remaining sensitivity and scaled4x4 spectrum',
        'Record rank tolerance1e-10 relative to largest singular value; one value, no threshold scan',
        'Do not turn saved optimization residuals or a local linear rank into independent physical truth or guaranteed uncertainty'],
    'budget':{'CPU_seconds_per_subject':120,'GPU':False,'attempts':1,'subjects_with_saved_structure':2,'no_new_weights':True},
    'interpretation':{
        'supports_hypothesis':'Strongly suppressed intrinsic pixel signatures and weak joint directions indicate poor local observability under these saved observations.',
        'opposes_hypothesis':'Strong signatures may oppose this local ambiguity explanation but do not validate the real K or eliminate correspondence/motion errors.',
        'non_claims':'No unique root-cause verdict, no calibrated K, no full-frame registration and no face<1px admission from this diagnostic alone.'},
    'stop_conditions':['Missing trustworthy material/gauge/identity: report unsupported','Input nonfinite or nonpositive depth: preserve exact failures, no cherry picking','CPU budget exhausted: preserve incomplete result, no repeated solver budget'],
    'source_material_review_sha256':sha(R121/'global-material-review.json'),
    'source_analysis_sha256':sha(R121/'global-fixedk-analysis.json')}
p=R/'camera-observability-plan.json'
if not p.exists():save(p,plan)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='C0a_registered_observability_diagnostic',running=None,
    next='Execute camera-observability-plan.json once on CPU after verifying source hashes and analytic Jacobians. No change of K or exported cameras. Native package loader next must use one unambiguous pose representation; its UI remains separate from current preview.',
    camera_diagnostic_plan='camera-observability-plan.json')
save(R/'queue-state.json',q)
state=read(C/'local-research-status.json');state.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'])
state['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='Current pinhole preview, report refresh and native-ray3d OpenFile rejection only; native interactive UI remains unverified')
state['renderer_readiness'].update(preview_native_contract_guard_UI_verified=True,native_interactive_GUI_verified=False,stage='C0a_registered_observability_diagnostic')
save(C/'local-research-status.json',state)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
for fact in ['Actual current preview UI2026.10.27.5 and report refreshed; historical119 model and89 cameras retained; tab15 untouched.',
             'Actual OpenFile ZIP native ray3d contract was rejected with original model preserved.',
             'Synthetic camera metadata duplicate pose fields rejected by portable parser; separate standard cameras emit one representation after transform equality check. Historical camera issue not inferred.']:
    if fact not in ledger['confirmed']:ledger['confirmed'].append(fact)
for n in ['viewer-ui-review.json','camera-observability-plan.json','covariance-sh-ply/portable-camera-adapter-check.json']:
    if n not in ledger['artifacts']:ledger['artifacts'].append(n)
ledger['unverified']=[x if x!='Current UI and company00 deployment' else 'Native interactive UI, official external viewers and company00 deployment' for x in ledger['unverified']]
save(R/'evidence-ledger.json',ledger)
receipt=read(R/'publication.json');receipt.update(updated=stamp,browser_ui_verified=True,
    browser_ui_scope='Existing report, pinhole preview/version and rejected native-ray3d OpenFile; not a native-renderer UI',
    native_interactive_ui_verified=False,remote00_verified=False)
save(R/'publication.json',receipt)
section='''<!-- ROBUST122_UI_BEGIN --><p><strong>本机实际界面核验：</strong>已刷新右侧进展及2026.10.27.5查看器，保留119轮204649点和89个相机。通过“打开文件”导入声明ray-3d的合成测试包时，界面明确拒绝并保留原模型；没有把原生三维响应静默当作屏幕高斯预览。历史tab15未动。该项不代表原生交互界面、官方SIBR、独立Spark页面或公司00已经验收。</p><p>下一项已预登记为<strong>输出内参可辨识性的只读诊断</strong>：固定已有观测，分别考察焦距和主点的像素变化能否与外参/三维结构的变化区分；不更新K、不重跑121候选、不用留出点拟合。现有沙发仅4帧、竖屏近共面地面结构作为失败基线的分析对象，不是新的训练初值。</p><!-- ROBUST122_UI_END -->'''
reports=[V/'body-smplx.html',S/'progress.html',L/'01_内参文件解析与待核实项.html']
for report in reports:
    txt=report.read_text(encoding='utf-8-sig')
    txt=re.sub(r'<!-- ROBUST122_UI_BEGIN -->.*?<!-- ROBUST122_UI_END -->','',txt,flags=re.S)
    txt=txt.replace('实际输出K、额外防抖仍未独立验证。','实际输出K仍未独立验证；用户随后已确认本批视频没有加防抖。')
    txt=txt.replace('真实去畸变输出K和防抖仍未知。','真实去畸变输出K仍未知；用户随后已确认没有加防抖。')
    assert txt.count('<!-- ROBUST122_END -->')==1
    report.write_text(txt.replace('<!-- ROBUST122_END -->',section+'<!-- ROBUST122_END -->'),encoding='utf-8',newline='\n')
for root in [V/'new-capture-118/calibration-note',S/'new-capture-118/calibration-note']:
    shutil.copy2(reports[-1],root/reports[-1].name)
print(json.dumps({'R1b':True,'preview_guard_actual_UI':True,'next_plan_registered':str(p),'native_interactive_UI':False,'training_started':False},ensure_ascii=False))
