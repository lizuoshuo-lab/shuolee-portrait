"""Publish existing evidence and the user's sidecar reply; never promotes a model."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import shutil

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
B = T/'shuolee'
O = B/'current/portrait-alignment-open-source-116'
P = O/'pixsfm'
V = T/'viewer'
now = dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)

capture = read(O/'capture-provenance.json')
capture.update(updated=now,
    user_sidecar_reply={'text':'没有我后面补', 'recorded':now,
                        'state':'not_available_now_user_may_supply_later',
                        'independent_image_diagnostics_continue':True},
    schema_search={'state':'bounded_search_complete_no_official_semantics_found',
                   'repeat_without_new_material':False})
save(O/'capture-provenance.json', capture)

setup = read(P/'setup-state.json')
guard = read(P/'setup-guard.json')
q = read(O/'queue-state.json')
q.update(updated=now, state='pixsfm_isolated_setup_in_progress', stage='S1_P1_setup',
         candidate_index=1, active_pid=setup['pid'] if setup['state']=='running' else None,
         active_process_kind='isolated dependency setup, not GPU training',
         setup_state=setup, setup_deadline_utc='2026-10-25T08:16:20+00:00',
         windows_wrapper_pid=int((P/'windows-wrapper.pid').read_text(encoding='utf-8-sig').strip()), historical_first_setup_guard=guard,
         training_started=False, quality_accepted=False,
         sidecar_state='user_will_supply_later_not_blocking',
         next_action='Inspect the verified setup process and logs; complete isolated bindings within the same setup deadline, then run the preregistered fixed-camera feature smoke test only with verified pretrained weights. No Gaussian training before alignment gates.')
save(O/'queue-state.json', q)

protocol = read(O/'protocol.json')
protocol['status'] = 'pixsfm_runtime_preparation_after_readonly_graph_preflight'
protocol['capture_priority']['next'] = 'User has no tdsp/schema available now and may supply it later. Keep original K/R/t unresolved; continue independent image diagnostics. Do not repeat the same schema search without new material.'
protocol['stages'][0]['state'] = 'completed_bounded_search_original_calibration_unresolved'
protocol['stages'][1]['state'] = 'P1_registered_and_readonly_graph_preflight_complete'
protocol['stages'][2]['state'] = 'P1_setup_running_no_optimization_yet'
save(O/'protocol.json', protocol)

candidate = read(O/'candidates.json')
candidate[0].update(state='isolated_native_build_running', local_runtime_verified=False,
                    setup_evidence='pixsfm/setup-state.json',
                    input_graph_evidence='pixsfm/sfm-track-support.json')
save(O/'candidates.json', candidate)

support = read(P/'sfm-track-support.json')
graph = read(P/'graph-preflight.json')
assert support['source_model_frames'] == 175
assert support['native_image_feature_index_vs_db_max_px_difference'] == 0
ledger_path = O/'evidence-ledger.json'
ledger = read(ledger_path) if ledger_path.exists() else {'entries':[]}
existing = {row['id']: row for row in ledger['entries']}
existing['user_original_sidecar_unavailable'] = {
    'id':'user_original_sidecar_unavailable', 'updated':now,
    'observation':'User says the original tdsp/SDK is unavailable now and may be provided later.',
    'conclusion':'Capture K/R/t semantics remain unconfirmed; no repeat empty search.'}
existing['original_sfm_feature_index_identity'] = {
    'id':'original_sfm_feature_index_identity', 'updated':now,
    'observation':'All175 COLMAP image point2D indices map to the original database features with exactly0 coordinate difference.',
    'evidence':'pixsfm/sfm-track-support.json',
    'limit':'Does not establish that cross-image matches are the same physical material point.'}
existing['original_sfm_regional_track_support'] = {
    'id':'original_sfm_regional_track_support', 'updated':now,
    'observation':'Approximate lower-face region has229 source tracks;51 have20+ consistent region observations and5 have50+. Background has85528 source tracks. The six-view smoke graph has only1 lower-face component across3+ views.',
    'evidence':['pixsfm/sfm-track-support.json','pixsfm/graph-preflight.json'],
    'limit':'These are approximate coverage counts on the fitting graph, not independent accuracy measures. Unequal region area and texture make raw counts incomparable as success rates; six-view smoke cannot establish full-face alignment.'}
ledger.update(updated=now, entries=list(existing.values()), root_cause_resolved=False)
save(ledger_path, ledger)

public = V/'body-camera116'
public.mkdir(exist_ok=True)
for name in ('capture-provenance.json','protocol.json','candidates.json','evidence-ledger.json'):
    shutil.copyfile(O/name, public/name)
for name in ('sfm-track-support.json','graph-preflight.json'):
    shutil.copyfile(P/name, public/name)

page = V/'body-smplx.html'
html = page.read_text(encoding='utf8')
start = html.index('<!-- ALIGNMENT116_BEGIN -->')
end = html.index('<!-- ALIGNMENT116_END -->')+len('<!-- ALIGNMENT116_END -->')
section = html[start:end]
section = re.sub(r'<!-- P1_PREFLIGHT_BEGIN -->.*?<!-- P1_PREFLIGHT_END -->\s*', '', section, flags=re.S)
block = '''<!-- P1_PREFLIGHT_BEGIN -->
<h3>当前核验：先区分索引映射、材料对应与相机精度</h3>
<p><strong>175帧中，COLMAP模型的特征索引与原始匹配数据库逐点一致，最大坐标差为0。</strong>这排除了这一级数据表索引串行错误；仍不能证明不同照片中匹配到的是同一块皮肤、衣料或镜框。</p>
<p>按既有遮罩和检测框近似划分区域，下半脸有229条原始SfM轨迹，其中51条在至少20帧落入该区域，5条达到50帧。预先选定的6视角运行检查只有1条覆盖至少3视角的下半脸匹配分量，因此它只用于检查算法是否运行，不能单独验收面部对齐。该统计尚未人工确认材料身份；不同区域的面积和纹理差异很大，不能用轨迹数量直接评判精度。</p>
<p>PixSfM正在隔离环境中准备Ceres、COLMAP和Python接口。未优化相机，未启动新一轮高斯训练。先固定074相机检查特征位置，再单独测试相机候选；原始数据库和图像保持不变。</p>
<p>你已告知原始tdsp/SDK资料后续补充。当前停止重复搜索同一资料，原始内外参继续标为未确认，图像诊断正常推进。</p>
<p><a href="body-camera116/sfm-track-support.json">175帧特征索引与轨迹覆盖</a> · <a href="body-camera116/graph-preflight.json">六视角运行检查输入</a> · <a href="body-camera116/evidence-ledger.json">本轮证据与结论边界</a></p>
<!-- P1_PREFLIGHT_END -->
'''
section = section.replace('<h3>什么条件下才开始下一轮训练？</h3>', block+'<h3>什么条件下才开始下一轮训练？</h3>')
assert '<!-- P1_PREFLIGHT_BEGIN -->' in section
page.write_text(html[:start]+section+html[end:], encoding='utf8')

plan = B/'long-term-plan.md'
text = plan.read_text(encoding='utf8')
marker = '<!-- PORTRAIT_ALIGNMENT_116_CURRENT_END -->'
insert = ('当前进度补充：用户暂无tdsp/SDK资料，后续补充；不等待、不重复空查。'
          'P1隔离环境准备中，setup截止2026-10-25 16:16:20（北京时间）。'
          '175帧原始SfM点索引到数据库坐标差为0，不能据此认定跨视角材料正确；'
          '6视角运行检查的下半脸支撑不足，不能代替175帧验收。尚未优化相机或新训高斯。\n\n')
text = re.sub(r'当前进度补充：.*?\n\n(?='+re.escape(marker)+')', '', text, flags=re.S)
assert marker in text
plan.write_text(text.replace(marker, insert+marker, 1), encoding='utf8')

status = read(B/'work-status.json')
status.update(updated=now, current_phase='portrait_alignment_open_source_116',
              local_training_running=False, goal_accepted=False)
status['next_bounded_work'] = {'run':str(O), 'phase':'pixsfm_isolated_setup',
    'new_training_started':False, 'active_setup_pid':setup['pid'],
    'remaining':'Complete isolated runtime, then validate material correspondences under fixed cameras before a separately evaluated camera candidate; tdsp pending user follow-up.'}
save(B/'work-status.json', status)
publication = read(O/'publication.json')
publication.update(updated=now, visual_review_pending=True,
                   new_model_created=False, latest_model_changed=False,
                   latest_diagnostic='alignment116', root_cause_resolved=False)
save(O/'publication.json', publication)
print(json.dumps({'queue_updated':True,'user_reply_saved':True,'report_updated':True,
                  'training_started':False,'model_promoted':False,'needs_actual_browser_review':True}))
