"""Record root's actual visual reviews; no training or renderer changes."""
import argparse
from update_leafwoman129_state import *

G=ROOT/'leafwoman-half12-dcbox-safe131'
SHOT=G/'gaussian-run/step-018000'
D=G/'delivery/step-018000'

def source():
    comparison=SHOT/'review/comparison129'
    independent=comparison/'first6-final-review131.json'
    assert read(independent)['actual_visual_review'] is True
    assert read(G/'final-process-exit.json')['exit_code']==0
    names=[f'{part}-{page:02d}.png' for part in ('body','arms','face') for page in (2,3)]
    receipt=dict(updated=datetime.now(timezone.utc).isoformat(),step=18000,optimizer_updates=17999,
        actual_visual_review=True,actual_process_session=33324,actual_process_exit_code=0,
        actual_CPU_review_session=75844,actual_CPU_review_exit_code=0,
        root_viewed_artifacts=[dict(path=str(comparison/n),sha256=sha(comparison/n)) for n in names],
        independent_first6_path=str(independent),independent_first6_sha256=sha(independent),
        combined_all12_body_arms_face_reviewed=True,
        world_ply_sha256=sha(SHOT/'world.ply'),snapshot_result_sha256=sha(SHOT/'result.json'),
        summary_sha256=sha(SHOT/'summary131-final.json'),
        observations=[
            'Same-level12918k and13118k source RGB appearance is close; no new gross collapse or color explosion.',
            'Cardigan stripes, shawl and fingers remain broad, smeared or fused; no clear geometric recovery.',
            'Frame18 double-eye/face ghost and large holes in visible masked white skirt persist.',
            'Side skirt edges and lower strips remain fragmented. Unknown cropped feet do not explain visible skirt failures.',
            'Cumulative alpha is slightly improved, source foreground RGB L1 slightly worse; no quality improvement claim.',
            'Mesh maximum offset is only0.651mm, smaller than1290.788mm; no stronger clothing expansion demonstrated.'],
        decision='Read-only native36 novel and paired-camera diagnostic export only; no additional training.',
        quality_accepted=False,novel_views_actually_reviewed=False)
    path=G/'root-review-18000.json';assert not path.exists();atomic(path,receipt)
    gate=dict(updated=receipt['updated'],decision='diagnostic_export_only',actual_visual_review=True,
        novel_pending=True,step=18000,optimizer_updates=17999,quality_accepted=False,
        world_ply_sha256=sha(SHOT/'world.ply'),snapshot_result_sha256=sha(SHOT/'result.json'),
        final_process_exit_sha256=sha(G/'final-process-exit.json'),
        review_evidence_path='/mnt/c/'+str(path)[3:].replace('\\','/'),review_evidence_sha256=sha(path))
    assert not (G/'review-18000-decision.json').exists();atomic(G/'review-18000-decision.json',gate)
    print('FINAL131_ACTUAL_SOURCE_REVIEW_GATE_WRITTEN')

def novel():
    names=[f'novel_{i:03d}.png' for i in range(36)]
    assert read(D/'validation.json')['novel_synthetic_views']==36
    receipt=dict(updated=datetime.now(timezone.utc).isoformat(),step=18000,optimizer_updates=17999,
        actual_novel_visual_review=True,actual_native_export_session=21646,actual_native_export_exit_code=0,
        viewed_orbit_pages={n:sha(D/'novel'/n) for n in ('orbit_00-11.png','orbit_12-23.png','orbit_24-35.png')},
        individually_viewed_indices=[0,9,18,27],all36_viewed=True,
        observations=['Front/side source-facing appearance has fused fingers and porous skirt; visible surfaces are not closed.',
            'Long pale strips remain below body; retained rather than clipped from export.',
            'Unobserved rear half is severely mottled/fragmented and has no reference ground truth.',
            'Native36 views do not demonstrate opacity, garment geometry or generalization acceptance.'],
        quality_accepted=False,novel_views_have_ground_truth=False,
        source_world_ply_sha256=sha(SHOT/'world.ply'),native_export_validation_sha256=sha(D/'validation.json'),
        portable_ply_sha256=sha(D/'point_cloud.ply'),novel_cameras_sha256=sha(D/'novel/cameras.json'),
        novel_png_sha256={n:sha(D/'novel'/n) for n in names},decision='diagnostic_package_only')
    assert not (G/'review-18000-novel-decision.json').exists()
    atomic(G/'review-18000-novel-decision.json',receipt)
    print('FINAL131_ACTUAL36_NOVEL_REVIEW_GATE_WRITTEN')

def ui():
    images=[G/'ui-final'/n for n in ('camera4-dark.png','camera4-gray.png','camera4-points.png','overview-dark.png')]
    novel=read(G/'review-18000-novel-decision.json')
    viewed=[D/'novel'/n for n in ('orbit_00-11.png','orbit_12-23.png','orbit_24-35.png')]
    viewed += [D/'novel'/f'novel_{i:03d}.png' for i in (0,9,18,27)]
    assert novel['actual_novel_visual_review'] and all(p.stat().st_size>1000 for p in images+viewed)
    receipt=dict(updated=datetime.now(timezone.utc).isoformat(),step=18000,optimizer_updates=17999,
        actual_visual_review=True,all36_continuous_synthetic_views_actually_reviewed=True,
        actual_native_export_session=21646,actual_native_export_exit_code=0,actual_package_exit_code=0,
        actual_OpenFile_scene_ZIP=True,actual_Gaussian_and_points=True,
        actual_gray_camera4_and_dark_overview=True,actual_points=201160,actual_paired_camera_count=1,
        actual_viewer_tab='62',actual_viewer_URL='http://127.0.0.1:8870/?sh=0&framing=fit',
        actual_model_title='持叶DC安全边界131 · 18000步终点 · 17999次实际更新 · 配套导入',
        package_sha256=sha(D/'scene.zip'),native_validation_sha256=sha(D/'validation.json'),
        viewed_images=[dict(path=str(p),sha256=sha(p)) for p in viewed],
        ui_images=[dict(path=str(p),sha256=sha(p)) for p in images],
        scientific_quality_accepted=False,official_SIBR_GUI_verified=False,
        independent_Spark_GUI_verified=False,company00_verified=False,
        observations=['OpenFile loaded201160points and1 SHA-bound camera frame4; SH0Gaussian and point modes both rendered.',
            'Gray source-camera view still has thin/soft porous clothing boundaries; crop omits some lower fragments.',
            'Dark overview deliberately includes all valid points and reveals the long pale lower fragments.',
            'No display-size, opacity, geometry or point filtering change was used to hide the defects.'])
    assert not (G/'native-final-and-viewer-review.json').exists()
    atomic(G/'native-final-and-viewer-review.json',receipt)
    print('FINAL131_ACTUAL_OPENFILE_UI_RECEIPT_WRITTEN')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['source','novel','ui']);a=parser.parse_args()
    {'source':source,'novel':novel,'ui':ui}[a.mode]()
