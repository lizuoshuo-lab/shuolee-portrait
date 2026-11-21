"""Record completed visual/browser review; does not accept a scientific winner."""
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'

def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    out=ROOT/'native-resolution-final-review-001.json';assert not out.exists()
    extra=json.loads((ROOT/'additional-frame-resolution-005/summary.json').read_text())
    a=extra['models']['low']['aggregate']['1'];b=extra['models']['full']['aggregate']['1']
    review=dict(status='reviewed',created=time.time(),accepted=False,goal_complete=False,
        primary_summary='native-resolution-cache-review-001/summary.json',
        visual=dict(primary_all8_both_targets=True,additional_all24_contact_sheets=True,
            source_split_disjoint=True,source_split_audit='native-resolution-extra-visual-review-001/manifest.json',
            inspected_native_full_crop_frames=['000024','000040'],
            inspected_path_stills=['distance-00','distance-15','distance-45','elevation-00','elevation-15','elevation-45'],
            finding='Full-resolution training sometimes preserves hair/clothing detail; face/glasses remain blurred or doubled. Large distance/pitch excursions still show torn surfaces and floaters in both models.'),
        browser=dict(report='http://127.0.0.1:8820/native-resolution.html',viewer='http://127.0.0.1:8820/',
            models=17,low_gaussians=201208,full_gaussians=316044,
            both_new_models_loaded=True,pose_preserved='000024',correct_ply_links=True,
            output_resolution_control=True,view_control=True,error_control=True,
            additional_frame_control=True,extra_group_controls=['all','far','higher_pitch'],extra_full_and_head=True,
            media_metadata=[dict(kind='distance',duration_seconds=6,width=1514,height=1380,ready_state=4),
                            dict(kind='elevation',duration_seconds=6,width=1514,height=1380,ready_state=4)],
            no_console_errors=True,no_horizontal_overflow=True,proof='resolution-browser-proof.png'),
        additional24_delta=dict(full_psnr=b['full_psnr']-a['full_psnr'],head_psnr=b['head']['psnr']-a['head']['psnr'],
            head_gradient_error=b['head']['normalized_gradient_error']-a['head']['normalized_gradient_error']),
        decision='Not accepted as human-clarity or geometry solution. Preserve pair; test post-warmup topology schedule in isolation.',
        limitations=['One seed; same-video development data','Extra24 renderer/metric encoding differs from primary; compare within each section',
            'Manual/interpolated head rectangles include background','No wide-pitch GT or metric-depth GT','Red-shirt subject is mostly static; not dynamic-human acceptance'],
        rejected_evidence=['legacy resolution-evaluation v1 metrics','sequential-backward memory optimization'],
        next_stage='native-late-protocol-001.json')
    write(out,review)
    for rel in ['native-resolution-cache-publication-queue.json','native-resolution-followup-queue.json']:
        p=ROOT/rel;j=json.loads(p.read_text());assert j['status']=='complete_pending_browser_and_visual_review'
        j.update(status='complete',review=out.name,accepted=False);write(p,j)
    p=ROOT/'viewer-resolution-registration-001/summary.json';j=json.loads(p.read_text());j.update(status='browser_verified',review=out.name);write(p,j)
    for folder in ['native-resolution-visual-review-001','native-resolution-extra-visual-review-001']:
        p=ROOT/folder/'manifest.json';j=json.loads(p.read_text());j.update(status='reviewed',review=out.name);write(p,j)
    p=ROOT/'native-resolution-observed-publication-001/manifest.json';j=json.loads(p.read_text());j.update(status='browser_verified',review=out.name);write(p,j)
    print(json.dumps({'status':review['status'],'accepted':False,'extra24_delta':review['additional24_delta']}))

if __name__=='__main__':main()
