"""Verify source identity/crop handoff using synthetic known pixels only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import material_review as mr, runtime as rt


def fixture(root):
    root.mkdir();rows=[]
    tracks=[dict(track_id=f'material_{i:03d}',region=region,identity_basis='material_candidate',
                 triangulation_frames=['frame_000','frame_004'],observations=[])
            for i,region in enumerate(['static_scene_candidate','face','body','unknown'])]
    for i in range(5):
        y,x=np.mgrid[:360,:480];pixels=np.stack([(x+i*13)%256,(y+i*7)%256,((x//12+y//12)%2)*110+60],axis=-1).astype('uint8')
        im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
        # Unlabelled synthetic patterns, never portrayed as human photographs.
        draw.rectangle((80+i*8,75,135+i*8,125),fill=(210,170,40),outline=(20,40,20),width=3)
        draw.ellipse((210+i*4,110,255+i*4,155),fill=(90,195,130),outline=(30,40,40),width=3)
        image=root/f'frame_{i:03d}.png';im.save(image)
        E=np.eye(4);E[0,3]=-.1*i
        row=dict(frame=f'frame_{i:03d}',name=image.name,source_frame_index=i*2,time_seconds=i/12,
            image=str(image),source_sha256=rt.sha(image),wh=[480,360],
            K=[[300.,0,240.5],[0,300.,180.5],[0,0,1]],w2c=E.tolist())
        rows.append(row)
        for j,uv in enumerate([[100.+i*8,90.],[225.+i*4,135.],[285.+i*2,230.],[.1,359.]]):
            tracks[j]['observations'].append(dict(observation_id=f'p{j}-f{i}',frame=row['frame'],
                image_sha256=row['source_sha256'],uv=uv,
                region_annotation=dict(region=tracks[j]['region'],reason='synthetic_region_annotation')))
    return rows,dict(schema='shuolee-fixed-tracks-v1',pixel_domain='undistorted_integer_centers',tracks=tracks)


def expect_error(fn):
    try:fn()
    except (ValueError,FileExistsError) as exc:return str(exc)
    raise AssertionError('Invalid review input was accepted.')


def check(evidence_dir):
    cases={}
    with tempfile.TemporaryDirectory(prefix='shuolee-source-review-') as tmp:
        root=Path(tmp);rows,bank=fixture(root/'sources');before=copy.deepcopy((rows,bank))
        hashes={p.name:rt.sha(p) for p in (root/'sources').iterdir()}
        output=root/'review';result=mr.write_material_review(output,rows,bank,dict(synthetic_fixture=True))
        data=rt.read(output/'manifest.json')
        assert result['tracks']==4 and result['observations']==20 and result['source_frames']==5
        assert (rows,bank)==before and hashes=={p.name:rt.sha(p) for p in (root/'sources').iterdir()}
        assert {o['observation_id'] for t in data['tracks'] for o in t['observations']}=={
            o['observation_id'] for t in bank['tracks'] for o in t['observations']}
        assert all(t['review_state']=='unreviewed' and not t['physical_material_identity_verified'] for t in data['tracks'])
        assert not data['training_admitted'] and result['tracks_reviewed']==0
        cases['all_source_members_preserved']=dict(tracks=4,observations=20,images=5,unknown_retained=True,no_false_review=True)
        sources={r['frame']:r for r in rows};assets={f['frame']:f for f in data['frames']}
        for frame in data['frames']:
            original=Path(sources[frame['frame']]['image']);copied=output/frame['asset']
            assert original.read_bytes()==copied.read_bytes()
        for t in data['tracks']:
            for o in t['observations']:
                source=np.array(Image.open(sources[o['frame']]['image']));image=np.array(Image.open(output/assets[o['frame']]['asset']))
                l,t0,r,b=o['crop_xyxy']
                assert np.array_equal(source[t0:b,l:r],image[t0:b,l:r])
                assert np.allclose(np.asarray(o['local_uv'])+o['window_origin_xy'],o['uv'],rtol=0,atol=1e-12)
        border=data['tracks'][-1]['observations'][0]
        assert border['clipped_at_image_edge'] and border['crop_xyxy']==[0,311,48,360]
        cases['native_pixels_and_subpixel_location']=dict(all_copy_hashes_match=True,
            all_crop_pixels_identical=True,edge_crop=border['crop_xyxy'],no_warp_or_burned_text=True)

        alt=copy.deepcopy(bank);alt['tracks'].reverse()
        for t in alt['tracks']:t['observations'].reverse()
        alternate=root/'reordered';mr.write_material_review(alternate,rows,alt,{})
        changed=rt.read(alternate/'manifest.json')
        assert [t['track_id'] for t in changed['tracks']]==[t['track_id'] for t in data['tracks']]
        assert all([o['frame'] for o in t['observations']]==[r['frame'] for r in rows] for t in changed['tracks'])
        assert [t['observations'] for t in changed['tracks']]==[t['observations'] for t in data['tracks']]
        cases['deterministic_source_order_without_metric_sort']=True

        invalid=[]
        bad=copy.deepcopy(bank);bad['tracks'][0]['observations'][0]['uv']=[-1,4]
        invalid.append(expect_error(lambda:mr.write_material_review(root/'bad_pixel',rows,bad,{})))
        bad=copy.deepcopy(bank);bad['tracks'][0]['observations'][0]['image_sha256']='a'*64
        invalid.append(expect_error(lambda:mr.write_material_review(root/'bad_hash',rows,bad,{})))
        bad=copy.deepcopy(bank);bad['tracks'][1]['observations'][0]['observation_id']=bad['tracks'][0]['observations'][0]['observation_id']
        invalid.append(expect_error(lambda:mr.write_material_review(root/'duplicate',rows,bad,{})))
        bad=copy.deepcopy(bank);bad['tracks'][0]['triangulation_frames']=['frame_000']
        invalid.append(expect_error(lambda:mr.write_material_review(root/'one_fit',rows,bad,{})))
        invalid.append(expect_error(lambda:mr.write_material_review(output,rows,bank,{})))
        path=Path(rows[0]['image']);prior=path.read_bytes();path.write_bytes(prior+b'x')
        invalid.append(expect_error(lambda:mr.write_material_review(root/'mutated',rows,bank,{})))
        path.write_bytes(prior)
        cases['invalid_source_and_overwrite_rejected']=invalid
        with patch.object(mr,'MAX_SOURCE_BYTES',1):
            message=expect_error(lambda:mr.write_material_review(root/'over_budget',rows,bank,{}))
        assert not (root/'over_budget').exists()
        cases['budget_does_not_silently_select_subset']=message

        no=copy.deepcopy(bank);no['tracks']=[];empty=root/'empty'
        mr.write_material_review(empty,rows,no,{})
        nothing=rt.read(empty/'manifest.json');assert nothing['total_tracks']==nothing['total_observations']==0
        assert len(nothing['frames_without_frozen_observations'])==5 and not nothing['training_admitted']
        hostile=copy.deepcopy(bank);hostile['tracks'][0]['track_id']='</script><img src=x onerror=alert(1)>'
        unsafe=root/'escaped';mr.write_material_review(unsafe,rows,hostile,{})
        assert '</script><img' not in (unsafe/'data.js').read_text()
        assert 'textContent' in (unsafe/'review.js').read_text() and 'innerHTML' not in (unsafe/'review.js').read_text()
        cases['empty_bank_and_text_identifiers']=dict(empty_is_not_acceptance=True,source_strings_not_html=True)
        if evidence_dir:
            if evidence_dir.exists():raise FileExistsError(evidence_dir)
            shutil.copytree(output,evidence_dir)
            (evidence_dir/'fixture.txt').write_text('Synthetic engineering UI fixture only. No human images, no real-camera validation.\n',encoding='utf-8')
    return dict(passed=True,groups=len(cases),cases=cases,synthetic_only=True,
        real_person_material_identity_verified=False,training_started=False,old_diagnosis_reopened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--evidence-dir',type=Path);args=parser.parse_args()
    result=check(args.evidence_dir);rt.save(args.report,result);print(json.dumps(result,indent=2))
