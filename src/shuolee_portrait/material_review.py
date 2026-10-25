"""A portable, source-only review of every frozen material-track observation.

No matching, geometry fit, residual ranking or automatic acceptance occurs.
Image copies keep the original bytes; crop positions are a viewing operation.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from . import runtime as rt
from .fixed_tracks import PIXEL_DOMAIN, cameras_from_records

PATCH_SIZE = 96
MAX_SOURCE_BYTES = 512 * 1024**2
MAX_OBSERVATIONS = 50_000
MAX_IMAGES = 256


def specification():
    return dict(patch_size_original_pixels=PATCH_SIZE, max_source_bytes=MAX_SOURCE_BYTES,
        max_observations=MAX_OBSERVATIONS, max_images=MAX_IMAGES,
        ordering='Tracks by original ID; observations by original camera order. No residual sort.',
        selection='All tracks and all observations, including unknown and unobservable ones.',
        image_operation='Original byte copies; browser clips a fixed native-pixel window. No image warping or burned-in labels.',
        limitation='Centred patches aid material inspection; they do not show geometric or camera alignment.')


def _crop(uv, wh):
    xy=np.asarray(uv,float);wh=np.asarray(wh,int)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or (xy<0).any() or (xy>wh-1).any():
        raise ValueError('Every observation must be inside its unchanged source image.')
    origin=np.floor(xy+.5).astype(int)-PATCH_SIZE//2
    lo=np.maximum(origin,0);hi=np.minimum(origin+PATCH_SIZE,wh)
    return dict(window_origin_xy=origin.tolist(),crop_xyxy=np.r_[lo,hi].tolist(),
        local_uv=(xy-origin).tolist(),clipped_at_image_edge=bool((lo!=origin).any() or (hi!=origin+PATCH_SIZE).any()))


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_material_review(output, rows, bank, source_evidence):
    """Build all-or-none, before any source review. No subset fallback on budget."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Material-review output must be fresh.')
    cameras_from_records(rows)
    if bank.get('schema')!='shuolee-fixed-tracks-v1' or bank.get('pixel_domain')!=PIXEL_DOMAIN:
        raise ValueError('Frozen integer-centre material observations required.')
    by_frame={r['frame']:r for r in rows};order={r['frame']:i for i,r in enumerate(rows)}
    ids=[t['track_id'] for t in bank['tracks']]
    if any(not isinstance(t,str) or not t for t in ids) or len(set(ids))!=len(ids):
        raise ValueError('Nonempty unique track IDs required.')
    observations=[o for t in bank['tracks'] for o in t['observations']]
    obs_ids=[o['observation_id'] for o in observations]
    if any(not isinstance(o,str) or not o for o in obs_ids) or len(set(obs_ids))!=len(obs_ids):
        raise ValueError('Observation IDs must remain unique across all tracks.')
    if len(observations)>MAX_OBSERVATIONS:
        raise ValueError('Review observation budget exceeded; no tracks were sampled or omitted.')
    needed={o['frame'] for o in observations}
    if not needed<=set(by_frame):raise ValueError('Observation refers to an unknown source frame.')
    chosen=[r for r in rows if r['frame'] in needed]
    if len(chosen)>MAX_IMAGES or sum(Path(r['image']).stat().st_size for r in chosen)>MAX_SOURCE_BYTES:
        raise ValueError('Portable image budget exceeded; no frame/track subset fallback.')
    frames=[];frames_map={}
    for i,row in enumerate(chosen):
        path=Path(row['image'])
        if rt.sha(path)!=row['source_sha256']:raise ValueError(f'Source image changed: {row["frame"]}')
        with Image.open(path) as im:
            if list(im.size)!=row['wh']:raise ValueError('Camera dimensions differ from source pixels.')
            suffix={"PNG":'.png',"JPEG":'.jpg',"WEBP":'.webp'}.get(im.format)
            if suffix is None:raise ValueError('Review requires PNG, JPEG or WebP source images.')
            im.verify()
        frame=dict(frame=row['frame'],order=order[row['frame']],source_frame_index=row['source_frame_index'],
            time_seconds=row['time_seconds'],wh=row['wh'],source_sha256=row['source_sha256'],
            asset=f'images/source_{i:06d}{suffix}')
        frames.append(frame);frames_map[row['frame']]=frame
    tracks=[]
    for track in sorted(bank['tracks'],key=lambda t:t['track_id']):
        seen=set();items=[]
        for o in sorted(track['observations'],key=lambda o:order[o['frame']]):
            row=by_frame[o['frame']]
            if o['frame'] in seen or o['image_sha256']!=row['source_sha256']:
                raise ValueError('Duplicate track frame or mismatched source-image identity.')
            seen.add(o['frame'])
            items.append(dict(observation_id=o['observation_id'],frame=o['frame'],uv=o['uv'],
                image_sha256=o['image_sha256'],role='triangulation' if o['frame'] in track['triangulation_frames'] else 'held',
                region_annotation=o.get('region_annotation'),**_crop(o['uv'],row['wh'])))
        fit=track['triangulation_frames']
        if len(items)<3 or len(fit)<2 or len(set(fit))!=len(fit) or not set(fit)<seen:
            raise ValueError('A material track must retain its distinct fit and held views.')
        tracks.append(dict(track_id=track['track_id'],region=track['region'],
            identity_basis=track['identity_basis'],region_annotation=track.get('region_annotation'),
            observations=items,review_state='unreviewed',physical_material_identity_verified=False))
    sources=dict(camera_records_sha256=_digest(rows),frozen_bank_sha256=_digest(bank),source_evidence=source_evidence)
    data=dict(schema='shuolee-material-source-review-v1',specification=specification(),sources=sources,
        frames=frames,tracks=tracks,total_tracks=len(tracks),total_observations=len(observations),
        frame_order=[r['frame'] for r in rows],frames_without_frozen_observations=[r['frame'] for r in rows if r['frame'] not in needed],
        image_warped=False,text_burned_into_images=False,cameras_optimized=False,
        physical_material_identity_verified=False,scene_staticity_verified=False,training_admitted=False)
    data['review_id']=_digest(data)
    output.mkdir(parents=True);(output/'images').mkdir()
    template=Path(__file__).with_name('material_review_assets')
    rt.save(output/'plan.json',dict(specification=specification(),sources=sources,review_id=data['review_id'],
        implementation_sha256=rt.sha(Path(__file__)),
        template_hashes={p.name:rt.sha(p) for p in template.iterdir() if p.is_file()}))
    for row in chosen:
        asset=output/frames_map[row['frame']]['asset'];shutil.copyfile(row['image'],asset)
        if rt.sha(asset)!=row['source_sha256'] or rt.sha(row['image'])!=row['source_sha256']:
            raise ValueError('Source or copied pixels changed during review generation.')
    rt.save(output/'manifest.json',data)
    # A local script works when opened from file:// too, without fetch/CORS or
    # an external service. All source strings are rendered using textContent.
    (output/'data.js').write_text('window.MATERIAL_REVIEW='+json.dumps(data,ensure_ascii=False,
        allow_nan=False).replace('<','\\u003c')+';\n',encoding='utf-8')
    for name in ('index.html','review.js','review.css'):
        shutil.copyfile(template/name,output/name)
    return dict(entry='material-review/index.html',manifest='material-review/manifest.json',
        review_id=data['review_id'],tracks=len(tracks),observations=len(observations),
        source_frames=len(chosen),tracks_reviewed=0,physical_material_identity_verified=False,
        bytes_copied=sum((output/f['asset']).stat().st_size for f in frames),no_residual_based_selection=True)
