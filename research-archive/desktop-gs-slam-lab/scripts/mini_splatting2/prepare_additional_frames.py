"""Localize previously unused video frames against a frozen SfM reconstruction.

No bundle adjustment, geometry update, or appearance training is permitted.
Frame selection is uniform by source index, before reading candidate metrics.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001'
VIDEO=LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
OUT=ROOT/'additional-frames-001'

def main():
    OUT.mkdir(exist_ok=False);images=OUT/'raw_images';shutil.copytree(BASE/'raw_images',images)
    db=OUT/'database.db';shutil.copy2(BASE/'database.db',db)
    ffmpeg=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
    tool=Path('C:/Users/Administrator/colmap-x64-windows-cuda');colmap=tool/'bin/colmap.exe'
    env=os.environ.copy();env['PATH']=str(tool/'bin')+os.pathsep+env['PATH'];env['QT_PLUGIN_PATH']=str(tool/'plugins')
    state=dict(status='preparing',started=time.time(),commands=[],selection='24 uniformly spaced unused source frame indices',
        training_forbidden=True,geometry_update_forbidden=True,scope='additional temporal frames in same video, not an independent capture')
    def save(): (OUT/'manifest.json').write_text(json.dumps(state,indent=2))
    def run(label,cmd):
        state['current']=label;state['commands'].append([str(x) for x in cmd]);save()
        with (OUT/f'{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    run('decode-frame-map',[ffmpeg,'-hide_banner','-i',VIDEO,'-vf','showinfo,fps=8,showinfo','-an','-f','null','-'])
    text=(OUT/'decode-frame-map.log').read_text(encoding='utf-8')
    raw={};selected=[]
    for line in text.splitlines():
        m=re.search(r'Parsed_showinfo_(0|2).*?n:\s*(\d+).*? checksum:([0-9A-F]+)',line)
        if not m:continue
        branch,index,checksum=m.groups()
        if branch=='0':raw.setdefault(checksum,[]).append(int(index))
        else:selected.append((int(index),checksum))
    assert len(selected)==59,(len(selected),len(raw))
    assert all(len(raw[c])==1 for _,c in selected),'Ambiguous original frame checksum'
    original={f'{n:06d}.jpg':raw[c][0] for n,c in selected}
    all_indices=sorted(i for v in raw.values() for i in v);used=set(original.values())
    available=sorted(set(all_indices)-used)
    query=[available[int(i)] for i in np.linspace(0,len(available)-1,24).round()]
    assert len(set(query))==24 and not (set(query)&used)
    state['source_video_sha256']=hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    state.update(source_frames=len(all_indices),original_source_indices=original,query_source_indices=query)
    # Freeze selection before rendering or comparing any model on these images.
    (OUT/'selection.json').write_text(json.dumps(dict(original=original,query=query,selection=state['selection']),indent=2));save()
    extracted=OUT/'extracted';extracted.mkdir()
    selector='+'.join(f'eq(n\\,{n})' for n in query)
    run('extract',[ffmpeg,'-hide_banner','-loglevel','warning','-n','-i',VIDEO,'-vf',f'select={selector},scale=760:1352',
        '-vsync','0','-q:v','1','-start_number','0',extracted/'%03d.jpg'])
    query_names=[]
    for i,n in enumerate(query):
        name=f'query_{n:06d}.jpg';shutil.copy2(extracted/f'{i:03d}.jpg',images/name);query_names.append(name)
    query_file=OUT/'queries.txt';query_file.write_text('\n'.join(query_names)+'\n')
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'));import read_write_model as rw
    source_model=BASE/'raw_sparse/0';cams,ims,points=rw.read_model(str(source_model),ext='.bin')
    assert len(cams)==1
    cam=next(iter(cams.values()))
    # Original database stores initialization intrinsics; use the frozen calibrated values.
    with sqlite3.connect(db) as connection:
        connection.execute('UPDATE cameras SET params=?,prior_focal_length=1 WHERE camera_id=?',(cam.params.astype(np.float64).tobytes(),cam.id));connection.commit()
    run('features',[colmap,'feature_extractor','--database_path',db,'--image_path',images,'--image_list_path',query_file,
        '--ImageReader.existing_camera_id',str(cam.id),'--SiftExtraction.use_gpu','0','--SiftExtraction.num_threads','4','--SiftExtraction.max_num_features','8192'])
    pairs=[]
    # Match each new frame only to existing model images, never query-to-query.
    for name,n in zip(query_names,query):
        nearest=sorted(original,key=lambda k:abs(original[k]-n))[:8]
        pairs.extend(f'{name} {reference}' for reference in nearest)
    pair_file=OUT/'pairs.txt';pair_file.write_text('\n'.join(pairs)+'\n')
    run('matching',[colmap,'matches_importer','--database_path',db,'--match_list_path',pair_file,'--match_type','pairs',
        '--SiftMatching.use_gpu','0','--SiftMatching.num_threads','4','--SiftMatching.guided_matching','1'])
    registered=OUT/'registered';registered.mkdir()
    run('registration',[colmap,'image_registrator','--database_path',db,'--input_path',source_model,'--output_path',registered,
        '--Mapper.num_threads','4','--Mapper.fix_existing_images','1','--Mapper.ba_refine_focal_length','0',
        '--Mapper.ba_refine_extra_params','0','--Mapper.abs_pose_max_error','4'])
    newcams,newims,newpts=rw.read_model(str(registered),ext='.bin')
    # Enforce the protocol by comparing actual output, not merely CLI flags.
    assert set(points)==set(newpts)
    assert all(np.array_equal(p.xyz,newpts[k].xyz) for k,p in points.items()),'Original geometry changed'
    assert all(np.allclose(im.qvec2rotmat(),newims[k].qvec2rotmat(),atol=1e-10) and np.allclose(im.tvec,newims[k].tvec,atol=1e-10) for k,im in ims.items()),'Original camera poses changed'
    assert all(np.allclose(c.params,newcams[k].params,atol=1e-10) for k,c in cams.items()),'Original calibration changed'
    state['frozen_reconstruction_verified']=True
    names={im.name:im for im in newims.values()};successful=[n for n in query_names if n in names]
    state['registered_queries']=len(successful);state['failed_queries']=[n for n in query_names if n not in names]
    if not successful:raise RuntimeError('No additional frame localized')
    dataset=OUT/'dataset'
    run('undistort',[colmap,'image_undistorter','--image_path',images,'--input_path',registered,'--output_path',dataset,'--output_type','COLMAP','--max_image_size','1352'])
    dc,di,dp=rw.read_model(str(dataset/'sparse'),ext='.bin');rows=[]
    for im in di.values():
        if im.name not in successful:continue
        c=dc[im.camera_id];assert c.model=='PINHOLE';R=im.qvec2rotmat();C=-R.T@im.tvec
        rows.append(dict(name=im.name,source_index=int(im.name[6:12]),width=int(c.width),height=int(c.height),fx=float(c.params[0]),fy=float(c.params[1]),
            cx=float(c.params[2]),cy=float(c.params[3]),position=C.tolist(),rotation=R.T.tolist(),
            correspondences=int((im.point3D_ids>=0).sum())))
    rows.sort(key=lambda x:x['source_index']);(OUT/'cameras.json').write_text(json.dumps(rows,indent=2))
    state.update(status='localized_not_evaluated',finished=time.time(),test_images=len(rows),
        data_used_for_appearance_training=False,reference_geometry_updated=False,
        evaluation_policy='Freeze candidate model hashes before rendering these frames; report correlated same-sequence interpolation, not new scene generalization')
    save();print(json.dumps(state),flush=True)

if __name__=='__main__':main()
