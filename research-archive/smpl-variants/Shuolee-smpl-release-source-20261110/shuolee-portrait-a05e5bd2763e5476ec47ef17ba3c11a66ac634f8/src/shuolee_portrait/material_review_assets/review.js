'use strict';
(() => {
  const data=window.MATERIAL_REVIEW;
  const el=id=>document.getElementById(id);
  const labels={face:'面部ROI',body:'非脸部前景',clothing:'衣物候选',static_scene_candidate:'背景候选（静止性未证实）',unknown:'未知／边界／区域不一致'};
  const bases={material_candidate:'纹理匹配候选',semantic_landmark:'语义关键点编号',unknown:'材料身份未知'};
  const reasons={inside_estimated_face_roi:'脸部检测框内',foreground_outside_face_not_clothing_segmentation:'前景中脸部以外的区域',
    outside_target_mask_staticity_unverified:'人物外部，尚未证实静止',uncertain_mask_boundary:'靠近人物轮廓边界',
    known_non_target_person_box:'其他人物范围',foreground_without_face_roi:'前景中缺少脸部区域记录',
    uncertain_face_roi_boundary:'靠近脸部检测框边界',synthetic_region_annotation:'模拟区域标签'};
  const frameMap=new Map(data.frames.map(f=>[f.frame,f]));
  let visible=data.tracks,selectedObservation=null;
  const option=(text,value)=>{const n=document.createElement('option');n.textContent=text;n.value=value;return n;};
  const paragraph=(text)=>{const n=document.createElement('p');n.textContent=text;return n;};
  el('scope').textContent=`完整保留 ${data.total_tracks} 条轨迹、${data.total_observations} 个观测，涉及 ${data.frames.length} 帧。高误差、未知区域及不可测轨迹未按误差筛除。`;
  el('review-id').textContent=`原始观测批次：${data.review_id.slice(0,16)} · 尚未审阅 · 不代表相机／人体质量通过`;
  if(data.sources.source_evidence.synthetic_fixture===true){
    const warning=paragraph('工程界面演示：以下为模拟纹理图案，不是实拍人像，也不是相机标定或重影改善结果。');
    warning.className='notice';document.querySelector('header').prepend(warning);
  }
  el('empty-frames').textContent=data.frames_without_frozen_observations.length?` 有 ${data.frames_without_frozen_observations.length} 帧没有冻结观测，具体帧号见清单。`:'';
  for(const region of [...new Set(data.tracks.map(t=>t.region))])el('region').append(option(labels[region]||region,region));
  function showContext(obs){
    const f=frameMap.get(obs.frame);selectedObservation=obs.observation_id;
    el('context-image').src=f.asset;el('context-image').alt=`${f.frame} 原始完整图像`;
    el('context-marker').setAttribute('viewBox',`0 0 ${f.wh[0]} ${f.wh[1]}`);
    // SVG coordinates have a corner origin; integer-centre source pixels add 0.5 once.
    el('context-point').setAttribute('cx',obs.uv[0]+.5);el('context-point').setAttribute('cy',obs.uv[1]+.5);
    el('context-point').setAttribute('r',Math.max(3,Math.min(...f.wh)/150));
    const [l,t,r,b]=obs.crop_xyxy;
    Object.entries({x:l,y:t,width:r-l,height:b-t}).forEach(([k,v])=>el('context-crop').setAttribute(k,v));
    el('context-info').textContent=`${f.frame} · 原视频帧 ${f.source_frame_index} · ${Number(f.time_seconds).toFixed(3)} 秒 · 观测 (${obs.uv.map(v=>Number(v).toFixed(2)).join(', ')}) · ${f.wh.join(' × ')} 原像素`;
    el('original-link').href=f.asset;
    for(const card of document.querySelectorAll('.card'))card.classList.toggle('selected',card.dataset.observation===selectedObservation);
  }
  function render(){
    const track=visible.find(t=>t.track_id===el('track').value);
    el('observations').replaceChildren();
    el('context-section').hidden=!track;el('track').disabled=!track;
    if(!track){
      el('track-title').textContent='没有可显示的冻结轨迹';el('track-info').textContent='没有观测不等于对齐通过。';
      el('filter-count').textContent=`当前显示 0 / ${data.total_tracks} 条轨迹`;
      el('previous').disabled=true;el('next').disabled=true;return;
    }
    const index=visible.indexOf(track);el('previous').disabled=index===0;el('next').disabled=index===visible.length-1;
    el('filter-count').textContent=`当前筛选显示 ${visible.length} / ${data.total_tracks} 条轨迹；第 ${index+1} 条。下面完整显示该轨迹全部 ${track.observations.length} 个观测。`;
    el('track-title').textContent=track.track_id;
    el('track-info').textContent=`${labels[track.region]||track.region} · ${bases[track.identity_basis]||'来源记录见清单'} · 未审阅。三角化／留出身份仅作记录，不作为材料判断依据。`;
    const z=Number(el('zoom').value),size=data.specification.patch_size_original_pixels;
    for(const obs of track.observations){
      const f=frameMap.get(obs.frame);const card=document.createElement('article');card.className='card';card.dataset.observation=obs.observation_id;
      const crop=document.createElement('div');crop.className='crop';crop.style.width=`${size*z}px`;crop.style.height=`${size*z}px`;
      const img=document.createElement('img');img.src=f.asset;img.alt=`${f.frame} 同一候选材料的原图局部`;img.loading='lazy';
      img.style.width=`${f.wh[0]*z}px`;img.style.height=`${f.wh[1]*z}px`;
      img.style.left=`${-obs.window_origin_xy[0]*z}px`;img.style.top=`${-obs.window_origin_xy[1]*z}px`;
      const point=document.createElement('span');point.className='point';point.setAttribute('aria-label','冻结观测落点');
      point.style.left=`${(obs.local_uv[0]+.5)*z}px`;point.style.top=`${(obs.local_uv[1]+.5)*z}px`;
      crop.append(img,point);card.append(crop,paragraph(`${f.frame} · 原帧 ${f.source_frame_index} · ${Number(f.time_seconds).toFixed(3)} 秒`),
        paragraph(`原图坐标 (${obs.uv.map(v=>Number(v).toFixed(2)).join(', ')}) · ${obs.role==='held'?'留出视角':'三角化视角'}`));
      if(obs.region_annotation)card.append(paragraph(`区域记录：${labels[obs.region_annotation.region]||obs.region_annotation.region} · ${reasons[obs.region_annotation.reason]||'详细原因见清单'}`));
      if(obs.clipped_at_image_edge)card.append(paragraph('窗口越过图像边缘；棋盘格不是源图像素。'));
      const button=document.createElement('button');button.type='button';button.textContent='在完整原图中定位';
      button.addEventListener('click',()=>{showContext(obs);el('context-title').scrollIntoView({block:'start'});});
      card.append(button);el('observations').append(card);
    }
    showContext(track.observations.find(o=>o.observation_id===selectedObservation)||track.observations[0]);
  }
  function filter(){
    const current=el('track').value;
    visible=data.tracks.filter(t=>el('region').value==='all'||t.region===el('region').value);
    el('track').replaceChildren(...visible.map(t=>option(`${t.track_id} · ${labels[t.region]||t.region} · ${t.observations.length}帧`,t.track_id)));
    if(visible.some(t=>t.track_id===current))el('track').value=current;
    render();
  }
  el('region').addEventListener('change',filter);el('track').addEventListener('change',render);
  el('zoom').addEventListener('change',render);
  el('markers').addEventListener('change',()=>document.body.classList.toggle('hide-markers',!el('markers').checked));
  for(const [id,delta] of [['previous',-1],['next',1]])el(id).addEventListener('click',()=>{
    const i=visible.findIndex(t=>t.track_id===el('track').value)+delta;
    if(i>=0&&i<visible.length){el('track').value=visible[i].track_id;render();}
  });
  filter();
})();
