"""Analytic size audit. Conditional mapping examples, not accepted calibration."""
from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0, '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait.camera_pixel_contract import rectified_contract, resize_matrix

HERE=Path(__file__).parent
LOG=Path('/mnt/d/00/工作日志/2026-10-27/内参文件解析')
data=json.loads((LOG/'02_参数解析与视频清单.json').read_text(encoding='utf-8'))
K=np.array(data['parsed']['camera_matrix'])
def contract(full, down, crop, k):
    return rectified_contract(dict(input_pixel_domain='rectified_pinhole',pixel_convention='integer_centers',
        fixed_processing_across_frames=True,K_rectified=k.tolist(),rectified_wh=full,
        downsampled_wh=down,crop_xywh=crop,training_wh=crop[2:],rectification_R=np.eye(3).tolist()))

rows=[]
for title, wh, down, crop in [
    ('landscape',[2704,1520],[2704,2028],[0,254,2704,1520]),
    ('portrait',[1080,1920],[2560,1920],[740,0,1080,1920])]:
    a=contract([4000,3000],down,crop,K)
    scaled=resize_matrix([4000,3000],[4096,3072])@K
    b=contract([4096,3072],down,crop,scaled)
    c=contract([4096,3072],down,crop,K)
    ka=np.array(a['K_integer_centers']);kb=np.array(b['K_integer_centers']);kc=np.array(c['K_integer_centers'])
    error=float(abs(ka-kb).max()); assert error<1e-9
    row=dict(orientation=title,actual_encoded_wh=wh,
        input_files=[x['file'] for x in data['videos'] if [x['width'],x['height']]==wh],
        assumption='Rectified K at 4000x3000 equals the first YAML K; uniform full-frame resize, centered crop, no extra rotation/EIS/zoom.',
        actual_preprocessing_recovered=False, candidate_only=True,
        rectified_wh=[4000,3000],downsampled_wh=down,crop_xywh=crop,
        K_conditional=ka.tolist(),
        alternative_4096_with_consistent_K_scaling_max_difference=error,
        alternative_4096_with_unscaled_original_K=kc.tolist(),
        unscaled_vs_scaled_delta_fx_fy_cx_cy=[float(kc[i,j]-ka[i,j]) for i,j in [(0,0),(1,1),(0,2),(1,2)]],
        physical_calibration_validated=False)
    if title=='landscape':
        old=np.array(data['current_hypothesis']['encoded_video_K_integer'])
        row['historical118_mapping_max_difference_px']=float(abs(ka-old).max())
    rows.append(row)
result=dict(user_clarification='Diagram numbers may be mistaken; sequence is intended. Landscape likely follows the same sequence.',
    diagram_numbers_are_not_measurements=True, conditional_examples=rows,
    conclusion='Correctly scaled 4:3 intermediate resolutions cancel. Under the same K and centered-ROI assumptions, the new landscape mapping equals historical118; changing order labels alone is not a new correction.',
    remaining_identifiability='Output size alone cannot recover rectification K, extra zoom/crop, EIS, or physical camera model.',
    new_training_started=False)
(HERE/'actual-video-size-audit120.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
