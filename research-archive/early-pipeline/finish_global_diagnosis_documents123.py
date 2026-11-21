from pathlib import Path
import json,shutil
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');G=T/'shuolee/current/robust-portrait-122/video10-global-initializer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
for folder,name in [(T/'viewer','body-smplx.html'),(Path('D:/00/工作/查看器'),'progress.html')]:
    p=folder/'latest-result.json';x=read(p);x['latestDiagnostic'].update(round='123',reportUrl=name+'#smplx123-video10-global-ambiguity');p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=read(G/'visual-review/review.json')
cn={1229:'两个中心都在空调的出风口／支架附近。能辨认同一装置，但模糊和透视变化使精确位置仍不确定。',1249:'两个中心都在帽子标志附近。物体一致，但柔性材料及边缘模糊仍需核验。',1291:'两个中心都在带玻璃的画框内，可见高光和模糊。尚未认证为静止材料上的同一点。',1295:'两个中心都在烤箱顶部附近。第一张明显模糊，精确定位不确定。',1298:'两个中心都在地毯边缘／地砖缝附近。边缘形状、透视和模糊影响精确对应。',244:'留出点位于空调装置，与拟合族1229属于同一局部，物理独立性尚未证实。',1297:'留出点位于地砖交界附近，两个视角的局部布局一致。右图左边有人体，但查询中心仍在地面。模糊和重复瓷砖纹理尚不足以证明亚像素对应。'}
s=(G/'index.html').read_text(encoding='utf-8')
for row in review['families']:s=s.replace(row['note'],cn[row['family']])
for a,b in {'fewer_than12sharedFIT':'共享拟合点少于12','fewer_than12inliers':'内点少于12','no_usable_pose':'未求得有效姿态','low_parallax':'视差过小','homography_competes':'存在平面替代解释','local checks passed':'仅通过局部检查'}.items():s=s.replace(a,b)
(G/'index.html').write_text(s,encoding='utf-8')
review.update(actual_three_frame_floor_crop_review=True,three_frame_floor_crop='heldout_floor_1297_frames_698_725_752.png')
(G/'visual-review/review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'index.html',dest/'index.html');shutil.copy2(G/'visual-review/review.json',dest/'visual-review/review.json')
print('Viewer diagnostic contract repaired (round/reportUrl); report localized. Historical model untouched.')
