from pathlib import Path
import cv2
import numpy as np
import json

root = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/gaussianavatars-smplx-confidence-fresh-003/comparison')
cap = cv2.VideoCapture(str(root / 'fixed-pose-path.mp4'))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
selected = [0, 8, 17, 26, 35, 44, 53, 62, 71]
tiles = []
for idx in selected:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Missing frame {idx}')
    width = 600
    height = round(frame.shape[0] * width / frame.shape[1])
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    band = np.full((30, width, 3), 255, np.uint8)
    cv2.putText(band, f'video frame {idx} / 71; 002 | 003', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,20), 1, cv2.LINE_AA)
    tiles.append(np.concatenate([band, frame]))
cap.release()
sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1) for i in range(0,9,3)], axis=0)
cv2.imwrite(str(root / 'path-review-contact-sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY,95])
(root/'path-metadata.json').write_text(json.dumps({'frames':count,'fps':fps,'seconds':count/fps,'sampled_frames':selected,'scope':'Decoded temporal samples for visual review; not a claim of new-view ground truth.'},indent=2))
print(json.dumps({'frames':count,'fps':fps,'contact_sheet':str(root/'path-review-contact-sheet.jpg')}))
