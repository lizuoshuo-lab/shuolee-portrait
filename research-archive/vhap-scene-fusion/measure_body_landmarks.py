"""Measure native RTMW landmarks with the original mirrored-image repeatability check."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose
from rtmlib.visualization.skeleton.coco133 import coco133


class BodyPose(RTMPose):
    def __init__(self, assets):
        normalization = json.loads((assets / "normalization.json").read_bytes())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.onnx_model = str(assets / "rtmw-l-384x288.onnx")
        self.session = ort.InferenceSession(self.onnx_model, sess_options=options, providers=["CPUExecutionProvider"])
        self.model_input_size = (288, 384)
        self.mean = np.asarray(normalization["mean"])
        self.std = np.asarray(normalization["std"])
        self.backend = "onnxruntime"
        self.device = "cpu"
        self.to_openpose = False


def run(scene, assets, output):
    pose = BodyPose(assets)
    keys = coco133["keypoint_info"]
    names = [keys[i]["name"] for i in range(133)]
    flip = [names.index(keys[i]["swap"]) if keys[i]["swap"] else i for i in range(133)]
    points, scores, disagreements, boxes, frames = [], [], [], [], []
    for row in json.loads((scene / "source-frames.json").read_bytes())["frames"]:
        name = row["name"]
        image = cv2.imread(str(scene / "orig_undist/images" / name))
        mask = cv2.imread(str(scene / "mask_undist/images" / name), 0)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Source and mask do not match at {name}")
        count, _, stats, _ = cv2.connectedComponentsWithStats((mask >= 128).astype(np.uint8))
        if count < 2:
            raise ValueError(f"No tracked actor at {name}")
        x, y, w, h, _ = stats[1 + np.argmax(stats[1:, 4])]
        height, width = mask.shape
        margin = .03 * max(w, h)
        box = np.clip([x-margin, y-margin, x+w+margin, y+h+margin], 0, [width-1, height-1]*2)
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        front, confidence = pose(color, bboxes=box[None])
        mirror = box.copy()
        mirror[[0, 2]] = width - 1 - box[[2, 0]]
        backward, other_conf = pose(color[:, ::-1].copy(), bboxes=mirror[None])
        aligned = backward[0][flip].copy()
        aligned[:, 0] = width - 1 - aligned[:, 0]
        observed = (front[0] + aligned) / 2
        if not np.isfinite(observed).all():
            raise ValueError(f"Nonfinite body landmarks at {name}")
        points.append(observed)
        scores.append(np.minimum(confidence[0], other_conf[0][flip]))
        disagreements.append(np.linalg.norm(front[0] - aligned, axis=1))
        boxes.append(box)
        frames.append(name)
        print(name, "shoulders", observed[5:7].round(1).tolist(), flush=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, names=frames, keypoints=points, scores=scores,
                            flip_disagreement=disagreements, boxes=boxes, joint_names=names)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    for key in ("scene", "assets", "output"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    cv2.setNumThreads(1)
    args = cli.parse_args()
    run(args.scene, args.assets, args.output)
