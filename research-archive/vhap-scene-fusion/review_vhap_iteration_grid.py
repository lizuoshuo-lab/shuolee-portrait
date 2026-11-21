"""Build an offline nine-panel review of a registered VHAP mesh iteration."""

from argparse import ArgumentParser
from contextlib import ExitStack
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from review_smplx_head_neck import mesh_overlay, panel, project


def load_mesh(archive, names, camera):
    if archive["names"].tolist() != names:
        raise ValueError("The mesh frame names must match the source archive")
    for key, value in camera.items():
        other = archive[key]
        if other.shape != value.shape or not np.allclose(other, value, rtol=0, atol=1e-8):
            raise ValueError(f"Mesh and source camera differ in {key}")
    v, f = archive["vertices_world"], archive["faces"]
    if (v.ndim != 3 or v.shape[0] != len(names) or v.shape[-1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f) or not np.issubdtype(f.dtype, np.integer)
            or f.min() < 0 or f.max() >= v.shape[1]):
        raise ValueError("Invalid mesh trajectory or triangle topology")
    tint = np.array([246, 210, 87])
    if "cloth_faces" in archive:
        cloth = archive["cloth_faces"]
        if (cloth.ndim != 1 or not np.issubdtype(cloth.dtype, np.integer)
                or np.any((cloth < 0) | (cloth >= len(f)))):
            raise ValueError("Invalid shirt/shoulder face IDs")
        tint = np.tile(tint, (len(f), 1))
        tint[cloth] = (60, 174, 241)
    return v, f, tint


def boundary(mask):
    mask = mask.astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) != 0


def projected_silhouette(shape, xy, faces):
    raster = np.zeros(shape, dtype=np.uint8)
    for triangle in np.rint(xy[faces] * 16).astype(np.int32):
        cv2.fillConvexPoly(raster, triangle, 1, lineType=cv2.LINE_8, shift=4)
    return raster != 0


def shoulder_observations(archive, names):
    if archive["names"].tolist() != names:
        raise ValueError("RTMW detections and source frames have different names")
    ids = [archive["joint_names"].tolist().index(name) for name in ("left_shoulder", "right_shoulder")]
    points = archive["keypoints"][:, ids]
    width = (archive["boxes"][:, 2] - archive["boxes"][:, 0])[:, None]
    valid = ((archive["scores"][:, ids] > .5) & (archive["flip_disagreement"][:, ids] < np.maximum(7., .06 * width))
             & np.isfinite(points).all(axis=-1))
    return points, valid


def observed_edges(rgb, alpha, row, hair, collar, shoulders, H):
    view = rgb.copy()
    silhouette = boundary(alpha >= 128)
    if "native_head_neck_bottom_px" in row:
        y, x = np.nonzero(silhouette)
        native = cv2.perspectiveTransform(np.column_stack((x, y))[None].astype(np.float64), np.linalg.inv(H))[0]
        keep = native[:, 1] < row["native_head_neck_bottom_px"] - 6
        silhouette[y[~keep], x[~keep]] = False
    view[silhouette] = (177, 175, 169)
    for edge, color in ((hair, (110, 255, 125)), (collar, (43, 175, 255))):
        if edge is not None:
            view[cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0] = color
    if shoulders is not None:
        points, eligible = shoulders
        used = np.flatnonzero(eligible)
        if len(used):
            warped = cv2.perspectiveTransform(points[None, used].astype(np.float64), H)[0]
            height, width = alpha.shape
            for side, (x, y) in zip(used, warped, strict=True):
                if 0 <= x < width and 0 <= y < height:
                    p = (round(x), round(y))
                    cv2.circle(view, p, 5, (255, 221, 81), -1, cv2.LINE_AA)
                    cv2.putText(view, ("L", "R")[side], (p[0] + 6, p[1] - 4), 0, .44, (255, 221, 81), 1, cv2.LINE_AA)
    return view, silhouette


def write_page(output, frames, initial_available, native_renders, restored_renders=False):
    options = "".join(f'<option value="{i}">{escape(r["name"])} · {r["timestamp_s"]:.2f} 秒</option>'
                      for i, r in enumerate(frames))
    payload = json.dumps([f'frames/{r["name"]}' for r in frames], ensure_ascii=False).replace("<", "\\u003c")
    note = "初始网格未提供。" if not initial_available else ""
    final_note = "底排两侧为 VHAP 原生 RGB 渲染。" if native_renders else "最后一格用网格颜色区分头颈与衣服。"
    if restored_renders:
        final_note = "底排是同一几何的 VHAP 原生 RGB：上版贴图、恢复旧贴图、87 帧重训贴图。"
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VHAP 九宫格逐帧评审</title>
<style>
body{{max-width:1160px;margin:auto;padding:22px 14px;background:#141719;color:#eef2f3;font:16px/1.55 system-ui,sans-serif}}
h1{{margin:0 0 10px;font-size:25px}} p{{max-width:75ch;color:#c0cbd0;margin:8px 0 14px}}
nav{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:16px 0}} button,select,a{{font:inherit;color:#d7f3fd}}
button,select{{background:#262d30;border:1px solid #5b696e;border-radius:5px;padding:7px 9px}}
button:disabled{{opacity:.4}} img{{display:block;width:100%;height:auto}}
</style>
<h1>VHAP 九宫格逐帧评审</h1>
<p>同一裁片、同一相机。观测格中绿色表示头发外缘，橙色表示衣领，蓝色 L/R 为 RTMW 肩点。轮廓对照的绿色、橙色和蓝色依次表示原图、上一版和本轮。{final_note}{note}</p>
<nav aria-label="帧选择"><button id="prev">上一帧</button><label>帧 <select id="pick">{options}</select></label>
<button id="next">下一帧</button><a id="save" download>下载当前九宫格</a></nav>
<img id="board" src="{escape(f'frames/{frames[0]["name"]}')}" width="1152" height="1152" alt="当前帧九宫格">
<script>
const paths={payload};
const pick=document.getElementById('pick'), board=document.getElementById('board'), save=document.getElementById('save');
const prev=document.getElementById('prev'), next=document.getElementById('next');
function show(i){{
  i=Math.min(paths.length-1,Math.max(0,i)); pick.value=String(i);
  board.src=paths[i]; save.href=paths[i];
  prev.disabled=i===0; next.disabled=i===paths.length-1;
}}
pick.onchange=()=>show(Number(pick.value));
prev.onclick=()=>show(Number(pick.value)-1);
next.onclick=()=>show(Number(pick.value)+1);
show(0);
</script></html>
"""
    (output / "index.html").write_text(page, encoding="utf-8")


def run(args):
    cv2.setNumThreads(1)
    if (args.initial_rgb is None) != (args.final_rgb is None):
        raise ValueError("Provide both initial and final VHAP RGB directories")
    native_renders = args.initial_rgb is not None
    if args.restored_rgb is not None and not native_renders:
        raise ValueError("Texture restoration comparison requires all three VHAP RGB directories")
    with ExitStack() as stack:
        source = stack.enter_context(ZipFile(args.frames))
        original_initialization = source.read("initialization.json")
        records = json.loads(original_initialization)["frames"]
        names = [r["name"] for r in records]
        if not names or len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError("Invalid source frame names")
        prior = stack.enter_context(np.load(args.prior, allow_pickle=False))
        camera = {k: prior[k] for k in ("native_K", "native_camera_to_world_rotation", "native_camera_center", "virtual_crop_homographies")}
        K, rotation, center, H = camera.values()
        if (K.shape, rotation.shape, center.shape, H.shape) != ((3, 3), (len(names), 3, 3), (len(names), 3), (len(names), 3, 3)):
            raise ValueError("Invalid registered camera array shape")
        final = stack.enter_context(np.load(args.final, allow_pickle=False))
        initial = stack.enter_context(np.load(args.initial, allow_pickle=False)) if args.initial else None
        meshes = [load_mesh(m, names, camera) if m is not None else None for m in (prior, initial, final)]
        outline = stack.enter_context(np.load(args.outline_targets, allow_pickle=False)) if args.outline_targets else None
        if outline is not None and (outline["names"].tolist() != names
                or outline["initialization_sha256"].item() != sha256(original_initialization).hexdigest()):
            raise ValueError("Outline targets came from a different crop sequence")
        hair, collars = (outline["hair_edge"], outline["collar"]) if outline is not None else (None, None)
        crop_hash, alpha_hash = (outline["crop_sha256"], outline["matte_sha256"]) if outline is not None else (None, None)
        shoulders = None
        if args.body_observations:
            shoulders = shoulder_observations(stack.enter_context(np.load(args.body_observations, allow_pickle=False)), names)

        args.output.mkdir(parents=True, exist_ok=False)
        boards = args.output / "frames"
        boards.mkdir()
        for i, (name, record) in enumerate(zip(names, records, strict=True)):
            V = np.asarray(record["virtual_from_native_cv"])
            projection = H[i] @ K
            if (not np.allclose(H[i], record["warp"], rtol=0, atol=1e-8)
                    or not np.allclose(projection, np.asarray(record["K_integer_pixel_centers"]) @ V, atol=1e-6)
                    or not np.allclose(rotation[i] @ V.T, record["world_from_virtual"], atol=1e-6)
                    or not np.allclose(center[i], record["camera_center_world"], atol=1e-6)):
                raise ValueError(f"Projection is not the registered source crop for {name}")
            rgb_bytes = source.read(f"images/{i:08d}.jpg")
            alpha_bytes = source.read(f"alpha_maps/{i:08d}.jpg")
            if outline is not None and (sha256(rgb_bytes).hexdigest() != crop_hash[i]
                    or sha256(alpha_bytes).hexdigest() != alpha_hash[i]):
                raise ValueError(f"Outline targets do not match the image or mask for {name}")
            rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            alpha = cv2.imdecode(np.frombuffer(alpha_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if rgb is None or alpha is None or rgb.shape[:2] != alpha.shape:
                raise ValueError(f"Invalid source image or mask for {name}")
            h, c = (hair[i], collars[i]) if hair is not None else (None, None)
            observed, source_edge = observed_edges(rgb, alpha, record, h, c,
                                                    tuple(v[i] for v in shoulders) if shoulders else None, H[i])
            overlays, masks = [], []
            for mesh in meshes:
                if mesh is None:
                    overlays.append(np.full_like(rgb, 27))
                    masks.append(None)
                    continue
                coordinates = (mesh[0][i] - center[i]) @ rotation[i]
                xy, depth = project(coordinates, projection)
                if (depth <= 1e-6).any():
                    raise ValueError(f"Mesh has nonpositive camera depth at {name}")
                overlays.append(mesh_overlay(rgb, coordinates, projection, mesh[1], tint=mesh[2]))
                masks.append(projected_silhouette(alpha.shape, xy, mesh[1]))
            contour = np.rint(rgb * .55).astype(np.uint8)
            for edge, color in ((source_edge, (115, 226, 117)), (boundary(masks[0]), (55, 160, 255)),
                                (boundary(masks[2]), (251, 212, 86))):
                contour[edge] = color
            if native_renders:
                render_images = []
                folders = (args.initial_rgb, args.restored_rgb, args.final_rgb) if args.restored_rgb else (args.initial_rgb, args.final_rgb)
                for folder in folders:
                    path = folder / f"{i:08d}.jpg"
                    rendered = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if rendered is None or rendered.shape != rgb.shape:
                        raise ValueError(f"VHAP RGB raster does not match its source crop: {path}")
                    render_images.append(rendered)
                left, right = render_images[0], render_images[-1]
                left_label, right_label = "VHAP RGB | initialization", "VHAP RGB | final"
            else:
                left = np.rint(rgb * (alpha[..., None] / 255.)).astype(np.uint8)
                mesh = meshes[-1]
                right = mesh_overlay(np.full_like(rgb, 28), (mesh[0][i] - center[i]) @ rotation[i],
                                     projection, mesh[1], tint=mesh[2])
                left_label, right_label = "Source foreground", "Final mesh | geometry shading"
            bottom_label = "Outlines | source / prior / final"
            if args.restored_rgb:
                contour = render_images[1]
                left_label, bottom_label, right_label = "VHAP RGB | previous texture", "VHAP RGB | restored old texture", "VHAP RGB | full sequence refit"
            labels = (f"{name} | Original", "Actual training mask", "Observed edges + shoulder detections",
                      "Prior mesh", "Round initialization" if initial is not None else "Initialization unavailable", "Final mesh",
                      left_label, bottom_label, right_label)
            images = (rgb, cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR), observed, *overlays, left, contour, right)
            tiles = [panel(img, label) for img, label in zip(images, labels, strict=True)]
            board = np.vstack([np.hstack(tiles[j:j + 3]) for j in (0, 3, 6)])
            if not cv2.imwrite(str(boards / name), board, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f"Could not write the review image for {name}")
    write_page(args.output, records, initial is not None, native_renders, args.restored_rgb is not None)
    print(f"Saved {len(records)} review boards and index.html to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("frames", "prior", "final", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("initial", "outline-targets", "body-observations", "initial-rgb", "restored-rgb", "final-rgb"):
        parser.add_argument(f"--{key}", type=Path)
    run(parser.parse_args())
