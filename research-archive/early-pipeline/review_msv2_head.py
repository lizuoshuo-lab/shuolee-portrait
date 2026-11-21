"""Read-only common-camera comparison of full-image and mixed head-view models.

Run in the completed baseline job's isolated Python runtime. --check-only never
imports torch; --frames 1 33 65 bounds the first real render. Display head crops
always come from a full 4x camera render, never an off-axis ROI camera render.
"""

import argparse
import gc
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VARIANTS = ("original", "interpolation2x", "full_swinir4x", "head_swinir4x")
STEMS = ("input", *VARIANTS)
LABELS = (
    "原图 · Lanczos 四倍参考",
    "原始完整图训练",
    "插值两倍完整图训练",
    "SwinIR 四倍完整图训练",
    "原图完整视角 + 头框四倍视角训练",
)
SCALE_NAMES = ((1, "original"), (2, "interpolation2x"), (4, "swinir4x"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def image_info(path):
    with Image.open(path) as image:
        require(
            image.format == "PNG" and image.mode == "RGB",
            "Expected lossless RGB PNG: " + str(path),
        )
        return list(image.size), hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def image_check(path, wh):
    with Image.open(path) as image:
        require(
            image.format == "PNG"
            and image.mode == "RGB"
            and list(image.size) == list(wh),
            "PNG dimensions/mode changed: " + str(path),
        )
        image.verify()


def unique(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), "Duplicate identities: " + key)
    return result


def matrix_equal(left, right):
    return np.allclose(np.asarray(left), np.asarray(right), atol=1e-10, rtol=0)


def implementation_sha(package):
    digest = hashlib.sha256()
    files = sorted(Path(package).glob("*.py"))
    require(bool(files), "Missing frozen pipeline package: " + str(package))
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_job(path):
    path = Path(path).resolve()
    job = read(path)
    root = Path(job["run_dir"]).resolve()
    require(
        path.parent == root, "Job must be read from its original frozen run directory"
    )
    package = root / "pipeline-source/msv2"
    require(
        implementation_sha(package) == job["implementation_sha256"],
        "Frozen public pipeline hash mismatch: " + str(root),
    )
    plan = read(root / "plan.json")
    source = {}
    protected = [
        path,
        root / "plan.json",
        root / "prepared/camera-inputs.json",
        root / "prepared/pixel-lineage.json",
    ]
    protected.extend(sorted(package.glob("*.py")))
    for item in plan["source_checks"]:
        relative = Path(item["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid native source path",
        )
        target = root / "source" / relative
        require(
            sha(target) == item["sha256"],
            "Frozen native source changed: " + str(target),
        )
        source[relative.as_posix()] = item["sha256"]
        protected.append(target)
    lineage = read(root / "prepared/pixel-lineage.json")
    require(lineage["passed"], "Native source pixel lineage is not verified")
    return dict(
        path=path,
        job=job,
        root=root,
        source=source,
        protected=protected,
        lineage=lineage,
    )


def check_cache(row):
    path = Path(row["image"]).parents[2] / "rgb-cache-1" / (row["frame"] + ".npy")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        require(
            array.dtype == np.uint8 and array.shape == (row["wh"][1], row["wh"][0], 3),
            "Actual camera cache header differs: " + str(path),
        )
    finally:
        array._mmap.close()
    return str(path)


def model_info(job, name):
    root = job["root"] / name
    execution = read(root / "execution.json")
    parity = read(root / "ply-parity.json")
    iterations = job["job"]["options"]["iterations"]
    require(
        execution["state"] == "training_completed"
        and execution["iterations"] == iterations
        and parity["passed"],
        "Model is not completed and export-verified: " + str(root),
    )
    path = root / f"model/point_cloud/iteration_{iterations}/point_cloud.ply"
    digest = sha(path)
    require(
        parity["sha256"] == digest and execution["ply_sha256"] == digest,
        "Frozen PLY identity mismatch: " + str(path),
    )
    job["protected"].extend(
        [
            path,
            root / "execution.json",
            root / "ply-parity.json",
            root / "training-inputs.json",
        ]
    )
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        job["protected"].append(root / "dataset/sparse/0" / name)
    return dict(
        path=str(path), sha256=digest, points=execution["points"], iterations=iterations
    )


def prepare(baseline_path, head_path):
    """Validate completed jobs and create deterministic display camera templates."""
    baseline, head = check_job(baseline_path), check_job(head_path)
    require(baseline["root"] != head["root"], "Use distinct baseline and head runs")
    require(
        baseline["source"] == head["source"],
        "Jobs do not use the same frozen native renderer/model source",
    )
    require(
        baseline["job"]["options"].get("sr_region", "full") == "full"
        and head["job"]["options"].get("sr_region") == "head",
        "Expected one full-image baseline and one mixed head-view job",
    )
    require(
        baseline["job"]["options"]["iterations"]
        == head["job"]["options"]["iterations"],
        "Training iteration counts differ",
    )
    base_rows = {}
    for scale, name in SCALE_NAMES:
        base_rows[scale] = read(baseline["root"] / name / "training-inputs.json")[
            "records"
        ]
    sources = base_rows[1]
    source_map = unique(sources, "frame")
    require(bool(sources), "Baseline has no source frames")
    sr_path = head["root"] / "swinir4x/superresolution.json"
    sr = read(sr_path)
    require(
        sr["state"] == "completed" and sr["sr_region"] == "head",
        "Head SR input preparation is not complete",
    )
    sr_map = unique(sr["records"], "name")
    mixed = read(head["root"] / "swinir4x/training-inputs.json")["records"]
    mixed_map = unique(mixed, "name")
    fulls = unique([row for row in mixed if row["view_type"] == "full"], "source_frame")
    require(
        set(fulls) == set(source_map)
        and set(sr_map) == {row["name"] for row in sources},
        "Jobs do not have exactly the same source frame groups",
    )
    by_scale = {scale: unique(rows, "frame") for scale, rows in base_rows.items()}
    require(
        all(set(rows) == set(source_map) for rows in by_scale.values()),
        "Baseline variants do not have identical source views",
    )
    head["protected"].append(sr_path)
    models = {
        name: model_info(baseline, native)
        for name, (_, native) in zip(VARIANTS[:3], SCALE_NAMES)
    }
    models[VARIANTS[3]] = model_info(head, "swinir4x")
    templates, exported = [], {}
    used_head_names = set()
    for index, source in enumerate(sources):
        frame, name = source["frame"], source["name"]
        require(
            Path(name).name == name and Path(frame).name == frame,
            "Unsafe frame/name identity",
        )
        native_wh = source["wh"]
        native_path = Path(source["native_image"])
        native_sha = source["native_sha256"]
        require(
            sha(native_path) == native_sha == source["source_sha256"],
            "Native baseline PNG identity differs",
        )
        require(
            sha(source["image"]) == native_sha,
            "Original model's full1 training PNG differs from the native source",
        )
        wh, rgb_sha = image_info(native_path)
        require(wh == native_wh, "Native source RGB size differs")
        baseline["protected"].extend([native_path, Path(source["image"])])
        full = fulls[frame]
        require(
            full["source_name"] == name
            and full["source_group"] == frame
            and full["native_sha256"] == native_sha,
            "Head full view source identity differs",
        )
        require(
            full["wh"] == native_wh
            and full["split"] == source["split"]
            and matrix_equal(full["K"], source["K"])
            and matrix_equal(full["w2c"], source["w2c"]),
            "Head full view changed original K/pose/split/dimensions",
        )
        for key in ("native_image", "image"):
            path = Path(full[key])
            require(
                sha(path) == native_sha,
                "Head job's full1/native PNG differs from baseline",
            )
            head["protected"].append(path)
        for scale, rows in by_scale.items():
            row = rows[frame]
            require(
                row["name"] == name
                and row["native_sha256"] == native_sha
                and row["split"] == source["split"]
                and row["wh"] == [scale * x for x in native_wh],
                "Baseline scaled view identity/dimensions differ",
            )
            require(
                matrix_equal(row["w2c"], source["w2c"])
                and matrix_equal(
                    row["K"], np.diag([scale, scale, 1]) @ np.asarray(source["K"])
                ),
                "Baseline scaled K/pose differs",
            )
        item = sr_map[name]
        require(
            item["native_sha256"] == item["output_sha256"] == native_sha,
            "Head SR full-image PNG is not the unchanged native image",
        )
        head_rows = []
        for crop in item["heads"]:
            bbox = crop["bbox_native"]
            require(
                len(bbox) == 4 and all(type(v) is int for v in bbox),
                "Head bbox must use integer exclusive corners",
            )
            x0, y0, x1, y1 = bbox
            require(
                0 <= x0 < x1 <= native_wh[0] and 0 <= y0 < y1 <= native_wh[1],
                "Head bbox outside source image",
            )
            row = mixed_map[crop["name"]]
            used_head_names.add(crop["name"])
            expected_k = (
                np.diag([4, 4, 1])
                @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
                @ np.asarray(source["K"])
            )
            require(
                row["view_type"] == "head"
                and row["native_bbox"] == bbox
                and row["source_frame"] == frame
                and row["source_name"] == name
                and row["source_group"] == frame,
                "Head training view source/bbox identity differs",
            )
            require(
                row["split"] == source["split"]
                and row["native_sha256"] == native_sha
                and row["wh"] == [4 * (x1 - x0), 4 * (y1 - y0)]
                and matrix_equal(row["K"], expected_k)
                and matrix_equal(row["w2c"], source["w2c"]),
                "Head actual training camera K/pose/split/dimensions differ",
            )
            path = Path(row["image"])
            require(
                sha(path) == crop["output_sha256"] == row["source_sha256"],
                "Head SR PNG changed",
            )
            image_check(path, row["wh"])
            receipt = crop["receipt"]
            require(
                receipt["bbox_native"] == bbox
                and receipt["source_rgb_sha256"] == rgb_sha
                and receipt["no_source_downsampling"]
                and receipt["source_full_rgb_unchanged"],
                "Head SR pixel lineage differs",
            )
            head["protected"].append(path)
            head_rows.append(row)
        exported[frame] = {
            "input": baseline["root"] / "export/frames" / frame / "input.png",
            "original": baseline["root"] / "export/frames" / frame / "original.png",
            "interpolation2x": baseline["root"]
            / "export/frames"
            / frame
            / "interpolation2x.png",
            "full_swinir4x": baseline["root"]
            / "export/frames"
            / frame
            / "swinir4x.png",
            "head_swinir4x": head["root"] / "export/frames" / frame / "swinir4x.png",
        }
        for path in exported[frame].values():
            image_check(path, native_wh)
            baseline["protected"].append(path)
        require(
            image_info(exported[frame]["input"])[1] == rgb_sha,
            "Published full input is not the same native RGB",
        )
        selected = head_rows[0] if head_rows else None
        common = by_scale[4][frame]
        template = dict(
            frame=frame,
            source_name=name,
            source_index=source.get("source_index"),
            source_ordinal=index + 1,
            split=source["split"],
            source_wh=native_wh,
            native_png_sha256=native_sha,
            source_rgb_sha256=rgb_sha,
            head_count=len(head_rows),
            selected_head_name=selected["name"] if selected else None,
            selection="first SR-manifest head only; other heads are not shown"
            if len(head_rows) > 1
            else "only detected head"
            if selected
            else "no head; no invented display crop",
            bbox_native=selected["native_bbox"] if selected else None,
            bbox_render_4x=[4 * v for v in selected["native_bbox"]]
            if selected
            else None,
            common_full4_camera=dict(
                wh=common["wh"],
                K=common["K"],
                w2c=common["w2c"],
                cache=check_cache(common),
            ),
            head_training_camera=dict(
                name=selected["name"],
                wh=selected["wh"],
                K=selected["K"],
                w2c=selected["w2c"],
                cache=check_cache(selected),
            )
            if selected
            else None,
        )
        templates.append(template)
    require(
        used_head_names == {row["name"] for row in mixed if row["view_type"] == "head"},
        "Unaccounted head training views",
    )
    protected_paths = set(baseline["protected"] + head["protected"])
    protected = {str(path): sha(path) for path in sorted(protected_paths)}
    template = dict(
        schema="msv2-head-review-camera-template-v1",
        source_frames=len(sources),
        displayed_frames=sum(r["head_count"] > 0 for r in templates),
        records=templates,
        full_display_scale=1,
        head_display_scale=4,
        display_method="Render every frozen model on the same complete baseline 4x camera, then crop the integer head bbox. No ROI-camera substitution.",
        camera_limit="Crop K preserves point projections; native off-axis FOV covariance clamp can differ. Optional crop-camera diagnostics are separate from display.",
    )
    context = dict(
        baseline_job=str(baseline["path"]),
        baseline_job_sha256=sha(baseline["path"]),
        head_job=str(head["path"]),
        head_job_sha256=sha(head["path"]),
        baseline_implementation_sha256=baseline["job"]["implementation_sha256"],
        head_implementation_sha256=head["job"]["implementation_sha256"],
        models=models,
        template_sha256=value_sha(template),
        protected_before=protected,
    )
    return baseline, head, by_scale, mixed_map, exported, template, context


def choose_frames(values, rows):
    if not values:
        return {row["frame"] for row in rows}
    by_ordinal = {str(row["source_ordinal"]): row["frame"] for row in rows}
    names = {row["frame"] for row in rows}
    chosen = set()
    for value in values:
        for part in value.split("/"):
            frame = by_ordinal.get(part, part)
            require(frame in names, "Unknown frame name/source ordinal: " + part)
            chosen.add(frame)
    return chosen


def find_font(supplied):
    for path in (
        [supplied]
        if supplied
        else [
            Path("/mnt/c/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
    ):
        if path and path.is_file():
            return str(path)
    raise FileNotFoundError("Supply --font with Chinese glyphs")


def board(folder, row, font):
    images = []
    for stem in STEMS:
        with Image.open(folder / (stem + "_head_4x.png")) as image:
            images.append(image.copy())
    width, height = images[0].size
    require(
        all(image.size == (width, height) for image in images),
        "Head panel sizes differ",
    )
    pad, header = 12, 70
    canvas = Image.new(
        "RGB", (3 * (width + 2 * pad), 2 * (height + header + pad)), "#f0f2f5"
    )
    draw = ImageDraw.Draw(canvas)
    title, detail = ImageFont.truetype(font, 18), ImageFont.truetype(font, 16)
    for index, (image, label) in enumerate(zip(images, LABELS)):
        x = (index % 3) * (width + 2 * pad) + pad
        y = (index // 3) * (height + header + pad)
        draw.text((x, y + 7), label, fill="#20242a", font=title)
        draw.text(
            (x, y + 36),
            f"{row['frame']} · {row['split']} · {width}×{height}",
            fill="#505761",
            font=detail,
        )
        canvas.paste(image, (x, y + header))
        image.close()
    x, y = 2 * (width + 2 * pad) + pad, height + header + pad
    draw.multiline_text(
        (x, y + header),
        "同一完整四倍相机渲染后裁剪\n原图仅作 Lanczos 四倍参考\n无模型或优化器更新\n不预先认定质量通过",
        fill="#505761",
        font=detail,
        spacing=8,
    )
    canvas.save(folder / "head_compare.png")
    canvas.close()


def expected_assets(row):
    x0, y0, x1, y1 = row["bbox_render_4x"]
    expected = {stem + "_head_4x.png": [x1 - x0, y1 - y0] for stem in STEMS}
    expected.update({stem + ".png": row["source_wh"] for stem in STEMS})
    expected["head_compare.png"] = [3 * (x1 - x0 + 24), 2 * (y1 - y0 + 82)]
    return expected


def validate_checkpoint(record, state, rows, complete=False):
    mapping = {row["frame"]: row for row in rows}
    require(
        set(state["records"]) <= set(mapping),
        "Checkpoint contains unknown/no-head frames",
    )
    for frame, entry in state["records"].items():
        expected = expected_assets(mapping[frame])
        for filename, digest in entry["sha256"].items():
            require(
                filename in expected
                or filename in {stem + "_full_4x.png" for stem in VARIANTS},
                "Unknown checkpoint asset",
            )
            path = record / "frames" / frame / filename
            require(sha(path) == digest, "Saved comparison asset changed: " + str(path))
            image_check(
                path,
                expected.get(filename, mapping[frame]["common_full4_camera"]["wh"]),
            )
        if complete:
            require(
                set(expected) <= set(entry["sha256"]),
                "A completed frame is missing images: " + frame,
            )
    if complete:
        require(
            set(state["records"]) == set(mapping),
            "The comparison is not complete for every detected-head source",
        )
    for frame, entry in state.get("crop_camera_diagnostics", {}).items():
        require(
            frame in mapping and entry["not_used_for_display"],
            "Invalid independent camera diagnostic",
        )
        for filename, digest in entry["sha256"].items():
            require(
                Path(filename).name == filename, "Unsafe camera diagnostic filename"
            )
            path = record / "crop_camera_diagnostic" / frame / filename
            require(sha(path) == digest, "Saved camera diagnostic changed")


def unchanged(context):
    for path, expected in context["protected_before"].items():
        require(
            sha(path) == expected,
            "Protected source/model/camera/export changed: " + path,
        )


def publish(record, state, rows):
    require(
        state["state"] == "completed" and state["protected_unchanged"],
        "Only a complete, protected comparison can be published",
    )
    options = "".join(
        f'<option value="{escape(row["frame"])}">{escape(row["frame"])} · {row["split"]}'
        + (f" · 展示首个/{row['head_count']}头框" if row["head_count"] > 1 else "")
        + "</option>"
        for row in rows
    )
    data = {
        row["frame"]: {
            "split": row["split"],
            "head_count": row["head_count"],
            "bbox": row["bbox_native"],
        }
        for row in rows
    }
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>MSV2 头部四倍对照</title>
<style>body{max-width:1400px;margin:24px auto;padding:0 20px;font:16px/1.7 system-ui;background:#f4f5f7;color:#20242a}.controls{display:flex;flex-wrap:wrap;gap:15px;align-items:center}select{font-size:16px;padding:6px}.scroll{overflow:auto;background:#e9ecf0;margin:18px 0}.scroll img{display:block;max-width:none}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid img{width:100%;height:auto}figure{margin:0}a{color:#2266aa}#links{display:flex;flex-wrap:wrap;gap:15px}[hidden]{display:none!important}</style>
<h1>MSV2 · 头部框四倍训练对照</h1><p>源原图参考与四个独立训练结果。头部图由冻结模型在同一完整四倍相机网格重新渲染后裁出；没有用 ROI 相机近似替代展示，也没有放大旧渲染图。完整图沿用各 run 导出的原始分辨率 PNG。此页不预先认定质量通过。</p>
<div class="controls"><label>源帧 <select id="frame">OPTIONS</select></label><label>区域 <select id="region"><option value="head">头部 · 共同四倍相机</option><option value="full">完整图 · 原始分辨率</option></select></label><label id="zoom-label">显示 <select id="zoom"><option value="1">1:1 实际像素</option><option value="2">200%</option></select></label></div><p id="note"></p><div class="scroll" id="head-panel"><a id="board-link" target="_blank" rel="noopener"><img id="board"></a></div><div class="grid" id="full-panel" hidden></div><div id="links"></div><p><a href="head_review.json">只读渲染核查</a> · <a href="head_view_templates.json">实际相机与头部模板</a></p>
<script>const frames=DATA,names=STEMS,labels=LABELS;const f=document.querySelector('#frame'),r=document.querySelector('#region'),z=document.querySelector('#zoom'),img=document.querySelector('#board');function zoom(){img.style.width=(img.naturalWidth*Number(z.value))+'px'}function update(){const head=r.value==='head',base='frames/'+f.value+'/',detail=frames[f.value];document.querySelector('#head-panel').hidden=!head;document.querySelector('#full-panel').hidden=head;document.querySelector('#zoom-label').hidden=!head;document.querySelector('#note').textContent=detail.split+' · 原图头框 ['+detail.bbox.join(', ')+']'+(detail.head_count>1?' · 多头框只展示SR记录中的首个':'')+(head?' · 五面板为无损PNG，三列两行；可横向滚动。':' · 完整图为原始分辨率；点击可单独打开。');document.querySelector('#links').replaceChildren();document.querySelector('#full-panel').replaceChildren();names.forEach((name,i)=>{const path=base+name+(head?'_head_4x':'')+'.png',a=document.createElement('a');a.href=path;a.target='_blank';a.rel='noopener';a.textContent=labels[i];document.querySelector('#links').append(a);if(!head){const fig=document.createElement('figure'),cap=document.createElement('figcaption'),link=document.createElement('a'),photo=document.createElement('img');cap.textContent=labels[i];link.href=path;link.target='_blank';link.rel='noopener';photo.src=path;link.append(photo);fig.append(cap,link);document.querySelector('#full-panel').append(fig)}});if(head){const path=base+'head_compare.png';document.querySelector('#board-link').href=path;img.onload=zoom;img.src=path;if(img.complete)zoom()}history.replaceState(null,'','#'+new URLSearchParams({frame:f.value,region:r.value,zoom:z.value}))}const q=new URLSearchParams(location.hash.slice(1));if(Object.hasOwn(frames,q.get('frame')))f.value=q.get('frame');r.value=q.get('region')==='full'?'full':'head';z.value=q.get('zoom')==='2'?'2':'1';f.onchange=r.onchange=z.onchange=update;update();</script></html>"""
    html = (
        html.replace("OPTIONS", options)
        .replace("DATA", json.dumps(data, ensure_ascii=False))
        .replace("STEMS", json.dumps(STEMS))
        .replace("LABELS", json.dumps(LABELS, ensure_ascii=False))
    )
    path = record / "查看头部对照.html"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-job", required=True, type=Path)
    parser.add_argument("--head-job", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument(
        "--frames",
        nargs="*",
        help="Exact frame names or 1-based source ordinals, e.g. 1 33 65",
    )
    parser.add_argument(
        "--keep-full-frames",
        nargs="*",
        help="Explicitly selected model full4 PNG evidence; default saves none",
    )
    parser.add_argument(
        "--diagnose-crop-camera",
        action="store_true",
        help="Separate native head-camera approximation diagnostic, never used for display",
    )
    parser.add_argument("--font", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only CPU input/checkpoint validation; does not import torch",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="CPU validation and page refresh; requires completed rendering",
    )
    args = parser.parse_args()
    require(
        not (args.check_only and args.publish_only), "Choose check-only or publish-only"
    )
    baseline, head, rows_by_scale, mixed_map, exported, template, context = prepare(
        args.baseline_job, args.head_job
    )
    record = args.record.resolve()
    require(
        all(
            not record.is_relative_to(job["root"])
            and not job["root"].is_relative_to(record)
            for job in (baseline, head)
        ),
        "Keep the review outside both protected run trees",
    )
    rows = [row for row in template["records"] if row["head_count"]]
    require(bool(rows), "No detected head views exist for comparison")
    chosen = (
        choose_frames(args.frames, template["records"])
        if args.frames
        else {row["frame"] for row in rows}
    )
    require(
        chosen <= {row["frame"] for row in rows},
        "Requested source has no detected head; no display box will be invented",
    )
    keep = (
        choose_frames(args.keep_full_frames, template["records"])
        if args.keep_full_frames
        else set()
    )
    require(
        keep <= chosen,
        "Keep-full frames must also be selected for this render invocation",
    )
    require(len(keep) <= 3, "Keep-full evidence is bounded to at most three source frames")
    manifest, template_path = (
        record / "head_review.json",
        record / "head_view_templates.json",
    )
    if manifest.exists():
        state = read(manifest)
        require(
            state["context"] == context and state["helper_sha256"] == sha(__file__),
            "Frozen evidence or report implementation changed since checkpoint",
        )
        require(read(template_path) == template, "Saved camera template changed")
        validate_checkpoint(record, state, rows, complete=state["state"] == "completed")
    else:
        state = dict(
            schema="msv2-readonly-head-review-v1",
            state="not_started",
            started=now(),
            context=context,
            helper_sha256=sha(__file__),
            model_updates=0,
            optimizer_updates=0,
            quality_accepted=False,
            records={},
            crop_camera_diagnostics={},
            source_frames=template["source_frames"],
            display_frames=len(rows),
            no_head_frames=template["source_frames"] - len(rows),
            full_png_method="Byte-verified copies of existing original-resolution run/export/frames PNGs",
            head_png_method=template["display_method"],
            source_reference="Lanczos4 original RGB reference; not super-resolution ground truth",
            covariance_limit=template["camera_limit"],
            page_ui_verified=False,
        )
    if args.check_only or args.publish_only:
        unchanged(context)
        if args.publish_only:
            validate_checkpoint(record, state, rows, complete=True)
            publish(record, state, rows)
        print(
            "HEAD_REVIEW_CPU_VALIDATED",
            json.dumps(
                dict(
                    source_frames=template["source_frames"],
                    display_frames=len(rows),
                    checkpoint_state=state["state"],
                    completed=state["state"] == "completed",
                    cuda_initialized=False,
                )
            ),
            flush=True,
        )
        return
    font = find_font(args.font)
    record.mkdir(parents=True, exist_ok=True)
    save(template_path, template)
    state.update(state="rendering", updated=now())
    save(manifest, state)
    sys.path.insert(0, str(baseline["root"] / "pipeline-source"))
    from msv2.engine import Engine, LazyCamera, save_image

    engine = Engine(baseline["path"])
    torch = engine.runtime()
    torch.cuda.set_per_process_memory_fraction(
        min(0.80, engine.config["gpu_memory_fraction"])
    )
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, device="cuda")
    try:
        for row in rows:
            frame = row["frame"]
            if frame not in chosen:
                continue
            folder = record / "frames" / frame
            folder.mkdir(parents=True, exist_ok=True)
            entry = state["records"].setdefault(frame, dict(sha256={}))
            for stem, src in exported[frame].items():
                filename = stem + ".png"
                if filename not in entry["sha256"]:
                    destination = folder / filename
                    shutil.copy2(src, destination)
                    require(
                        sha(destination) == context["protected_before"][str(src)],
                        "Full PNG copy failed byte verification",
                    )
                    entry["sha256"][filename] = sha(destination)
            filename = "input_head_4x.png"
            if filename not in entry["sha256"]:
                with Image.open(exported[frame]["input"]) as image:
                    enlarged = image.resize(
                        tuple(row["common_full4_camera"]["wh"]),
                        Image.Resampling.LANCZOS,
                    )
                    enlarged.crop(row["bbox_render_4x"]).save(folder / filename)
                    enlarged.close()
                entry["sha256"][filename] = sha(folder / filename)
            save(manifest, state)
        common_map = rows_by_scale[4]
        for variant in VARIANTS:
            model = GaussianModel(3)
            model.load_ply(context["models"][variant]["path"])
            model.active_sh_degree = 3
            with torch.no_grad():
                for uid, row in enumerate(rows):
                    frame = row["frame"]
                    if frame not in chosen:
                        continue
                    entry, folder = state["records"][frame], record / "frames" / frame
                    filename = variant + "_head_4x.png"
                    full_name = variant + "_full_4x.png"
                    diagnostic_needed = (
                        args.diagnose_crop_camera
                        and variant == VARIANTS[-1]
                        and frame not in state["crop_camera_diagnostics"]
                    )
                    if (
                        filename in entry["sha256"]
                        and (frame not in keep or full_name in entry["sha256"])
                        and not diagnostic_needed
                    ):
                        continue
                    camera = LazyCamera(common_map[frame], uid)
                    full = render(camera, model, pipe, background)["render"].clamp(0, 1)
                    x0, y0, x1, y1 = row["bbox_render_4x"]
                    cropped = full[:, y0:y1, x0:x1].contiguous()
                    if filename not in entry["sha256"]:
                        save_image(folder / filename, cropped)
                        entry["sha256"][filename] = sha(folder / filename)
                    if frame in keep and full_name not in entry["sha256"]:
                        save_image(folder / full_name, full)
                        entry["sha256"][full_name] = sha(folder / full_name)
                    # Release the large full canvas before an optional ROI render.
                    display_cpu = (
                        cropped.cpu().numpy().copy() if diagnostic_needed else None
                    )
                    del full, cropped, camera
                    torch.cuda.empty_cache()
                    if diagnostic_needed:
                        camera = LazyCamera(mixed_map[row["selected_head_name"]], uid)
                        approximate = render(camera, model, pipe, background)[
                            "render"
                        ].clamp(0, 1)
                        actual = approximate.cpu().numpy()
                        delta = np.abs(actual - display_cpu)
                        diagnostic = record / "crop_camera_diagnostic" / frame
                        diagnostic.mkdir(parents=True, exist_ok=True)
                        save_image(diagnostic / "native_head_camera.png", approximate)
                        difference = np.rint(
                            np.clip(delta.max(axis=0), 0, 1) * 255
                        ).astype(np.uint8)
                        Image.fromarray(difference).save(
                            diagnostic / "absolute_max_rgb_difference.png"
                        )
                        state["crop_camera_diagnostics"][frame] = dict(
                            model=variant,
                            not_used_for_display=True,
                            reference="Complete common4 camera render, cropped after native rendering",
                            candidate="Actual native off-axis head training camera",
                            max_abs=float(delta.max()),
                            mean_abs=float(delta.mean()),
                            rms=float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                            bitwise_equal=bool(np.array_equal(actual, display_cpu)),
                            equivalence_claimed=False,
                            sha256={
                                name: sha(diagnostic / name)
                                for name in (
                                    "native_head_camera.png",
                                    "absolute_max_rgb_difference.png",
                                )
                            },
                        )
                        del camera, approximate, actual, delta, display_cpu
                        torch.cuda.empty_cache()
                    state.update(updated=now())
                    save(manifest, state)
                    print("HEAD_COMMON_FULL4_RENDER", variant, frame, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for row in rows:
            frame = row["frame"]
            if (
                frame in chosen
                and "head_compare.png" not in state["records"][frame]["sha256"]
            ):
                folder = record / "frames" / frame
                board(folder, row, font)
                state["records"][frame]["sha256"]["head_compare.png"] = sha(
                    folder / "head_compare.png"
                )
        complete = len(state["records"]) == len(rows) and all(
            set(expected_assets(row)) <= set(state["records"][row["frame"]]["sha256"])
            for row in rows
        )
        validate_checkpoint(record, state, rows, complete=complete)
        unchanged(context)
        state.update(
            state="completed" if complete else "partial",
            updated=now(),
            protected_unchanged=True,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        if complete:
            state["completed"] = now()
        save(manifest, state)
        if complete:
            publish(record, state, rows)
    except Exception as error:
        state.update(state="failed", updated=now(), error=str(error))
        save(manifest, state)
        raise
    print("HEAD_REVIEW_CHECKPOINT", len(chosen), state["state"], flush=True)


if __name__ == "__main__":
    main()
