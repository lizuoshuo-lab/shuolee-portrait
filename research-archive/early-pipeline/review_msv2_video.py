"""Inspect frozen MSV2 models on one common high-resolution camera grid.

This local postprocessor changes report assets only. It never initializes an
optimizer, updates a model, alters a training image, or uses display boxes in
training/camera recovery. Run with the completed job's isolated CUDA Python.
"""

from pathlib import Path
from types import SimpleNamespace
from html import escape
import argparse
import gc
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEMS = ("input", "original", "interpolation2x", "swinir4x")
LABELS = (
    "视频原图 · Lanczos 放大参考",
    "原始图训练 · 四倍渲染",
    "插值两倍训练 · 四倍渲染",
    "SwinIR 四倍训练 · 四倍渲染",
)
TRAINING_LABELS = {1: "原始完整图", 2: "Lanczos 插值两倍", 4: "SwinIR 超分四倍"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--frames", nargs="*", help="Optional exact frame names for a bounded first inspection")
    parser.add_argument("--font", type=Path, help="Optional TrueType/OpenType font with Chinese glyphs")
    parser.add_argument("--check-only", action="store_true", help="Validate completed image assets without initializing CUDA")
    parser.add_argument("--publish-only", action="store_true", help="Validate completed image assets and refresh HTML without CUDA")
    return parser.parse_args()


def font_path(supplied):
    candidates = [supplied] if supplied else [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path and path.is_file():
            return str(path)
    raise FileNotFoundError("Provide --font with a font containing Chinese glyphs")


def protected_files(engine, record, rows, names, sha):
    """Hash original evidence that this report must leave unchanged."""
    paths = [engine.job_file, engine.root / "plan.json", record / "comparison.json"]
    for name in names.values():
        paths.extend([
            engine.root / name / f"model/point_cloud/iteration_{engine.iterations}/point_cloud.ply",
            engine.root / name / "training-inputs.json",
            engine.root / name / "execution.json",
            engine.root / name / "ply-parity.json",
            record / name / "metrics.json",
            record / name / "delivery.json",
            record / name / "cameras.json",
        ])
    for row in rows:
        for stem in STEMS:
            paths.append(record / "frames" / row["frame"] / (stem + ".png"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Completed original evidence is missing: " + str(path))
    return {str(path): sha(path) for path in paths}


def validate_regions(regions, rows, record):
    if regions.get("used_for_training") or regions.get("used_for_camera_recovery"):
        raise ValueError("The supplied boxes must be explicitly display-only")
    entries = regions["records"]
    lookup = {entry["frame"]: entry for entry in entries}
    if len(lookup) != len(entries) or set(lookup) != {row["frame"] for row in rows}:
        raise ValueError("Display boxes must identify exactly the completed source frames")
    for row in rows:
        entry = lookup[row["frame"]]
        width, height = row["wh"]
        if entry["wh"] != row["wh"]:
            raise ValueError("Display boxes use a different image coordinate domain")
        x0, y0, x1, y1 = entry["face"]
        if any(type(value) is not int for value in entry["face"]) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Invalid face display rectangle: " + row["frame"])
        with Image.open(record / "frames" / row["frame"] / "input.png") as image:
            if image.size != (width, height):
                raise ValueError("Report source RGB dimensions changed")
            actual = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
        if actual != entry["image_rgb_sha256"]:
            raise ValueError("Display boxes do not belong to the exact original RGB: " + row["frame"])
    return lookup


def board(folder, row, font):
    images = []
    for stem in STEMS:
        with Image.open(folder / (stem + "_face_4x.png")) as image:
            images.append(image.convert("RGB"))
    if len({image.size for image in images}) != 1:
        raise ValueError("All faces must use the same camera region and dimensions")
    width, height = images[0].size
    pad, header = 16, 78
    canvas = Image.new("RGB", (2 * (width + 2 * pad), 2 * (height + header + pad)), "#f0f2f5")
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype(font, 20)
    small = ImageFont.truetype(font, 17)
    for index, image in enumerate(images):
        x = (index % 2) * (width + 2 * pad) + pad
        y = (index // 2) * (height + header + pad)
        draw.text((x, y + 8), LABELS[index], font=large, fill="#20242a")
        draw.text((x, y + 39), f"{row['frame']} · {'留出' if row['split'] == 'test' else '训练'} · {width}×{height}", font=small, fill="#505761")
        canvas.paste(image, (x, y + header))
        image.close()
    canvas.save(folder / "face_compare.png")
    canvas.save(folder / "face_compare.jpg", quality=97, subsampling=0)
    canvas.close()


def validate_outputs(state, rows, record, sha):
    if set(state["records"]) != {row["frame"] for row in rows}:
        raise ValueError("Not every source frame has a completed high-resolution comparison")
    for row in rows:
        entry = state["records"][row["frame"]]
        if set(entry["sha256"]) != set(STEMS):
            raise ValueError("Missing rendered variant for " + row["frame"])
        folder = record / "frames" / row["frame"]
        for stem in STEMS:
            path = folder / (stem + "_face_4x.png")
            if sha(path) != entry["sha256"][stem]:
                raise ValueError("Stored face image changed: " + str(path))
            with Image.open(path) as image:
                if list(image.size) != entry["wh"]:
                    raise ValueError("Face image dimensions disagree with the common-grid crop")
                image.verify()
        width, height = entry["wh"]
        expected = (2 * (width + 32), 2 * (height + 94))
        for extension in ("png", "jpg"):
            with Image.open(folder / ("face_compare." + extension)) as image:
                if image.size != expected:
                    raise ValueError("Comparison board dimensions disagree with 2-by-2 layout")
                image.verify()


def publish(engine, record, rows, state, regions):
    names = {1: "original", 2: "interpolation2x", 4: "swinir4x"}
    table = ""
    for scale, name in names.items():
        metrics = json.loads((record / name / "metrics.json").read_text(encoding="utf-8-sig"))
        delivery = json.loads((record / name / "delivery.json").read_text(encoding="utf-8-sig"))
        if not delivery["parity_passed"] or metrics["execution"]["state"] != "training_completed":
            raise ValueError("Cannot publish an unfinished variant")
        metric = metrics["means"]["test"]
        width, height = rows[0]["wh"]
        table += f'<tr><td>{TRAINING_LABELS[scale]}</td><td>{width*scale} × {height*scale}</td><td>{delivery["points"]:,}</td><td>{metric["psnr"]:.3f}</td><td>{metric["ssim"]:.4f}</td><td><a href="{name}/{escape(delivery["bundle"])}">PLY 与相机</a></td></tr>'
    options = "".join(f'<option value="{escape(row["frame"])}">{escape(row["frame"])} · {"留出" if row["split"] == "test" else "训练"}</option>' for row in rows)
    frame_details = {entry["frame"]: {"source_index": entry["source_index"], "nearest_display_only": entry["provenance"]["method"] == "nearest_display_only"} for entry in regions["records"]}
    full_panels = "".join(f'<figure><figcaption>{escape(label)}</figcaption><a id="full-link-{stem}" target="_blank" rel="noopener"><img id="full-image-{stem}"></a></figure>' for stem, label in zip(STEMS, ["视频／数据集原图", "原始图训练", "插值两倍训练", "SwinIR四倍训练"]))
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MSV2 高清面部与完整图对照</title>
<style>body{margin:28px auto;max-width:1400px;padding:0 22px;font:16px/1.65 system-ui;background:#f5f6f7;color:#20242a}h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.panel{background:white;padding:18px;margin:18px 0;border-radius:12px}.hint{color:#555}.controls{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center}select{padding:8px;font-size:16px}a{color:#2266aa}.image-scroll{overflow-x:auto;margin:14px 0;padding-bottom:12px;border:1px solid #e3e6ea;background:#f0f2f5}.image-scroll a{display:block;width:max-content}.image-scroll img{display:block;width:auto;max-width:none;height:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0}.grid img{width:100%;height:auto}.status{font-size:14px;color:#555}#links{display:flex;gap:10px 20px;flex-wrap:wrap}[hidden]{display:none!important}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>MSV2 · 高清面部与完整图对照</h1>
<p>{{COUNT}} 个配准视角，{{ITERATIONS}} 步，{{METRIC}} / config/fast。完整背景参与训练，无 mask、无训练裁剪，从源图直接放大；各倍率独立初始化模型与优化器。</p>
<table><tr><th>训练输入</th><th>目标尺寸</th><th>高斯点数</th><th>留出 PSNR</th><th>留出 SSIM</th><th>通用导出</th></tr>{{TABLE}}</table>
<p class="hint">表中指标统一在原始完整图像尺寸计算，未用超分图作为真值。高清面部是三个冻结模型在相同四倍完整相机网格上的重新渲染，再裁出相同位置；不是对旧渲染图进行插值。视频原图仅作 Lanczos 四倍参考。训练及导出检查通过与人物几何、连续视角质量分别记录，本页不预先认定哪版更好。</p>
<div class="panel"><div class="controls"><label>帧 <select id="frame">{{OPTIONS}}</select></label><label>区域 <select id="region"><option value="face" selected>高清面部 · 四倍渲染</option><option value="full">完整画面 · 原分辨率</option></select></label><label id="zoom-control">显示 <select id="zoom"><option value="1" selected>1:1 · 实际像素</option><option value="2">200% · 放大查看</option></select></label></div>
<p class="hint" id="face-note">面部为两行两列无损 PNG，所有模型使用同一相机、裁剪与像素密度。默认实际像素，窄屏可横向滚动；点击可单独打开。</p><p class="status" id="source-status"></p><p class="status" id="image-status" aria-live="polite"></p>
<div class="image-scroll" id="face-panel"><a id="direct" target="_blank" rel="noopener"><img id="compare" alt="同帧视频参考及三种训练倍率模型的高清面部"></a></div><div class="grid" id="full-panel" hidden>{{FULL_PANELS}}</div><div id="links"></div></div>
<p><a href="comparison.json">完整指标</a> · <a href="面部高清对照.json">高清渲染与一致性核查</a> · <a href="展示区域.json">纯展示区域及源帧对应</a></p>
<script>
const frames={{FRAME_DETAILS}},names=['input','original','interpolation2x','swinir4x'],labels=['视频原图 · Lanczos四倍参考','原始图训练 · 四倍渲染','插值两倍训练 · 四倍渲染','SwinIR四倍训练 · 四倍渲染'];
const f=document.querySelector('#frame'),r=document.querySelector('#region'),z=document.querySelector('#zoom'),image=document.querySelector('#compare'),status=document.querySelector('#image-status');
function applyZoom(){if(r.value!=='face')return;image.style.width=(image.naturalWidth*Number(z.value))+'px';if(image.naturalWidth)status.textContent='无损图 '+image.naturalWidth+' × '+image.naturalHeight+' 像素 · '+(z.value==='1'?'1:1 实际像素':'200% 放大')+' · 两行两列';}
function update(writeHash=true){const face=r.value==='face',base='frames/'+f.value+'/',details=frames[f.value];document.querySelector('#zoom-control').hidden=!face;document.querySelector('#face-note').hidden=!face;document.querySelector('#face-panel').hidden=!face;document.querySelector('#full-panel').hidden=face;document.querySelector('#source-status').textContent='源视频帧 '+details.source_index+(details.nearest_display_only?' · 本帧仅展示框使用最近一次二维检测，未参与训练或相机恢复':' · 二维展示框对应相同原始 RGB，未参与训练或相机恢复');const target=document.querySelector('#links');target.replaceChildren();for(let i=0;i<names.length;i++){const stem=names[i],file=stem+(face?'_face_4x':'')+'.png',a=document.createElement('a');a.href=base+file;a.target='_blank';a.rel='noopener';a.textContent=face?labels[i]:['视频原图','原始版渲染','插值两倍版渲染','SwinIR四倍版渲染'][i];target.appendChild(a);if(!face){document.querySelector('#full-image-'+stem).src=base+file;document.querySelector('#full-link-'+stem).href=base+file;}}if(face){const path=base+'face_compare.png';document.querySelector('#direct').href=path;image.alt=f.value+' · 视频放大参考与三冻结模型四倍渲染';status.textContent='正在加载 '+f.value+'…';image.onload=applyZoom;image.onerror=()=>{status.textContent='图片未就绪，请检查高清渲染记录。';};image.src=path;if(image.complete&&image.naturalWidth)applyZoom();}else{status.textContent='完整画面保留原图分辨率渲染；点击各图可单独打开。';}if(writeHash){const params=new URLSearchParams({frame:f.value,region:r.value});if(face&&z.value!=='1')params.set('zoom',z.value);history.replaceState(null,'','#'+params.toString());}}
function readHash(){const params=new URLSearchParams(location.hash.slice(1)),frame=params.get('frame');if(frame&&Object.hasOwn(frames,frame))f.value=frame;r.value=params.get('region')==='full'?'full':'face';z.value=params.get('zoom')==='2'?'2':'1';update(false);}
f.onchange=r.onchange=z.onchange=()=>update();window.addEventListener('hashchange',readHash);readHash();
</script></html>'''
    replacements = {"COUNT": str(len(rows)), "ITERATIONS": f"{engine.iterations:,}", "METRIC": escape(engine.config["importance_metric"]), "TABLE": table, "OPTIONS": options, "FULL_PANELS": full_panels, "FRAME_DETAILS": json.dumps(frame_details, ensure_ascii=False)}
    for key, value in replacements.items():
        html = html.replace("{{" + key + "}}", value)
    for filename in ("comparison.html", "查看完整图倍率对照.html"):
        path = record / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(path)


def main():
    args = parse_args()
    job = args.job.resolve()
    snapshot = job.parent / "pipeline-source"
    if not (snapshot / "msv2/engine.py").is_file():
        raise FileNotFoundError("The completed job's frozen public pipeline is missing")
    sys.path.insert(0, str(snapshot))
    from msv2.engine import Engine, LazyCamera, NAMES, save_image
    from msv2.io import now, read_json as read, save_json as save, sha256 as sha
    from msv2.pipeline import implementation_fingerprint

    engine = Engine(job)
    if implementation_fingerprint(snapshot / "msv2") != engine.job["implementation_sha256"]:
        raise ValueError("Frozen pipeline implementation integrity check failed")
    for source in read(engine.root / "plan.json")["source_checks"]:
        if sha(engine.root / "source" / source["path"]) != source["sha256"]:
            raise ValueError("Frozen native renderer source integrity check failed")
    record = args.record.resolve()
    regions = read(args.regions)
    rows = read(engine.root / "original/training-inputs.json")["records"]
    rows4 = read(engine.root / "swinir4x/training-inputs.json")["records"]
    if len(rows) != len(rows4) or not rows:
        raise ValueError("The common-grid camera records are incomplete")
    boxes = validate_regions(regions, rows, record)
    manifest = record / "面部高清对照.json"
    models = {}
    for scale, name in NAMES.items():
        execution = read(engine.root / name / "execution.json")
        if execution["state"] != "training_completed" or execution["iterations"] != engine.iterations or not read(engine.root / name / "ply-parity.json")["passed"]:
            raise ValueError("Only completed, verified frozen models may be inspected")
        path = engine.root / name / f"model/point_cloud/iteration_{engine.iterations}/point_cloud.ply"
        models[name] = {"path": str(path), "sha256": sha(path), "points": execution["points"]}
    protected = protected_files(engine, record, rows, NAMES, sha)
    if manifest.exists():
        state = read(manifest)
        if state["job_sha256"] != sha(job) or state["regions_sha256"] != sha(args.regions) or state["models"] != models or state["protected_before"] != protected:
            raise ValueError("Frozen input evidence differs from this report's saved state")
    else:
        if args.check_only or args.publish_only:
            raise FileNotFoundError("No completed high-resolution manifest exists")
        state = dict(schema="msv2-high-resolution-face-review-v1", state="rendering", started=now(),
            run_root=str(engine.root), job=str(job), job_sha256=sha(job), regions_sha256=sha(args.regions),
            source_video_sha256=regions.get("source_video_sha256", regions.get("video_sha256")),
            source_wh=rows[0]["wh"], render_wh=rows4[0]["wh"], render_scale=4, models=models,
            method="Full-frame unchanged native rasterizer at common 4x K; crop after full render, no cropped-camera approximation",
            source_reference="Original full RGB Lanczos4 reference, not super-resolution ground truth",
            model_updates=0, optimizer_updates=0, used_for_training=False, used_for_camera_recovery=False,
            quality_accepted=False, protected_before=protected, records={}, parity_checks=[])
        save(manifest, state)
    if args.check_only or args.publish_only:
        validate_outputs(state, rows, record, sha)
        if state["state"] != "completed":
            raise ValueError("High-resolution rendering is not complete")
        if args.publish_only:
            publish(engine, record, rows, state, regions)
        print("HD_REPORT_VALIDATED", len(rows), "CUDA initialized: false", flush=True)
        return
    chosen = set(args.frames or [row["frame"] for row in rows])
    unknown = chosen - {row["frame"] for row in rows}
    if unknown:
        raise ValueError("Unknown source frame names: " + ", ".join(sorted(unknown)))
    indices = [index for index, row in enumerate(rows) if row["frame"] in chosen]
    font = font_path(args.font)
    torch = engine.runtime()
    torch.cuda.set_per_process_memory_fraction(min(0.80, engine.config["gpu_memory_fraction"]))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device="cuda")
    for scale, name in NAMES.items():
        model = GaussianModel(3)
        model.load_ply(models[name]["path"])
        model.active_sh_degree = 3
        with torch.no_grad():
            for index in indices:
                row, row4 = rows[index], rows4[index]
                frame = row["frame"]
                if frame != row4["frame"] or row["split"] != row4["split"] or row4["wh"] != [4 * value for value in row["wh"]]:
                    raise ValueError("Native and common-4x camera identities/dimensions disagree")
                if not np.array_equal(np.asarray(row["w2c"]), np.asarray(row4["w2c"])) or not np.allclose(np.diag([4., 4., 1.]) @ np.asarray(row["K"]), row4["K"], atol=1e-10, rtol=0):
                    raise ValueError("Common-grid rendering changed a pose or failed to scale K")
                folder = record / "frames" / frame
                destination = folder / (name + "_face_4x.png")
                existing = state["records"].get(frame)
                if existing and name in existing["sha256"]:
                    if sha(destination) != existing["sha256"][name]:
                        raise ValueError("Existing high-resolution render was changed")
                    continue
                camera = LazyCamera(row4, index)
                full = render(camera, model, pipe, background)["render"].clamp(0, 1)
                x0, y0, x1, y1 = box4 = [4 * value for value in boxes[frame]["face"]]
                cropped = full[:, y0:y1, x0:x1].contiguous()
                save_image(destination, cropped)
                preview = engine.root / name / "preview" / (frame + ".png")
                if scale == 4 and preview.is_file():
                    with Image.open(preview) as image:
                        if image.size != tuple(row4["wh"]):
                            raise ValueError("Native preview resolution differs from the 4x grid")
                        reference = np.asarray(image.crop(box4)).astype(np.int16)
                    with Image.open(destination) as image:
                        actual = np.asarray(image).astype(np.int16)
                    delta = np.abs(reference - actual)
                    check = dict(frame=frame, scale=scale, preview_sha256=sha(preview), max_byte_error=int(delta.max()), mean_byte_error=float(delta.mean()))
                    if check["max_byte_error"] > 2:
                        raise ValueError("Frozen native preview and high-resolution crop disagree: " + str(check))
                    state["parity_checks"].append(check)
                del full, cropped, camera
                torch.cuda.empty_cache()
                reference = folder / "input_face_4x.png"
                # An unregistered existing asset may belong to an older report.
                # Only trust an input reference already bound in this manifest;
                # otherwise regenerate it from the exact verified source RGB.
                if not existing or "input" not in existing["sha256"]:
                    with Image.open(folder / "input.png") as image:
                        enlarged = image.resize(tuple(row4["wh"]), Image.Resampling.LANCZOS)
                        enlarged.crop(box4).save(reference)
                        enlarged.close()
                elif sha(reference) != existing["sha256"]["input"]:
                    raise ValueError("The registered source reference was changed")
                entry = state["records"].setdefault(frame, dict(frame=frame, split=row["split"],
                    source_index=boxes[frame]["source_index"], display_box_provenance=boxes[frame]["provenance"],
                    source_wh=row["wh"], render_wh=row4["wh"],
                    face_box_source=boxes[frame]["face"], face_box_render=box4, wh=[x1-x0, y1-y0],
                    K=row4["K"], w2c=row4["w2c"], sha256={}))
                entry["sha256"].update({name: sha(destination), "input": sha(reference)})
                state["updated"] = now()
                save(manifest, state)
                print("HD_FACE_RENDER", name, frame, entry["wh"], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for index in indices:
        row = rows[index]
        if set(state["records"][row["frame"]]["sha256"]) != set(STEMS):
            raise ValueError("A requested comparison is missing a model")
        board(record / "frames" / row["frame"], row, font)
    complete = len(state["records"]) == len(rows) and all(set(entry["sha256"]) == set(STEMS) for entry in state["records"].values())
    if complete:
        validate_outputs(state, rows, record, sha)
        if protected_files(engine, record, rows, NAMES, sha) != state["protected_before"]:
            raise ValueError("Original evidence changed while preparing the report")
        state.update(state="completed", completed=now(), frame_count=len(rows), protected_unchanged=True,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            display="Lossless 2-by-2 panels at equal 4x pixel density; original full renders and metrics preserved",
            page_ui_verified=False)
        save(manifest, state)
        publish(engine, record, rows, state, regions)
    print("HD_FACE_COMPARISON_READY", len(indices), state["state"], flush=True)


if __name__ == "__main__":
    main()
