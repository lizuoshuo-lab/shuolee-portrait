"""Compare five frozen models at exact face pixels, reusing four old HD renders.

Use the baseline job's isolated Python. --check-only is CPU-only and writes
nothing. Only the new model is ever loaded for GPU rendering; an old crop that
does not contain the new face bbox is rejected, never enlarged or fabricated.
"""

import argparse
import gc
import hashlib
from html import escape
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import review_msv2_head as old
import publish_msv2_pipeline_viewer as publisher


OLD_MODELS = old.VARIANTS
NEW_MODEL = "face_composite4x"
STEMS = ("input", *OLD_MODELS, NEW_MODEL)
LABELS = (
    "原始RGB · Lanczos四倍参考",
    "原始完整图1倍训练",
    "插值两倍完整图训练",
    "SwinIR四倍完整图训练",
    "旧整图1倍＋头部4倍视图",
    "面部mask合成完整图4倍训练",
)


def display_labels(layout, scope="face"):
    return (
        (
            *LABELS[:-1],
            f"mask外原图1倍＋{'整头' if scope == 'head' else '面部'}子图4倍 · 同次更新",
        )
        if layout == publisher.PAIRED_LAYOUT
        else LABELS
    )


def check_new_job(path):
    """Verify the new snapshot, including the packaged ROI patch when present."""
    path = Path(path).resolve()
    job = old.read(path)
    root = Path(job["run_dir"]).resolve()
    old.require(path.parent == root, "Read the new job from its original frozen run")
    package = root / "pipeline-source/msv2"
    old.require(
        publisher.implementation_sha(package) == job["implementation_sha256"],
        "New frozen pipeline/ROI patch hash differs",
    )
    protected = [
        path,
        root / "plan.json",
        root / "prepared/camera-inputs.json",
        root / "prepared/pixel-lineage.json",
    ]
    protected.extend(
        sorted(
            p
            for p in package.iterdir()
            if p.is_file() and p.suffix in (".py", ".patch")
        )
    )
    sources = {}
    for item in old.read(root / "plan.json")["source_checks"]:
        relative = Path(item["path"])
        old.require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe native source path",
        )
        target = root / "source" / relative
        old.require(
            old.sha(target) == item["sha256"], "New frozen native source changed"
        )
        sources[relative.as_posix()] = item["sha256"]
        protected.append(target)
    lineage = old.read(root / "prepared/pixel-lineage.json")
    old.require(lineage["passed"], "New input lineage failed")
    return dict(
        path=path,
        job=job,
        root=root,
        source=sources,
        protected=protected,
        lineage=lineage,
    )


def crop_in_old(old_bbox, face_bbox, wh):
    for box in (old_bbox, face_bbox):
        old.require(
            isinstance(box, list)
            and len(box) == 4
            and all(type(v) is int for v in box),
            "Display boxes must use integer exclusive native corners",
        )
        old.require(
            0 <= box[0] < box[2] <= wh[0] and 0 <= box[1] < box[3] <= wh[1],
            "Display bbox is outside native RGB",
        )
    x0, y0, x1, y1 = old_bbox
    a, b, c, d = face_bbox
    old.require(
        x0 <= a < c <= x1 and y0 <= b < d <= y1,
        "New face bbox is outside the old HD head crop: stop this frame and re-render the old models on the common full4 camera; resizing is forbidden",
    )
    return [4 * (a - x0), 4 * (b - y0), 4 * (c - x0), 4 * (d - y0)]


def prepare(args):
    baseline, previous, by_scale, _, _, previous_template, previous_context = (
        old.prepare(args.baseline_job, args.old_head_job)
    )
    record = args.old_record.resolve()
    previous_state = old.read(record / "head_review.json")
    saved_template = old.read(record / "head_view_templates.json")
    old.require(
        previous_state["state"] == "completed"
        and previous_state["context"] == previous_context,
        "Old four-model HD review is not completed or no longer matches its frozen jobs",
    )
    old.require(
        previous_state["helper_sha256"] == old.sha(old.__file__)
        and saved_template == previous_template,
        "Old HD helper/camera-template binding differs",
    )
    old.validate_checkpoint(
        record,
        previous_state,
        [row for row in saved_template["records"] if row["head_count"]],
        complete=True,
    )
    new = check_new_job(args.new_job)
    old.require(
        new["job"]["schema"] == "msv2-job-v1"
        and new["job"]["options"]["sr_region"] in ("face", "head"),
        "Expected the new complete-image face-composition job",
    )
    old.require(
        new["source"] == baseline["source"],
        "New and old models do not use the same frozen native renderer/model source",
    )
    old.require(
        new["job"]["options"]["iterations"] == baseline["job"]["options"]["iterations"],
        "Completed training budgets differ",
    )
    pipeline_state = old.read(new["root"] / "state.json")
    plan = old.read(new["root"] / "plan.json")
    expected_scales = set(new["job"]["options"]["scales"])
    old.require(
        pipeline_state["state"] == plan["state"] == "completed"
        and pipeline_state.get("training_completed") is True
        and plan.get("training_completed") is True
        and not pipeline_state.get("failures")
        and set(pipeline_state["completed_scales"]) == expected_scales
        and 4 in expected_scales,
        "New requested pipeline is not fully completed",
    )
    models = dict(previous_context["models"])
    models[NEW_MODEL] = old.model_info(new, "swinir4x")
    run = new["root"] / "swinir4x"
    sr_path = run / "superresolution.json"
    sr = old.read(sr_path)
    paired = sr.get("view_layout") == publisher.PAIRED_LAYOUT
    scope = sr.get("mask_scope", "face")
    old.require(
        sr["state"] == "completed"
        and sr["sr_region"] == scope
        and scope in ("face", "head")
        and (paired or sr["training_mask_used"] is False)
        and sr["additional_training_views"] is False
        and sr["full_frame_4x_generated"] is (not paired),
        "Face composition inputs are not complete whole-image4 inputs",
    )
    sr_rows = old.unique(sr["records"], "name")
    training = old.read(run / "training-inputs.json")["records"]
    trained = old.unique(training, "frame")
    sources = by_scale[1]
    if paired:
        old.require(
            new["job"]["options"]["iterations"] == 18000,
            "Paired HD review requires completed18000 steps",
        )
        old.require(
            plan.get("fresh_gaussians") is True
            and plan.get("fresh_optimizer") is True
            and plan.get("old_fitted_state_loaded") is False,
            "Paired review requires fresh independent training",
        )
        publisher.verify_variant(
            dict(
                run=new["root"],
                iterations=18000,
                job=new["job"],
                frames=len(sources),
                slug="paired-hd",
                date="20000101",
            ),
            4,
        )
        new["protected"].extend(
            p for p in (new["root"] / "export/swinir4x").rglob("*") if p.is_file()
        )
    old.require(
        set(trained) == set(sources)
        and len(training) == len(sources)
        and set(sr_rows) == {row["name"] for row in sources.values()},
        "The new job must have exactly one complete view per original source",
    )
    previous_rows = old.unique(saved_template["records"], "frame")
    protected = dict(previous_context["protected_before"])
    protected_paths = [
        record / "head_review.json",
        record / "head_view_templates.json",
        Path(old.__file__),
        Path(publisher.__file__),
        sr_path,
        new["root"] / "state.json",
        *new["protected"],
    ]
    templates = []
    for index, (frame, source) in enumerate(sources.items()):
        row, full4, old_row = trained[frame], by_scale[4][frame], previous_rows[frame]
        input_camera = source if paired else full4
        old.require(
            row["name"] == source["name"]
            and row["native_sha256"] == source["native_sha256"]
            and row["split"] == source["split"]
            and row["wh"] == input_camera["wh"]
            and old.matrix_equal(row["K"], input_camera["K"])
            and old.matrix_equal(row["w2c"], input_camera["w2c"]),
            "New source identity/split/full4 camera differs: " + frame,
        )
        native = Path(row["native_image"])
        old.require(
            old.sha(native) == source["native_sha256"],
            "New native RGB differs from the old source: " + frame,
        )
        item = sr_rows[source["name"]]
        old.require(
            item["native_sha256"] == source["native_sha256"]
            and item["output_sha256"] == row["source_sha256"],
            "New whole4 PNG source receipt differs: " + frame,
        )
        old.require(item["face_count"] == len(item["faces"]), "New face count differs")
        composition = item["composition"]
        old.require(
            composition["mask_scope"] == scope
            and (
                paired
                or (
                    composition["one_complete_training_image"]
                    and composition["added_views"] == 0
                    and composition["full_frame_4x_generated"]
                    and composition["outside_mask_equal_reference"]
                )
            )
            and composition["source_rgb_sha256"] == old_row["source_rgb_sha256"],
            "New composition is not bound to one unchanged native source: " + frame,
        )
        image = Path(row["image"])
        old.require(old.sha(image) == row["source_sha256"], "New training PNG changed")
        old.image_check(image, input_camera["wh"])
        protected_paths.extend([native, image])
        union = np.zeros((source["wh"][1], source["wh"][0]), bool)
        for face in item["faces"]:
            name = face["mask_name"]
            old.require(Path(name).name == name, "Unsafe native face-mask name")
            mask_dir = item.get("mask_dir", "face-masks")
            old.require(
                mask_dir in ("face-masks", "head-masks"), "Unsafe mask directory"
            )
            path = run / mask_dir / name
            old.require(old.sha(path) == face["mask_sha256"], "Face-mask PNG changed")
            with Image.open(path) as mask:
                pixels = np.asarray(mask)
                old.require(
                    mask.mode == "L"
                    and mask.format == "PNG"
                    and list(mask.size) == source["wh"]
                    and np.all((pixels == 0) | (pixels == 255))
                    and mask.getbbox() is not None
                    and list(mask.getbbox()) == face["bbox_native"],
                    "Native face mask/bbox differs",
                )
                union |= pixels > 0
            protected_paths.append(path)
        union_sha = hashlib.sha256(union.tobytes()).hexdigest()
        old.require(
            composition[
                "union_native_mask_sha256" if paired else "union_mask_native_sha256"
            ]
            == union_sha
            and composition[
                "union_native_mask_pixels" if paired else "union_mask_native_pixels"
            ]
            == int(union.sum()),
            "Saved face masks do not match the composition support",
        )
        faces = item["faces"]
        template = dict(
            frame=frame,
            source_ordinal=index + 1,
            source_index=source.get("source_index"),
            split=source["split"],
            source_name=source["name"],
            source_wh=source["wh"],
            native_png_sha256=source["native_sha256"],
            source_rgb_sha256=old_row["source_rgb_sha256"],
            face_count=len(faces),
            input_layout=sr.get("view_layout", publisher.COMPOSITE_LAYOUT),
            input_scope=scope,
            mask_source=sr.get("mask_source"),
            selection="first face in the completed SR receipt only; other faces are not displayed",
            bbox_native=None,
            bbox_render_4x=None,
            old_hd_crop_box=None,
            old_hd={},
            common_full4_camera=old_row["common_full4_camera"],
            native_image=str(native),
        )
        if faces:
            bbox = faces[0]["bbox_native"]
            old.require(
                old_row["bbox_native"] is not None
                and frame in previous_state["records"],
                "No old head HD support exists for the new face: " + frame,
            )
            try:
                subcrop = crop_in_old(old_row["bbox_native"], bbox, source["wh"])
            except ValueError as error:
                raise ValueError(frame + ": " + str(error)) from error
            for name in OLD_MODELS:
                path = record / "frames" / frame / (name + "_head_4x.png")
                digest = previous_state["records"][frame]["sha256"][path.name]
                old.require(
                    old.sha(path) == digest, "Old HD image changed: " + str(path)
                )
                expected_wh = [
                    4 * (old_row["bbox_native"][2] - old_row["bbox_native"][0]),
                    4 * (old_row["bbox_native"][3] - old_row["bbox_native"][1]),
                ]
                old.image_check(path, expected_wh)
                template["old_hd"][name] = dict(
                    path=str(path), sha256=digest, wh=expected_wh
                )
                protected_paths.append(path)
            template.update(
                bbox_native=bbox,
                bbox_render_4x=[4 * v for v in bbox],
                old_head_bbox_native=old_row["bbox_native"],
                old_hd_crop_box=subcrop,
                selected_face_mask_sha256=faces[0]["mask_sha256"],
            )
        templates.append(template)
    for path in sorted(set(protected_paths)):
        digest = old.sha(path)
        old.require(
            str(path) not in protected or protected[str(path)] == digest,
            "Protected evidence changed before review",
        )
        protected[str(path)] = digest
    template = dict(
        schema="msv2-face-composite-HD-template-v1",
        source_frames=len(templates),
        displayed_frames=sum(row["face_count"] > 0 for row in templates),
        no_face_frames=[row["frame"] for row in templates if not row["face_count"]],
        records=templates,
        old_method="Exact integer secondary crop from verified common-full4 old HD images; no old-model render/resize",
        new_method="Only new model: complete baseline full4 camera render then crop; never ROI-camera render",
        unsupported_crop_policy="Stop; old models require an explicit new complete-camera render, never rescale the available crop",
    )
    context = dict(
        old_record=str(record),
        old_hd_manifest_sha256=old.sha(record / "head_review.json"),
        baseline_job=str(baseline["path"]),
        old_head_job=str(previous["path"]),
        new_job=str(new["path"]),
        baseline_job_sha256=old.sha(baseline["path"]),
        old_head_job_sha256=old.sha(previous["path"]),
        new_job_sha256=old.sha(new["path"]),
        models=models,
        new_layout=sr.get("view_layout", publisher.COMPOSITE_LAYOUT),
        new_scope=scope,
        new_mask_source=sr.get("mask_source"),
        template_sha256=old.value_sha(template),
        protected_before=protected,
    )
    return baseline, by_scale[4], template, context


def reuse_frame(row, folder):
    """No resizing of old renders: a 4x integer subcrop is already exact HD."""
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    bbox4 = row["bbox_render_4x"]
    with Image.open(row["native_image"]) as image:
        enlarged = image.convert("RGB").resize(
            tuple(row["common_full4_camera"]["wh"]), Image.Resampling.LANCZOS
        )
        enlarged.crop(bbox4).save(folder / "input_face_4x.png")
        enlarged.close()
    result["input_face_4x.png"] = old.sha(folder / "input_face_4x.png")
    for name in OLD_MODELS:
        source = row["old_hd"][name]
        old.require(
            old.sha(source["path"]) == source["sha256"],
            "Old HD changed before secondary crop",
        )
        path = folder / (name + "_face_4x.png")
        with Image.open(source["path"]) as image:
            image.crop(row["old_hd_crop_box"]).save(path)
        result[path.name] = old.sha(path)
    return result


def board(folder, row, font):
    x0, y0, x1, y1 = row["bbox_render_4x"]
    width, height = x1 - x0, y1 - y0
    cell_width, pad, header = max(width, 340), 12, 66
    canvas = Image.new(
        "RGB", (3 * (cell_width + 2 * pad), 2 * (height + header + pad)), "#f0f2f5"
    )
    draw = ImageDraw.Draw(canvas)
    title, detail = ImageFont.truetype(font, 18), ImageFont.truetype(font, 15)
    for index, (name, label) in enumerate(
        zip(
            STEMS,
            display_labels(row.get("input_layout"), row.get("input_scope", "face")),
        )
    ):
        x = (index % 3) * (cell_width + 2 * pad) + pad
        y = (index // 3) * (height + header + pad)
        draw.text((x, y + 6), label, font=title, fill="#20242a")
        draw.text(
            (x, y + 34),
            f"{row['frame']} · {row['split']} · {width}×{height}",
            font=detail,
            fill="#555555",
        )
        with Image.open(folder / (name + "_face_4x.png")) as image:
            old.require(
                image.size == (width, height),
                "All six panels must have identical exact face pixels",
            )
            canvas.paste(image, (x, y + header))
    canvas.save(folder / "face_compare.png")


def expected_assets(row):
    x0, y0, x1, y1 = row["bbox_render_4x"]
    width, height = x1 - x0, y1 - y0
    return {
        **{name + "_face_4x.png": [width, height] for name in STEMS},
        "face_compare.png": [3 * (max(width, 340) + 24), 2 * (height + 78)],
    }


def validate(record, state, rows, complete=False):
    lookup = old.unique(rows, "frame")
    old.require(
        set(state["records"]) <= set(lookup), "Checkpoint has an unknown/no-face frame"
    )
    for frame, entry in state["records"].items():
        expected = expected_assets(lookup[frame])
        for name, digest in entry["sha256"].items():
            old.require(name in expected, "Unknown comparison asset")
            path = record / "frames" / frame / name
            old.require(old.sha(path) == digest, "Saved face HD changed: " + str(path))
            old.image_check(path, expected[name])
        if complete:
            old.require(
                set(entry["sha256"]) == set(expected),
                "Incomplete six-panel frame: " + frame,
            )
    if complete:
        old.require(
            set(state["records"]) == set(lookup),
            "Not every detected-face source has a completed comparison",
        )


def publish(record, state, rows):
    old.require(
        state["state"] == "completed" and state["protected_unchanged"],
        "Only completed HD output may be published",
    )
    options = "".join(
        f'<option value="{escape(row["frame"])}">{escape(row["frame"])} · {row["split"]}</option>'
        for row in rows
    )
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>面部mask整图合成高清对照</title><style>body{margin:24px;font:16px/1.7 system-ui;background:#f5f6f7;color:#222}select{font-size:16px;padding:6px}.scroll{overflow:auto;margin:18px 0;background:#e9ecf0}.scroll img{display:block;max-width:none}a{color:#2266aa}#links{display:flex;gap:16px;flex-wrap:wrap}</style><h1>MSV2 · 面部mask整图合成高清对照</h1><p>六面板：原始RGB参考和五个独立完成的模型。旧四个模型复用已校验的完整四倍相机高清图，按整数偏移再次裁剪；新模型在同一完整四倍相机上渲染后裁剪。原图参考为Lanczos四倍，不是高分辨率真值。未认定质量通过。</p><p>COVERAGE</p><label>帧 <select id="frame">OPTIONS</select></label> <label>显示 <select id="zoom"><option value="1">1:1 实际像素</option><option value="2">200%</option></select></label><p id="note"></p><div class="scroll"><a id="direct" target="_blank" rel="noopener"><img id="compare"></a></div><div id="links"></div><p><a href="face_composite_review.json">高清核查与模型绑定</a> · <a href="face_composite_view_templates.json">面部框、完整相机与旧图裁剪对应</a></p><script>const names=STEMS,labels=LABELS,details=DETAILS;const f=document.querySelector('#frame'),z=document.querySelector('#zoom'),im=document.querySelector('#compare');function zoom(){im.style.width=(im.naturalWidth*Number(z.value))+'px'}function update(){const base='frames/'+f.value+'/',r=details[f.value];document.querySelector('#note').textContent=r.split+' · 原图面部框 ['+r.bbox.join(', ')+']'+(r.face_count>1?' · 仅展示SR记录中的第一个面部':'')+' · 六图均为同一完整相机像素密度；可横向滚动。';document.querySelector('#links').replaceChildren();names.forEach((name,i)=>{const a=document.createElement('a');a.href=base+name+'_face_4x.png';a.textContent=labels[i];a.target='_blank';a.rel='noopener';document.querySelector('#links').append(a)});document.querySelector('#direct').href=base+'face_compare.png';im.onload=zoom;im.src=base+'face_compare.png';if(im.complete)zoom();history.replaceState(null,'','#'+new URLSearchParams({frame:f.value,zoom:z.value}))}const q=new URLSearchParams(location.hash.slice(1));if(Object.hasOwn(details,q.get('frame')))f.value=q.get('frame');z.value=q.get('zoom')==='2'?'2':'1';f.onchange=z.onchange=update;update();</script></html>"""
    details = {
        row["frame"]: dict(
            split=row["split"], bbox=row["bbox_native"], face_count=row["face_count"]
        )
        for row in rows
    }
    for key, value in dict(
        OPTIONS=options,
        COVERAGE=f"{state['source_frames']} 个原始源帧，{len(rows)} 帧存在可展示面部mask；无mask源帧不虚构展示框。",
        STEMS=json.dumps(STEMS),
        LABELS=json.dumps(
            display_labels(
                state.get("context", {}).get("new_layout"),
                state.get("context", {}).get("new_scope", "face"),
            ),
            ensure_ascii=False,
        ),
        DETAILS=json.dumps(details, ensure_ascii=False),
    ).items():
        html = html.replace(key, value)
    paired = state.get("context", {}).get("new_layout") == publisher.PAIRED_LAYOUT
    if paired:
        html = html.replace("面部mask整图合成高清对照", "子母图联合训练高清对照")
    if state.get("context", {}).get("new_scope") == "head":
        html = (
            html.replace("子母图联合训练高清对照", "整头子母图联合训练高清对照")
            .replace("面部框", "整头区域框")
            .replace("面部mask", "整头mask")
            .replace("第一个面部", "第一个整头区域")
        )
    path = record / (
        "查看子母图高清对照.html" if paired else "查看面部整图合成对照.html"
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-job", "old-head-job", "new-job", "old-record", "record"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--frames",
        nargs="*",
        help="Exact names or 1-based source ordinals, e.g. 1 33 65",
    )
    parser.add_argument("--font", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args(argv)
    old.require(
        not (args.check_only and args.publish_only), "Choose check-only or publish-only"
    )
    baseline, common, template, context = prepare(args)
    record = args.record.resolve()
    old.require(
        record != args.old_record.resolve()
        and not record.is_relative_to(args.old_record.resolve())
        and not args.old_record.resolve().is_relative_to(record),
        "Use a separate new review record, outside the old HD record",
    )
    for path in (args.baseline_job, args.old_head_job, args.new_job):
        run = path.resolve().parent
        old.require(
            not record.is_relative_to(run) and not run.is_relative_to(record),
            "Keep the report outside every protected run",
        )
    rows = [row for row in template["records"] if row["face_count"]]
    old.require(bool(rows), "No actual face mask exists for a six-panel comparison")
    chosen = (
        old.choose_frames(args.frames, template["records"])
        if args.frames
        else {row["frame"] for row in rows}
    )
    old.require(
        chosen <= {row["frame"] for row in rows},
        "A requested source has no face mask; no display rectangle will be invented",
    )
    manifest = record / "face_composite_review.json"
    template_path = record / "face_composite_view_templates.json"
    if manifest.exists():
        state = old.read(manifest)
        old.require(
            state["context"] == context and state["helper_sha256"] == old.sha(__file__),
            "Checkpoint input/source/helper binding changed",
        )
        old.require(
            old.read(template_path) == template,
            "Saved exact-face camera template changed",
        )
        validate(record, state, rows, complete=state["state"] == "completed")
    else:
        state = dict(
            schema="msv2-five-model-face-composite-HD-v1",
            state="not_started",
            started=old.now(),
            context=context,
            helper_sha256=old.sha(__file__),
            source_frames=template["source_frames"],
            display_frames=len(rows),
            no_face_frames=template["no_face_frames"],
            old_model_GPU_renders=0,
            new_model_GPU_renders=0,
            model_updates=0,
            optimizer_updates=0,
            quality_accepted=False,
            records={},
            old_method=template["old_method"],
            new_method=template["new_method"],
            page_ui_verified=False,
        )
    if args.check_only or args.publish_only:
        old.unchanged(context)
        if args.publish_only:
            validate(record, state, rows, complete=True)
            publish(record, state, rows)
        print(
            "FACE_COMPOSITE_HD_CPU_VALIDATED",
            json.dumps(
                dict(
                    source_frames=template["source_frames"],
                    display_frames=len(rows),
                    state=state["state"],
                    cuda_initialized=False,
                )
            ),
            flush=True,
        )
        return
    font = old.find_font(args.font)
    record.mkdir(parents=True, exist_ok=True)
    old.save(template_path, template)
    state.update(state="rendering", updated=old.now())
    old.save(manifest, state)
    try:
        for row in rows:
            frame = row["frame"]
            if frame not in chosen:
                continue
            entry = state["records"].setdefault(frame, dict(sha256={}))
            required = {name + "_face_4x.png" for name in ("input", *OLD_MODELS)}
            if not required <= set(entry["sha256"]):
                entry["sha256"].update(reuse_frame(row, record / "frames" / frame))
                old.save(manifest, state)
        sys.path.insert(0, str(baseline["root"] / "pipeline-source"))
        from msv2.engine import Engine, LazyCamera, save_image

        engine = Engine(baseline["path"])
        torch = engine.runtime()
        torch.cuda.set_per_process_memory_fraction(
            min(0.8, engine.config["gpu_memory_fraction"])
        )
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render

        model = GaussianModel(3)
        model.load_ply(context["models"][NEW_MODEL]["path"])
        model.active_sh_degree = 3
        pipe = SimpleNamespace(
            convert_SHs_python=False, compute_cov3D_python=False, debug=False
        )
        background = torch.zeros(3, device="cuda")
        with torch.no_grad():
            for uid, row in enumerate(rows):
                frame = row["frame"]
                if frame not in chosen:
                    continue
                entry = state["records"][frame]
                folder = record / "frames" / frame
                name = NEW_MODEL + "_face_4x.png"
                if name not in entry["sha256"]:
                    camera = LazyCamera(common[frame], uid)
                    full = render(camera, model, pipe, background)["render"].clamp(0, 1)
                    x0, y0, x1, y1 = row["bbox_render_4x"]
                    cropped = full[:, y0:y1, x0:x1].contiguous()
                    save_image(folder / name, cropped)
                    entry["sha256"][name] = old.sha(folder / name)
                    state["new_model_GPU_renders"] += 1
                    del full, cropped, camera
                    torch.cuda.empty_cache()
                    old.save(manifest, state)
                if "face_compare.png" not in entry["sha256"]:
                    board(folder, row, font)
                    entry["sha256"]["face_compare.png"] = old.sha(
                        folder / "face_compare.png"
                    )
                    old.save(manifest, state)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        complete = len(state["records"]) == len(rows) and all(
            set(expected_assets(row)) == set(state["records"][row["frame"]]["sha256"])
            for row in rows
        )
        validate(record, state, rows, complete=complete)
        old.unchanged(context)
        state.update(
            state="completed" if complete else "partial",
            updated=old.now(),
            protected_unchanged=True,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        if complete:
            state["completed"] = old.now()
        old.save(manifest, state)
        if complete:
            publish(record, state, rows)
    except Exception as error:
        state.update(state="failed", updated=old.now(), error=str(error))
        old.save(manifest, state)
        raise
    print("FACE_COMPOSITE_HD_CHECKPOINT", len(chosen), state["state"], flush=True)


if __name__ == "__main__":
    main()
