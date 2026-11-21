"""Re-render frozen 59-view models at a common 4x grid for face inspection."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import importlib.util
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment", HERE / "standard_msv2_demo.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
NAMES = ["input", "original", "interpolation2x", "swinir4x"]
LABELS = ["视频原图 · Lanczos 放大参考", "原始图训练 · 四倍渲染", "插值两倍训练 · 四倍渲染", "SwinIR 四倍训练 · 四倍渲染"]
MANIFEST = ex.RECORD / "面部高清对照.json"


def protected_files():
    paths = []
    for scale, name in ex.NAMES.items():
        paths.extend([ex.ROOT / name / "model/point_cloud/iteration_18000/point_cloud.ply",
                      ex.ROOT / name / "training-inputs.json", ex.ROOT / name / "execution.json",
                      ex.RECORD / name / "metrics.json", ex.RECORD / name / "delivery.json"])
    for folder in (ex.RECORD / "frames").iterdir():
        for stem in NAMES:
            paths.extend([folder / (stem + ".png"), folder / (stem + "_face.png")])
    return {str(p): ex.sha(p) for p in paths}


def board(folder, frame, split):
    images = [Image.open(folder / (n + "_face_4x.png")).convert("RGB") for n in NAMES]
    assert len({im.size for im in images}) == 1
    w, h = images[0].size
    pad, header = 16, 78
    canvas = Image.new("RGB", (2 * (w + 2 * pad), 2 * (h + header + pad)), "#f0f2f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 20)
    small = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 17)
    for i, im in enumerate(images):
        x, y = (i % 2) * (w + 2 * pad) + pad, (i // 2) * (h + header + pad)
        draw.text((x, y + 8), LABELS[i], font=font, fill="#20242a")
        draw.text((x, y + 39), f"{frame} · {'留出' if split == 'test' else '训练'} · {w}×{h}", font=small, fill="#505761")
        canvas.paste(im, (x, y + header))
        im.close()
    canvas.save(folder / "face_compare.png")
    canvas.save(folder / "face_compare.jpg", quality=97, subsampling=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="*")
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    rows = ex.read(ex.ROOT / "original/training-inputs.json")["records"]
    rows4 = ex.read(ex.ROOT / "swinir4x/training-inputs.json")["records"]
    boxes = ex.read(ex.RECORD / "original/metrics.json")["boxes"]
    assert len(rows) == len(rows4) == 59
    if MANIFEST.exists():
        state = ex.read(MANIFEST)
        assert state["run_root"] == str(ex.ROOT)
    else:
        state = dict(state="rendering", started=ex.now(), run_root=str(ex.ROOT),
                     source_wh=rows[0]["wh"], render_wh=rows4[0]["wh"], render_scale=4,
                     method="Full-frame native rasterization at common 4x intrinsics; crop AFTER rendering; no ROI-camera approximation",
                     source_reference="Full original RGB Lanczos enlarged 4x, then cropped; not super-resolution ground truth",
                     model_updates=0, optimizer_updates=0, quality_accepted=False,
                     protected_before=protected_files(), records={}, parity_checks=[])
        ex.save(MANIFEST, state)
    selected = [i for i, row in enumerate(rows) if not args.frames or row["frame"] in args.frames]
    torch = ex.runtime()
    torch.cuda.set_per_process_memory_fraction(0.80)
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    for scale, name in ex.NAMES.items():
        model = GaussianModel(3)
        model.load_ply(str(ex.ROOT / name / "model/point_cloud/iteration_18000/point_cloud.ply"))
        model.active_sh_degree = 3
        with torch.no_grad():
            for i in selected:
                row, row4 = rows[i], rows4[i]
                frame = row["frame"]
                folder = ex.RECORD / "frames" / frame
                destination = folder / (name + "_face_4x.png")
                if frame in state["records"] and name in state["records"][frame].get("sha256", {}):
                    assert ex.sha(destination) == state["records"][frame]["sha256"][name]
                    continue
                assert row["frame"] == row4["frame"] and row["split"] == row4["split"]
                assert np.array_equal(np.asarray(row["w2c"]), np.asarray(row4["w2c"]))
                assert np.allclose(np.diag([4., 4., 1.]) @ np.asarray(row["K"]), row4["K"], atol=1e-10, rtol=0)
                camera = ex.LazyCamera(row4, i)
                full = render(camera, model, pipe, bg)["render"].clamp(0, 1)
                x0, y0, x1, y1 = box4 = [4 * v for v in boxes[frame]["face"]]
                crop = full[:, y0:y1, x0:x1].contiguous()
                ex.save_image(destination, crop)
                if scale == 4 and frame in ("frame_000009", "frame_000025", "frame_000041"):
                    with Image.open(ex.ROOT / name / "preview" / (frame + ".png")) as im:
                        reference = np.asarray(im.crop(box4)).astype(np.int16)
                    with Image.open(destination) as im:
                        actual = np.asarray(im).astype(np.int16)
                    delta = np.abs(reference - actual)
                    check = dict(frame=frame, scale=scale, max_byte_error=int(delta.max()), mean_byte_error=float(delta.mean()))
                    assert check["max_byte_error"] <= 2, check
                    state["parity_checks"].append(check)
                del full, crop, camera
                torch.cuda.empty_cache()
                reference_path = folder / "input_face_4x.png"
                if not reference_path.exists():
                    with Image.open(folder / "input.png") as im:
                        assert im.size == tuple(row["wh"])
                        enlarged = im.resize(tuple(row4["wh"]), Image.Resampling.LANCZOS)
                        enlarged.crop(box4).save(reference_path)
                        enlarged.close()
                record = state["records"].setdefault(frame, dict(frame=frame, split=row["split"],
                    face_box_source=boxes[frame]["face"], face_box_render=box4,
                    wh=[x1-x0, y1-y0], K=row4["K"], w2c=row4["w2c"], sha256={}))
                record["sha256"].update({name: ex.sha(destination), "input": ex.sha(reference_path)})
                state["updated"] = ex.now()
                ex.save(MANIFEST, state)
                print("FACE_RENDER", name, frame, record["wh"], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for i in selected:
        row = rows[i]
        assert set(state["records"][row["frame"]]["sha256"]) == set(NAMES)
        board(ex.RECORD / "frames" / row["frame"], row["frame"], row["split"])
    if args.finish:
        assert len(state["records"]) == len(rows)
        for row in rows:
            folder = ex.RECORD / "frames" / row["frame"]
            record = state["records"][row["frame"]]
            assert set(record["sha256"]) == set(NAMES)
            for name, sha in record["sha256"].items():
                path = folder / (name + "_face_4x.png")
                assert ex.sha(path) == sha
                with Image.open(path) as im:
                    assert list(im.size) == record["wh"]
            with Image.open(folder / "face_compare.png") as im:
                im.verify()
        assert protected_files() == state["protected_before"], "Original evidence changed"
        state.update(state="completed", completed=ex.now(), frame_count=len(rows), protected_unchanged=True,
                     peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                     display="Lossless 2-by-2 panels; all four faces have identical crop and pixel dimensions")
        ex.save(MANIFEST, state)
    print("FACE_COMPARISON_READY", len(selected), state["state"], flush=True)


if __name__ == "__main__":
    main()
