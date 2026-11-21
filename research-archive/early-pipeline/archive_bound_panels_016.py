"""Verify the completed 016 bound replay and append its JPEGs to the image record."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageOps


PREFIX = "3DGS_016_smpl逐帧驱动_"
CONTACT = "分析_016_smpl逐帧驱动入选原图_24帧.jpg"
IDENTITY = ("frame", "pose_index", "source_frame_index", "source_sha256", "mask_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contact_sheet(rows, path):
    tile_w, tile_h = 330, 219
    sheet = Image.new("RGB", (6 * tile_w, 4 * tile_h), "#171717")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(sorted(rows, key=lambda item: item["source_frame_index"])):
        x, y = (i % 6) * tile_w, (i // 6) * tile_h
        with Image.open(row["image"]) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_w - 8, tile_h - 25), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + 4))
        draw.text((x + 8, y + tile_h - 18), f'{row["frame"]}   encoded frame {row["source_frame_index"]}', fill="white")
    sheet.save(path, quality=95, subsampling=0)


def archive(run, destination, *, prior_run=None, publish=False):
    run, destination = Path(run).resolve(strict=True), Path(destination).resolve(strict=True)
    require(destination.is_dir() and destination.name == "016" and destination.parent.name == "记录",
            "Choose the existing 记录/016 image directory.")
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        require((expected is None or digest == expected) and pins.get(path, digest) == digest,
                f"Hash mismatch: {path}")
        pins[path] = digest
        return path

    def load(path, expected=None):
        return json.loads(pin(path, expected).read_text(encoding="utf-8-sig"))

    status = load(run / "status.json")
    require(status.get("stage") == "complete" and status.get("state") == "completed"
            and status.get("format_passed") is True, "Wait for the full bound review to complete.")
    result = load(run / "result.json")
    require(result.get("state") == "completed" and result.get("format_passed") is True
            and result.get("steps") == 10_000 and result.get("frames") == 24
            and status.get("sha256") == result["sha256"], "Expected the completed 24-frame stable result.")
    prereg = load(run.parent / "preregistration.json")
    info = load(run / "run.json")
    decoded = load(run / "decoded.json")
    require(prereg["source_sha256"] == info["video_sha256"] == decoded["video_sha256"], "Original video identity changed.")
    pin(info["video"], prereg["source_sha256"])
    require(result["ply"] == "export/point_cloud.ply" and result["scene"] == "export/scene.zip",
            "Unexpected terminal export paths.")
    pin(run / result["ply"], result["sha256"])
    pin(run / result["scene"], result["scene_sha256"])
    fit = load(run / "fit/official-fit-result.json")
    selection = load(run / "fit/selection.json")
    require(fit.get("state") == "complete" and fit.get("execution_kind") == "fresh_official_sequence"
            and fit["selection"] == selection and selection["arc"] == "half" and selection["full_roster"] is True,
            "The original 24-pose selection is incomplete or changed.")
    pin(run / "fit/fitted.npz", fit["fit_sha256"])
    rows_path = run / "data/training-inputs.json"
    rows = load(rows_path, fit["training_inputs_sha256"])["records"]
    replay_path = run / "gaussian-run/step-010000/posed-replay/result.json"
    panel_path = run / "audit/bound-panels/result.json"
    require(result["bound_replay"] == replay_path.relative_to(run).as_posix()
            and result["bound_panels"] == panel_path.relative_to(run).as_posix(), "Unexpected review paths.")
    replay = load(replay_path, result["bound_replay_sha256"])
    panels = load(panel_path, result["bound_panels_sha256"])
    selected, held = selection["selected_frames"], selection["retained_holdouts"]
    reference = selection["reference_frame"]
    expected = [reference] + [frame for frame in selected if frame != reference]
    require(len(expected) == len(set(expected)) == 24 and set(expected).isdisjoint(held)
            and expected == [row["frame"] for row in rows]
            == [row["frame"] for row in replay["records"]] == [row["frame"] for row in panels["panels"]]
            and replay["retained_holdouts"] == held, "Review roster differs from the protected training selection.")
    for receipt in (result, replay, panels):
        require(receipt["state"] == "completed" and receipt["reference_frame_id"] == reference
                and receipt["reference_pose_index"] == 0, "Reference identity differs.")
    require(replay["optimizer_updates"] == panels["optimizer_updates"] == 0
            and panels["same_frame_bound_gaussians"] is True and panels["native_crops"] is True
            and replay["training_inputs_sha256"] == fit["training_inputs_sha256"]
            and panels["replay_receipt_sha256"] == result["bound_replay_sha256"]
            and panels["mesh_sha256"] == replay["mesh_sha256"] and replay["mesh_file"] == "meshes.npz",
            "The panels do not bind the completed same-frame replay.")
    pin(replay_path.parent / replay["mesh_file"], replay["mesh_sha256"])
    snapshot = replay_path.parent.parent
    bound = load(snapshot / "bound-replay.json", replay["bound_replay_sha256"])
    require(bound["bound_ply"] == "bound-local.ply" and bound["sh_degree"] == 0
            and bound["frames"] == [{key: item[key] for key in IDENTITY} for item in replay["records"]],
            "The trained bound model did not supply those exact pose IDs.")
    pin(snapshot / bound["bound_ply"], bound["bound_ply_sha256"])
    native = {row["frame"]: row for row in decoded["records"]}
    sources = []
    for slot, (row, posed, panel) in enumerate(zip(rows, replay["records"], panels["panels"], strict=True)):
        frame = row["frame"]
        require(re.fullmatch(r"frame_\d{6}", frame) and slot == posed["pose_index"] == panel["pose_index"]
                and row["source_frame_index"] == posed["source_frame_index"] == panel["source_frame_index"]
                == native[frame]["source_frame_index"]
                and row["source_sha256"] == posed["source_sha256"] == panel["source_sha256"]
                == native[frame]["source_sha256"]
                and row["mask_sha256"] == posed["mask_sha256"]
                and posed["image"] == panel["panel"] == frame + ".png"
                and panel["gaussian_sha256"] == posed["image_sha256"], f"Mixed evidence for {frame}.")
        pin(row["image"], row["source_sha256"])
        pin(row["mask"], row["mask_sha256"])
        pin(replay_path.parent / posed["image"], posed["image_sha256"])
        source = pin(panel_path.parent / panel["panel"], panel["panel_sha256"])
        sources.append((source, destination / f"{PREFIX}{frame}.jpg"))

    changed = None
    if prior_run is not None:
        prior = Path(prior_run).resolve(strict=True)
        prior_info = load(prior / "run.json")
        require(prior_info["video_sha256"] == info["video_sha256"], "Prior roster is from a different video.")
        prior_fit = load(prior / "fit/official-fit-result.json")
        previous = load(prior / "data/training-inputs.json", prior_fit["training_inputs_sha256"])["records"]
        current_pairs = {(row["frame"], row["source_frame_index"]) for row in rows}
        previous_pairs = {(row["frame"], row["source_frame_index"]) for row in previous}
        changed = current_pairs != previous_pairs

    report = dict(state="verified", publication_requested=publish, frames=len(sources),
                  prior_roster_changed=changed, reference_frame=reference,
                  video_sha256=info["video_sha256"], finished_result_sha256=pins[run / "result.json"],
                  panel_receipt_sha256=result["bound_panels_sha256"], replay_receipt_sha256=result["bound_replay_sha256"],
                  selected=[dict(frame=row["frame"], encoded_frame=row["source_frame_index"]) for row in rows],
                  images=[])
    if not publish:
        return report

    with TemporaryDirectory(prefix="bound016-record-") as temporary:
        staged = []
        for source, target in sources:
            jpeg = Path(temporary) / target.name
            with Image.open(source) as panel:
                panel.convert("RGB").save(jpeg, quality=95, subsampling=0)
            staged.append((jpeg, target))
        if changed:
            jpeg = Path(temporary) / CONTACT
            contact_sheet(rows, jpeg)
            staged.append((jpeg, destination / CONTACT))
        files = []
        for jpeg, target in staged:
            with Image.open(jpeg) as converted:
                converted.verify()
            digest = sha(jpeg)
            require(not target.is_symlink() and (not target.exists() or sha(target) == digest),
                    f"Existing record differs; preserving {target}.")
            files.append((jpeg, target, digest))
        require(all(sha(path) == digest for path, digest in pins.items()), "A source changed during conversion.")
        created = []
        try:
            for jpeg, target, digest in files:
                if not target.exists():
                    with jpeg.open("rb") as source, target.open("xb") as output:
                        created.append(target)
                        copyfileobj(source, output)
                require(sha(target) == digest, f"Archived JPEG hash differs: {target}.")
                report["images"].append(dict(path=str(target), sha256=digest))
        except BaseException:
            for target in created:
                target.unlink(missing_ok=True)
            raise
    report["state"] = "archived"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prior-run", type=Path, help="Compare an earlier complete roster and include an original-frame sheet if it differs.")
    parser.add_argument("--publish", action="store_true", help="Append verified JPEGs. The default only verifies.")
    args = parser.parse_args()
    report = archive(args.run, args.destination, prior_run=args.prior_run, publish=args.publish)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
