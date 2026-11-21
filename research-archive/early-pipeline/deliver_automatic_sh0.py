"""Deliver a completed SH0 run using the existing rigid exporter and viewer package."""

import argparse
import fcntl
import inspect
import json
import shutil
import subprocess
from pathlib import Path

from run_official_gaussians123 import now, read, save, sha


STEP = 18000
SNAPSHOT = f"step-{STEP:06d}"
TRAINED = "completed_pending_quality_review"
MESH_FILE = "reference-mesh.obj"
MESH_PROVENANCE = "mesh-provenance.json"
VIEWER = Path("/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer")
WINDOWS_NODE = Path("/mnt/d/Program Files/nodejs/node.exe")

PACKAGE = """
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [folder, viewer] = process.argv.slice(2);
const scene = await import(pathToFileURL(path.join(viewer, 'scene-package.js')));
const {storedZip} = await import(pathToFileURL(path.join(viewer, 'camera-formats.js')));
const file = name => path.join(folder, name);
const read = async name => JSON.parse(await fs.readFile(file(name), 'utf8'));
const ply = await fs.readFile(file('point_cloud.ply'));
const [camera, context, validation, source] = await Promise.all([
  read('cameras.json'), read('context.json'), read('validation.json'), read('source-receipt.json')
]);
const meshName = 'reference-mesh.obj', meshProvenanceName = 'mesh-provenance.json';
const mesh = await fs.readFile(file(meshName));
const meshProvenance = await read(meshProvenanceName);
if (meshProvenance.mesh_sha256 !== await scene.sha256(mesh) ||
    meshProvenance.camera_sha256 !== await scene.sha256(await fs.readFile(file('cameras.json'))) ||
    meshProvenance.ply_sha256 !== validation.sha256 || meshProvenance.reference_frame !== source.reference_frame) {
  throw new Error('Reference mesh does not match its Gaussian and camera');
}
const portable = await scene.scenePackage(Uint8Array.from(ply).buffer, 'ply', camera, context);
const files = Object.fromEntries((await scene.unpackSceneZip(portable.bytes)).map(entry => [entry.name, entry.data]));
files['validation.json'] = JSON.stringify(validation, null, 2);
files['source-receipt.json'] = JSON.stringify(source, null, 2);
const decoder = new TextDecoder(), encoder = new TextEncoder();
const manifest = JSON.parse(decoder.decode(files['scene.json']));
meshProvenance.packaged_camera_sha256 = manifest.cameras.sha256;
const provenanceBytes = encoder.encode(JSON.stringify(meshProvenance, null, 2) + '\\n');
const provenanceHash = await scene.sha256(provenanceBytes);
manifest.referenceMesh = {file: meshName, sha256: meshProvenance.mesh_sha256, provenance: meshProvenanceName,
  provenanceSha256: provenanceHash, referenceFrameId: meshProvenance.reference_frame,
  vertexCount: meshProvenance.vertex_count, triangleCount: meshProvenance.triangle_count};
files['scene.json'] = JSON.stringify(manifest, null, 2) + '\\n';
files[meshName] = mesh;
files[meshProvenanceName] = provenanceBytes;
files['README.txt'] = decoder.decode(files['README.txt']) +
  '\\nreference-mesh.obj is the trained shared-offset SMPL-X surface for the same reference pose and camera.\\n';
const zip = storedZip(files);
const entries = (await scene.unpackSceneZip(zip)).map(entry => {
  const item = new File([entry.data], path.basename(entry.name));
  item.sceneRelativePath = entry.name;
  return item;
});
const models = await scene.primaryModelFiles(entries);
if (models.length !== 1 || await scene.sha256(await models[0].arrayBuffer()) !== validation.sha256) {
  throw new Error('Scene archive does not contain the validated Gaussian PLY');
}
const paired = await scene.pairedCameras(models[0], entries, models);
if (!paired.verifiedBinding || paired.cameras.length !== 1 || paired.referenceFrameId !== source.reference_frame) {
  throw new Error('Scene archive does not contain the matching reference camera');
}
for (const [name, hash] of [[meshName, meshProvenance.mesh_sha256], [meshProvenanceName, provenanceHash]]) {
  const entry = entries.find(item => item.sceneRelativePath === name);
  if (!entry || await scene.sha256(await entry.arrayBuffer()) !== hash) {
    throw new Error('Scene archive mesh readback failed');
  }
}
await fs.writeFile(file(meshProvenanceName + '.tmp'), provenanceBytes);
await fs.rename(file(meshProvenanceName + '.tmp'), file(meshProvenanceName));
const temporary = file('scene.zip.tmp');
await fs.writeFile(temporary, zip);
await fs.rename(temporary, file('scene.zip'));
console.log(JSON.stringify({passed: true, points: validation.points, bytes: zip.length,
  sha256: await scene.sha256(zip), cameraCount: paired.cameras.length, verifiedBinding: true,
  referenceFrameId: paired.referenceFrameId, shDegree: 0, sibrGuiVerified: false,
  meshSha256: meshProvenance.mesh_sha256, meshProvenanceSha256: provenanceHash,
  meshVertexCount: meshProvenance.vertex_count, meshTriangleCount: meshProvenance.triangle_count}));
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_receipt(out):
    run = out / "gaussian-run"
    snapshot = run / SNAPSHOT
    plan_path, terminal_path = out / "plan.json", run / "result.json"
    inputs_path, result_path = out / "data/training-inputs.json", snapshot / "result.json"
    exit_path, progress_path = out / "process-exit.json", out / "progress.json"
    plan, terminal, result, progress, exit_status = map(read, (
        plan_path, terminal_path, result_path, progress_path, exit_path,
    ))
    require(exit_status["exit_code"] == 0 and exit_status["stage"] == "training"
            and exit_status["training_function_returned"] is True
            and exit_status["terminal_result_sha256"] == sha(terminal_path),
            "Terminal trainer exit and result must agree")
    require(progress["state"] == terminal["state"] == TRAINED and progress["pid"] is None
            and progress["step"] == terminal["steps"] == result["step"] == STEP,
            "The terminal 18000-step snapshot is required")

    paths = plan["source_hashes"]
    require(paths and all(sha(Path(path)) == digest for path, digest in paths.items()),
            "Training source, inputs or code differ from the pinned plan")
    rows = read(inputs_path)["records"]
    frames = [row["frame"] for row in rows]
    require(rows and len(rows) == result["frames_rendered"] == plan["training_frames"],
            "Training roster differs from the final snapshot")
    require(len(set(frames)) == len(frames) and set(frames) == set(plan["selected_frames"])
            and str(inputs_path) in paths, "Training images differ from the pinned roster")
    require(result["reference_slot"] == 0 and result["active_sh_degree"] == plan["model_sh_degree"] == 0,
            "Expected a first-slot reference and SH0")
    ref = rows[0]
    require(ref["frame"] == result["reference_frame"] == plan["reference_frame"]
            and sha(Path(ref["image"])) == ref["source_sha256"], "Reference image does not match the training roster")
    require(result["points"] > 0 and result["points"] == progress["points"] == terminal["points"],
            "Trainer point counts disagree")
    world_hash = sha(snapshot / "world.ply")
    require(world_hash == result["world_ply_sha256"], "Terminal source PLY hash mismatch")

    evidence = [plan_path, terminal_path, result_path, exit_path, progress_path, inputs_path,
                out / "fit/surface-cache.npz", snapshot / "current-meshes.npz",
                snapshot / "binding.npz", snapshot / "checkpoint.pt"]
    return dict(step=STEP, terminal_verified=True, training_state=TRAINED, quality_accepted=False, points=result["points"],
                source_ply_sha256=world_hash, reference_frame=ref["frame"], reference_size=ref["wh"],
                reference_corner_K=ref["K"], reference_w2c=ref["w2c"], reference_image_sha256=ref["source_sha256"],
                actual_frames=frames, training_files_sha256={str(path.relative_to(out)): sha(path) for path in evidence},
                exporter_files_sha256={name: sha(Path(__file__).with_name(name)) for name in (
                    "deliver_automatic_sh0.py", "export_cross_subject_sh0124.py", "export_demo_body_snapshot123.py",
                    "verify_demo_head_export123.py",
                )}, viewer_files_sha256={name: sha(VIEWER / name) for name in ("scene-package.js", "camera-formats.js")})


def export_reference_mesh(out, folder, source, validation):
    import numpy as np

    snapshot = out / "gaussian-run" / SNAPSHOT
    mesh_path, binding_path = snapshot / "current-meshes.npz", snapshot / "binding.npz"
    cache_path = out / "fit/surface-cache.npz"
    source_hashes = source["training_files_sha256"]
    mesh_hash, binding_hash, cache_hash = (source_hashes[str(path.relative_to(out))]
                                           for path in (mesh_path, binding_path, cache_path))
    require(sha(mesh_path) == mesh_hash and sha(binding_path) == binding_hash and sha(cache_path) == cache_hash,
            "Reference mesh inputs changed during delivery")
    frames = source["actual_frames"]
    slot = 0
    with np.load(mesh_path, allow_pickle=False) as mesh, np.load(binding_path, allow_pickle=False) as binding, \
            np.load(cache_path, allow_pickle=False) as cache:
        require(all(archive["frame_ids"].tolist() == frames for archive in (mesh, binding, cache))
                and frames[slot] == source["reference_frame"], "Reference mesh is not in the registered pose slot")
        camera_vertices = mesh["vertices_camera"][slot].astype(np.float64)
        faces = mesh["faces"]
        base, linear = cache["base_vertices"][slot], cache["linear"][slot]
        static_offset = binding["static_offset"]
        allowed = cache["clothing_vertices"].astype(bool) & ~cache["protected_face_vertices"].astype(bool)
        require(faces.dtype.kind in "iu" and faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0
                and np.array_equal(faces, cache["faces"]) and faces.min() >= 0 and faces.max() < len(base)
                and len(binding["triangle_ids"]) == source["points"], "Reference mesh topology differs from Gaussian binding")
        require(camera_vertices.shape == static_offset.shape == base.shape and base.ndim == 2 and base.shape[1] == 3
                and linear.shape == (len(base), 3, 3) and allowed.shape == (len(base),)
                and np.isfinite(camera_vertices).all() and (camera_vertices[:, 2] > .05).all()
                and np.isfinite(static_offset).all(), "Invalid trained reference mesh geometry")
        offsets = static_offset * allowed[:, None]
        replay_world = base + (linear @ offsets[..., None]).squeeze(-1)

    E = np.asarray(source["reference_w2c"], dtype=np.float64)
    R = np.asarray(validation["rotation"], dtype=np.float64)
    origin = np.asarray(validation["origin_world"], dtype=np.float64)
    scale = float(validation["scale"])
    require(E.shape == (4, 4) and R.shape == (3, 3) and origin.shape == (3,)
            and np.isfinite(E).all() and np.isfinite(origin).all() and np.isfinite(scale) and scale > 0
            and np.allclose(R @ R.T, np.eye(3), atol=1e-8, rtol=0) and np.linalg.det(R) > 0,
            "Invalid PLY coordinate transform for the reference mesh")
    replay_camera = replay_world @ E[:3, :3].T + E[:3, 3]
    replay_error = float(np.max(np.abs(replay_camera - camera_vertices)))
    require(replay_error < 2e-6, "Stored mesh differs from the replay of the trained shared offsets")
    world = np.linalg.solve(E[:3, :3], (camera_vertices - E[:3, 3]).T).T
    vertices = scale * ((world - origin) @ R.T)

    [camera] = read(folder / "cameras.json")
    basis = np.asarray(camera["rotation"], dtype=np.float64)
    eye = np.asarray(camera["position"], dtype=np.float64)
    exported_camera = (vertices - eye) @ basis
    camera_error = float(np.max(np.abs(exported_camera - scale * camera_vertices)))
    K = np.asarray(source["reference_corner_K"], dtype=np.float64)
    focal, center = np.array([camera["fx"], camera["fy"]]), np.array([camera["cx"], camera["cy"]])
    require(camera["img_name"] == source["reference_frame"] and camera["pixelConvention"] == "pixel_corners"
            and [camera["width"], camera["height"]] == source["reference_size"]
            and np.array_equal(focal, K.diagonal()[:2]) and np.array_equal(center, K[:2, 2])
            and camera_error < 2e-6 and (exported_camera[:, 2] > 0).all(),
            "Mesh and PLY reference cameras differ")
    projected = exported_camera[:, :2] / exported_camera[:, 2:] * focal + center
    native = camera_vertices[:, :2] / camera_vertices[:, 2:] * focal + center
    pixel_error = float(np.linalg.norm(projected - native, axis=1).max())
    require(pixel_error < 1e-5, "Reference mesh projection changed in export")

    target = folder / MESH_FILE
    temporary = target.with_suffix(".obj.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Trained SMPL-X shared-offset surface in the paired Gaussian PLY coordinates\n")
        np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces.astype(np.int64) + 1, fmt="f %d %d %d")
    data = np.loadtxt(temporary, dtype=str)
    v, f = data[:len(vertices)], data[len(vertices):]
    require(v.shape == (len(vertices), 4) and f.shape == (len(faces), 4)
            and (v[:, 0] == "v").all() and (f[:, 0] == "f").all()
            and np.array_equal(v[:, 1:].astype(float), vertices)
            and np.array_equal(f[:, 1:].astype(np.int64) - 1, faces), "Reference OBJ roundtrip failed")
    temporary.replace(target)
    save(folder / MESH_PROVENANCE, dict(
        schema="shuolee-reference-mesh-v1", mesh_file=MESH_FILE, mesh_sha256=sha(target),
        ply_sha256=validation["sha256"], camera_sha256=validation["serialized_camera_sha256"],
        source_meshes_sha256=mesh_hash, source_binding_sha256=binding_hash, source_surface_cache_sha256=cache_hash,
        reference_frame=source["reference_frame"], reference_slot=slot, step=STEP, sh_degree=0,
        vertex_count=len(vertices), triangle_count=len(faces), coordinate_space=validation["coordinate_space"],
        rotation=R.tolist(), origin_world=origin.tolist(), scale=scale, shared_vertexoffset_applied=True,
        nonzero_offset_vertices=int(np.count_nonzero(np.any(offsets, axis=1))),
        offset_max_m=float(np.linalg.norm(offsets, axis=1).max()), source_mesh_replay_max_abs_m=replay_error,
        camera_space_max_abs=camera_error, reprojection_max_px=pixel_error,
        format_passed=True, quality_accepted=False,
    ))


def package(folder):
    native_node = shutil.which("node")
    paths = [str(folder), str(VIEWER)]
    if not native_node:
        require(WINDOWS_NODE.is_file(), "The existing viewer packager requires Node.js")
        paths = [subprocess.check_output(["wslpath", "-w", path], text=True).strip() for path in paths]
    completed = subprocess.run([native_node or str(WINDOWS_NODE), "--input-type=module", "-", *paths],
                               input=PACKAGE, text=True, stdout=subprocess.PIPE, check=True)
    return json.loads(completed.stdout)


def export(out):
    import export_cross_subject_sh0124 as exporter

    # The existing exporter also serves bounded trials. Only this entry guard
    # changes for a training function that returned after all 18000 steps.
    code = inspect.getsource(exporter.main)
    old_guard = "assert progress['state']=='bounded_control_completed',progress"
    require(code.count(old_guard) == 1, "The existing export entry guard changed")
    scope = dict(vars(exporter), ROOT=out.parent)
    code = code.replace(old_guard, f"assert progress['state']=={TRAINED!r},progress")
    exec(compile(code, "<automatic-sh0-terminal-export>", "exec"), scope)
    scope["main"](out.name, STEP)


def deliver(out: Path) -> dict:
    """Return the final package receipt after export and format readback."""
    out = Path(out).resolve()
    delivery = out / "delivery"
    folder, publication = delivery / SNAPSHOT, delivery / "publication.json"
    delivery.mkdir(exist_ok=True)
    with (delivery / ".automatic-sh0.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = source_receipt(out)
        if publication.exists():
            previous = read(publication)
            require(previous["state"] == "delivered_pending_quality_review"
                    and previous["source_receipt_sha256"] == sha(folder / "source-receipt.json")
                    and read(folder / "source-receipt.json") == source
                    and previous["ply_sha256"] == sha(folder / "point_cloud.ply")
                    and previous["camera_sha256"] == sha(folder / "cameras.json")
                    and previous.get("mesh_sha256") == sha(folder / MESH_FILE)
                    and previous.get("mesh_provenance_sha256") == sha(folder / MESH_PROVENANCE)
                    and previous["package_sha256"] == sha(folder / "scene.zip"),
                    "An existing delivery differs from this terminal run")
            return previous

        if not folder.exists():
            export(out)

        validation = read(folder / "validation.json")
        fields = read(folder / "sh0-field-check.json")
        ply_hash, camera_hash = sha(folder / "point_cloud.ply"), sha(folder / "cameras.json")
        require(validation["format_passed"] and validation["native_source_ply_parity"]["passed"]
                and validation["native_rotated_render_parity"]["passed"] and fields["passed"]
                and fields["sha256"] == validation["sha256"] == ply_hash
                and validation["serialized_camera_sha256"] == camera_hash
                and validation["reference_frame"] == source["reference_frame"]
                and validation["source_sha256"] == source["source_ply_sha256"]
                and validation["points"] == fields["points"] == source["points"],
                "Existing export does not pass source, native parity, camera and SH0 checks")
        save(folder / "source-receipt.json", source)
        export_reference_mesh(out, folder, source, validation)
        package_result = package(folder)
        require(package_result["sha256"] == sha(folder / "scene.zip")
                and package_result["meshSha256"] == sha(folder / MESH_FILE)
                and package_result["meshProvenanceSha256"] == sha(folder / MESH_PROVENANCE),
                "Scene archive or reference mesh hash mismatch")
        save(folder / "package-result.json", package_result)
        receipt = dict(updated=now(), state="delivered_pending_quality_review", quality_accepted=False,
                       viewer_ui_review="pending", run=out.name, step=STEP, points=source["points"],
                       reference_frame=source["reference_frame"], artifact_dir=str(folder),
                       ply_path=str(folder / "point_cloud.ply"), camera_path=str(folder / "cameras.json"),
                       package_path=str(folder / "scene.zip"), package_sha256=package_result["sha256"],
                       ply_sha256=ply_hash, camera_sha256=camera_hash, format_passed=True,
                       mesh_path=str(folder / MESH_FILE), mesh_sha256=package_result["meshSha256"],
                       mesh_provenance_path=str(folder / MESH_PROVENANCE),
                       mesh_provenance_sha256=package_result["meshProvenanceSha256"],
                       mesh_vertex_count=package_result["meshVertexCount"],
                       mesh_triangle_count=package_result["meshTriangleCount"],
                       source_receipt_sha256=sha(folder / "source-receipt.json"))
        save(publication, receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    print(json.dumps(deliver(parser.parse_args().out), ensure_ascii=False))
