"""Export a standalone head and neck sequence from the fitted SMPL-X surface."""

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np

from review_smplx_head_neck import mesh_overlay, panel


def boundary_loops(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold edge in the head and neck topology")
    graph = {}
    for a, b in edges[counts == 1]:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in graph.values()):
        raise ValueError("Expected a closed ring around each open mesh boundary")
    remaining = set(graph)
    loops = []
    while remaining:
        start = min(remaining)
        path = [start]
        previous, current = start, graph[start][0]
        while current != start:
            path.append(current)
            a, b = graph[current]
            previous, current = current, b if a == previous else a
        remaining.difference_update(path)
        loops.append(np.asarray(path, dtype=np.int32))
    return loops


def clip(source_faces, weights, cutoff):
    """Clip each source triangle once and retain barycentric correspondence across frames."""
    inside = weights >= cutoff
    vertex_index = {}
    endpoints, fractions, faces, parents = [], [], [], []

    def register(a, b):
        key = (min(a, b), max(a, b))
        if key not in vertex_index:
            vertex_index[key] = len(endpoints)
            endpoints.append(key)
            first, second = key
            fractions.append(0. if first == second else (cutoff - weights[first]) / (weights[second] - weights[first]))
        return vertex_index[key]

    for parent, face in source_faces:
        polygon = []
        for a, b in zip(face, np.roll(face, -1), strict=True):
            if inside[a]:
                polygon.append(register(a, a))
            if inside[a] != inside[b]:
                polygon.append(register(a, b))
        for i in range(1, len(polygon) - 1):
            faces.append([polygon[0], polygon[i], polygon[i + 1]])
            parents.append(parent)
    return (np.asarray(endpoints, dtype=np.int32), np.asarray(fractions, dtype=np.float32),
            np.asarray(faces, dtype=np.int32), np.asarray(parents, dtype=np.int32))


def extract(mesh, model, mapping, masks):
    source_faces = mesh["faces"]
    if not np.array_equal(source_faces, model["f"]) or int(mesh["neck_joint_index"]) != 12:
        raise ValueError("Expected the reference SMPL-X faces and neck joint")
    included = np.isin(source_faces, mapping).all(axis=1)
    reference_faces = source_faces[included]
    loops = boundary_loops(reference_faces)
    if len(loops) != 2:
        raise ValueError("Expected the FLAME neck and mouth boundaries")
    neutral_y = model["v_template"][:, 1]
    bottom = min(loops, key=lambda ring: neutral_y[ring].mean())
    parents = model["kintree_table"][0]
    head_neck = {12}
    for child, parent in enumerate(parents):
        if int(parent) in head_neck:
            head_neck.add(child)
    weights = model["weights"][:, sorted(head_neck)].sum(axis=1)
    face_neck = mapping[np.unique(np.r_[masks["face"], masks["neck"]])]
    rim_weight, skin_weight = weights[bottom].max(), weights[face_neck].min()
    if skin_weight <= rim_weight:
        raise ValueError("The mapped neck surface overlaps the old lower rim in skinning weight")
    cutoff = float((rim_weight + skin_weight) / 2)
    ends, fraction, faces, source_ids = clip(zip(np.flatnonzero(included), reference_faces, strict=True), weights, cutoff)
    if not np.isin(face_neck, ends[ends[:, 0] == ends[:, 1], 0]).all():
        raise RuntimeError("The new mesh must retain every mapped face and neck point")
    posed = mesh["vertices_world"]
    vertices = posed[:, ends[:, 0], :] * (1 - fraction[None, :, None]) + posed[:, ends[:, 1], :] * fraction[None, :, None]
    rings = boundary_loops(faces)
    neck_rings = [ring for ring in rings if np.all(ends[ring, 0] != ends[ring, 1])]
    if len(rings) != 2 or len(neck_rings) != 1:
        raise RuntimeError("Expected a single neck boundary in addition to the original mouth opening")
    for frame in vertices:
        a, b, c = frame[faces].transpose(1, 0, 2)
        if not np.isfinite(frame).all() or np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) <= 1.e-10):
            raise RuntimeError("Nonfinite coordinates or degenerate head and neck triangle")
    return vertices, faces, ends, fraction, neck_rings[0], source_ids, cutoff


def review(args, records, names, vertices, faces, source, frame_archive):
    fixed = {row["name"] for row in records if abs(row["baseline_yaw_deg"]) >= 80}
    if not fixed:
        raise ValueError("No pure side frames to include in the review")
    command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", "1152x384", "-framerate", "3", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output / "all-frames.mp4")]
    K, centers, rotations = (source[key] for key in ("native_K", "native_camera_center", "native_camera_to_world_rotation"))
    sheet = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as video:
        try:
            for index, (name, record, frame) in enumerate(zip(names, records, vertices, strict=True)):
                encoded = frame_archive.read(f"images/{index:08d}.jpg")
                rgb = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise OSError(f"The registered RGB crop cannot be decoded: {name}")
                warp = np.asarray(record["warp"])
                virtual = np.asarray(record["virtual_from_native_cv"])
                projection = warp @ K
                if (not np.allclose(projection, np.asarray(record["K_integer_pixel_centers"]) @ virtual, atol=1e-6)
                        or not np.allclose(rotations[index] @ virtual.T, record["world_from_virtual"], atol=1e-6)
                        or not np.allclose(centers[index], record["camera_center_world"], atol=1e-6)):
                    raise ValueError(f"The mesh and RGB crop use different cameras: {name}")
                camera = (frame - centers[index]) @ rotations[index]
                overlay = mesh_overlay(rgb, camera, projection, faces)
                isolated = mesh_overlay(np.full_like(rgb, 32), camera, projection, faces)
                row = np.hstack([panel(rgb, f"{name} | source"), panel(overlay, "Head and neck overlay"),
                                 panel(isolated, "Head and neck mesh")])
                video.stdin.write(row.tobytes())
                if name in fixed:
                    sheet.append(row)
        finally:
            video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError("Failed to encode the head and neck review")
    if not cv2.imwrite(str(args.output / "profile-comparison.jpg"), np.vstack(sheet)):
        raise OSError("Failed to save the profile comparison")
    return [name for name in names if name in fixed]


def main(args):
    with args.masks.open("rb") as stream:
        masks = pickle.load(stream, encoding="latin1")
    with (np.load(args.smplx_frames, allow_pickle=False) as mesh,
          np.load(args.model, allow_pickle=False) as model,
          ZipFile(args.frames) as archive):
        mapping = np.load(args.correspondence, allow_pickle=False)
        names = mesh["names"].tolist()
        records = json.loads(archive.read("initialization.json"))["frames"]
        if not names or len(set(names)) != len(names) or names != [r["name"] for r in records]:
            raise ValueError("Expected the same registered frame sequence in the mesh and image archive")
        vertices, faces, ends, fractions, neck, parents, cutoff = extract(mesh, model, mapping, masks)
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output / "head-neck-frames.npz", names=names, vertices_world=vertices, faces=faces,
                            timestamps_s=[r["timestamp_s"] for r in records], source_vertex_endpoints=ends,
                            source_edge_interpolation=fractions, source_face_ids=parents, neck_boundary_vertex_ids=neck,
                            neck_joint_world=mesh["joints_world"][:, 12], head_joint_world=mesh["joints_world"][:, 15],
                            scene_scale=mesh["scene_scale"], native_K=mesh["native_K"],
                            native_camera_to_world_rotation=mesh["native_camera_to_world_rotation"],
                            native_camera_center=mesh["native_camera_center"],
                            virtual_crop_homographies=[r["warp"] for r in records])
        fixed = review(args, records, names, vertices, faces, mesh, archive)
        sample = names.index(fixed[len(fixed) // 2])
        with (args.output / f"head-neck-{Path(names[sample]).stem}.obj").open("x", encoding="utf-8") as stream:
            np.savetxt(stream, vertices[sample], fmt="v %.8f %.8f %.8f")
            np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        report = {"frames": len(names), "fps": 3, "vertices_per_frame": vertices.shape[1], "faces_per_frame": len(faces),
                  "neck_ring_vertices": len(neck), "cutoff": cutoff, "profile_frames": fixed,
                  "scene_scale": float(mesh["scene_scale"]), "source_topology": "SMPL-X with FLAME head correspondence"}
        (args.output / "review.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("smplx-frames", "model", "correspondence", "masks", "frames", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    cv2.setNumThreads(1)
    main(parser.parse_args())
