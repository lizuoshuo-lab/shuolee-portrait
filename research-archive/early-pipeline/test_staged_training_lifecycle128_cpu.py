"""Bounded CPU checks only; no person data, CUDA initialization or fitting."""
import hashlib
import inspect
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

import staged_training_lifecycle128 as stage


CHECKS = []


def seed(value=128):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def trace_loader(original, n, explicit):
    seed()
    generator = torch.Generator().manual_seed(923) if explicit else None
    if original:
        loader = DataLoader(list(range(n)), batch_size=None, shuffle=True,
                            num_workers=0, generator=generator)
    else:
        loader = stage.CheckpointableCameraLoader(list(range(n)), camera_ids=list(range(n)),
                            batch_size=None, shuffle=True, num_workers=0, generator=generator)
    iterator, rows = iter(loader), []
    for _ in range(n * 3 + 2):
        try:
            value = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            value = next(iterator)
        rows.append((value, torch.rand(3).tolist(), random.random(), float(np.random.rand())))
    return rows, stage.capture_rng_state(), generator.get_state() if explicit else None


class Model:
    """Tiny CPU fixture with the actual14-field capture/7-group restore contract."""
    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.timestep = 0
        self.num_timesteps = 5
        self._xyz = torch.nn.Parameter(torch.randn(4, 3))
        self._features_dc = torch.nn.Parameter(torch.randn(4, 1, 3))
        self._features_rest = torch.nn.Parameter(torch.randn(4, 1, 3))
        self._scaling = torch.nn.Parameter(torch.randn(4, 3))
        self._rotation = torch.nn.Parameter(torch.randn(4, 4))
        self._opacity = torch.nn.Parameter(torch.randn(4, 1))
        self.static_offset = torch.nn.Parameter(torch.zeros(4, 3))
        self.binding = torch.arange(4)
        self.binding_counter = torch.ones(4, dtype=torch.int32)
        self.max_radii2D = torch.zeros(4)
        self.xyz_gradient_accum = torch.zeros(4, 1)
        self.denom = torch.zeros(4, 1)
        self.spatial_lr_scale = 1.0

    def training_setup(self, options):
        params = [self._xyz, self._features_dc, self._features_rest, self._opacity,
                  self._scaling, self._rotation, self.static_offset]
        self.optimizer = torch.optim.Adam([dict(params=[p], name=name, lr=0.001 * (i + 1))
                                for i, (name, p) in enumerate(zip(stage.GROUPS, params))], eps=1e-15)
        self.xyz_gradient_accum = torch.zeros(4, 1)
        self.denom = torch.zeros(4, 1)

    def select_mesh_by_timestep(self, timestep):
        self.timestep = timestep
        self.derived_mesh = self.static_offset + self._xyz

    def capture(self):
        return (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
                self._scaling, self._rotation, self._opacity, self.binding, self.binding_counter,
                self.max_radii2D, self.xyz_gradient_accum, self.denom,
                self.optimizer.state_dict(), self.spatial_lr_scale)

    def restore(self, cap, options):
        (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
         self._scaling, self._rotation, self._opacity, self.binding, self.binding_counter,
         self.max_radii2D, accum, denom, state, self.spatial_lr_scale) = cap
        self.training_setup(options)
        self.xyz_gradient_accum, self.denom = accum, denom
        self.optimizer.load_state_dict(state)


class Pair:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(124)
        self.calls = 0
        self.first_colors = []

    def draw(self):
        chosen = torch.rand(3, generator=self.generator)
        self.calls += 1
        if self.calls <= 3:
            self.first_colors.append(chosen.tolist())
        return chosen


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def toy_run(dest, stop, source, resume=None, override_identity=None, tamper_optimizer=False):
    seed()
    model, paired = Model(), Pair()
    options = SimpleNamespace(iterations=18000, position_lr_max_steps=18000, original_option=0.05)
    frames = [f"camera_{i}" for i in range(5)]
    counts, traces = [0] * 5, []

    class Dataset:
        def __len__(self):
            return 5

        def __getitem__(self, i):
            counts[i] += 1
            return i

    def get():
        return dict(counts=list(counts), traces=list(traces))

    def put(state):
        counts[:], traces[:] = state["counts"], state["traces"]

    identity = dict(subject="CPU synthetic only", options=vars(options), frames=frames)
    if override_identity:
        identity.update(override_identity)
    life = stage.StagedTrainingLifecycle(model=model, paired_background=paired,
            camera_ids=frames, identity=identity, source_hashes={str(source): stage.sha256_file(source)},
            total_iterations=18000, stop_iterations=[stop], optimization=options,
            counters_getter=get, counters_setter=put)
    proxy = life.torch_proxy(torch)
    model.training_setup(options)
    first = 0
    if resume:
        cap, first = proxy.load(resume)
        model.restore(cap, options)
        if tamper_optimizer:
            model.optimizer.state[model.static_offset]["exp_avg"].add_(0.1)
    loader = life.camera_loader(Dataset(), batch_size=None, shuffle=True, num_workers=8,
                               pin_memory=True, persistent_workers=True)
    iterator = iter(loader)
    for step in range(first + 1, stop + 1):
        try:
            camera = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            camera = next(iterator)
        model.select_mesh_by_timestep(camera)
        draws = [random.random(), float(np.random.rand()), float(torch.rand(()))]
        background = paired.draw()
        coefficients = sum(draws) + float(background.sum()) + camera
        loss = sum((p.square().sum() + p.sum() * coefficients)
                   for group in model.optimizer.param_groups for p in group["params"])
        loss.backward()
        model.max_radii2D += 0.25
        model.xyz_gradient_accum += 0.125
        model.denom += 1
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        traces.append(dict(step=step, camera=camera, draws=draws, background=background.tolist(), loss=float(loss)))
        if step == stop:
            try:
                proxy.save((model.capture(), step), dest / f"chkpnt{step}.pth")
            except stage.ReviewStop as stopped:
                assert Path(stopped.manifest).is_file()
            else:
                raise AssertionError("Missing clean staged stop")
    assert sum(counts) == stop and paired.calls == stop
    assert all(p.grad is None for g in model.optimizer.param_groups for p in g["params"])
    return cpu_tree(dict(capture=model.capture(), offset=model.static_offset, counters=get(),
                         RNG=stage.capture_rng_state(), paired=paired.generator.get_state(),
                         sampler=loader.state_dict())), life


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    for n in (1, 5, 24):
        for explicit in (False, True):
            assert stage._equal(trace_loader(True, n, explicit), trace_loader(False, n, explicit))
            CHECKS.append(f"original_DataLoader_RandomSampler_n{n}_explicit_generator_{explicit}_3epochs_exact")
    with tempfile.TemporaryDirectory(prefix="staged-lifecycle128-cpu-") as tmp:
        root = Path(tmp)
        source = root / "frozen-fixture.txt"
        source.write_text("no person data; frozen CPU lifecycle fixture\n")
        continuous, staged = root / "continuous", root / "staged"
        continuous.mkdir(); staged.mkdir()
        full, _ = toy_run(continuous, 26, source)
        first, _ = toy_run(staged, 8, source)
        assert first["sampler"]["sampler"]["active"]["cursor"] == 3
        final, life = toy_run(staged, 26, source, resume=staged / "chkpnt8.pth")
        assert stage._equal(full, final)
        assert life.last_resume_receipt["next_iteration"] == 9
        assert life.prior_compute_seconds > 0
        assert all(int(v["step"]) == 26 for v in final["capture"][12]["state"].values())
        assert torch.count_nonzero(final["offset"]) > 0
        CHECKS.extend(["CPU_continuous26_vs_stage8_resume26_all_params_7Adam_offset_exact",
                       "four_rng_families_CPU_state_and_next_draws_exact_CUDA_uninitialized",
                       "camera_remainder_and_cross_epoch_order_exact",
                       "paired_generator_calls_and_colors_exact",
                       "original18000_clock_and_resume9_retained",
                       "compute_prior_preserved_no_clock_reset",
                       "post_update_gradients_cleared_and_checkpoint_manifest_SHA_bound"])
        try:
            toy_run(staged, 26, source, resume=staged / "chkpnt8.pth", override_identity={"frames_changed": True})
        except AssertionError:
            CHECKS.append("identity_mismatch_rejected")
        else:
            raise AssertionError("Changed identity accepted")
        try:
            toy_run(staged, 26, source, resume=staged / "chkpnt8.pth", tamper_optimizer=True)
        except AssertionError:
            CHECKS.append("post_restore_Adam_mutation_rejected_using_independent_native_snapshot")
        else:
            raise AssertionError("Changed restored Adam accepted")
        marker = staged / "chkpnt8.pth.lifecycle.json"
        good = marker.read_bytes()
        manifest = json.loads(good)
        manifest["sidecar_sha256"] = "0" * 64
        marker.write_text(json.dumps(manifest))
        try:
            toy_run(staged, 26, source, resume=staged / "chkpnt8.pth")
        except AssertionError:
            CHECKS.append("sidecar_hash_mismatch_rejected")
        else:
            raise AssertionError("Changed sidecar accepted")
        finally:
            marker.write_bytes(good)
    assert not torch.cuda.is_initialized()
    H = Path(__file__).resolve().parent
    result = dict(passed=True, scope="bounded_CPU_lifecycle_only", GPU_executed=False,
        CUDA_initialized=False, person_data_read=False, person_fit_or_training=False,
        native_GPU_control_still_required=True, quality_accepted=False,
        helper_sha256=stage.sha256_file(H / "staged_training_lifecycle128.py"),
        test_sha256=stage.sha256_file(__file__), torch_version=str(torch.__version__),
        numpy_version=np.__version__, original_RandomSampler_source_sha256=hashlib.sha256(
            inspect.getsource(RandomSampler).encode()).hexdigest(),
        checks=CHECKS, check_count=len(CHECKS))
    output = H / "staged_training_lifecycle128_cpu_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, check_count=len(CHECKS), receipt=str(output),
                         helper_sha256=result["helper_sha256"])))


if __name__ == "__main__":
    main()
