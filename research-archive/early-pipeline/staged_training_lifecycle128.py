"""Adapter-only, state-complete stops at original post-update checkpoints.

This module does not train, alter a solver, or edit the upstream repository.
Use the pinned original ``training`` function with ``opt.iterations`` unchanged::

    life = StagedTrainingLifecycle(model=model, paired_background=paired,
        camera_ids=frames, identity=identity, source_hashes=hashes,
        total_iterations=18000, stop_iterations=[1000], optimization=opt)
    namespace.update(torch=life.torch_proxy(torch), DataLoader=life.camera_loader)
    try:
        training(dataset, opt, pipe, [], saves, [1000], checkpoint, -1)
    except ReviewStop as stop:
        # Normal staged completion, not a failed training or quality acceptance.
        ...

For resume construct the same model/assets/offset Parameter and a fresh ``life``
with the same identity and total clock. Pass the prior *native* chkpntNNN.pth to
the original checkpoint argument. Its load restores offset before base.restore;
the first iter(DataLoader), after base.restore, verifies Adam/model state and
restores the sampler/paired generator/all RNGs. Start at original first_iter+1.

The camera adapter supports the current deterministic map-style camera dataset,
batch_size=None, shuffle=True, replacement=False, num_workers=0. It reproduces
the original RandomSampler's lazy seed, randperm and empty-tail draw, including
the DataLoader base-seed draw. It does not substitute a new sampling distribution.
No exact-continuation claim is made until the caller's native GPU control passes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


SCHEMA = "staged-original-checkpoint128-v1"
GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "static_offset")
PARAM_CAPTURE = (1, 2, 3, 6, 4, 5)


class ReviewStop(RuntimeError):
    """Successful native checkpoint + complete sidecar; review may be offline."""
    def __init__(self, checkpoint, manifest):
        self.checkpoint = str(checkpoint)
        self.manifest = str(manifest)
        super().__init__("checkpoint_ready_for_offline_review: " + self.checkpoint)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity_hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _equal(a, b):
    if torch.is_tensor(a):
        return (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, Mapping):
        return isinstance(b, Mapping) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _snapshot_tree(value):
    """Independent values, not aliases to Parameters/Adam tensors loaded next."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(v) for v in value)
    if isinstance(value, list):
        return [_snapshot_tree(v) for v in value]
    return value


def capture_rng_state():
    """Does not initialize CUDA in a CPU-only process."""
    initialized = torch.cuda.is_initialized()
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                cuda_initialized=initialized,
                torch_cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if initialized else [])


def restore_rng_state(state):
    if state["cuda_initialized"]:
        assert torch.cuda.is_initialized(), "Recreate CUDA model before restoring CUDA RNG"
        assert len(state["torch_cuda"]) == torch.cuda.device_count(), "CUDA device topology changed"
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    else:
        assert not torch.cuda.is_initialized(), "CPU checkpoint cannot silently become a CUDA continuation"
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


class _SamplerEpoch:
    def __init__(self, sampler, state=None):
        self.owner = sampler
        self.started = False
        self.order = []
        self.cursor = 0
        self.tail_consumed = False
        self.generator = None
        if state is not None:
            self.started = state["started"]
            self.order = list(state["order"])
            self.cursor = int(state["cursor"])
            self.tail_consumed = state["tail_consumed"]
            assert all(type(i) is int for i in self.order)
            assert not self.started or sorted(self.order) == list(range(len(sampler)))
            assert 0 <= self.cursor <= len(self.order)
            if self.started:
                self.generator = sampler.generator or torch.Generator(device="cpu")
                self.generator.set_state(state["generator_state"])

    def __iter__(self):
        return self

    def __next__(self):
        n = len(self.owner)
        if not self.started:
            # This is deliberately lazy, as RandomSampler.__iter__ is a generator.
            if self.owner.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                self.generator = torch.Generator(device="cpu").manual_seed(seed)
            else:
                self.generator = self.owner.generator
            self.order = torch.randperm(n, generator=self.generator).tolist()
            self.started = True
        if self.cursor < n:
            value = self.order[self.cursor]
            self.cursor += 1
            self.owner.last_index = value
            return value
        if not self.tail_consumed:
            # Original num_samples==n still evaluates randperm(n)[:0] on exhaustion.
            torch.randperm(n, generator=self.generator)[:0].tolist()
            self.tail_consumed = True
        raise StopIteration

    def state_dict(self):
        return dict(started=self.started, order=list(self.order), cursor=self.cursor,
                    tail_consumed=self.tail_consumed,
                    generator_state=self.generator.get_state().clone() if self.started else None)


class CheckpointableRandomSampler(RandomSampler):
    """Exact no-replacement/n-samples RandomSampler, with resumable current epoch."""
    def __init__(self, data_source, generator=None):
        super().__init__(data_source, replacement=False, num_samples=None, generator=generator)
        assert len(data_source) > 0
        assert generator is None or str(generator.device) == "cpu"
        self.epoch = -1
        self.active = None
        self.last_index = None
        self._resume_once = False

    def __iter__(self):
        if self._resume_once:
            self._resume_once = False
            return self.active
        assert self.active is None or self.active.tail_consumed, "Cannot abandon a partially sampled epoch"
        self.epoch += 1
        self.active = _SamplerEpoch(self)
        return self.active

    def state_dict(self):
        assert self.active is not None
        return dict(length=len(self), epoch=self.epoch, last_index=self.last_index,
                    explicit_generator=self.generator is not None,
                    generator_state=self.generator.get_state().clone() if self.generator is not None else None,
                    active=self.active.state_dict())

    def load_state_dict(self, state):
        assert state["length"] == len(self)
        assert state["explicit_generator"] == (self.generator is not None)
        if self.generator is not None:
            self.generator.set_state(state["generator_state"])
        self.epoch = int(state["epoch"])
        self.last_index = state["last_index"]
        self.active = _SamplerEpoch(self, state["active"])
        self._resume_once = True


class _CameraIterator:
    def __init__(self, owner, inner):
        self.owner, self.inner = owner, inner

    def __iter__(self):
        return self

    def __next__(self):
        camera = next(self.inner)
        self.owner.sampled_count += 1
        self.owner.last_camera_id = self.owner.camera_ids[self.owner.sampler.last_index]
        return camera


class CheckpointableCameraLoader:
    """The existing zero-worker DataLoader with a serializable RandomSampler."""
    def __init__(self, dataset, *, camera_ids, lifecycle=None, **kwargs):
        assert kwargs.pop("batch_size", None) is None
        assert kwargs.pop("shuffle", True) is True
        assert kwargs.pop("sampler", None) is None
        assert kwargs.pop("batch_sampler", None) is None
        assert not kwargs.pop("drop_last", False)
        assert kwargs.pop("collate_fn", None) is None
        assert kwargs.pop("worker_init_fn", None) is None
        assert kwargs.pop("timeout", 0) == 0
        # These are the already-existing local camera runtime settings, not new workers.
        kwargs.pop("num_workers", None)
        kwargs.pop("persistent_workers", None)
        kwargs.pop("pin_memory", None)
        generator = kwargs.pop("generator", None)
        assert not kwargs, "Unsupported DataLoader options: " + repr(sorted(kwargs))
        self.camera_ids = tuple(camera_ids)
        assert len(dataset) == len(self.camera_ids) and len(set(self.camera_ids)) == len(self.camera_ids)
        self.sampler = CheckpointableRandomSampler(dataset, generator=generator)
        self.loader = DataLoader(dataset, batch_size=None, sampler=self.sampler, shuffle=False,
                                 num_workers=0, persistent_workers=False, pin_memory=False, generator=generator)
        self.lifecycle = lifecycle
        self.sampled_count = 0
        self.last_camera_id = None
        self._active = None

    def __len__(self):
        return len(self.camera_ids)

    def __iter__(self):
        pending = self.lifecycle._pending if self.lifecycle is not None else None
        if pending is not None:
            self.load_state_dict(pending["sidecar"]["camera_loader"])
        # Constructor consumes DataLoader's base seed. On resume _finish_resume rewinds
        # this incidental reconstruction draw after the iterator/model are reconstructed.
        self._active = _CameraIterator(self, iter(self.loader))
        if pending is not None:
            self.lifecycle._finish_resume(self)
        return self._active

    def state_dict(self):
        assert self._active is not None
        return dict(camera_ids=list(self.camera_ids), sampled_count=self.sampled_count,
                    last_camera_id=self.last_camera_id, sampler=self.sampler.state_dict())

    def load_state_dict(self, state):
        assert state["camera_ids"] == list(self.camera_ids), "Camera roster/order changed"
        self.sampled_count = int(state["sampled_count"])
        self.last_camera_id = state["last_camera_id"]
        self.sampler.load_state_dict(state["sampler"])


class _TorchProxy:
    def __init__(self, module, lifecycle):
        self.module, self.lifecycle = module, lifecycle

    def __getattr__(self, name):
        return getattr(self.module, name)

    def save(self, obj, path, *args, **kwargs):
        if (isinstance(obj, tuple) and len(obj) == 2 and type(obj[1]) is int
                and obj[1] in self.lifecycle.stop_iterations):
            return self.lifecycle.save_after_update(obj, path, *args, **kwargs)
        return self.module.save(obj, path, *args, **kwargs)

    def load(self, path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and re.fullmatch(r"chkpnt\d+\.pth", Path(path).name):
            return self.lifecycle.load_before_model_restore(path, *args, **kwargs)
        return self.module.load(path, *args, **kwargs)


class StagedTrainingLifecycle:
    def __init__(self, *, model, paired_background, camera_ids, identity, source_hashes,
                 total_iterations, stop_iterations, optimization=None,
                 counters_getter=None, counters_setter=None, compute_seconds_getter=None,
                 extra_parameters=None, expected_group_names=GROUPS):
        self.model, self.paired = model, paired_background
        self.camera_ids = tuple(camera_ids)
        assert self.camera_ids and len(set(self.camera_ids)) == len(self.camera_ids)
        self.total_iterations = int(total_iterations)
        self.stop_iterations = frozenset(int(s) for s in stop_iterations)
        assert all(0 < s < self.total_iterations for s in self.stop_iterations), "Review stop must be before original terminal step"
        self.identity = json.loads(_canonical(identity))
        self.source_hashes = {str(Path(p).resolve()): digest.lower() for p, digest in source_hashes.items()}
        assert self.source_hashes, "Bind at least the frozen input/code manifest"
        self.expected_group_names = tuple(expected_group_names)
        self.extra_parameters = dict(extra_parameters) if extra_parameters is not None else {"static_offset": model.static_offset}
        assert all(isinstance(p, torch.nn.Parameter) for p in self.extra_parameters.values())
        self.counters_getter = counters_getter or (lambda: {})
        self.counters_setter = counters_setter
        # The getter reports this process/phase only; the stored prior is added here.
        self.compute_seconds_getter = compute_seconds_getter or (lambda: time.monotonic() - self._started)
        self._started = time.monotonic()
        self.prior_compute_seconds = 0.0
        self._pending = None
        self.loader = None
        self.last_resume_receipt = None
        self.optimization = None
        if optimization is None:
            optimization = identity.get("options", identity.get("optimization"))
        if optimization is not None:
            self.bind_optimization(optimization)

    def bind_optimization(self, optimization):
        self.optimization = optimization
        self._optimization_identity = json.loads(_canonical(dict(optimization) if isinstance(optimization, Mapping) else vars(optimization)))
        assert self._optimization_identity["iterations"] == self.total_iterations
        return self

    def _contract(self):
        assert self.optimization is not None, "Pass optimization=opt or call bind_optimization(opt)"
        current = dict(self.optimization) if isinstance(self.optimization, Mapping) else vars(self.optimization)
        assert json.loads(_canonical(current)) == self._optimization_identity, "Optimization/18k clock changed"
        return dict(identity=self.identity, source_hashes=self.source_hashes,
                    camera_ids=list(self.camera_ids), total_iterations=self.total_iterations,
                    optimization=self._optimization_identity,
                    expected_group_names=list(self.expected_group_names),
                    torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                    cuda_version=torch.version.cuda,
                    backend=dict(threads=torch.get_num_threads(),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        cudnn_benchmark=torch.backends.cudnn.benchmark))

    def _verify_sources(self):
        for path, expected in self.source_hashes.items():
            assert re.fullmatch(r"[0-9a-f]{64}", expected)
            assert sha256_file(path) == expected, "Frozen source changed: " + path

    def torch_proxy(self, module=torch):
        assert module is torch, "Use the same torch module as the model"
        return _TorchProxy(module, self)

    def camera_loader(self, dataset, **kwargs):
        assert self.loader is None, "Only the one original training camera loader is supported"
        self.loader = CheckpointableCameraLoader(dataset, camera_ids=self.camera_ids, lifecycle=self, **kwargs)
        return self.loader

    def _validate_model(self, capture=None, require_zero_grad=True):
        cap = self.model.capture()
        assert isinstance(cap, tuple) and len(cap) == 14
        groups = self.model.optimizer.param_groups
        assert [g["name"] for g in groups] == list(self.expected_group_names)
        assert all(len(g["params"]) == 1 for g in groups)
        by_name = {g["name"]: g["params"][0] for g in groups}
        for name, index in zip(GROUPS[:6], PARAM_CAPTURE):
            assert by_name[name] is cap[index], "Optimizer is bound to wrong Gaussian Parameter: " + name
        for name, parameter in self.extra_parameters.items():
            assert getattr(self.model, name) is parameter and by_name[name] is parameter
        if require_zero_grad:
            assert all(p.grad is None for g in groups for p in g["params"]), "Not a post-Adam/zero_grad boundary"
        if capture is not None:
            assert _equal(cap, capture), "Restored capture/Adam state differs from saved native tuple"
        return cap

    @staticmethod
    def _paths(checkpoint):
        checkpoint = Path(checkpoint)
        return checkpoint, Path(str(checkpoint) + ".lifecycle.pt"), Path(str(checkpoint) + ".lifecycle.json")

    def save_after_update(self, native_tuple, checkpoint, *args, **kwargs):
        cap, step = native_tuple
        assert type(step) is int and step in self.stop_iterations and step < self.total_iterations
        assert self._pending is None and self.loader is not None
        assert self.loader.sampled_count == step, "Sampling/global iteration mismatch"
        assert int(self.paired.calls) == step, "Paired-background/global iteration mismatch"
        assert re.fullmatch(r"chkpnt" + str(step) + r"\.pth", Path(checkpoint).name)
        self._verify_sources()
        self._validate_model(cap)
        checkpoint, sidecar_path, manifest_path = self._paths(checkpoint)
        assert checkpoint.parent.is_dir()
        assert not any(p.exists() for p in (checkpoint, sidecar_path, manifest_path)), "Do not overwrite earlier evidence"
        contract = self._contract()
        counters = json.loads(_canonical(self.counters_getter()))
        seconds = self.prior_compute_seconds + float(self.compute_seconds_getter())
        assert np.isfinite(seconds) and seconds >= self.prior_compute_seconds
        sidecar = dict(schema=SCHEMA, contract=contract, contract_sha256=_identity_hash(contract),
                       completed_iteration=step, checkpoint_phase="after_density_adam_zero_grad",
                       extra_parameters={k: p.detach().cpu().clone() for k, p in self.extra_parameters.items()},
                       model_metadata={k: int(getattr(self.model, k)) for k in ("active_sh_degree", "max_sh_degree", "timestep", "num_timesteps")},
                       camera_loader=self.loader.state_dict(),
                       paired=dict(generator_state=self.paired.generator.get_state().cpu().clone(),
                                   generator_device=str(self.paired.generator.device), calls=int(self.paired.calls),
                                   first_colors=list(self.paired.first_colors)),
                       counters=counters, cumulative_compute_seconds=seconds,
                       compute_seconds_scope="sum_of_lifecycle_process_intervals_including_resume_validation_excluding_offline_review_and_work_before_life_construction",
                       rng=capture_rng_state(), quality_accepted=False)
        # Manifest is the commit marker; a crash before it leaves an unresumable artifact.
        temp_native = Path(str(checkpoint) + ".tmp")
        temp_sidecar = Path(str(sidecar_path) + ".tmp")
        temp_manifest = Path(str(manifest_path) + ".tmp")
        assert not any(p.exists() for p in (temp_native, temp_sidecar, temp_manifest))
        torch.save(native_tuple, temp_native, *args, **kwargs)
        sidecar["native_checkpoint_sha256"] = sha256_file(temp_native)
        torch.save(sidecar, temp_sidecar)
        manifest = dict(schema=SCHEMA, completed_iteration=step, total_iterations=self.total_iterations,
                        native_checkpoint_sha256=sidecar["native_checkpoint_sha256"],
                        sidecar_sha256=sha256_file(temp_sidecar), contract_sha256=sidecar["contract_sha256"],
                        checkpoint_phase=sidecar["checkpoint_phase"], state="checkpoint_ready_for_offline_review",
                        cumulative_compute_seconds=seconds, quality_accepted=False)
        temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_native, checkpoint)
        os.replace(temp_sidecar, sidecar_path)
        os.replace(temp_manifest, manifest_path)
        raise ReviewStop(checkpoint, manifest_path)

    def load_before_model_restore(self, checkpoint, *args, **kwargs):
        assert self._pending is None and self.loader is None
        checkpoint, sidecar_path, manifest_path = self._paths(checkpoint)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == SCHEMA
        assert sha256_file(checkpoint) == manifest["native_checkpoint_sha256"]
        assert sha256_file(sidecar_path) == manifest["sidecar_sha256"]
        self._verify_sources()
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        assert sidecar["schema"] == SCHEMA
        assert sidecar["contract"] == self._contract()
        assert sidecar["contract_sha256"] == _identity_hash(self._contract()) == manifest["contract_sha256"]
        assert sidecar["native_checkpoint_sha256"] == manifest["native_checkpoint_sha256"]
        assert sidecar["checkpoint_phase"] == manifest["checkpoint_phase"] == "after_density_adam_zero_grad"
        kwargs.setdefault("weights_only", False)  # Only locally generated, SHA-bound artifacts.
        native = torch.load(checkpoint, *args, **kwargs)
        assert type(native) is tuple and len(native) == 2 and len(native[0]) == 14
        assert native[1] == manifest["completed_iteration"] == sidecar["completed_iteration"]
        assert 0 < native[1] < self.total_iterations
        assert set(sidecar["extra_parameters"]) == set(self.extra_parameters)
        with torch.no_grad():
            for name, parameter in self.extra_parameters.items():
                saved = sidecar["extra_parameters"][name]
                assert saved.dtype == parameter.dtype and saved.shape == parameter.shape
                parameter.copy_(saved.to(parameter.device))
        self._pending = dict(sidecar=sidecar, native=native,
                             expected_capture=_snapshot_tree(native[0]),
                             checkpoint=str(checkpoint), manifest=manifest)
        return native

    def _finish_resume(self, loader):
        pending = self._pending
        assert pending is not None
        sidecar, native = pending["sidecar"], pending["native"]
        self._validate_model(pending["expected_capture"])
        for name, parameter in self.extra_parameters.items():
            assert _equal(parameter, sidecar["extra_parameters"][name])
        meta = sidecar["model_metadata"]
        for name in ("active_sh_degree", "max_sh_degree", "num_timesteps"):
            assert int(getattr(self.model, name)) == meta[name], "Model identity changed: " + name
        with torch.no_grad():
            self.model.select_mesh_by_timestep(meta["timestep"])
        assert str(self.paired.generator.device) == sidecar["paired"]["generator_device"]
        self.paired.generator.set_state(sidecar["paired"]["generator_state"])
        self.paired.calls = sidecar["paired"]["calls"]
        self.paired.first_colors = list(sidecar["paired"]["first_colors"])
        if sidecar["counters"]:
            assert self.counters_setter is not None, "Cannot discard saved adapter counters"
            self.counters_setter(json.loads(_canonical(sidecar["counters"])))
            assert _equal(json.loads(_canonical(self.counters_getter())), sidecar["counters"])
        self.prior_compute_seconds = float(sidecar["cumulative_compute_seconds"])
        # Rewind the incidental DataLoader base-seed draw without rebuilding the
        # iterator again. Explicit generator is independent of the global CPU RNG.
        saved_sampler = sidecar["camera_loader"]["sampler"]
        if loader.sampler.generator is not None:
            loader.sampler.generator.set_state(saved_sampler["generator_state"])
        restore_rng_state(sidecar["rng"])
        assert _equal(loader.state_dict(), sidecar["camera_loader"])
        assert _equal(capture_rng_state(), sidecar["rng"])
        self.last_resume_receipt = dict(schema=SCHEMA, completed_iteration=native[1],
            next_iteration=native[1] + 1, checkpoint=pending["checkpoint"],
            native_checkpoint_sha256=pending["manifest"]["native_checkpoint_sha256"],
            model_Adam_offset_verified=True, camera_stream_restored=True,
            all_rng_restored=True, paired_background_restored=True,
            cumulative_compute_seconds=self.prior_compute_seconds,
            native_GPU_continuation_control_still_required=True, quality_accepted=False)
        self._pending = None
