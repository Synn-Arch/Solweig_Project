# SPDX-License-Identifier: GPL-3.0-only
"""G2.0: temporal checkpoint format, write policy, bitwise roundtrip.

Design: ``docs/incremental_design_tool/realtime_collaboration/design/
r5-temporal-checkpoints-and-fast-serve.md`` section G2.0 (binding).

Checkpoints are REPRODUCE aids captured at revision boundaries: they bound
how far a later replay (G2.1) must rewind on the time axis. The op log and
the temporal result store stay authoritative — a checkpoint never leads
them. The hard gate here is the r4a/r4b discipline: a checkpoint that does
not roundtrip bitwise is a defect, and a torn/corrupt checkpoint is
rejected at load, never served.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_gpu.incremental.checkpoints import (
    DEFAULT_KEEP,
    FINGERPRINT_KEYS,
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointRecord,
    coverage_checkpoint,
    coverage_manifest_from_store,
    fingerprint_mismatch_reason,
    list_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    prune_checkpoints,
    select_replay_checkpoint,
    write_checkpoint,
)
from solweig_gpu.incremental.edit_types import EditCommand
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.store import (
    TemporalResultEntry,
    TemporalResultStore,
)

from tests.test_incremental_executor import (
    _executor,
    _stub_local,
    _validated,
    _veg_command,
)
from tests.test_incremental_worker import (
    TINY_ORIGIN,
    TINY_PIXEL,
    _make_tiny_site,
)

# The canonical cross-timestep thermal state the solver threads k -> k+1
# (R5b supplement, verified against the code): the six full-tile planes
# allocated at utci_process.py:614-619 and mutated only by
# TsWaveDelay_2015a (solweig.py:451-488) plus the scalars the loop
# carries — CI/firstdaytime/timeadd and Twater, the midnight-recomputed
# water temperature (utci_process.py:728, 790-805). timestepdec is
# DERIVED per step, never carried. next_step: the state is after
# completing timesteps 0..next_step-1; replay resumes AT next_step.
THERMAL_TENSOR_NAMES = (
    "Tgmap1",
    "Tgmap1E",
    "Tgmap1S",
    "Tgmap1W",
    "Tgmap1N",
    "TgOut1",
)
THERMAL_SCALARS = {
    "CI": 0.87,
    "firstdaytime": 1.0,
    "timeadd": 0.0,
    "Twater": 288.15,
}
# The series container stays part of the format (generality); the
# canonical payload uses no series today.
THERMAL_SERIES = {"Twater_daily": (288.15, 289.2, 290.01)}

# Input digests that DETERMINE the thermal state (R5b supplement): a
# checkpoint is applicable to a replay iff these match — revisions can
# skip under mixed direct-edit+realtime, so (workspace, revision) is a
# weak key.
INPUT_FINGERPRINT = {
    "composed_scene": "a" * 64,
    "resolved_landcover": "b" * 64,
    "model_parameters": "c" * 64,
    "met_prefix": "d" * 64,
}


def _thermal_state(seed: int = 0) -> dict[str, np.ndarray]:
    """Representative float32 maps, deliberately carrying NaN and -0.0.

    The ground-temperature state is a weighted accumulation
    (TsWaveDelay_2015a); nothing in the format may assume finiteness, so
    the roundtrip proof must be byte-exact for NaN payloads and signed
    zeros, not ``==``-equal.
    """
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for index, name in enumerate(THERMAL_TENSOR_NAMES):
        plane = rng.standard_normal((5, 4)).astype(np.float32) * (10.0 + index)
        plane[0, 0] = np.float32("nan")
        plane[1, 1] = -0.0
        state[name] = plane
    return state


_FINGERPRINT_DEFAULT = object()


def _record(
    scene_revision: int = 3,
    *,
    next_step: int | None = 8,
    thermal: dict[str, np.ndarray] | None = None,
    fingerprint: dict[str, str] | None | object = _FINGERPRINT_DEFAULT,
) -> CheckpointRecord:
    store = TemporalResultStore()
    store.publish(
        [
            TemporalResultEntry(
                node_id="utci",
                time_index=t,
                scene_revision=scene_revision,
                job_id=f"job-{scene_revision}",
                mode="local",
                write_window=RasterWindow(0, 4, 0, 4),
                patch_path=Path(f"/scenario/rev-{scene_revision:06d}/patch"),
            )
            for t in range(3)
        ]
    )
    if fingerprint is _FINGERPRINT_DEFAULT:
        fingerprint = dict(INPUT_FINGERPRINT)
    return CheckpointRecord(
        scene_revision=scene_revision,
        coverage_manifest=coverage_manifest_from_store(store),
        thermal_tensors=(
            dict(_thermal_state()) if thermal is None else dict(thermal)
        ),
        thermal_scalars=dict(THERMAL_SCALARS),
        thermal_series=dict(THERMAL_SERIES),
        next_step=next_step,
        input_fingerprint=fingerprint,  # type: ignore[arg-type]
    )


@pytest.fixture()
def executor(tmp_path: Path):
    grid, site = _make_tiny_site(tmp_path / "site")
    ex = _executor(tmp_path, grid=grid, site=site)
    return ex


class TestBitwiseRoundtrip:
    def test_thermal_tensors_roundtrip_bitwise(self, tmp_path: Path) -> None:
        record = _record()
        revision_dir = write_checkpoint(record, tmp_path / "checkpoints")

        loaded = load_checkpoint(revision_dir)
        assert loaded.scene_revision == record.scene_revision
        assert loaded.next_step == record.next_step == 8
        assert sorted(loaded.thermal_tensors) == sorted(THERMAL_TENSOR_NAMES)
        for name in THERMAL_TENSOR_NAMES:
            original = record.thermal_tensors[name]
            restored = loaded.thermal_tensors[name]
            assert restored.dtype == np.float32
            assert restored.shape == original.shape
            # Byte equality, not ==: NaN and -0.0 must survive untouched.
            assert restored.tobytes() == original.tobytes()

    def test_torch_tensors_bridge_bitwise(self, tmp_path: Path) -> None:
        torch_state = {
            name: torch.from_numpy(plane.copy())
            for name, plane in _thermal_state(seed=3).items()
        }
        record = _record(
            thermal={"Tgmap1": torch_state["Tgmap1"], "TgOut1": torch_state["TgOut1"]}
        )
        revision_dir = write_checkpoint(record, tmp_path / "checkpoints")
        loaded = load_checkpoint(revision_dir)
        for name in ("Tgmap1", "TgOut1"):
            restored = torch.from_numpy(loaded.thermal_tensors[name])
            assert restored.dtype == torch.float32
            # Bit-for-bit torch equality including NaN bit patterns.
            assert torch.equal(
                restored.view(torch.int32), torch_state[name].view(torch.int32)
            )

    def test_scalars_and_series_roundtrip_exactly(self, tmp_path: Path) -> None:
        record = _record()
        revision_dir = write_checkpoint(record, tmp_path / "checkpoints")
        loaded = load_checkpoint(revision_dir)
        assert loaded.thermal_scalars == THERMAL_SCALARS
        assert loaded.thermal_series == {
            name: tuple(values) for name, values in THERMAL_SERIES.items()
        }

    def test_input_fingerprint_roundtrips_and_is_digest_covered(
        self, tmp_path: Path
    ) -> None:
        revision_dir = write_checkpoint(_record(), tmp_path / "checkpoints")
        loaded = load_checkpoint(revision_dir)
        assert loaded.input_fingerprint == INPUT_FINGERPRINT

        # The fingerprint lives inside the payload digest: tampering with
        # it must reject the checkpoint at load, never serve a guess.
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["payload"]["input_fingerprint"]["composed_scene"] = "f" * 64
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(CheckpointError, match="payload"):
            load_checkpoint(revision_dir)

    def test_coverage_manifest_roundtrips(self, tmp_path: Path) -> None:
        record = _record(scene_revision=2)
        revision_dir = write_checkpoint(record, tmp_path / "checkpoints")
        loaded = load_checkpoint(revision_dir)
        assert len(loaded.coverage_manifest) == 3
        for original, restored in zip(
            record.coverage_manifest, loaded.coverage_manifest
        ):
            assert restored.node_id == original.node_id
            assert restored.time_index == original.time_index
            assert restored.scene_revision == original.scene_revision
            assert restored.job_id == original.job_id
            assert restored.mode == original.mode
            assert restored.write_window == original.write_window
            assert restored.patch_path == original.patch_path

    def test_manifest_references_patches_without_copying_payloads(
        self, tmp_path: Path
    ) -> None:
        record = _record()
        revision_dir = write_checkpoint(record, tmp_path / "checkpoints")
        # The checkpoint directory holds metadata + thermal tensors only:
        # every patch payload stays where the result store put it.
        npy_files = sorted(p.name for p in (revision_dir / "tensors").glob("*.npy"))
        assert npy_files == [f"{name}.npy" for name in sorted(THERMAL_TENSOR_NAMES)]
        assert record.coverage_manifest[0].patch_path == Path(
            "/scenario/rev-000003/patch"
        )
        assert not (revision_dir / "patch").exists()

    def test_roundtrip_across_fresh_load_latest(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        write_checkpoint(_record(scene_revision=1), root)
        loaded = load_latest_checkpoint(root)
        assert loaded is not None
        assert loaded.scene_revision == 1
        assert (
            loaded.thermal_tensors["Tgmap1"].tobytes()
            == _thermal_state()["Tgmap1"].tobytes()
        )

    def test_coverage_checkpoint_from_store_snapshots_every_window(
        self,
    ) -> None:
        store = TemporalResultStore()
        store.publish(
            [
                TemporalResultEntry(
                    node_id="tmrt",
                    time_index=0,
                    scene_revision=4,
                    job_id="job-a",
                    mode="local",
                    write_window=RasterWindow(0, 4, 0, 4),
                    patch_path=Path("/patches/w0"),
                ),
                TemporalResultEntry(
                    node_id="tmrt",
                    time_index=0,
                    scene_revision=4,
                    job_id="job-a",
                    mode="local",
                    write_window=RasterWindow(4, 8, 4, 8),
                    patch_path=Path("/patches/w1"),
                ),
            ]
        )
        record = coverage_checkpoint(scene_revision=4, store=store)
        assert record.scene_revision == 4
        assert record.thermal_tensors == {}
        assert record.next_step is None
        assert record.input_fingerprint is None
        assert [
            (entry.write_window, entry.patch_path)
            for entry in record.coverage_manifest
        ] == [
            (RasterWindow(0, 4, 0, 4), Path("/patches/w0")),
            (RasterWindow(4, 8, 4, 8), Path("/patches/w1")),
        ]


class TestTornCheckpointRejection:
    def _write(self, tmp_path: Path) -> Path:
        return write_checkpoint(_record(), tmp_path / "checkpoints")

    def test_truncated_tensor_bytes_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        tensor = revision_dir / "tensors" / "Tgmap1.npy"
        raw = tensor.read_bytes()
        tensor.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(CheckpointError, match="Tgmap1"):
            load_checkpoint(revision_dir)

    def test_flipped_tensor_checksum_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        digest = meta["payload"]["thermal_tensor_checksums"]["Tgmap1"]
        meta["payload"]["thermal_tensor_checksums"]["Tgmap1"] = (
            "0" if digest[0] != "0" else "1"
        ) + digest[1:]
        meta_path.write_text(json.dumps(meta))
        # The checksum map is INSIDE the payload digest (G2.1 hardening):
        # tampering it is caught at the digest layer, before any tensor
        # bytes are even read. (Flipping the .npy BYTES instead — see
        # test_bit_flipped_tensor_bytes_rejected — still lands on the
        # per-tensor checksum compare.)
        with pytest.raises(CheckpointError, match="payload"):
            load_checkpoint(revision_dir)

    def test_np_load_eoferror_rejected_typed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # np.load can surface a truncated file as EOFError (numpy-version
        # dependent); the load fence must convert EVERY unreadable-tensor
        # failure to the typed refusal, never leak it.
        from solweig_gpu.incremental import checkpoints as checkpoints_module

        revision_dir = self._write(tmp_path)

        def truncated_np_load(*args, **kwargs):
            raise EOFError("Truncated file read")

        monkeypatch.setattr(checkpoints_module.np, "load", truncated_np_load)
        with pytest.raises(CheckpointError, match="unreadable thermal tensor"):
            load_checkpoint(revision_dir)

    def test_bit_flipped_tensor_bytes_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        tensor = revision_dir / "tensors" / "TgOut1.npy"
        raw = bytearray(tensor.read_bytes())
        raw[-1] ^= 0x01  # flip the low bit of the last float's mantissa
        tensor.write_bytes(bytes(raw))
        with pytest.raises(CheckpointError, match="checksum"):
            load_checkpoint(revision_dir)

    def test_flipped_payload_digest_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["payload"]["coverage_manifest"][0]["job_id"] = "tampered"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(CheckpointError, match="payload"):
            load_checkpoint(revision_dir)

    def test_corrupt_json_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        (revision_dir / "checkpoint.json").write_text("{not json")
        with pytest.raises(CheckpointError):
            load_checkpoint(revision_dir)

    def test_wrong_schema_version_rejected(self, tmp_path: Path) -> None:
        revision_dir = self._write(tmp_path)
        meta_path = revision_dir / "checkpoint.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = SCHEMA_VERSION + 1
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(CheckpointError, match="schema"):
            load_checkpoint(revision_dir)

    def test_load_latest_falls_back_to_previous_valid(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        write_checkpoint(_record(scene_revision=1), root)
        write_checkpoint(_record(scene_revision=2), root)
        # Tear the newest revision the way a kill mid-persist would leave
        # it: manifest present, tensor bytes truncated.
        tensor = root / "rev-000002" / "tensors" / "Tgmap1.npy"
        raw = tensor.read_bytes()
        tensor.write_bytes(raw[: len(raw) // 3])
        loaded = load_latest_checkpoint(root)
        assert loaded is not None
        assert loaded.scene_revision == 1

    def test_load_latest_none_when_nothing_valid(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        revision_dir = write_checkpoint(_record(), root)
        (revision_dir / "checkpoint.json").unlink()
        assert load_latest_checkpoint(root) is None
        assert load_latest_checkpoint(tmp_path / "missing") is None


class TestRotationPolicy:
    def test_keep_last_k_plus_revision0_base(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        for revision in range(6):
            write_checkpoint(_record(scene_revision=revision), root)
        assert list_checkpoints(root) == (0, 3, 4, 5)
        assert (root / "rev-000000" / "checkpoint.json").exists()

    def test_base_inside_window_keeps_exactly_k(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        for revision in range(3):  # rev 0 is within the newest three
            write_checkpoint(_record(scene_revision=revision), root)
        assert list_checkpoints(root) == (0, 1, 2)

    def test_explicit_keep_parameter(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        for revision in range(5):
            write_checkpoint(_record(scene_revision=revision), root, keep=1)
        assert list_checkpoints(root) == (0, 4)

    def test_prune_returns_deleted_revisions(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        for revision in range(5):
            write_checkpoint(_record(scene_revision=revision), root, keep=5)
        deleted = prune_checkpoints(root, keep=2)
        assert [p.name for p in deleted] == ["rev-000001", "rev-000002"]
        assert list_checkpoints(root) == (0, 3, 4)

    def test_default_keep_is_three(self) -> None:
        assert DEFAULT_KEEP == 3


class TestAtomicWrite:
    def test_failed_rename_records_no_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solweig_gpu.incremental import checkpoints as checkpoints_module

        root = tmp_path / "checkpoints"

        def broken_rename(src, dst):
            raise OSError("simulated kill between stage and rename")

        monkeypatch.setattr(checkpoints_module.os, "rename", broken_rename)
        with pytest.raises(OSError, match="simulated kill"):
            write_checkpoint(_record(), root)
        monkeypatch.undo()

        assert list_checkpoints(root) == ()
        assert load_latest_checkpoint(root) is None
        # No final directory ever became visible; staging leftovers (if
        # any) are invisible to the revision listing and to loads.
        assert not (root / "rev-000003" / "checkpoint.json").exists()

    def test_failed_rewrite_keeps_previous_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from solweig_gpu.incremental import checkpoints as checkpoints_module

        root = tmp_path / "checkpoints"
        write_checkpoint(_record(scene_revision=1), root)

        def broken_rename(src, dst):
            raise OSError("simulated kill before replace")

        monkeypatch.setattr(checkpoints_module.os, "rename", broken_rename)
        with pytest.raises(OSError):
            write_checkpoint(_record(scene_revision=2), root)
        monkeypatch.undo()

        loaded = load_latest_checkpoint(root)
        assert loaded is not None
        assert loaded.scene_revision == 1


class TestFloat32Fence:
    def test_float64_tensor_refused_loudly(self, tmp_path: Path) -> None:
        narrowed = {
            name: plane.astype(np.float64)
            for name, plane in _thermal_state().items()
        }
        with pytest.raises(CheckpointError, match="float32"):
            write_checkpoint(_record(thermal=narrowed), tmp_path / "checkpoints")
        assert list_checkpoints(tmp_path / "checkpoints") == ()

    def test_int_tensor_refused_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="float32"):
            write_checkpoint(
                _record(thermal={"Tgmap1": np.zeros((4, 4), dtype=np.int32)}),
                tmp_path / "checkpoints",
            )

    def test_mixed_shape_planes_refused_loudly(self, tmp_path: Path) -> None:
        # The thermal chain is elementwise and full-tile: every plane must
        # describe the SAME grid, or a windowed consumer would slice a
        # patchwork. Refuse the write, never persist the mismatch.
        ragged = dict(_thermal_state())
        ragged["TgOut1"] = np.zeros((6, 4), dtype=np.float32)
        with pytest.raises(CheckpointError, match="shape"):
            write_checkpoint(_record(thermal=ragged), tmp_path / "checkpoints")
        assert list_checkpoints(tmp_path / "checkpoints") == ()


class TestInputFingerprint:
    """R5b supplement: applicability is keyed by INPUT DIGESTS, not
    (workspace, revision) — revisions can skip under mixed direct-edit +
    realtime, and a stale-amplitude scene silently changes the science
    (solver.py:1580-1608). A mismatch is a typed refusal + cold replay +
    a fallback reason for telemetry: never an exception to the serve
    path, never a guess."""

    def test_fingerprint_keys_are_the_four_input_digests(self) -> None:
        assert FINGERPRINT_KEYS == (
            "composed_scene",
            "resolved_landcover",
            "model_parameters",
            "met_prefix",
        )

    def test_match_returns_none_reason(self) -> None:
        record = _record()
        assert fingerprint_mismatch_reason(record, INPUT_FINGERPRINT) is None

    def test_missing_fingerprint_is_a_reason_never_a_guess(self) -> None:
        record = _record(fingerprint=None)
        reason = fingerprint_mismatch_reason(record, INPUT_FINGERPRINT)
        assert reason is not None
        assert "no input fingerprint" in reason

    def test_differing_digest_is_a_reason(self) -> None:
        expected = dict(INPUT_FINGERPRINT)
        expected["composed_scene"] = "e" * 64
        reason = fingerprint_mismatch_reason(_record(), expected)
        assert reason is not None
        assert "composed_scene" in reason

    def test_select_newest_matching_checkpoint(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        write_checkpoint(_record(scene_revision=1), root)
        write_checkpoint(_record(scene_revision=2), root)
        checkpoint, reason = select_replay_checkpoint(root, INPUT_FINGERPRINT)
        assert checkpoint is not None and reason is None
        assert checkpoint.scene_revision == 2
        assert checkpoint.next_step == 8

    def test_select_falls_back_to_older_match_then_reports_cold(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "checkpoints"
        older = dict(INPUT_FINGERPRINT)
        newer = dict(INPUT_FINGERPRINT)
        newer["met_prefix"] = "e" * 64  # met edit arrived: newest is stale
        write_checkpoint(_record(scene_revision=1, fingerprint=older), root)
        write_checkpoint(_record(scene_revision=2, fingerprint=newer), root)

        # The current inputs match only the older checkpoint: warm start
        # from it (fingerprint, not revision, decides).
        checkpoint, reason = select_replay_checkpoint(root, older)
        assert checkpoint is not None and reason is None
        assert checkpoint.scene_revision == 1

        # Nothing at all matches the newest inputs: cold replay + reason.
        checkpoint, reason = select_replay_checkpoint(
            root, {**INPUT_FINGERPRINT, "composed_scene": "9" * 64}
        )
        assert checkpoint is None
        assert reason is not None and "composed_scene" in reason

    def test_select_reports_unreadable_newest_as_cold(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        write_checkpoint(_record(scene_revision=1), root)
        revision_dir = write_checkpoint(_record(scene_revision=2), root)
        tensor = revision_dir / "tensors" / "Tgmap1.npy"
        tensor.write_bytes(tensor.read_bytes()[:-4])  # torn newest
        checkpoint, reason = select_replay_checkpoint(root, INPUT_FINGERPRINT)
        assert checkpoint is not None and reason is None
        assert checkpoint.scene_revision == 1

        # Newest torn AND older mismatched: cold replay, reason says so.
        write_checkpoint(
            _record(scene_revision=1, fingerprint=dict(INPUT_FINGERPRINT,
                                                       met_prefix="1" * 64)),
            root,
        )
        checkpoint, reason = select_replay_checkpoint(
            root, INPUT_FINGERPRINT
        )
        assert checkpoint is None
        assert reason is not None

    def test_select_reports_absence_as_cold(self, tmp_path: Path) -> None:
        checkpoint, reason = select_replay_checkpoint(
            tmp_path / "checkpoints", INPUT_FINGERPRINT
        )
        assert checkpoint is None
        assert reason is not None and "no checkpoint" in reason


class TestExecutorPublishHook:
    """The write-policy hook: a fully published batch offers a checkpoint.

    The executor site is ``_record_outcome`` after the temporal store
    publish — the single point where a batch is fully published at a
    revision. A failed checkpoint write must never fail the published
    batch (checkpoints are reproduce aids, never a source of truth).
    """

    def _checkpoint_root(self, executor) -> Path:
        return executor.results_root / executor._scenario_id / "checkpoints"

    def test_publish_writes_checkpoint_at_revision(self, executor) -> None:
        _stub_local(executor, {})
        executed = executor.execute([executor.validate(_veg_command())])
        assert executed.published
        assert executed.scene_revision == 1

        root = self._checkpoint_root(executor)
        assert list_checkpoints(root) == (1,)
        loaded = load_checkpoint(root / "rev-000001")
        assert loaded.scene_revision == 1
        assert loaded.thermal_tensors == {}
        assert loaded.input_fingerprint is None
        # The manifest carries the freshly published (node, time) coverage.
        manifest_keys = {
            (entry.node_id, entry.time_index)
            for entry in loaded.coverage_manifest
        }
        assert set(executed.store_keys) <= manifest_keys
        for entry in loaded.coverage_manifest:
            assert entry.patch_path is not None
            assert entry.patch_path.exists()

    def test_noop_batch_writes_no_checkpoint(self, executor) -> None:
        command = EditCommand(
            edit_id="view-1",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id="output_view",
            operation="select_layer",
            old_state={"layers": ("utci",)},
            new_state={"layers": ("tmrt",)},
            requested_outputs=("tmrt",),
            requested_times=None,
        )
        executed = executor.execute([_validated(command)])
        assert executed.status == "no-op"
        root = self._checkpoint_root(executor)
        assert list_checkpoints(root) == ()
        assert not root.exists()

    def test_consecutive_publishes_keep_rotation_window(self, executor) -> None:
        _stub_local(executor, {})
        executor.execute([executor.validate(_veg_command(edit_id="veg-1"))])
        tree_state = {
            "tree_id": "t1",
            "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
            "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
            "height_m": 4.0,
            "canopy_radius_m": 2.0,
        }
        executor.execute(
            [
                executor.validate(
                    _veg_command(
                        edit_id="veg-2",
                        revision=1,
                        operation="move",
                        old_state=tree_state,
                        new_state={
                            **tree_state,
                            "height_m": 6.0,
                        },
                    )
                )
            ]
        )
        root = self._checkpoint_root(executor)
        assert list_checkpoints(root) == (1, 2)

    def test_unexpected_checkpoint_fault_never_fails_publish(
        self, executor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Best-effort means best-effort: ANY fault class in the checkpoint
        # write (not just CheckpointError/OSError) is logged and swallowed
        # — the published batch stands, the fault is telemetry, never an
        # error surfaced to the edit.
        from solweig_gpu.incremental import executor as executor_module

        def exploding_write(record, root, **kwargs):
            raise RuntimeError("simulated checkpoint store fault")

        monkeypatch.setattr(executor_module, "write_checkpoint", exploding_write)
        _stub_local(executor, {})
        with caplog.at_level("WARNING", logger="solweig_gpu.incremental.executor"):
            executed = executor.execute([executor.validate(_veg_command())])
        assert executed.published
        assert executed.scene_revision == 1
        assert any(
            "checkpoint" in record.getMessage() and "simulated" in record.getMessage()
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# G2.1 item 3: capture at the full solve, commit at publication only
# ---------------------------------------------------------------------------


def _fingerprint_inputs(seed: int = 0, rows: int = 4, cols: int = 5):
    """Deterministic scene/forcing inputs for the digest-spelling tests."""
    rng = np.random.default_rng(seed)
    return {
        "building_dsm": rng.uniform(0.0, 12.0, (rows, cols)).astype(np.float32),
        "canopy": rng.uniform(0.0, 8.0, (rows, cols)).astype(np.float32),
        "dem": rng.uniform(0.0, 2.0, (rows, cols)).astype(np.float32),
        "resolved_landcover": rng.integers(1, 8, (rows, cols)).astype(np.float32),
        "model_parameters": {"albedo_b": 0.2},
        "met_prefix": rng.uniform(280.0, 310.0, (6, 25)),
    }


class TestThermalFingerprintDigests:
    """The applicability-key spelling lives in ONE place so the G2.1
    consumer recomputes the identical digests over its CURRENT inputs.

    The load-bearing property is the met-prefix boundary: a met edit at a
    row INSIDE the consumed prefix changes the digest (the carried state
    heard the old row — cold replay), while an edit BEYOND it does not
    (the state never read that row — warm start applies). That boundary is
    what makes ``warm(k..N, state@k)`` an honest fast path rather than a
    guess."""

    def _digest(self, **overrides) -> dict[str, str]:
        from solweig_gpu.incremental.checkpoints import thermal_fingerprint

        inputs = _fingerprint_inputs()
        inputs.update(overrides)
        return thermal_fingerprint(**inputs)

    def test_same_inputs_same_digest(self) -> None:
        assert self._digest() == self._digest()

    def test_every_key_is_a_sha256_hex_digest(self) -> None:
        digest = self._digest()
        assert set(digest) == set(FINGERPRINT_KEYS)
        for value in digest.values():
            assert len(value) == 64 and int(value, 16) >= 0

    def test_canopy_edit_changes_only_composed_scene(self) -> None:
        baseline = self._digest()
        edited = self._digest(canopy=_fingerprint_inputs(7)["canopy"])
        assert edited["composed_scene"] != baseline["composed_scene"]
        for key in ("resolved_landcover", "model_parameters", "met_prefix"):
            assert edited[key] == baseline[key]

    def test_landcover_edit_changes_only_resolved_landcover(self) -> None:
        baseline = self._digest()
        edited = self._digest(
            resolved_landcover=_fingerprint_inputs(9)["resolved_landcover"]
        )
        assert edited["resolved_landcover"] != baseline["resolved_landcover"]
        for key in ("composed_scene", "model_parameters", "met_prefix"):
            assert edited[key] == baseline[key]

    def test_params_edit_changes_only_model_parameters(self) -> None:
        baseline = self._digest()
        edited = self._digest(model_parameters={"albedo_b": 0.35})
        assert edited["model_parameters"] != baseline["model_parameters"]
        for key in ("composed_scene", "resolved_landcover", "met_prefix"):
            assert edited[key] == baseline[key]

    def test_none_and_empty_params_digest_identically(self) -> None:
        # Physics-identical spellings (run_full_tile treats falsy as the
        # legacy path): the applicability key must not distinguish them.
        assert self._digest(model_parameters=None) == self._digest(
            model_parameters={}
        )

    def test_met_edit_inside_prefix_changes_met_prefix(self) -> None:
        inputs = _fingerprint_inputs()
        from solweig_gpu.incremental.checkpoints import thermal_fingerprint

        baseline = thermal_fingerprint(**inputs)
        edited_table = inputs["met_prefix"].copy()
        edited_table[2, 11] += 1.0  # row 2 < next_step 6: INSIDE the prefix
        edited = thermal_fingerprint(**{**inputs, "met_prefix": edited_table})
        assert edited["met_prefix"] != baseline["met_prefix"]
        for key in ("composed_scene", "resolved_landcover", "model_parameters"):
            assert edited[key] == baseline[key]

    def test_met_row_beyond_prefix_does_not_change_met_prefix(self) -> None:
        # The warm-start property: rows the state never consumed cannot
        # invalidate it. A 3-row prefix over the same table's first rows
        # digests identically whether row 5 (outside) is edited or not.
        inputs = _fingerprint_inputs()
        from solweig_gpu.incremental.checkpoints import thermal_fingerprint

        short = {**inputs, "met_prefix": inputs["met_prefix"][:3]}
        edited_table = inputs["met_prefix"].copy()
        edited_table[5, 14] += 1.0
        short_edited = {
            **inputs,
            "met_prefix": edited_table[:3],
        }
        assert (
            thermal_fingerprint(**short)["met_prefix"]
            == thermal_fingerprint(**short_edited)["met_prefix"]
        )

    def test_absent_landcover_is_a_stable_spelling(self) -> None:
        assert self._digest(resolved_landcover=None) == self._digest(
            resolved_landcover=None
        )
        assert self._digest(resolved_landcover=None)[
            "resolved_landcover"
        ] != self._digest()["resolved_landcover"]


class TestFullSolveCapture:
    """G2.1 item 3: the worker captures the FULL-TILE final thermal state
    after ``_solve_full`` and the executor commits it at PUBLICATION only
    — superseded jobs persist nothing (a stale capture must never leak
    into a later revision's checkpoint), and local publications stay
    coverage-only (no full-tile state exists for them)."""

    def _checkpoint_root(self, executor) -> Path:
        return executor.results_root / executor._scenario_id / "checkpoints"

    def test_full_publication_commits_thermal_checkpoint(
        self, executor, tmp_path: Path
    ) -> None:
        from solweig_gpu.incremental.solver import (
            compose_full_scene_tensors,
            run_full_tile,
        )
        from tests.test_incremental_executor import _met_edit

        # A meteorology edit is site-global: the dirty set is the full grid
        # and the fast path refuses on a fresh store (no published tmrt),
        # so the job routes through the REAL full solve.
        executed = executor.execute([_met_edit(executor)])
        assert executed.published
        assert executed.mode == "full"

        root = self._checkpoint_root(executor)
        assert list_checkpoints(root) == (1,)
        loaded = load_checkpoint(root / "rev-000001")
        assert loaded.scene_revision == 1
        assert sorted(loaded.thermal_tensors) == sorted(THERMAL_TENSOR_NAMES)
        assert loaded.next_step == executor.cache.time_steps
        assert loaded.input_fingerprint is not None
        assert set(loaded.input_fingerprint) == set(FINGERPRINT_KEYS)
        # Twater is established at row 0 under land cover on this site.
        assert "Twater" in loaded.thermal_scalars
        for name in ("CI", "firstdaytime", "timeadd"):
            assert name in loaded.thermal_scalars
        for name, plane in loaded.thermal_tensors.items():
            assert plane.dtype == np.float32
            assert plane.shape == (executor.cache.rows, executor.cache.cols)
        # The publication hook attached the coverage manifest.
        assert loaded.coverage_manifest

        # The captured planes are the physics state, bitwise: a fresh
        # deterministic rerun of the same inputs must reproduce them.
        worker = executor._worker
        _arrays, state = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=worker.forcing(),
            site_dir=worker.site_dir,
            scratch_dir=tmp_path / "scratch_verify",
            landcover_overlay=worker._landcover_overlay,
            model_parameters=worker._model_parameters,
            return_final_state=True,
        )
        for name in THERMAL_TENSOR_NAMES:
            assert (
                loaded.thermal_tensors[name].tobytes()
                == state[name].detach().cpu().numpy().tobytes()
            ), name

        # The fingerprint spelling is stable end-to-end: recomputing the
        # digests over the same current inputs reproduces the stored ones.
        from solweig_gpu.incremental.checkpoints import thermal_fingerprint

        scene = compose_full_scene_tensors(worker.cache, worker.layer)
        resolved_lc = (
            np.asarray(worker.cache.landcover) if "landcover" in worker.cache else None
        )
        assert loaded.input_fingerprint == thermal_fingerprint(
            building_dsm=scene.a.numpy(),
            canopy=scene.canopy.numpy(),
            dem=scene.dem.numpy(),
            resolved_landcover=resolved_lc,
            model_parameters=worker._model_parameters,
            met_prefix=worker.forcing().met_table[: loaded.next_step],
        )

    def test_superseded_full_job_persists_nothing_and_capture_does_not_leak(
        self, executor
    ) -> None:
        from tests.test_incremental_executor import _met_edit

        calls = {"n": 0}

        def provider():
            calls["n"] += 1
            # Pass the windowing (ck1) and pre-solve (ck2) fences so the
            # full solve runs and CAPTURES, then advance the scene at the
            # pre-publish fence (ck3): the job is superseded AFTER the
            # capture exists — the leak scenario under test.
            return 1 if calls["n"] < 3 else 2

        original_provider = executor._worker._revision_provider
        executor._worker._revision_provider = provider
        executed = executor.execute([_met_edit(executor)])
        assert executed.status == "superseded"
        root = self._checkpoint_root(executor)
        assert list_checkpoints(root) == ()
        assert not root.exists()

        # The superseded job's capture must not leak into the NEXT
        # publication: a local-mode batch after it publishes coverage-only
        # (the capture is cleared at the start of every run).
        executor._worker._revision_provider = original_provider
        _stub_local(executor, {})
        executed = executor.execute([executor.validate(_veg_command())])
        assert executed.published
        assert executed.mode == "local"
        assert list_checkpoints(root) == (1,)
        loaded = load_checkpoint(root / "rev-000001")
        assert loaded.thermal_tensors == {}
        assert loaded.input_fingerprint is None
        assert loaded.next_step is None
