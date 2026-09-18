# SPDX-License-Identifier: GPL-3.0-only
"""Site-identity pinning tests (U-D1, intake item g).

The old grid guard checked rows/cols/pixel_size only: a site cache that
is pixel-identical but GEOGRAPHICALLY different (shifted origin, same
dimensions) would ride a live scenario undetected. The pin records
site id + rows + cols + pixel + origin x/y + time_steps at scenario
creation and every later mutation (edit, reset) verifies the live
registry geometry against it — a mismatch is a typed 409
``site_identity_mismatch`` carrying the pinned and current identities.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_api import (  # noqa: E402
    ORIGIN,
    PIXEL,
    ROWS,
    COLS,
    SITE_ID,
    TIME_STEPS,
    FakeSolver,
    make_app,
    make_site_cache,
    make_tree,
    wait_for_job,
)

from solweig_gpu.server.store import SiteIdentityMismatch, Store


def _identity(
    *,
    origin: tuple[float, float] = ORIGIN,
    rows: int = ROWS,
    cols: int = COLS,
    pixel: float = PIXEL,
    site_id: str = SITE_ID,
    time_steps: int = TIME_STEPS,
) -> dict:
    return {
        "site_id": site_id,
        "rows": rows,
        "cols": cols,
        "pixel_size_m": pixel,
        "origin_x_m": origin[0],
        "origin_y_m": origin[1],
        "time_steps": time_steps,
    }


# ---------------------------------------------------------------------------
# Store-level pin semantics
# ---------------------------------------------------------------------------


def test_store_pins_and_detects_origin_shift(tmp_path: Path) -> None:
    store = Store(tmp_path / "state" / "app.db", results_root=tmp_path / "results")
    record, _ = store.create_scenario(
        site_id=SITE_ID,
        name="pinned",
        scenario_id="sc-pin",
        site_identity=_identity(),
    )
    assert record.site_identity == _identity()

    # The live geometry matching the pin verifies (and returns the record).
    verified = store.verify_site_identity("sc-pin", _identity())
    assert verified.scenario_id == "sc-pin"

    # rows/cols/pixel identical, origin shifted: the OLD guard was blind
    # to exactly this; the pin refuses it.
    with pytest.raises(SiteIdentityMismatch) as info:
        store.verify_site_identity("sc-pin", _identity(origin=(ORIGIN[0] + 500.0, ORIGIN[1])))
    assert info.value.expected == _identity()
    assert info.value.actual["origin_x_m"] == ORIGIN[0] + 500.0
    assert info.value.actual["rows"] == ROWS  # everything else matches

    # Dimension and pixel-size swaps are refused too.
    with pytest.raises(SiteIdentityMismatch):
        store.verify_site_identity("sc-pin", _identity(rows=ROWS + 8))
    with pytest.raises(SiteIdentityMismatch):
        store.verify_site_identity("sc-pin", _identity(pixel=PIXEL * 2))
    with pytest.raises(SiteIdentityMismatch):
        store.verify_site_identity("sc-pin", _identity(site_id="other-site"))
    store.close()


def test_store_identity_comparison_is_numeric_tolerant(tmp_path: Path) -> None:
    """int 2 vs float 2.0 is the same geometry, not a mismatch."""
    store = Store(tmp_path / "state" / "app.db", results_root=tmp_path / "results")
    store.create_scenario(
        site_id=SITE_ID,
        name="pinned",
        scenario_id="sc-num",
        site_identity={
            "site_id": SITE_ID,
            "rows": 32,
            "cols": 48,
            "pixel_size_m": 2,
            "origin_x_m": 1000,
            "origin_y_m": 2000,
        },
    )
    store.verify_site_identity(
        "sc-num",
        {
            "site_id": SITE_ID,
            "rows": 32.0,
            "cols": 48,
            "pixel_size_m": 2.0,
            "origin_x_m": 1000.0,
            "origin_y_m": 2000.0,
        },
    )
    store.close()


def test_store_lazily_pins_legacy_null_rows(tmp_path: Path) -> None:
    """A pre-upgrade row (NULL pin) adopts the live identity it has been
    running under on its first verified request, then guards it."""
    store = Store(tmp_path / "state" / "app.db", results_root=tmp_path / "results")
    store.create_scenario(
        site_id=SITE_ID, name="legacy", scenario_id="sc-legacy"
    )  # no site_identity: the pre-v3 shape
    record = store.require_scenario("sc-legacy")
    assert record.site_identity is None

    pinned = store.verify_site_identity("sc-legacy", _identity())
    assert pinned.site_identity == _identity()

    # Now the pin exists and a swapped cache is refused.
    with pytest.raises(SiteIdentityMismatch):
        store.verify_site_identity(
            "sc-legacy", _identity(origin=(ORIGIN[0], ORIGIN[1] - 250.0))
        )
    store.close()


def test_pin_covers_time_steps_and_skips_legacy_absence(tmp_path: Path) -> None:
    """u-d1b nit c: the pin covers ``time_steps`` — the registry's
    geometry() carries it (jobs.py) and the executor-level grid identity
    compares it, so a cache regenerated with a different time-axis
    length is a different site. A legacy pin that never recorded the
    field cannot conflict on it (the None-skip)."""
    store = Store(tmp_path / "state" / "app.db", results_root=tmp_path / "results")
    store.create_scenario(
        site_id=SITE_ID,
        name="pinned",
        scenario_id="sc-ts",
        site_identity=_identity(),
    )
    store.verify_site_identity("sc-ts", _identity())  # recorded and matching

    with pytest.raises(SiteIdentityMismatch) as info:
        store.verify_site_identity("sc-ts", _identity(time_steps=TIME_STEPS + 2))
    assert info.value.expected["time_steps"] == TIME_STEPS
    assert info.value.actual["time_steps"] == TIME_STEPS + 2

    # A legacy pin recorded before time_steps joined the field set: the
    # absent field is skipped, not treated as a conflict.
    legacy = {key: value for key, value in _identity().items() if key != "time_steps"}
    store.create_scenario(
        site_id=SITE_ID, name="legacy", scenario_id="sc-old-ts", site_identity=legacy
    )
    store.verify_site_identity("sc-old-ts", _identity(time_steps=TIME_STEPS + 9))
    store.close()


def test_verify_takes_no_write_transaction_when_pinned(tmp_path: Path) -> None:
    """u-d1b nit e: pure verification (the common case — every edit and
    reset) reads the pin outside any write transaction; only the lazy
    pin of a legacy NULL row writes."""
    store = Store(tmp_path / "state" / "app.db", results_root=tmp_path / "results")
    store.create_scenario(
        site_id=SITE_ID,
        name="pinned",
        scenario_id="sc-ro",
        site_identity=_identity(),
    )
    writes: list[int] = []
    original_write = store._write

    @contextlib.contextmanager
    def counting_write():
        writes.append(1)
        with original_write() as conn:
            yield conn

    store._write = counting_write  # type: ignore[method-assign]

    # Matching verification: no write transaction at all.
    verified = store.verify_site_identity("sc-ro", _identity())
    assert verified.scenario_id == "sc-ro"
    assert writes == []

    # Mismatched verification refuses WITHOUT a write transaction too.
    with pytest.raises(SiteIdentityMismatch):
        store.verify_site_identity("sc-ro", _identity(rows=ROWS + 8))
    assert writes == []
    store.close()

    # The lazy pin (legacy NULL row) still takes exactly one write.
    legacy_store = Store(
        tmp_path / "state2" / "app.db", results_root=tmp_path / "results2"
    )
    legacy_store.create_scenario(site_id=SITE_ID, name="legacy", scenario_id="sc-w")
    legacy_writes: list[int] = []
    legacy_original = legacy_store._write

    @contextlib.contextmanager
    def counting_legacy_write():
        legacy_writes.append(1)
        with legacy_original() as conn:
            yield conn

    legacy_store._write = counting_legacy_write  # type: ignore[method-assign]
    legacy_store.verify_site_identity("sc-w", _identity())
    assert len(legacy_writes) == 1
    legacy_store.close()


# ---------------------------------------------------------------------------
# Endpoint behaviour: same site_id, shifted-origin cache, shared state root
# ---------------------------------------------------------------------------


def _shifted_site_cache(source: Path, target: Path, *, dx: float, dy: float) -> Path:
    """A pixel-identical, geographically SHIFTED copy of the site cache
    (same site_id, dims, pixel; different geographic origin)."""
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["origin_x_m"] = float(manifest["origin_x_m"]) + dx
    manifest["origin_y_m"] = float(manifest["origin_y_m"]) + dy
    for entry in manifest.get("rasters", {}).values():
        geotransform = list(entry["geotransform"])
        geotransform[0] = float(geotransform[0]) + dx
        geotransform[3] = float(geotransform[3]) + dy
        entry["geotransform"] = geotransform
    manifest_path.write_text(json.dumps(manifest))
    return target


@pytest.fixture()
def pinned_apps(tmp_path: Path):
    """One scenario created under the true site cache, then the SAME state
    root served by an app whose registry maps the same site_id to a
    shifted-origin cache — the swapped-data deployment the pin refuses."""
    solver = FakeSolver()
    true_cache = make_site_cache(tmp_path / "true")
    shifted_cache = _shifted_site_cache(
        true_cache, tmp_path / "shifted" / SITE_ID, dx=1000.0, dy=-2000.0
    )
    state_root = tmp_path / "state"
    true_app = make_app(
        tmp_path, solver, site_cache=true_cache, state_root=state_root
    )
    swapped_app = make_app(
        tmp_path,
        FakeSolver(),
        site_cache=shifted_cache,
        state_root=state_root,
    )
    with TestClient(true_app) as true_client:
        response = true_client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "pinned", "initial_state": "baseline"},
        )
        assert response.status_code == 201, response.text
        scenario_id = response.json()["scenario_id"]
        with TestClient(swapped_app) as swapped_client:
            yield true_client, swapped_client, scenario_id


def test_edit_under_shifted_origin_is_refused(pinned_apps) -> None:
    true_client, swapped_client, scenario_id = pinned_apps

    # Sanity: the two identities differ ONLY in origin — the exact blind
    # spot of the rows/cols/pixel guard.
    response = swapped_client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree("tree_01")}],
        },
        headers={"Idempotency-Key": "identity-edit-1"},
    )
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "site_identity_mismatch"
    pinned = error["pinned_identity"]
    current = error["current_identity"]
    assert pinned["origin_x_m"] == ORIGIN[0]
    assert current["origin_x_m"] == ORIGIN[0] + 1000.0
    assert current["rows"] == pinned["rows"]
    assert current["cols"] == pinned["cols"]
    assert current["pixel_size_m"] == pinned["pixel_size_m"]

    # The refusal is durable: the scenario did not advance, and the TRUE
    # site still accepts the same edit shape.
    assert true_client.get(f"/api/v1/scenarios/{scenario_id}").status_code == 200
    accepted = true_client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree("tree_01")}],
        },
        headers={"Idempotency-Key": "identity-edit-true"},
    )
    assert accepted.status_code == 202, accepted.text
    job = wait_for_job(true_client, accepted.json()["job_id"])
    assert job["status"] == "complete"


def test_reset_under_shifted_origin_is_refused(pinned_apps) -> None:
    _true_client, swapped_client, scenario_id = pinned_apps
    response = swapped_client.post(
        f"/api/v1/scenarios/{scenario_id}/reset", json={"base_scene_version": 0}
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "site_identity_mismatch"


def test_same_identity_mutations_still_succeed(pinned_apps) -> None:
    """Regression: the pin adds no friction to the matching deployment."""
    true_client, _swapped_client, scenario_id = pinned_apps
    edit = true_client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree("tree_01")}],
        },
        headers={"Idempotency-Key": "identity-regression"},
    )
    assert edit.status_code == 202, edit.text
    assert wait_for_job(true_client, edit.json()["job_id"])["status"] == "complete"

    reset = true_client.post(
        f"/api/v1/scenarios/{scenario_id}/reset", json={"base_scene_version": 1}
    )
    assert reset.status_code == 200, reset.text


def test_legacy_scenario_row_is_lazily_pinned_through_the_api(tmp_path: Path) -> None:
    """A scenario row created pre-upgrade (NULL pin) is pinned from the live
    registry on its first verified request, then guarded."""
    solver = FakeSolver()
    cache = make_site_cache(tmp_path / "true")
    state_root = tmp_path / "state"
    app = make_app(tmp_path, solver, site_cache=cache, state_root=state_root)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "legacy", "initial_state": "baseline"},
        )
        scenario_id = response.json()["scenario_id"]

        # Age the row back to the pre-pin shape.
        db_path = state_root / "store.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "UPDATE scenarios SET site_identity_json = NULL WHERE scenario_id = ?",
            (scenario_id,),
        )
        connection.commit()
        connection.close()

        edit = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree("tree_01")}],
            },
            headers={"Idempotency-Key": "legacy-pin"},
        )
        assert edit.status_code == 202, edit.text

        # The pin now exists (the row was lazily upgraded) and a shifted
        # cache is refused.
        connection = sqlite3.connect(db_path)
        pinned_json = connection.execute(
            "SELECT site_identity_json FROM scenarios WHERE scenario_id = ?",
            (scenario_id,),
        ).fetchone()[0]
        connection.close()
        assert pinned_json is not None
        pinned = json.loads(pinned_json)
        assert pinned["origin_x_m"] == ORIGIN[0]

        shifted_cache = _shifted_site_cache(
            cache, tmp_path / "shifted" / SITE_ID, dx=64.0, dy=0.0
        )
        swapped_app = make_app(
            tmp_path,
            FakeSolver(),
            site_cache=shifted_cache,
            state_root=state_root,
        )
        with TestClient(swapped_app) as swapped_client:
            refused = swapped_client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [{"operation": "add", "tree": make_tree("tree_02")}],
                },
                headers={"Idempotency-Key": "legacy-refuse"},
            )
            assert refused.status_code == 409
            assert refused.json()["error"]["code"] == "site_identity_mismatch"
