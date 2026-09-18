# SPDX-License-Identifier: GPL-3.0-only
"""CPU-side result publication lease (T15, mirroring the T13 OutputLease).

The T13 GPU residency model publishes results through an
:class:`~solweig_gpu.cuda.residency.OutputLease`: ``publish()`` refuses
while events are still pending (:class:`PublishPendingError`), fences on
the output slot's generation (:class:`LeaseStaleError`), and ``release()``
abandons without publishing. This module is the CPU analogue for the job
runner's exact-lane publication: the "completion" is the solve having
produced its result AND the job row still being the one that produced it.

The lease is a NARROW seam over the existing store fences — it delegates
to :meth:`Store.publish_result`, so every durable guarantee (version
fence ``StaleResultError``, in-transaction running-check
``ResultNotPublishable``, idempotent-complete ``ResultAlreadyPublished``)
is unchanged. What it adds is the ORDERING discipline the finalize path
must not get wrong:

* mint from a ``running`` job row (anything else is stale at mint);
* ``mark_complete()`` once the result exists (a job row that left
  ``running`` in between means the lease's slot was reused — stale);
* ``publish()`` only after completion, at most once per lease (a second
  publish on the same lease is the double-publish class T13 fences).

Typing follows T13 so the two lanes read the same way; every error here
subclasses :class:`solweig_gpu.server.store.StoreError` for the existing
route/finalize handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from solweig_gpu.server.store import Store, StoreError

if TYPE_CHECKING:  # pragma: no cover - import cycle only under type check
    from solweig_gpu.server.jobs import JobRecord

__all__ = [
    "LeaseStaleError",
    "PublishPendingError",
    "ResultPublishLease",
]


class PublishPendingError(StoreError):
    """``publish()`` before ``mark_complete()`` — completion is pending.

    The T13 ``events pending`` analogue: the lease exists, the solver's
    result may even be in hand at the caller, but the lease's own
    completion protocol has not run. Publishing now would let a result
    whose job-row validity was never re-checked reach the store.
    """

    advice = "call mark_complete() on the lease before publishing"

    def body(self) -> dict[str, Any]:
        return {"code": "publish_pending", "advice": self.advice}


class LeaseStaleError(StoreError):
    """The job row this lease was minted from is no longer ``running``.

    The T13 slot-generation analogue: the slot was reused (the job was
    cancelled or finished by another path) between mint and publish, so
    the result this lease would publish is a discard by contract. The
    job row already carries its own outcome; the lease adds none.
    """

    advice = (
        "the job left status 'running' after this lease was minted; the "
        "result is a discard (the job row already records its outcome)"
    )

    def __init__(self, job_id: str, status: str) -> None:
        self.job_id = str(job_id)
        self.status = str(status)
        super().__init__(
            f"publish lease for job {self.job_id!r} is stale: the job row is "
            f"{self.status!r}, not 'running'"
        )

    def body(self) -> dict[str, Any]:
        return {
            "code": "lease_stale",
            "job_id": self.job_id,
            "status": self.status,
            "advice": self.advice,
        }


class ResultPublishLease:
    """One job's fenced publication slot (mint → complete → publish)."""

    def __init__(self, store: Store, scenario_id: str, scene_version: int, job_id: str) -> None:
        self._store = store
        self.scenario_id = str(scenario_id)
        self.scene_version = int(scene_version)
        self.job_id = str(job_id)
        self._completed = False
        self._published = False

    @classmethod
    def from_job(cls, store: Store, job: "JobRecord") -> "ResultPublishLease":
        """Mint a lease for one job row; refuse a non-running row.

        Minting from a terminal row is stale by definition — the outcome
        that row carries is the truth, and a late solve's result must
        never overwrite it.
        """
        if job.status != "running":
            raise LeaseStaleError(job.job_id, job.status)
        return cls(store, job.scenario_id, int(job.target_scene_version), job.job_id)

    # -- fence state ---------------------------------------------------------

    def _require_running(self) -> None:
        live = self._store.get_job(self.job_id)
        if live is None or live.status != "running":
            raise LeaseStaleError(
                self.job_id, live.status if live is not None else "deleted"
            )

    def mark_complete(self) -> None:
        """Record that the lease's result is complete (T13 completion).

        Re-fences the job row at completion time: a job that left
        ``running`` between mint and completion had its slot reused.
        """
        self._require_running()
        self._completed = True

    # -- publication ---------------------------------------------------------

    def publish(
        self,
        *,
        manifest: Mapping[str, Any],
        payload: bytes,
        acked_sequence: int | None = None,
    ) -> None:
        """Publish through the store, fenced on completion and freshness.

        Raises :class:`PublishPendingError` before ``mark_complete()``,
        :class:`LeaseStaleError` when the job row left ``running``, and a
        plain double-publish refusal on the second call. Everything the
        store itself fences (``StaleResultError``,
        ``ResultNotPublishable``, ``ResultAlreadyPublished``) propagates
        unchanged.
        """
        if not self._completed:
            raise PublishPendingError(
                f"job {self.job_id!r} lease has no completed result to publish; "
                + PublishPendingError.advice
            )
        if self._published:
            raise StoreError(
                f"lease for job {self.job_id!r} already published scene version "
                f"{self.scene_version}; mint a new lease per publication"
            )
        self._require_running()
        self._published = True
        self._store.publish_result(
            self.scenario_id,
            self.scene_version,
            manifest=dict(manifest),
            payload=payload,
            exact=True,
            job_id=self.job_id,
            acked_sequence=acked_sequence,
        )

    def release(self) -> None:
        """Abandon the lease without publishing (T13 release).

        Idempotent; a released lease can no longer publish (its
        completion, if any, is voided — exactly an abandon).
        """
        self._completed = False
        self._published = True
