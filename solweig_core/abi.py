"""Torch-neutral buffer ABI (DESIGN.ko.md 5.3, TASKS T02).

Every array handed across the new core boundary is an :class:`ArrayView`
carrying, as TYPES and not conventions:

* ``owner`` — a hard reference to the numpy buffer the view borrows from
  (the view keeps the owner alive; a stale free is impossible while the
  view lives);
* ``dtype`` / ``shape`` / ``strides`` — byte strides, honoured verbatim;
* ``global_origin`` — where element ``[0, 0]`` of the view sits in the
  LOGICAL domain grid (row0, col0; a third leading entry for time stacks);
* ``logical_domain_id`` — which logical domain that origin refers to;
* ``residency`` — ``'cpu'`` is the only value solweig_core can host today;
* ``read_only`` — the view's borrow discipline.

``logical_domain`` (where reference arithmetic and boundary rules live) and
the physical work window (which cells are actually computed — see
``solweig_core.request.PhysicalWindow``) are SEPARATE fields on SEPARATE
objects: collapsing them is how window shrinking silently changes ray
termination and reductions (DESIGN 5.3).

Non-contiguity is never silently treated as contiguity. A kernel that
cannot honour strides must either call :meth:`ArrayView.require_contiguous`
(refusing strided input with :class:`NonContiguousError`) or take a single
boundary copy through :meth:`ArrayView.to_contiguous` (dtype and logical
order preserved). Logical-order equality across differently-strided views
of the same values is checked with :meth:`ArrayView.same_logical_values`.

Stale-borrow policy (pick documented in T02): the owner reference keeps the
buffer ALIVE, and mutation after view creation is ALLOWED for plain views
(the legacy path legitimately mutates planes mid-solve) but DETECTED for
*frozen* views — :meth:`ArrayView.freeze` pins a digest of the logical
bytes and :meth:`ArrayView.assert_not_mutated` re-checks it, raising
:class:`StaleBufferError`. The adapter seam freezes read-only INPUT planes
at the boundary and asserts them unchanged after the solve.

NO torch import is permitted anywhere in ``solweig_core``; numpy is the
only array dependency of this module (enforced by
``tests/ultrafast/test_abi.py``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from solweig_core.status import (
    AbiValidationError,
    NonContiguousError,
    RefusalReason,
    StaleBufferError,
    TypedRefusal,
)

__all__ = [
    "RESIDENCY_CPU",
    "KNOWN_RESIDENCIES",
    "ArrayView",
]

#: The only residency solweig_core can host today. A buffer claiming any
#: other residency fails ABI validation on this host — including ``'cuda'``,
#: which the core can never allocate or borrow (it never imports torch).
RESIDENCY_CPU = "cpu"
KNOWN_RESIDENCIES = (RESIDENCY_CPU,)

_DIGEST_CHUNK = 1 << 20


def _logical_digest(array: np.ndarray) -> str:
    """sha256 of the array's bytes in LOGICAL (C) order, strides honoured.

    Reading through ``.flat``/``ascontiguousarray`` respects the strides, so
    two views with identical logical values digest identically even when
    their byte layouts differ (transpose, slice, negative stride).
    """
    h = hashlib.sha256()
    for start in range(0, int(array.size), _DIGEST_CHUNK):
        stop = min(start + _DIGEST_CHUNK, int(array.size))
        h.update(np.ascontiguousarray(array.flat[start:stop]).tobytes())
    return h.hexdigest()


@dataclass
class ArrayView:
    """A typed borrow of a numpy buffer, honouring dtype/shape/strides.

    Construct through :meth:`from_numpy` (which records the real strides —
    never ``assume`` contiguity). The dataclass is not frozen only so
    :meth:`freeze` can pin a digest; shape/strides/dtype/origin are not to
    be mutated after construction, and :meth:`validate` re-checks them
    against the owner on demand.
    """

    owner: np.ndarray
    dtype: np.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # BYTES per axis (numpy convention)
    global_origin: tuple[int, ...]
    logical_domain_id: str
    residency: str = RESIDENCY_CPU
    read_only: bool = False
    _frozen_digest: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate at construction: a hand-built view that lies about its
        # owner (shape/strides/dtype/rank/residency/borrow direction) must
        # never reach a kernel. from_numpy re-validates on demand.
        self.validate()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        *,
        global_origin: tuple[int, ...] = (0, 0),
        logical_domain_id: str,
        read_only: bool | None = None,
        freeze: bool = False,
    ) -> "ArrayView":
        """Borrow ``array`` (no copy) under the ABI.

        ``read_only`` defaults to the owner's own writability (a stricter
        view over a writable owner is fine; the reverse is refused by
        :meth:`validate`). ``freeze=True`` pins a digest of the logical
        bytes for later :meth:`assert_not_mutated` checks.
        """
        if not isinstance(array, np.ndarray):
            raise AbiValidationError(
                TypedRefusal(
                    RefusalReason.ABI_INVALID,
                    f"owner must be numpy.ndarray, got {type(array).__name__}",
                )
            )
        if not isinstance(logical_domain_id, str) or not logical_domain_id:
            raise AbiValidationError(
                TypedRefusal(
                    RefusalReason.ABI_INVALID,
                    "logical_domain_id must be a non-empty string",
                )
            )
        view = cls(
            owner=array,
            dtype=array.dtype,
            shape=tuple(int(s) for s in array.shape),
            strides=tuple(int(s) for s in array.strides),
            global_origin=tuple(int(o) for o in global_origin),
            logical_domain_id=logical_domain_id,
            read_only=(not array.flags.writeable) if read_only is None else bool(read_only),
        )
        view.validate()
        if freeze:
            view._frozen_digest = _logical_digest(array)
        return view

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Re-check the ABI invariants; raise :class:`AbiValidationError`.

        Any inconsistency refuses loudly — including a residency other than
        ``'cpu'`` (a buffer claiming CUDA residency cannot exist in this
        torch-free core) and a writable borrow of a read-only owner (the
        cache memmaps are read-only; a writable view over them would be a
        lie the kernel could act on).
        """
        if not isinstance(self.owner, np.ndarray):
            raise self._refusal("owner is not a numpy ndarray")
        if self.residency not in KNOWN_RESIDENCIES:
            raise self._refusal(
                f"residency {self.residency!r} is not hostable by solweig_core "
                f"(known: {list(KNOWN_RESIDENCIES)}); a CUDA-resident buffer "
                "cannot exist in this torch-free core"
            )
        ndim = self.owner.ndim
        if len(self.shape) != ndim or len(self.strides) != ndim:
            raise self._refusal(
                f"shape/strides rank ({len(self.shape)}/{len(self.strides)}) "
                f"does not match owner rank {ndim}"
            )
        if tuple(int(s) for s in self.owner.shape) != tuple(self.shape):
            raise self._refusal(
                f"declared shape {self.shape} != owner shape {self.owner.shape}"
            )
        if tuple(int(s) for s in self.owner.strides) != tuple(self.strides):
            raise self._refusal(
                f"declared strides {self.strides} != owner strides "
                f"{self.owner.strides} (bytes); strides are honoured, not assumed"
            )
        if self.dtype != self.owner.dtype:
            raise self._refusal(
                f"declared dtype {self.dtype} != owner dtype {self.owner.dtype}"
            )
        if len(self.global_origin) != ndim:
            raise self._refusal(
                f"global_origin rank {len(self.global_origin)} != owner rank {ndim}"
            )
        if any(int(o) < 0 for o in self.global_origin):
            raise self._refusal(f"negative global origin {self.global_origin}")
        if any(int(s) < 0 for s in self.shape):
            raise self._refusal(f"negative shape {self.shape}")
        for axis, (size, stride) in enumerate(zip(self.shape, self.strides)):
            if size > 1 and stride == 0:
                raise self._refusal(
                    f"zero byte stride on axis {axis} of size {size}: overlapping "
                    "memory is not a valid borrow"
                )
        if not self.read_only and not self.owner.flags.writeable:
            raise self._refusal(
                "writable borrow of a read-only owner (cache memmaps are "
                "read-only); declare the view read_only or copy at the boundary"
            )

    def _refusal(self, detail: str) -> AbiValidationError:
        return AbiValidationError(
            TypedRefusal(
                RefusalReason.ABI_INVALID,
                detail,
                context={"plane_domain": self.logical_domain_id},
            )
        )

    # ------------------------------------------------------------------
    # contiguity discipline
    # ------------------------------------------------------------------
    def is_contiguous(self) -> bool:
        """C-contiguity computed FROM shape/strides (never assumed).

        Mirrors numpy's rule: the last axis stride must equal the itemsize
        and each earlier axis's stride must equal the following axes' span.
        """
        itemsize = self.dtype.itemsize
        expected = itemsize
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size > 1 and stride != expected:
                return False
            if size > 1:
                expected *= size
        return True

    def require_contiguous(self) -> "ArrayView":
        """Return ``self`` if C-contiguous, else refuse with a typed error.

        The seam for kernels that cannot honour strides: they call this (and
        get :class:`NonContiguousError` on strided input) or
        :meth:`to_contiguous` (boundary copy). Silently indexing a strided
        view as if contiguous is the bug this method exists to prevent.
        """
        if not self.is_contiguous():
            raise NonContiguousError(
                TypedRefusal(
                    RefusalReason.NON_CONTIGUOUS,
                    f"strides {self.strides} are not C-contiguous for shape "
                    f"{self.shape}; handle strides, copy at the boundary "
                    "(to_contiguous), or refuse",
                    context={"plane_domain": self.logical_domain_id},
                )
            )
        return self

    def to_contiguous(self) -> "ArrayView":
        """Boundary copy with dtype and logical order preserved.

        The single sanctioned copy for stride-honouring kernels at the ABI
        boundary (DESIGN 5.3): one copy, dtype/order preserved, metadata
        carried over. The returned view borrows the COPY (fresh owner).
        """
        if self.is_contiguous():
            return self
        copy = np.ascontiguousarray(self.owner)
        return ArrayView(
            owner=copy,
            dtype=copy.dtype,
            shape=tuple(copy.shape),
            strides=tuple(copy.strides),
            global_origin=self.global_origin,
            logical_domain_id=self.logical_domain_id,
            residency=self.residency,
            read_only=self.read_only,
        )

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        """The borrowed owner (no copy). Mutation is visible through the
        view; frozen views detect it via :meth:`assert_not_mutated`."""
        return self.owner

    @property
    def nbytes(self) -> int:
        return int(self.owner.nbytes)

    # ------------------------------------------------------------------
    # staleness (freeze-on-lend, opt-in)
    # ------------------------------------------------------------------
    def freeze(self) -> "ArrayView":
        """Pin a digest of the logical bytes (freeze-on-lend)."""
        self._frozen_digest = _logical_digest(self.owner)
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen_digest is not None

    def assert_not_mutated(self) -> None:
        """Raise :class:`StaleBufferError` if a frozen owner changed."""
        if self._frozen_digest is None:
            from solweig_core.status import AbiError

            raise AbiError(
                "assert_not_mutated on a view that was never frozen; call "
                "freeze() or from_numpy(freeze=True) first"
            )
        current = _logical_digest(self.owner)
        if current != self._frozen_digest:
            raise StaleBufferError(
                TypedRefusal(
                    RefusalReason.STALE_BUFFER,
                    "owner mutated after view creation (frozen digest mismatch)",
                    context={
                        "plane_domain": self.logical_domain_id,
                        "frozen_sha256": self._frozen_digest,
                        "current_sha256": current,
                    },
                )
            )

    # ------------------------------------------------------------------
    # logical-order equality
    # ------------------------------------------------------------------
    def same_logical_values(self, other: "ArrayView") -> bool:
        """Raw-bit equality of the LOGICAL cell sequence, strides ignored.

        A transpose or negative-stride view of the same values compares
        equal here (same logical order by index) while comparing unequal at
        the byte-layout level — the property the seam must preserve.
        """
        if not isinstance(other, ArrayView):
            return NotImplemented
        if self.dtype != other.dtype or self.shape != other.shape:
            return False
        return np.array_equal(
            np.ascontiguousarray(self.owner), np.ascontiguousarray(other.owner)
        )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary for run records."""
        return {
            "dtype": str(self.dtype.str),
            "shape": list(self.shape),
            "strides": list(self.strides),
            "global_origin": list(self.global_origin),
            "logical_domain_id": self.logical_domain_id,
            "residency": self.residency,
            "read_only": self.read_only,
            "contiguous": self.is_contiguous(),
            "frozen": self.frozen,
        }
