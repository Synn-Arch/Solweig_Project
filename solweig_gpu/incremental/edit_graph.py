# SPDX-License-Identifier: GPL-3.0-only
"""Versioned dependency graph for the universal editing engine.

The graph is CODE-DEFINED: the node/edge sets below mirror
``docs/incremental_design_tool/universal_editing/dependency_graph.yaml``
(reconciled to the code by the U-A audit, 2026-09-02), but the runtime
never reads the docs tree. A drift-guard test loads the YAML and asserts
the sets are identical, so documentation and code cannot diverge silently.

Semantics
---------

**Kinds.** ``source`` nodes are editable inputs, ``derived`` nodes are pure
functions of their inputs, ``stateful`` nodes additionally accumulate
across timesteps (``surface_thermal_state``), ``output`` nodes are
published products, and ``view`` nodes select presentation without any
scientific stage.

**Invalidation dimensions.** Every dirty node carries four dimensions
(mission invariant ``dependency_graph_is_versioned``):

1. *stage* — which nodes are dirty: the downstream closure of the changed
   sources (:meth:`EditGraph.downstream`);
2. *spatial* — windows vs full tile, decided per node by the planner;
3. *temporal* — which timesteps, decided per node by the planner;
4. *version* — :class:`SceneGraphState` bumps a per-node version counter
   for every node in the dirty closure and leaves the rest untouched, so
   "inputs' versions unchanged" is a mechanical reusability test.

**Determinism.** :meth:`EditGraph.topological_order` is Kahn's algorithm
with a lexicographic (node-id) tie-break, so the order is unique for a
given graph — the reproducibility invariant the planner relies on.

**Acyclicity.** The graph is acyclic today and stays enforced: constructing
an :class:`EditGraph` with a cycle raises :class:`EditGraphError`.

Conservative choice: a node counts as dirty as soon as *any* path from a
changed source reaches it (plain transitive closure, no cancellation), and
reusability is exactly the complement of the closure. E.g. a
``vegetation_dsm`` edit dirties ``relative_geometry`` (direct edge exists)
even though a tree edit rarely changes relative heights materially — the
graph, not a per-edit guess, decides.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

__all__ = [
    "EditGraph",
    "EditGraphError",
    "GRAPH_SCHEMA_VERSION",
    "GraphNode",
    "NodeKind",
    "SceneGraphState",
    "default_edit_graph",
]

#: Bump when the node/edge sets or their semantics change (result patches
#: record this version; publication layers reject graph-version drift).
GRAPH_SCHEMA_VERSION = 1


class EditGraphError(RuntimeError):
    """Graph construction or traversal failure (unknown node, cycle)."""


class NodeKind(str, Enum):
    """Node role in the invalidation graph."""

    SOURCE = "source"
    DERIVED = "derived"
    STATEFUL = "stateful"
    OUTPUT = "output"
    VIEW = "view"


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One node of the dependency graph."""

    node_id: str
    kind: NodeKind


# ---------------------------------------------------------------------------
# Canonical node/edge sets (mirror dependency_graph.yaml — do not edit one
# without the other; tests/test_incremental_edit_engine.py enforces parity).
# ---------------------------------------------------------------------------

GRAPH_NODES: tuple[GraphNode, ...] = (
    GraphNode("dem", NodeKind.SOURCE),
    GraphNode("building_dsm", NodeKind.SOURCE),
    GraphNode("vegetation_dsm", NodeKind.SOURCE),
    GraphNode("landcover", NodeKind.SOURCE),
    GraphNode("meteorology", NodeKind.SOURCE),
    GraphNode("selected_date_time", NodeKind.SOURCE),
    GraphNode("wind_coefficients", NodeKind.SOURCE),
    GraphNode("model_parameters", NodeKind.SOURCE),
    GraphNode("output_selection", NodeKind.VIEW),
    GraphNode("relative_geometry", NodeKind.DERIVED),
    GraphNode("walls", NodeKind.DERIVED),
    GraphNode("wall_aspect", NodeKind.DERIVED),
    GraphNode("building_visibility", NodeKind.DERIVED),
    GraphNode("vegetation_visibility", NodeKind.DERIVED),
    GraphNode("svf", NodeKind.DERIVED),
    GraphNode("solar_atmospheric_state", NodeKind.DERIVED),
    GraphNode("time_shadow", NodeKind.DERIVED),
    GraphNode("radiation", NodeKind.DERIVED),
    GraphNode("surface_thermal_state", NodeKind.STATEFUL),
    GraphNode("tmrt", NodeKind.OUTPUT),
    GraphNode("utci", NodeKind.OUTPUT),
    GraphNode("wbgt", NodeKind.OUTPUT),
)

#: Directed edges ``[producer, consumer]`` mirroring the reconciled YAML.
GRAPH_EDGES: tuple[tuple[str, str], ...] = (
    ("dem", "relative_geometry"),
    ("building_dsm", "relative_geometry"),
    ("vegetation_dsm", "relative_geometry"),
    ("building_dsm", "walls"),
    ("building_dsm", "wall_aspect"),
    ("building_dsm", "building_visibility"),
    ("dem", "vegetation_visibility"),
    ("relative_geometry", "vegetation_visibility"),
    ("vegetation_dsm", "vegetation_visibility"),
    ("building_visibility", "svf"),
    ("vegetation_visibility", "svf"),
    ("meteorology", "solar_atmospheric_state"),
    ("selected_date_time", "solar_atmospheric_state"),
    ("solar_atmospheric_state", "time_shadow"),
    ("building_visibility", "time_shadow"),
    ("vegetation_visibility", "time_shadow"),
    ("walls", "time_shadow"),
    ("wall_aspect", "time_shadow"),
    ("svf", "radiation"),
    ("time_shadow", "radiation"),
    ("landcover", "radiation"),
    ("meteorology", "radiation"),
    ("model_parameters", "radiation"),
    ("landcover", "surface_thermal_state"),
    ("meteorology", "surface_thermal_state"),
    ("radiation", "surface_thermal_state"),
    ("radiation", "tmrt"),
    ("surface_thermal_state", "tmrt"),
    ("meteorology", "tmrt"),
    ("tmrt", "utci"),
    ("meteorology", "utci"),
    ("wind_coefficients", "utci"),
    ("tmrt", "wbgt"),
    ("meteorology", "wbgt"),
    ("wind_coefficients", "wbgt"),
    ("time_shadow", "wbgt"),
)


class EditGraph:
    """Immutable, acyclic, deterministically ordered dependency graph."""

    __slots__ = ("_children", "_edges", "_kinds", "_nodes", "_order", "_parents")

    def __init__(
        self,
        nodes: Iterable[GraphNode],
        edges: Iterable[tuple[str, str]],
    ) -> None:
        node_list = tuple(nodes)
        edge_list = tuple(edges)
        kinds: dict[str, NodeKind] = {}
        for node in node_list:
            if not isinstance(node.node_id, str) or not node.node_id:
                raise EditGraphError("node ids must be non-empty strings")
            if not isinstance(node.kind, NodeKind):
                raise EditGraphError("node kinds must be NodeKind values")
            if node.node_id in kinds:
                raise EditGraphError(f"duplicate node id {node.node_id!r}")
            kinds[node.node_id] = node.kind
        children: dict[str, list[str]] = {node_id: [] for node_id in kinds}
        parents: dict[str, list[str]] = {node_id: [] for node_id in kinds}
        seen_edges: set[tuple[str, str]] = set()
        for producer, consumer in edge_list:
            for endpoint in (producer, consumer):
                if endpoint not in kinds:
                    raise EditGraphError(
                        f"edge ({producer!r}, {consumer!r}) references "
                        f"unknown node {endpoint!r}"
                    )
            if producer == consumer:
                raise EditGraphError(f"self-edge on node {producer!r}")
            edge = (producer, consumer)
            if edge in seen_edges:
                raise EditGraphError(f"duplicate edge {edge!r}")
            seen_edges.add(edge)
            children[producer].append(consumer)
            parents[consumer].append(producer)

        self._nodes = node_list
        self._edges = edge_list
        self._kinds = kinds
        self._children = {
            node_id: tuple(sorted(targets)) for node_id, targets in children.items()
        }
        self._parents = {
            node_id: tuple(sorted(sources)) for node_id, sources in parents.items()
        }
        # Cycle check + canonical order in one pass; raises EditGraphError
        # on a cycle (the graph is acyclic today and stays enforced).
        self._order = self._kahn_order()

    # -- structure ----------------------------------------------------------

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        return self._nodes

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return self._edges

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(self._kinds)

    @property
    def schema_version(self) -> int:
        return GRAPH_SCHEMA_VERSION

    def contains(self, node_id: str) -> bool:
        return node_id in self._kinds

    def kind(self, node_id: str) -> NodeKind:
        try:
            return self._kinds[node_id]
        except KeyError:
            raise EditGraphError(f"unknown node {node_id!r}") from None

    def is_executable(self, node_id: str) -> bool:
        """True when the node is a computed stage (not a source or view)."""
        return self.kind(node_id) not in (NodeKind.SOURCE, NodeKind.VIEW)

    def parents(self, node_id: str) -> tuple[str, ...]:
        """Direct inputs of ``node_id``, sorted (deterministic)."""
        self.kind(node_id)  # existence check
        return self._parents[node_id]

    def children(self, node_id: str) -> tuple[str, ...]:
        """Direct consumers of ``node_id``, sorted (deterministic)."""
        self.kind(node_id)
        return self._children[node_id]

    # -- ordering and closures ----------------------------------------------

    def _kahn_order(self) -> tuple[str, ...]:
        """Kahn's algorithm with lexicographic tie-break (unique order)."""
        remaining_parents = {
            node_id: len(self._parents[node_id]) for node_id in self._kinds
        }
        ready = [
            node_id for node_id, count in remaining_parents.items() if count == 0
        ]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            node_id = heapq.heappop(ready)
            order.append(node_id)
            for child in self._children[node_id]:
                remaining_parents[child] -= 1
                if remaining_parents[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(self._kinds):
            cyclic = sorted(
                node_id
                for node_id, count in remaining_parents.items()
                if count > 0
            )
            raise EditGraphError(
                "dependency graph contains a cycle among: " + ", ".join(cyclic)
            )
        return tuple(order)

    def topological_order(self) -> tuple[str, ...]:
        """The deterministic topological order (stable tie-break by id)."""
        return self._order

    def downstream(self, node_ids: Iterable[str]) -> frozenset[str]:
        """Transitive closure of consumers, *including* the input nodes.

        This is the dirty set for a source change: conservative plain
        reachability, no cancellation and no per-edit exceptions.
        """
        closure: set[str] = set()
        stack: list[str] = []
        for node_id in node_ids:
            self.kind(node_id)  # existence check
            if node_id not in closure:
                closure.add(node_id)
                stack.append(node_id)
        while stack:
            current = stack.pop()
            for child in self._children[current]:
                if child not in closure:
                    closure.add(child)
                    stack.append(child)
        return frozenset(closure)

    def upstream(self, node_ids: Iterable[str]) -> frozenset[str]:
        """Transitive closure of inputs, *including* the input nodes."""
        closure: set[str] = set()
        stack: list[str] = []
        for node_id in node_ids:
            self.kind(node_id)
            if node_id not in closure:
                closure.add(node_id)
                stack.append(node_id)
        while stack:
            current = stack.pop()
            for parent in self._parents[current]:
                if parent not in closure:
                    closure.add(parent)
                    stack.append(parent)
        return frozenset(closure)


_DEFAULT_EDIT_GRAPH: EditGraph | None = None


def default_edit_graph() -> EditGraph:
    """The canonical scene graph (built once, then cached)."""
    global _DEFAULT_EDIT_GRAPH
    if _DEFAULT_EDIT_GRAPH is None:
        _DEFAULT_EDIT_GRAPH = EditGraph(GRAPH_NODES, GRAPH_EDGES)
    return _DEFAULT_EDIT_GRAPH


# ---------------------------------------------------------------------------
# Versioned scene state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneGraphState:
    """Per-node version counters plus the graph-level scene revision.

    Immutable value object: :meth:`advance` returns a new state and never
    mutates in place, so a planner run can hold a consistent snapshot while
    edits land elsewhere. ``node_versions`` is exposed read-only (a
    :class:`types.MappingProxyType` over a copy).

    Bump semantics along the four invalidation dimensions:

    - *stage*: exactly the downstream closure of the changed nodes gets +1;
    - *version*: unchanged nodes keep their counter — mechanically
      identifying reusable caches ("inputs' versions unchanged");
    - *spatial/temporal*: not tracked here; they live per-plan in each
      :class:`~solweig_gpu.incremental.edit_types.NodeImpact`.

    Not hashable by design (mapping field); equality is structural.
    """

    graph: EditGraph
    scene_revision: int = 0
    node_versions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, EditGraph):
            raise EditGraphError("graph must be an EditGraph")
        if isinstance(self.scene_revision, bool) or not isinstance(
            self.scene_revision, int
        ):
            raise EditGraphError("scene_revision must be an integer")
        if self.scene_revision < 0:
            raise EditGraphError("scene_revision must be non-negative")
        versions = dict(self.node_versions)
        for node_id in versions:
            if not self.graph.contains(node_id):
                raise EditGraphError(f"version entry for unknown node {node_id!r}")
        object.__setattr__(
            self, "node_versions", MappingProxyType(versions)
        )

    @classmethod
    def initial(cls, graph: EditGraph | None = None) -> "SceneGraphState":
        """Revision 0 with every node at version 0."""
        return cls(graph if graph is not None else default_edit_graph(), 0, {})

    def version(self, node_id: str) -> int:
        """Current version counter of ``node_id`` (0 when never bumped)."""
        self.graph.kind(node_id)  # existence check
        return self.node_versions.get(node_id, 0)

    def advance(self, changed_nodes: Iterable[str]) -> "SceneGraphState":
        """Bump the closure of ``changed_nodes`` and the scene revision.

        The dirty closure is the *stage* dimension; per-node counters are
        the *version* dimension; the +1 scene revision is the publication
        token publication layers compare against.
        """
        changed = tuple(changed_nodes)
        closure = self.graph.downstream(changed)
        versions = dict(self.node_versions)
        for node_id in sorted(closure):
            versions[node_id] = versions.get(node_id, 0) + 1
        return SceneGraphState(self.graph, self.scene_revision + 1, versions)
