#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "incremental_design_tool"
UNI = DOC / "universal_editing"
AGENT = DOC / "agent"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    required = [
        UNI / "index.md",
        UNI / "editable_scope.md",
        UNI / "dependency_graph.md",
        UNI / "edit_adapter_contract.md",
        UNI / "execution_pipeline.md",
        UNI / "frontend_spec.md",
        UNI / "validation_matrix.md",
        UNI / "SUPERSEDES_TREE_ONLY_SCOPE.md",
        UNI / "dependency_graph.yaml",
        UNI / "edit_registry.yaml",
        UNI / "validation_gates.yaml",
        AGENT / "goal_condition_universal.txt",
        AGENT / "mission_universal.yaml",
        AGENT / "master_prompt_universal.md",
        AGENT / "multi_agent_orchestration_universal.md",
        AGENT / "subagent_prompt_universal.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"missing files: {missing}"

    goal = (AGENT / "goal_condition_universal.txt").read_text(encoding="utf-8")
    assert len(goal) <= 4000, f"goal is {len(goal)} characters"
    for marker in ["Tree placement is only a reference example", "EditAdapter", "MULTI-AGENT", "DONE"]:
        assert marker in goal, marker

    graph = load_yaml(UNI / "dependency_graph.yaml")
    node_ids = [node["id"] for node in graph["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    adjacency = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for source, target in graph["edges"]:
        assert source in node_ids, source
        assert target in node_ids, target
        adjacency[source].append(target)
        indegree[target] += 1

    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    assert visited == len(node_ids), "dependency graph must be acyclic"

    registry = load_yaml(UNI / "edit_registry.yaml")
    statuses = set(registry["status_values"])
    adapter_ids = [adapter["id"] for adapter in registry["adapters"]]
    assert len(adapter_ids) == len(set(adapter_ids))
    groups = set()
    for adapter in registry["adapters"]:
        assert adapter["status"] in statuses
        assert adapter["source_nodes"]
        assert all(node in node_ids for node in adapter["source_nodes"])
        assert adapter["operations"]
        if adapter["status"] not in {"scientific_extension", "view_only"}:
            assert adapter["validation_fixtures"], adapter["id"]
        groups.add(adapter["group"])
    assert {"geometry", "surface", "environment", "advanced", "analysis"} <= groups

    mission = load_yaml(AGENT / "mission_universal.yaml")
    assert mission["multi_agent"]["required"] is True
    assert mission["multi_agent"]["one_writer_per_file"] is True
    assert mission["scientific_oracle"] == "current_full_domain_solweig"
    assert set(mission["source_families"]) <= set(node_ids)
    gate_ids = {gate["id"] for gate in load_yaml(UNI / "validation_gates.yaml")["gates"]}
    assert set(mission["mandatory_gates"]) == gate_ids

    print(f"Universal interactive spec valid; goal length={len(goal)}; adapters={len(adapter_ids)}; nodes={len(node_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
