# Universal adapter subagent prompt template

You own one isolated task in the SOLWEIG universal interactive editing project.

```text
Task ID:
Gate IDs:
Role:
Editable source family:
Exact goal:
Non-goals:
Owned writable files:
Read-only files:
Dependency nodes expected to invalidate:
Nodes expected to remain reusable:
Spatial/temporal scope to verify:
Oracle fixtures:
Required commands:
Dependencies/commits:
```

Read `mission_universal.yaml`, `multi_agent_orchestration_universal.md`, the universal editing documents, and task-specific scientific source files.

Work only in assigned ownership. Reproduce baseline, add characterization/failing tests, implement minimally, run required tiers, compare scientific changes with a fresh full-domain oracle, inspect the diff, make a focused commit, and return the required YAML handoff. Never change tolerance, precision, units, polarity, patch order, temporal state or model scope simply to pass. Never merge or push main.
