# Subagent task prompt template

You are a subagent under the SOLWEIG-GPU incremental-design lead.

Assignment:
- Task ID: `<TASK_ID>`
- Phase/gates: `<PHASE_AND_GATES>`
- Role: `<ROLE>`
- Goal: `<ONE_TESTABLE_GOAL>`
- Non-goals: `<NON_GOALS>`
- Owned writable files: `<FILES>`
- Read-only files: `<FILES>`
- Dependencies: `<TASKS_OR_COMMITS>`

Read `agent/mission.yaml`, `agent/multi_agent_orchestration.md`, and the task-specific docs.

Rules:
1. Work only in assigned ownership.
2. Reproduce the supplied baseline first.
3. Add/adjust a failing or characterization test when applicable.
4. Make the minimum correct change.
5. Run required narrow tests and validation tiers.
6. Scientific changes compare against a fresh full-domain oracle from the same commit/environment.
7. Never alter tolerance, precision, units, raster semantics, patch order, temporal state, or model scope just to pass.
8. Keep the branch clean and make a focused commit.
9. Never merge/push main.
10. Stop only for a `mission.yaml` hard stop.

Return:

```yaml
task_id:
status: PASS|FAIL|BLOCKED
branch:
commit_sha:
owned_files_changed: []
commands_run: []
tests:
  passed:
  failed:
metrics: {}
scientific_assumptions: []
risks: []
known_failures: []
recommended_merge_order: []
next_step:
```
