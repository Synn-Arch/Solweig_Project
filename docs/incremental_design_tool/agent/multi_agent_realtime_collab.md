# Multi-agent plan for collaborative real-time optimization

Use 8-12 concurrent agents when isolated work is available. The lead owns integration, the mission gates, shared file ownership, and final main commits.

## Wave R0

- Performance reproduction agent
- Scientific parity auditor
- Queue/state cartographer
- Universal adapter/dependency cartographer
- Frontend collaboration auditor
- Telemetry/schema agent

## Wave R1/R2

- Operation-log and idempotency implementer
- Deterministic reducer/conflict-semantics implementer
- Epoch scheduler/admission implementer
- Fast/exact queue isolation implementer
- Collaboration transport/frontend implementer
- Concurrency/property-test implementer

## Wave R3-R6

- Cheap exact-kernel implementer
- Geometry delta-visibility/SVF implementer
- Temporal checkpoint/sparse-replay implementer
- Kernel profiler/fusion implementer
- Independent scientific reviewer
- Independent differential runner
- Independent performance reproducer

Only one agent writes each scientific-core file at a time. Reviewer and differential runner cannot author the code they verify.

## Wave R7/R8

- Compensation-model/calibration implementer
- Out-of-domain and drift-monitor implementer
- Load/soak/chaos agent
- Frontend exactness/provenance agent
- Scheduler fairness/admission red team
- Scientific compensation red team
- Release auditor

## Required handoff

```yaml
task_id:
branch:
commit_sha:
owned_files_changed: []
status: PASS|FAIL|BLOCKED
commands: []
tests: {}
scientific_metrics: {}
performance_metrics: {}
assumptions: []
risks: []
next_step:
```

A finding without reproducible commands and commit/cache/model/environment metadata is not accepted as performance evidence.
