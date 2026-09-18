# 근거 목록

모든 repository source는 `AlanSynn/solweig-project@e0d19fc98220a333ea312efe4604780803bbdd17`에 고정되어 있다.
아래의 line 범위는 조회 범위를 나타낸다. 큰 문서의 모든 문장을 완전 감사했다는 뜻은 아니다. 과거 문서와 후속 source/ledger가 충돌하면 최신 구현과 독립 evidence를 재확인한다.
설계 작성 중 SOLWEIG benchmark나 scientific test를 실행하지 않았다. 이 패키지에서 실제 실행한 검사는 standalone 비교기 self-test와 artifact consistency check다.

## [R1] P8 performance summary

경로: `docs/incremental_design_tool/benchmarks/2026-09-02-p8-performance/p8_summary.md`
조회 범위: 본문; dated 2026-09-02
사용 근거: M1 baseline, measured phase costs, UTCI extent sensitivity, existing optimizations
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/benchmarks/2026-09-02-p8-performance/p8_summary.md
```

## [R2] Performance optimization handoff

경로: `docs/incremental_design_tool/benchmarks/2026-09-04-perf-optimization-handoff.md`
조회 범위: 1–500 범위의 연결된 조회; dated 2026-09-04
사용 근거: 37.5–39 s breakdown, read/write fractions, W2 work count, historical hypotheses
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/benchmarks/2026-09-04-perf-optimization-handoff.md
```

## [R3] R6 hot-path and independent review ledger

경로: `docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md`
조회 범위: 206–210
사용 근거: Later evidence: 12% launch estimate, 6-degree ring 66.6%, amplitude escalation, rejected cache, mutation-sensitive oracle anchor
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md
```

## [R4] CPU/GPU and incremental parity verification report

경로: `docs/parity_verification_2026-09-07.md`
조회 범위: 본문 1–230 범위
사용 근거: Recorded same-backend full solve parity, cross-device differences, CPU-only incremental limitation and Linux timings; scratch deletion
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/parity_verification_2026-09-07.md
```

## [R5] Shadow source

경로: `solweig_gpu/shadow.py`
조회 범위: 1–330; shadow()
사용 근거: Exact march state machine, previous-step stop, zero padding, first-step reset, device allocation behavior
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/shadow.py
```

## [R6] Persistent vegetation occlusion state source

경로: `solweig_gpu/incremental/veg_svf_state.py`
조회 범위: 1–380; module contract, march_offsets, regime/clamp helpers
사용 근거: Current R4 Phase A implementation and guards, exact dz association
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/incremental/veg_svf_state.py
```

## [R7] R4 design and corrections

경로: `docs/incremental_design_tool/realtime_collaboration/design/r4-veg-svf-occlusion-state.md`
조회 범위: 1–220 and 240–460; addenda take precedence
사용 근거: Corridor union C, clamp counterexamples, nonbinary vbsh, one-step regime, Phase A performance caveat
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/realtime_collaboration/design/r4-veg-svf-occlusion-state.md
```

## [R8] Incremental solver source

경로: `solweig_gpu/incremental/solver.py`
조회 범위: 1200–1410; _patch_march_window, _sky_patch_geometry, _fold_veg_svf_from_planes
사용 근거: Logical read extent, canonical patch/annulus order, cached exact geometry and scalar corrections
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/incremental/solver.py
```

## [R9] R5 warm-start and restart ledger

경로: `docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md`
조회 범위: 197–205
사용 근거: G2.1/G2.2 implementation, applicability fingerprint, own-next_step checks, reviewed warm/cold parity
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md
```

## [R10] Radiation source

경로: `solweig_gpu/solweig.py`
조회 범위: 1410–1615; define_patch_characteristics
사용 근거: Repeated scalar terms, directional predicates, ordered patch loops, reflection dependency
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/solweig.py
```

## [R11] UTCI source

경로: `solweig_gpu/calculate_utci.py`
조회 범위: 1–95 and 255–400; polynomial and calculator
사용 근거: Long ordered polynomial, masked compaction, log/exp/pow and default constants
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/calculate_utci.py
```

## [R12] Source-family integration and met fast-path review ledger

경로: `docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md`
조회 범위: 70–95 and 140–145
사용 근거: Mixed-family dirty-coverage pitfalls, wind/UHII affinity, sparse time-index scatter defect and remediation
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/agent/SUBAGENT_LEDGER.md
```

## [R13] Pinned main commit metadata and WORKLOG changes

경로: `commit metadata / WORKLOG diff`
조회 범위: GitHub commit endpoint; e0d19fc
사용 근거: Latest incident/provenance status at inspected HEAD; not a claim about a later deployment
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/commit/e0d19fc98220a333ea312efe4604780803bbdd17
```

## [R14] Temporal checkpoint source

경로: `solweig_gpu/incremental/checkpoints.py`
조회 범위: 1–180
사용 근거: Six float32 planes, four scalar slots, input fingerprint, checksums, coverage-only versus thermal checkpoint
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/solweig_gpu/incremental/checkpoints.py
```

## [R15] Existing CPU optimization design

경로: `docs/incremental_design_tool/cpu_optimization.md`
조회 범위: 본문의 CPU/packing/streaming/Numba sections
사용 근거: Prior structural optimizations and historical parity/performance qualifications
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/cpu_optimization.md
```

## [R16] Computation-first optimization roadmap

경로: `docs/incremental_design_tool/realtime_collaboration/optimization_roadmap.md`
조회 범위: 1–230
사용 근거: Existing R0–R8 roadmap; its tolerance/surrogate option is NOT adopted under the new strict contract
확인일: 2026-09-07

```text
https://github.com/AlanSynn/solweig-project/blob/e0d19fc98220a333ea312efe4604780803bbdd17/docs/incremental_design_tool/realtime_collaboration/optimization_roadmap.md
```

## [W1] Numba Performance Tips

사용 근거: Compiled loops, nopython, fast-math and reduction-order caveats; vendor example speedups are not SOLWEIG forecasts.
확인일: 2026-09-07

```text
https://numba.readthedocs.io/en/stable/user/performance-tips.html
```

## [W2] PyTorch Numerical accuracy

사용 근거: Same mathematical expression does not imply cross-device, shape, or release-level bitwise equality.
확인일: 2026-09-07

```text
https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
```

## [W3] CUDA Math API: single-precision intrinsics

사용 근거: Explicit rounding and noncontracting add/mul intrinsics; not a guarantee of matching Torch transcendental behavior.
확인일: 2026-09-07

```text
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
```

## [W4] Numba-CUDA Maintenance Notice

사용 근거: Maintenance-mode notice and Numba-CUDA-MLIR direction as observed on 2026-09-07.
확인일: 2026-09-07

```text
https://nvidia.github.io/numba-cuda/
```

## [W5] CUDA Programming Guide: CUDA Graphs

사용 근거: Graph-based execution and launch optimization; exact design here is a proposed integration.
확인일: 2026-09-07

```text
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
```

## [W6] Using Goals in Codex

사용 근거: /goal command, bounded completion contract and evidence-based success. No claim that this chat has started an autonomous Goal.
확인일: 2026-09-07

```text
https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
```

