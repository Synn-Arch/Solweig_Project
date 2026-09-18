# 구현 task cards

이 파일은 `DESIGN.ko.md`의 실행 순서를 작은 변경으로 나눈다. 모든 task는 초기 상태 `NOT_RUN`이다. 현재 저장소에 제안 module/test/benchmark CLI가 존재한다고 가정하지 않는다. 아래 새 entry point는 T00/T01에서 먼저 구현하고, 그 뒤 동일 명령을 계속 사용한다.

## 공통 작업 규칙

Task 하나가 완료되려면 변경 전 failing/characterization evidence, 변경 후 test log, independent oracle comparison, 필요한 benchmark, reviewer 기록이 있어야 한다. Fast test만 통과한 source는 production default로 바꾸지 않는다. Oracle와 acceptance target은 author의 writable scope 밖이다.

표준 handoff에는 `task_id`, `base_sha`, `candidate_sha`, `owned_files`, `commands`, `source_origin`, `input_hashes`, `gate_status`, `mismatch_count`, `raw_artifacts`, `perf_delta`, `limitations`, `rollback`, `next_command`를 적는다. “테스트 통과”라는 문장만으로는 완료가 아니다.

## T00. 현재 HEAD, 실행 경로, baseline audit

**Depends:** 없음. **Owner:** baseline engineer. **Writable:** 새 `benchmarks/ultrafast/` harness, 별도 worklog/manifest. Scientific source, live state는 read-only.

Current `git status`, HEAD, user changes를 확인한다. 검토 SHA와 달라졌다면 shadow, solver, R4/R5/R6, scheduler, source adapters의 변경을 보고한다. 원본 입력은 read-only로 두고 scratch output root를 분리한다. Active incident의 해결 여부를 확인하고 local regression을 재현한다. Live service를 재시작하지 않는다.

Oracle는 별도 pinned worktree/process다. 각 process가 import한 `solweig_gpu`의 `__file__` 및 source hash를 남긴다. Editable install 또는 `PYTHONPATH` 때문에 두 process가 같은 candidate를 읽지 않도록 한다.

**Fixture:** site_500 대표 tree case, current E1~E4, 40×80 non-square synthetic, 4 m one-step witness, one met warm-start sequence. Real data가 없으면 해당 gate는 BLOCKED다.

**Deliverable:** 환경/input/fixture manifest, 동일 작업 2회 raw repeat comparison, exclusive stage timings, cold/warm 구분, canonical/legacy profile matrix 초안.

**Exit:** Baseline이 자신과 raw-byte equality를 보이는지 확인하고 실패를 숨기지 않는다. Main benchmark 수치는 이전 문서 복사가 아니라 새 실행 결과다.

## T01. 엄격한 bitwise harness와 primitive characterization

**Depends:** T00. **Writable:** `tests/ultrafast/`, `solweig_core/profile.py` 초안, immutable trace exporter. Oracle scientific functions는 변경 금지.

`tools/bitwise_check.py`를 읽고 raw `uint32` comparison을 repository harness에 연결한다. Output dtype/shape/global index/coverage를 검증한다. NaN-aware equality를 acceptance에 사용하지 않는다. Provided checker는 출발점이며 packed integer state, stage trace, profile identity 검사는 별도로 추가한다.

**RED witnesses:** +0 vs -0, 서로 다른 NaN payload, 1-ULP delta, wrong global time index, wrong shape, oracle import-origin mismatch. 동일 NaN payload와 noncontiguous but logically identical view는 통과해야 한다.

Add/sub/mul/div/pow/log/exp/trig/round/max를 scalar/broadcast/dense/compacted context로 실행한다. Vector/tail 경계와 non-power-of-two scale을 포함한다.

**Deliverable:** primitive compatibility matrix, first mismatch report, frozen numeric-profile JSON. `benchmarks/ultrafast/run.py`의 `baseline`, `compare`, `bench`, `report` subcommands를 구현하고 CLI help를 test한다.

**Exit:** Deliberate 1-bit change가 반드시 FAIL이고 tests가 oracle를 변경해 green을 만들 수 없다.

## T02. Torch-neutral buffer ABI와 명시적 device

**Depends:** T01. **Writable:** `solweig_core/{abi,request,status,dispatch}.py`, 기존 solver/worker의 좁은 adapter seam, 새 boundary tests.

Logical domain, physical window, dtype/stride/origin/ownership/residency를 타입으로 분리한다. 기존 tuple contract는 adapter에서 유지한다. 초기 backend는 legacy math로 roundtrip하여 behavior change 없이 seam만 만든다.

Global CUDA availability에 따른 내부 allocation을 새 core에서 금지한다. CPU-target request는 GPU가 보이는 환경에서도 CPU buffer만 사용한다. `.cpu().numpy()` conversion은 boundary에서 한 번만 허용하고 반복 inner loop에는 없음을 검사한다.

**RED witnesses:** CUDA-visible CPU request의 mixed-device allocation, stale borrowed buffer, noncontiguous input을 contiguous로 오인, wrong origin/shape.

**Exit:** 새 adapter를 사용한 CPU legacy result가 original direct path와 byte-identical. Numerical source rewrite와 이 task를 섞지 않는다.

## T03. 정확한 step table과 amplitude policy

**Depends:** T01, T02. **Writable:** `solweig_core/step_tables.py`, `tests/ultrafast/test_step_tables.py`.

기존 `march_offsets`, `_patch_first_step_dz`, R6 amplitude selection을 읽고 executed `dx,dy,dz` 및 stop count를 추출한다. SVF shadow와 wall-height march를 구분한다. Logical domain/angle/scale/amplitude/profile을 key에 넣는다.

**RED witnesses:** 이전 stop 값 대신 새 dz로 cutoff, rows/cols 전도, cardinal quadrant boundary, azimuth-zero handling 혼동, dz association 변경, R6 escalation 제거. Zenith wrapper의 no-march 처리도 확인한다.

첫 table producer는 원본 trace를 사용할 수 있다. 이 단계에서 Torch-free라고 주장하지 않는다. 이후 T14에서 일반 input의 producer까지 대체한다.

**Exit:** 모든 trace fixture의 step count, offsets, dz bits가 원본과 같다. Real-site full output도 동일하다.

## T04. Dense serial Numba shadow kernel

**Depends:** T03. **Writable:** `solweig_core/numba_cpu/march.py`, dense march tests. 기존 `shadow.py`는 read-only.

`bush == 0` guard가 성립하는 domain부터 구현한다. 한 target의 private state가 table의 step을 순서대로 실행한다. State-machine manifest에 각 source expression/rounding boundary를 대응시킨다. `max`와 masked assignment를 단순 if/else로 치환할 때 NaN semantics를 확인한다.

**RED witnesses:** OOB zero→-inf, first-step reset 이동, shifted trunk를 target-local trunk 대신 사용, veg suppression 순서 변경, final 2.0→bool. Negative DSM과 non-square scene을 포함한다.

**Exit:** Per-step trace와 final shadow/veg/vb 값 raw equality. Serial dense case의 correctness만 인증한다. Sparse/parallel/SIMD나 bush 지원까지 완료했다고 하지 않는다.

## T05. Sparse closure와 row-run/CSR builder

**Depends:** T04. **Writable:** `solweig_core/sparse_work.py`, sparse work tests, R4 wrapper의 optional descriptor seam.

Composed pre/post source diffs를 얻는다. `D_p = C ∪ inverse_shift(C, pre_offsets ∪ post_offsets)`의 보수적인 superset을 생성한다. Existing guard를 유지한다. Sorted row runs 또는 chunked bitsets를 prototype하고 memory ceiling을 둔다.

**RED witnesses:** `∪C` 삭제, old footprint 누락, overlap-hidden canopy deletion, smaller-clamped amplitude transition, regime narrowing, OOB/corner, empty changed raster지만 global stop context 변경.

**Exit:** Fresh full oracle에서 변경된 모든 `(cell,patch)`가 closure 안에 있다. Untouched state equality를 매 edit 후 확인한다. Guard가 필요한 case를 단지 test에서 제거하지 않는다.

## T06. Sparse Numba march와 safe CPU parallelism

**Depends:** T04, T05. **Writable:** march/work-list runner, tests, microbench configuration.

Dense kernel과 같은 state-machine body에 sparse targets를 넣는다. Serial sparse를 먼저 통과시키고 chunk/row-run 단위 parallelism을 추가한다. Per-target step 순서를 바꾸지 않는다. Threshold 이하 workload는 serial로 routing한다.

**RED witnesses:** pointer/target duplicate ownership, small work의 thread-overhead regression, target id overflow, cancellation 후 recycled buffer write, physical crop가 logical stop를 바꾸는 경우.

**Exit:** Dense serial ↔ sparse serial ↔ sparse parallel(1/2/4/8 threads) raw equality. Same-work dense/fused vs sparse/fused ablation과 pair-step/bytes counts. Work-list build까지 포함한 total stage latency로 판단한다.

## T07. Direct packed state와 canonical SVF fold

**Depends:** T06. **Writable:** `solweig_core/bitplanes.py`, `numba_cpu/svf_fold.py`, packed/fold tests.

원래 patch-major/annulus-minor recurrence와 directional branch, trunk correction, clamp, svftotal을 그대로 구현한다. Packed input을 직접 소비하고 affected chunks에만 output을 만든다. 동일 packed byte를 여러 threads가 read-modify-write하지 않는다.

**RED witnesses:** all values→bool, tail padding nondeterminism, shared-byte race, pre-summed annulus weights, reordered patch reduction, transVeg/last constant mutation. Fold reference는 candidate fold를 공유하지 않는 original `svf_calculator`여야 한다.

**Exit:** 10개 scalar family, svftotal, packed roundtrip, downstream output의 raw equality. Regime fallback을 유지하고 typed extended encoding은 아직 enable하지 않는다.

## T08. Radiation hoist 및 ordered fusion

**Depends:** T07. **Writable:** `numba_cpu/radiation.py`, reference trace adapter, radiation tests. Existing source는 oracle로 보존한다.

`define_patch_characteristics`의 repeated invariants를 hoist하고 cell-owned ordered loops를 만든다. First sky pass와 reflection dependency를 유지한다. Directions의 strict inequalities를 별도로 고정한다. 이어서 measured bottleneck 순서로 Kside/GVF 관련 stage를 이식하되 다른 march를 재사용하기 전에 별도 manifest를 만든다.

**RED witnesses:** reflection을 incomplete sky sum으로 계산, direction boundary 통합, scalar math context 변경, geometry-dependent sunlit mask를 stale reuse, halo 누락.

**Exit:** Radiation intermediate families와 Tmrt/shadow raw equality. Stage fusion과 full time-loop performance 모두 보고한다. Selected outputs만 비교해서 hidden radiation divergence를 놓치지 않는다.

## T09. UTCI compatibility kernel

**Depends:** T01, T08. **Writable:** `numba_cpu/utci.py`, math compatibility module, typed expression manifest, UTCI tests.

Original AST association과 pow semantics를 보존한다. Primitive matrix에서 불일치한 log/exp/pow를 먼저 해결한다. Compacted vector의 original valid ordinal/context를 유지하는 sparse variant와 dense-compatible variant를 구분한다.

**RED witnesses:** Horner, new FMA contraction, float64 promotion/float32 blanket cast, pow→multiply, mask에서 NaN 추가 제거, scalar saturation-pressure broadcast, odd-length valid compacted vector tail mismatch.

**Exit:** Primitive부터 full UTCI까지 raw equality. 해결되지 않은 primitive는 compatibility fallback으로 남기고 hybrid status를 표시한다. 이를 숨기고 Torch-free gate를 통과시키지 않는다.

## T10. 기존 R5 활용과 exact sparse temporal replay

**Depends:** T08, T09. **Writable:** thermal adapter/kernel, current checkpoint consumer의 좁은 extension, temporal tests.

G2.1의 현재 warm-start path를 먼저 측정한다. Six planes/four scalars 및 next_step semantics를 그대로 유지한다. Fingerprint는 candidate checkpoint의 own next_step에 대해 계산한다. Geometry-history와 met-prefix invalidation을 분리한다.

Spatial state reuse는 upstream radiation/GVF dependency까지 증명한 뒤 추가한다. Full-tile/partial checkpoint contract 변경은 별도 schema migration과 test가 필요하다.

**RED witnesses:** 이전 scene의 noon checkpoint 재사용, Twater/CI 누락, next_step off-by-one, late met edit의 과도한 cold replay, t index scatter 오류, same-revision torn state load.

**Exit:** Warm/cold/fresh-full equality, selected/sparse/full time coverage correctness, 실제 steps_solved 감소. Coverage-only checkpoint를 thermal hit로 세지 않는다.

## T11. Full-domain fallback와 모든 지원 edit family

**Depends:** T07, T08, T09, T10. **Writable:** native/Numba full path, fallback router, integration tests.

초기 bush/one-step/large closure/mixed-family fallback이 Torch를 호출해도 migration은 가능하다. 최종 release를 위해 이 fallback의 정확한 full-domain implementation을 제공한다. Bush global predicates는 original per-step global reduction 의미를 유지한다. Building regeneration의 walls/SVF/land-cover/forcing chain은 별도 input provenance로 실행한다.

**RED witnesses:** unsupported case를 silently no-op, huge work list로 OOM, building edit에서 met/paint overlay 누락, lower-density route가 더 느린데 무조건 sparse, fallback result의 wrong profile.

**Exit:** 기존에 허용된 edit family는 지원을 잃지 않고 exact result를 반환한다. Direct full path와 original full oracle의 모든 requested variable/time bands가 같다.

## T12. CUDA canonical kernels

**Depends:** T03, T04, T07, T08, T09의 명세가 안정됨. CPU task와 병렬 착수 가능하되 integration은 앞 gate에 종속. **Writable:** `native/cuda/`, CUDA tests, build config.

Explicit device/buffer ownership 및 strict arithmetic primitive부터 구현한다. CPU canonical primitive bits와 맞춘 뒤 march/fold/radiation/UTCI를 순서대로 연결한다. Native CUDA legacy profile은 별도 test matrix에 둔다.

**RED witnesses:** contraction, FTZ difference, approximate intrinsic, CPU/GPU library discrepancy, FP reduction/atomic, packed-byte race, original compacted context 손실.

**Exit:** Canonical CUDA ↔ canonical CPU raw equality와 legacy CUDA ↔ original CUDA raw equality를 각각 보고한다. 같은 executable이 profile을 어떻게 구분하는지도 test한다. GPU 미접근이면 BLOCKED이며 emulator만으로 GPU PASS를 선언하지 않는다.

## T13. GPU residency, graphs, dispatch

**Depends:** T12. **Writable:** CUDA runtime/pool/graph cache, dispatch cost model, stress benchmarks.

Site arrays를 resident로 유지하고 changed chunks 및 output만 transfer한다. Graphs는 stable buffer lifetime과 capacity buckets가 확립된 뒤 평가한다. Auto dispatch는 동일 certified profile에 한정한다.

**RED witnesses:** graph가 해제된 pointer 사용, padding targets가 결과에 쓰임, async output 완료 전 publish, cancelled buffer 조기 재사용, different-profile cache hit, CPU-selected request의 CUDA allocation.

**Exit:** Transfer/synchronization/graph build 포함 end-to-end benchmark. Tiny edit는 CPU가 더 빠르면 CPU를 선택한다. Graph-off/on, host-staged/resident ablation 및 leak/stress test를 남긴다.

## T14. CPU AOT 및 진짜 Torch-free runtime

**Depends:** T11, T09 compatibility 완성. **Writable:** native CPU core, packaging/import tree, clean-env tests.

Numba와 같은 manifest를 C++ AOT로 내려 작은 runtime을 만든다. Numba가 이미 모든 목표를 만족하면 native rewrite의 필요성은 evidence로 재평가하되 Torch-free와 lightweight 목표 자체는 유지한다. Step-table 일반 생성, scene compose, fallback, scalar forcing 계산까지 dependency를 audit한다.

Torch, GDAL, plotting, notebook, compiler toolchain을 runtime install에서 제거/분리한다. Cached known-site constants만으로 passing하고 새로운 scale/forcing에서 Torch를 import하는 경로를 남기지 않는다.

**RED witnesses:** transitively imported torch, hidden `.to()` adapter, Torch tensor type annotation import, first unseen scale의 oracle call, packaging extra 누락, ISA mismatch.

**Exit:** Torch 없는 clean environment에서 supported workload matrix 실행. `sys.modules`, dependency graph, installed footprint, RSS, cold/warm startup을 검증한다. Oracle extras는 별도 환경으로 남긴다.

## T15. Publication, mixed edit, liveness

**Depends:** T02 이후 병렬 준비, final integration은 T10/T11/T13. **Writable:** 좁은 scheduler/publication seams, protocol tests.

기존 operation durability/reducer/version contract를 유지하고 backend async completion을 통합한다. Current mixed-plane incident regression을 포함한다. Adaptive epoch/selected-time streaming은 별도 flag와 coverage test로 추가한다.

**RED witnesses:** dense offset time mis-scatter, stale exact publication, accepted operation loss, full-coverage false claim, mixed-family partial no-op wedge, restart 후 pending epoch 영구 정체, slow client가 compute thread를 block.

**Exit:** 중간 transition을 포함한 every accepted sequence의 exact result가 fresh oracle와 같다. Latest-state age와 supersession counts가 보고되며 fast visual response로 wrong scientific state를 숨기지 않는다.

## T16. 비용 모델, 목표 검증, independent release review

**Depends:** T00~T15 해당 mandatory gate. **Writable:** benchmark/report files와 review 기록. Scientific code는 reviewer에게 read-only.

동일 fixtures/config에 대해 factorized ablation과 final workload matrix를 실행한다. Primary CPU 4-core 및 single GPU를 고정하고 cold/warm, fallback, throughput/freshness를 분리한다. `acceptance_targets.yaml`의 목표를 바꾸지 않는다.

**Exit:** 모든 hard gate와 final target의 PASS/FAIL/BLOCKED/TARGET_MISSED 표, raw JSONL, confidence/sample count, memory/startup/dependency report, exact hashes, unverified cases. Non-author reviewer가 최소 critical mutation과 independent full-domain witness를 직접 재실행한다.

목표 미달이면 남은 병목의 exclusive time 및 next candidate를 제시한다. Unavailable data/GPU 또는 budget 종료를 success로 바꾸지 않는다. Release candidate는 verified source commit을 가리켜야 한다.

## T17. 선택적 후속 연구: mandatory 작업 이후에만

2-bit extended vbsh encoding과 더 넓은 regime proof, prefix-accumulator suffix fold, equality-triggered temporal convergence, cache-aware ray-state reuse를 각각 독립 task로 다룬다. 성능 목표 미달을 핑계로 proof 없는 상태 재사용을 넣지 않는다.

각 후보는 `hypothesis → exact equivalence argument → counterexample search → independent oracle test → measured ablation` 순서다. 실패한 후보와 이유도 남긴다. 기본 fallback보다 느리거나 memory 목표를 깨면 채택하지 않는다.

---

## 제안 CLI contract

아래 명령은 **T01에서 구현한 뒤** 사용한다. 현재 저장소의 기존 command라고 가정하지 않는다. Site 입력 위치는 T00 manifest에서 발견·고정한 경로를 읽고 여기에 private path를 하드코딩하지 않는다.

```bash
python benchmarks/ultrafast/run.py baseline \
  --manifest benchmarks/ultrafast/fixtures_manifest.json \
  --profile canonical_cpu_v1 --output "$SOLWEIG_ULTRA_ARTIFACTS/baseline"

python -m pytest tests/ultrafast/test_step_tables.py tests/ultrafast/test_march_dense.py -q

python benchmarks/ultrafast/run.py compare \
  --reference "$SOLWEIG_ULTRA_ARTIFACTS/baseline" \
  --candidate "$SOLWEIG_ULTRA_ARTIFACTS/candidate" --strict-bits --require-full-coverage

python benchmarks/ultrafast/run.py bench \
  --targets docs/incremental_design_tool/ultrafast_bitwise/acceptance_targets.yaml \
  --variant cpu-numba --suite representative --warmup 5 --repeats 100 \
  --output "$SOLWEIG_ULTRA_ARTIFACTS/bench"

python benchmarks/ultrafast/run.py report \
  --artifacts "$SOLWEIG_ULTRA_ARTIFACTS" --require-evidence
```

기존 repository test suite와 frontend test commands는 T00에서 실제 package/config를 확인해 추가한다. 문서에 기록된 옛 test count를 현재 count의 expected value로 강제하지 않는다.
