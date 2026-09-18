# SOLWEIG 초경량 실시간 편집 엔진 설계
## CPU 우선 최적화, CUDA 병행, 엄격한 bitwise parity

- 작성일: 2026-09-07
- 검토 저장소: `AlanSynn/solweig-project`
- 검토 기준: `e0d19fc98220a333ea312efe4604780803bbdd17`
- 문서 상태: 소스 및 저장소에 기록된 실험을 검토한 **구현 설계**. 이 문서 작성 과정에서 SOLWEIG 실행, 실제 site benchmark, CUDA 검증을 새로 수행하지 않았다.
- 수치 구분: **기존 측정**은 저장소 보고서의 수치, **설계 목표**는 앞으로 달성 여부를 측정할 수치, **가설**은 실험으로 채택 여부를 결정할 제안이다.
- 사용 위치: 이 패키지를 `docs/incremental_design_tool/ultrafast_bitwise/`에 놓는 것을 기준으로 한다.
- 함께 읽을 파일: `TASKS.ko.md`, `GOAL_PROMPT.ko.txt`, `acceptance_targets.yaml`, `SOURCES.md`.

## 1. 결정 요약

**Torch를 제거하는 방향에 동의한다. 다만 성능 향상의 단위는 라이브러리 교체가 아니라 계산 범위, 연산 실행 방식, 상태 재사용이다.** 권장 순서는 다음과 같다.

1. 독립적인 full-domain oracle과 raw-bit 검사기를 고정한다.
2. 기존 R4 persistent occlusion state를 유지하고, dense/rectangle replay를 정확한 sparse corridor kernel로 대체한다.
3. SVF fold와 radiation의 patch loop를 cell-owned compiled kernel로 융합한다. 같은 셀 안에서는 기존 연산 순서를 그대로 보존한다.
4. 기존 R5 warm-start와 input fingerprint를 재사용한다. geometry edit에 이전 장면의 thermal state를 잘못 재사용하지 않는다.
5. Numba CPU 구현으로 알고리즘과 parity를 먼저 확립한다. 실제 배포 크기와 startup이 병목이면 같은 명세를 C++ AOT로 내린다.
6. CUDA C++ backend를 같은 명세에서 구현한다. CPU와 같은 값을 내는 canonical profile과 기존 GPU 결과를 보존하는 legacy GPU profile을 구분한다.
7. 모든 reachable fallback까지 대체한 뒤 production runtime에서 Torch와 무거운 preprocessing 의존성을 제거한다.

**최종 권장 스택**은 Python control plane + 작은 typed-array/native compute core + 선택적 CUDA module이다. API, edit log, publication protocol을 성급히 재작성하지 않는다. 수치 kernel을 빠르게 만들기 위해 서버까지 Rust/C++로 바꿀 필요는 현재 근거가 없다.

Numba는 첫 번째 구현 도구로 적합하다. 하지만 Numba, LLVM runtime, JIT cache까지 포함한 배포가 반드시 가장 가볍지는 않다. 따라서 `cpu-numba`는 검증과 빠른 실험에, `cpu-native`는 AOT 배포와 엄격한 compiler control에 각각 역할을 둔다. 성능 차이가 없고 메모리 목표를 충족하면 Numba 경로를 유지해도 된다. [W1]

## 2. 현재 상태: 무엇을 이미 했고, 무엇이 실제 병목인가

### 2.1 서로 다른 날짜의 수치를 섞지 않는다

| 근거 | 환경과 작업 | 저장소에 기록된 결과 | 해석 |
|---|---|---|---|
| P8, 2026-09-02 | M1 Pro, 500×500, 대표 tree edit, 24 steps | p50 54.21 s, p95 59.14 s | 초기 구조 최적화 이후의 과거 baseline |
| W1/W2/W4, 2026-09-04 | M1 Pro native, 선택 시각 `[12]`, 실제로 causal prefix 13 steps | 약 37.5~39 s, SVF 약 20 s, time loop 약 17 s | 당시 phase breakdown |
| R6, 2026-09-05 | 후속 march profile | SVF 16.365 s 중 추정 launch overhead 1.9642 s, 약 12%; 6° ring 66.6% | 과거의 “40% launch overhead” 주장은 이 HEAD에서 재현되지 않음 |
| 검증 보고서, 2026-09-07 | Linux, 4 CPU cores, warm 500×500/24 steps, E1~E4 | incremental 20.4/21.0/21.0/14.0 s | 최신 문서의 CPU incremental 결과. 위 macOS 실험과 직접적인 speedup 비교 금지 |

이 표는 원자료를 다시 실행한 결과가 아니다. commit, edit, output coverage, hardware, cache state가 다르다. 구현 시작 시 **동일한 현재 HEAD, 동일한 입력과 작업**으로 새 baseline을 만들어야 한다. [R1][R2][R3][R4]

### 2.2 현재 incremental exact 경로는 GPU 경로가 아니다

9월 7일 검증 보고서는 batch GPU 실행과 incremental 실행을 구분한다. Incremental harness는 `cuda=False`이고 solver에는 CPU device pin이 있다. CUDA-visible 환경에서 `shadow.py`의 mixed-device 오류도 기록되어 있다. 따라서 “이미 GPU로 실행되는 incremental solver를 CPU로 내린다”는 출발점은 맞지 않는다. 실제 작업은 **현재 CPU incremental 경로의 재설계 + GPU incremental backend 신설 + 기존 GPU batch 호환성 유지**다. [R4][R5]

### 2.3 재구현하면 안 되는 기존 기반

R4 Phase A의 `veg_svf_state.py`에는 packed vegetation occlusion state, pre/post corridor, regime guard, canonical SVF refold가 이미 있다. R5는 checkpoint 저장만 있는 것이 아니라 G2.1 warm-start consumer 및 G2.2 restart 검증까지 후속 ledger에 기록되어 있다. `checkpoints.py` 상단의 일부 설명보다 실제 caller와 최신 ledger를 함께 읽어야 한다. R6의 amplitude escalation도 이미 들어갔다. 이 세 기능을 “새로 만드는 작업”으로 중복 구현하지 않는다. [R6][R7][R8][R9]

### 2.4 가장 중요한 후속 최적화 근거

R6에서는 가장 큰 march가 compute-dominated이고 6° ring이 SVF 시간의 약 2/3다. 따라서 Python dispatch만 제거하는 최적화의 상한을 전체 rewrite의 상한으로 취급해서는 안 된다. 반대로 `@njit`만 붙이면 수십~수백 배 빨라진다고 주장할 근거도 없다.

구별해야 할 효과는 세 가지다.

- **Dispatch 제거:** 같은 cell-step 수를 더 적은 호출로 실행한다.
- **Memory-traffic 제거:** 매 step의 whole-array temporary, zero-fill, mask, tensor pass를 register/local state로 바꾼다.
- **Work elimination:** 실제 영향을 받을 `(cell, patch)`만 march하고 나머지 state를 그대로 재사용한다.

세 효과를 분리한 ablation이 필요하다. R6의 12% 수치만 제거한다는 가정이면 해당 SVF stage의 speedup은 약 `1 / (1 - 0.12) = 1.14×`에 불과하다. 이것은 **dispatch-only 가정의 계산**이지 fused sparse kernel의 예측치가 아니다. [R3]

### 2.5 현재 소스에서 확인된 높은 위험도 항목

`shadow()`는 per-step 2D scratch를 비우고 shifted slice를 채운 뒤 여러 elementwise tensor operation을 실행한다. 함수 내부/도우미에서 GPU availability에 따라 device를 결정하고, 마지막에 `torch.cuda.empty_cache()`도 호출한다. `define_patch_characteristics()`는 다수의 patch별 연산과 반복 scalar/trigonometric 계산을 포함한다. `utci_calculator()`는 boolean compaction 후 `log`, `exp`, `pow`와 긴 다항식 chain을 계산한다. 이들은 각각 서로 다른 방식으로 이식해야 한다. [R5][R10][R11]

## 3. Bitwise parity의 정확한 계약

### 3.1 세 종류의 parity를 혼동하지 않는다

`O_cpu(x)`를 고정된 기존 CPU full-domain oracle, `O_cuda(x)`를 고정된 기존 CUDA oracle이라고 하자. 새 backend를 `N_cpu`, `N_cuda`라고 하면 다음 계약은 서로 다르다.

| 계약 | 의미 | 본 설계의 처리 |
|---|---|---|
| CPU backward parity | `bits(N_cpu(x)) == bits(O_cpu(x))` | 필수 |
| CUDA backward parity | `bits(N_cuda_legacy(x)) == bits(O_cuda(x))` | 기존 CUDA batch 지원을 보존하는 필수 회귀 gate |
| Cross-device parity | `bits(N_cuda_canonical(x)) == bits(N_cpu(x))` | 같은 workspace를 CPU/GPU 사이에서 이동시키기 위한 필수 gate |

저장소 보고서에는 기존 CPU/GPU 사이에서 TMRT 최대 0.368°C, UTCI 최대 0.091°C의 차이가 기록되어 있다. 따라서 이미 서로 다른 두 oracle에 대해 **하나의 결과가 둘 다 같을 수는 없다**. 이것은 부동소수점 구현을 개선하면 풀리는 모순이 아니라 equality의 추이성 문제다. [R4]

**선택:** production exact workspace는 고정 CPU oracle을 기준으로 하는 `canonical_cpu_v1` semantics를 사용한다. CUDA에서도 이 profile을 구현하되 primitive부터 같은 bit를 내는지 검증한다. 기존 GPU 결과를 유지해야 하는 batch 작업은 `legacy_cuda_v1` profile로 분리한다. 이미 존재하는 CPU architecture별 차이가 발견되면 그 legacy profile도 별도로 식별한다.

CPU↔GPU equality가 아직 통과하지 않았다면 canonical workspace의 GPU 실행을 enable하지 않는다. `legacy_cuda_v1`에서만 통과한 결과를 canonical exact로 표기하거나 두 profile의 cache를 섞는 것은 금지한다. CPU fallback으로 동작할 수는 있지만 이를 GPU 최적화 완료로 보고하지 않는다.

### 3.2 Oracle 선택과 불변성

초기 primary CPU 환경은 사용 가능한 고정 Linux CPU 환경 하나로 선택하고 `reference_environment.json`에 정확한 CPU/ISA, OS, dependency build를 기록한다. 9월 7일 보고서의 환경을 그대로 사용할 수 있다면 재현에 유리하지만, 데이터와 실행 파일이 없는데 재현했다고 가정하지 않는다.

Oracle 계층은 다음과 같다.

1. **Independent full-domain physics oracle:** 최적화 후보와 다른 worktree/process에서 실행하는 고정 기존 full solve. authoritative scientific result다.
2. **Legacy incremental snapshot:** 같은 operation sequence에 대한 기존 incremental 결과. 회귀 위치를 찾기 위한 보조 reference다.
3. **Primitive trace oracle:** 원본 함수에서 필요한 cell/patch/step의 입력·중간 결과 bits를 수집한다.

1과 2가 불일치하면 그 차이를 새 구현의 허용 오차로 물려받지 않는다. 기존 버그인지 domain/coverage mismatch인지 최소 재현을 만들고, 독립된 correctness 변경으로 처리한다. 오래된 P8의 1-ULP 기록이 최신 모든 실행에서도 남아 있다고 단정해서도 안 된다. R6에서 해결된 amplitude 문제를 다시 미해결로 취급하지 않는다. [R1][R3][R12]

Oracle 코드, golden data, 비교기를 candidate 구현과 함께 바꿔서 green으로 만드는 것은 금지한다. Oracle pin 변경이 필요한 경우 old/new oracle 차이를 별도 보고하고 명시적인 승인 없이 gate를 교체하지 않는다.

### 3.3 Equality는 raw bits다

Float32 출력의 정의는 **dtype, logical shape, C-order cell sequence, 각 원소의 32-bit pattern**이다. 다음도 구별한다.

- `+0.0`과 `-0.0`
- 서로 다른 NaN payload/sign
- invalid mask와 `-999` sentinel
- sparse patch의 global time index 및 window origin

`np.allclose`, `max_abs == 0`, `np.array_equal(equal_nan=True)`는 이 계약의 합격 판정기가 아니다. 특히 마지막 비교는 signed zero와 NaN payload를 구별하지 않는다. `view(uint32)` 기반 element 비교와 raw-byte digest를 함께 사용한다. 제공한 `tools/bitwise_check.py`는 작은 standalone 시작점이며 repository 전체 correctness harness를 대체하지 않는다.

논리 배열 byte equality와 ZIP/GeoTIFF 파일 전체의 byte equality는 구분한다. timestamp나 container metadata를 바꾼 경우라도 scientific plane parity는 따로 검사한다. 기존 API가 payload checksum까지 계약한다면 그 payload의 encoder version과 endian도 고정해야 한다.

### 3.4 Reference execution context도 입력이다

같은 수식이라도 기존 tensor implementation은 shape, stride, vector/scalar lane, dtype promotion, backend에 따라 결과가 달라질 수 있다. PyTorch도 수학적으로 동등한 연산이 항상 bitwise 같다고 보장하지 않는다. 이 저장소에는 UTCI boolean compaction에 따른 extent-dependent 1-ULP 차이의 과거 기록이 있다. [W2][R1][R11]

다음 항목을 numerical profile에 넣는다.

```text
reference source SHA and source-file hashes
Python / Torch / NumPy versions and build identifiers
compiler, LLVM, math-library identifiers and flags
CPU ISA / vector width / GPU architecture and code artifact hash
rounding mode, FTZ/DAZ behavior, FMA contraction policy
per-operator dtype and scalar-promotion rules
logical full domain, indexing order, relevant tensor layout
compaction order and reference vector/tail dispatch policy
model parameter bits, patch-geometry bits, forcing/input digests
```

기존 scalar constant의 Python float64 계산 결과와 tensor로 cast되는 위치도 기록한다. 소스의 모든 숫자를 일괄 `float32`로 바꾸는 것은 parity 보존 작업이 아니다.

### 3.5 Parallelism의 안전 경계

안전 후보는 **서로 독립적인 셀 또는 `(cell, patch)` state machine을 병렬로 실행하는 것**이다. 같은 셀의 patch sum을 thread별 partial sum으로 나누고 마지막에 합치는 것은 기본적으로 금지한다.

Kernel fusion은 허용한다. 단, 원래 separate tensor op가 만들던 rounding boundary를 유지해야 한다. 예를 들어 원래 `mul` 뒤 `add`였다면 하나의 FMA로 바꾸지 않는다. 원래 math primitive 내부에서 FMA를 사용했다면 그 내부 구현까지 무조건 FMA-off로 바꾸어서도 안 된다. 바깥 expression의 contraction과 primitive 구현 자체를 구분한다.

## 4. 목표, 범위, latency 정의

### 4.1 목표의 지위

다음 값은 아직 달성하지 않은 **공격적인 설계 목표**다. `acceptance_targets.yaml`이 machine-readable source of truth다. 에이전트는 수치를 느슨하게 바꾸거나 “현재 측정값보다 큰 budget”으로 재설정할 수 없다.

| Class | 고정 작업 | 중간 milestone p95 | 최종 목표 p95 |
|---|---|---:|---:|
| UX | local drag geometry feedback, scientific exact claim 없음 | 16.7 ms | 16.7 ms |
| CPU selected | site_500 대표 single-tree edit, 요청 `[12]`, 필요한 causal replay 포함 | 1,000 ms | 100 ms |
| CUDA selected | 동일 edit/coverage, canonical profile, upload/download 포함 | 250 ms | 33.3 ms |
| CPU full-day local | 동일 edit, 24 bands 정확한 결과 | 10,000 ms | 3,000 ms |
| CUDA full-day local | 동일 edit, 24 bands, canonical profile | 3,000 ms | 1,000 ms |
| CPU full fallback | site_500 warm full recompute, 24 bands | 별도 기록 | 15,000 ms |
| CUDA full fallback | 동일 full recompute | 별도 기록 | 5,000 ms |

대표 fixture는 과거 h10/r4 사례를 고정하되 baseline 시 실제 rasterization, exact closure, output coverage를 manifest로 동결한다. CPU 성능 target은 고정된 reference CPU의 물리 코어 4개 budget, GPU는 기록된 단일 NVIDIA device를 기준으로 한다. “4 vCPU면 어느 VM에서나 동일 성능”이라는 주장은 하지 않는다. 1/2 cores, ARM CPU, noisy host는 별도 matrix다.

건물 regeneration, 매우 낮은 sun, 광범위한 met/land-cover edit는 대표 tree SLO에 몰래 제외하거나 섞지 않고 별도 class로 보고한다. 입력 admission 범위 밖을 silent approximation으로 처리하지 않는다. 정확한 fallback이 늦게 끝나는 경우도 오류와 구분해 드러낸다.

### 4.2 측정 구간

`selected exact latency`는 local benchmark에서 **edit의 control-plane 수락부터 exact patch가 client에서 검증·적용 가능한 상태가 되는 시점**까지다. 다음을 포함한다.

```text
admission + epoch/coalescing wait + queue wait + planning
+ rasterization + invalidation + cache/step-table access
+ SVF + all causally necessary thermal steps + requested output
+ packing + encoding + local transport + decode/checksum/application
```

추가로 `compute_only_ms`와 server publication latency를 기록하되 이를 end-to-end target과 바꿔 쓰지 않는다. WAN RTT를 engine 성능으로 보장하지 않는다. Browser geometry feedback이 60 Hz라는 사실은 scientific result가 60 Hz라는 뜻이 아니다.

연속 edit에서 completion된 일부 작업만 표본으로 고르는 생존자 편향을 막기 위해 `latest_scene_age_ms`, accepted epoch 수, superseded target 수, 정체된 epoch 수를 함께 측정한다. 중간 scene을 coalesce하더라도 accepted operation이 유실되어서는 안 된다.

### 4.3 메모리와 배포 목표

최종 AOT CPU worker의 목표는 warm steady RSS ≤256 MiB, representative job peak RSS ≤768 MiB, 20개 warm scenario 관리 시 process peak ≤1.5 GiB다. mmap된 resident page, scratch, checkpoint, code, thread stack을 포함한다. 별도 offline cache builder는 제외하되 그 제외를 보고서에 명시한다. Numba 개발 backend가 이를 초과하면 CPU parity의 실패는 아니지만 lightweight release gate는 미달이다.

GPU 목표는 representative workload에서 runtime-owned live device allocation ≤1 GiB, 20개 scenario에서 ≤2 GiB다. CUDA context/driver 포함 total residency도 별도 보고한다. NVML total과 allocator statistics를 혼동하지 않는다.

Prebuilt site를 사용하는 AOT process-to-ready p95 목표는 CPU 3,000 ms, CUDA 5,000 ms다. Cold first edit, unseen signature/JIT, site-cache build는 별도 전체 시간을 반드시 보고한다. Ready 시점만 앞당기고 첫 요청으로 초기화 비용을 숨기지 않는다.

Torch-free는 “주요 파일에 `import torch`가 없다”가 아니다. Torch가 설치되지 않은 clean environment에서 import, startup, 모든 지원 edit family, fallback, serialization이 실행되어야 한다.

## 5. Architecture와 migration boundary

### 5.1 유지할 control plane

기존 edit validation, canonical operation reduction, scene revision, stale-result fence, result coverage, durable publication은 유지한다. 최신 HEAD에는 mixed legacy/native edit chain으로 exact lane이 정체되는 incident도 기록되어 있다. Kernel 최적화가 이 문제를 해결했다고 간주하지 말고, 현재 시작 HEAD에서 incident의 해결 여부와 regression test를 먼저 확인한다. [R13]

새 numerical engine의 경계는 다음과 같다.

```text
Browser / existing API / canonical operation log
                 |
        dependency planner + revision scheduler
                 |
       Backend-neutral SolveRequest / ArrayView
                 |
      +----------+----------------------+---------------------+
      |                                 |                     |
 cpu-numba (migration)             cpu-native (AOT)       cuda-native
      |                                 |                     |
      +--------- explicit numerical profile + trace schema ---+
                 |
      immutable result patches + coverage + hashes
                 |
      existing atomic publication / client stale fence
```

Oracle는 이 경로 바깥의 pinned environment에 둔다. Candidate가 자신의 결과를 oracle로 재사용하는 구조를 만들지 않는다.

### 5.2 제안 module 배치

아래는 새로 만들 대상이다. 현재 저장소에 이미 존재한다고 주장하는 경로가 아니다.

```text
solweig_core/
  abi.py                 # NumPy/buffer interface; Torch import 금지
  profile.py             # semantics id와 runtime certification
  request.py             # logical domain, physical windows, time coverage
  step_tables.py         # 기존 offset semantics를 확장한 typed table
  sparse_work.py         # CSR/row-run work lists, dependency closure
  bitplanes.py           # binary/extended encoding + unique-writer pack
  numba_cpu/
    march.py
    svf_fold.py
    radiation.py
    thermal.py
    utci.py
  native/
    include/solweig_core.h
    src/...
    cuda/...
  dispatch.py
  status.py              # typed refusal / fallback, no implicit device choice

tests/ultrafast/
  test_primitives.py
  test_step_tables.py
  test_march_dense.py
  test_march_sparse.py
  test_svf_fold.py
  test_radiation.py
  test_utci.py
  test_temporal.py
  test_backend_contract.py
  test_publication.py
  test_no_torch.py

benchmarks/ultrafast/
  run.py                 # 에이전트가 구현할 제안 entry point
  fixtures_manifest.json
  environments/
  reports/
  raw/
```

기존 `incremental/solver.py`와 `worker.py`에 얇은 backend adapter를 추가한다. Phase별 feature flag와 fallback reason을 남긴다. 초기에는 dense legacy compatibility bridge를 허용하되 `.cpu().numpy()`를 patch/step loop 안에서 반복하는 구현은 통합하지 않는다.

### 5.3 Buffer ABI

각 array view에는 pointer 또는 owner reference, dtype, shape, byte strides, global origin, logical domain id, residency, read-only flag가 있어야 한다. Stride/shape를 무시하고 contiguous라고 가정하는 kernel은 boundary에서 복사하거나 명시적으로 거절한다.

`logical_domain`은 reference arithmetic과 boundary rule을 정의한다. `physical_work_window`는 어떤 셀을 실제로 계산하는지 정의한다. 두 값을 분리하지 않으면 window 축소가 ray termination, compaction, global reductions를 바꾸는 버그를 반복한다.

권장 numeric buffer types는 categorical uint8/bool, coordinate int32 또는 범위 검증된 int64, scientific planes float32, 기존 source가 요구하는 scalar float64다. dtype는 성능 선호가 아니라 oracle의 실제 연산별 dtype inventory에 의해 최종 결정한다.

### 5.4 Packaging

`runtime-cpu`에는 core, 작은 array/API 의존성과 lossless codec만 둔다. GDAL, geospatial fetching, plotting, notebook, Torch oracle, compiler toolchain은 각각 `preprocess`, `reference`, `dev`, `cuda` extras 또는 별도 build image로 옮긴다.

Numba cache는 startup에 미리 warm한다. JIT 첫 실행 비용을 warm latency에서 숨기지 않고 cold-start 보고서에 포함한다. Cache key에 ABI, source, compiler, CPU feature/profile을 포함한다. 배포에서 target과 맞지 않는 JIT cache를 재사용하지 않는다.

최종 native core는 wheel에 AOT binary를 넣어 runtime compiler를 요구하지 않는다. ISA 별 wheel 또는 baseline ISA + 검증된 dispatch를 사용한다. 임의 `-march=native` binary를 다른 node에 복사하지 않는다. 원본 코드의 GPL 헤더와 derivative source obligations는 기존 프로젝트 정책에 맞게 보존한다. 별도의 법률적 호환성 판정을 이 설계가 대신하지는 않는다.

## 6. 실행 명세: compiler보다 먼저 고정할 것

### 6.1 Kernel마다 operation manifest를 만든다

각 kernel에 대해 다음을 기록한다.

```text
inputs / outputs / intermediate live state
input validity domain and explicit unsupported cases
ordered operation graph with stable op_id
operand dtype, scalar-promotion point, rounding boundary
comparison semantics: >, >=, ==, NaN, signed zero
branch predicates and their domain: per-cell / per-patch / global
source coordinate map and out-of-bounds value
reduction order, aliasing, mutation order
oracle source symbol and immutable source hash
```

Handwritten NumPy 수식을 먼저 만들고 그 수식을 oracle로 삼는 방식은 피한다. Original primitive의 결과와 직접 비교한다. High-level algebra가 같아도 op graph가 다르면 우선 별도 candidate로 취급한다.

### 6.2 Common strict settings

Numba 시작점은 `njit(cache=True, nogil=True, fastmath=False)`다. 먼저 serial kernel을 통과시키고, 독립 cell ownership이 검증된 뒤 `parallel=True`/`prange`를 추가한다. 이것만으로 Torch bitwise parity가 보장되지는 않는다. 수학 함수 lowering과 dtype promotion을 확인해야 한다. Numba의 fast-math와 reduction parallelization은 연산 순서를 바꿀 수 있다. [W1]

C++에서는 해당 compiler의 strict floating-point 옵션을 고정한다. 예를 들어 GCC/Clang 계열에서 `-fno-fast-math`, `-ffp-contract=off`를 출발점으로 삼고, 실제 assembly와 primitive tests로 효과를 확인한다. `volatile`을 모든 intermediate에 붙이는 방식으로 메모리 왕복을 강제하지 않는다. SSA value의 연산 경계를 compiler가 보존하도록 검증한다.

CUDA에서는 strict arithmetic이 필요한 곳에 `__fadd_rn`, `__fmul_rn` 등 명시적 intrinsic을 사용할 수 있다. 이들은 contraction 제어 수단이지 Torch transcendental library와의 일치 보증이 아니다. `--fmad=false`, `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`는 검토할 기본 build policy이며 PTX/SASS 및 각 primitive의 oracle 결과를 함께 확인한다. [W3]

### 6.3 고정할 constants

Patch angle, annulus weight, model parameter, scalar forcing-derived table은 source expression으로 생성한 **원본 bits**를 저장한다. 사람이 소수 자릿수를 줄여 쓴 JSON float를 다시 parsing해서 original tensor와 같을 것이라고 가정하지 않는다. Binary payload 또는 uint32/uint64 bit pattern과 metadata를 사용한다.

상수 table이 input forcing/scale에 의존하면 그 dependency를 key에 포함한다. 미래의 임의 angle/scale를 지원해야 하는 API에서 알려진 site의 table만 hard-code해서 성능을 달성하는 것은 금지한다. General table generation까지 Torch-free로 대체하는 task를 별도로 완료한다.

## 7. 최우선 kernel: sparse exact shadow march

### 7.1 현재 `shadow()`의 보존해야 할 의미

아래는 구현 시 확인할 source-level contract다. 원본 `shadow()`와 wrapper의 zenith 처리 및 amplitude policy를 함께 기준으로 삼는다. [R5][R6][R3]

| 항목 | 원본 의미 | 잘못된 빠른 구현 |
|---|---|---|
| angle zero | `shadow()`는 azimuth 0에 특수 치환을 한다 | wall-height march에도 같은 치환을 무조건 적용 |
| offset rounding | branch별 `torch.round`와 부호 규칙 | 정수 DDA/Bresenham으로 임의 대체 |
| stop | loop 진입 시 **이전** `dx,dy,dz` 검사 후 새 step 계산 | 새 dz가 amplitude를 넘는 step을 미리 제거 |
| dz | `((ds * index) * (tan(altitude) / scale))`의 실제 dtype/association | `(ds * index * tan(altitude)) / scale`로 재결합 |
| OOB | temporary는 0으로 초기화되고 in-bounds만 `source - dz` | OOB를 `-inf` 또는 `0 - dz`로 처리 |
| building | running max `f` 갱신 후 `sh` 갱신 | vegetation만 보고 building suppression 생략 |
| vegetation | max → building suppression → vb accumulator 순서 | max와 suppression을 교환 |
| first step | first-vegetation 조건, target-local trunk gate, vb reset | first-step을 일반 step과 합치거나 shifted trunk 사용 |
| bush | marched extent 전체에 의존하는 reduction predicate | 각 ray의 bush 값만 보고 같은 함수라고 주장 |
| final output | vb accumulator threshold와 veg subtraction 후 반전 | 모든 output을 무조건 boolean cast |

특히 기존 loop는 마지막 overshoot step을 실행할 수 있다. Step table은 이 **실행된 step set**을 담아야 한다. 실제 값이 대부분 0/1이라는 이유로 float accumulator를 integer로 바꾸는 것도 첫 구현에서는 하지 않는다. 별도의 범위/rounding proof 없이 state representation을 바꾸지 않는다.

### 7.2 Step table

기존 `march_offsets()`를 확장하되 별개의 근사 geometry generator를 만들지 않는다. 제안 구조는 다음과 같다.

```text
StepTableKey:
  semantics_profile, kernel_variant, source_geometry_hash
  angle_input_bits, scale_bits
  logical_rows, logical_cols
  amplitude_policy_id, executed_amplitude_bits
  boundary_policy_id

StepTable:
  count
  dx[count]: int32
  dy[count]: int32
  dz_bits[count]: uint32
  optional_debug: previous_stop_state / branch_id
```

`kernel_variant`는 SVF `shadow`와 `shadowingfunction_wallheight_23`를 구분한다. 이 둘은 이름이 비슷해도 동일 함수가 아니다. Table 생성 후 모든 offset, dz, count를 원본 trace와 byte 비교한다.

**R6 보존:** 실행 amplitude는 단순 `A_eff`가 아니라 현재 reference의 mode/patch/time-dependent amplitude selection 결과다. Banded scene에서 absolute `scene.amaxvalue`로 escalation하는 R6 동작을 유지한다. R4 state path의 기존 안전 guard와 R6 replay path의 escalation을 서로 바꿔 쓰지 않는다. Table producer가 호출 경로별 실제 선택을 기록하고 full-domain oracle로 확인한다. [R3][R6]

일반화 가능한 최적화는 angle/scale/logical shape별 potential step sequence를 먼저 만들고 amplitude별 정확한 prefix length를 선택하는 것이다. 단, 기존 while의 previous-state semantics와 first-step exception을 그대로 반영해야 한다. 단순 `dz <= amplitude` mask는 대체 명세가 아니다.

### 7.3 변경 footprint를 객체 bbox가 아닌 composed surface에서 확정

Tree layer는 여러 canopy와 baseline을 max-combine한다. 겹친 tree를 지웠을 때 아래 canopy가 드러날 수 있다. 따라서 후보 영역은 old/new object footprint의 union으로 얻되, 실제 source 변화는 old/new composed `vegdsm`, `vegdsm2`, `bush`와 필요한 `a` 값을 비교해 확정한다.

```text
C_candidate = union(old_footprints, new_footprints)
C = cells where any march input differs after exact recomposition
```

단순 byte diff는 값 비교보다 보수적인 superset이 될 수 있다. 어떤 equality를 사용하는지 명시하고 signed zero/NaN policy에 일관되게 적용한다. Baseline과 겹친 모든 object contributor를 spatial index로 조회해서 기존 max-composition 순서를 보존한다. “삭제된 tree가 current winner가 아니면 무조건 no-op” 같은 shortcut은 composition의 tie/NaN semantics까지 증명한 뒤에만 쓴다.

### 7.4 정확한 corridor closure

`Δ_p(pre)`와 `Δ_p(post)`를 해당 path가 reference semantics로 실행하는 offset set이라 하자. 기존 R4가 허용하는 regime에서 계산 대상은 다음의 conservative superset이다.

```text
D_p = C ∪ union over d in (Δ_p(pre) ∪ Δ_p(post)) of { t : t + d ∈ C }
```

`C` 자체를 반드시 넣는다. First-step target-local trunk gate 때문이다. Closure를 tile 경계에 clip하되 실제 source read는 logical domain을 유지한다. Changed surface가 없고 모든 관련 global parameter/stop context가 동일할 때만 empty-work shortcut을 허용한다.

**기존 guard는 그대로 남긴다.** 최소한 다음 상황은 첫 sparse release에서 기존 정확한 replay/fallback 경로로 간다.

- pre 또는 post의 bush가 0이 아닌 경우
- amplitude가 변하고, 더 작은 effective-amplitude scene에서 `bound > scene_amaxvalue`인 경우
- packed state의 one-step regime 전환 또는 representability guard가 실패한 경우
- scene/cache/profile identity mismatch, checksum corruption, unknown dependency

`bound`와 amplitude는 전체 composed scene에서 원본 식으로 계산한다. Steady-state clamped scene을 항상 fallback시키는 잘못된 guard를 만들지 않는다. Threshold의 strict `>`와 amplitude-change conjunct를 각각 regression test로 고정한다. R4 proof 및 후속 correction이 명세의 근거다. [R6][R7]

### 7.5 Dense masks 대신 sparse work list

현재 dense shift-OR corridor 생성 자체가 병목이 되지 않도록 두 구현을 비교한다.

**CSR/row runs:** patch별 sorted `(row, c0, c1)` run을 만들고 shift된 run을 union한다. 얇은 corridor와 연속 footprint에 유리하다. 필요한 경우 row run을 sorted cell-id CSR로 펼친다.

**Chunked bitsets:** 64×64 또는 32×32 chunk에 integer bitset을 두고 shifted footprint를 OR한다. Union/popcount는 integer 연산이다. GPU/CPU에서 compaction할 때 deterministic sorted order를 유지한다.

둘 다 실제 성능으로 고른다. Full-domain 153개 dense bool mask를 매 step 새로 만들거나 수천만 pair를 무제한 materialize하지 않는다. `pair_count`, `row_run_count`, estimated bytes에 상한을 두고 high-density이면 dense fused kernel로 routing한다.

### 7.6 CPU kernel ownership

Serial correctness kernel에서 각 target은 private state를 갖고 table의 step을 순서대로 실행한다.

```text
for (patch p, target t) in work_list:
    initialize f, sh, vegsh, vbsh and first-step state EXACTLY as reference
    for s in StepTable[p].executed_order:
        source = t + (dx[s], dy[s])
        load shifted a/canopy/trunk or exact ZERO temporary value
        apply ordered primitive graph for this step
    apply ordered final graph
    write independent output slot
```

이것은 architecture pseudocode다. `max`의 NaN/tie rule, `sh`의 두 masked assignment, signed zero 및 dtype를 생략했으므로 그대로 production code로 옮겨서는 안 된다. 실제 구현은 operation manifest를 따른다.

Fast path에서는 global bush predicate가 항상 false라는 guard가 있으므로 target 간 의존성이 없다. Row-run 안의 cells를 SIMD lanes로 배치하거나 `prange`로 분리할 수 있다. 순서는 **cell 간**에만 바뀌고 한 cell의 step 순서는 바뀌지 않는다.

Sparse가 작은 경우 task scheduling overhead가 지배할 수 있다. Work unit은 개별 cell 하나가 아니라 contiguous row-run 또는 chunk로 묶고, chunk size를 benchmark한다. Nested Torch/OpenMP/Numba thread pool을 동시에 켜지 않는다.

### 7.7 예상 효과를 계산하는 방법

기존 dense 작업량의 근사는 다음과 같다.

```text
dense_cell_steps = Σ_p area(march_window[p]) × executed_steps[p]
sparse_cell_steps = Σ_p |D_p| × executed_steps[p]
```

이 값은 비교 가능한 structural metric이지만 그대로 latency가 되지는 않는다. Sparse gather locality, register pressure, work-list 생성, fold, serialization 비용을 별도로 측정한다.

모델은 `T ≈ a·pair_steps + b·bytes_touched + c·work_items + d`와 같이 fitted term으로 만들고 held-out edit에 예측 오차를 보고한다. 초당 연산 수를 임의 가정하여 sub-millisecond 성능을 보장하지 않는다.

## 8. Packed state와 canonical SVF fold

### 8.1 기존 binary 가정을 일반화하지 않는다

500×500×153 float32 stack은 153,000,000 bytes, 약 145.9 MiB다. Binary stack이면 cell당 20 bytes로 약 4.77 MiB가 된다. 그러나 `vbsh`는 특정 one-step 조건에서 2.0이 될 수 있다는 실제 counterexample이 존재한다. 현재 R4는 관련 regime을 거절해 binary state의 전제를 지킨다. [R6][R7]

첫 구현은 이 guard를 보존한다. 이후 `uint8` 또는 2-bit extended state를 추가하는 것은 별도 task다. `{0,1,2}` 전제가 모든 허용 input에 성립하는지도 먼저 증명/검증한다. 모르는 값은 bit truncation하지 않고 typed refusal로 보낸다.

**2-bit로 2.0을 저장할 수 있다는 사실은 corridor proof를 자동으로 확장하지 않는다.** Amplitude/regime 변화가 untouched ray에 새 값을 만들 수 있으므로 closure와 invalidation proof도 별도로 갱신해야 한다.

### 8.2 Packed-byte write race

다른 patch의 bit라도 같은 cell의 같은 byte를 공유한다. 아래 구현은 race다.

```text
parallel over p:
    packed[cell, p // 8] = modify_bit(packed[cell, p // 8], p % 8)
```

대안은 다음 둘 중 하나다.

- `(cell, byte_group)`를 한 thread가 독점하고 8개 patch 결과를 조립한다.
- 독립 uint8 result buffer에 결과를 쓴 후 unique-owner pack stage에서 합친다.

불필요한 full dense cube 대신 affected chunk만 uint8 staging한다. Integer atomic을 사용하더라도 set/clear와 concurrent revision의 순서를 명확히 정의해야 한다. 첫 구현은 unique-owner 방식으로 단순화한다.

Unused tail bits는 항상 같은 값으로 초기화한다. Serialization/hash가 padding byte까지 포함한다면 padding도 deterministic해야 한다. Abort된 edit의 dirty chunk가 committed state에 섞이지 않도록 COW 또는 transactional staging을 사용한다.

### 8.3 Fold는 ordered recurrence다

기존 `_fold_veg_svf_from_planes`는 patch 순서와 annulus-weight 순서에 따라 10개 accumulator를 갱신한다. 이 recurrence를 같은 cell 안에 유지한다. [R8]

```text
for cell in consumed_dirty_cells:      # 이 축만 병렬화
    initialize ten accumulators as reference
    for ring in canonical_ring_order:
        for patch in canonical_patch_order_within_ring:
            v, b = decode exact values
            for annulus in canonical_annulus_order:
                perform each original multiply then add
                apply original directional branch predicates
    apply trunk-absent correction, clamps, svftotal in original order
```

금지하는 변환은 annulus weights 사전 합산, patch 순서 정렬 변경, `dot`/GEMM 대체, tree별 float contribution delta, `old_sum - old_term + new_term`이다. 이들 대부분은 수학적으로 같아도 rounded recurrence가 다르다.

한 cell의 모든 입력 bits가 같다면 **그 cell의 완성된 scalar bundle 전체를 재사용**하는 것은 후보가 된다. 이는 float delta update와 다르다. 완전한 dependency fingerprint가 일치하는 경우에만 whole-cell cache hit로 처리한다.

### 8.4 Unpack하지 않고 소비한다

Fold와 radiation kernel이 packed value를 직접 읽게 만든다. Memory layout은 CPU에서 cell-major 20-byte stack과 AoSoA chunk를, GPU에서 patch-major work-list/warp coalescing을 비교한다. Layout 변환 비용도 benchmark에 포함한다.

기존 downstream API 때문에 매 solve마다 3개의 float32 cube를 전부 복원하면 packing의 이점이 크게 사라진다. Migration 중에는 adapter에서 consumed chunk만 unpack하고, 최종 backend는 direct packed input을 받는다.

## 9. Radiation loop fusion

### 9.1 목표 함수

우선순위는 `define_patch_characteristics`, 그 안의 repeated patch arithmetic, 뒤이어 `Kside_veg_v2022a`, `gvf_2018a`, `sunonsurface_2018a` 및 direct-sun march의 현재 실제 profile 순서다. 과거 inclusive `cumtime`에서 parent와 child를 더해 총 비용으로 보고하지 않는다. [R1][R10]

### 9.2 Hoist 가능한 것과 아닌 것

Patch geometry에만 의존하는 trig/steradian/direction predicate는 reference bits로 미리 생성할 수 있다. 동일 timestep의 `Ta`, `ewall`, `SBC`로 반복 계산되는 shaded/vegetation surface term도 입력 shape와 dtype context를 보존한다면 한 번 계산해 재사용할 후보가 된다.

`Tgwall`, `asvf`, geometry masks, solar position에 의존하는 term은 그 dependency가 바뀌는 범위에서만 재사용한다. “angle은 고정이므로 sunlit/shaded도 고정”이라고 단순화하지 않는다. Cardinal direction boundary는 SVF fold와 radiation에서 부등호가 다르므로 공용 helper로 합치기 전에 각 truth table을 고정한다. [R10]

### 9.3 Cell-owned fusion

각 cell에 대해 patch loop를 순서대로 실행하며 필요한 radiation accumulator를 register/local variable에 둔다. Output에 필요 없는 whole-array temporary는 생성하지 않는다. Source가 만드는 각 multiply/add의 rounding boundary는 유지한다.

`define_patch_characteristics`에는 sky sum이 완료된 뒤 reflection을 계산하는 후속 loop가 있다. 첫 loop의 final `Ldown_sky`가 다음 단계의 입력이므로 dependency barrier를 유지한다. 두 loop를 임의로 한 번의 patch traversal로 합치지 않는다. 한 cell의 첫 loop를 끝내고 그 cell의 reflection loop를 실행하는 구조는 다른 셀과의 의존성이 없다는 audit 후에 가능하다.

GPU에서는 모든 term을 거대한 하나의 kernel로 융합하면 register spill이 생길 수 있다. Ordered first pass, reflection pass, final combine 정도의 coarse kernels와 monolithic kernel을 비교한다. Launch 수가 적다는 이유만으로 더 빠르다고 판정하지 않는다.

### 9.4 GVF와 direct-sun march는 별도 proof

SVF `shadow`의 sparse kernel을 다른 march에 그대로 적용하지 않는다. Wall direction, wall height, spatial extent, thermal field, global reductions, `firstdaytime` 등의 dependency가 다르다. 기존 exact read window와 6° floor는 유지한다. 새로운 sparse target execution이 같은 logical domain의 source를 읽는지 독립적으로 검증한다.

한 cell의 radiation이 바뀐다는 사실과 그 cell만 다시 계산하면 된다는 사실은 다르다. GVF와 주변 surface terms가 읽는 halo까지 dirty dependency closure에 포함해야 한다.

## 10. UTCI: 가장 늦게 단순화할 수치 경계

### 10.1 왜 `fastmath=False`만으로 해결되지 않는가

현재 함수는 valid mask로 Ta/RH/Tmrt/wind를 compact하고, log/exp/pow와 긴 다항식을 계산한다. Torch와 Numba의 primitive 구현, scalar promotion, vector width/tail 처리, pow lowering이 같다는 보장이 없다. 기존 문서에서도 output extent를 줄였을 때 1-ULP 차이가 생긴 사례가 있다. [R11][R1][W2]

따라서 “원소별 함수이므로 dirty pixels만 gather해서 같은 식을 실행하면 bitwise 같다”는 추론은 금지한다. 먼저 원본의 actual primitive behavior를 측정한다.

### 10.2 Primitive characterization

다음 matrix를 `test_primitives.py`와 immutable trace fixtures로 만든다.

```text
operators: add/sub/mul/div, comparisons, max/min, round,
           pow integer/float exponents, log/exp, sin/cos/tan, sqrt
shapes: 0,1,2,7,8,9,15,16,17,31,32,33,63,64,65 and real extents
layouts: contiguous, sliced, transpose where accepted, shifted alignment
values: domain boundaries, nextafter neighbors, signed zeros,
        subnormals, valid sentinels, NaN/infinity where source accepts them
contexts: scalar tensor / broadcast scalar / dense tensor / compacted tensor
backends: pinned CPU oracle, candidate CPU serial/SIMD, legacy CUDA, canonical CUDA
```

NaN을 invalid_mask에 새로 넣는 것처럼 원본과 다른 validation을 추가하지 않는다. 원본 함수의 `<= -999` 조건에서 NaN이 어떻게 경로를 타는지도 보존하거나 별도 API validation change로 분리한다.

### 10.3 Ordered expression IR

다항식은 기존 expression AST의 association을 그대로 읽어 typed SSA graph로 내린다. Each node에는 `op_id`, dtype, operands, constant bits, primitive implementation id가 있다. 같은 IR에서 Numba reference-style kernel과 C++/CUDA implementation을 만들고, graph manifest를 review한다.

허용되는 CSE는 **동일 input bits, 동일 primitive, 동일 execution context**를 가진 동일 subexpression의 결과 재사용뿐이다. `Ta**4`를 `(Ta*Ta)*(Ta*Ta)`로 바꾸거나 6차식을 Horner form으로 바꾸는 것은 자동 허용이 아니다. 원본 pow의 특수 exponent lowering도 검사한다.

### 10.4 Transcendental compatibility strategy

순서는 다음과 같다.

1. 원본 ATen dispatch와 실제 math library/build를 기록한다. SLEEF, libm, vector implementation 중 무엇이 사용되는지 추측하지 말고 확인한다.
2. 해당 primitive의 arithmetic behavior를 작은 compatibility layer에서 재현한다. 기존 coefficient table과 internal polynomial을 옮기는 경우 licensing/source provenance를 보존한다.
3. Compaction-induced behavior가 발견되면 reference의 global valid ordinal 및 vector/tail context를 metadata로 유지한다. Selected sparse cells가 원래 dense evaluation에서 어떤 lane/kernel path였는지 재현한다.
4. 동일 timestep의 forcing-only term을 precompute할 때도 원본 context와 같은 bit가 나오는지 검사한다. Scalar 한 번 계산해 전 셀에 broadcast하면 달라질 수 있다.
5. 아직 exact primitive가 없는 동안은 해당 stage만 reference-compatible implementation으로 유지한다. 이 상태의 release는 `hybrid`, Torch-free가 아니다.

Correctly rounded 새 math library가 원본의 approximate result보다 정확하더라도 backward bitwise parity를 자동으로 만족하지 않는다. 더 정확한 값으로 바꾸는 일은 별도 model/numerical version이다.

### 10.5 Sparse UTCI의 조건부 활성화

Global mask/ordinal을 그대로 만들고 active cells만 실행하는 variant를 검증한다. Sparse gather가 original lane dispatch를 바꾸는 구간은 dense compatibility stage로 routing한다. 실제 매력은 전체 radiation/SVF를 생략할 수 있는 wind/UHII-only family 등에서 나온다. 해당 fast path의 source affinity와 sparse global time scatter는 이미 기존 구현과 review에 기록되어 있으므로 그대로 보존한다. [R12]

## 11. Temporal causality와 checkpoint 활용

### 11.1 이미 있는 기능을 확장한다

R5 G2.1은 warm-start plumbing 및 met-radiation consumer, G2.2는 restart 검증까지 기록되어 있다. 먼저 현재 caller가 실제로 warm route를 타는지 metrics와 tests로 확인한다. Coverage-only checkpoint의 존재를 thermal reuse 완료라고 해석해서는 안 된다. [R9]

Thermal state는 6개 planes `Tgmap1`, `Tgmap1E`, `Tgmap1S`, `Tgmap1W`, `Tgmap1N`, `TgOut1` 및 `CI`, `firstdaytime`, `timeadd`, `Twater` scalar를 포함한다. `next_step`는 이미 계산한 마지막 step이 아니라 **다음 실행 step**이다. [R14]

### 11.2 Geometry edit에서 정오 state를 재사용하면 안 되는 이유

장면 geometry가 하루 전체에 적용되는 edit라면, 정오 heat output뿐 아니라 아침부터의 surface thermal history도 달라질 수 있다. 이전 장면의 정오 checkpoint는 “시각이 같다”는 이유로 유효하지 않다.

반면 t=20 이후 meteorological row만 바뀌고 t<20의 inputs가 bitwise 같다면 prefix state 재사용이 가능하다. Candidate checkpoint의 **자기 `next_step`에 대한** met-prefix fingerprint를 계산해야 한다. Current request의 다른 prefix length로 검사하면 false hit 또는 false miss가 생긴다. [R9][R14]

### 11.3 확장할 spatial checkpoint 정책

가설: source/dependency closure가 고정된 뒤 dirty cells의 causal chain만 replay하고 untouched cells의 state를 COW로 유지한다. 적용 조건은 thermal recurrence가 cell-local이라는 사실만이 아니라 upstream radiation/GVF input history도 unchanged라는 것이다.

- Scene geometry가 바뀐 cell 및 그 영향을 받는 radiation/GVF dependency closure를 구한다.
- 해당 cells에는 earliest dirty time부터 recurrence를 다시 실행한다.
- Global scalar forcing/day-boundary dependency가 바뀌면 관련 cells 전체로 invalidation을 확대한다.
- Unchanged cells의 checkpoint bytes를 그대로 유지한다.
- New checkpoint는 publication과 동일 revision lineage를 가져야 한다.

기존 full-tile checkpoint 계약을 변경해야 하므로 초기 kernel port와 분리한다. 여섯 float32 full-tile planes는 500×500에서 checkpoint당 6 MB다. 24개를 무조건 복제하면 144 MB/scenario가 된다. Sparse/COW checkpoints의 수, anchor 위치, retention은 memory budget과 warm-hit telemetry로 결정한다. 모든 시각에 dense checkpoint를 저장하지 않는다.

### 11.4 Partial time coverage

Selected time `[12]` 요청에서 causal prefix 13 steps가 필요할 수 있다. 이때 `[12]`만 출력하는 것과 13 bands를 계산하는 것과 24 bands를 publish하는 것은 서로 다르다.

Manifest는 `(variable, global time index, spatial coverage, revision, profile)`을 명시해야 한다. 1-band sparse patch를 배열의 index 0이라는 이유로 global t=0에 넣지 않는다. 기존 review에서 이 class의 실제 served/durable mismatch가 발견되고 수정되었다. [R12]

### 11.5 Equality-triggered early convergence

추가 가설: dirty recurrence가 어떤 step에서 reference future inputs가 동일하고 carried state의 모든 bits도 이전 committed state와 같아지면, 이후 suffix를 그대로 재사용할 수 있다. Equality는 여섯 planes와 네 scalar 및 future-input identity를 모두 포함한다. 온도 차이가 작다는 이유나 한 output의 equality만으로 조기 종료하지 않는다. 구현 우선순위는 기본 sparse/fusion보다 낮다.

## 12. CPU backend 세부 전략

### 12.1 Port 순서

`step tables → dense serial march → sparse serial march → parallel march → canonical fold → radiation → UTCI → temporal integration` 순서를 지킨다. Dense serial port는 최종 성능을 위한 것이 아니라 GPU/Numba/C++가 공유할 exact state-machine reference를 만드는 단계다.

첫 Numba kernel에 Torch object, Python dictionary, exception-heavy control flow를 넣지 않는다. Python boundary에서 typed arrays를 검증하고 kernel은 fixed-signature buffer만 받는다. Object mode를 허용해 놓고 native code라고 보고하지 않는다.

### 12.2 Allocation과 memory traffic

Permanent site arrays는 read-only mmap 또는 resident buffers로 공유한다. Per-worker arena는 cell/chunk/timestep별 scratch를 재사용한다. Scalar-only stage에서 full H×W constant tensor를 만들지 않는다. 전체 scene 재조합이 필요하지 않은 edits는 object overlap closure에서만 patch-compose한다.

단, reference가 tensor shape-dependent arithmetic을 수행하는 stage는 virtual logical layout을 유지한다. Memory allocation을 줄이는 것과 numerical shape semantics를 바꾸는 것을 구분한다.

### 12.3 Threading

Control plane, fast exact request, long exact work에 합의된 CPU budget을 둔다. 실제 host가 4 cores라면 각 library가 4 threads씩 생성하지 않도록 단일 ownership policy를 둔다. Worker별 affinity, thread count, SMT, NUMA placement를 report에 기록한다.

Candidate test matrix는 serial, 2, 4, 8 threads이며 raw bits가 같아야 한다. Threshold 이하 sparse work는 serial로 실행한다. R6에서 8 threads가 1 thread보다 느린 큰 march 사례도 기록되어 있으므로 core 수에 비례한 speedup을 가정하지 않는다. [R3]

### 12.4 Numba 유지 대 C++ AOT 전환의 기준

Numba가 correctness/latency/memory/startup target을 만족하면 먼저 채택한다. C++ 전환은 다음 중 하나가 측정되었을 때 우선한다.

- JIT compiler/runtime residency가 lightweight target을 막는다.
- 필요한 reference arithmetic primitive를 Numba lowering으로 정확히 제어하기 어렵다.
- SIMD layout 또는 thread scheduling에서 native 구현의 재현 가능한 이점이 있다.
- CUDA와 동일 generated core를 유지하는 편이 duplicated implementation보다 검증 가능하다.

C++가 자동으로 더 빠르다는 가정으로 전체 module을 한 번에 번역하지 않는다. Numba와 native를 동일 ABI/trace/fixture에 놓고 비교한다.

## 13. GPU backend와 CPU 공통 semantics

### 13.1 CUDA C++를 production 우선안으로 선택

Numba CPU와 CUDA를 같은 Python syntax로 쓰면 구현이 쉬울 수 있지만, 현재 NVIDIA Numba-CUDA 문서는 maintenance mode와 Numba-CUDA-MLIR로의 후속 개발 방향을 명시한다. 장기 production GPU core는 명시적인 rounding, memory ownership, generated code inspection이 가능한 CUDA C++를 우선안으로 둔다. Numba-CUDA 또는 MLIR 경로는 비교 후보이며 deprecated namespace를 무심코 장기 기반으로 고정하지 않는다. [W4]

이는 MLIR이 느리거나 부정확하다는 판정이 아니다. 이 프로젝트의 핵심 요구인 legacy bitwise semantics를 가장 좁은 implementation surface로 통제하기 위한 설계 선택이다.

### 13.2 Kernel mapping

March에서는 patch별 sorted target work list를 사용하고 warp의 인접 lanes가 인접 target cell을 읽도록 한다. 각 thread는 한 ray의 step sequence를 순서대로 실행한다. Branch divergence와 long 6° rays 때문에 workload를 ring/step-count bucket으로 나눌 수 있지만 **각 ray의 실행 step을 바꾸지 않는다**.

SVF/radiation에서는 thread가 cell을 소유하고 ordered patch loop를 실행한다. FP atomics, warp tree reduction으로 canonical sum을 대체하지 않는다. Thread block의 execution order는 output ownership이 독립적이면 결과에 영향을 주지 않아야 한다.

### 13.3 Persistent residency

Static site, packed building state, patch geometry는 device에 유지한다. Edit마다 changed raster chunk와 sparse work descriptor만 올리고 필요한 output chunk만 내려받는다. Site 전체, full-day cube, dense constant arrays를 매 요청 upload하지 않는다.

Pinned host staging buffer와 async copies는 bounded pool로 관리한다. Device buffer free/alloc 및 `empty_cache()`를 per-patch hot path에서 호출하지 않는다. Async copy가 완료되기 전에 revision을 publish하지 않으며 cancellation 후 buffer를 다른 job에 재사용할 때 completion event를 확인한다.

### 13.4 CUDA Graphs는 마지막 launch 최적화

Work buffers와 kernel DAG가 안정된 뒤 graph capture/replay를 평가한다. Sparse capacity buckets와 active count로 buffer address를 고정할 수 있다. Padding target은 읽기/쓰기 모두 mask하고 original logical extent나 UTCI compaction context를 바꾸지 않는다.

Graph cache는 device/context, profile, kernel binary, capacity/layout, site residency generation과 parameter lifetime을 key로 가져야 한다. Graph가 살아 있는 동안 참조 buffer가 해제되지 않도록 한다. Graphs는 launch overhead를 줄이는 수단이며 계산량·transfer·parity 문제를 해결하는 수단은 아니다. [W5]

### 13.5 Canonical CPU arithmetic을 GPU에서 재현

기본 add/mul/div/sqrt는 explicit operation boundary와 rounding policy를 검사한다. Transcendental과 NaN payload 처리, float64 scalar path는 별도 compatibility implementation이 필요할 수 있다. CPU용 table의 exact bits를 device로 복사할 수 있는 invariants는 그렇게 한다.

전체 finite domain에 대한 모든 legacy primitive의 일반적인 equality를 테스트만으로 증명했다고 말하지 않는다. Domain restrictions, primitive implementation reasoning, exhaustive 가능한 subdomain, randomized/adversarial test, supported profile 범위를 함께 명시한다. Profile 밖의 hardware/math mode는 uncertified다.

### 13.6 Device dispatch

Auto dispatch는 **동일 canonical semantics가 인증된 backend 사이에서만** 가능하다. 비용 모델은 다음을 비교한다.

```text
T_cpu = queue_cpu + compose_cpu + work_cpu + encode_cpu
T_gpu = queue_gpu + staging + H2D + kernels + D2H + encode_cpu
```

작은 edit에서는 CPU가 빠를 수 있다. GPU selection은 device availability만 보고 결정하지 않는다. Certification 실패, unavailable device, memory pressure는 explicit CPU fallback reason으로 남긴다. Legacy GPU profile의 buffer나 checkpoint를 canonical profile로 암묵 재사용하지 않는다.

## 14. Dependency cache, invalidation, concurrency

### 14.1 Cache key는 결과를 결정하는 모든 입력을 포함한다

R6에서 `(revision, timestep)` march cache는 입력 순수성 조건을 만족하지 못했고, 실제 반복 edit에서는 hit도 제한적이라는 이유로 거절되었다. 같은 실수를 반복하지 않는다. [R3]

Stage cache의 공통 key는 다음을 포함한다.

```text
semantics profile + kernel/source version + ABI/schema version
site id + immutable baseline/cache manifest hash
composed input chunk hashes and dependency closure identity
model/forcing/geometry exact bits or stable digests
logical domain + boundary policy + physical layout when numerically relevant
amplitude selection and exact stop/table identity
requested variable/time coverage where relevant
```

Revision은 lineage를 검증하는 용도이지 numerical input fingerprint를 대신하지 않는다. Backend binary hash는 certification artifact에 기록한다. 같은 canonical profile 사이에서 cache 공유를 허용하려면 그 profile에 대한 equality가 인증되어 있어야 한다.

### 14.2 Cache invalidation matrix

| Edit family | 우선 재사용 | 반드시 다시 검토할 dependency |
|---|---|---|
| View/time selection only | 존재하는 정확한 result bytes | requested coverage, profile, revision |
| Wind/UHII-only met | 검증된 기존 affinity에 따라 radiation/Tmrt | 해당 time의 UTCI와 validity/compaction context |
| Radiation-affecting met | unchanged geometry/SVF | earliest dirty time, thermal prefix, global day state |
| Tree add/move/resize/delete | immutable building state, untouched visibility | composed old/new surfaces, closure, amplitude/regime, thermal history |
| Land-cover paint | 영향 없는 geometry visibility | model material parameters, GVF/thermal dependency, planner stroke coverage |
| Building edit | baseline 외의 unaffected input | building/wall/SVF regeneration chain, scenario-specific manifest |
| Model parameters | source affinity가 증명된 stage만 | globally used constants, output coverage, profile/model version |
| Reset/undo/replay | 동일 input fingerprint의 exact bytes | accepted op order, lineage, no stale state resurrection |

각 row는 구현 전 현재 `edit_registry`와 adapters의 실제 source affinity를 audit한다. 표를 이유로 지원하지 않던 edit를 조용히 no-op 처리하지 않는다.

### 14.3 Transaction boundary

Immutable baseline과 committed scenario state를 나누고, edit 결과는 staging state에 작성한다. Publish 직전에 scene revision, accepted watermark, numerical profile, coverage, buffer completion을 재검사한다. 성공 시 numerical state와 output manifest가 같은 lineage로 보인다.

Cancelled/superseded job은 buffer를 안전하게 정리하되 newer committed state를 덮어쓰지 않는다. 같은 byte를 다른 jobs가 쓰지 않는다. Crash 시 partial state가 valid checksum이나 latest revision처럼 보이지 않도록 atomic rename/commit discipline을 유지한다.

### 14.4 No-op와 coalescing

연속 pointer events는 canonical scene transition으로 coalesce할 수 있다. 그러나 accepted operation의 effect와 required plan coverage는 모두 보존해야 한다. Whole-batch value equality와 mixed-family batch 안의 한 source가 같다는 것은 다르다. 기존 ledger에는 value-diff만 dirty로 잡았다가 planner stroke coverage를 누락해 publication이 실패한 사례가 있다. [R12][R13]

Single-user latency를 위해 기존 100 ms epoch/debounce를 줄이는 방안은 별도 scheduling task로 평가한다. Compute target이 100 ms인데 대기 자체가 100 ms이면 target에 도달할 수 없다. Epoch를 adaptive하게 줄이더라도 reducer determinism, idempotency, conflict resolution은 변하지 않아야 한다. 빠른 visual movement를 exact scientific publication으로 오인시키지 않는다.

## 15. UI와 transport: 속도를 숨기지 말고 측정한다

Drag geometry는 즉시 렌더한다. Exact result가 오기 전에는 이전 heatmap의 revision을 명시하거나 pending 상태를 표시한다. 근사 열장, downsampled physics, stale result를 exact로 표기하지 않는다. 이 설계의 optimization success는 exact bytes에 대해서만 판단한다.

Transport는 요청한 variable/time과 실제 dirty rectangle 또는 chunk만 보낸다. Float32를 float16으로 바꾸지 않는다. Lossless compression과 byte shuffle은 후보지만 compression/decompression 시간을 포함해 identity encoding과 비교한다. Browser codec 지원은 runtime feature detection으로 확인하고, 지원되지 않는 codec을 강제로 보내지 않는다.

Client decode/checksum은 worker에서 처리하고 GPU texture는 dirty region만 갱신한다. Full texture를 frame마다 재업로드하지 않는다. Time-index scatter, window origin, schema/profile id, checksum, revision을 적용 전에 검사한다. 이미 적용한 newer exact revision보다 오래된 patch는 버린다.

Large full-day result는 selected-time result의 publication을 막지 않도록 coverage-aware streaming을 고려한다. 단, partial series를 full coverage로 보고하면 안 된다. Client와 server가 각 `(variable,t,window)`의 coverage를 정확히 합성할 수 있을 때만 enable한다.

## 16. Benchmark 설계

### 16.1 재현 가능한 baseline부터 만든다

P0에서 다음을 저장한다.

```text
baseline/commit.txt
baseline/reference_environment.json
baseline/input_manifest.json
baseline/fixture_manifest.json
baseline/commands.json
baseline/raw_runs.jsonl
baseline/stage_samples.jsonl
baseline/parity_summary.json
baseline/known_failures.json
```

Source, test, fixture generator, reproducibility scripts는 repository에 남긴다. Private input raster와 대형 golden planes는 승인된 local artifact root에 두고 digest 및 재생 명령만 버전 관리한다. 기존 9월 7일 scratch benchmark 파일은 삭제되었다고 보고되어 있으므로 ephemeral `/tmp` 경로를 재현 수단으로만 남기지 않는다. [R4]

### 16.2 필수 ablation

| Variant | 바꾸는 것 | 답할 질문 |
|---|---|---|
| B0 | 고정 현재 reference | 실제 현재 end-to-end baseline인가? |
| B1 | 동일 dense work, Numba fused march | 호출/temporary 제거 효과는 얼마인가? |
| B2 | 동일 sparse work list, legacy arithmetic | work elimination만의 효과는 얼마인가? |
| B3 | sparse + compiled march/fold | interaction을 포함한 실제 이득은? |
| B4 | compiled radiation/UTCI 추가 | time loop가 새 병목인가? |
| B5 | verified checkpoint/coverage reuse | 어떤 edit family에서 이득인가? |
| B6 | native CPU AOT | Numba 대비 latency/RSS/startup 이득인가? |
| B7 | canonical CUDA, graphs off/on | transfer 포함 GPU가 언제 이기는가? |

B2가 기존 path로 표현하기 어렵다면 이를 명시하고 direct work-count counterfactual만 보고한다. 없는 implementation의 시간을 만들어 넣지 않는다. 모든 후보는 동일 output coverage와 zero-mismatch gate를 통과한 뒤 비교한다.

### 16.3 Workload matrix

- Site scale: 작은 synthetic 32×48/40×80/96×96, real site_500, 더 큰 synthetic/허용된 real site.
- Edits: first add, repeat move, resize, delete, overlapping canopies, hidden tree deletion, distant disjoint edits, boundary/corner, amplitude raiser/dropper, land cover, building, met and mixed-family sequences.
- Time: selected early/noon/late, sparse `[1,12,23]`, full 24, midnight boundary, forcing-prefix edit.
- Runtime: cold process, warm process/cold cache, warm cache, fresh scene, 100th edit, restart, concurrent client, CPU contention.
- Hardware: fixed primary CPU 4 cores, 1/2/8-thread sweeps, ARM CPU secondary, one fixed CUDA GPU, forced CPU on CUDA-visible host.

작은 grid에서 sparse routing이 느린 경우도 숨기지 않는다. Small grid에서는 GVF margin 때문에 LOCAL이 FULL로 바뀔 수 있다. Candidate가 선택한 실제 route, write/read fractions, cell-step count를 항상 기록한다. [R3][R6]

### 16.4 Timing과 통계

Python boundary는 monotonic clock을 쓴다. CUDA는 device event elapsed time과 synchronized end-to-end wall time을 둘 다 기록한다. Asynchronous enqueue 시간만 GPU latency라고 보고하지 않는다.

Warmup은 최소 5회이며 compilation, page fault, cache building 상태를 따로 기록한다. Expensive baseline은 paired runs 최소 30회를 목표로 하되 가능한 표본 수와 신뢰구간을 공개한다. Fast candidate는 최소 100회 p50/p95, p99 claim에는 최소 1,000개 completed epoch와 supersession/freshness coverage가 필요하다. 적은 표본으로 p99를 주장하지 않는다.

A/B 순서는 가능한 한 randomized paired blocks로 실행하고 host load, CPU frequency/thermal state, GPU contention, thread affinity를 기록한다. Profiler를 켠 시간과 production timing을 혼동하지 않는다. CProfile의 inclusive time끼리 더하지 않는다. GPU/CPU profiler overhead를 별도 control run으로 확인한다.

### 16.5 Counters

```text
accepted_epoch_count, superseded_target_count, latest_scene_age_ms
ack_ms, epoch_wait_ms, queue_ms, plan_ms, compose_ms
closure_ms, pair_count, pair_steps, dense_equivalent_pair_steps
svf_march_ms, svf_fold_ms, radiation_ms, thermal_ms, utci_ms
h2d_bytes/ms, d2h_bytes/ms, graph_build_ms, graph_replay_ms
checkpoint_hit/reason, replay_start, steps_solved
pack_ms, encode_ms, payload_bytes, decode_ms, publish_ms
fallback_reason, selected_backend, numerical_profile
RSS/PSS, allocation peak, device live/total bytes, cold_compile_ms
mismatch_count, first_mismatch_locator, fixture/source hashes
```

Profiler samples가 wall-time stage를 대체하지 않는다. `no-op`, `cache_hit`, `full_replay`, `sparse_replay`, `legacy_profile`을 분리해서 보고한다.

## 17. 대안 비교와 채택 우선순위

| 방안 | 본 작업에서의 가치 | 핵심 위험 | 결정 |
|---|---|---|---|
| Torch eager 유지 + 작은 hoist | 작은 변경으로 일부 비용 절감 | dominant dense work는 남음 | migration baseline과 control variant |
| 전체 Torch→NumPy 치환 | dependency 줄이기 | 같은 many-array-pass 구조면 큰 이득 불확실; primitive mismatch | standalone 목표로 삼지 않음 |
| Numba compiled loops | state-machine 구현·디버깅이 비교적 직접적 | LLVM/math lowering, JIT/RSS | CPU 첫 구현 |
| C++ AOT + controlled SIMD | 작은 runtime, explicit control | build/portability와 검증 부담 | 최종 경량화 후보 |
| CUDA C++ | GPU residency와 coarse kernels, rounding control | cross-device math와 launch/transfer | production GPU 우선안 |
| 기존 compiler 자동 fusion | 적은 source 변경으로 후보 생성 | reassociation, reductions, shape-specialization의 opaque behavior | exact harness를 통과할 때만 비교 후보 |
| approximate horizon/LOD/surrogate | 별도 제품에서는 latency 도움 가능 | 현재 bitwise 계약 위반 | 본 goal에서 제외 |
| winning-occluder-only index | 일부 visibility 모델에서 빠름 | 현재 suppression/reset/value semantics를 표현하지 못할 수 있음 | proof 없는 대체 금지 |
| incremental float SVF sum | edit당 계산 감소 | summation history가 바뀜 | 금지 |
| 완성된 cell/bit state 재사용 | 정확한 dependency hit에서 계산 제거 | invalidation bug | 우선 채택 |

추가 research candidate로 “첫 변경 patch 이전의 exact accumulator prefix”를 저장하고 그 지점부터 suffix fold를 재실행할 수 있다. 이것은 old contribution을 빼는 방식과 다르다. 동일 prefix input bits와 동일 accumulator bits를 재사용하므로 원래 recurrence를 이어갈 수 있다. 다만 10개 accumulator의 여러 prefix checkpoint는 memory cost가 크며 변경 patch가 초반이면 이득이 거의 없다. 기본 full canonical fold를 먼저 최적화한 뒤 별도 ablation으로 판단한다.

완전히 다른 ray-tracing acceleration structure나 analytical visibility로 재설계하는 것도 연구 후보지만, 현재 discrete step/reset/zero-padding semantics와 동치라는 proof가 필요하다. 연속 기하학적으로 더 정확한 ray tracing이 reference와 같은 bit를 보장하지 않는다.

## 18. Validation: output 비교보다 앞 단계까지 본다

### 18.1 Gate hierarchy

| Gate | 검사 | 실패 시 |
|---|---|---|
| G0 | baseline reproducibility, environment/input/source pin | 최적화 측정 중단, 원인 기록 |
| G1 | primitive bits/dtype/context | 해당 primitive/backend 미인증 |
| G2 | step table/offset/dz/termination | march integration 금지 |
| G3 | per-ray ordered state and outputs | sparse/parallelization 금지 |
| G4 | corridor closure and untouched-state invariance | legacy exact fallback 유지 |
| G5 | packed roundtrip and canonical fold vs independent oracle | packed path 비활성 |
| G6 | radiation/UTCI intermediate+final outputs | 해당 compiled stage 비활성 |
| G7 | temporal warm/cold/full equivalence | cold replay fallback |
| G8 | full-domain operation-sequence parity and API composition | release 금지 |
| G9 | CPU/CUDA profile certification | cross-device dispatch 금지 |
| G10 | performance/memory/startup/torch-free | optimization 목표 미달, 수치 완화 금지 |
| G11 | concurrency, restart, latest-state liveness | interactive release 금지 |

### 18.2 Fixtures

Synthetic fixture는 실제 153-patch geometry를 사용한다. Arbitrary random patch count가 original anisotropic code를 실행하지 못하는 경우 무의미한 test가 된다.

필수 numerical cases:

```text
non-square grids, all four quadrants, cardinal/sector boundaries
azimuth 0 and nextafter neighbors; altitude/zenith wrapper behavior
scale values corresponding to 1/2/3/4 m plus non-power-of-two values
zero/negative/positive DEM, no vegetation, no buildings
amplitude constant, raiser, dropper, strict clamp-bound equality
one-step vbsh=2 witness, multi→one regime transition
bush nonzero/negative refusal; boundary OOB zero semantics
overlap/max-composition, delete tallest, undo and re-add
day/night, midnight, selected/sparse/full time coverage
invalid sentinel, signed zero, NaN payload where accepted
```

Real-site fixture는 read-only cache로 재실행하며 every edit 뒤 fresh full-domain oracle와 비교한다. Unchanged outside-window bytes까지 검사해서 dirty closure 누락을 잡는다. Local output 안의 max error만으로 통과시키지 않는다.

### 18.3 Sequence/state-machine tests

Fast suite는 deterministic seed로 최소 수십 개 sequence를 실행하고 nightly는 1,000+ accepted edit transitions를 누적한다. Add→move→resize→delete→undo, mixed met/tree/paint/building, resets, duplicate idempotency keys, out-of-order requests, restart를 포함한다.

반드시 각 transition 직후 비교한다. 마지막 결과만 baseline과 같으면 중간 wrong-exact publication을 놓친다. Equivalent operation reordering은 실제로 commute한다는 proof가 있는 family에만 metamorphic property로 사용한다.

### 18.4 Mutation tests: test가 진짜로 실패하는가

다음 deliberate defect가 최소 하나의 지정 test를 반드시 실패시켜야 한다.

| Mutation | 잡아야 할 test |
|---|---|
| corridor에서 `∪ C` 삭제 | target-local first-step witness |
| previous-state while을 post-update check로 변경 | overshoot-step trace |
| dx/dy 또는 rows/cols 교환 | non-square quadrant fixture |
| dz association 변경 | non-power-of-two scale threshold |
| OOB zero를 -inf로 변경 | negative-target boundary witness |
| first-step vb reset 제거/이동 | one-step and multi-step comparison |
| clamp guard의 changed conjunct 삭제/strict bound 변경 | safe steady-clamp + equality boundary |
| R6 amplitude escalation 삭제 | independent `svf_calculator` witness |
| vbsh 2를 bool로 pack | nonbinary representability witness |
| packed byte를 parallel read-modify-write | adversarial pack ownership/stress test |
| annulus 합산/patch reduction tree 변경 | independent fold-anchor fixture |
| polynomial Horner/FMA 또는 transcendental 교체 | primitive/UTCI bit witness |
| 이전 scene checkpoint 강제 사용 | geometry history warm-vs-cold |
| sparse t index를 dense offset으로 scatter | multi-time API compose test |
| stale revision publish 허용 | concurrent supersession barrier test |
| golden과 candidate가 같은 helper를 공유 | source-origin/immutable-oracle check |

Mutation이 output에 실제 영향을 주지 않는 case는 “검증 성공”이 아니라 witness가 부적절할 수 있다. 수학적으로 value-inert인 변형이라면 그 이유와 별도 structural coverage를 기록한다. R6 review에서도 shared replay helper 때문에 mutation을 못 잡는 test가 발견되어 independent oracle anchor가 추가되었다. [R3]

### 18.5 Failure localization

Mismatch report에는 다음을 남긴다.

```text
fixture / edit index / scene revision / profile / backend
variable, global time, global row/col
first divergent stage → op_id → patch → step
input bit patterns, expected bit pattern, actual bit pattern
shape/stride/valid ordinal, reference and candidate source hashes
raw mismatch count, diagnostic ULP distribution
```

ULP는 원인 진단용이며 1 ULP를 통과시키는 threshold로 사용하지 않는다. GPU race가 의심되면 동일 configuration 반복, launch geometry 변경, sanitizer, stress test를 추가한다. Result equality가 우연히 맞는 한 번의 run으로 race-free라고 주장하지 않는다.

## 19. Agent 작업 체계

### 19.1 Agent에게 큰 재설계를 한 번에 맡기지 않는다

`TASKS.ko.md`의 작은 task card를 dependency 순서로 수행한다. 각 card에는 writable files, input contract, red test, 구현 범위, exit evidence, rollback이 있다. Author와 reviewer가 같은 oracle 수정 권한을 갖지 않는다.

기본 loop:

```text
read current gate + source → reproduce baseline → write characterization/red test
→ implement one bounded change → narrow tests → independent full-domain differential
→ paired benchmark → non-author review → integration tests → focused commit
→ update worklog with exact next command
```

### 19.2 작업 경계와 보호

원본 `Input_rasters`, `Input_subset`, `site-cache`, `state` 및 live services는 수정하지 않는다. Scratch output은 별도 root에 둔다. Permission 없는 GPU/CPU resource를 사용하지 않는다. 기존 unrelated worktree와 user changes를 reset/clean/force-push하지 않는다.

Numerical changes와 scheduling/protocol changes를 같은 commit에 섞지 않는다. Source comments의 과거 “불가능” 또는 “≤5초” 결론은 새로운 evidence로 반박할 수 있지만, parity guard를 지우는 근거가 되지는 않는다.

### 19.3 Evidence status

- `PASS`: 실제 실행과 지정 evidence가 있다.
- `FAIL`: gate를 실행했고 계약을 만족하지 못했다.
- `BLOCKED`: 필요한 data/hardware/permission/toolchain이 없다. 구체적인 missing capability를 기록한다.
- `NOT_RUN`: 아직 시도하지 않았다.
- `TARGET_MISSED`: correctness는 통과했지만 performance/lightweight 목표가 미달이다.

GPU가 없으면 CPU 경로 구현을 계속할 수 있지만 GPU parity를 PASS로 표기하지 않는다. Real data가 없으면 synthetic gate와 real-data gate를 분리한다. Request budget/agent runtime limit으로 중단하면 best verified commit, remaining bottleneck, exact next command를 남기고 goal completed라고 하지 않는다.

### 19.4 Goal의 완료 조건

Final exact target, raw-bit gates, Torch-free reachable runtime, certified GPU canonical profile, legacy GPU regression, memory/startup와 concurrency gates가 모두 evidence와 함께 통과해야 전체 goal을 완료한다. 중간 milestone을 release 후보로 제공할 수는 있지만 final success와 구분한다.

한 번의 Numba 시도가 실패했다고 목표가 불가능하다고 결론 내리지 않는다. Failure가 arithmetic compatibility인지, dense work volume인지, thread dispatch인지 분해하고 task card에 있는 다음 후보를 실험한다. 반대로 반복되는 무근거 tweak를 무제한 시도하지 말고 explicit resource/iteration budget 안에서 verified progress를 남긴다.

## 20. 위험 목록과 중단 기준

| 위험 | 영향 | 초기 방어 | 해제 조건 |
|---|---|---|---|
| CPU/GPU legacy oracle 불일치 | 단일 exact identity 모순 | numerical profiles 분리 | canonical implementation equality 인증 |
| Transcendental/extent mismatch | 1-ULP 이상의 결과 변화 | primitive trace + compatibility stage | context matrix와 full oracle 통과 |
| Sparse closure 누락 | 멀리 있는 cell이 wrong exact | 기존 R4 guard/fallback | analytic dependency proof + adversarial tests |
| R6 escalation 손실 | one-step value divergence | current amplitude policy 재사용 | independent oracle witness 통과 |
| Packed write race | 반복 실행 비결정성 | unique byte owner | stress/sanitizer 및 bitwise repeats |
| Thermal state 오염 | history-dependent wrong result | fingerprint와 cold replay | warm/cold/full-domain equality |
| Shared oracle helper | test의 자기검증 | pinned process/source origin | mutation이 독립적으로 실패 |
| Coalescing/publication 오류 | accepted edit 유실/영구 pending | 기존 durable protocol 유지 | mixed-plane liveness regression |
| JIT/allocator cold cost 은폐 | 실제 첫 interaction 느림 | cold/warm 분리 | startup/total residency gate |
| 대표 fixture 편향 | 다른 edit에서 큰 regression | route/scale/family matrix | 모든 지원 class의 correctness 및 성능 공개 |

정확한 구현이지만 성능이 목표에 못 미치면 bottleneck report를 남긴다. Approximation, shorter reach, reduced precision, skipped physics로 target을 맞추지 않는다.

## 21. 실행 순서: 첫 구현에서 무엇을 할 것인가

**첫 번째 묶음은 oracle와 small-kernel proof다.** Current HEAD의 independent baseline, strict comparator, non-square/one-step/amplitude witnesses를 만들고 step tables 및 dense Numba march를 검증한다. 이 묶음은 성능 약속을 하지 않는다.

**두 번째 묶음은 지배적인 작업량을 줄인다.** Existing R4 state에 sparse CSR/row-run execution을 연결하고 direct packed fold를 넣는다. 같은 reference state에 대해 dense와 sparse를 비교한다. 이 단계에서 6° ring의 pair-step 및 bytes-touched 감소가 실제로 나오는지 확인한다.

**세 번째 묶음은 time loop다.** Radiation ordered fusion, UTCI primitive compatibility, existing R5 consumer의 실제 warm-hit를 개선한다. Selected-time latency에 필요한 causality를 제거하지 않는다.

**네 번째 묶음은 production backends다.** CPU AOT와 CUDA canonical profile, resident memory와 optional graph replay, measured backend dispatch를 추가한다. 이후 Torch-free clean runtime, fallback coverage, concurrency와 client-applied latency를 검증한다.

## 22. 최종 판단

이 코드에서 가장 유망한 경로는 **Torch를 Numba로 기계적으로 치환하는 것**이 아니라, **기존 reference의 정확한 실행 의미를 보존하는 sparse, fused, compiled incremental engine으로 바꾸는 것**이다.

특히 read window가 tile의 약 99%라는 사실은 계산도 그 전체에 대해 반복해야 한다는 뜻이 아니다. Read domain은 그대로 유지하고, 영향을 받은 targets만 같은 source sequence로 계산하는 것이 핵심이다. 이 차이를 이용하면 physics reach를 줄이지 않고도 큰 structural reduction을 시도할 수 있다.

다만 실제 100 ms CPU / 33.3 ms GPU exact target 달성 여부는 아직 검증되지 않았다. 이 설계는 그 목표를 주장하는 문서가 아니라, 어떤 계산을 제거하고 어떤 rounding을 보존하며 어떤 증거가 있어야 성공이라고 부를 수 있는지를 고정한 실행 명세다.

---

## 참고 근거

`[R#]`는 저장소의 고정 commit 문서/소스, `[W#]`는 외부 공식 문서다. 각 경로, 읽은 범위, 확인일, URL은 `SOURCES.md` 및 `sources.json`에 기록했다. 설계의 새 module, ABI, 목표 수치, execution profile, task gate는 본 문서의 제안이며 현재 구현이나 실측 결과가 아니다.
