# Ultrafast Bitwise — 성능·패리티 보고 README

> living document. 각 task 종결 시 갱신. 원본 카드: `TASKS.ko.md`, 설계: `DESIGN.ko.md`,
> 목표: `acceptance_targets.yaml` (변경 불가). 상세 증거: `WORKLOG_ULTRAFAST.md` +
> `~/Workspace/solweig_ultrafast_artifacts/t*/`.

## 한눈에 보기

원본(torch) full-domain oracle 대비 **raw float32 bit 동일성**(uint32 view,
signed-zero·NaN payload 포함, allclose/1-ULP 허용 없음)을 유지하면서
증분 편집 엔진의 각 계산 단계를 torch-free CPU 커널로 재작성.
성능 수치는 전부 **동일 호스트·동일 fixture·n회 반복 median**이며 원본과
패리티가 **증명된** 상태의 수치만 기재한다.

- 기준 호스트(CPU 레인): 10-core Apple M1 Pro, macOS arm64, torch 2.14.0 CPU.
  (4-core pinned acceptance host는 미확보 — T16에서 확보 필요.)
- 기준 호스트(GPU 레인): `cura` — 4× NVIDIA RTX A6000 49GB (sm_86),
  driver 550.144.03, CUDA 12.0, 72-core Xeon, 187GB RAM.

## 단계별 성능 (CPU canonical, site_500 = 500×500)

| 단계 | 원본 oracle | 후보 (스레드) | 가속비 | 패리티 증명 |
|---|---|---|---|---|
| T04 dense march (1회, warm) | — (원본은 전체 루프만) | **0.0241 s/march** | — | 74 frozen traces 0 mismatch |
| T05 closure 빌드 (F1 편집) | dense 14.51 s | **0.80 s** (build 0.735 + march 0.065) | **~18×** | closure completeness 0 miss |
| T06 sparse march | (dense 대비) | t8 1.87 ms (synthetic 55% mask) | dense 대비 6.4× | dense↔sparse↔t1/2/4/8 raw 0 mismatch |
| T07 SVF fold (full 500²) | svf_calculator 69.1 s | **t4 0.202 s** (+pack 0.082 s 1회) | **~340×** | 11 출력 전부 site_500 0 mismatch |
| T07 affected 10% fold | — | **t4 0.024 s** | — | masked==full, unmasked==base bit-identical |
| T08 radiation (24 timestep) | 43.81 s | **fused 13.27 s** (warm/day-step 0.766 s) | **3.22×** | intermediates 1746 ok/0 bad, fused 685/685 |
| T09 UTCI (1 timestep, 25만 lane) | torch 0.0727 s | **0.0733 s** (1.008×) | 패리티 | dense/sparse/site chain 0 mismatch |
| T14a 시작 JIT (warm cache, 프로세스당) | 7.15 s (재컴파일) | **1.27 s** | **~5.6×** | cross-process bit parity(cold컴파일=warm재생=in-process) |
| T10 thermal warm replay (resume@20) | fresh-full 24 steps 14.26 s | **4 steps 1.83 s** | steps 24→4 | warm/cold/fresh 전 24t·20 plane uint32 0 mismatch |
| T11 full-solve fallback + router | oracle torch 경로 | cold **996.98 ms/step** / warm replay 695.85 ms/step | — | met identity 전 24t·전 필드 raw 0 mismatch; full-solve identity 21 plane + ret_CI 24t 0 mismatch; edited-met은 warm-vs-fresh 내부 differential(외부 오라클 부재 — 한계 명시) |

전체 원본 solve (site_500, 24 timestep): **49.31 s** — 이 중 radiation 88.9%.
T00 기록 원본 full time loop: 315.8 s (구버전 측정).

누적 테스트: `tests/ultrafast` **531 passed / 82 skipped** (T13 시점 — CUDA 파일 로컬 skip 포함,
**cura 회귀 94P** 별도). T11 시점 527P/54S, T09 시점 423P/2S(dc61d85).

## 패리티 증명 방법론 (전 단계 공통)

1. **오라클 분리**: oracle은 pinned 별도 워크트리/프로세스, 후보와 changed helper
   공유 없음. golden 변경으로 green 만들기 금지.
2. **raw bit 비교**: `float32 → uint32 view` 직접 비교. NaN payload, ±0 구분.
3. **독립 differential**: dense full recompute 대비 sparse/병렬/증분 경로.
4. **mutation testing**: task별 의미 있는 mutation ≥1개(관행상 5–7개)를
   지정 test가 kill하는지 검증 (author 계열 + lead 독립 계열 분리).
5. **순수성**: 런타임 모듈 torch-free (grep + subprocess import 검증).
   torch는 test/bench의 oracle 비교용으로만 허용.

## 트랜잭덴탈 프리미티브 정책 (T01 → T09 성과)

- **CLEAN** (torch-free 재계산 입증): f32 plane 산술, f64-adder accumulation,
  SLEEF 포트(xexpf, xlogf_u1 — torch 내부 u10 symbol→u1 body pin, xpowf),
  scalar-exponent pow exact forms (e ∈ {−2..3}),
  first-NaN-operand nadd/nmul/ndiv wrapper.
- **P9 발견**: torch `pow(Tensor, Scalar)` chunk-tail 레인은 opmath f64 pow
  (double-rounding) — layout 모델로 정확 재현 (T01 pow_float 발산 원인 폐쇄).
- **FROZEN** (torch 재계산 불가, captured bits + digest pin): torch `**4` plane,
  sin/cos/tan/exp 스칼라 체인 (T08 radiation family). hybrid status는
  `radiation.py` 헤더 + `math_compat.HYBRID_STATUS`에 정직 기록.

## GPU 레인 (cura, T12/T13)

| 항목 | 상태 |
|---|---|
| 원격 환경 | `~/solweig_ultrafast/` — 완전 삭제 계약 (`MANIFEST.md` 참조). repo @ dc61d85, site 데이터 전송 완료 (766MB) |
| **원본 oracle GPU baseline** | **측정 완료** (독립 verifier, read-only). 아래 표 |
| T12 CUDA canonical kernels | **PASS** (2026-09-08) — 5단계 전부 canonical CUDA ↔ canonical CPU raw-bit 동등, legacy kernel-scope 재현. 아래 표 |
| T13 residency/graphs/dispatch | **PASS** (2026-09-08) — 상주 + 변경 chunk 전송 + graph + measured dispatch, 94P(cura). 아래 표 |
| acceptance target (CUDA 33.3ms selected-time) | T16에서 측정. 미달 시 TARGET_MISSED 로 정직 보고 |

### 원본 oracle GPU baseline (site_500, 24 timestep, A6000)

| 구성 | median (n=3) | 비고 |
|---|---|---|
| gpu_harness (bundle SVF, amax 57.85) | **7.38 s** | legacy_cuda_v1 PRIMARY reference |
| gpu_fullbatch (svf_calculator GPU, amax 228.34) | 8.60 s | 2차 reference (원본 batch semantics) |
| cura CPU 대조 (72-core, CUDA hidden) | 19.75 s | **2.7× GPU 가속** |

스테이지 분해 (GPU/CPU, 초): radiation 6.88/21.27, Lcyl 3.75/13.79, define_patch 3.25/13.63,
gvf 1.93/3.39, sunonsurface 1.88/3.28, march 0.38/0.87, utci 0.10/0.69.
GPU 이용률 mean 51% / max 68%, 1232 MiB, ~121 W (물리 GPU 0 pin).

**패리티 lattice 사실** (T12 게이트의 기반):
- **shadow cube `6c62bcea…` = 보편 불변** — GPU·CPU·모든 composition·로컬 macOS t08
  capture 전부 bit-identical. march 포트의 첫 이정표.
- tmrt/utci는 4개의 서로 다른 bit-pattern (gpu-harness / gpu-fullbatch / cura-cpu /
  local-macos-cpu) — **환경 간 bit 동등성 불가** (물리 소스 sha 동일인데도 libm/OS 차이로
  최대 3.05e-5 °C). 따라서 legacy bit-gate는 **cura env (torch 2.5.1+cu121)에 pin**.
- 결정성: cura에서 cross-process bit-identical (GPU·CPU 각 3회, SVF 19-tuple 포함) —
  exact-equality gate 타당.
- legacy torch GPU↔CPU 편차 (참고, canonical 목표 아님): Tmrt day 최대 0.345 °C,
  UTCI 최대 0.0852 °C, march wallsh/wallsun 8/13 day-t 불일치, SVF 1-ulp + 희소
  threshold-flip.

전체 증거: `~/Workspace/solweig_ultrafast_artifacts/cura_oracle_baseline/`
(`legacy_cuda_v1_index.json` — digest manifest, provenance, 원격 미러).

### T12 CUDA canonical 스테이지 성능 (site_500 500×500, A6000, kernel-only = CUDA events / E2E)

| 커널 | kernel-only | E2E (H2D+D2H 포함) | 비고 |
|---|---|---|---|
| march_svf_shadow | 0.253 ms | 1.27 ms | canonical==legacy bits |
| march_wallheight23 | 0.299 ms | 1.30 ms | |
| svf fold | 0.734 ms | 3.14 ms | 11 출력 raw 동등 |
| utci_dense | 0.551 ms | 1.70 ms | 전 lane bit-identical |
| utci_sparse (n=237,500) | 0.259 ms | 1.03 ms | |
| **rad_day** | **12.99 ms** | 43.1 ms | H2D 202 MB ≈ 26 ms가 E2E 지배 → T13 residency |
| rad_night | 0.008 ms | 13.4 ms | |

빌드 프로파일: canonical = `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`
(전 상수 exact hex float literal), legacy_cuda_v1 = `--fmad=true` (미인증, 캐릭터화만).
상세·게이트 증거: WORKLOG_ULTRAFAST.md T12 + `native/cuda/` (pin data 72 MB 포함,
content-digest 관리).

### T13 residency/graphs/dispatch 성능 (site_500 500×500, A6000, E2E wall — 전송+sync 포함)

| 구성 | rad_day E2E | H2D/call | 비고 |
|---|---|---|---|
| T12 host-staged wrapper | 50.1 / 50.9 ms | 202 MB | T12 기록 43.1ms는 최초 warm 레인 |
| **resident 정착 (steady)** | **18.5 / 18.9 ms** | 0 | 정적 ~178MB 상주, H2D 26ms 제거 |
| resident 40-row 편집 | 19.8 / 19.9 ms | 1.05 MB | 변경 chunk만 (uint8-view bit 검출) |
| graph replay (rad_day) | 21.0 / 21.5 ms | 1.05 MB | replay wall 12.97ms ≈ kernel 12.99ms |
| rad_night | 13.6 → **11.1 ms** | | D2H+fence 오버헤드 바닥 |
| UTCI sparse (n=250k) | 1.00 → **0.79 ms** | | graph build 2.2ms, ~11회 상각 |
| dispatch (tiny 16-lane) | cpu 0.0010 vs cuda 0.080 ms | | **cpu 선택 + device 할당 0** (실측 기반) |
| dispatch (full-site) | cuda 13.9 vs cpu 860.5 ms | | cuda 선택 |

패리티: resident == T12 wrapper == oracle (transitive, T12 digest gate 동일 suite green).
Fence 7종: publish / stale-lease(lead 보강) / reuse / padding / freed-pointer / profile-key /
CPU-selected zero-alloc. Stress 400 cycle alloc delta 0 · free-mem drift 0B.
상세: WORKLOG_ULTRAFAST.md T13 + `solweig_ultrafast_artifacts/t13/`.

### cura 사용·삭제 계약

- 이 작업이 이 호스트에 만드는 모든 것은 `~/solweig_ultrafast/` 단일 루트
  (repo/, venv/, data/, work/). sudo·시스템 설치·공유 conda·cron 없음.
- 원격 호스트 특이점: 시스템 전역 PYTHONPATH 오염 → 모든 실행은
  `env -u PYTHONPATH` wrapper (`~/solweig_ultrafast/py`) 경유.
- **전부 끝나면**: 증거를 laptop artifacts로 회수 후 `rm -rf ~/solweig_ultrafast`.
  절차는 MANIFEST.md TEARDOWN 섹션 참조.

## acceptance targets 현황 (T16에서 최종 판정)

| 목표 | 상태 |
|---|---|
| site_500 selected-time exact E2E p95 CPU 100 ms | NOT_RUN (스테이지 조립 T11 완료 — cold 997 ms/step·warm 696 ms/step 실측은 목표 미달 수준, 측정·판정은 T16) |
| canonical CUDA 33.3 ms | NOT_RUN (T13에서 rad_day resident E2E 18.5ms·편집 19.8ms 실측 — selected-time 전체 조립·판정은 T16) |

목표는 baseline이 아니다. 미달은 미달로 보고한다.
