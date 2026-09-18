# SOLWEIG 실시간 exact engine 설계 패키지

검토 기준은 `AlanSynn/solweig-project@e0d19fc98220a333ea312efe4604780803bbdd17`이다. 이 패키지는 source/report 검토를 바탕으로 한 설계와 실행 지침이며, SOLWEIG의 새 구현이나 실측 speedup을 포함하지 않는다.

## 파일

| 파일 | 용도 |
|---|---|
| `DESIGN.ko.md` | 전체 설계: 현재 병목, parity contract, sparse kernels, CPU/CUDA, temporal state, benchmark와 release gates |
| `TASKS.ko.md` | T00~T17 task cards: dependency, writable scope, failing witness, exit evidence |
| `GOAL_PROMPT.ko.txt` | `/goal`로 시작하는 전체 실행 프롬프트 |
| `acceptance_targets.yaml` | 목표와 hard constraint의 machine-readable source of truth. 아직 측정되지 않은 목표 |
| `SOURCES.md`, `sources.json` | 고정 source 경로와 공식 외부 문서의 provenance |
| `tools/bitwise_check.py` | Float32 `.npy` raw-bit 비교 시작점. NaN payload와 signed zero 포함 |
| `tools/test_bitwise_check.py` | 비교기 자체 test 18개. SOLWEIG physics 검증과 별개 |
| `VALIDATION.md` | 패키지 생성 과정에서 실제로 실행한 검사와 수행하지 않은 검사 |
| `SHA256SUMS.txt` | 배포 파일 digest |

## 저장소에 넣기

ZIP의 `solweig_realtime_bitwise_design/` **내부 파일들**을 다음 위치에 놓는다. 기존 numerical/source 파일을 덮어쓰는 패키지가 아니다.

```text
docs/incremental_design_tool/ultrafast_bitwise/
  DESIGN.ko.md
  TASKS.ko.md
  GOAL_PROMPT.ko.txt
  acceptance_targets.yaml
  ...
```

파일을 넣은 뒤 Codex의 Goal 지원 환경에서 `GOAL_PROMPT.ko.txt` 전체를 입력한다. 짧은 시작 명령을 사용할 때는 아래와 같이 전체 프롬프트 파일을 실행 계약으로 지정한다. `/goal`은 outcome과 evidence/constraints를 붙이는 공식 command surface다. [W6: SOURCES.md]

```text
/goal docs/incremental_design_tool/ultrafast_bitwise/GOAL_PROMPT.ko.txt의 실행 계약 전체를 읽고 수행하라. DESIGN.ko.md, TASKS.ko.md, acceptance_targets.yaml을 기준으로 실제 구현과 검증을 진행하되, 독립 full-domain oracle에 대한 raw-bit parity를 유지하라. 최종 목표는 대표 site_500 selected-time exact end-to-end p95 CPU 100 ms, canonical CUDA 33.3 ms이며 아직 실측되지 않은 목표다. 목표를 완화하거나 미검증 gate를 PASS로 처리하지 마라. T00 baseline과 T01 oracle/primitive 검증부터 시작하라.
```

설치/배포·live state 수정 권한은 이 프롬프트가 새로 부여하지 않는다. 사용 가능한 local resources에서 read-only 입력과 별도 scratch root를 사용한다. User/system budget 또는 hardware/data blocker가 있으면 partial status를 남기며 goal success로 바꾸지 않는다.

## 비교기 self-test

NumPy가 있는 Python 환경에서 패키지 root 기준으로 실행한다.

```bash
python -m unittest discover -s tools -p 'test_*.py' -v
```

두 float32 `.npy` 파일의 exact 비교:

```bash
python tools/bitwise_check.py reference.npy candidate.npy \
  --require-nonempty --report comparison.json
```

Exit 0은 이 두 배열의 dtype/shape/raw element bits가 같다는 뜻이다. Profile, scene revision, global time, spatial coverage 또는 물리적으로 올바른 결과임을 뜻하지 않는다. 이들은 repository integration harness에서 별도로 검사해야 한다.

## 먼저 읽을 핵심

기존 incremental exact 경로는 검토된 보고서에서 CPU-only다. R4/R5/R6는 상당 부분 이미 구현되어 있다. `vbsh`의 2.0 예외와 R6 amplitude escalation을 삭제하면 안 된다. 기존 CPU/GPU output이 다른 만큼 canonical cross-device profile과 legacy GPU profile을 분리한다.

핵심 최적화는 Torch 의존성 삭제 그 자체가 아니라 **read domain을 보존한 sparse target execution + ordered compiled fusion + 검증된 state reuse**다. 100 ms/33.3 ms는 계획의 합격 목표이지 이 패키지가 입증한 성능이 아니다.
