# 패키지 검증 기록

확인일: 2026-09-07

## 실제 수행한 검사

이 기록은 **설계 패키지와 standalone 비교기**에 한정된다.

- DESIGN.ko.md: balanced code fences, defined citations, language/punctuation scan PASS
- TASKS.ko.md: balanced code fences, defined citations, language/punctuation scan PASS
- README.ko.md: balanced code fences, defined citations, language/punctuation scan PASS
- Python helper and self-test syntax: PASS
- Goal command prefix, pinned review SHA, primary target consistency: PASS
- YAML syntax and primary hard constraints: PASS
- Task cards T00–T17 present: PASS
- Design-versus-measurement disclosure: PASS
- Standalone comparator self-tests: 18 passed (not SOLWEIG tests)

비교기 실행 명령:

```bash
python -m unittest discover -s tools -p 'test_*.py' -v
```

원본 실행 log는 `tools/SELF_TEST_OUTPUT.txt`에 있다. Signed zero, NaN payload/sign/signaling bit, 1-ULP change, dtype/shape/byteorder, strided/scalar/empty arrays, chunk coordinates, input immutability, NPY/memmap CLI와 overwrite guard를 검사했다.

## 수행하지 않은 검사

- SOLWEIG source를 새로 빌드하거나 실행하지 않았다.
- 실제 site_500, 전체 numerical pipeline, CPU/GPU differential을 실행하지 않았다.
- Numba/C++/CUDA 최적화 kernel을 구현하거나 benchmark하지 않았다.
- 설계의 100 ms/33.3 ms, memory/startup 수치를 달성했다고 검증하지 않았다.
- Repository를 수정, commit, push 또는 production deployment하지 않았다.

저장소의 과거 실험 수치는 해당 보고서의 기록이다. 이 패키지의 self-test 통과가 그 과거 실험을 독립 재현한 것은 아니다.
