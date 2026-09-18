# OPERATIONAL_STABILITY.md — realtime studio 운용 안정성 검토

**날짜:** 2026-09-11 · **작성:** main session (검증·편집) + opus counter-analysis (설계 반검증)
**방법:** 모든 file:line은 main session이 코드에서 직접 대조 확인. 생산 환경(fly)과 로컬 공유 월드 실측 포함.
**범위:** 세션 수명/presence, ETA 진행 계약, 큐 안정성, op-history 성장, 프론트 구조. 물리 경로 변경 없음.

> **한 줄 요약:** 서버는 깨지지 않았다 — 그러나 (1) 세션이 영원히 남고, (2) 완료 예정 시각이 클라이언트에 도달하지 않으며, (3) 워커 1개 + 원장 무한 성장이 공유 월드의 미래를 깨는 구조적 시한폭탄 3개가 확인되었다. 전부 물리·정확성 계약을 건드리지 않고 고칠 수 있다.

---

## 실측 증거 (2026-09-11)

| 현상 | 실측 | 근원 |
| --- | --- | --- |
| "with Editor 1 · … · Editor 12" 영구 표시 | 공유 월드 13 ops / **12개 서로 다른 actor** — 테스트 에이전트 세션이 하나도 소멸 안 됨 | 클라이언트 로스터가 "본 편집" 기반, 만료 없음 |
| "updating · 4416 s" (74분 경과) + "usually about 4 min" 공존 | exact 작업이 **프로세스 전체 단일 워커 스레드** 직렬화 | `jobs.py:924` `JobRunner` "Single background worker thread" |
| 토스트 "Preview updated — precise map queued / 0.00 s" — 완료 감각 없음 | SSE는 완료 시 `exact_revision`만 실음, 진행 이벤트 없음 | `broadcast.py:201-217` job 이벤트 브로드캐스트 부재 |

---

## AXIS 1 — 세션 수명 / PRESENCE

**진단(확인됨).** 클라이언트 로스터는 op 유래·추가 전용·만료 없음
(`op_builder.mjs:453-462` `presenceFromOperations`; `app.mjs:2874-2880` `observeActor` 추가만 함;
`app.mjs:2882-2895` 렌더). 툴팁은 정직("Not a guaranteed roster") — **라벨은 정직, 영구성이 거짓**.
서버는 살아있는 구독자를 정확히 앎 (`broadcast.py:242-257` subscribe, `:259-273` unsubscribe
(generator-finally), `:275-279` subscriber_count). 그러나 로스터를 아무에게도 알리지 않음.
하트비트 5 s마다 epoch 스케줄러 스레드에서 발화(`epochs.py:95, 557-569`)且 privileged 전달로
drop-oldest 큐를 생존(`broadcast.py:72-78, 219-238`). 죽은 TCP는 자가 정리: heartbeat write 실패 →
GeneratorExit → finally-unsubscribe. **구독자 TTL 불필요.**

**부수 발견(위험).** privileged `_enqueue`가 용량 검사를 완전히 우회 (`broadcast.py:95-99`) —
이벤트 루프는 살아있는데 읽지 않는 구독자가 하트비트 프레임을 무한 축적 (대당 ~15 MB/일).
privileged 보류 프레임에 cap(직전 하트비트 대체) 필요.

**제안.**
- **Now:** join/leave 이벤트가 아니라 **하트비트 안에 로스터**. `GET /events?actor_id=`
  (검증·길이 제한); hub가 워크스페이스별 live-actor set 유지; `heartbeat_payload`에
  `roster:[actor ids]` 추가. join/leave 대비 우위: 하트비트는 privileged(지연 생존)·멱등 full-state
  (프레임 유실에도 불일치 없음)·**히스테리시스 불필요** (재접속 폭풍 최악 = 5 s 늦은 로스터, 자가 수정).
  join/leave는 drop-oldest 모델에서 유실되고 양단 grace window 필요.
- **Now(클라이언트):** 3단 로스터 — live(하트비트 로스터; "Editor 3" 서수는 first-seen
  `rt.roster`에서 상속해 안정), 축약 "N editors here earlier"(`rt.roster` − live), "solo session".
  정직성 툴팁 유지.
- **Later:** 로스터 항목에 actor별 마지막 활동 시각(마지막 수용 op 시각).

**위험.** 같은 actor의 멀티 탭은 actor id로 중복 제거. 서버 로스터를 op에서 유도하면
**forever-roster 재발** — 금지. 추가 필드 방식(additive)이라 구형 클라이언트 무시 가능.

## AXIS 2 — ETA / 진행 계약

**진단(확인됨).** 정직한 ETA 기계가 폴라 경로엔 end-to-end 존재: `job_eta_fields`
(`jobs.py:506-545`) — p80/이력/정적만, 추측 금지(`store.py:2528-2591`); 클라이언트는
`#jobStatusDetail`에만 렌더 (`app.mjs:127-175`, wiring 4009/4029). 토스트는 정적 카피
("Preview updated — precise map queued", `app.mjs:2327`). "updating · 4416 s" 필은
**마지막 `exact_revision` 이벤트 이후 경과시간**(`realtime_client.mjs:875`)에 클라이언트 자체
측정 이력 절을 붙인 것(`app.mjs:2680-2684`) — 과거 실행에 대한 서술일 뿐 현재 큐가 아님.

**구조적 공백(핵심).** exact 작업은 **서버가 mint** (`jobs.py:1322-1481`)하고 **클라이언트는
job id를 못 받음** — SSE는 완료 시 `exact_revision`만 (`jobs.py:1613-1657`), job 이벤트
브로드캐스트 없음. 네이티브 플레인 세션은 폴라할 job이 없어 **카운트다운이 오늘 원리적으로 불가능**.
또 RUNNING eta는 모드 확정 전까지 null(executor 경로는 최종 스테이지에서 모드 보고,
`jobs.py:519-521`) — elapsed 74분과 "usually 4 min"이 공존한 이유.

**제안.**
- **Now(클라이언트만):** deadline = poll 시각 + `eta_seconds`(basis 있을 때); 남은시간 단조
  (eta 개정 때만 재계산); exact 패치 도착 시점에 "last map update HH:MM" 스탬프
  (`app.mjs:3787-3807` 성공 경로) — 토스트 + 배지 title에. eta null이면 경과만(현행 유지).
- **Next(서버, additive):** 워크스페이스별 **`exact_progress` SSE 이벤트** — job mint / dispatch /
  mode-known / terminal 전환 시: `{job_id, status, target_revision, queue_position, eta_seconds,
  eta_basis}`. 이것이 하중을 지는 수정 — 모든 구독자가 job 소유 없이 카운트다운 획득.
  전환 구동(틱당 아님); 직렬화 펜스는 이미 존재(`broadcast.py:186-199`).
- **Next:** QUEUED job의 큐 인지 eta = 실행 중 job의 eta + 자체 추정 (둘 다 알 때만).
  오늘 queued 비-reconcile은 `(None, None)`(`jobs.py:540-545`) → "Queued — position N" 렌더 —
  null을 정직하게 유지.
- **Later:** 기존 progress 콜백(`jobs.py:1842-1865`)을 이벤트에 브리지(~1 s, 풀솔브만).

**위험.** 카운트다운은 조용히 0 아래로 내려가면 안 됨 — "taking longer than usual · typically N"으로
클램프. RUNNING eta null은 정상(모드 미확정) — 경과만이 정직한 상태. SSE는 편의 플레인
(broadcast 유실 모델), GET이 진실 — 클라이언트는 스냅샷 필드에서 재유도. 숫자 발명 금지
(p80/정적 basis 규율 `store.py:2528-2591` 이미 옳음).

## AXIS 3 — 하나의 공유 월드에서 큐 안정성

**진단.** 후보 정책(단일-flight, supersede, coalesce)은 **이미 존재** — 기각:
워크스페이스별 single-flight(mint 시 target ≥ current인 queued/running exact job이면 mint 중단,
`jobs.py:1424-1444`), mint/dispatch 시 stale-pending supersede(`store.py:2298-2309`,
`jobs.py:1470`, 스캔 `jobs.py:1691-1743`), 500 ms coalescing + adaptive flag
(`jobs.py:1745-1797`), 죽은 세션 audience gate(`jobs.py:1292-1320`), idle-reconcile
grace/cap/backoff(`jobs.py:1483-1589`).

**실제로 불안정한 것 3가지.**
1. **프로세스 전체 워커 1개** (`jobs.py:924, 1042-1045`). 어떤 워크스페이스의 풀솔브 —
   감시 없는 private hatch의 idle-reconcile 포함 — 이 공유 월드의 live chase를 수분 봉쇄.
   워크스페이스 간 기아, 선점 없음.
2. **chase-at-newest가 풀솔브 직렬화**: 완료마다 최신 리비전으로 다음 job mint(`jobs.py:1359-1421`);
   building 편집은 full-heavy 경로(exact-lane full 비중 0.95, `executor_bridge.py:190-196`) →
   활성 공유 월드는 수분짜리 풀솔브 연속. 관측된 74분은 N개 연속 풀솔브(및/또는 외부 job 끼임)지
   미-coalesce 홍수가 아님. **봇이 큐를 깬 게 아님 — 제품이 그것을 보여주지 않을 뿐(axis 2).**
3. **`executor_bridge.py:1548`가 `operations_since(workspace, 0)`** — 솔브마다 op-history 풀스캔 후
   Python에서 span 필터. 솔브당 O(history) 낭비, 공유 월드 나이와 함께 성장.

**제안.** **Now:** 정책 변경 없음 — 가시성(axis 2) 먼저. **Next:** **reconcile 전용 워커 분리**
(job은 이미 `exact_lane.reconcile` 태그; 큐 태그 검사로 라우팅) — 유지보수 풀솔브가 live chase
봉쇄 못 함. 단, fly vm CPU 여유 먼저 측정 — 작은 박스에서 동시 2솔브는 스래싱 가능.
**Next:** 브리지 읽기를 span-bounded SQL로 (`server_sequence > consumed_seq AND <= span_hi`),
zero-loss 펜스(`executor_bridge.py:1555-1563`) 불변. **Later:** 없음 — supersede-on-finish가
이미 stale 결과 폐기(`jobs.py:1948-1967`).

**위험.** publish 펜스는 어떤 레인 분할에서도 exactly-once 유지(publish_fence lease
`jobs.py:1981-1993`; store version-staleness `store.py:2610-2620`). Supersession은
결과가 버려질 작업만 건너뜀 — 누적 순서·조성 불변, bitwise-equality 주장 불변.
ops 내구성 불변(수용 시 append, `routes_realtime.py:91-97`); fast lane은 vegetation-epochs-only 불변.

## AXIS 4 — OP-HISTORY 성장

**진단(설계상 무한 확인).** `realtime_operations`은 의도적으로 `sweep_retention` 밖
(`store.py:736-740, 801`; sweep은 idempotency_keys + edit_events만 `store.py:1034-1086`).
Catch-up GET 깊이 제한 없음 (`routes_realtime.py:171-198` → 무한 SELECT `store.py:1810-1818`);
클라이언트도 `appliedOperationIds` 영구 보유(`realtime_client.mjs:905`).

**더 나쁜 3개 표면(부수 발견).** (a) `realtime_canonical_state` — 리비전당 풀 folded-state JSON,
append-only(`store.py:802-810`); 바이트 ≈ 리비전 × 상태 크기. (b) exact 버전별 `results` 행 +
payload **파일** 무삭제(`store.py:2598-2636`) — **fly 볼륨 벽을 처음 치는 것**. (c) `realtime_epochs`
영구. 검증된 프루닝 위험: incident-3 원장 앵커링이 id의 네이티브 ops를 "마지막 원장 이벤트 이후"부터
재폴드(`executor_bridge.py:1518-1533`) — 앵커가 프루닝 바닥보다 오래되면 **부분 네이티브 히스토리를
폴드 = 조용히 틀린 과학**. 오늘날 blanket truncation은 안전하지 않음.

**제안.**
- **Now:** O(history) 읽기 수정(위 axis 3) + catch-up 본문에 count + oldest-sequence 메타데이터
  (성장 관측 가능화).
- **Next:** catch-up GET에 `prune_floor` 필드(초기 0) + typed "history pruned — rebuild from
  snapshot" 본문 `{snapshot_revision, exact_result_url}`; 그 다음 이미 존재하는 canonical-state
  테이블 위에 스냅샷 경로 구축; 그 다음에야 floor = min(terminal epoch들의 consumed_seq,
  가장 오래된 원장 앵커) − audit tail 아래의 ops/epochs/canonical 행 삭제(워크스페이스당 1 트랜잭션).
  `operation_id` UNIQUE 멱등성과 `server_sequence` 단조성은 구조적으로 생존(서열 재사용 없음; PK 불변).
  Canonical-state 보존도 같은 floor 적용(epoch은 최신 행에서 폴드). Results: 조성은 target 이하
  모든 버전을 디코드(`jobs.py:2606-2657`) → re-baselining으로만 프루닝 — 현재 버전으로 풀 1개
  발행 후 구버전 drop, idle-reconcile 기계 재사용.
- **Later:** 일회용 데모 월드의 학기별 fork/reset 계약.

**위험.** 잘못된 프루닝은 조용한 부패가 아니라 **요란한 job 실패**(unfoldable-op 펜스
`jobs.py:810-817`)로 드러남 — 펜스 유지. 스냅샷 경로 없는 catch-up cap은 늦은 참여자만 깨뜨림 —
스냅샷 먼저. 물리 경로 변경 없음; 읽기 범위·6° 하한·정밀도 무관.

## AXIS 5 — app.mjs 구조적 안정성

**진단.** 5,325줄이지만 이미 섹션화되어 있고 순수·테스트 고정 빌더가 인라인 추출됨
(`app.mjs:107-458`; `tests/realtime_ux, job_eta, catchup_copy, frame_sample`가 고정).
위험은 크기가 아니라 **관습만이 소유한 4개의 교차-섹션 불변**:
one-plane-per-gesture(`realtimePlaneReady` `app.mjs:2578-2583` 모든 submit 게이트; legacy 폴백은
exact_session), 프레임 캘리브레이션 잠금(samples → solve → residual → lock; `op_builder.mjs:230-283`,
wiring `app.mjs:3470+`), 배지 정직성(알 수 없는/예약된 와이어 클래스의 보수적 `effectiveClass`
폴백, `app.mjs:2621-2754` + `realtime_client.mjs:370-395`), compare-guarded DOM writes
(`setJobStatusDetail` 패턴 `app.mjs:2018-2022`, 전면 사용).

**제안.** 이미-순인 이음매로 추출, 리라이트 없음:
- **Now:** `status_copy.mjs` (`app.mjs:107-458`, ~350줄, 순수 + 테스트 고정; 허브가 한 릴리스
  재export).
- **Next:** `rt_submit.mjs` — submit 경로 + held flow + stall 영수증 조회 (`app.mjs:3186-3413`).
  가장 상태적·타이머 많은 블록(held-row 카운트다운 interval, admission/network hold 구분,
  같은 operation_id 재제출 규율) — 작은 인터페이스(submitEnvelope, hold 정책, 파이프라인 스테이지
  콜백) 뒤로 추출. `rt_adoption.mjs` — 원격 op 채택·프레임 캘리브레이션·presence·pulses
  (`app.mjs:3470-3683`). 프레임 캘리브레이션 불변(samples → solveFrameFromSamples →
  frameResidualOk → lock; `op_builder.mjs:230-283`)은 **반드시 한 덩어리로** 이동 — 부분 샘플
  집합에서 lock이 발동해도 요란하게 실패하는 것이 없어 조용히 깨지기 가장 쉬운 불변.
- **Later:** `rt_events.mjs` — fast/exact 이벤트 브리지 (`app.mjs:3684-3809`: markRowsPainted,
  renderVerifyOwed, markVerifySettled, maybeFetchExactResult). 우선순위 1(exact_progress SSE)과
  **같은 변경** 안에서 추출 — 계약 변경과 이동이 하나의 리뷰 가능한 diff.

**하지 말 것 (Do NOT).**
- 5단계 파이프라인 바, 결과클래스 배지 렌더러 추출 금지 — 허브 `elements` 맵의 15+ 엘리먼트 참조를
  만짐; 추출은 결합을 제거하지 않고 이전만 함.
- 프레임워크·상태 라이브러리 도입 금지 — compare-guarded DOM write 규율(`app.mjs:2018-2022`
  패턴; setConnectionState·renderTransportPill·renderClassBadge에 적용)이 그 자체가
  anti-flicker/UX 계약. 반응형 프레임워크는 다른 의미론으로 재작성하게 만듦.
- 4개의 교차-섹션 불변을 **이름 붙은 한 곳**(헤더 블록 또는 `INVARIANTS.md`: 불변 → 소유 모듈 →
  고정 테스트 매핑)에 기록. 오늘 관습만이 소유 — 진짜 single-writer 위험은 줄 수가 아니라 이것.

**위험.** 모든 추출은 one-writer + worklog-card 규율 따름(카드 완료 시 저자가 자기 WORKLOG
엔트리 작성). 우선순위 1–3이 그 섹션들을 만질 때 기회적으로 추출, 단독 churn으로는 never.
추출된 모듈은 허브 re-export 제거 전에 기존 고정 테스트가 green 유지(status_copy.mjs는 re-export
한 릴리스 유지).

---

## 우선순위 (합의 확정)

1. **Axis 2 — `exact_progress` SSE 이벤트 + 클라이언트 카운트다운.** mint/dispatch/mode-known/
   terminal 전환 시 `{job_id, status, target_revision, queue_position, eta_seconds, eta_basis}` +
   클라이언트 deadline = 이벤트/폴라 시각 + eta(단조 잔여), 토스트·배지에 "last map update HH:MM".
   실제 사용자 불만의 본체 — 74분 침묵을 고치는 것은 이것뿐. additive 계약, 저렴. 구조적 뿌리:
   exact job은 서버 mint(`jobs.py:1322-1481`) + 클라이언트는 id를 못 봄 — SSE는 완료
   `exact_revision`뿐(`jobs.py:1613-1657`).
2. **Axis 4-Now — 브리지 span 쿼리 + catch-up 성장 관측화.** `executor_bridge.py:1548`의
   솔브당 풀스캔을 span-bounded SQL로. 공유 월드 나이와 함께 자라는 실비용 제거, 계약 변경 제로.
3. **Axis 1 — 하트비트 속 로스터.** `GET /events?actor_id=` + hub 워크스페이스별 live-actor set +
   privileged 5 s 하트비트에 roster 필드; 클라이언트 3단 로스터(live / "N editors here earlier" /
   solo). 아울러 privileged 하트비트 큐 cap(`broadcast.py:95-99` — 웨지된 루프당 ~15 MB/일 무한
   축적). 작고 additive; 거짓말하는 UI를 죽임. TTL 기계 불필요(연결 끊김 unsubscribe + 하트비트
   write 실패로 ≤10–15 s 신선도).
4. **Axis 3-Next — reconcile 전용 워커 분리.** `exact_lane.reconcile` job을 둘째 JobRunner
   스레드로 라우팅 — 유지보수 풀솔브가 live chase 봉쇄 금지. **fly vm CPU 여유 측정 후에만.**
   워크스페이스별 1-running-+-1-pending coalescing 후보는 **기각** — 이미 구현됨
   (single-flight `jobs.py:1424-1444`, supersede `store.py:2298-2309` + `jobs.py:1691-1743`,
   audience gate `jobs.py:1292-1320`).
5. **Axis 4-Next — 스냅샷 + prune_floor + canonical/results 보존.** 디스크 벽은 op 테이블이 아니라
   results payload 파일 + `realtime_canonical_state`(리비전당 풀 상태 JSON). 스냅샷 계약이 프루닝보다
   먼저; floor = min(terminal epoch들의 consumed_seq, 가장 오래된 원장 앵커) − audit tail
   (원장-앵커 부분폴드 위험 `executor_bridge.py:1518-1533`); results는 re-baselining으로만
   (조성은 target 이하 전 버전 디코드 `jobs.py:2606-2657`).
6. **Axis 5 — `status_copy.mjs` 추출(Now).** `rt_submit.mjs`/`rt_adoption.mjs`는 우선순위 1–3이 그
   섹션을 만질 때 기회적으로. 위생 작업 — 위 항들 뒤에 순서화.

**합의된 기각.** join/leave presence 이벤트(하트비트 full-state가 유실 모델에서 우위),
워크스페이스 큐 coalescing 신설(이미 존재), blanket op-table truncation(원장 앵커 부분폴드 위험),
프론트 리라이트/프레임워크(compare-guarded write 규율이 곧 계약).

## 불변 (어떤 제안도 이것을 깨지 않는다)

- 과학 제약 (원문): "물리적 읽기 범위를 축소하거나, 6° 하늘 하한을 올리거나, 비트 단위 동등성을
  주장하면서 누적을 재정렬하거나, 정밀도를 조용히 낮추지 마십시오."
- ops 내구성: 수용 시 append, 스킵돼도 원장 유지 (`routes_realtime.py:91-97`).
- publish 펜스 exactly-once; unfoldable-op은 요란한 실패.
- ETA 숫자 발명 금지 — p80 이력 또는 정적 폴백 basis만.
- 배지·presence 카피 정직성 (측정한 것만 말한다).
- 데이터 정책: 사이트 데이터 git/이미지 금지, 명시적 파일 스테이징만.
