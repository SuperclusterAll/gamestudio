# 모듈별 설명

각 파일이 무엇을 책임지고, **왜 그 자리에 있는지**. 전체 그림은 [ARCHITECTURE.md](ARCHITECTURE.md),
설계 판단의 근거는 [DESIGN-DECISIONS.md](DESIGN-DECISIONS.md).

의존 방향은 이렇다. 화살표는 "import 한다"이고, 순환이 없다.

```mermaid
graph BT
    models["models.py<br/>스키마 · 상태"]
    prompts["prompts.py<br/>역할 프롬프트"]
    required["required_art.py<br/>아트 판정"]
    sprites["sprites.py<br/>이미지 후처리"]
    comfyui["comfyui.py<br/>워크플로"]

    agents["agents.py<br/>모델 어댑터 · 정적 QA"]
    godot["godot.py<br/>엔진 어댑터"]
    tools["agent_tools.py<br/>HTML 도구"]
    gtools["godot_tools.py<br/>Godot 도구"]
    codeagent["code_agent.py<br/>에이전트 조립"]
    graph["graph.py<br/>오케스트레이션"]
    server["server.py<br/>대시보드"]
    cli["cli.py"]

    agents --> models
    agents --> prompts
    agents --> required
    godot --> required
    tools --> agents
    tools --> sprites
    tools --> comfyui
    gtools --> tools
    gtools --> godot
    codeagent --> agents
    graph --> codeagent
    graph --> tools
    graph --> gtools
    graph --> godot
    server --> graph
    cli --> graph
```

---

## `models.py` — 계약

**책임**: 에이전트 간에 오가는 모든 것의 타입.

| 스키마 | 쓰임 |
|---|---|
| `GameConcept` | 기획 Agent 산출 — 제목·루프·조작·참조 게임 |
| `ImplementationPlan` | **구현 계약** — 메커닉·승패조건·상태전이·수용테스트 |
| `ArtDirection` | 팔레트·이미지 프롬프트·에셋 후보 목록 |
| `RequirementCheck` / `DesignReview` | 설계 감사 결과 |
| `QAReport` | 검증 판정 (pass/repair + findings) |
| `SupervisorDecision` | 에스컬레이션 판단 (action + reason + instructions) |
| `StudioState` | LangGraph 상태 — 노드 간 계약 전부 |

**왜 이렇게**: 워커들이 **타입이 있는 객체를 그대로** 주고받는다. 메시지 문자열로 직렬화했다
재파싱하지 않는다. `ImplementationPlan`이 QA가 대조할 계약이 되고, 그 계약이 곧 게임의 정의다.

**흉터 두 개**가 스키마에 남아 있다.

- `ArtDirection.palette` 등에 `default_factory` — 없으면 "추가 효과 없음"이 필드 누락으로 와서 검증 실패
- `RequirementCheck.evidence`가 선택 — 모든 요구사항에 근거를 요구했더니 출력이 계약 크기에 비례해
  자라다가 예산을 넘겨 배열 중간에서 잘렸다. 거부한 항목만 이유를 쓴다.

---

## `prompts.py` — 역할 정의

**책임**: 시스템 프롬프트 6종 + 장르별 레퍼런스 게임 표.

`IDEA_SYSTEM`의 맨 앞 문단이 중요하다.

> 플레이어의 요청이 여기 다른 모든 지침보다 우선한다. 특정 게임을 재현해 달라면 재현하라 —
> 같은 규칙, 같은 조작, 같은 승패 조건. 변형을 더하지 말고, 더 독창적인 쪽으로 틀지 말라.

`GENRE_REFERENCES`는 장르별로 "빌릴 만한 검증된 루프"를 담은 표다 — **15개 장르 × 5~7개, 총 87개**.
장르가 정해지면 그 행에서 **런 시드로 3개를 뽑아** 프롬프트에 넣는다. 행이 통째로 들어가던 시절에는
장르 하나가 곧 고정된 대표작 3개였고, 같은 장르의 두 실행이 늘 같은 자리에서 출발했다.

각 항목은 `이름(한글 표기): 빌릴 메커닉` 꼴이다. 한글 표기는 장식이 아니라 두 군데서 쓰인다 —
플레이어가 이름을 댄 게임을 셔플에서 지켜 내고(테트리스를 요청했는데 비쥬얼드가 대신 붙는 일 방지),
드롭박스를 안 건드린 브리프가 어떤 장르를 말하는지 읽어 낸다.

`GENRE_KEYWORDS`는 그 읽어 내기에 쓰는 루프 단어 표다. 테마어가 아니라 루프어만 담는다 — "우주"는
슈팅일 수도 레이싱일 수도 있지만 "편대·탄막·발사"는 한 가지 루프다.

---

## `agents.py` — 모델 어댑터 (990줄)

가장 큰 파일이고, 성격이 다른 다섯 가지가 들어 있다.

### 1. 모델 생성

```python
_bedrock_client()   # (service, region) 당 하나. 생성에 ~2초 걸림
_build_model()      # (model, region, max_tokens) 당 하나. lru_cache
_model()            # 위를 감싼 진입점
qa_model_id()       # 검증·판단용 모델을 한 곳에서 결정
cache_control_for() # Anthropic 모델에만 프롬프트 캐시 포인트
```

`qa_model_id()`가 한 곳인 이유: 설계 감사와 에스컬레이션 판단이 **같은 모델**을 써야 한다.
한 런이 A 모델로 감사하고 B 모델로 그 감사에 대한 조치를 정하면 안 된다.

### 2. `StreamAccumulator` — 스트리밍 누적

```python
class StreamAccumulator:   # 조각을 리스트에 append, 마지막에 한 번만 조립
def stream_turn(model, messages, on_preview=None)
```

**왜 직접 만들었나**: `gathered = gathered + piece`가 모든 LangChain 예제의 모양이지만
이 길이의 답변에는 함정이다. `+` 마다 `AIMessageChunk`가 새로 만들어지고, 그때마다 누적된
툴 인자 전체를 다시 파싱한다. 24KB 게임 하나 누적에 **1.112초 → 0.0007초**(실측).

### 3. `_structured` — 구조화 출력

스키마 검증 실패 시 재시도하되, **원인에 따라 다르게** 대응한다.

- 진짜 스키마 위반 → 검증기 불만을 붙여 같은 예산으로 재요청
- 출력 예산 초과로 잘림 → **예산을 2배로 키우고** "더 짧게 답하라"로 지시 변경

둘을 구분하지 않으면, 잘림이 원인일 때 3번 다 같은 자리에서 실패한다.

### 4. `static_qa` — HTML 결정론적 검증

키워드 스캔 + node로 JS 문법 파싱. 내용별 캐시가 걸려 있다(같은 HTML = 같은 답).

차단하는 것은 게임을 못 돌게 하는 것뿐이다. `restart()` 버튼이 있는데 "재시작 없음"으로 읽혀
멀쩡한 게임을 떨어뜨린 적이 있어서, 니즈 검사는 전부 참고로 내렸다.

### 5. 기획 어댑터

```python
create_concept()      # 자동 장르 배정 · 최근 산출물 회피 · 플레이어 요청 우선
create_art()          # 재수립이면 지적사항 + 기존 스프라이트를 받음
run_deep_director()   # Deep Agents 감독 — 스텝·시간·토큰 예산을 직접 강제
player_requested()    # 사람이 직접 쓴 브리프인지 판별
resolve_auto_genre()  # 런 시드 → 장르 배정 (브리프가 비었을 때만)
infer_genre()         # 플레이어가 쓴 문장 → 장르 (이름 댄 게임 우선, 없으면 루프 단어 2개 이상)
genre_label()         # 이 브리프가 속한 표의 행
sample_references()   # 그 행에서 런 시드로 3개 — 이름 댄 게임은 항상 남김
recent_productions()  # 출력 폴더의 매니페스트를 장기 기억으로
```

---

## `graph.py` — 오케스트레이션 (1309줄)

**책임**: 노드 정의, 라우팅, 예산 강제, 엔진 분기.

### 노드

| 노드 | 하는 일 |
|---|---|
| `supervisor_node` | 유일한 라우팅 지점 |
| `idea_node` | 컨셉 |
| `design_document_node` | 구현 계약 + 리뷰용 기획서 |
| `approval_node` | `interrupt()` — 사람 승인 |
| `art_node` | 아트 계획 (+ 재수립 시 이미지 직접 생성) |
| `code_node` | 코드 Agent 실행 |
| `qa_node` | 2단 검증 |
| `repair_node` | 텍스트 재작성 (HTML 전용) |
| `package_node` / `abandoned_node` / `rejected_node` | 종료 |

### 라우팅 핵심

```python
_NEXT_AFTER = {"idea": "design_document", "design_document": "approval",
               "art": "code", "code": "qa", "repair": "qa"}
```

이 딕셔너리가 "제작 사다리"다. 모델이 개입하지 않는다.

```python
_affordable_actions(state)  # 예산이 지불 가능한 수만 남긴다
_fallback_action(...)       # 감독이 못 쓸 수를 고르면 덮어씀
_needs_new_art(findings)    # "없는 그림"을 요구하는 지적이면 art로 강제
```

`_affordable_actions`가 **선택지 자체를 지운다**는 게 핵심이다. 프롬프트로 "쓰지 마세요"라고
부탁하는 것과 다르다.

### 엔진 분기

```python
HTML5, GODOT = "html5", "godot"
_engine(state)      # 인식 못 하는 값은 HTML5 (옛 체크포인트 호환)
_is_godot(state)
```

분기 지점은 **딱 세 곳**이다 — `code_node`(도구·프롬프트), `qa_node`(검증), `package_node`(산출물).
기획·승인·아트는 전부 공유한다.

### QA 판정

```python
@dataclass(frozen=True)
class _Verdict:
    unmet: list[str]       # 우리 계약 중 리뷰어가 명시적으로 거부한 것
    advisories: list[str]  # 기타 지적 — 기록만
    unreviewed: list[str]  # 언급조차 안 된 것
    blocking: bool
```

`unmet`이 우리 계약과 **실제로 매칭된 것만** 센다. 예전엔 리뷰어가 스스로 지어낸 요구사항까지
거부 비율에 들어가서, 실제 계약은 다 충족한 빌드가 차단됐다.

---

## `code_agent.py` — 에이전트 조립

`create_agent`(LangChain) 위에 미들웨어를 얹는다. 예전에 손으로 돌리던 루프가 전부 여기로 왔다.

| 손으로 하던 것 | 지금 |
|---|---|
| 도구 호출 예산 | `ModelCallLimitMiddleware(run_limit=20)` |
| 일시 오류 재시도 | `ModelRetryMiddleware` |
| 이력 압축 | `ContextEditingMiddleware(ClearToolUsesEdit)` |
| 진행 상황 보고 | `StudioObservability.wrap_model_call` |
| 긴 빌드의 계획 유지 | `TodoListMiddleware` |

`ContextEditingMiddleware`의 trigger가 8,000으로 **게임 하나 크기보다 일부러 낮다**.
20KB 게임이 `write_game_file` 인자로도, `read_game_file` 결과로도 이력에 남아 매 턴 재전송되면
한 루프에서 265k 입력 토큰을 쓴다.

`wrap_model_call`은 모델을 직접 스트리밍한다 — 긴 생성이 화면에 보이게 하려고. 실패하면
에이전트 자신의 호출로 조용히 넘어간다. 진행 표시 기능 때문에 빌드가 죽으면 안 된다.

---

## `agent_tools.py` — HTML 도구 (438줄)

| 도구 | 역할 |
|---|---|
| `write_game_file` | 게임 전체 쓰기 **+ 정적 QA 판정 동봉** |
| `read_game_file` | 부분 읽기 (`outline=True`로 구조 맵) |
| `run_static_qa` | 자기가 쓰지 않은 초안 재검사용 |
| `repair_html` | 수정 + 판정 동봉 |
| `list_game_assets` | 파일명이 아니라 **그리기 계약**을 반환 |
| `generate_comfyui_image` | 스프라이트 생성 (배경 제거 · 방향 계약 포함) |

**`write`가 판정을 함께 돌려주는 이유**: `static_qa`는 모델 호출이 없고 캐시까지 된다.
그걸 물어보려고 턴을 쓰면 빌드 루프 왕복이 두 배가 된다.

`list_game_assets`가 계약을 반환하는 이유: 파일명만으로는 배경이 제거됐는지, 어느 방향으로
그려졌는지 알 수 없다. 방향을 모르면 진행 방향으로 한 번 더 회전시켜 엉뚱한 곳을 보게 된다.

`generate_comfyui_image`의 `facing` 인자가 방향 계약의 출발점이다. 이미지에서 방향을
**감지하지 않는다** — 느리고 자주 틀린다. 생성 시점에 고정하고 그 계약을 코드까지 전달한다.

---

## `godot.py` — 엔진 어댑터 (426줄)

바이너리에 대해 실측한 두 사실이 설계를 결정했다.

- `--headless --quit-after N`: 파스·런타임·리소스 오류를 stderr로 보고하지만 **exit 0**.
  종료 코드에 정보가 없고 출력만이 신호다.
- `--headless --check-only --script`: 파스 에러에 **exit 1**. 유일하게 믿을 수 있는 종료 코드다.

```python
check_scripts()       # 파일 단위 컴파일 (exit code로 게이트)
run_project()         # 헤드리스 실행 후 stderr 파싱
static_project_qa()   # 구조 검사 — 엔진이 못 보는 것
parse_errors()        # 메시지 + at: 줄을 합쳐 주소 있는 findings로
export_web()          # 기회주의적 웹 빌드
write_launch_script() # run.bat
```

**`static_project_qa`가 왜 필요했나**: 엔진은 던져지는 것만 본다. `func _ready(): pass`뿐인
프로젝트는 아무것도 안 던진다 — 임포트되고, 5초 돌고, exit 0이고, **통과로 보고됐다.**
입력도 점수도 승패도 없는데. HTML에 `static_qa`가 있는 이유와 같은 자리다.

`run.bat`은 `-e`를 주지 않는다. Godot 도움말이 `-e/--editor`를 "씬 실행 대신 편집기 실행"으로
정의하므로, **없는 것이 곧 실행 버튼**이다. 경로는 `%~dp0.`이라 폴더를 옮겨도 동작한다.

---

## `godot_tools.py` — Godot 도구

HTML 도구와 같은 모양이되 **경로로 주소를 매긴다.** Godot 게임은 디렉터리이기 때문이다.

경계가 둘 있다.

```python
WRITABLE_SUFFIXES = {".gd", ".tscn", ".tres", ".godot", ".cfg", ...}
_resolve(state, path)   # 워크스페이스 밖으로 못 나감
```

- **경로**: 모델이 자유 텍스트로 주므로 형식이 아니라 실제 경계다. `../`나 절대경로는 거부.
  단, 앞의 `/`는 `res://`처럼 프로젝트 상대로 **가둔다** — 탈출이 아니다.
- **확장자**: `.res`·`.import` 같은 엔진 소유 바이너리를 모델이 손으로 쓰면 아무도 설명 못 하는
  방식으로 깨진다.

워크스페이스가 곧 Godot 프로젝트 루트다. 그래서 `generate_comfyui_image`가 만든 스프라이트가
`<workspace>/assets`에 떨어지고 `res://assets/<name>.png`로 바로 참조된다 — 복사 단계도,
동기화할 두 번째 위치도 없다.

---

## `sprites.py` — 이미지 후처리 (288줄)

두 문제를 푼다. 같은 문제의 양끝이다.

**1. 텍스트-이미지 모델은 알파 채널을 못 그린다.** 항상 꽉 찬 사각형이 온다.

크로마키를 프롬프트로 **요구**하고, 색 판정 + 연결성으로 잘라낸다. 색만 보면 피사체의 초록 부분에
구멍이 뚫리고, 연결성만 보면 비슷한 색 피사체를 관통한다. 둘 다 필요하다.

판정 기준은 "순수 `#00FF00`에 가까운가"가 아니라 **"초록이 지배적인가"**다. 실측한 배경이
`(6,224,10)`, `(22,254,91)`, `(117,212,113)` — 명백히 초록이지만 순수 키는 한 번도 아니었다.

**2. 모델은 "앞"이 어딘지 모른다.** 같은 우주선을 두 번 요청하면 오른쪽, 위, 3/4 뷰가 나온다.

픽셀에서 방향을 감지하지 않는다. **생성 시점에 계약으로 고정**하고 그 계약을 코드까지 전달한다.

캔버스 회전이 벡터 v를 (cos·vx − sin·vy, sin·vx + cos·vy)로 보내고 `atan2(vy,vx)`는 오른쪽
이동에서 0이므로 — 오른쪽을 향해 그린 아트는 보정각 0, 위(−Y)를 향한 아트는 +π/2다.

---

## `required_art.py` — 아트 판정

작지만 두 엔진이 공유하는 정의가 들어 있다.

```python
required_assets(asset_plan, existing)  # 재수립이 의무화한 스프라이트
missing_required(required, assets_dir) # 아직 안 만들어진 것
unused_sprites(body, sprites)          # 만들었는데 안 쓰는 것
```

`unused_sprites`가 양방향으로 보수적이다. 테트리스는 조각을 루프로 다룬다 —
`load("res://assets/block-" + kind + ".png")` — 그래서 리터럴 `block-i.png`가 소스에 없다.
`res://` 경로를 조립하는 코드가 하나라도 있으면 **아무것도 보고하지 않는다.** 읽어서 알 수 없는
문제이고, 틀린 "낭비했습니다"는 rethink 한 바퀴를 태워야 반증된다.

---

## `server.py` — 대시보드 (590줄)

FastAPI + WebSocket. 로컬 전용이다.

| 엔드포인트 | |
|---|---|
| `POST /api/runs` | 실행 시작 |
| `POST /api/runs/{id}/decision` | 승인/거부 |
| `POST /api/runs/{id}/launch` | **로컬 프로세스 시작** (Godot) |
| `GET /api/model-status` | Bedrock·ComfyUI·Godot 가용성 |
| `GET /games/{id}/...` | 완성된 게임 서빙 |
| `WS /ws` | 실시간 |

`launch`가 유일하게 로컬 프로세스를 띄우므로 의도적으로 좁다 — **요청에서 명령줄로 가는 값이
하나도 없다.** 실행 가능한 건 파이프라인이 직접 쓴 `run.bat` 뿐이고, 위치는 `GAME_OUTPUT_ROOT/<run_id>/`
고정이며 `run_id`는 패턴 검사를 통과해야 한다.

체크포인터가 SQLite인 이유: 승인 게이트는 durable interrupt인데 뒤에 메모리 세이버가 있으면
**프로세스와 함께 죽는다.** 기획서를 두고 퇴근했다가 재시작하면 그 런은 복구 불가였다.

`Run.public()`이 `game_html`·`design_review`·`messages`를 뺀다. 초당 여러 번 나가는 페이로드다.

---

## `cli.py` / `comfyui.py` / `fallback_game.py`

- **`cli.py`** — 대시보드 없이 한 판 돌리는 진입점. 같은 그래프를 쓴다.
- **`comfyui.py`** — 제공된 UI 워크플로 JSON을 `/prompt` API 형식으로 컴파일.
- **`fallback_game.py`** — 테스트 픽스처. **제작에 쓰이지 않는다.** 모델이 안 되면 실행이 실패하지,
  고정 게임으로 대체하지 않는다.

---

## 테스트 (2594줄 · 119개)

| 파일 | 고정하는 것 |
|---|---|
| `test_hitl_graph.py` | 승인 게이트, 라우팅, 예산, QA 판정, 감독 결정 |
| `test_godot_mode.py` | 엔진 옵션 경계 — HTML 경로가 안 건드려짐 |
| `test_godot_qa.py` | Godot 구조 검사 |
| `test_qa_policy.py` | QA 강도 · 모델 선택 · 잘림 재시도 |
| `test_required_art.py` | replan/re-execute 루프와 강제 |
| `test_efficiency.py` | 캐싱 · 호출 절감 · 지속성 · 요청 우선순위 |
| `test_sprites.py` | 배경 제거 · 방향 계약 |
| `test_game_contract.py` / `test_dashboard_surface.py` | 정적 QA · 대시보드 표면 |

테스트 독스트링이 **왜 그 동작인지**를 담고 있다. 대부분 실제로 겪은 실패에서 나왔다.
