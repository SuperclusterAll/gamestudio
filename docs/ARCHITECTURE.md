# 아키텍처

사람이 기획서를 승인하면, 여러 AI 에이전트가 실제로 플레이 가능한 게임을 만들어 내는 파이프라인.
HTML5 Canvas 단일 파일 또는 Godot 4 프로젝트 중 하나로 산출한다.

- **모듈별 상세** → [MODULES.md](MODULES.md)
- **왜 이렇게 설계했는가** → [DESIGN-DECISIONS.md](DESIGN-DECISIONS.md)

---

## 1. 무엇을 만드는 시스템인가

한 번의 실행(run)은 이렇게 끝난다.

| 엔진 | 산출물 | 검증 방식 |
|---|---|---|
| `html5` | `index.html` 한 장 (외부 의존성 0) | 키워드 스캔 + JS 문법 파싱 |
| `godot` | Godot 4 프로젝트 + `run.bat` (+ 가능하면 웹 빌드) | **엔진이 실제로 실행** |

두 경로는 **기획 단계를 전부 공유**하고, 산출물이 달라지는 세 단계(코드·검증·패키징)에서만 갈라진다.

### 설계의 중심 원칙 세 가지

**1. 사람의 승인이 진짜 게이트다.**
기획서가 나오면 그래프가 멈춘다(`interrupt()`). 승인 없이는 아트도 코드도 시작하지 않는다.
이 대기는 SQLite 체크포인트에 저장되므로 대시보드를 재시작해도 살아남는다.

**2. 판단이 필요한 곳에만 모델을 쓴다.**
supervisor는 8번 방문에 모델 호출 1번이다. 나머지는 딕셔너리 조회다.

**3. 검증은 막을 수 있는 것만 막는다.**
게임이 동작하지 않게 만드는 것만 차단하고, 나머지는 기록만 남긴다.
"멀쩡한 게임을 QA가 떨어뜨려 예산을 태우는" 실패를 여러 번 겪고 나온 원칙이다.

---

## 2. 전체 구조

```mermaid
graph TB
    subgraph client["브라우저"]
        UI["대시보드<br/>index.html / app.js"]
    end

    subgraph server["FastAPI 서버 (로컬)"]
        API["REST + WebSocket"]
        SVC["StudioService<br/>실행 구동 · 이벤트 중계"]
    end

    subgraph graph_["LangGraph 파이프라인"]
        SUP{{"supervisor<br/>단일 라우팅 허브"}}
        W["워커 노드 7개"]
    end

    subgraph data["data/ — 스튜디오 자신의 상태 (git 무시)"]
        CK[("SQLite 체크포인트<br/>승인 대기가 재시작을 넘김")]
        VDB[("ChromaDB 아트 메모리<br/>프롬프트 · 역할 · 판정")]
    end

    subgraph external["외부"]
        BR["Amazon Bedrock<br/>Sonnet 4.6 / Haiku 4.5"]
        CF["ComfyUI<br/>스프라이트 생성"]
        GD["Godot 4.7<br/>헤드리스 검증"]
    end

    FS[("게임 산출물<br/>GAME_OUTPUT_DIR")]

    UI <-->|"WebSocket 실시간"| API
    API --> SVC
    SVC <--> CK
    SVC --> SUP
    SUP <--> W
    W --> BR
    W --> CF
    W --> GD
    W --> FS
    W -->|"생성한 프롬프트 저장"| VDB
    W -->|"RAG 조회 — 아트 기획"| VDB
    UI -->|"게임 실행 · 재개발 요청"| API
    UI -->|"이미지 좋음/별로"| API
    API -->|"사람 판정 기록"| VDB
```

### 실행 흐름 (정상 경로)

```mermaid
sequenceDiagram
    actor U as 사용자
    participant D as 대시보드
    participant G as 그래프
    participant M as Bedrock
    participant V as 아트 메모리<br/>(ChromaDB)
    participant F as 워크스페이스

    U->>D: 장르 · 엔진 · 브리프 입력
    D->>G: 실행 시작
    G->>M: 총괄 감독 (범위 결정 1회, 끌 수 있음)
    G->>M: 기획 Agent → GameConcept
    G->>M: 기획 문서 → ImplementationPlan
    G-->>D: ⏸ 기획서 승인 대기 (체크포인트 저장)
    U->>D: 승인
    D->>G: 재개

    rect rgba(80,140,200,.12)
        note over G,V: RAG — 지난 런에서 통한 표현을 먼저 꺼낸다
        G->>V: recall(visual_direction, 장르)
        V-->>G: 좋게 평가된 프롬프트 3개 (bad 제외)
    end
    G->>M: 아트 기획 (예시 포함) → ArtDirection

    loop 코드 Agent 도구 루프 (최대 50 호출)
        G->>M: 다음 행동 결정
        G->>F: 파일 쓰기 / 검증
        opt 스프라이트가 필요하면
            G->>F: ComfyUI 생성 + 배경 제거
            G->>V: remember(프롬프트 · 역할 · 장르 · 기하학)
            V->>V: 자동 판정 — 120px 미만이면 bad
        end
    end
    G->>G: QA 1단 — 결정론적 검증
    G->>M: QA 2단 — 설계 감사 (Haiku)
    G->>F: 패키징 + 매니페스트 (+ 소비량 기록)
    G-->>D: 완료

    rect rgba(115,230,210,.12)
        note over U,V: 피드백 루프 ① 이미지 — 좋은 프롬프트가 다음 런으로
        D->>V: GET 지금 쓰는 스프라이트 + 프롬프트
        U->>D: 좋음 / 별로
        D->>V: judge(verdict_by="human")
    end

    rect rgba(255,200,120,.12)
        note over U,G: 피드백 루프 ② 게임 — 기획은 그대로, 코드부터 다시
        U->>D: 보완점 입력 ("점프가 무겁다")
        D->>G: revise — 기획·계약·워크스페이스 유지
        G->>G: stage="art" 로 진입 → code → QA → 패키징
        note over V: 재생성된 스프라이트는 판정이 초기화되고<br/>사람이 좋다고 한 옛 프롬프트는 보관된다
    end
```

---

## 3. 그래프 토폴로지

실제 컴파일된 그래프를 그대로 옮긴 것이다 (`build_graph().get_graph().draw_mermaid()`).

```mermaid
graph TD
    START([시작]) --> SUP
    SUP{{supervisor}}

    SUP -.-> idea
    SUP -.-> design_document
    SUP -.-> approval
    SUP -.-> art
    SUP -.-> code
    SUP -.-> qa
    SUP -.-> repair
    SUP -.-> package
    SUP -.-> abandoned
    SUP -.-> rejected

    idea --> SUP
    design_document --> SUP
    approval --> SUP
    art --> SUP
    code --> SUP
    qa --> SUP
    repair --> SUP

    package --> END([끝])
    abandoned --> END
    rejected --> END
```

**허브-스포크 구조다.** 모든 워커가 supervisor로 돌아오고, 간선을 고르는 곳은 supervisor 하나뿐이다.
이건 supervisor 패턴이 맞다 — 다만 **라우팅을 LLM이 아니라 코드가 결정한다**는 점이 `create_supervisor` 같은
라이브러리 구현과 다르다. 이유는 [DESIGN-DECISIONS.md §1](DESIGN-DECISIONS.md#1-supervisor-패턴을-직접-구현한-이유)에 있다.

### supervisor가 실제로 판단하는 것

| 진입 상황 | 동작 | 모델 호출 |
|---|---|---|
| 첫 방문 | 총괄 감독이 프로덕션 브리프 작성 | **있음** |
| `stage == approval` | 사람의 결정을 그대로 분기 | 없음 |
| `stage == qa` + 통과 | package로 | 없음 |
| `stage == qa` + 실패 | `_escalate` — 다음 조치 결정 | **있음** |
| 그 외 전부 | `_NEXT_AFTER[stage]` 조회 | 없음 |

실측: supervisor 방문 8회 중 모델 호출 1회. (+ QA 실패 시 1회)

---

## 4. 파이프라인 단계

```mermaid
flowchart LR
    R["참조 이미지 (선택)<br/>비전 호출 1회 → 글"] -.-> A
    A["기획 Agent<br/>GameConcept"] --> B["기획 문서<br/>ImplementationPlan"]
    B --> C{"사람 승인"}
    C -->|승인| D["아트 기획<br/>ArtDirection"]
    C -->|거부| X["중단"]
    D --> E["코드 Agent<br/>게임 제작 + 이미지 생성"]
    E --> F["QA 검증"]
    F -->|통과| G["패키징"]
    F -->|실패| H{"감독 판단"}
    H -->|code| E
    H -->|art| D
    H -->|repair| I["자동 수정"] --> F
    H -->|예산 소진| J["미통과 배포"]
    G --> K{"사람이 플레이"}
    K -->|보완 요청| E
    K -->|이미지 별로| D
```

재개발은 **같은 폴더의 같은 파일**로 돌아온다. 이미지 시간 예산은 새로 받고, 다시 그리는 것은
사람이 "별로"라고 한 스프라이트뿐이다.

### 단계별 모델 사용

| 단계 | 모델 | 출력 예산 | 호출 |
|---|---|---|---|
| 총괄 감독 | Sonnet 4.6 | 1,024 | 런당 1회 · 범위 결정만 (`DIRECTOR_TIMEOUT_SECONDS=0`이면 생략) |
| 기획 Agent | Sonnet 4.6 | 8,000 | 1회 |
| 기획 문서 | Sonnet 4.6 | 8,000 | 1회 |
| 아트 기획 | Sonnet 4.6 | 8,000 | 1회 (+ 재수립 1회) |
| **코드 Agent** | Sonnet 4.6 | 16,000 | **최대 20회** |
| QA 설계 감사 | **Haiku 4.5** | 12,000 | 정적 검증 통과 시 |
| 감독 에스컬레이션 | **Haiku 4.5** | 2,000 | QA 실패 시만 |
| 자동 수정 | Sonnet 4.6 | 32,000 | 감독이 선택 시만 (HTML 전용) |

시간·비용의 압도적 다수는 **코드 Agent**다. 나머지를 다 합쳐도 못 미친다.

---

## 5. 상태 모델

`StudioState`(TypedDict)가 노드 간 계약 전부다. 타입이 있는 Pydantic 객체를 그대로 주고받는다 —
메시지 문자열로 직렬화했다 재파싱하지 않는다.

```mermaid
graph LR
    subgraph routing["라우팅 계약"]
        S1[stage] --- S2[next_step]
    end
    subgraph artifacts["산출 계약"]
        A1[concept] --> A2[implementation_plan]
        A2 --> A3[design_document]
        A3 --> A4[approval]
        A4 --> A5[art]
        A5 --> A6["game_html /<br/>godot_project_path"]
        A6 --> A7[qa / design_review]
    end
    subgraph budget["예산 계량기"]
        B1[rethink_cycles]
        B2[repair_attempts]
        B3[art_revised]
        B4[required_assets]
    end
```

**의도적으로 넣지 않는 것**: 코드 Agent의 대화 이력(`messages`).
읽는 곳이 없는데 게임 전체를 툴 인자로 품고 있어서, 체크포인트마다 재직렬화되고 rethink마다 누적됐다.

---

## 6. 검증 체계

같은 2단 구조를 두 엔진이 공유하되, 1단의 성격이 다르다.

```mermaid
flowchart TD
    subgraph s1["1단 — 결정론적 (모델 없음)"]
        H1["HTML: 키워드 스캔<br/>+ node로 JS 문법 파싱"]
        G1["Godot: 프로젝트 구조 검사<br/>→ 스크립트 컴파일<br/>→ 헤드리스 실행"]
    end
    subgraph s2["2단 — 설계 감사 (Haiku)"]
        R["구현 계약 대조<br/>요구사항별 통과/미통과"]
    end
    V{"_verdict<br/>판정"}

    H1 -->|통과| R
    G1 -->|통과| R
    R --> V
    V -->|"미충족 > 50%"| FAIL["repair"]
    V -->|"그 이하"| PASS["pass<br/>(미충족 항목은 기록)"]
```

**Godot이 질적으로 강하다.** HTML은 실행할 수 없는 게임을 키워드로 추측하지만,
Godot은 엔진이 실제로 돌려서 없는 노드·null 역참조를 **파일·줄 번호와 함께** 잡는다.

### 차단 기준

| | 차단 | 기록만 |
|---|---|---|
| HTML | canvas/루프/키입력 부재, 외부 의존성, JS 문법 오류, 불완전 HTML | 점수·재시작 미검출, 미사용 스프라이트 |
| Godot | main_scene 부재, `_process` 없음, 입력 처리 없음, 네트워크 API, 깨진 `res://` 경로 | WASD 미지원, 점수·재시작 미검출, 미사용 스프라이트 |
| 공통 | 계약의 50% 초과 미충족, 감사 결과 공백 | 그 이하 미충족, 리뷰어의 기타 지적(최대 5건) |

---

## 7. 에스컬레이션 예산

QA 실패 시 감독이 쓸 수 있는 수는 코드로 잘라낸다. 프롬프트로 부탁하지 않는다.

```mermaid
stateDiagram-v2
    [*] --> QA실패
    QA실패 --> 선택지계산: _affordable_actions
    선택지계산 --> code: rethink 남음
    선택지계산 --> art: rethink 남음 + 미재수립
    선택지계산 --> repair: repair 남음 + HTML
    선택지계산 --> abandoned: 선택지 없음
    code --> [*]: 코드 Agent 재실행
    art --> [*]: 아트 재수립 → code
    repair --> [*]: 텍스트 재작성
    abandoned --> [*]: 미해결 기록 후 배포
```

| 예산 | 기본값 | 비고 |
|---|---|---|
| `MAX_RETHINK_CYCLES` | 1 | 가장 비싼 수 — 코드 Agent 루프 전체 + 재감사 |
| `MAX_REPAIR_ATTEMPTS` | 1 | HTML 전용. Godot은 여러 파일이라 성립 안 함 |
| 아트 재수립 | 런당 1회 | `art_revised` 플래그 |

**Godot은 `repair`가 없어 실질 재시도가 rethink 1회뿐이다.** 부족하면 `MAX_RETHINK_CYCLES=2`.

---

## 8. 아트 파이프라인

`art` 노드는 **그림을 그리지 않는다.** 목록(`asset_plan`)만 쓴다. 실제 생성은 code가 한다 —
만들면서 무엇이 필요한지 발견하기 때문이다.

예외가 하나 있다. **재수립일 때는 art가 직접 생성한다.**

```mermaid
flowchart TD
    A["QA: 적 스프라이트가 없다"] --> B["supervisor<br/>(감독이 code를 골라도 art로 강제)"]
    B --> C["art 재수립<br/>required_assets 산출"]
    C --> D["ComfyUI 직접 호출<br/>모델 호출 0회"]
    D --> E["배경 제거 + 트리밍"]
    E --> F["code: 게임에 그려 넣기만"]
```

생성된 PNG는 네 가지 후처리를 거친다.

1. **크로마키 배경 제거** — 프롬프트가 크로마 배경을 요구하고, 색 판정 + 연결성으로 잘라낸다.
   피사체가 초록이면 배경을 **매젠타**로 요청한다(초록 슬라임은 그린스크린과 함께 잘려나간다).
   잘라내는 쪽은 어느 키를 요청했는지 묻지 않고 **실제로 칠해진 색**을 읽는다
2. **알파 바운딩박스 트리밍** — 640px 프레임 속 130px 차를 그대로 쓰면 자기 여백 안의 점이 된다
3. **배경이 남았으면 다시 뽑기** — 제거 거부·90% 이상 불투명·잘라낸 결과가 원본 캔버스 그대로,
   이 셋을 전부 보고 다른 시드로 한 번 더 시도한다
4. **방향 계약 기록** — `sprites.json`에 facing을 남겨 코드가 회전 보정각을 알 수 있게 한다

애니메이션은 **프레임을 한 장에 몰아 그린 뒤 잘라낸다.** 따로 뽑으면 프레임마다 다른 캐릭터가
오는데, 한 장 안에서는 화풍이 드리프트할 수 없다. 잘라낸 뒤 좌우가 뒤집힌 프레임을 되돌리고,
혼자 다른 색인 프레임과 중복 포즈를 버린다.

| `facing` | 생성 방향 | 게임에서 |
|---|---|---|
| `right` (기본) | +X | `ctx.rotate(Math.atan2(vy, vx))` |
| `up` | −Y | `ctx.rotate(Math.atan2(vy, vx) + Math.PI/2)` |
| `none` | 없음 | 회전 금지 |

### 아트 메모리 — 결과로 배우는 RAG

아트 기획은 `image_prompt`를 **아무 시각적 근거 없이** 썼다. 스무 번째 런이 첫 번째 런만큼이나
맹목적이었고, 프롬프트가 PNG와 함께 버려졌기 때문에 나아질 방법이 없었다.

```mermaid
flowchart LR
    subgraph 생성["생성 (런 N)"]
        A["아트 기획<br/>image_prompt 작성"] --> B["ComfyUI 생성"]
        B --> C["배경 제거 + 트리밍"]
        C --> D["프롬프트 + role + 장르<br/>+ 기하학 저장"]
    end
    subgraph 판정["판정"]
        D --> E{"자동 판정<br/>기하학만 본다"}
        E -->|"120px 미만<br/>여백 90% 초과"| F["bad — 사람에게 안 묻는다"]
        E -->|그 외| G["미판정"]
        G --> H["대시보드 평가 패널<br/>사람이 좋음/별로"]
    end
    subgraph 조회["조회 (런 N+1)"]
        H --> I[("ChromaDB<br/>data/art-memory/")]
        F --> I
        I --> J["visual_direction 으로 조회<br/>bad 제외 · 장르/역할 필터"]
        J --> K["아트 기획 프롬프트에<br/>예시 3개 주입"]
    end
    K -.->|"다음 런의"| A
```

**무엇이 임베딩되나**

```
문서(embedded)  "[미로 추격 · enemy] 둥근 유령, 붉은 단색, 굵은 검은 외곽선, 프레임을 꽉 채움"
메타데이터      role · genre · prompt · kind · width · height · removed_share
                verdict("good"|"bad"|"") · verdict_by("human"|"auto"|"") · verdict_note
id              "<런 폴더>:<파일명>"
```

장르와 역할을 문서 앞에 붙이는 이유는 **비슷하게 들리는 두 프롬프트가 서로 다른 요청**이기 때문이다.
"둥근 붉은 캐릭터"는 적일 수도 수집품일 수도 있다.

**`role`은 `kind`가 아니다**

| | 값 | 무엇을 정하나 |
|---|---|---|
| `kind` | `sprite` / `backdrop` | **어떻게 자를지** — 배경을 남길지 |
| `role` | player · enemy · projectile · pickup · obstacle · terrain · effect · ui · backdrop | **무엇인지** — 조회 키 |

역할이 자유 텍스트가 아니라 고정 목록인 이유: *"슈팅 장르에서 좋게 평가된 **적** 프롬프트"* 조회는
**모든 적이 자기를 적이라고 부르기로 합의해야** 성립한다.

**사람 피드백이 하는 일**

| 판정 | 누가 | 다음 런에서 |
|---|---|---|
| `bad` (자동) | 기하학 | 조회에서 제외 |
| `bad` (사람) | 리뷰어 | 조회에서 제외 |
| `good` (사람) | 리뷰어 | **예시로 주입 + "플레이어가 좋다고 평가"로 표시** |
| 미판정 | — | 같은 장르면 예시로 쓰임 (자동이 이미 실패는 걸렀으므로) |

자동 판정이 **실패만** 붙이는 이유: *"작게 나왔다"* 는 배경 제거가 이미 기록하는 기하학에 보이지만
*"보기 좋다"* 는 안 보인다.

**그림이 아니라 말이 전이된다**

주입되는 것은 프롬프트 텍스트이고, 같은 프롬프트로 생성해도 z_image_turbo는 시드마다 다른 그림을 낸다.
그래서 **복제가 아니라 학습**이고, IPAdapter나 img2img로 픽셀을 물려주는 것과 근본적으로 다르다.
저작권 문제가 없는 이유이기도 하다 — 코퍼스가 전부 자기 산출물이다.

**재개발하면 평가셋이 다시 만들어진다**

재개발은 같은 폴더·같은 런 id로 돈다. 재생성된 스프라이트는 같은 행을 덮어쓰고 **판정이 초기화된다**
(새 이미지이므로 옛 판정이 설명하지 않는다). 평가 패널이 보여 주는 셋은 **디스크가 정한다** — 지금
`assets/`에 있는 파일만. 두 빌드 전의 게임을 설명하는 평가셋은 리뷰어에게 **없는 이미지를 판정하게**
만든다.

다만 사람이 `good`이라 한 프롬프트는 덮어쓰기 전에 보관한다. 이미지는 사라져도 **교훈은 남는다** —
보관본은 평가 패널에 안 나오고(설명할 이미지가 없으니) 조회에는 잡힌다(프롬프트는 통했으니).

**모든 실패가 조용하다.** chromadb가 없든 파일이 잠겼든 **예시 없이 진행**한다. ComfyUI·Godot과 같은
계약이다 — 빌드를 깰 수 있는 기억은 없는 것만 못하다.

---


## 9. 실시간 관측

대시보드는 "지금 누가 무슨 도구를 쓰는지"를 그대로 본다.

```mermaid
flowchart LR
    N["노드 / 미들웨어"] -->|"get_stream_writer()"| ST["커스텀 스트림"]
    ST --> SVC["StudioService._drive"]
    SVC -->|"kind=usage"| U["토큰·비용 집계"]
    SVC -->|"kind=model_call/tool_call/..."| L["에이전트 로그"]
    SVC -->|"step/text"| P["생성 중 원본 출력"]
    U & L & P --> WS["WebSocket 브로드캐스트"]
```

- 라이브 프리뷰는 **0.25초 타이머**로 만든다. 토큰마다 만들면 긴 생성이 자기 길이에 대해 2차가 된다.
- 브로드캐스트에서 `game_html`·`design_review`·`messages`는 뺀다. UI가 안 쓰는데 20KB+다.
- 비용 추정은 **캐시 읽기를 정가로 세지 않는다**(입력가의 1/10).

---

## 10. 측정과 관측

### 평가 하네스 — 프롬프트를 측정한다

`tests/`의 모든 테스트는 모델을 스텁 처리한다. 코드에는 맞는 선택이지만, 그래서 **프롬프트 자체는
측정되지 않는다** — 그리고 이 파이프라인의 동작 대부분은 프롬프트에 있다.

```bash
python -m game_studio.evaluate                            # 전체 (실제 비용 발생)
python -m game_studio.evaluate --case clone-tetris
python -m game_studio.evaluate --out evals/new.json --baseline evals/baseline.json
```

기획 단계만 돌린다(케이스당 모델 2회). 지표는 **거의 전부 결정론적**이다 — 판단이 필요한
"이게 정말 테트리스인가"는 구조적으로 답한다: 충실한 클론은 재현 대상을 `reference_games`에
적어야 하므로, 그 이름이 왔는지만 보면 된다.

| 지표 | 잡는 것 |
|---|---|
| `genre_kept` | 요청/배정 장르대로 나왔나 |
| `genre_assigned` | 자동 기획이 앵커 가능한 장르를 받았나 |
| `clone_named` | 재현 요청한 게임을 실제로 참조했나 |
| `has_references` | 검증된 루프에 앵커했나 |
| `contract_sized` | 코드 Agent 예산 안에 끝낼 계약인가 |
| 다양성 | 자동 기획 간 제목·코어 루프 중복도 |

### 트레이스

`@traceable`이 파이프라인 노드 11개 + **코드 Agent 내부**에 붙어 있다. 후자가 중요한데,
가장 비싼 단계(모델 호출 최대 20회)가 트레이스에서 노드 하나로 뭉쳐 보였기 때문이다.

| 스팬 | 기록 |
|---|---|
| `code-agent-turn` | 제공된 도구, 입출력 토큰, 캐시 읽기, stop reason |
| `code-agent-tool` | 도구 이름, 인자(200자 절단) |
| `structured-output` | 스키마·모델·예산, 재시도와 예산 증액 |

런 단위로 engine·모델·이미지 생성 여부가 메타데이터에 붙어 트레이스 목록에서 비교 가능하다.
게임 전문이 트레이스로 새지 않도록 입력은 전부 요약해서 기록한다.

`.env`에 `LANGSMITH_TRACING=true` + API 키만 넣으면 켜진다. 코드 변경은 필요 없다.

> `@traceable`은 비활성 상태에서도 호출당 약 37µs다(실측: 20k 호출 752ms vs 평범한 함수 2ms).
> 턴·도구 경계에는 무시할 수준이지만 **청크 단위 경로에는 절대 넣으면 안 된다** —
> `StreamAccumulator.add`는 게임 하나에 수천 번 호출된다.

---

## 11. 파일 지도

```
src/game_studio/
├── graph.py          1487  LangGraph 오케스트레이션 — 노드 · 라우팅 · 예산
├── agents.py         1234  모델 어댑터 · 스트리밍 · 정적 QA · 장르 참조 · 총괄 감독
├── server.py         1205  FastAPI 대시보드 — REST · WebSocket · 실행 구동 · 체크포인트 정리
├── agent_tools.py     481  HTML 코드 Agent 도구 + ComfyUI 이미지 생성
├── godot.py           473  Godot 엔진 어댑터 — 검증 · 익스포트 · 런처
├── prompts.py         354  역할별 시스템 프롬프트 · 장르 레퍼런스 15×87
├── art_memory.py      301  아트 프롬프트 기억 — ChromaDB · 역할 9종 · 자동/사람 판정
├── evaluate.py        300  평가 하네스 — 15케이스 · 7지표 · 실제 모델 호출
├── sprites.py         288  스프라이트 후처리 — 배경 제거 · 방향 계약
├── code_agent.py      287  create_agent 조립 — 예산 · 재시도 · 관측
├── models.py          232  Pydantic 스키마 + StudioState + 계약 상한
├── godot_tools.py     197  Godot 코드 Agent 도구
├── cli.py             103  대시보드 없이 실행
├── required_art.py    101  필수 아트 판정 · 미사용 스프라이트 검출
└── comfyui.py          38  ComfyUI 워크플로 컴파일

tests/                8000+ 11개 파일 · 244개 테스트
web/                   467  대시보드 프론트엔드
```

---

## 12. 실행에 필요한 것

| 구성요소 | 필수 | 없으면 |
|---|---|---|
| Amazon Bedrock 자격증명 | **예** | 실행 자체가 거부됨 (503) |
| Node.js | 아니오 | JS 문법 검사만 건너뜀 |
| ComfyUI | 아니오 | Canvas 전용으로 제작 |
| Godot 4 | Godot 모드만 | 드롭다운에서 해당 옵션 비활성 |
| Godot 익스포트 템플릿 | 아니오 | 웹 빌드 없이 `run.bat`으로 실행 |
| Pillow | 아니오 | 스프라이트가 불투명 사각형으로 남음 |

**"없으면 조용히 통과"는 없다.** 전부 명시적으로 보고한다.

---

## 13. 주요 설정

`.env`로 전부 조절 가능하다. 자주 만지는 것만 추림.

```bash
# 모델
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6            # 기획·코딩
BEDROCK_QA_MODEL_ID=global.anthropic.claude-haiku-4-5-...      # 검증·판단
# BEDROCK_FALLBACK_MODEL_IDS=us.anthropic.claude-sonnet-4-6    # 분당 스로틀 시 넘어갈 프로파일
# BEDROCK_DAILY_CAP_MODEL_ID=global.anthropic.claude-haiku-...  # 일일 한도 소진 시 쓸 모델(빈 값이면 중단)

# QA 강도
QA_REJECT_TOLERANCE=0.5        # 계약의 이 비율을 넘게 미충족해야 차단
MAX_RETHINK_CYCLES=2           # 가장 비싼 손잡이
MAX_REPAIR_ATTEMPTS=2
ADVISORY_FINDING_LIMIT=5

# 코드 Agent
CODE_AGENT_MODEL_CALLS=70

# 이미지 (로컬 ComfyUI — 청구되는 건 없고 묶이는 건 GPU 시간이다)
COMFYUI_CFG=1.0                     # Z-Image Turbo가 증류된 조건. 올리면 느려진다
COMFYUI_TIME_BUDGET_SECONDS=600     # 런당 생성 시간. 다 쓰면 Canvas 도형으로 넘어간다
COMFYUI_SPRITE_PIXELS=512           # 스프라이트 캔버스 상한 (배경만 1024)
COMFYUI_MAX_ASSETS=14               # PNG 장수 — 폭주 방지용이지 예산이 아니다
MAX_REFERENCE_IMAGES=3              # 런 시작 시 올릴 수 있는 참조 이미지

# 비용
BEDROCK_PROMPT_CACHE_TTL=5m    # off 로 끌 수 있음

# 지속성
CHECKPOINT_DB=<프로젝트>/data/studio-checkpoints.sqlite         # :memory: 로 옵트아웃
STUDIO_DATA_DIR=<프로젝트>/data                                 # 체크포인트·벡터 DB 위치
```

전체 목록은 `.env` 주석에 있다.
