# 문서

| 문서 | 무엇이 들어 있나 | 언제 읽나 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 시스템 구조, 다이어그램 9개, 파이프라인 단계, 상태 모델, 검증 체계, 예산, 설정 | 전체 그림을 잡을 때 |
| [MODULES.md](MODULES.md) | 파일별 책임과 그 자리에 있는 이유, 의존 관계 | 특정 코드를 고칠 때 |
| [DESIGN-DECISIONS.md](DESIGN-DECISIONS.md) | 설계 판단 17건과 실측 근거 | "왜 이렇게 했지?" 싶을 때 |

## 빠른 참조

**파이프라인 한 줄 요약**
총괄 감독 → 기획 → 구현 계약 → **사람 승인** → 아트 계획 → 코드 Agent → QA 2단 → 패키징

**두 엔진**
`html5`는 단일 `index.html`(외부 의존성 0), `godot`은 Godot 4 프로젝트 + `run.bat`.
기획 단계는 전부 공유하고, 코드·검증·패키징에서만 갈라진다.

**핵심 원칙 세 가지**
1. 사람의 승인이 진짜 게이트다 (SQLite 체크포인트로 재시작을 견딤)
2. 판단이 필요한 곳에만 모델을 쓴다 (supervisor 8회 방문 = 모델 1회)
3. 검증은 게임을 못 돌게 하는 것만 막는다

**실패를 숨기지 않는다**
Bedrock 없음 → 거부. Godot 없음 → 건너뜀 명시. QA 미통과 → 미해결 항목과 함께 배포.
고정 게임으로 대체하는 경로는 없다.

## 자주 만지는 설정

```bash
QA_REJECT_TOLERANCE=0.5     # 계약의 이 비율을 넘게 미충족해야 차단
MAX_RETHINK_CYCLES=1        # 가장 비싼 손잡이 (코드 Agent 루프 전체 + 재감사)
CODE_AGENT_MODEL_CALLS=20   # 코드 Agent 호출 예산
BEDROCK_PROMPT_CACHE_TTL=5m # off 로 끌 수 있음
```

전체 목록과 주석은 `.env`에 있다.
