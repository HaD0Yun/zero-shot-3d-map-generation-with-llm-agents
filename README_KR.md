# Dual-Agent PCG 시스템

**LLM 에이전트를 활용한 Zero-shot 3D 맵 생성: 절차적 콘텐츠 생성을 위한 이중 에이전트 아키텍처**

[arXiv:2512.10501](https://arxiv.org/abs/2512.10501) 논문 기반

🌐 **[프로젝트 페이지](https://had0yun.github.io/zero-shot-3d-map-generation-with-llm-agents/index-ko.html)** | 📄 [arXiv 논문](https://arxiv.org/abs/2512.10501)

[English](./README.md) | **한국어**

## 개요

본 프로젝트는 Zero-shot 절차적 콘텐츠 생성(PCG) 파라미터 설정을 위한 Dual-Agent Actor-Critic 아키텍처의 완전한 구현체입니다. 태스크별 파인튜닝 없이 기성 LLM을 PCG 도구와 연동할 수 있게 해줍니다.

### 핵심 특징

- **Zero-shot**: 추가 학습 없이 자연어로 3D 맵 파라미터 생성
- **Dual-Agent**: Actor-Critic 구조로 안정적인 파라미터 검증
- **Training-free**: 사전 훈련 불필요, 기성 LLM 그대로 활용
- **Tool-agnostic**: API 문서만 제공하면 어떤 PCG 도구든 적용 가능

## 아키텍처

```
                     사용자 프롬프트 (P_user)
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                       오케스트레이터                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   컨텍스트 매니저                        │  │
│  │  - 상태 교체 전략 (고정 크기 버퍼)                        │  │
│  │  - API 문서 주입                                        │  │
│  │  - 사용 예시 관리                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
│      ┌─────────────────────┴─────────────────────┐           │
│      │                                           │           │
│      ▼                                           ▼           │
│  ┌───────────────────┐                   ┌───────────────────┐│
│  │   ACTOR 에이전트   │      S_i         │  CRITIC 에이전트   ││
│  │   (의미 해석기)    │ ───────────────► │   (정적 검증기)    ││
│  │                   │                   │                   ││
│  │                   │ ◄─────────────── │                   ││
│  │  Temperature:     │     피드백        │  Temperature:     ││
│  │     0.4           │                   │     0.2           ││
│  └───────────────────┘                   └───────────────────┘│
│                            │                                  │
│                   ┌────────┴────────┐                        │
│                   │    승인 여부?    │                        │
│                   └────────┬────────┘                        │
│                     Yes    │    No                           │
│                      │     │     │                           │
│                      ▼     │     ▼                           │
│                  [반환]    │  [반복 또는 최선의 결과]          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   RefinementResult (S_final)
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# API 키 설정
export ANTHROPIC_API_KEY=your-key-here  # Linux/Mac
set ANTHROPIC_API_KEY=your-key-here     # Windows
```

## 사용법

### 실제 API로 실행

```bash
python -m dual_agent_pcg.main
```

### Mock 모드로 실행 (API 불필요)

```bash
python -m dual_agent_pcg.main --mock
```

---

## 🚀 빠른 시작 - 프롬프트만 복사하세요!

**설정하기 귀찮으신가요?** 아래 프롬프트를 복사해서 좋아하는 LLM CLI (Claude, ChatGPT 등)에 붙여넣기만 하세요. `$ARGUMENTS`를 원하는 맵 설명으로 바꾸면 끝!

<details>
<summary><strong>📋 전체 프롬프트 보기 (클릭)</strong></summary>

```markdown
# Dual-Agent PCG Map Generation

You are executing the **Zero-shot Dual-Agent PCG Refinement Protocol** (arXiv:2512.10501).

## User Request (P_user)

**Map Description**: $ARGUMENTS

---

## Protocol

You will alternate between two roles until convergence (max 3 iterations):

### ACTOR ROLE (Semantic Interpreter)
Generate a Parameter Trajectory Sequence as JSON. You must:
- Translate the user's intent into specific PCG tool configurations
- Include concrete parameter values (NO placeholders like "TBD")
- Ground all tool names and parameters in the API Documentation below
- Identify risks and assumptions

### CRITIC ROLE (Static Verifier)  
Review the trajectory against documentation. Apply the 5-dimension framework:
1. **Tool Selection**: Does each tool exist exactly as named?
2. **Parameter Correctness**: All required params present? Values in valid range?
3. **Logic & Sequence**: Generators before modifiers? Dependencies satisfied?
4. **Goal Alignment**: Does trajectory achieve user's requirements?
5. **Completeness**: Any missing steps?

**CONSERVATIVE POLICY**: Only flag issues you're CERTAIN about.

---

## Execution Flow

1. [ACTOR] Generate initial trajectory S₀
2. [CRITIC] Review S₀ → produce feedback
3. IF issues found AND iteration < 3:
   [ACTOR] Revise trajectory based on feedback → S₁
   [CRITIC] Review S₁
   ... repeat until approved or max iterations
4. Output final approved trajectory

---

## API Documentation (D)

### Generators (must be called before modifiers)

#### CellularAutomataGenerator
Creates organic landmass patterns. Ideal for islands, continents, caves.

**Required Parameters:**
- `width` (int): Grid width [16, 256]
- `height` (int): Grid height [16, 256]  
- `fill_probability` (float): Initial fill [0.0, 1.0]
  - 0.3-0.4: scattered landmasses
  - 0.45-0.55: balanced, connected
  - 0.6-0.7: larger, solid masses
- `iterations` (int): Smoothing passes [1, 10]
- `birth_limit` (int): Birth threshold [0, 8] (typically 4)
- `death_limit` (int): Death threshold [0, 8] (typically 3)

**Optional:** `seed` (int)

#### PerlinNoiseGenerator
Creates smooth heightmaps. Ideal for elevation, mountains, hills.

**Required Parameters:**
- `width` (int): Grid width [16, 512]
- `height` (int): Grid height [16, 512]
- `scale` (float): Noise scale [0.01, 1.0]
  - 0.01-0.03: large, smooth features
  - 0.04-0.08: good for mountains
  - 0.1+: rough, detailed
- `octaves` (int): Detail layers [1, 8]
- `persistence` (float): Amplitude falloff [0.0, 1.0]

**Optional:** `seed` (int), `lacunarity` (float, default 2.0)

### Modifiers (apply after generators)

#### HeightLayerModifier
Creates discrete elevation zones.

**Required Parameters:**
- `layer_count` (int): Number of layers [1, 10]
- `layer_heights` (list[float]): Thresholds, ascending order
- `blend_factor` (float): Transition smoothness [0.0, 0.5]

#### ScatterModifier
Scatters objects on terrain.

**Required Parameters:**
- `object_type` (str): One of "rock", "tree", "grass_clump", "bush", "flower"
- `density` (float): Scatter density [0.0, 1.0]
- `valid_layers` (list[int]): Layer indices to scatter on

#### GrassDetailModifier
Adds grass coverage to a layer.

**Required Parameters:**
- `target_layer` (int): Layer index (0-indexed)
- `coverage` (float): Coverage percentage [0.0, 1.0]

---

## Output Format

Output the FINAL APPROVED trajectory as JSON:

{
  "final_trajectory": {
    "trajectory_summary": "<overview>",
    "tool_plan": [
      {
        "step": 1,
        "objective": "<what this achieves>",
        "tool_name": "<EXACT tool name>",
        "arguments": { ... },
        "expected_result": "<success criteria>"
      }
    ],
    "risks": ["<potential issues>"]
  }
}

---

## BEGIN PROTOCOL

Now execute the Dual-Agent refinement for: **$ARGUMENTS**

Start with [ACTOR] generating the initial trajectory S₀.
```

</details>

> 💡 사용 예제가 포함된 전체 프롬프트는 [`.opencode/command/Map.md`](.opencode/command/Map.md)를 참조하세요.

---

## OpenCode 통합 (권장)

**API 키 불필요!** OpenCode CLI 내에서 Dual-Agent PCG 시스템을 직접 사용할 수 있습니다.

### 빠른 설정

1. `.opencode` 디렉토리를 사용자 홈으로 복사:

```bash
# Windows
xcopy /E /I .opencode %USERPROFILE%\.opencode

# Linux/Mac
cp -r .opencode ~/.opencode
```

2. OpenCode를 실행하고 `/Map` 명령어 사용:

```bash
opencode -c
```

```
/Map 중앙에 분화구 호수가 있는 화산섬
```

### 작동 방식

OpenCode 통합은 인증된 Claude 세션을 활용하여 별도의 API 키가 필요 없습니다:

```
User: /Map 3개 고도 구역이 있는 산악 지형
         │
         ▼
    ┌─────────────────┐
    │  Map.md 커맨드  │ ← 프로토콜 조율
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐     ┌─────────┐
│  Actor  │     │ Critic  │
│  Agent  │────▶│  Agent  │
│ (t=0.4) │◀────│ (t=0.2) │
└─────────┘     └─────────┘
             │
             ▼
    JSON 파라미터 출력
```

### OpenCode 파일 구조

```
.opencode/
├── agent/
│   ├── pcg-actor.yaml    # Actor 에이전트 (temperature: 0.4)
│   └── pcg-critic.yaml   # Critic 에이전트 (temperature: 0.2)
└── command/
    └── Map.md            # API 문서가 포함된 메인 슬래시 커맨드
```

### 비교: Python API vs OpenCode

| 기능 | Python API | OpenCode |
|------|-----------|----------|
| API 키 필요 | ✅ 필요 | ❌ **불필요** |
| 설정 복잡도 | pip install + 환경변수 | 폴더 복사 |
| Temperature 제어 | ✅ 0.4/0.2 | ✅ 0.4/0.2 |
| 토큰 추적 | 정확함 | 추정치 |
| 사용 방식 | 스크립트/코드 | `/Map` 명령어 |

---

## 프로그래밍 방식 사용

```python
import asyncio
from dual_agent_pcg.models import SystemConfig
from dual_agent_pcg.llm_providers import create_provider
from dual_agent_pcg.orchestrator import Orchestrator

async def main():
    # 논문 설정과 일치하는 구성 (Section 4.1)
    config = SystemConfig(
        actor_temperature=0.4,   # 창의성과 지시 준수의 균형
        critic_temperature=0.2,  # 일관되고 확신 있는 피드백
        max_iterations=1         # 논문 설정
    )
    
    # 프로바이더 생성
    llm = create_provider("anthropic", api_key="your-key")
    
    # 오케스트레이터 생성
    orchestrator = Orchestrator(
        config=config,
        llm_provider=llm,
        api_documentation="...",  # PCG 도구 문서
        usage_examples=["..."]    # 예시 궤적
    )
    
    # 실행
    result = await orchestrator.execute("산악 지형 생성...")
    
    if result.success:
        print("승인된 궤적:", result.final_trajectory)
    else:
        print("최선의 결과:", result.final_trajectory)

asyncio.run(main())
```

## 프로젝트 구조

```
dual_agent_pcg/
├── .opencode/                    # OpenCode 통합
│   ├── agent/
│   │   ├── pcg-actor.yaml       # Actor 에이전트 설정 (t=0.4)
│   │   └── pcg-critic.yaml      # Critic 에이전트 설정 (t=0.2)
│   └── command/
│       └── Map.md               # /Map 슬래시 커맨드
├── __init__.py                   # 패키지 초기화
├── models.py                     # Pydantic 모델 (ActorOutput, CriticFeedback 등)
├── prompts.py                    # Actor와 Critic용 시스템 프롬프트
├── llm_providers.py              # LLM 프로바이더 추상화 (Anthropic, OpenAI, Mock)
├── orchestrator.py               # Algorithm 1 구현
├── main.py                       # 실행 가능한 예제
├── config.yaml                   # 설정 파일
└── requirements.txt              # 의존성
```

## 논문 설정 (Section 4.1)

| 설정 | 값 | 근거 |
|------|-----|------|
| LLM 모델 | Claude 4.5 Sonnet | 논문 선택 |
| Actor Temperature | 0.4 | "창의성과 지시 준수의 균형" |
| Critic Temperature | 0.2 | "일관되고 확신 있는 피드백" |
| 최대 반복 횟수 | 1 | "모든 실험에서 1로 설정" |

## 핵심 컴포넌트

### Actor 에이전트 (의미 해석기)
- 자연어를 파라미터 궤적 시퀀스로 변환
- 출력: `{trajectory_summary, tool_plan, risks}`
- 도구 직접 실행 금지

### Critic 에이전트 (정적 검증기)
- 문서 기준으로 궤적 평가
- 5차원 리뷰 프레임워크:
  1. 도구 선택
  2. 파라미터 정확성
  3. 논리 및 순서
  4. 목표 정렬
  5. 완성도
- 보수적 정책: 확실한 경우에만 블로킹 이슈로 표시

### 컨텍스트 매니저
- 상태 교체 전략 구현 (Section 3.3)
- 고정 크기 컨텍스트 버퍼
- 이전 궤적을 덮어씀 (추가가 아님)

## Algorithm 1: Zero-shot Dual-Agent PCG 정제

```
입력: P_user, D, E, K
출력: S_final

Context_actor ← {P_user, D, E}
S_0 ← Actor(Context_actor)
i ← 0

while i < K do:
    Feedback ← Critic(S_i, D, E)
    if Feedback = empty then
        return S_i
    Context_actor ← UpdateContext(S_i, Feedback)
    S_{i+1} ← Actor(Context_actor)
    i ← i + 1

return S_K
```

### 변수 설명

| 변수 | 설명 |
|------|------|
| P_user | 사용자의 자연어 프롬프트 |
| D | API 문서 (PCG 도구 명세) |
| E | 사용 예시 (검증된 궤적들) |
| K | 최대 반복 횟수 |
| S_i | i번째 반복의 파라미터 궤적 시퀀스 |

## 라이선스

본 구현체는 교육 및 연구 목적입니다.

## 인용

```bibtex
@article{her2025zeroshot3dmap,
  title={Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation},
  author={Her, Lim Chien and Yan, Ming and Bai, Yunshu and Li, Ruihao and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.10501},
  year={2025}
}
```

## 관련 링크

- **논문**: [arXiv:2512.10501](https://arxiv.org/abs/2512.10501)
- **GitHub**: [HaD0Yun/zero-shot-3d-map-generation-with-llm-agents](https://github.com/HaD0Yun/zero-shot-3d-map-generation-with-llm-agents)
