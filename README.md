# agentic_rag — 한국 노동법 리트리버 에이전트

**naive → advanced(하이브리드·리랭킹) → agentic(반복 검색·자가 판정) RAG를 한 코드베이스에서 구현하고, 각 단계의 기여를 701쿼리 평가셋으로 실측한 프로젝트.**

사내 HR 컴플라이언스 봇(한국 노동법) 시나리오로 만들었지만, 코드는 코퍼스 무관 — 데이터를 갈아끼우면 그대로 동작합니다. 로컬 GPU 1장(RTX 4090)에서 임베더·리랭커·9B LLM 3모델을 동시 서빙합니다.

```
Q: 육아휴직 기간에도 연차휴가가 발생하나요?
→ 에이전트가 스스로 "육아휴직 중 연차유급휴가 발생"으로 리라이팅
→ 근로기준법 제60조 리랭크 1위 → 제60조⑥ 정확 인용 답변
```

## 핵심 결과 (ablation)

전 조합을 고용노동부 질의회시 기반 **701쿼리 qrels**(조문 단위 정답)로 평가. 전체 표와 해석은 [`docs/ablation_results.md`](docs/ablation_results.md).

| 파이프라인 | R@1 | R@8 | MRR | latency p50 |
|---|---|---|---|---|
| BM25 단독 | 0.136 | 0.238 | 0.292 | 0.01s |
| 하이브리드(RRF) | 0.212 | 0.487 | 0.477 | 0.01s |
| dense 단독 | 0.237 | 0.536 | 0.520 | 0.01s |
| **하이브리드 + 0.6B 크로스인코더** | 0.286 | 0.575 | **0.584** | **0.16s** |
| 하이브리드 + 9B LLM 리랭크 | 0.295 | 0.577 | 0.603 | 4.3s |
| react 에이전트 (max_steps 6) | 0.224 | 0.515 | 0.504 | 112s |

세 가지 발견:

1. **전용 리랭커는 즉시 값을 한다.** 0.6B 크로스인코더가 MRR +11pp를 쿼리당 0.15초에 산다. 9B LLM 리랭크는 추가 +2pp에 30배 비용 — 스위트스팟은 전용 0.6B.
2. **약한 sparse는 하이브리드를 해친다.** 형태소 분석 없는 한국어 BM25가 RRF 융합을 오염시켜, 리랭크 없이는 dense 단독이 하이브리드를 이긴다. 리랭커를 얹으면 하이브리드가 회복 — "BM25는 후보 풀 확장, 노이즈는 리랭커가 필터"가 실측으로 확인됨.
3. **에이전틱 반복이 항상 이기지는 않는다 (정직한 네거티브).** 이 평가셋에서 react 에이전트는 1-shot 검색+리랭크에 못 미쳤다. 원인 분리: ① LLM이 메타데이터 필터(`law_names`)를 오용해 0건 검색을 만들고, 스텝 예산이 작으면 만회를 못 함(빈손 s2 17% → s6 8%), ② 질의회시 질문은 이미 법률 용어로 정제돼 있어 에이전트의 핵심 무기인 **리라이팅이 발휘될 여지가 없음**. 일상어 질문에서는 리라이팅이 top-30 밖 조문을 1위로 끌어올리는 것을 라이브로 확인 — 일상어 변형 평가셋이 후속 과제.

## 아키텍처

```mermaid
flowchart TD
    Q[질문] --> P["reason (Qwen3.5-9B)\n리라이팅 · 디컴포지션 · 충분성 판정"]
    P -->|act| S[statute_search]
    P -->|act| W["web_search (ddgs)"]
    S --> P
    W --> P
    P -->|finish: cid 선택| R["resolve() — 코드가 전문 해소\n(환각 구조적 차단)"]
    R --> A[답변 합성 + 근거 조문]

    subgraph statute_search 내부 2단계
        S1["하이브리드 top-30\ndense(Qwen3-Emb 0.6B) + BM25 → RRF"] --> S2["크로스인코더 top-8\nQwen3-Reranker-0.6B"]
    end
    S -.-> S1
```

설계 포인트:

- **리라이팅/디컴포지션은 툴이 아니라 policy 지침** — 툴로 빼면 LLM 호출만 늘어난다. 충분성 판정(CRAG식)도 reason 스텝이 겸한다.
- **리랭커는 툴 내부 파이프라인** — 쓸지 말지는 LLM이 결정할 일이 아니다 (ablation 플래그로만 on/off).
- **finish 계약: LLM은 조문 id(cid)만 고른다.** 전문은 검색 세션에서 코드로 해소하므로 LLM이 조문 텍스트를 재타이핑하며 환각을 섞을 경로가 구조적으로 없다.
- **관측 절단**: 툴 결과는 압축 뷰(cid/ref/snippet)로만 scratchpad에 들어간다 — 컨텍스트 폭발 방지.
- 에이전트 프레임워크는 [basic_agent](https://github.com/Taeksu-Kim/basic_agent)(자작 LangGraph 스켈레톤, HITL·체크포인팅 포함)를 vendoring.

## 데이터 · 평가셋 (전부 스크립트로 재현)

| | 소스 | 규모 |
|---|---|---|
| 코퍼스 | HF [`ducut91/korean-statutes`](https://huggingface.co/datasets/ducut91/korean-statutes) (MIT) → 노동법 8종 필터 | 1,787조문 |
| 평가셋 | law.go.kr DRF API — 고용노동부 질의회시(`moelCgmExpc`) + 법제처 해석례(`expc`) | 701쿼리 |

- **1조문 = 1청크 = 1포인트.** 조문이 자연스러운 의미 단위(중앙값 275자)라 추가 청킹 없음 — miss가 긴 조문에 몰리지 않는 것을 데이터로 확인(오히려 짧은 정의·준용 조항이 어렵다).
- **qrels 자동 구축**: 질의회시 회답의 조문 인용(`「근로기준법」 제60조제2항`, "같은 법 시행령", 약칭 정의 등)을 정규식 상태기계로 파싱해 정답 라벨 생성 ([`evaluation/citations.py`](evaluation/citations.py)). 질의요지가 곧 자연스러운 실무 질문이라 검색 난이도가 현실적이다.
- 멀티홉이 도메인 필연: 법률 조문 693개 중 시행령·부령 위임이 338개, 조문 간 상호참조 평균 3.9회.

## 실행

```bash
# 0. 서빙 스택 (Qdrant + 임베더 + 리랭커 + 9B, RTX 4090 24GB 기준)
bash scripts/serve.sh start

# 1. 데이터 구축
python scripts/build_corpus.py      # HF → 노동법 8종 1,787조문
python scripts/build_index.py       # Qdrant 하이브리드 인덱스
python scripts/build_evalset.py     # 질의회시 수집 → qrels

# 2. 단일 질문 (에이전트)
PYTHONPATH=. python scripts/ask.py "육아휴직 기간에도 연차휴가가 발생하나요?" --trace

# 3. ablation (재개 가능, --workers로 병렬)
PYTHONPATH=. python scripts/run_ablations.py retrieval
PYTHONPATH=. python scripts/run_ablations.py agent --steps 2,4,6 --sample 120
PYTHONPATH=. python scripts/run_ablations.py report   # → docs/ablation_results.md
```

모델 서빙: vLLM으로 Qwen3.5-9B(정책·합성), Qwen3-Embedding-0.6B(`--runner pooling`), Qwen3-Reranker-0.6B(`/score` + 공식 채팅 템플릿 — 템플릿 없이는 순위가 무의미해지는 것 실측). 상세 플래그는 [`scripts/serve.sh`](scripts/serve.sh).

## 구조

```
agent/        # vendored 범용 에이전트 프레임워크 (react/plan_execute, TDD)
retriever/    # 검색 스택: embedder / index / search(2단계) / reranker / tool / agent
evaluation/   # citations(qrels 파싱) / metrics(recall@k, MRR) / runner(재개 가능)
scripts/      # build_corpus·index·evalset, ask, run_ablations, serve.sh
docs/         # design_and_plan.md(설계·결정로그), ablation_results.md(결과·해석)
data/         # gitignored — 위 스크립트로 재현
```

**테스트 159개, 전부 fake 기반** (네트워크·실서버·실모델 없이 `pytest` 한 방) — 임베더/리랭커/LLM/웹은 Protocol 뒤에 두고 fake로 갈아끼운다.

## 한계 · 후속

- 평가셋이 법률 용어 질문이라 리라이팅 이득이 안 보인다 → 일상어 변형 평가셋으로 에이전트 축 재측정
- react의 메타데이터 필터 오용 가드 (0건 결과 시 필터 해제 재시도를 툴 레벨에서)
- 멀티턴(대화 히스토리) 데모 UI — Gradio, 스텝 트레이스 스트리밍 (진행 중)
