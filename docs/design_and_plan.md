# Retriever Agent — 설계 & 작업 계획

> 이 문서는 세션이 바뀌어도 작업을 이어갈 수 있도록 **모든 결정과 문맥**을 담는다.
> 작성: 2026-07-17. 상태: **Phase 0 진행 중 (골격 생성됨) + 평가셋 소스 스파이크 완료(최대 리스크 해소).**

## 0. 프로젝트 정체성

- **무엇**: 독립 실행 + 다른 에이전트의 서브에이전트 겸용 **리트리버 에이전트**.
  복잡한 질문을 스스로 리라이팅/디컴포지션하며 반복 검색하고, 리랭킹·충분성
  판정을 거쳐 **구조화된 근거(조문) 목록**을 반환한다.
- **왜**: 포트폴리오 — 금융 프로젝트(news_reaction_agent)와 별개로 **RAG 전반
  커버 증명**. naive→advanced(하이브리드·리랭킹·리라이팅)→agentic(반복·분해·
  자가판정·멀티소스)→conversational(멀티턴)까지. 의도적 제외: GraphRAG,
  지저분한 인제스천(PDF/OCR), 임베더 파인튜닝, RAGAS류 faithfulness 평가.
- **시나리오**: 사내 HR 컴플라이언스 봇 (한국 노동법). 단, 코드는 코퍼스
  무관(데이터 갈아끼우면 동작).
- **레포**: 별도 공개 레포 **`agentic_rag`** (github.com/Taeksu-Kim/agentic_rag 예정
  — RAG 키워드 노출이 목적이라 이름에 박음; GitHub topics: rag, agentic-rag,
  hybrid-search, reranker, qdrant). 범용 프레임워크 `agent/`는 basic_agent에서 **vendoring**
  (news_reaction_agent와 같은 방식). **데이터는 gitignore**하고 빌드 스크립트로
  재현 (참고: 이 코퍼스는 법령 조문 = 저작권법 제7조 비보호저작물 + MIT
  데이터셋이라 커밋해도 합법 — 유저 결정으로 일단 제외, 나중에 포함 전환 가능).

### 관련 레포 관계도

```
basic_agent (github.com/Taeksu-Kim/basic_agent)   ← 범용 프레임워크 원본 (@0d5d812: HITL 포함)
  ├─ vendored → news_reaction_agent (금융, 공개)
  └─ vendored → 이 프로젝트 (공개 예정)
원본 작업 사본: "/mnt/d/workspace/stock dataset/agent/" (basic_agent의 git working tree)
```

**프레임워크 수정 규칙**: 범용 기능은 반드시 basic_agent에서 TDD로 작업·푸시 후
각 프로젝트에 rsync로 vendor 동기화 (`rsync -a --delete --exclude .git
--exclude __pycache__ --exclude .pytest_cache <원본>/agent/ ./agent/`).

## 1. 환경 (그대로 안 지키면 삽질하는 것들)

- conda env **stock-dataset** (Python 3.12). 실행:
  `/home/ubuntu/miniconda3/envs/stock-dataset/bin/python` (conda run은 stdout 삼킴).
- **WSL2**: loopback 깨짐 → 서버 접근은 `10.5.0.2`. vite/gradio 등은 `0.0.0.0`
  바인드. `/mnt/d`는 9p라 느림 — 대용량 IO/Qdrant 스토리지는 ext4(`~/`)에.
- **Qdrant 네이티브 서버** (임베디드 모드는 GIL 캡 ~50/s, 서버는 ~180/s):
  `QDRANT__STORAGE__STORAGE_PATH=~/qdrant_storage QDRANT__SERVICE__HOST=0.0.0.0 ~/qdrant/qdrant`
  기존 컬렉션 `articles`(금융 179,778포인트)와 같은 인스턴스에 컬렉션만 추가.
- 테스트: **TDD, 전부 fake** — 테스트 스위트에 네트워크/실서버/실모델 금지
  (basic_agent 82 tests, news repo 153 tests가 이 규칙).

## 2. 모델 & 서빙 (4090 24GB, 3-프로세스 동시 상주)

| 포트 | 모델 | gpu-mem-util | 역할 |
|---|---|---|---|
| 8000 | Qwen3.5-9B-FP8-dynamic | **0.60** (기존 0.70에서 인하) | react policy (리라이팅·분해·충분성판정·LLM리랭크) |
| 8001 | Qwen/Qwen3-Embedding-0.6B | 0.10 | dense 임베딩 (`--runner pooling`, /v1/embeddings) |
| 8002 | **Qwen/Qwen3-Reranker-0.6B** | 0.10 | 2단계 크로스인코더 리랭킹 |

- 합 ~0.80. **기동은 큰 것(9B)부터 순차** (동시 기동시 프로파일링 충돌 OOM).
- 빠듯하면 9B에 `--kv-cache-dtype fp8`, 0.6B들에 `--max-model-len 2048~4096`.
- **9B는 반드시 로컬 경로로 로드** — HF repo id는 메모리 프로파일링에서 데드락:
  `"/mnt/d/workspace/stock dataset/models/Qwen3.5-9B-FP8-dynamic"`.
  전체 명령은 stock dataset의 CLAUDE.md "Local vLLM serving" 절 참조
  (CUDA_HOME=cu13 경로, VLLM_USE_FLASHINFER_SAMPLER=0, --enforce-eager).
- ⚠️ **오픈 이슈**: Qwen3-Reranker의 vLLM 서빙 플래그 미검증 — score/pooling
  태스크 + hf_overrides(sequence classification 변환)가 필요할 수 있음.
  Phase 2에서 실검증 후 여기 업데이트할 것.
- 리랭커는 평상시 안 띄워도 됨(v1 파이프라인은 9B 겸용 리랭크). 전용 리랭커는
  ablation 실험 때 3-프로세스 구성으로 전환.

## 3. 데이터

### 코퍼스 (확정)

- **HF `ducut91/korean-statutes`** — MIT, law.go.kr 원천, 전체 200,633 조문 /
  5,474 법령, 단일 CSV `korean-statutes-v2.csv` (231MB).
  카드에 "연구·상업·교육 무제한 이용·수정·재배포" 명시. 검증 완료(2026-07-17).
- 스키마: `LawNameKor, LawTypeName(법률/대통령령/고용노동부령), DepartmentName,
  ClauseNumber, ClauseTitle, ClauseContent, EffectiveDate, PromulgationDate, ...`
  — 조문 단위로 이미 정리돼 있어 **1조문 = 1청크 = 1포인트** (추가 청킹 불필요).
- **노동법 8종 패키지로 필터 → 1,787 조문** (실측):
  근로기준법(132)+시행령(73)+시행규칙(23), 남녀고용평등법(71/39/23),
  최저임금법(34/27/8), 근로자퇴직급여보장법(68/79/19), 기간제법(26/7/2),
  산업안전보건법(181/124/246), 고용보험법(147/208/197), 근로자참여법(34/11/8).
  ※ 기간제법 시행규칙의 LawTypeName은 "노동부령"(옛 명칭) — 필터는 법령명으로.
- **멀티홉 구조 실측**: 법률 조문 693개 중 시행령 위임("대통령령으로 정하")
  227개 + 부령 위임 111개, 조문 간 상호참조 평균 3.9회/조문 → 쿼리 디컴포지션이
  도메인 필연.
- 탈락 후보(재검토 방지): bowang0911/KoreanLegalQAChunkRetrieval(라이선스 없음),
  ggh5454/korean-legal-qa-dataset(CC-BY지만 질문에 정답 조문명이 박힌 템플릿 —
  검색 난이도 0), lbox·KLAID(CC-BY-NC/-ND), Rootpye/lawdata(zip, 구조 불명).

### 평가셋 — 소스 검증 완료 (2026-07-17 스파이크; 최대 리스크였음 → 해소)

law.go.kr **DRF Open API** 두 소스 실검증 (예제 키 `OC=test`로 응답 확인;
실수집 시 open.law.go.kr에서 본인 OC(이메일 id) 무료 발급 권장):

1. **고용노동부 행정해석(질의회시)** — `target=moelCgmExpc` (메인 소스):
   - 목록: `http://www.law.go.kr/DRF/lawSearch.do?OC=<oc>&target=moelCgmExpc&type=XML&query=<검색어>&display=100&page=N`
   - 상세: `http://www.law.go.kr/DRF/lawService.do?OC=<oc>&target=moelCgmExpc&ID=<법령해석일련번호>&type=XML`
   - 필드: `안건명, 안건번호(예: 근로기준정책과-3084), 해석일자, 질의요지, 회답`
   - **질의요지 = 자연스러운 실무 질문, 회답 = 조문 인용 촘촘**("「근로기준법」
     제60조제2항…") → 인용 정규식 파싱 = qrels 자동 구축.
   - 수량(키워드 검색 건수, 중복 있음): 근로기준법 214 / 기간제 416 / 산업안전 998 /
     임금 897 / 휴가 384 / 해고 226 → 풀 수천 건, 목표 200~500 여유.
2. **법제처 법령해석례** — `target=expc` (골드 멀티홉 보조):
   - 같은 URL 패턴. 필드: `안건명, 질의요지, 회답, 이유`.
   - 노동 8법 관련 **127건** — 수는 적지만 법 간 교차(근기법×기간제법 등)의
     진짜 멀티홉 질문 다수 → 어려운 평가 subset.
- 저작권: 국가 작성 해석·회신(저작권법 제7조 계열) + 공공데이터 → 평가셋
  재배포 가능(출처표시).
- 파싱 규칙: 「법령명」 제N조(제M항) 패턴 → 코퍼스 조문 id 매칭. 코퍼스 밖
  법령 인용은 라벨 제외(커버리지 통계로 기록).
- 보조: 위임 사슬(법 제N조→시행령 제M조)에서 멀티홉 질문 LLM 생성 + 수동 검수.
- 지표: recall@k, MRR (조문 단위 정답).

## 4. 아키텍처

### 에이전트 (react 타입 — plan_execute 아님)

근거: 검색 품질은 실행 전 알 수 없어 스텝마다 적응(검색→평가→재검색)이 본질.
디컴포지션은 DAG가 아니라 policy가 하위 질의를 순차 발행하는 것으로 충분.
서브에이전트 용도라 max_steps(4~6)로 비용 상한.

```
질문 → reason(9B: 리라이팅/디컴포지션/충분성판단)
     ⇄ act: statute_search | web_search | evidence_pack
     → finish: 구조화 근거 목록
```

### 툴 3종 (+policy 지침)

1. **statute_search(query, k, filters?)** — 내부 2단계:
   하이브리드 top-30 (dense Qwen3-Emb + BM25 fastembed → RRF)
   → 리랭커 top-8. 리랭커는 툴 **내부** 파이프라인(LLM이 결정할 일 아님),
   ablation용 on/off 플래그. 금융 `retrieval/` 모듈(embedder/index/search 패턴)
   재사용하되 PIT 필터 대신 메타데이터 필터(법령명/법종류).
2. **web_search(query)** — ddgs(구 duckduckgo-search, 키 불필요). 최신 개정
   확인 등 보조. 백엔드 교체 가능하게 프로토콜로 추상화.
3. **evidence_pack(...)** — dedup + 구조화 출력 조립 (LLM 없는 기계적 툴).

- **리라이팅/디컴포지션은 별도 툴 아님** — react policy 프롬프트의 지침
  (툴로 빼면 LLM 호출만 +1). 충분성 판정(CRAG식)도 reason 스텝이 겸함.
- **HITL 게이트 불필요** (읽기 전용) → checkpointer 없이 컴파일
  (= "서브에이전트에 HITL 금지" 제약과 정합).

### Qdrant 컬렉션 `statutes`

named dense(1024) + BM25 sparse (기존 `articles`와 동일 하이브리드 구조).
payload: `law_name, law_type, clause_no, clause_title, effective_date`.
임베딩 텍스트: `{law_name} 제{clause_no}조({clause_title})\n{clause_content}`.

### 히스토리 (멀티턴) — 2층 구분

| | 무엇 | 어디 |
|---|---|---|
| 대화 히스토리 | 유저↔봇 턴 | **UI 세션이 관리** → `arun(..., history=[...])`로 주입 (프레임워크 이미 지원) |
| thread 체크포인트 | HITL 재개용 실행 상태 | SQLite 30-thread 캡 (이 프로젝트에선 미사용) |

후속질문("그럼 육아휴직은?")의 지시어 해소 = policy 리라이팅 지침이 history 참조.

### UI — Gradio (React 프론트 아님)

근거: 커스텀 React는 택수증권으로 이미 증명됨; 이 데모의 볼거리는 **에이전트
스텝 트레이스**(분해된 쿼리, 재검색 횟수, 리랭크 점수, 근거 조문 카드)라
Gradio 채팅 + 접이식 중간이벤트가 적합. Streamlit은 rerun 모델이라 스트리밍
번거로움. `ui/app.py` 단일 파일, `server_name="0.0.0.0"`(WSL).
답변 생성 + 조문 인용까지 포함(검색만 하고 끝나지 않게).

## 5. 프레임워크 선행 확장 (basic_agent에서 TDD)

1. **AgentTool 다중 인자**: 현재 `query_arg` 하나만 서브그래프 초기 상태로 전달
   → 추가 kwargs(`filters` 등) 통과 옵션.
2. **구조화 출력**: react의 finish가 `final`(str)만 씀 → `ReactState`에 구조화
   채널(예: `evidence`) 추가 + `AgentTool(result_key=...)`로 추출 (result_key
   파라미터는 이미 존재).

## 6. 평가 & ablation 매트릭스 (핵심 산출물)

| 축 | 값 |
|---|---|
| 검색 | dense only / BM25 only / hybrid(RRF) |
| 리랭크 | 없음 / 9B LLM 리랭크 / Qwen3-Reranker-0.6B |
| 에이전트 | 1-shot 검색 / react 반복(리라이팅·디컴포지션) |
| 스텝 예산 | max_steps 2 / 4 / 6 |

+ latency 병기 → "전용 리랭커가 언제부터 값을 하는가", "에이전틱 반복이 recall을
얼마나 올리는가"가 이 프로젝트의 프론티어 표.

## 7. Phase 계획

| # | 내용 | 완료 기준 |
|---|---|---|
| 0 | 프로젝트 초기화: git init, vendored agent/, .gitignore(데이터 제외), scripts/build_corpus.py (HF 다운로드→노동 패키지 필터→data/corpus/) | 1,787 조문 parquet 재현 가능 |
| 0.5 | **평가셋 선행 구축** (유저 지시로 리스크 순서상 앞당김): moelCgmExpc+expc 수집 → 조문 인용 파싱 → qrels + 커버리지 리포트 | 200+ 쿼리 |
| 1 | 프레임워크 확장 2개 (basic_agent에서 TDD→푸시→vendor 동기화) | basic_agent 테스트 전부 green |
| 2 | 인덱싱: retriever/index 모듈 + build_index 스크립트, 리랭커 vLLM 서빙 검증 | `statutes` 컬렉션 1,787포인트, 하이브리드 쿼리 동작 |
| 3 | 툴 3종 TDD (fake embedder/qdrant/reranker/web) | 테스트 green |
| 4 | react policy 프롬프트(리라이팅·분해·충분성 지침) + 에이전트 조립 + 소규모 수동 검증 | 샘플 질문에서 멀티스텝 검색 트레이스 확인 |
| 5 | (0.5로 이동됨 — 필요 시 평가셋 확장만) | — |
| 6 | ablation 러너 + 매트릭스 표 | §6 표 완성 |
| 7 | README (한국어, Mermaid, 결과표) + 레포 푸시 | 공개 레포 |
| 8 | Gradio 데모 UI (`ui/app.py`) + 스크린샷 | 멀티턴 + 트레이스 시연 |

## 8. 레포 구조 (목표)

```
retriever_agent/          # 폴더명; 레포명 미정 (agentic-retriever / law-retriever-agent)
  agent/                  # vendored basic_agent
  retriever/              # tools(search 2-stage/web/pack), index, payload, config
  evaluation/             # qrels, metrics, runner
  scripts/                # build_corpus.py, build_index.py, build_evalset.py, run_ablations.py
  ui/app.py               # Gradio 데모
  data/                   # gitignored (corpus/, eval/) — build_corpus.py로 재현
  docs/design_and_plan.md # 이 문서
```

## 9. 결정 로그 (재논의 방지)

- react > plan_execute: 적응 루프가 본질, 이중 오케스트레이션 회피 (2026-07-17)
- 리랭커 v1 = 9B 겸용(rerank+충분성 한 호출), 전용 0.6B는 ablation 변수 (〃)
- 리라이터/디컴포지션 = policy 지침, 툴 아님 (〃)
- 웹서치 = ddgs 무료, 프로토콜로 백엔드 교체 가능 (〃)
- UI = Gradio, 히스토리 = UI 세션→history 파라미터 (〃)
- 코퍼스 = ducut91/korean-statutes 노동 8종 1,787조문; 평가 = 질의회시 qrels (〃)
- 데이터는 gitignore (합법이지만 유저 결정; 스크립트로 재현) (〃)
- 레포명 = `agentic_rag` (RAG 키워드 노출 목적 + 기존 snake_case 일관성) (〃)
- 평가셋 소스 확정: law.go.kr DRF API `moelCgmExpc`(질의회시 메인) + `expc`(멀티홉 127건) — 스파이크로 실검증 (〃)
