# Retriever Agent — 설계 & 작업 계획

> 이 문서는 세션이 바뀌어도 작업을 이어갈 수 있도록 **모든 결정과 문맥**을 담는다.
> 작성: 2026-07-17 (갱신 07-19). 상태: **Phase 0~8 실장 완료.** 레버 3종(union/
> struct/kiwi) + 퓨샷 리라이터 A/B(무효과·기각) + 멀티턴 응축(contextualize.py) +
> **Phase 8 Gradio UI(ui/app.py) + 데모 캡처(docs/media/, playwright)** 완료.
> 남은 것 = 커밋(유저 지시로 일괄 보류 중) + 선택적 고도화(§10). 최종 사다리(격식 701 R@8, kiwi 컬렉션 statutes_kiwi):
> sparse 0.507 / hybrid 0.570 / hybrid+ce 0.616(LLM 0회) / **struct+ce 0.640**.
> 제품 스코프(확답형 188, LLM 재라벨 핵심조문): **R@8 0.782, any-hit 0.862**.
> 에이전트는 시딩으로 1-shot 동률. 남은 진성 실패 ~14-22% = 어휘 갭(임베더 체급)
> + 교차 인용. 다음 배치: 임베더 업그레이드 실험 / τ 필터 → 조건부 세트 판정 →
> 전문 에스컬레이션 / 관측 재설계(상위 2건 600자) / Phase 8 UI.
> 평가 자산: answer_types.parquet(확답 188/조건부 494/판단불가 19),
> primary_labels.parquet(LLM 핵심조문), qrels_realistic.parquet(일상어 120).
>
> **레인 구분 (2026-07-19 유저 합의로 재정렬):** 개선 카드를 두 레인으로 나눈다.
> **(A) 파이프라인 레버** = 코퍼스·유스케이스 무관한 검색 스택 자체의 개선 —
> 포트폴리오 본편. **(B) 실서비스 고도화** = "이 도메인의 질문 분포를 안다"가
> 전제라 운영하며 로그가 쌓여야 값을 하는 옵션 카드 — §10으로 분리, 여기선 구현
> 안 함(설계·근거만 기록).
>
> **(A) 파이프라인 레버 — 남은 것:**
> 1. ~~퓨샷 리라이터~~ **완료·기각(무효과)**: 창작 퓨샷으로 리라이터를 질적
>    개선했으나 검색 지표 불변(일상어 struct 120중 2건만 변동) — "리라이터가
>    약한 탓" 대안 설명 제거, 리랭커-지배 결론 강화. 퓨샷은 트레이스 가독성
>    이득으로 코드 유지. (`docs/ablation_results.md` §퓨샷 리라이터)
> 2. 리랭커/임베더 체급 실험 (R@3↔R@8 갭 9.6pp = 리랭커 변별력; qrels 701이
>    리랭커 파인튜닝 데이터로 재사용 가능) — 남은 진성 어휘 갭의 단일 최대 레버.
> 3. 조건부 세트 판정 라운드 (τ필터 → 압축 뷰 1호출로 커버리지·재계획 → 플래그
>    청크만 전문 정독 — 상시 8병렬 아님, 비용 사다리 설계 합의됨)
> 4. 관측 재설계 (상위 2건 본문 600자 + 엣지 필드 — 위임 가시율 10%→73%+보험)
>
> **(A) 남은 시연/마무리 (현재 배치):**
> 5. 쿼리 디컴포지션 + 멀티턴 실험 몇 개 — agentic RAG가 실제로 동작함을 시연.
> 6. Phase 8 Gradio UI (스텝 스트리밍 + HITL 신청 데모) + 데모 영상/캡처 → 마무리.
>    스코프 라우팅 기능화(answer_type 분류를 파이프라인 앞단 게이트로)는 옵션.
>
> **(B) 실서비스 고도화 레버는 §10 참조** (doc2query·HyDE·과거질문 검색·
> co-citation — 전부 유스케이스/누적 데이터 전제, 여기선 구현 보류).
>
> **세션 시작 절차**: ① `bash scripts/serve.sh start` (Qdrant+임베더+리랭커+9B —
> EMBED_GPU=1 모드였음, .env EMBEDDER_URL=:8001) ② `pytest` 194개 green 확인
> ③ 재현 커맨드: `PYTHONPATH=. python scripts/run_ablations.py retrieval --arms
> struct+ce --collection statutes_kiwi --sparse kiwi --outdir data/eval/ablation_kiwi`.
> 평가 스코어는 data/eval/ablation*/. 이 문서 §결정 로그 먼저 읽을 것.
> 레포 공개됨(github.com/Taeksu-Kim/agentic_rag). 결과 = `docs/ablation_results.md`
> (v1 매트릭스 + v3 타깃 재측정), 실패 추적 = `docs/postmortem.md` (유저 지시로
> 추가된 컨셉 — 프론티어 모델 페어코딩의 실패 기록).
> 핵심 결론: ① 0.6B CE 리랭커 = 스위트스팟 (formal +11pp MRR, 일상어 갭도 흡수
> −1.4pp) ② react는 수정(필터 목록·정규화·0건 폴백, evidence=LLM픽5+세션 채움,
> 프레임워크 group(1) 버그픽스) 후에도 1-shot+CE 미역전 — 일상어 평가셋
> (qrels_realistic.parquet, 120)에서도 기각. 에이전트 가치 = 해석가능성/멀티소스.
> ③ 남은 개선 카드: 원 질문 그대로 검색 1회 상시 포함(미측정).
> 다음 = **Phase 8 (Gradio UI + HITL 신청 데모)**. basic_agent 업스트림 완료 여부는
> 커밋 로그 확인 (group(0) 픽스 + 회귀 테스트 2개).
>
> Phase 6 구현 내역 (2026-07-18, 156 tests green):
> - `evaluation/metrics.py` (recall@k, MRR) + `evaluation/runner.py`
>   (arm별 JSONL **재개 가능** 러너, per-query latency, summarize, 마크다운 표).
> - `retriever/search.py`에 `mode="hybrid"|"dense"|"sparse"` (단독 모드는 top-level
>   `query_filter` 사용 — 단독 벡터 쿼리에선 정상 적용됨).
> - `retriever/reranker.py::LLMReranker` — 9B에 0~10 점수 (batch 10, doc 400자 절단,
>   파싱 실패 배치는 1단계 순서 보존 폴백).
> - **VLLMReranker 수정**: 서빙 max-model-len 2048 초과 시 400 → 문서 1,200자
>   클라 절단 + `truncate_prompt_tokens` 안전망 (좌측 절단이라 안전망으로만).
> - `scripts/run_ablations.py` — `retrieval|agent|report` 서브커맨드. arm =
>   `{dense|sparse|hybrid}[+ce|+llm]`, `react-s{2,4,6}`. 에이전트는 고정 샘플
>   120개(seed 42, `data/eval/ablation/sample_ids.json`에 고정), report가 동일
>   부분집합 비교표 포함 → `docs/ablation_results.md`.
> - 실행 순서(백그라운드, `~/agentic_rag_logs/ablation_run.log`): 저렴한 6 arm 전체
>   701쿼리 → hybrid+llm 전체 → react s2/4/6 ×120 → report. dense+llm/sparse+llm은
>   비용 대비 통찰이 적어 기본 제외 (필요 시 `--arms`로 추가).
> - 실측 latency/query: hybrid 0.08s / +ce 0.18s / +llm ~5s / react-s4 ~77s.
>   9B LLM timeout 300s 필수 (thinking 2048토큰 ≈ 65s+, 기본 120s에서 실측 타임아웃).
>
> Phase 4 완료 내역 (2026-07-18):
> - 서빙: `scripts/serve.sh` — **EMBED_GPU=1 모드로 3모델 GPU 동주 성공**
>   (`--kv-cache-memory-bytes`로 vLLM 0.24 회계 우회; 임베더 512MiB, 9B 1.5GiB).
>   .env EMBEDDER_URL=:8001(GPU). CPU 임베더(:8011, cpu_embed_server.py)는 예비
>   (GPU 버전과 코사인 0.9998+ 일치 검증됨).
> - 에이전트 라이브 검증: "육아휴직 기간에도 연차휴가가 발생하나요?" →
>   9B가 스스로 법률용어 리라이팅("육아휴직 중 연차유급휴가 발생") →
>   근기법 60조 리랭크 1위 → **합성 답변이 60조⑥ 정확 인용**. 완주.
> - 프레임워크 추가(95 tests): `build_react_agent(system=)` 도메인 지침 주입,
>   policy 재시도(에러 피드백)+우아한 종료, scratchpad 관측 1,500자 절단
>   (컨텍스트 폭발 방지), OpenAICompatLLM sampling 오버라이드+max_tokens 1024 기본
>   +HTTP 에러 본문 노출.
> - 리트리버(139 tests 전체 green): 답변은 루프 finish에 의존하지 않고
>   **_synthesize_answer가 근거 조문(각 2,500자)으로 별도 합성** (스키마로 답변
>   필드 강제 — thinking 덤프 방지). 9B 습성: 유사쿼리 재검색 루프 → 캡 도달이
>   흔함(향후 개선 여지, ablation에는 영향 없음 — 근거는 수집됨).
> - 실행: `PYTHONPATH=. python scripts/ask.py "<질문>" [--trace] [--no-rerank]`
> - **커밋 전부 보류 중** (유저 지시: 작업 종료 후 일괄). basic_agent 원본
>   ("/mnt/d/workspace/stock dataset/agent")도 미커밋 변경 있음 — 커밋 시
>   news_reaction_agent vendor 재동기화도 함께.

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

- 실측 배치(2026-07-18): 9B 0.68(가중치 12.73GiB — mamba 하이브리드라 예상보다 큼,
  KV 1.41GiB) + 임베더 0.10 + 리랭커 0.10, len 4096(소형 2종).
- **기동은 큰 것(9B)부터 순차 + 완료 확인 후 다음** — 동시에 띄우거나 죽이면
  프로파일링 어서션("Initial free memory ... current free memory")으로 연쇄 실패.
- ⚠ **윈도우측 GPU 점유 주의**: 윈도우 앱이 ~4GB를 먹으면 합계가 한계를 넘어
  WDDM 스필 -> 생성이 0.5 tok/s로 붕괴(타임아웃). 벤치마크/데모 전에 윈도우측
  GPU 사용 앱을 닫을 것. `nvidia-smi memory.used`로 베이스라인 확인.
- 빠듯하면 9B에 `--kv-cache-dtype fp8`, 0.6B들에 `--max-model-len 2048~4096`.
- **9B는 반드시 로컬 경로로 로드** — HF repo id는 메모리 프로파일링에서 데드락:
  `"/mnt/d/workspace/stock dataset/models/Qwen3.5-9B-FP8-dynamic"`.
  전체 명령은 stock dataset의 CLAUDE.md "Local vLLM serving" 절 참조
  (CUDA_HOME=cu13 경로, VLLM_USE_FLASHINFER_SAMPLER=0, --enforce-eager).
- ✅ **리랭커 서빙 검증 완료 (2026-07-18)**. vLLM 서빙 레시피:

      vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling \
        --served-model-name qwen3-reranker --max-model-len 8192 \
        --gpu-memory-utilization 0.12 --port 8002 \
        --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'

  ⚠ **공식 채팅 템플릿 필수** — raw 쿼리/문서로 /score를 치면 순위가 무의미
  (실측: 정답 조문 최하위). `retriever/reranker.py::format_score_inputs`가
  prefix/suffix + <Instruct>/<Query>/<Document> 템플릿을 적용하며, 적용 시
  정답 0.98 / 오답 0.001로 분리. /score와 /v1/score 둘 다 동작.
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
| 8 | Gradio 데모 UI (`ui/app.py`) + 스크린샷 — **스텝 스트리밍**(LangGraph astream -> Gradio 제너레이터; 수제 SSE는 스킵 결정) + **HITL 가상 신청 시나리오**(답변 후 ask_human "신청하시겠어요?" -> 정보 수집 -> 가상 submit_application 툴 -> 접수번호 반환; InMemorySaver, UI 전용 레지스트리라 벤치마크와 분리) | 멀티턴 + 트레이스 + HITL 신청 흐름 시연 |

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
- evidence_pack 툴 폐기 -> finish 계약으로 대체: LLM은 cid만 고르고 전문 해소는
  `StatuteSearchTool.resolve()`가 코드로 (환각 차단, LLM 호출 절약) (2026-07-18)
- react system 프롬프트 주입 확장: `build_react_agent(system=...)` — 기본 ReAct
  계약에 suffix (basic_agent 90 tests) (〃)
- UI 실시간 상태: Gradio 제너레이터 스트리밍으로 충분, 수제 SSE/FastAPI는 보류
  (API 서비스로 노출할 때만 재검토) (〃)
- HITL 가상 신청 시나리오는 UI 전용 그래프에만 (평가 그래프는 statute_search만
  등록 -> 벤치마크 오염 없음) (〃)
- 개선 카드를 파이프라인 레버(범용) vs 실서비스 고도화(누적 데이터 전제)로 분리:
  후자는 §10으로 빼고 포트폴리오 본편에선 구현 안 함 (2026-07-19)
- 퓨샷 리라이터 = 검색 지표 무효과로 기각, 트레이스 가독성 위해 코드만 유지;
  doc2query는 코드 셸빙(비용이 코퍼스 크기에 비례하는 패턴 회피) (〃)
- 멀티턴 = 프레임워크(react)가 history를 안 쓰므로(build_user_prompt는 query만)
  리트리버 레벨 condense-question으로 구현(`retriever/contextualize.py`) — basic_agent
  불변, 표준 대화형 RAG 패턴 (2026-07-19)
- Phase 8 UI = `ui/app.py`(Gradio 6, 멀티턴 채팅 + 접이식 트레이스 thought + 근거
  조문 표). 데모 드라이버 = `scripts/demo.py`(build_live/answer_turn 재사용).
  캡처 = `scripts/capture_demo.py`(playwright, 시스템 chrome channel — 라이브
  구동 스크린샷·영상 → docs/media/). 실제 서빙 백엔드로 UI 경로 E2E 검증 완료 (〃)

## 10. 실서비스 고도화 레버 (누적 데이터 전제 — 본편 미구현)

전부 "이 도메인의 실제 질문 분포를 안다"가 전제라, HR 봇을 실제로 운영하며
질의 로그·피드백이 쌓였을 때 켜는 옵션이다. 지금(정적 코퍼스 + 학술 평가셋)
구현하면 평가셋 과적합 위험이 크고 범용성 주장과도 어긋나므로 **설계·근거만
기록**한다. 파이프라인 레버(§상단 A)와 명확히 구분하는 것 자체가 설계 판단.

- **doc2query 역질문 증강** — 조문마다 9B가 예상 질문 2~3개를 생성해 임베딩
  텍스트에 부착(docT5query), 일상어↔규범어 갭을 문서 쪽에서 닫는다. 코드·배관
  실장·TDD 완료(`retriever/doc2query.py`, `scripts/gen_doc2query.py`,
  `build_index.py --doc2query`) 후 **실행 보류** — 인덱싱 비용이 LLM 추론에
  비례(코퍼스·청크 커질수록 부담)해 정적 코퍼스에선 값이 안 맞는다는 유저 판단.
  운영 로그로 "실제로 자주 묻는 각도"를 알면 그때 표적 생성이 정당해진다.
- **HyDE** — 쿼리 시점에 9B가 "답이 될 조문 초안"을 써서 그걸 임베딩. doc2query와
  같은 갭을 쿼리 쪽에서 닫음. 이 데이터셋엔 먹힐 여지가 있으나 **범용성이 약해
  회피**(도메인 특화 가정이 큼) — 유저 판단.
- **과거 질문–질문 검색 (LLM 0회)** — 평가 제외 질의회시 질문을 별도 컬렉션에
  임베딩, 새 질문과 질문끼리 매칭 → 매칭된 과거 질의의 인용 조문을 후보 풀에
  합류. 일상어↔일상어라 어휘 갭이 없고 HR 봇의 질문 반복성과 맞음. co-citation
  수집과 데이터 공유. ⚠ 평가셋 701 마이닝 제외 필수(누수).
- **동시 인용(co-citation) 그래프** — 질의회시 대량 수집 풀에서 "함께 인용되는
  조문 쌍"을 엣지로 → 교차 법령 연결 보강. `retriever/edges.py` 재사용. ⚠ 평가셋
  제외.
- **용어 사전(글로서리) 확장** — 병렬 풀에서 동시출현으로 "잘렸다↔해고" 정렬을
  마이닝해 kiwi BM25 쿼리 확장. 결정적·LLM 0회지만 효과 상한이 낮아 후순위.
