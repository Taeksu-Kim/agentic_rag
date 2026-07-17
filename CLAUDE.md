# CLAUDE.md

## Status

**설계 확정, Phase 0(초기화) 착수 전.** 모든 설계 결정·작업 계획·환경 주의사항은
**`docs/design_and_plan.md`가 단일 진실** — 작업 전 반드시 읽을 것.

## 프로젝트 한 줄

독립 실행 + 서브에이전트 겸용 **리트리버 에이전트** (react 타입, agentic RAG).
시나리오: HR 컴플라이언스 봇(한국 노동법 1,787조문, HF ducut91/korean-statutes).
범용 프레임워크는 basic_agent(github.com/Taeksu-Kim/basic_agent)를 vendoring.
별도 공개 레포 예정 (레포명 미정), **data/는 gitignore** (스크립트로 재현).

## 환경 요약 (상세는 설계 문서 §1~2)

- conda env `stock-dataset` — `/home/ubuntu/miniconda3/envs/stock-dataset/bin/python`
- WSL: loopback 깨짐 → 서버는 `10.5.0.2`; `/mnt/d`는 느린 9p
- 모델: 9B(:8000, util 0.60) + Qwen3-Embedding-0.6B(:8001) + Qwen3-Reranker-0.6B(:8002)
  — 9B는 로컬 경로 로드 필수, 기동은 큰 것부터 순차
- Qdrant 네이티브 서버(~/qdrant, 스토리지 ~/qdrant_storage), 컬렉션 `statutes` 예정
- **TDD, 테스트는 전부 fake** (네트워크/실서버 금지)
