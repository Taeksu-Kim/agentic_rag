"""2단계 리랭킹: 쿼리-문서 쌍 스코어러 (크로스인코더).

* ``VLLMReranker`` -- Qwen3-Reranker-0.6B를 vLLM ``/score``로 서빙한 것에 POST.
  (서빙 플래그는 docs/design_and_plan.md §2 -- sequence-classification 변환 필요.)
* ``FakeReranker`` -- 토큰 겹침 비율 (tests only).

⚠ Qwen3-Reranker는 **공식 채팅 템플릿이 필수**다. raw 쿼리/문서를 그대로 score에
넣으면 순위가 무의미해진다 (실측: 정답 조문이 최하위 -> 템플릿 적용 시 0.98로 1위).
``format_score_inputs``가 그 템플릿을 만든다.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from retriever import config

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a legal question, retrieve the statute clauses that answer it"


def format_score_inputs(
    query: str, docs: Sequence[str], *, instruction: str = DEFAULT_INSTRUCTION
) -> tuple[str, list[str]]:
    """(text_1, text_2[]) for the vLLM /score API in the official pair format."""
    text_1 = f"{_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
    text_2 = [f"<Document>: {d}{_SUFFIX}" for d in docs]
    return text_1, text_2


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        """Relevance score per doc (higher = more relevant)."""
        ...


class FakeReranker:
    """쿼리 토큰이 문서에 포함된 비율 (deterministic, tests only)."""

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        toks = [t for t in query.split() if t]
        if not toks:
            return [0.0] * len(docs)
        return [sum(t in d for t in toks) / len(toks) for d in docs]


_LLM_RERANK_SYSTEM = (
    "너는 한국 노동법 검색 리랭커다. 질문과 각 조문의 관련성을 0~10 점수로 매겨라. "
    "질문에 직접 답하는 조문이 10, 무관한 조문이 0이다. "
    "조문 개수와 같은 길이의 scores 배열로만 답하라."
)

_LLM_RERANK_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
    "required": ["scores"],
}


class LLMReranker:
    """범용 LLM(9B)에게 관련성 점수를 물어보는 리랭커 (ablation의 '9B 겸용' 축).

    배치(기본 10문서)로 나눠 호출 — 문서 전문을 다 넣으면 8k 컨텍스트가 넘친다.
    파싱 실패/길이 불일치 배치는 1단계 순서를 보존하는 미세 점수로 폴백
    (전부 0을 주면 다른 배치의 정상 점수와 섞여 순서가 깨진다).
    """

    def __init__(self, llm, *, doc_chars: int = 400, batch: int = 10) -> None:
        self._llm = llm
        self.doc_chars = doc_chars
        self.batch = batch

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        import json

        scores: list[float] = []
        for start in range(0, len(docs), self.batch):
            chunk = docs[start:start + self.batch]
            listing = "\n\n".join(
                f"[{i + 1}] {d[:self.doc_chars]}" for i, d in enumerate(chunk)
            )
            user = f"질문: {query}\n\n조문 {len(chunk)}개:\n{listing}"
            try:
                got = json.loads(self._llm.complete(
                    _LLM_RERANK_SYSTEM, user, schema=_LLM_RERANK_SCHEMA))["scores"]
                if len(got) != len(chunk):
                    raise ValueError("length mismatch")
                scores.extend(float(s) for s in got)
            except (ValueError, KeyError, TypeError):
                # 폴백: 이 배치는 1단계 순서 유지 (전역 인덱스 기반 미세 감쇠)
                scores.extend(-1e-6 * (start + i) for i in range(len(chunk)))
        return scores


class VLLMReranker:
    """vLLM score API (cross-encoder pair scoring)."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float = 300.0, instruction: str = DEFAULT_INSTRUCTION,
                 doc_chars: int = 1000, query_chars: int = 600,
                 max_len: int = 2048) -> None:
        # timeout 300s: GPU가 윈도우측 점유로 간헐 스톨하면 120s로는 실측 타임아웃
        self.base_url = (base_url or config.RERANKER_URL).rstrip("/")
        self.model = model or config.RERANKER_MODEL
        self.timeout = timeout
        self.instruction = instruction
        # 쿼리+문서 쌍이 서빙 max-model-len(2048)을 넘으면 400 — 그리고
        # truncate_prompt_tokens를 줘도 개별 텍스트만 잘려 쌍 합계가 한도를 넘으면
        # 요청이 **스케줄 불가로 무한 대기**한다 (실측: 2,439자 질의회시 쿼리 행).
        # 쿼리·문서 모두 클라이언트에서 절단해 쌍이 항상 한도 안에 들게 보장한다.
        self.doc_chars = doc_chars
        self.query_chars = query_chars
        self.max_len = max_len

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        import requests  # lazy

        text_1, text_2 = format_score_inputs(
            query[:self.query_chars], [d[:self.doc_chars] for d in docs],
            instruction=self.instruction)
        r = requests.post(
            f"{self.base_url}/score",
            json={"model": self.model, "text_1": text_1, "text_2": text_2,
                  "truncate_prompt_tokens": self.max_len},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["score"] for d in data]
