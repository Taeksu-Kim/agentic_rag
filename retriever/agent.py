"""리트리버 에이전트: react 루프 + statute_search(+web_search) 조립.

policy에 심는 도메인 지침이 이 에이전트의 심장이다 — 특히 **리라이팅**:
일상어 질문은 조문 어휘와 어긋나 1단계 검색이 정답을 아예 못 가져온다
(실측: "육아휴직 기간에도 연차휴가가 발생하나요?" -> top-30 밖; 법률 용어
리라이팅 -> 1위. docs/design_and_plan.md 참조). 리랭커는 1단계가 놓친 문서를
못 살리므로 어휘 갭은 오직 리라이팅/디컴포지션으로 푼다.

finish 계약: ``result = {"cids": [...]}`` — LLM은 근거 조문의 cid만 고르고,
전문 해소는 ``StatuteSearchTool.resolve()``가 코드로 한다 (환각 차단).
"""

from __future__ import annotations

from typing import Any, Optional

from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.react.graph import arun, build_react_agent
from retriever.tool import StatuteSearchTool
from retriever.web import WebSearchTool

SYSTEM_GUIDE = """너는 한국 노동법 조문 검색 에이전트다. 질문에 대한 근거 조문을 찾아라.

검색 규칙:
1. **리라이팅 필수**: 일상어 질문을 조문이 쓸 법한 **법률 용어의 자연어 명사구**로
   바꿔 검색하라. 예: "연차가 발생하나요" -> "연차 유급휴가 출근율 산정",
   "잘렸어요" -> "해고". AND/OR 같은 불리언 문법은 쓰지 마라 (효과 없음).
   첫 검색이 빗나가면 **다른** 법률 용어로 바꿔 다시 검색하라.
2. **디컴포지션**: 질문에 쟁점이 여러 개면(예: 육아휴직 + 연차) 쟁점별로 나눠
   각각 검색하라. 법-시행령 위임("대통령령으로 정하는")이 보이면 시행령도 찾아라.
3. **중복 검색 금지 / 충분성 판단**: 직전 검색과 거의 같은 쿼리를 반복하지 마라.
   같은 조문이 이미 상위에 나왔다면 그 쟁점은 해결된 것이다 — 모든 쟁점이
   커버되면 즉시 finish하라. snippet이 쟁점을 다루면 충분하다.
4. law_names 필터 값은 검색 결과의 law_name 표기 그대로 (예: "근로기준법 시행령"
   — 띄어쓰기 포함). 확실치 않으면 필터 없이 검색하라.
5. web_search는 조문 밖 보조 정보(최신 개정 소식 등)에만 사용.

finish 형식: final에는 근거 조문 내용에 기반한 **한국어 답변**을 반드시 쓰고
(빈 값 금지), result에는 {"cids": ["<검색 결과에 나온 cid 그대로>", ...]} 로
근거 조문 cid를 담아라. 검색 결과에 없던 cid를 지어내지 마라."""


def build_statute_agent(
    *,
    llm: LLMClient,
    client: Any,
    dense: Any,
    sparse: Any,
    reranker: Any = None,
    collection: str = "statutes",
    web_backend: Any = None,
    max_steps: int = 6,
) -> tuple[Any, StatuteSearchTool]:
    """(compiled graph, search_tool) — search_tool은 세션 해소용으로 함께 반환."""
    search_tool = StatuteSearchTool(
        client=client, collection=collection, dense=dense, sparse=sparse, reranker=reranker
    )
    registry = ToolRegistry()
    registry.add(search_tool)
    if web_backend is not None:
        registry.add(WebSearchTool(backend=web_backend))
    graph = build_react_agent(
        llm=llm, registry=registry, max_steps=max_steps, system=SYSTEM_GUIDE
    )
    return graph, search_tool


SYNTH_SYSTEM = (
    "너는 한국 노동법 상담 어시스턴트다. 주어진 조문 근거만으로 질문에 답하라. "
    "조문에 없는 내용을 지어내지 말고, 근거 조문을 명시하며 간결히 답하라."
)


_SYNTH_SCHEMA = {  # thinking 모델의 사고 덤프 방지 — 답변 필드를 스키마로 강제
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _synthesize_answer(llm: LLMClient, question: str, evidence: list[dict[str, Any]]) -> str:
    """근거 조문 전문으로 답변 1회 생성 — 루프의 finish에 의존하지 않는 안전망.

    (실측: 9B는 유사 쿼리 재검색 루프에 빠져 finish 없이 스텝 캡에 걸리는 일이
    잦다. 검색 루프의 역할은 근거 수집까지로 한정하고, 답변은 확보된 조문으로
    별도 합성하는 편이 항상 안정적이다. 조문은 2,500자까지 — 과도한 절단은
    정답 항(예: 60조⑥)을 날려먹는다.)
    """
    import json as _json

    ctx = "\n\n".join(e["text"][:2500] for e in evidence[:5])
    text = llm.complete(SYNTH_SYSTEM, f"질문: {question}\n\n근거 조문:\n{ctx}\n\n답변:",
                        schema=_SYNTH_SCHEMA)
    try:
        return str(_json.loads(text).get("answer", "")) or text
    except (ValueError, AttributeError):
        return text


async def run_statute_agent(
    graph: Any,
    search_tool: StatuteSearchTool,
    question: str,
    *,
    history: Optional[list[Any]] = None,
    fallback_k: int = 5,
    synth_llm: Optional[LLMClient] = None,
) -> dict[str, Any]:
    """한 질문 실행 -> {answer, evidence(전문 해소됨), steps(트레이스)}.

    ``synth_llm``이 주어지면, 루프가 답(final) 없이 끝났을 때 근거 조문으로
    답변을 합성한다.
    """
    search_tool.reset()  # 이전 질문의 세션이 새면 안 된다
    state = await arun(graph, question, history=history or [])

    cids = []
    result = state.get("result")
    if isinstance(result, dict):
        cids = [c for c in result.get("cids", []) if isinstance(c, str)]
    evidence = search_tool.resolve(cids)
    if not evidence:  # cid 미지정/전부 환각 -> 세션 최고점 폴백
        evidence = search_tool.top_session(k=fallback_k)

    answer = state.get("final", "") or ""
    if not answer.strip() and synth_llm is not None and evidence:
        answer = _synthesize_answer(synth_llm, question, evidence)

    return {
        "answer": answer,
        "evidence": evidence,
        "steps": state.get("scratchpad", []),
        "iterations": state.get("iteration", 0),
    }
