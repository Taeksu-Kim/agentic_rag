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

작업 절차:
1. **쟁점 분해 먼저**: 첫 reason에서 질문의 법적 쟁점을 목록으로 나열하라
   (하나뿐이면 하나). 법-시행령 위임("대통령령으로 정하는")이 예상되는 쟁점은
   시행령 조문 찾기를 별도 쟁점으로 세라.
2. **쟁점당 검색 1회**: 각 쟁점을 조문이 쓸 법한 **법률 용어의 자연어 명사구**로
   리라이팅해 검색하라. 예: "연차가 발생하나요" -> "연차 유급휴가 출근율 산정",
   "잘렸어요" -> "해고". AND/OR 같은 불리언 문법은 쓰지 마라 (효과 없음).
3. **커버 판정 후 다음 쟁점으로**: score가 높은(0.8+) 조문이 쟁점을 다루면 그
   쟁점은 해결이다 — 같은 쟁점을 표현만 바꿔 재검색하며 스텝을 낭비하지 마라.
   반대로 전부 낮으면(0.5 미만) 어휘가 틀린 것이니 **완전히 다른** 법률 용어로
   한 번만 재시도하라. 모든 쟁점이 커버되면 즉시 finish하라.
4. law_names 필터는 statute_search 설명의 **유효 법령명 목록 표기 그대로**만 써라.
   후보 법령이 여럿이면 **한 번에 여러 개**를 넣어라(예: 법 + 그 시행령).
   확실치 않으면 필터 없이 검색하라.
5. web_search는 조문 밖 보조 정보(최신 개정 소식 등)에만 사용.

finish 형식: final에는 근거 조문 내용에 기반한 **한국어 답변**을 반드시 쓰고
(빈 값 금지), result에는 {"cids": ["<검색 결과에 나온 cid 그대로>", ...]} 를 담되
**모든 쟁점에서 찾은 최고 조문을 빠짐없이** 포함하라 — 마지막 쟁점의 조문만 담고
앞 쟁점에서 찾아둔 조문을 빠뜨리는 실수를 하지 마라. 없던 cid를 지어내지 마라."""


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
    valid_laws: Any = None,
) -> tuple[Any, StatuteSearchTool]:
    """(compiled graph, search_tool) — search_tool은 세션 해소용으로 함께 반환.

    ``valid_laws``: 코퍼스의 법령명 목록 — 주면 툴 설명에 박혀 필터 오타/오특정이
    급감한다 (핀포인트 실측). 코퍼스 무관성을 위해 주입 파라미터로 받는다.
    """
    search_tool = StatuteSearchTool(
        client=client, collection=collection, dense=dense, sparse=sparse,
        reranker=reranker, valid_laws=valid_laws
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

    # 컨텍스트 예산: 4건 x 2,000자 + 질문 1,500자 ≈ 최악 ~6k 토큰 — 8192 한도 안.
    # (기존 5 x 2,500 + 무제한 질문은 꼬리에서 한도 초과 가능 — 설계 스윕에서 발견)
    ctx = "\n\n".join(e["text"][:2000] for e in evidence[:4])
    text = llm.complete(SYNTH_SYSTEM, f"질문: {question[:1500]}\n\n근거 조문:\n{ctx}\n\n답변:",
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
    max_evidence: int = 8,
    max_llm_picks: int = 5,
    seed_raw_search: bool = True,
    synth_llm: Optional[LLMClient] = None,
) -> dict[str, Any]:
    """한 질문 실행 -> {answer, evidence(전문 해소됨), steps(트레이스)}.

    evidence = LLM이 finish에서 고른 cid(최대 ``max_llm_picks``) + 세션 고점수
    조문으로 ``max_evidence``까지 채움. LLM 큐레이션만 믿지 않는 이유(ablation
    실측): finish가 마지막 쟁점의 저점수 조문 8개만 골라, 앞 스텝에서 0.99점으로
    찾아둔 정답이 통째로 밀려나는 실패가 관측됐다. 세션 점수 채움이 그 방어선.

    ``seed_raw_search``: 루프 시작 전에 **원 질문 그대로** 검색 1회를 세션에
    심는다 — 에이전트의 쟁점별 좁은 쿼리가 원문 통짜 검색의 커버리지를 대체하며
    잃는 손실(실측: 합집합 0.567 vs 대체 0.520)의 하한 방어선. LLM 호출이 아니라
    검색 1회라 비용은 ~0.2s.

    ``synth_llm``이 주어지면, 루프가 답(final) 없이 끝났을 때 근거 조문으로
    답변을 합성한다.
    """
    search_tool.reset()  # 이전 질문의 세션이 새면 안 된다
    if seed_raw_search:
        try:
            search_tool.run(query=question)  # 세션 누적용 — 관측은 버림
        except Exception:
            pass  # 시딩 실패가 본 실행을 막으면 안 된다
    state = await arun(graph, question, history=history or [])

    cids = []
    result = state.get("result")
    if isinstance(result, dict):
        cids = [c for c in result.get("cids", []) if isinstance(c, str)]
    evidence = search_tool.resolve(cids)[:max_llm_picks]
    seen = {e["cid"] for e in evidence}
    for r in search_tool.top_session(k=max_evidence):
        if len(evidence) >= max_evidence:
            break
        if r["cid"] not in seen:
            evidence.append(r)
            seen.add(r["cid"])

    answer = state.get("final", "") or ""
    if not answer.strip() and synth_llm is not None and evidence:
        answer = _synthesize_answer(synth_llm, question, evidence)

    return {
        "answer": answer,
        "evidence": evidence,
        "steps": state.get("scratchpad", []),
        "iterations": state.get("iteration", 0),
    }
