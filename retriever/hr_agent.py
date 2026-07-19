"""HR 컴플라이언스 에이전트 (UI 전용) = 조문 검색 + 처리성 업무 툴 + HITL.

build_statute_agent(검색만)의 상위 버전: 같은 statute_search에 처리 툴(연차 계산·
신청)과 ``ask_human``(입력 요청)을 얹고 **체크포인터**를 달아 HITL(interrupt/재개)을
지원한다. 에이전트가 상황을 보고 스스로 "검색 / 계산 / 신청(승인 필요)"을 호출한다.

**평가 그래프(build_statute_agent)는 이 파일을 임포트하지 않는다** — 처리 툴이
벤치마크에 새지 않도록 UI 경로에서만 사용.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.core.hitl import ask_human
from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.react.graph import build_react_agent
from retriever.hr_tools import AnnualLeaveCalcTool, LeaveRequestTool
from retriever.tool import StatuteSearchTool

# 검색 전용 SYSTEM_GUIDE를 상속하지 않는다 — 그쪽의 "쟁점 커버되면 즉시 finish"
# 편향이 처리(계산·신청) 단계로의 진행을 막는다(실측). HR 에이전트는 검색과 처리를
# 모두 다루는 독립 지침을 쓴다.
HR_GUIDE = """너는 한국 노동법 HR 어시스턴트다. 사용자의 요청 유형에 맞게 도구를 골라 처리한다.

도구:
- statute_search(query, law_names): 노동법 조문 검색. query는 법률 용어 명사구로.
- calculate_annual_leave(hire_date): 입사일(YYYY-MM-DD)로 올해 연차 일수 계산.
- submit_leave_request(start_date, days): 연차 사용 신청 제출(제출 직전 사람 승인 게이트).
- ask_human(question): 계산·신청에 필요한 값(입사일/사용일/일수)이 대화에 없을 때만 묻는다.

판단 규칙:
1. **규정을 묻는 질문**이면 statute_search로 근거 조문을 찾아 답한다. 질문이 여러
   쟁점이면 쟁점마다 한 번씩, 조문이 쓸 법률 용어로 리라이팅해 검색한다.
2. **"계산해줘"** 요청이면 계산이 목적이다 — 규정 설명이 굳이 필요 없으면 검색을
   건너뛰고 바로 calculate_annual_leave를 호출한다. 입사일이 없으면 ask_human으로 묻는다.
3. **"신청해줘"** 요청이면 start_date·days를 확보(없으면 ask_human)한 뒤
   submit_leave_request를 호출한다 — 이 도구는 실행 전 사람 승인을 받는다.
4. 대화에 이미 있는 값은 다시 묻지 마라.

finish(반드시): final에는 **결과를 담은 한국어 답변**을 꼭 써라(빈 값 금지).
- 계산했으면 계산 결과(예: "올해 연차 16일")와 근거를 쓴다.
- 신청했으면 접수 결과(접수번호·일정)를 쓴다.
- 조문만 찾았으면 조문 근거로 답한다.
result에는 {"cids": [...]}로 인용한 조문 cid가 있으면 담되, 없으면 생략해도 된다.
없는 cid를 지어내지 마라."""


def build_hr_agent(
    *,
    llm: LLMClient,
    client: Any,
    dense: Any,
    sparse: Any,
    reranker: Any = None,
    collection: str = "statutes",
    checkpointer: Any = None,
    max_steps: int = 8,
    valid_laws: Any = None,
) -> tuple[Any, StatuteSearchTool]:
    """(compiled graph, search_tool). checkpointer가 있어야 HITL(interrupt/재개) 동작."""
    search_tool = StatuteSearchTool(
        client=client, collection=collection, dense=dense, sparse=sparse,
        reranker=reranker, valid_laws=valid_laws,
    )
    registry = ToolRegistry()
    registry.add(search_tool)
    registry.add(AnnualLeaveCalcTool())
    registry.add(LeaveRequestTool())
    registry.register("ask_human", ask_human,
                      description="처리에 필요한 정보를 사용자에게 묻고 답을 받는다. args: question(str).",
                      args_schema={"type": "object", "properties": {"question": {"type": "string"}},
                                   "required": ["question"]})
    graph = build_react_agent(llm=llm, registry=registry, max_steps=max_steps,
                              checkpointer=checkpointer, system=HR_GUIDE)
    return graph, search_tool
