"""HR 처리성 업무 툴 — 조문 검색(정보 조회) 다음의 실무 처리 단계.

에이전트 파이프라인에 그대로 얹는 BaseTool들이다(특정 케이스용 GUI 하드코딩이
아님): 에이전트가 상황을 보고 스스로 호출하고, 신청 툴은 ``requires_approval``로
HITL 승인 게이트를 탄다. **UI 전용 그래프에만 등록** — 평가 그래프는 미접촉.

- ``AnnualLeaveCalcTool``  : 입사일 -> 연차 일수 (근기법 제60조 공식, 결정적).
- ``LeaveRequestTool``     : 연차 신청 접수 (requires_approval=True → 그래프가 interrupt).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from agent.core.tools import BaseTool


def _parse(d: str) -> date:
    return datetime.strptime(d.strip(), "%Y-%m-%d").date()


def _completed_years(hire: date, as_of: date) -> int:
    """만 근속연수(완성된 해 수)."""
    y = as_of.year - hire.year - ((as_of.month, as_of.day) < (hire.month, hire.day))
    return max(0, y)


class AnnualLeaveCalcTool(BaseTool):
    name = "calculate_annual_leave"
    description = (
        "근로자의 입사일로 올해 연차 유급휴가 일수를 계산한다(근기법 제60조). "
        "args: hire_date='YYYY-MM-DD' (필수), as_of='YYYY-MM-DD' (선택, 기본 오늘). "
        "입사일을 모르면 먼저 ask_human으로 물어라."
    )
    args_schema = {
        "type": "object",
        "properties": {"hire_date": {"type": "string"}, "as_of": {"type": "string"}},
        "required": ["hire_date"],
    }

    def run(self, hire_date: str, as_of: str | None = None) -> dict[str, Any]:
        try:
            hire = _parse(hire_date)
            today = _parse(as_of) if as_of else date.today()
        except (ValueError, TypeError):
            return {"error": "날짜 형식이 올바르지 않습니다. 'YYYY-MM-DD'로 입력하세요."}
        years = _completed_years(hire, today)
        if years < 1:  # 1년 미만: 1개월 개근당 1일 (최대 11)
            months = (today.year - hire.year) * 12 + (today.month - hire.month) \
                - (1 if today.day < hire.day else 0)
            days = max(0, min(11, months))
            detail = (f"입사 {hire_date} 기준 근속 1년 미만 → 1개월 개근당 1일 = {days}일 "
                      f"(근기법 제60조제2항)")
        else:
            days = min(25, 15 + (years - 1) // 2)  # 15 + 3년차부터 2년마다 +1, 상한 25
            gasan = days - 15
            detail = (f"입사 {hire_date} 기준 근속 {years}년차 → 기본 15일"
                      + (f" + 가산 {gasan}일" if gasan else "")
                      + f" = {days}일 (근기법 제60조제1·4항, 상한 25일)")
        return {"days": days, "years": years, "detail": detail}


class LeaveRequestTool(BaseTool):
    name = "submit_leave_request"
    description = (
        "연차 유급휴가 사용을 신청서로 제출한다. args: start_date='YYYY-MM-DD', "
        "days=정수. 제출 전 사람의 승인을 받는다(승인 게이트)."
    )
    args_schema = {
        "type": "object",
        "properties": {"start_date": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["start_date", "days"],
    }
    requires_approval = True  # HITL: 그래프가 실행 전 interrupt로 승인을 받는다

    def run(self, start_date: str, days: int) -> dict[str, Any]:
        h = hashlib.md5(f"{start_date}|{days}".encode()).hexdigest()[:4].upper()
        return {
            "status": "submitted",
            "receipt_no": f"A-{start_date.replace('-', '')[:6]}-{h}",
            "start_date": start_date,
            "days": int(days),
            "message": f"{start_date}부터 연차 {days}일 신청서가 제출되었습니다. 결재 대기 중.",
        }
