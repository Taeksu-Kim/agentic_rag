"""HR 처리 툴: 연차 계산 + 연차 신청(HITL 승인) — 결정적, 네트워크 없음."""

import pytest

from retriever.hr_tools import AnnualLeaveCalcTool, LeaveRequestTool


# -- 연차 계산 (근기법 60조: 1년 15일, 3년부터 2년마다 +1, 최대 25) --

@pytest.mark.parametrize("hire,asof,days,years", [
    ("2024-03-02", "2025-06-01", 15, 1),   # 1년차 -> 15
    ("2023-01-01", "2025-06-01", 15, 2),   # 2년차 -> 15
    ("2022-01-01", "2025-06-01", 16, 3),   # 3년차 -> 16 (가산 1)
    ("2021-03-02", "2025-06-01", 16, 4),   # 4년차 -> 16
    ("2020-01-01", "2025-06-01", 17, 5),   # 5년차 -> 17 (가산 2)
    ("1990-01-01", "2025-06-01", 25, 35),  # 상한 25
])
def test_annual_leave_days(hire, asof, days, years):
    out = AnnualLeaveCalcTool().run(hire_date=hire, as_of=asof)
    assert out["days"] == days
    assert out["years"] == years
    assert hire in out["detail"]


def test_annual_leave_under_one_year_is_monthly():
    # 1년 미만: 1개월 개근당 1일 (최대 11)
    out = AnnualLeaveCalcTool().run(hire_date="2025-01-01", as_of="2025-06-15")
    assert out["years"] == 0
    assert out["days"] == 5  # 1~5월 개근분 5일


def test_annual_leave_bad_date_returns_error():
    out = AnnualLeaveCalcTool().run(hire_date="어제")
    assert "error" in out


# -- 연차 신청 (HITL 게이트는 그래프가 담당; 툴은 requires_approval 표식 + 접수 처리) --

def test_leave_request_requires_approval_flag():
    assert LeaveRequestTool().requires_approval is True


def test_leave_request_returns_receipt():
    out = LeaveRequestTool().run(start_date="2026-08-01", days=3)
    assert out["status"] == "submitted"
    assert out["receipt_no"].startswith("A-")
    assert out["start_date"] == "2026-08-01" and out["days"] == 3


def test_leave_request_receipt_is_deterministic_per_args():
    a = LeaveRequestTool().run(start_date="2026-08-01", days=3)["receipt_no"]
    b = LeaveRequestTool().run(start_date="2026-08-01", days=3)["receipt_no"]
    assert a == b  # 같은 신청 = 같은 접수번호(멱등 데모)
