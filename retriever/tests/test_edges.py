"""조문 그래프 엣지 추출 (합성 미니 코퍼스, 오프라인)."""

from retriever.edges import EdgeIndex, build_edges, parent_law

CLAUSES = [
    {"law_name": "산업안전보건법", "law_type": "법률", "clause_no": "17",
     "clause_title": "안전관리자",
     "clause_content": "안전관리자의 수·자격은 대통령령으로 정한다."},
    {"law_name": "산업안전보건법 시행령", "law_type": "대통령령", "clause_no": "16",
     "clause_title": "안전관리자의 선임 등",
     "clause_content": "법 제17조제1항에 따라 안전관리자를 두어야 하는 사업의 종류는 별표와 같다."},
    {"law_name": "산업안전보건법 시행규칙", "law_type": "고용노동부령", "clause_no": "9",
     "clause_title": "서류 제출",
     "clause_content": "영 제16조제6항에 따라 증명서류를 제출해야 한다. 법 제17조에 따른 선임도 같다."},
    {"law_name": "산업안전보건법", "law_type": "법률", "clause_no": "168",
     "clause_title": "벌칙",
     "clause_content": "제17조를 위반한 자는 벌금에 처한다. 「근로기준법」 제60조를 준용한다."},
    {"law_name": "근로기준법", "law_type": "법률", "clause_no": "60",
     "clause_title": "연차 유급휴가", "clause_content": "연차 조문."},
    {"law_name": "고용보험법 시행령", "law_type": "대통령령", "clause_no": "1-2",
     "clause_title": "보수 제외 금품",
     "clause_content": "법 제2조제5호 본문에서 정하는 금품이란 비과세 소득을 말한다."},
]


def test_parent_law():
    assert parent_law("산업안전보건법 시행규칙") == "산업안전보건법 시행령"
    assert parent_law("산업안전보건법 시행령") == "산업안전보건법"
    assert parent_law("산업안전보건법") is None


def test_delegation_edges_resolve_to_root_law():
    edges = set(build_edges(CLAUSES))
    assert ("산업안전보건법 시행령|16", "산업안전보건법|17", "위임") in edges
    # 시행규칙: "영 제16조" -> 시행령, "법 제17조" -> 법률
    assert ("산업안전보건법 시행규칙|9", "산업안전보건법 시행령|16", "위임") in edges
    assert ("산업안전보건법 시행규칙|9", "산업안전보건법|17", "위임") in edges


def test_penalty_and_explicit_edges():
    edges = set(build_edges(CLAUSES))
    assert ("산업안전보건법|168", "산업안전보건법|17", "벌칙") in edges
    assert ("산업안전보건법|168", "근로기준법|60", "명시참조") in edges


def test_dangling_refs_dropped():
    edges = build_edges(CLAUSES)
    # "법 제2조제5호" -> 고용보험법|2 는 코퍼스에 없음 -> 엣지 없어야
    assert not any(d == "고용보험법|2" for _, d, _ in edges)


def test_edge_index_bidirectional_with_priority():
    idx = EdgeIndex(build_edges(CLAUSES))
    comps = idx.companions("산업안전보건법|17")
    cids = [c for c, _ in comps]
    assert "산업안전보건법 시행령|16" in cids   # 역방향(참조받음)
    assert cids[0] == "산업안전보건법 시행령|16" or comps[0][1] == "위임"  # 위임 우선
    assert "산업안전보건법|168" in cids          # 벌칙 역참조
    assert len(idx.companions("산업안전보건법|17", cap=1)) == 1
