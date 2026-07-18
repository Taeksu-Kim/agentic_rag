"""Statute-citation parser: 해석례/질의회시 텍스트 → (법령명, 조문번호) 라벨.

정확도 우선(정밀도 > 재현율): 애매한 참조는 버린다 — qrels에 오답이 섞이는 것이
빠지는 것보다 나쁘다.
"""

from evaluation.citations import extract_citations

KNOWN = [
    "근로기준법", "근로기준법 시행령", "근로기준법 시행규칙",
    "기간제 및 단시간근로자 보호 등에 관한 법률",
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    "최저임금법", "최저임금법 시행령", "산업안전보건법",
]


def test_basic_bracketed_citation():
    text = "「근로기준법」 제60조제2항에 따라 연차휴가가 부여된다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "60")]


def test_branch_article_uijo():
    text = "「근로기준법」 제74조의2에 따른 태아검진 시간"
    assert extract_citations(text, KNOWN) == [("근로기준법", "74-2")]


def test_same_law_relative_reference():
    text = "「근로기준법」 제50조에서 근로시간을 정하고, 같은 법 제61조에 따라 촉진한다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "50"), ("근로기준법", "61")]


def test_same_law_enforcement_decree():
    text = "「근로기준법」 제55조 및 같은 법 시행령 제30조에 따른 유급휴일"
    assert extract_citations(text, KNOWN) == [("근로기준법", "55"), ("근로기준법 시행령", "30")]


def test_dongbeop_synonym():
    text = "「최저임금법」 제5조와 동법 시행령 제5조를 함께 본다."
    assert extract_citations(text, KNOWN) == [("최저임금법", "5"), ("최저임금법 시행령", "5")]


def test_alias_defined_inline():
    text = (
        "「기간제 및 단시간근로자 보호 등에 관한 법률」(이하 “기간제법”이라 함) 제4조와 "
        "기간제법 제8조가 적용된다."
    )
    assert extract_citations(text, KNOWN) == [
        ("기간제 및 단시간근로자 보호 등에 관한 법률", "4"),
        ("기간제 및 단시간근로자 보호 등에 관한 법률", "8"),
    ]


def test_unknown_law_excluded():
    text = "「파견근로자 보호 등에 관한 법률」 제5조와 「근로기준법」 제2조를 본다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "2")]


def test_unknown_law_does_not_capture_context():
    # 모르는 법령이 컨텍스트가 된 뒤의 '같은 법'은 그 모르는 법령을 가리킴 -> 제외
    text = "「파견근로자 보호 등에 관한 법률」 제5조와 같은 법 제6조를 본다."
    assert extract_citations(text, KNOWN) == []


def test_bare_article_follows_context():
    # 나열: 「근로기준법」 제50조, 제69조 본문 -> 둘 다 근로기준법
    text = "「근로기준법」 제50조, 제69조 본문에 따른 근로시간"
    assert extract_citations(text, KNOWN) == [("근로기준법", "50"), ("근로기준법", "69")]


def test_hang_enumeration_single_article():
    # 항 나열은 조 하나: 제60조 제1항·제3항 및 제4항 -> ("근로기준법", "60")만
    text = "「근로기준법」 제60조제1항ㆍ제3항 및 제4항에 따른 유급휴가"
    assert extract_citations(text, KNOWN) == [("근로기준법", "60")]


def test_bare_article_without_context_is_dropped():
    text = "제5조에 따라 처리한다."  # 어느 법인지 알 수 없음 -> 버림
    assert extract_citations(text, KNOWN) == []


def test_deduplication_preserves_order():
    text = "「근로기준법」 제60조, 같은 법 제60조, 같은 법 제61조"
    assert extract_citations(text, KNOWN) == [("근로기준법", "60"), ("근로기준법", "61")]


def test_middle_dot_normalization():
    # 남녀고용평등법의 'ㆍ'가 '·'(U+00B7)로 표기돼도 매칭
    text = "「남녀고용평등과 일·가정 양립 지원에 관한 법률」 제19조에 따른 육아휴직"
    assert extract_citations(text, KNOWN) == [
        ("남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률", "19")
    ]


def test_unbracketed_known_law_name():
    # 질의회시 회답은 「」 없이 쓰는 경우가 많다
    text = "근로기준법 제10조제2항에 의하면 대통령령이 정하는 바에 따라 적용할 수 있다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "10")]


def test_unbracketed_decree_longest_match():
    # "근로기준법 시행령"이 "근로기준법"보다 먼저(길게) 매칭돼야 함
    text = "근로기준법 시행령 제30조 및 근로기준법 제55조 참조"
    assert extract_citations(text, KNOWN) == [("근로기준법 시행령", "30"), ("근로기준법", "55")]


def test_dongbeop_decree_unbracketed_context():
    text = "근로기준법 제10조제2항과 동법 시행령 제1조의2를 본다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "10"), ("근로기준법 시행령", "1-2")]


def test_buchik_articles_are_dropped():
    # 부칙 조문은 본문 조문이 아니다 -> 다음 법령 언급 전까지 버림
    text = "근로기준법 부칙 제2조에 의거 신고할 수 있고, 「최저임금법」 제5조를 본다."
    assert extract_citations(text, KNOWN) == [("최저임금법", "5")]


def test_buchik_then_rel_recovers():
    text = "「근로기준법」 부칙 제1조에서 시행일을 정하며, 같은 법 제50조가 적용된다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "50")]


def test_unknown_unbracketed_law_does_not_steal_context():
    # 코퍼스 밖 법령이 「」 없이 인용돼도 직전 known 컨텍스트에 귀속되면 안 된다
    text = "「산업안전보건법」 제63조의 도급인 책임이며, 공동주택관리법 제7조에 따라 위탁한다."
    assert extract_citations(text, KNOWN) == [("산업안전보건법", "63")]


def test_unknown_unbracketed_decree_does_not_steal_context():
    text = "근로기준법 제50조를 보고, 파견법 시행령 제3조도 참고한다."
    assert extract_citations(text, KNOWN) == [("근로기준법", "50")]


def test_rel_after_unknown_unbracketed_is_dropped():
    text = "공동주택관리법 제6조와 같은 법 제7조에 따른 관리주체"
    assert extract_citations(text, KNOWN) == []
