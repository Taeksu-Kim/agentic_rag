"""조문 그래프 엣지: 위임(법↔시행령↔시행규칙)·벌칙 참조를 정규식으로 추출.

동기(실측): 1단계 풀 미스의 34%가 "이미 적중한 조문의 같은-법-계열 동반 조문".
하위법령 조문은 "법 제N조에 따라…" 역참조로 시작하는 관례가 있어(1,094개 중
753개) LLM 없이 코드로 그래프를 깔 수 있다. 검색 후 적중 조문의 동반 조문을
규칙으로 부착하는 데 쓴다 (parent-document retriever의 법령판).

엣지 방향: src(참조하는 조문) -> dst(참조되는 조문). 확장은 양방향 인덱스로.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping

_JO = r"제(\d+)조(?:의(\d+))?"
# "법 제N조" — 하위법령이 모법을 부르는 관례 ("이하 '법'이라 한다").
# 앞이 한글이면 다른 법령명의 끝("…평등법 제19조")이므로 제외.
_LAW_REF = re.compile(r"(?<![가-힣])법\s*" + _JO)
# "영 제N조" — 시행규칙이 시행령을 부르는 관례 ("운영 제도" 류 오매치도 한글 제외로 차단)
_DECREE_REF = re.compile(r"(?<![가-힣])영\s*" + _JO)
# 「법명」 제N조 — 명시 참조 (코퍼스 내 법령만 엣지化)
_EXPL_REF = re.compile(r"「([^」]+)」\s*" + _JO)
# 벌칙/과태료 조문의 같은 법 내부 참조 "제N조"
_INTERNAL = re.compile(_JO)
_PENALTY_TITLE = re.compile(r"벌칙|과태료|양벌")


def _cid(law: str, jo: str, ui: str | None) -> str:
    return f"{law}|{jo}-{ui}" if ui else f"{law}|{jo}"


def parent_law(law_name: str) -> str | None:
    """'X 시행령'/'X 시행규칙' -> 'X 시행령'의 모체. 법률이면 None."""
    if law_name.endswith(" 시행규칙"):
        return law_name[: -len(" 시행규칙")] + " 시행령"
    if law_name.endswith(" 시행령"):
        return law_name[: -len(" 시행령")]
    return None


def build_edges(clauses: Iterable[Mapping]) -> list[tuple[str, str, str]]:
    """[(src_cid, dst_cid, kind)] — kind: 위임 | 벌칙 | 명시참조.

    clauses: law_name/law_type/clause_no/clause_title/clause_content 매핑들.
    코퍼스에 존재하는 cid로만 엣지를 만든다 (dangling 방지는 호출측 corpus set).
    """
    rows = list(clauses)
    known = {f"{r['law_name']}|{r['clause_no']}" for r in rows}
    edges: set[tuple[str, str, str]] = set()

    for r in rows:
        law, text = r["law_name"], str(r["clause_content"])
        src = f"{law}|{r['clause_no']}"
        base = parent_law(law)

        if base is not None:
            # "법 제N조" -> 계열의 최상위 법률
            root = parent_law(base) or base
            for m in _LAW_REF.finditer(text):
                dst = _cid(root, m.group(1), m.group(2))
                if dst in known:
                    edges.add((src, dst, "위임"))
            if base != root:  # 시행규칙의 "영 제N조" -> 시행령
                for m in _DECREE_REF.finditer(text):
                    dst = _cid(base, m.group(1), m.group(2))
                    if dst in known:
                        edges.add((src, dst, "위임"))

        for m in _EXPL_REF.finditer(text):  # 「법명」 제N조 (교차 참조 포함)
            dst = _cid(m.group(1).strip(), m.group(2), m.group(3))
            if dst in known and dst != src:
                edges.add((src, dst, "명시참조"))

        if _PENALTY_TITLE.search(str(r.get("clause_title") or "")):
            for m in _INTERNAL.finditer(text):
                dst = _cid(law, m.group(1), m.group(2))
                if dst in known and dst != src:
                    edges.add((src, dst, "벌칙"))
    return sorted(edges)


class EdgeIndex:
    """양방향 조회: companions(cid) = 이 조문을 참조하는 조문 + 이 조문이 참조하는 조문."""

    def __init__(self, edges: Iterable[tuple[str, str, str]]) -> None:
        self._fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._rev: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for src, dst, kind in edges:
            self._fwd[src].append((dst, kind))
            self._rev[dst].append((src, kind))

    def companions(self, cid: str, *, cap: int = 6) -> list[tuple[str, str]]:
        """(cid, kind) 목록 — 위임 짝 우선, cap개 상한 (폭주 방지).

        cap을 넉넉히 두는 이유: 어느 동반 조문이 질문에 유효한지는 여기서 정하지
        않고 다운스트림 CE 리랭크가 자른다 (관련성 판정은 리랭커 담당 원칙).
        """
        out = self._fwd.get(cid, []) + self._rev.get(cid, [])
        out = sorted(set(out), key=lambda x: (x[1] != "위임", x[0]))
        return out[:cap]
