"""StatuteSearchTool: 에이전트가 부르는 조문 검색 툴 (2단계 검색 래핑 + 세션 누적).

설계 (evidence_pack 툴을 대체 — docs/design_and_plan.md §9):
* ``run()``은 LLM에게 **압축 뷰**(cid/ref/snippet/score)만 돌려준다 — 전문을
  scratchpad에 넣으면 컨텍스트가 폭발한다.
* 전체 hit는 ``session``에 누적. 에이전트가 finish에서 고른 cid들을
  ``resolve()``가 세션에서 **코드로** 전문 해소 — LLM이 조문 텍스트를 다시
  타이핑하지 않으므로 환각이 구조적으로 차단된다 (모르는 cid는 버림).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from qdrant_client import QdrantClient

from agent.core.tools import BaseTool
from retriever.embedder import DenseEmbedder, SparseEmbedder
from retriever.reranker import Reranker
from retriever.search import build_filter, search_statutes

SNIPPET_LEN = 160
MAX_K = 10


class StatuteSearchTool(BaseTool):
    name = "statute_search"
    description = (
        "노동법 조문(법률/시행령/시행규칙)을 검색한다. query는 조문이 쓸 법한 "
        "법률 용어로 쓸 것. law_names로 특정 법령만 좁힐 수 있다."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer"},
            "law_names": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }

    def __init__(self, *, client: QdrantClient, collection: str,
                 dense: DenseEmbedder, sparse: SparseEmbedder,
                 reranker: Optional[Reranker] = None,
                 default_k: int = 8, prefetch_limit: int = 30,
                 valid_laws: Optional[Sequence[str]] = None) -> None:
        self._client = client
        self._collection = collection
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._default_k = default_k
        self._prefetch_limit = prefetch_limit
        self.session: dict[str, dict[str, Any]] = {}  # cid -> full record (best score)
        # 법령명은 닫힌 소규모 집합(노동 8종 x 법/령/규칙 = 24) — 목록을 LLM에게
        # 보여주면 필터 오타가 사라지고 오특정도 3% 수준으로 떨어진다 (핀포인트 실측:
        # 목록 제공 시 유효 표기 29/30, R@8 +6.3pp). 공백 제거 매칭으로 잔여 오타 흡수.
        self._valid_laws = list(valid_laws) if valid_laws else None
        self._law_norm = ({n.replace(" ", ""): n for n in self._valid_laws}
                          if self._valid_laws else {})
        if self._valid_laws:
            self.description = (
                "노동법 조문(법률/시행령/시행규칙)을 검색한다. query는 조문이 쓸 법한 "
                "법률 용어로 쓸 것. law_names로 법령을 좁힐 수 있고 **여러 법령을 한 번에** "
                "넣어도 된다(예: 법과 그 시행령). 유효한 법령명(이 표기 그대로만): "
                + ", ".join(self._valid_laws)
            )

    def _normalize_laws(self, law_names: Optional[Sequence[str]]) -> Optional[list[str]]:
        """공백 변형("산업안전보건법시행령")을 정확 표기로 교정. 미지의 이름은 유지
        (-> 0건 -> 필터 해제 폴백이 흡수)."""
        if not law_names:
            return None
        return [self._law_norm.get(str(n).replace(" ", ""), str(n)) for n in law_names]

    def run(self, query: str, k: int | None = None,
            law_names: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
        k = min(k or self._default_k, MAX_K)  # LLM이 k를 크게 불러도 상한
        law_names = self._normalize_laws(law_names)

        def _search(names):
            return search_statutes(
                self._client, self._collection,
                dense_vec=self._dense.encode([query])[0],
                sparse_vec=self._sparse.encode_query([query])[0],
                k=k, prefetch_limit=self._prefetch_limit,
                flt=build_filter(law_names=names),
                reranker=self._reranker, query_text=query if self._reranker else None,
            )

        hits = _search(law_names)
        filter_dropped = False
        if not hits and law_names:
            # LLM이 존재하지 않는 법령명 표기로 필터를 걸면 0건이 된다 (ablation 실측:
            # max_steps=2에서 빈손 17%의 주범). 코드가 필터를 해제하고 재검색해 준다.
            hits = _search(None)
            filter_dropped = True
        out: list[dict[str, Any]] = []
        for h in hits:
            p = h.payload
            rec = {"cid": p["cid"], "law_name": p["law_name"], "law_type": p["law_type"],
                   "clause_no": p["clause_no"], "clause_title": p["clause_title"],
                   "text": p["text"], "score": float(h.score)}
            prev = self.session.get(p["cid"])
            if prev is None or rec["score"] > prev["score"]:
                self.session[p["cid"]] = rec
            body = p["text"].split("\n", 1)[-1]
            out.append({
                "cid": p["cid"],
                "ref": f"{p['law_name']} 제{p['clause_no']}조({p['clause_title']})",
                "snippet": body[:SNIPPET_LEN],
                "score": round(float(h.score), 4),
            })
        if filter_dropped:
            out.insert(0, {"note": (
                f"law_names={list(law_names)} 필터와 일치하는 조문이 0건이라 필터를 "
                "해제하고 재검색했다. 필터 값은 검색 결과의 law_name 표기 그대로만 유효하다.")})
        return out

    # -- evidence resolution (agent wrapper uses these; not exposed to the LLM) --

    def resolve(self, cids: Sequence[str]) -> list[dict[str, Any]]:
        """finish에서 고른 cid들 -> 세션의 전문 레코드 (모르는 cid는 버림)."""
        return [self.session[c] for c in cids if c in self.session]

    def top_session(self, k: int = 5) -> list[dict[str, Any]]:
        """cid 미지정 시 폴백: 세션 최고점 순."""
        return sorted(self.session.values(), key=lambda r: r["score"], reverse=True)[:k]

    def reset(self) -> None:
        self.session = {}
