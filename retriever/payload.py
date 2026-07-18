"""Qdrant payload + point id.

``cid``("법령명|조문번호")는 qrels 라벨과 같은 표기라 평가에서 그대로 조인된다.
Point id는 cid의 uuid5 — 재실행해도 같은 조문은 같은 포인트(멱등 upsert).
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping

from retriever.text import build_embedding_text

_NS = uuid.uuid5(uuid.NAMESPACE_URL, "agentic-rag/statutes")


def point_id(cid: str) -> str:
    return str(uuid.uuid5(_NS, cid))


def build_payload(clause: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cid": f"{clause['law_name']}|{clause['clause_no']}",
        "law_name": clause["law_name"],
        "law_type": clause["law_type"],
        "clause_no": str(clause["clause_no"]),
        "clause_title": clause.get("clause_title") or "",
        "effective_date": clause.get("effective_date") or "",
        "text": build_embedding_text(clause),
    }
