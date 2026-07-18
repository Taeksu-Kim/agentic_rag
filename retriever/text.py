"""조문 -> 임베딩 텍스트: "{법령명} 제{N}조[의{M}]({제목})\n{본문}"."""

from __future__ import annotations

from typing import Any, Mapping


def _article_label(clause_no: str) -> str:
    no = str(clause_no)
    if "-" in no:  # 가지조문: "74-2" -> 제74조의2
        base, branch = no.split("-", 1)
        return f"제{base}조의{branch}"
    return f"제{no}조"


def build_embedding_text(clause: Mapping[str, Any]) -> str:
    header = f"{clause['law_name']} {_article_label(clause['clause_no'])}"
    title = clause.get("clause_title") or ""
    if title:
        header += f"({title})"
    return f"{header}\n{clause['clause_content']}"
