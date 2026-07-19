"""조문 -> 임베딩 텍스트: "{법령명} 제{N}조[의{M}]({제목})\n{본문}[\n예상 질문...]".

``questions``(doc2query 역질문)는 **인덱스 텍스트 전용** — payload ``text``는
질문 없이 만든다 (리랭커/에이전트가 읽는 본문은 원문 그대로).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def _article_label(clause_no: str) -> str:
    no = str(clause_no)
    if "-" in no:  # 가지조문: "74-2" -> 제74조의2
        base, branch = no.split("-", 1)
        return f"제{base}조의{branch}"
    return f"제{no}조"


def build_embedding_text(clause: Mapping[str, Any],
                         questions: Optional[Sequence[str]] = None) -> str:
    header = f"{clause['law_name']} {_article_label(clause['clause_no'])}"
    title = clause.get("clause_title") or ""
    if title:
        header += f"({title})"
    text = f"{header}\n{clause['clause_content']}"
    if questions:
        text += "\n예상 질문:\n" + "\n".join(f"- {q}" for q in questions)
    return text
