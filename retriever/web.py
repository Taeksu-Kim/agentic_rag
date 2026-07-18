"""WebSearchTool: 백엔드 교체형 웹 검색 (기본 ddgs — 키 불필요).

법령 코퍼스 밖 보조 정보(최신 개정 소식 등)용. 백엔드는 프로토콜이라
Tavily/Google로 갈아끼울 수 있다. 테스트는 FakeSearchBackend.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent.core.tools import BaseTool


@runtime_checkable
class SearchBackend(Protocol):
    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """[{title, url, snippet}] 반환."""
        ...


class FakeSearchBackend:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.results[:k]


class DdgsBackend:
    """DuckDuckGo (ddgs 라이브러리, 무료/키 없음). lazy import."""

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        from ddgs import DDGS  # lazy

        with DDGS() as ddgs:
            rows = ddgs.text(query, max_results=k)
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": r.get("body", "")} for r in (rows or [])]


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "웹 검색. 조문 밖 보조 정보(최신 개정, 시행 시기, 일반 해설)가 필요할 때만 "
        "사용. 법 조문 자체는 statute_search로 찾을 것."
    )
    args_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
    }

    def __init__(self, backend: SearchBackend | None = None, default_k: int = 5) -> None:
        self._backend = backend or DdgsBackend()
        self._default_k = default_k

    def run(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        return self._backend.search(query, k or self._default_k)
