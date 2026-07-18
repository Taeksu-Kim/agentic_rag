"""kiwi 형태소 BM25 sparse 임베더 — fastembed BM25의 한국어 대체.

동기(실측): 기본 BM25는 어절 통짜 토큰이라 "휴가를 != 휴가가" — sparse 축이
사실상 죽어 있었다(R@8 0.238, 미스에서 짧은 절차성 조문 4배 농축인데 렉시컬
구제 채널 부재). kiwi로 조사·어미를 벗겨 내용 형태소(NNG/NNP/NNB/SN/SL/XR)만
남기면 BM25의 전제(용어=의미 단위, IDF 유효)가 성립한다. 조문 번호도 보존
("제17조" -> 17/SN + 조/NNB).

설계: fastembed BM25와 같은 분해 — 문서측 = tf 정규화 가중치, 쿼리측 = IDF.
Qdrant 내적이 곧 BM25 점수. IDF/avgdl은 코퍼스로 fit하고 JSON으로 저장/로드
(쿼리 시점에 같은 통계를 써야 한다). 토크나이저는 주입 가능(테스트는 fake).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

from qdrant_client import models

KEEP_TAGS = {"NNG", "NNP", "NNB", "SN", "SL", "XR"}


def _default_tokenizer():
    from kiwipiepy import Kiwi  # lazy — 테스트는 fake 주입

    kiwi = Kiwi()

    def tokenize(text: str) -> list[str]:
        return [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]

    return tokenize


def _tok_index(tok: str) -> int:
    return int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


class KiwiBM25SparseEmbedder:
    """SparseEmbedder 프로토콜 구현 (encode=문서측, encode_query=IDF측)."""

    def __init__(self, tokenizer: Optional[Callable[[str], list[str]]] = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self._tokenize = tokenizer or _default_tokenizer()
        self.k1 = k1
        self.b = b
        self.idf: dict[str, float] = {}
        self.avgdl: float = 1.0

    # -- 코퍼스 통계 --

    def fit(self, corpus_texts: Sequence[str]) -> "KiwiBM25SparseEmbedder":
        docs = [self._tokenize(t) for t in corpus_texts]
        n = len(docs)
        df: dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
        self.avgdl = (sum(len(d) for d in docs) / n) if n else 1.0
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"idf": self.idf, "avgdl": self.avgdl},
                                         ensure_ascii=False))

    def load(self, path: str | Path) -> "KiwiBM25SparseEmbedder":
        d = json.loads(Path(path).read_text())
        self.idf, self.avgdl = d["idf"], d["avgdl"]
        return self

    # -- SparseEmbedder 프로토콜 --

    def encode(self, texts: Sequence[str]) -> list[models.SparseVector]:
        out = []
        for text in texts:
            toks = self._tokenize(text)
            dl = len(toks) or 1
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            idx, val = [], []
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            for t, f in tf.items():
                idx.append(_tok_index(t))
                val.append(f * (self.k1 + 1) / (f + norm))
            out.append(models.SparseVector(indices=idx or [0], values=val or [0.0]))
        return out

    def encode_query(self, texts: Sequence[str]) -> list[models.SparseVector]:
        out = []
        for text in texts:
            seen: dict[int, float] = {}
            for t in set(self._tokenize(text)):
                w = self.idf.get(t)
                if w:  # 코퍼스 미등장 토큰은 매칭 불가 -> 제외
                    seen[_tok_index(t)] = w
            out.append(models.SparseVector(indices=list(seen) or [0],
                                           values=list(seen.values()) or [0.0]))
        return out
