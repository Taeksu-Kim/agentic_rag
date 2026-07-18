"""해석례/질의회시 텍스트에서 조문 인용을 추출해 (법령명, 조문번호)로 정규화.

qrels 자동 구축의 심장부. 정확도 우선(정밀도 > 재현율) — 어느 법령인지 확정할 수
없는 참조("같은 법"의 선행 법령이 코퍼스 밖인 경우, 법령 컨텍스트 없는 맨 제N조)는
라벨로 만들지 않고 버린다.

처리하는 패턴:
* ``「근로기준법」 제60조제2항``            -> ("근로기준법", "60")   (항은 무시, 조 단위)
* ``근로기준법 제10조제2항`` (괄호 없음)     -> 질의회시 회답의 흔한 표기 — known 법령명 직접 매칭
* ``제74조의2``                            -> 가지조문 "74-2" (코퍼스 clause_no 표기)
* ``같은 법 / 동법 (시행령|시행규칙)?``     -> 직전 법령 컨텍스트로 해소
* ``(이하 "기간제법"이라 함)``              -> 본문 내 약칭 정의를 알아내 별칭 매칭
* ``부칙 제N조``                           -> 본문 조문이 아님 — 다음 법령 언급까지 버림
* 컨텍스트 이후의 맨 ``제N조`` 나열          -> 컨텍스트 법령에 귀속
"""

from __future__ import annotations

import re
from typing import Optional

# 가운뎃점 이형 통일 (남녀고용평등과 일ㆍ가정 ...)
_DOTS = {"·": "ㆍ", "・": "ㆍ", "‧": "ㆍ"}

_ALIAS_RE = re.compile(
    r"「(?P<full>[^」]+)」\s*\((?:이하\s*)?[“”\"']?(?P<alias>[^“”\"'()]+?)[“”\"']?\s*(?:이?라\s*(?:함|한다))\.?\)"
)

_LAW = r"「(?P<law>[^」]+)」"
_REL = r"(?P<rel>(?:같은\s*|동\s?)법)(?P<rel_suffix>\s*시행(?:령|규칙))?"
_BUCHIK = r"(?P<buchik>부\s?칙)"
# 괄호 없이 인용된 '모르는' 법령: ...법/법률(시행령|시행규칙)? 바로 뒤에 제N조가
# 따라오는 경우만 법령 토큰으로 본다 (컨텍스트를 가로채 known 법령에 오귀속되는 것 방지)
_ULAW = r"(?P<ulaw>[가-힣ㆍ]{1,40}?(?:법률|법)(?:\s*시행(?:령|규칙))?)(?=\s*제\d+조)"
_ART = r"제(?P<jo>\d+)조(?:의(?P<ui>\d+))?"


def _norm(s: str) -> str:
    for a, b in _DOTS.items():
        s = s.replace(a, b)
    return s.strip()


def extract_citations(text: str, known_laws: list[str]) -> list[tuple[str, str]]:
    """텍스트에서 코퍼스 내 법령의 조문 인용을 (법령명, 조문번호)로 추출.

    known_laws에 없는 법령의 인용(및 그 법령을 컨텍스트로 갖는 상대 참조)은 버린다.
    반환은 등장 순서 유지 + 중복 제거.
    """
    text = _norm(text)
    known = {_norm(k) for k in known_laws}

    # 본문에 정의된 약칭 -> 정식 명칭
    aliases: dict[str, str] = {}
    for m in _ALIAS_RE.finditer(text):
        full = _norm(m.group("full"))
        if full in known:
            aliases[_norm(m.group("alias"))] = full

    # known 법령명은 괄호 없이도 법령 토큰 (긴 이름 먼저 — "X 시행령"이 "X"보다 우선)
    kname = "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True))
    parts = [_LAW, f"(?P<kname>{kname})", _REL, _BUCHIK, _ULAW, _ART]
    if aliases:  # 약칭 토큰 (긴 것 먼저)
        alt = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
        parts.insert(2, f"(?P<alias>{alt})")
    token_re = re.compile("|".join(parts))

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    context: Optional[str] = None   # 마지막으로 언급된 법령 (모르는 법령도 추적)
    context_known = False

    for m in token_re.finditer(text):
        if m.group("law") is not None:
            context = _norm(m.group("law"))
            context_known = context in known
        elif m.group("kname") is not None:
            context = _norm(m.group("kname"))
            context_known = True
        elif aliases and m.groupdict().get("alias") is not None:
            context = aliases[_norm(m.group("alias"))]
            context_known = True
        elif m.group("rel") is not None:
            if context is None:
                context_known = False
                continue
            # "같은 법" 기준은 모법: 컨텍스트가 시행령/규칙이면 접미어를 벗긴다
            base = re.sub(r"\s*시행(?:령|규칙)$", "", context)
            suffix = (m.group("rel_suffix") or "").strip()
            context = f"{base} {suffix}".strip() if suffix else base
            context_known = context in known
        elif m.group("buchik") is not None:
            # 부칙 조문 번호는 본문과 다른 체계 — 다음 법령 토큰까지 라벨 생성 중단
            context_known = False
        elif m.group("ulaw") is not None:
            # 코퍼스 밖 법령이 컨텍스트가 됨 — 이후 조문은 known 아님
            context = _norm(m.group("ulaw"))
            context_known = False
        else:  # article
            if not context_known or context is None:
                continue
            jo, ui = m.group("jo"), m.group("ui")
            clause = f"{jo}-{ui}" if ui else jo
            key = (context, clause)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out
