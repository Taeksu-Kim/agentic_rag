"""라이브 시연: 쟁점 분해(단발 복합 질문) + 멀티턴 응축 — 승리 구성으로.

승리 구성 = 컬렉션 statutes_kiwi + kiwi 형태소 BM25 + 0.6B CE 리랭커.
UI(Phase 8)의 백엔드 드라이버이기도 하다: `build_live()`가 그래프·도구를,
`answer_turn()`이 (응축->에이전트->답변) 한 턴을 캡슐화.

    PYTHONPATH=. python scripts/demo.py            # 큐레이션 시나리오 2종
    PYTHONPATH=. python scripts/demo.py --ask "질문"  # 단발
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_live(max_steps: int = 4):
    import pandas as pd
    from qdrant_client import QdrantClient

    from agent.core.llm import OpenAICompatLLM
    from retriever import config
    from retriever.agent import build_statute_agent
    from retriever.embedder import VLLMDenseEmbedder
    from retriever.kiwi_bm25 import KiwiBM25SparseEmbedder
    from retriever.reranker import VLLMReranker

    # enable_thinking=False: 데모 지연의 근본은 9B thinking(reason 스텝당 6~80s, 일부
    # 스텝은 thought 필드 안에서 폭주). 스키마 제약 하 no-think는 스텝당 ~2s로 깨끗·
    # 예측가능(실측 70s→31s, 스키마 붙이면 6.5s→2.2s). SYSTEM_GUIDE가 명시적이라
    # 쟁점분해/리라이팅 품질 유지. repetition_penalty는 잔여 반복 방어. UI/데모 전용
    # (ablation은 run_ablations가 자체 thinking-on LLM을 만들어 영향 없음).
    NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
    # max_tokens 1024: no-think action JSON은 짧다(~150자). HR 루프가 검색 관측을
    # 누적하면 입력이 6k 토큰+ 이라, 출력 상한을 1024로 낮춰야 8192 컨텍스트 안에 든다
    # (2048이면 입력 6145+출력 2048=8193 초과로 400 에러, 실측).
    llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL, timeout=300.0,
                          sampling={"temperature": 0.2, "frequency_penalty": 0.5,
                                    "repetition_penalty": 1.15, "max_tokens": 1024, **NO_THINK})
    # 합성(C안): thinking ON — 조문을 읽고 추론(예: 60조⑥ '육아휴직=출근 간주')해야
    # 답이 정확하다. no-think 합성은 60조를 검색해놓고 "명시 안 됨" 유보하는 저하 실측.
    # repetition_penalty로 thinking의 \r\r 폭주(65s 실측) 방어, max_tokens 1024 추론 여유.
    synth_llm = OpenAICompatLLM(base_url=config.LLM_URL, model=config.LLM_MODEL, timeout=180.0,
                                sampling={"temperature": 0.2, "repetition_penalty": 1.15,
                                          "max_tokens": 1024})
    laws = sorted(pd.read_parquet(ROOT / "data/corpus/labor_statutes.parquet").law_name.unique())
    sparse = KiwiBM25SparseEmbedder().load(ROOT / "data/corpus/kiwi_bm25_stats.json")
    client, dense, reranker = QdrantClient(url=config.QDRANT_URL), VLLMDenseEmbedder(), VLLMReranker()
    graph, tool = build_statute_agent(
        llm=llm, client=client, dense=dense, sparse=sparse, reranker=reranker,
        collection="statutes_kiwi", max_steps=max_steps, valid_laws=laws,
    )
    # HITL 처리 에이전트(계산·신청 툴 + ask_human + 승인 게이트). 체크포인터 필수.
    from langgraph.checkpoint.memory import InMemorySaver
    from retriever.hr_agent import build_hr_agent
    saver = InMemorySaver()
    hr_graph, hr_tool = build_hr_agent(
        llm=llm, client=client, dense=dense, sparse=sparse, reranker=reranker,
        collection="statutes_kiwi", checkpointer=saver, max_steps=4, valid_laws=laws,
    )
    return {"llm": llm, "synth_llm": synth_llm, "graph": graph, "tool": tool,
            "hr_graph": hr_graph, "hr_tool": hr_tool}


async def answer_turn(live, question: str, history=None):
    """한 대화 턴: (history 있으면) 응축 -> 에이전트 -> 답변/근거/트레이스."""
    from retriever.agent import run_statute_agent
    from retriever.contextualize import condense_question

    standalone = condense_question(live["llm"], history or [], question)
    out = await run_statute_agent(live["graph"], live["tool"], standalone,
                                  synth_llm=live.get("synth_llm", live["llm"]))
    out["standalone"] = standalone
    return out


async def astream_turn(live, question: str, history=None):
    """한 턴을 **스텝 단위로 스트리밍**하는 제너레이터 (UI 라이브 표시용).

    이벤트: {"type": "condensed"|"status"|"search"|"final", ...}
    - condensed: 멀티턴 응축 결과(원문과 다를 때만 의미)
    - search:    쟁점별 리라이팅 쿼리 + 최상위 검색 결과 (생기는 즉시)
    - final:     합성 답변 + 근거 조문
    """
    from agent.react.graph import _config
    from retriever.agent import _synthesize_answer
    from retriever.contextualize import condense_question

    llm, graph, tool = live["llm"], live["graph"], live["tool"]
    standalone = condense_question(llm, history or [], question)
    yield {"type": "condensed", "standalone": standalone, "raw": question}

    tool.reset()
    yield {"type": "status", "msg": "원 질문 검색으로 후보 풀 시딩…"}
    try:
        tool.run(query=standalone)  # seed_raw_search (하한 방어선)
    except Exception:
        pass

    yield {"type": "status", "msg": "쟁점 분해 · 법률용어 리라이팅 · 조문 검색…"}
    seen, final_state = 0, None
    inp = {"query": standalone, "history": [], "iteration": 0}
    async for state in graph.astream(inp, _config(None), stream_mode="values"):
        final_state = state
        sp = state.get("scratchpad", [])
        while seen < len(sp):
            step = sp[seen]
            seen += 1
            if step.get("tool") == "statute_search":
                obs = step.get("observation")
                top = next((o for o in obs if isinstance(o, dict) and "ref" in o), {}) if isinstance(obs, list) else {}
                yield {"type": "search", "n": sum(1 for s in sp[:seen] if s.get("tool") == "statute_search"),
                       "query": step.get("args", {}).get("query", ""),
                       "thought": step.get("thought", ""),
                       "top_ref": top.get("ref", ""), "top_score": top.get("score")}

    yield {"type": "status", "msg": "근거 조문으로 답변 작성…"}
    cids = []
    result = (final_state or {}).get("result")
    if isinstance(result, dict):
        cids = [c for c in result.get("cids", []) if isinstance(c, str)]
    evidence = tool.resolve(cids)[:5]
    seen_c = {e["cid"] for e in evidence}
    for r in tool.top_session(k=8):
        if len(evidence) >= 8:
            break
        if r["cid"] not in seen_c:
            evidence.append(r)
            seen_c.add(r["cid"])
    # C안: 답변은 항상 thinking 합성기로 생성(루프의 no-think final을 신뢰하지 않음).
    # 근거 조문을 읽고 추론한 정확한 답을 보장한다. evidence가 없을 때만 루프 final 폴백.
    synth = live.get("synth_llm", llm)
    if evidence:
        answer = _synthesize_answer(synth, standalone, evidence)
    else:
        answer = (final_state or {}).get("final", "") or ""
    yield {"type": "final", "answer": _normalize_numbers(answer), "evidence": evidence,
           "raw": question, "standalone": standalone,
           "iterations": (final_state or {}).get("iteration", 0)}


# ---- HITL 처리 에이전트 드라이버 (UI용) ----
# 한 신청이 interrupt(입력요청/승인)로 여러 UI 상호작용에 걸치므로, "실행/재개 →
# 결과 또는 다음 interrupt" 단위로 노출한다. 9B finish가 비어도 툴 출력(계산 detail·
# 접수 message)을 authoritative 결과로 쓴다.

def _hr_result(live, state) -> dict:
    """실행/재개 후 상태 -> {interrupt|None, searches, answer, done}."""
    from retriever.agent import _synthesize_answer
    scratch = state.get("scratchpad", [])
    # 검색 스텝을 쿼리 텍스트로 중복 제거(9B가 같은 쿼리를 반복 검색하는 습성 — 표시만 정리).
    searches, seen_q = [], set()
    for s in scratch:
        if s.get("tool") != "statute_search":
            continue
        q = s.get("args", {}).get("query", "")
        if q in seen_q:
            continue
        seen_q.add(q)
        obs = s.get("observation") or []
        searches.append({"n": len(searches) + 1, "query": q,
                         "top_ref": next((o["ref"] for o in obs if isinstance(o, dict) and "ref" in o), ""),
                         "top_score": next((o["score"] for o in obs if isinstance(o, dict) and "score" in o), None)})

    if "__interrupt__" in state:
        return {"interrupt": state["__interrupt__"][0].value, "searches": searches, "done": False}

    # 완료: 처리 툴 출력(권위) 우선, 없으면 검색 근거로 합성
    calc = next((s["observation"] for s in reversed(scratch)
                 if s.get("tool") == "calculate_annual_leave" and isinstance(s.get("observation"), dict)), None)
    subm = next((s["observation"] for s in reversed(scratch)
                 if s.get("tool") == "submit_leave_request" and isinstance(s.get("observation"), dict)), None)
    parts = []
    if calc or subm:  # 처리 턴: 툴 출력이 권위(9B finish 안 믿음)
        if calc and "detail" in calc:
            parts.append(f"**연차 계산**: {calc['detail']}")
        if subm and subm.get("status") == "submitted":
            parts.append(f"**신청 접수됨**: {subm['message']} (접수번호 {subm['receipt_no']})")
    else:  # 검색 턴: no-think finish 대신 thinking 합성기로 품질 확보(astream_turn과 동일)
        ev = live["hr_tool"].top_session(k=5)
        if ev:
            parts.append(_normalize_numbers(_synthesize_answer(
                live.get("synth_llm", live["llm"]), state.get("query", ""), ev)))
        elif (state.get("final") or "").strip():
            parts.append(_normalize_numbers(state["final"]))
    return {"interrupt": None, "searches": searches, "answer": "\n\n".join(p for p in parts if p),
            "evidence": live["hr_tool"].top_session(k=8), "done": True}


async def hr_start(live, question: str, thread_id: str) -> dict:
    from agent.react.graph import arun
    live["hr_tool"].reset()
    return _hr_result(live, await arun(live["hr_graph"], question, thread_id=thread_id))


async def hr_resume(live, verdict, thread_id: str) -> dict:
    from agent.react.graph import aresume
    return _hr_result(live, await aresume(live["hr_graph"], verdict, thread_id=thread_id))


def synth_stream(synth_llm, question, evidence):
    """근거 조문으로 답변을 **토큰 단위 스트리밍** 생성(제너레이터). thinking ON은
    <think> 태그 없이 추론을 평문으로 뱉어 스트리밍에 안 맞으므로 no-think로 흘린다.
    (요약형 답변이라 no-think로 충분; 근거 조문을 명시 인용하도록 프롬프트로 유도.)"""
    import json
    import requests

    SYS = ("너는 한국 노동법 상담 어시스턴트다. 주어진 근거 조문만으로 질문에 "
           "**결론부터 명확히** 답하라. 조문에 없는 내용은 지어내지 말고, 인용한 조문 "
           "번호를 함께 밝혀라. 군더더기 없이 3~5문장으로.")
    ctx = "\n\n".join(e["text"][:2000] for e in evidence[:4])
    user = f"질문: {question[:1500]}\n\n근거 조문:\n{ctx}\n\n답변:"
    payload = {"model": synth_llm.model, "stream": True, "max_tokens": 700,
               "temperature": 0.2, "repetition_penalty": 1.15,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}
    with requests.post(f"{synth_llm.base_url}/chat/completions",
                       headers={"Authorization": f"Bearer {synth_llm.api_key}"},
                       json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data.strip() == b"[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                yield delta


async def hr_stream(live, thread_id: str, *, question=None, resume=None):
    """HR 에이전트를 **스텝 단위로 스트리밍** — 질문 입력 즉시 상태가 순차로 바뀌게.

    이벤트: {'type':'status', 'msg'} / {'type':'step', tool, query, obs} /
            {'type':'interrupt', payload, searches} / {'type':'final', result}
    question이면 새 실행, resume면 interrupt 재개.
    """
    from agent.react.graph import _config
    from langgraph.types import Command

    graph, cfg = live["hr_graph"], _config(thread_id)
    if question is not None:
        live["hr_tool"].reset()
        inp = {"query": question, "history": [], "iteration": 0}
    else:
        inp = Command(resume=resume)

    yield {"type": "status", "msg": "쟁점 분해 · 법률용어 리라이팅 중…"}
    seen, last, last_planned, proc_done = 0, None, None, False
    async for state in graph.astream(inp, cfg, stream_mode="values"):
        last = state
        # reason 단계가 만든 다음 검색 쿼리(=리라이팅 결과) — 검색 실행 전에 먼저 노출
        act = state.get("action") or {}
        if act.get("tool") == "statute_search":
            q = act.get("args", {}).get("query", "")
            if q and q != last_planned:
                last_planned = q
                yield {"type": "planned", "query": q}
        # act 단계 완료 = 검색 결과
        sp = state.get("scratchpad", [])
        while seen < len(sp):
            step = sp[seen]
            seen += 1
            tool = step.get("tool")
            yield {"type": "step", "tool": tool,
                   "query": step.get("args", {}).get("query", ""),
                   "obs": step.get("observation")}
            # 처리 툴은 결과가 곧 답 — 실행되면 즉시 종료(그 뒤 finish reason 낭비 스킵)
            if tool in ("calculate_annual_leave", "submit_leave_request"):
                proc_done = True
        if proc_done:
            break

    snap = graph.get_state(cfg)
    payload = None
    for t in getattr(snap, "tasks", []) or []:
        intr = getattr(t, "interrupts", None)
        if intr:
            payload = intr[0].value
            break
    if snap.next and payload is not None:  # interrupt 대기(폼/승인)
        res = _hr_result(live, last or {})
        yield {"type": "interrupt", "payload": payload, "searches": res["searches"]}
        return

    result = _hr_result(live, last or {})
    scratch = (last or {}).get("scratchpad", [])
    is_proc = any(s.get("tool") in ("calculate_annual_leave", "submit_leave_request") for s in scratch)
    if is_proc or not result.get("evidence"):  # 처리 결과(툴 출력)는 즉시 표시
        yield {"type": "final", "result": result}
        return
    # 검색 답변은 토큰 스트리밍 생성
    yield {"type": "answer_start", "searches": result["searches"], "evidence": result["evidence"]}
    ev = live["hr_tool"].top_session(k=5)
    full = ""
    for tok in synth_stream(live.get("synth_llm", live["llm"]), (last or {}).get("query", ""), ev):
        full += tok
        yield {"type": "answer_delta", "text": tok}
    yield {"type": "answer_done", "answer": _normalize_numbers(full),
           "searches": result["searches"], "evidence": result["evidence"]}


def _normalize_numbers(text: str) -> str:
    """9B가 조문 번호에 끼워 넣는 공백 정리: '제 60 조 제 6 항' -> '제60조제6항'."""
    import re
    text = re.sub(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", lambda m: f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else ""), text)
    text = re.sub(r"제\s*(\d+)\s*(항|호|목|款)", r"제\1\2", text)
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)  # '1 0%' 류 분리 숫자 결합
    return text


def _print(out, question):
    print(f"\nQ: {question}")
    if out["standalone"] != question:
        print(f"  (응축된 독립형 질문: {out['standalone']})")
    print("-" * 68)
    searches = [s for s in out["steps"] if s.get("tool") == "statute_search"]
    for i, s in enumerate(searches, 1):
        print(f"  [검색 {i}] {s.get('args', {}).get('query', '')}")
    print(f"\nA: {out['answer']}\n")
    print("근거 조문:")
    for e in out["evidence"][:5]:
        print(f"  - {e['law_name']} 제{e['clause_no']}조({e['clause_title']})  score={e['score']:.3f}")
    print()


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", help="단발 질문")
    args = ap.parse_args()
    live = build_live()

    if args.ask:
        _print(await answer_turn(live, args.ask), args.ask)
        return

    print("\n========== 시나리오 1: 복합 질문 (쟁점 분해) ==========")
    q1 = "육아휴직 중에도 연차휴가가 발생하는지, 그리고 육아휴직 기간이 퇴직금 산정에 포함되는지 알려줘"
    _print(await answer_turn(live, q1), q1)

    print("\n========== 시나리오 2: 멀티턴 (후속 질문 응축) ==========")
    history = []
    t1 = "수습기간에도 최저임금을 다 줘야 하나요?"
    o1 = await answer_turn(live, t1, history)
    _print(o1, t1)
    history.append((t1, o1["answer"]))
    t2 = "그럼 그 기간에 해고하면 예고수당은요?"  # 대명사·생략 -> 응축 필요
    _print(await answer_turn(live, t2, history), t2)


if __name__ == "__main__":
    asyncio.run(amain())
