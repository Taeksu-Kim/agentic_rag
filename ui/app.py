"""Gradio 데모 UI — 대화형 노동법 리트리버 에이전트 (승리 구성).

한 턴 = (멀티턴 응축 -> 쟁점 분해 react 루프 -> 근거 조문 합성 답변). 에이전트의
검색 쟁점 트레이스를 접이식 'thought'로, 근거 조문을 점수와 함께 보여 준다.

    PYTHONPATH=. python ui/app.py            # 로컬 (loopback 깨진 WSL은 아래)
    GRADIO_SERVER_NAME=0.0.0.0 PYTHONPATH=. python ui/app.py

백엔드(9B/임베더/리랭커/Qdrant)가 떠 있어야 한다: `bash scripts/serve.sh start`.
"""

from __future__ import annotations

import asyncio
import re
import uuid

import gradio as gr

from scripts.demo import build_live, hr_stream

_LIVE = None


def _live():
    global _LIVE
    if _LIVE is None:
        _LIVE = build_live()
    return _LIVE


def _clause_body(e, limit: int = 380) -> str:
    """조문 전문에서 본문(①②③…)만 추출 — 헤더(법령명 제N조)·중복 제목 라인 제거."""
    parts = e.get("text", "").split("\n", 1)
    body = parts[1] if len(parts) > 1 else parts[0]
    lines = body.split("\n")
    if lines and re.match(r"^제\s*\d+\s*조", lines[0].strip()):  # 중복 '제N조(제목)' 라인 제거
        lines = lines[1:]
    body = "\n".join(l for l in lines if l.strip()).strip()
    if len(body) > limit:
        body = body[:limit].rstrip() + " …"
    return body


def _trace_md(raw, condensed, searches, status=None) -> str:
    """각 검색 반복(쟁점)을 **독립 STEP**으로 — STEP N = (N-1 리라이팅 → N-2 검색·결과).
    STEP 1 사이클이 끝나면 STEP 2가 뒤이어 나오는 멀티스텝 루프를 보여준다. 이모지 없음."""
    lines = [f"**질문**: {raw}"]
    if condensed:
        lines.append(f"_맥락 반영: {condensed}_")
    for s in searches:
        n = s["n"]
        blk = [f"**STEP {n} · 쟁점 검색**",
               f"　{n}-1  법률용어 리라이팅: **「{s['query']}」**"]
        if s.get("searching"):
            blk.append(f"　{n}-2  하이브리드 검색 → _검색 중…_")
        else:
            score = f"　·　score {s['top_score']:.3f}" if s.get("top_score") is not None else ""
            blk.append(f"　{n}-2  하이브리드 검색 → **{s['top_ref'] or '(결과 없음)'}**{score}")
        lines.append("\n".join(blk))
    last = f"**STEP {len(searches) + 1} · 근거 조문으로 최종 답변 생성**"
    if status:
        last += f"\n　*…{status}*"
    lines.append(last)
    return "\n\n".join(lines)


def _evidence_md(evidence) -> str:
    """근거 조문을 **실제 전문**으로 — 상위 2건 본문 인용, 나머지는 목록. 이모지 없음."""
    if not evidence:
        return "**③ 근거 조문**\n\n_해당 조문 없음_"
    out = ["**③ 근거 조문 (실제 전문 인용)**"]
    for e in evidence[:2]:
        head = f"{e['law_name']} 제{e['clause_no']}조({e['clause_title']})"
        body = _clause_body(e)
        quoted = "\n".join(f"> {ln}" for ln in body.split("\n"))
        out.append(f"**{head}** · score {e['score']:.3f}\n{quoted}")
    rest = evidence[2:8]
    if rest:
        refs = ", ".join(f"{e['law_name']} 제{e['clause_no']}조({e['score']:.2f})" for e in rest)
        out.append(f"_그 외 참고: {refs}_")
    return "\n\n".join(out)


PLACEHOLDER = "예: 연차는 어떻게 생기고 육아휴직 기간도 출근으로 쳐주나요?  (계산·신청도: '내 연차 계산해줘')"

EXAMPLES = [
    "연차는 며칠 생기고, 육아휴직이나 출산휴가 다녀온 기간도 출근으로 쳐주는지, 안 쓰면 어떻게 되는지 알려줘",
    "내 연차가 며칠인지 계산해줘",
    "부당하게 해고당한 것 같은데 어떻게 구제받고, 해고예고수당은 받을 수 있나요?",
    "임신 중인데 야근이나 주말근무를 시켜도 되나요?",
]


def _trace_msg(raw, searches):
    return gr.ChatMessage(role="assistant", content=_trace_md(raw, "", searches))


def _apply(result, raw, chat, state):
    """HR 결과(interrupt|done) -> (chatbot, state, form_grp, form_prompt, form_input, appr_grp, appr_prompt, msg)."""
    chat = list(chat)
    if result["searches"]:
        chat.append(_trace_msg(raw, result["searches"]))

    if result["interrupt"]:
        p = result["interrupt"]
        if p.get("type") == "question":
            q = str(p.get("question", "")).strip().strip("'\"")
            return (chat, {**state, "awaiting": "input"},
                    gr.update(visible=True), gr.update(value=f"**입력이 필요합니다**\n\n{q}"),
                    gr.update(value=""), gr.update(visible=False), gr.update(),
                    gr.update(interactive=False, placeholder="↓ 아래 폼에 입력하세요"))
        args = p.get("args", {})  # tool_approval
        summary = (f"**신청 내용을 확인하고 승인해 주세요**\n\n"
                   f"- 연차 사용 시작일: **{args.get('start_date')}**\n"
                   f"- 사용 일수: **{args.get('days')}일**")
        return (chat, {**state, "awaiting": "approval"},
                gr.update(visible=False), gr.update(), gr.update(),
                gr.update(visible=True), gr.update(value=summary),
                gr.update(interactive=False, placeholder="위에서 승인/취소를 선택하세요"))

    # done
    chat.append(gr.ChatMessage(role="assistant", content=result.get("answer") or "_결과 없음_"))
    if result.get("evidence") and result.get("searches"):
        chat.append(gr.ChatMessage(role="assistant", content=_evidence_md(result["evidence"])))
    return (chat, {**state, "awaiting": None},
            gr.update(visible=False), gr.update(), gr.update(),
            gr.update(visible=False), gr.update(),
            gr.update(interactive=True, placeholder=PLACEHOLDER))


def _live_out(chat, raw, searches, status, st):
    """처리 중 라이브 표시: 검색 턴이면 STEP 트레이스+상태, 아니면 상태만. 입력창 잠금."""
    content = _trace_md(raw, "", searches, status) if searches else f"**처리 중**\n\n_…{status}_"
    live = gr.ChatMessage(role="assistant", content=content)
    return (chat + [live], st,
            gr.update(visible=False), gr.update(), gr.update(),
            gr.update(visible=False), gr.update(),
            gr.update(value="", interactive=False, placeholder="처리 중…"))


def _drive(make_agen, raw, chat, st):
    """hr_stream을 소비하며 UI를 순차 갱신하는 제너레이터. 마지막은 _apply(폼/승인/답변)."""
    loop = asyncio.new_event_loop()
    agen = make_agen()
    cycles, status = [], "시작하는 중…"   # cycles: 쿼리별 {n, query, top_ref, top_score, searching}
    yield _live_out(chat, raw, cycles, status, st)   # 질문 직후 즉시 상태 표시
    try:
        while True:
            try:
                ev = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            t = ev["type"]
            if t == "status":
                status = ev["msg"]
                yield _live_out(chat, raw, cycles, status, st)
            elif t == "planned":  # 리라이팅된 쿼리 노출(검색 중) — 사이클 시작
                if not any(c["query"] == ev["query"] for c in cycles):
                    cycles.append({"n": len(cycles) + 1, "query": ev["query"],
                                   "top_ref": "", "top_score": None, "searching": True})
                    status = "조문 검색 중…"
                    yield _live_out(chat, raw, cycles, status, st)
            elif t == "step" and ev.get("tool") == "statute_search":  # 검색 결과 — 사이클 완료
                obs = ev.get("obs") or []
                top = next((o for o in obs if isinstance(o, dict) and "ref" in o), {})
                c = next((c for c in cycles if c["query"] == ev["query"] and c["searching"]), None)
                if c:
                    c.update(top_ref=top.get("ref", ""), top_score=top.get("score"), searching=False)
                elif not any(c["query"] == ev["query"] for c in cycles):
                    cycles.append({"n": len(cycles) + 1, "query": ev["query"],
                                   "top_ref": top.get("ref", ""), "top_score": top.get("score"),
                                   "searching": False})
                status = "다음 쟁점 리라이팅 중…"
                yield _live_out(chat, raw, cycles, status, st)
            elif t == "answer_start":  # 검색 답변 토큰 스트리밍 시작
                trace = gr.ChatMessage(role="assistant", content=_trace_md(raw, "", ev["searches"]))
                ans_text = ""
                yield (chat + [trace, gr.ChatMessage(role="assistant", content="_답변 생성 중…_")],
                       st, gr.update(visible=False), gr.update(), gr.update(),
                       gr.update(visible=False), gr.update(),
                       gr.update(value="", interactive=False, placeholder="답변 생성 중…"))
            elif t == "answer_delta":
                ans_text += ev["text"]
                yield (chat + [trace, gr.ChatMessage(role="assistant", content=ans_text)],
                       st, gr.update(visible=False), gr.update(), gr.update(),
                       gr.update(visible=False), gr.update(), gr.update())
            elif t == "answer_done":
                final = [trace, gr.ChatMessage(role="assistant", content=ev["answer"] or "_결과 없음_")]
                if ev.get("evidence"):
                    final.append(gr.ChatMessage(role="assistant", content=_evidence_md(ev["evidence"])))
                yield (chat + final, {**st, "awaiting": None},
                       gr.update(visible=False), gr.update(), gr.update(),
                       gr.update(visible=False), gr.update(),
                       gr.update(interactive=True, placeholder=PLACEHOLDER))
                return
            elif t == "interrupt":
                yield _apply({"interrupt": ev["payload"], "searches": ev["searches"]}, raw, chat, st)
                return
            elif t == "final":
                yield _apply(ev["result"], raw, chat, st)
                return
    finally:
        loop.close()


def build_ui():
    with gr.Blocks(title="노동법 HR 에이전트") as demo:
        gr.Markdown(
            "# 노동법 HR 에이전트\n"
            "일상어 질문을 **쟁점 분해 → 법률용어 리라이팅 → 하이브리드 검색 + CE 리랭크**로 풀어 "
            "근거 조문을 인용해 답하고, **연차 계산·신청 같은 처리 업무**는 도구를 호출해 수행합니다"
            "(신청은 **사람 승인 후** 제출 — HITL). _근거 조문에 없으면 지어내지 않습니다._"
        )
        state = gr.State({"awaiting": None, "thread_id": None, "q": ""})
        chatbot = gr.Chatbot(height=520, show_label=False, autoscroll=True)

        with gr.Group(visible=False) as form_grp:   # ask_human 입력 폼
            form_prompt = gr.Markdown()
            with gr.Row():
                form_input = gr.Textbox(show_label=False, scale=4, elem_id="formbox",
                                        placeholder="예: 2021-03-02  또는  2026-08-01, 3일")
                form_submit = gr.Button("제출", variant="primary", scale=1)

        with gr.Group(visible=False) as appr_grp:    # 신청 승인 게이트
            appr_prompt = gr.Markdown()
            with gr.Row():
                approve = gr.Button("승인 · 제출", variant="primary")
                reject = gr.Button("취소")

        msg = gr.Textbox(placeholder=PLACEHOLDER, show_label=False, autofocus=True, elem_id="mainbox")
        with gr.Row():
            send = gr.Button("보내기", variant="primary")
            clear = gr.Button("대화 초기화")
        gr.Examples(EXAMPLES, inputs=msg)

        OUT = [chatbot, state, form_grp, form_prompt, form_input, appr_grp, appr_prompt, msg]

        def on_send(message, chat, st):
            message = (message or "").strip()
            if not message:
                yield (chat, st, *([gr.update()] * 6))
                return
            chat = list(chat) + [gr.ChatMessage(role="user", content=message)]  # 질문 즉시 표시
            tid = uuid.uuid4().hex[:8]
            st = {**st, "thread_id": tid, "q": message}
            yield from _drive(lambda: hr_stream(_live(), tid, question=message), message, chat, st)

        def on_form(value, chat, st):
            yield from _drive(lambda: hr_stream(_live(), st["thread_id"], resume=(value or "").strip()),
                              st.get("q", ""), chat, st)

        def on_approve(chat, st):  # 제너레이터 함수여야 Gradio가 스트리밍으로 인식(lambda 불가)
            yield from _drive(lambda: hr_stream(_live(), st["thread_id"], resume=True),
                              st.get("q", ""), chat, st)

        def on_reject(chat, st):
            yield from _drive(lambda: hr_stream(_live(), st["thread_id"],
                                                resume={"feedback": "사용자가 신청을 취소함"}),
                              st.get("q", ""), chat, st)

        send.click(on_send, [msg, chatbot, state], OUT)
        msg.submit(on_send, [msg, chatbot, state], OUT)
        form_submit.click(on_form, [form_input, chatbot, state], OUT)
        form_input.submit(on_form, [form_input, chatbot, state], OUT)
        approve.click(on_approve, [chatbot, state], OUT)
        reject.click(on_reject, [chatbot, state], OUT)
        clear.click(lambda: ([], {"awaiting": None, "thread_id": None, "q": ""},
                             gr.update(visible=False), gr.update(), gr.update(),
                             gr.update(visible=False), gr.update(),
                             gr.update(interactive=True, placeholder=PLACEHOLDER)),
                    None, OUT, queue=False)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(theme=gr.themes.Soft())
