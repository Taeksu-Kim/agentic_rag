"""라이브 HITL 시나리오를 실제 브라우저로 구동해 스크린샷 + 영상(mp4/GIF) 캡처.

시나리오(연차): ① 복합 질문(검색 STEP 트레이스) → ② "내 연차 계산"(입력 폼 HITL)
→ ③ "신청"(승인 게이트 HITL) → 접수. 배속 없음(실시간).

전제: `ui/app.py`가 127.0.0.1:7860에 떠 있고 백엔드(serve.sh) 가동.
출력: docs/media/ (hitl_*.png, demo.gif, demo.mp4).

    PYTHONPATH=. python scripts/capture_demo.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "docs" / "media"
URL = "http://127.0.0.1:7860/"


async def _shot(page, name):
    await page.wait_for_timeout(700)
    await page.screenshot(path=str(MEDIA / name))
    print("captured", name)


async def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, channel="chrome")
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 940}, device_scale_factor=2,
            record_video_dir=str(MEDIA), record_video_size={"width": 1280, "height": 940})
        page = await ctx.new_page()
        await page.goto(URL, wait_until="domcontentloaded")
        box = page.locator("#mainbox textarea")
        await box.wait_for(state="visible", timeout=30000)
        await page.wait_for_timeout(1000)

        async def type_send(text):  # 천천히 타이핑(입력이 실시간으로 보이게)
            await box.click()
            await box.press_sequentially(text, delay=45)
            await page.wait_for_timeout(700)
            await box.press("Enter")

        async def wait_done():   # 처리 완료 = 입력창 재활성(제너릭, 출력 문자열 무관)
            await page.wait_for_function(
                "() => { const t=document.querySelector('#mainbox textarea');"
                "return t && !t.disabled; }", timeout=260000)

        async def scroll_top(text):   # STEP 트레이스 등 특정 구조 마커를 상단으로(스크린샷 프레이밍용)
            await page.evaluate("""(t) => {
              const el=[...document.querySelectorAll('*')].find(e=>e.children.length===0
                && e.textContent && e.textContent.includes(t));
              if(el){const box=el.closest('.message-row,[class*=message],.bubble')||el;
                     box.scrollIntoView({block:'start',behavior:'instant'});}
            }""", text)

        async def scroll_bottom():   # 챗봇을 맨 아래로 — 최신 결과 노출(UI autoscroll 보조, 제너릭)
            await page.evaluate("""() => {
              const els=[...document.querySelectorAll('*')].filter(e=>e.scrollHeight>e.clientHeight+40
                && getComputedStyle(e).overflowY!=='visible');
              let best=null; for(const e of els){ if(!best||e.clientHeight>best.clientHeight) best=e; }
              if(best) best.scrollTop=best.scrollHeight;
            }""")

        # ① 복합 질문 -> 검색 STEP 트레이스 → 답변 스트리밍 → 조문 전문
        await type_send("연차는 며칠 생기고, 육아휴직이나 출산휴가 다녀온 기간도 출근으로 쳐주는지, 안 쓰면 어떻게 되는지 알려줘")
        await wait_done()                    # 답변+조문까지 완료(입력창 재활성)
        await page.wait_for_timeout(1000)
        await scroll_top("STEP 1")           # (1) 검색 STEP 트레이스 스크린샷
        await page.wait_for_timeout(5000)
        await _shot(page, "hitl_0_search.png")
        await scroll_top("최종 답변 생성")     # (2) 생성된 답변 스크린샷
        await page.wait_for_timeout(4500)
        await _shot(page, "hitl_0b_answer.png")
        await scroll_bottom()                # 최신으로 복귀(다음 턴 autoscroll 정상화)
        await page.wait_for_timeout(2000)

        # ② 계산 요청 -> 입력 폼(HITL)
        await type_send("내 연차가 며칠인지 계산해줘")
        await page.get_by_text("입력이 필요합니다").wait_for(timeout=150000)
        await scroll_bottom()
        await page.wait_for_timeout(3000)
        await _shot(page, "hitl_1_form.png")
        fb = page.locator("#formbox textarea")
        await fb.click()
        await fb.press_sequentially("2021-03-02", delay=90)   # 입사일 천천히 입력
        await page.wait_for_timeout(1500)
        await page.get_by_role("button", name="제출").click()
        await wait_done()                    # 계산 완료(제너릭 대기)
        await scroll_bottom()                # 계산 결과가 최신 → 맨 아래로
        await page.wait_for_timeout(4500)
        await _shot(page, "hitl_2_calc.png")
        await page.wait_for_timeout(1500)

        # ③ 신청 -> 승인 게이트(HITL)
        await type_send("2026-08-01부터 3일 연차 사용 신청해줘")
        await page.get_by_text("승인해 주세요").wait_for(timeout=150000)
        await scroll_bottom()
        await page.wait_for_timeout(4000)
        await _shot(page, "hitl_3_approval.png")
        await page.get_by_role("button", name="승인").click()
        await wait_done()                    # 신청 접수 완료(제너릭 대기)
        await scroll_bottom()                # 접수 결과 최신 → 맨 아래로
        await page.wait_for_timeout(4500)
        await _shot(page, "hitl_4_receipt.png")
        await page.wait_for_timeout(2000)

        await ctx.close()
        await browser.close()
        vids = sorted(MEDIA.glob("*.webm"))
        if vids:
            webm = MEDIA / "_demo.webm"
            vids[-1].rename(webm)
            _encode(webm, MEDIA)
            webm.unlink(missing_ok=True)
            print("encoded -> demo.gif, demo.mp4")
    print(f"done -> {MEDIA}")


def _encode(webm, media) -> None:
    """webm -> 실시간(1배속) GIF(인라인 재생, 8fps/720px) + mp4(H.264, 화질용)."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        webm.rename(media / "demo.webm")
        print("ffmpeg 없음 — webm 유지")
        return
    pal, vf = str(media / "_pal.png"), "fps=8,scale=680:-1:flags=lanczos"  # GitHub 10MB 한도 여유
    subprocess.run([ff, "-y", "-i", str(webm), "-vf", f"{vf},palettegen=max_colors=96", pal],
                   capture_output=True)
    subprocess.run([ff, "-y", "-i", str(webm), "-i", pal, "-lavfi",
                    f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3", str(media / "demo.gif")],
                   capture_output=True)
    subprocess.run([ff, "-y", "-i", str(webm), "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1000:-2", "-crf", "30", str(media / "demo.mp4")], capture_output=True)
    Path(pal).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
