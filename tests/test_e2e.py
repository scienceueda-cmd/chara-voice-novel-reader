import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

async def run_e2e_test():
    print("Starting E2E test...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: print(f"BROWSER ERROR: {err}"))

        errors = []
        page.on("console", lambda msg: print(f"  [{msg.type}] {msg.text}") if msg.type in ("error","warn") else None)

        print("Navigating to http://127.0.0.1:8000/")
        await page.goto("http://127.0.0.1:8000/")
        await page.wait_for_selector("#novelUrlInput")

        print("Step 1: Fetching novel...")
        await page.fill("#novelUrlInput", "https://ncode.syosetu.com/n0022gd/")
        await page.click("#fetchNovelBtn")

        try:
            await page.wait_for_selector("#novelTitle", state="visible", timeout=20000)
        except Exception:
            await page.screenshot(path=str(SCREENSHOT_DIR / "fail_step1.png"))
            raise RuntimeError("FAIL: Novel metadata did not load")

        print("  Title loaded OK")

        print("Step 1: Selecting episode 1...")
        # Wait for episode rows and click first
        await page.wait_for_selector(".episode-row", state="attached", timeout=10000)
        await page.locator(".episode-row").first.click()
        await page.wait_for_timeout(500)

        print("Step 1: Going to Step 2...")
        await page.click("#startAnalyzeBtn")

        try:
            await page.wait_for_selector("#analyzeBtn", state="visible", timeout=10000)
        except Exception:
            await page.screenshot(path=str(SCREENSHOT_DIR / "fail_step2.png"))
            raise RuntimeError("FAIL: Step 2 did not become visible")

        print("  Step 2 visible OK")

        print("Step 2: Running AI analysis (may take up to 90s)...")
        await page.click("#analyzeBtn")

        # Wait for loading overlay to DISAPPEAR first (API call done)
        try:
            await page.wait_for_selector(".loading-overlay.active", state="hidden", timeout=90000)
        except Exception:
            pass  # overlay may already be gone

        # Now wait for char-cards - use "attached" not "visible" since they may be off-screen
        try:
            await page.wait_for_selector(".char-card", state="attached", timeout=10000)
        except Exception:
            await page.screenshot(path=str(SCREENSHOT_DIR / "fail_analysis.png"))
            # Check for toast error
            toast_text = await page.locator(".toast.error").inner_text() if await page.locator(".toast.error").count() > 0 else "none"
            raise RuntimeError(f"FAIL: Char cards not found. Toast: {toast_text}")

        cards = await page.locator(".char-card").count()
        print(f"  {cards} characters loaded OK")
        await page.screenshot(path=str(SCREENSHOT_DIR / "step2_chars.png"))

        print("Step 2: Going to Step 3 (generateScript)...")
        # Take screenshot to see state after analysis
        await page.screenshot(path=str(SCREENSHOT_DIR / "after_analysis.png"))
        # Check if goToStep3Btn exists and its visibility
        btn_visible = await page.evaluate("() => { const b = document.getElementById('goToStep3Btn'); return b ? {visible: b.offsetParent !== null, classes: b.className, display: getComputedStyle(b).display} : null; }")
        print(f"  goToStep3Btn state: {btn_visible}")
        # Force click via JS regardless of visibility
        await page.evaluate("() => document.getElementById('goToStep3Btn').click()")


        # Wait for script generation loading to finish
        try:
            await page.wait_for_selector(".loading-overlay.active", state="hidden", timeout=90000)
        except Exception:
            pass

        try:
            await page.wait_for_selector("#synthesizeBtn", state="visible", timeout=15000)
        except Exception:
            await page.screenshot(path=str(SCREENSHOT_DIR / "fail_step3.png"))
            raise RuntimeError("FAIL: Step 3 (synthesize btn) not visible")

        print("  Step 3 visible OK")
        await page.screenshot(path=str(SCREENSHOT_DIR / "step3_ready.png"))

        print("Step 3: Starting synthesis...")
        await page.click("#synthesizeBtn")

        try:
            print("Step 3: Waiting for streaming playback to start...")
            await page.wait_for_selector(".script-line.playing", state="attached", timeout=60000)
            print("  Streaming playback started successfully!")
            
            print("Step 3: Waiting for synthesis to complete entirely (up to 5 mins)...")
            await page.wait_for_selector("#audioPlayerWrap.visible", state="attached", timeout=300000)
        except Exception:
            await page.screenshot(path=str(SCREENSHOT_DIR / "fail_synthesis.png"))
            raise RuntimeError("FAIL: Audio player did not appear after synthesis")

        print("  Audio player visible OK")
        await page.screenshot(path=str(SCREENSHOT_DIR / "success.png"))

        print("\n✅ ALL STEPS PASSED - E2E Test Complete!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
