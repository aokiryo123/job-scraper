import os
import asyncio
import pandas as pd
import gradio as gr
from playwright.async_api import async_playwright
import io

# --- PlaywrightのHuggingface Spaces対応 ---
# 起動時の読み込みブロックを防ぐため、初回のスクレイピング実行時に
# ブラウザのインストールを行うように変更
# （モジュールのトップレベルで実行するとタイムアウトの「Starting...」になるため）

CONCURRENT_LIMIT = 5
WAIT_TIME = 1

async def fetch_and_extract(context, url, selector, semaphore, index):
    """
    1つのURLから抽出を行い、結果の辞書を返す
    """
    async with semaphore:
        page = await context.new_page()
        # マニュアル・ステルス設定
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """)

        result_text = ""
        try:
            print(f"開始: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # セレクターが出るまで待機
            try:
                await page.wait_for_function(
                    f"document.querySelector('{selector}')?.innerText.trim().length > 0",
                    timeout=15000
                )
                if WAIT_TIME > 0:
                    await asyncio.sleep(WAIT_TIME)
                
                element = await page.query_selector(selector)
                if element:
                    result_text = await element.inner_text()
                    # 改行が含まれるとTSVが崩れるため、空白に置換
                    result_text = result_text.replace('\n', ' ').replace('\r', '').strip()
                else:
                    result_text = "(ERROR: Element not found)"
            except:
                result_text = "(ERROR: Timeout/Selector not found)"

            print(f"  取得成功: {url} -> {result_text[:30]}...")
            
        except Exception as e:
            result_text = f"(ERROR: {str(e)})"
            print(f"  失敗: {url}")
        finally:
            await page.close()
            
        return {"_index": index, "url": url, "selector": selector, "result": result_text}

async def scrape_from_text(input_tsv_text):
    """
    画面から入力されたTSVテキストを直接読み込み、スクレイピングを実行して
    結果をDataFrameとして返す
    """
    if not input_tsv_text.strip():
        yield "エラー: 入力データが空です。", ""
        return

    yield "ブラウザを起動しています...", ""

    try:
        # 文字列としてTSVを読み込み
        df_input = pd.read_csv(io.StringIO(input_tsv_text), sep='\t')
        df_input.columns = [c.strip() for c in df_input.columns]
        
        # 必須カラムのチェック
        if 'url' not in df_input.columns or 'selector' not in df_input.columns:
            yield "エラー: TSVには 'url' と 'selector' の列が必要です。", ""
            return
            
        records = df_input[['url', 'selector']].to_dict('records')
    except Exception as e:
        yield f"入力データの解析に失敗しました: {e}", ""
        return

    playwright = None
    browser = None
    
    try:
        playwright = await async_playwright().start()
        
        # Huggingface Spacesではサンドボックス無効化等のオプションが必要な場合があります
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-http2',
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
            locale='ja-JP',
            timezone_id='Asia/Tokyo',
            viewport={'width': 1920, 'height': 1080},
            extra_http_headers={
                'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'Upgrade-Insecure-Requests': '1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
            })
        
        semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
        
        tasks = [
            asyncio.create_task(fetch_and_extract(context, r['url'], r['selector'], semaphore, idx)) 
            for idx, r in enumerate(records)
        ]
        
        results = []
        total = len(tasks)
        
        # 件数が多い場合にUIが反応しなくなるのを防ぐため、即座に1回目のyieldを実施
        yield f"スクレイピング開始... (0/{total} 件完了)", ""

        for i, completed_task in enumerate(asyncio.as_completed(tasks), 1):
            res = await completed_task
            results.append(res)
            
            # 入力された順序（_index）でソートしてから出力する
            sorted_results = sorted(results, key=lambda x: x["_index"])
            
            # 出力用 DataFrame では _index 列を削除して作成
            df_current = pd.DataFrame([{k: v for k, v in r.items() if k != "_index"} for r in sorted_results])
            yield f"実行中... ({i}/{total} 件完了)", df_current.to_csv(sep='\t', index=False)

        sorted_results = sorted(results, key=lambda x: x["_index"])
        df_final = pd.DataFrame([{k: v for k, v in r.items() if k != "_index"} for r in sorted_results])
        yield "スクレイピングが完了しました。", df_final.to_csv(sep='\t', index=False)

    except Exception as e:
        df_error = pd.DataFrame(results) if 'results' in locals() else pd.DataFrame()
        yield f"予期せぬエラーが発生しました: {e}", df_error.to_csv(sep='\t', index=False) if not df_error.empty else ""

    finally:
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()

# --- Gradio UIの構築 ---
def define_ui():
    with gr.Blocks(title="Playwright Scraper App") as app:
        gr.Markdown("# WEB Scraper")
        gr.Markdown("URLとSelectorを入力し、自動取得した結果を確認・コピーできます。")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 入力データ (TSV形式)")
                gr.Markdown("1行目に `url` と `selector` をタブ区切りで記述し、2行目以降にデータを入力してください。")
                default_tsv = "url\tselector\nhttps://example.com\th1"
                input_textbox = gr.Textbox(
                    label="Input TSV", 
                    value=default_tsv, 
                    lines=10, 
                    elem_classes="text-monospace"
                )
                run_btn = gr.Button("スクレイピング実行", variant="primary")
                status_text = gr.Textbox(label="実行ステータス", interactive=False)
                
            with gr.Column():
                gr.Markdown("### 抽出結果")
                gr.Markdown("スクレイピングが完了すると、ここに結果のTSVが表示されます。")
                output_tsv = gr.Textbox(
                    label="Output",
                    info="テキストボックス内をクリックして全選択(Ctrl+A)＆コピー(Ctrl+C)してExcelやスプレッドシートに貼り付けてください。",
                    interactive=False,
                    lines=15,
                    elem_classes="text-monospace"
                )
                
        # ボタンクリック時に非同期関数を呼び出す
        run_btn.click(
            fn=scrape_from_text,
            inputs=[input_textbox],
            outputs=[status_text, output_tsv]
        )
        
    return app

demo = define_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
