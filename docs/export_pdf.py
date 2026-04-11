"""
Gera PDF do ML Canvas a partir do ml_canvas.html.

Uso:
    cd docs
    python export_pdf.py

Saída: docs/ml_canvas.pdf
"""

import http.server
import os
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8765
DOCS_DIR = Path(__file__).parent
OUTPUT = DOCS_DIR / "ml_canvas.pdf"


def start_server():
    os.chdir(DOCS_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # silencia logs
    server = http.server.HTTPServer(("localhost", PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    print("Iniciando servidor local...")
    server = start_server()
    time.sleep(1)

    url = f"http://localhost:{PORT}/ml_canvas.html"

    print(f"Abrindo canvas em {url} ...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        # Aguarda o canvas renderizar completamente
        page.wait_for_selector("#canvas-grid .section", timeout=8000)
        time.sleep(1)

        print(f"Exportando PDF para {OUTPUT} ...")
        page.pdf(
            path=str(OUTPUT),
            format="A3",
            landscape=True,
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        )
        browser.close()

    server.shutdown()
    print(f"PDF gerado com sucesso: {OUTPUT}")


if __name__ == "__main__":
    main()
