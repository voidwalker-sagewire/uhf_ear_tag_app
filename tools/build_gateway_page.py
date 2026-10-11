"""Generate the dedicated Gateway entry point from the proven scanner."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
page = (root / "index.html").read_text(encoding="utf-8")
page = page.replace("<head>", '<head>\n<meta name="reaper-scout-route" content="gateway-rs-tx-007">', 1)
(root / "gateway.html").write_text(page, encoding="utf-8")
