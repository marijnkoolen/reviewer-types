"""Build a self-contained report/<name>.html from report/<name>.md: converts
via pandoc, embeds every referenced figure as a base64 data URI so the file
works standalone (no figures/ folder needed alongside it), and wraps it in a
themed page (light/dark, follows the dataviz skill's chart-chrome tokens).

Requires pandoc on PATH.

Usage: python src/build_report_html.py [report_name] [page_title]
  report_name defaults to "report" (report/report.md -> report/report.html)
Writes report/<report_name>.html
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
REPORT_NAME = sys.argv[1] if len(sys.argv) > 1 else "report"
PAGE_TITLE = sys.argv[2] if len(sys.argv) > 2 else "Structure of online book reviews: gold-set findings (Phase A)"
MD_PATH = REPORT_DIR / f"{REPORT_NAME}.md"
HTML_OUT = REPORT_DIR / f"{REPORT_NAME}.html"

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --page: #f9f9f7;
  --ink-primary: #0b0b0b;
  --ink-secondary: #52514e;
  --ink-muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --link: #2a78d6;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #1a1a19;
    --page: #0d0d0d;
    --ink-primary: #ffffff;
    --ink-secondary: #c3c2b7;
    --ink-muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --link: #3987e5;
    --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19;
  --page: #0d0d0d;
  --ink-primary: #ffffff;
  --ink-secondary: #c3c2b7;
  --ink-muted: #898781;
  --grid: #2c2c2a;
  --baseline: #383835;
  --link: #3987e5;
  --border: rgba(255,255,255,0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
}
.wrap {
  max-width: 880px;
  margin: 0 auto;
  padding: 48px 24px 96px;
}
h1, h2, h3 { line-height: 1.25; }
h1 { font-size: 1.7rem; margin-bottom: 0.3em; }
h2 {
  font-size: 1.25rem;
  margin-top: 2.4em;
  padding-top: 0.6em;
  border-top: 1px solid var(--grid);
}
h3 { font-size: 1.05rem; }
p, li { color: var(--ink-primary); }
em { color: var(--ink-secondary); }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.1em 0.4em;
  font-size: 0.9em;
}
hr { border: none; border-top: 1px solid var(--grid); margin: 2.5em 0; }
figure { margin: 1.5em 0; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.2em 0;
  font-size: 0.92rem;
}
th, td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid var(--grid);
}
th { color: var(--ink-secondary); font-weight: 600; }
tbody tr:hover { background: var(--surface); }
#TOC {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 24px;
  margin: 1.5em 0 2.5em;
}
#TOC ul { margin: 0.3em 0; padding-left: 1.3em; }
#TOC > ul { list-style: none; padding-left: 0; }
#TOC a { color: var(--ink-secondary); }
.theme-toggle {
  position: fixed;
  top: 16px;
  right: 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 0.85rem;
  cursor: pointer;
  color: var(--ink-secondary);
}
"""

TOGGLE_JS = """
(function () {
  var btn = document.querySelector('.theme-toggle');
  var root = document.documentElement;
  btn.addEventListener('click', function () {
    var current = root.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Switch to light' : 'Switch to dark';
  });
})();
"""


def embed_images(html, base_dir):
    def replace(match):
        src = match.group(1)
        if src.startswith("http") or src.startswith("data:"):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return match.group(0).replace(f'src="{src}"', f'src="data:image/png;base64,{data}"')

    return re.sub(r'<img [^>]*src="([^"]+)"[^>]*>', replace, html)


def rewrite_cross_report_links(html):
    """report.md <-> report_phase_c.md cross-links point at the sibling .md
    file (correct when reading the markdown source directly); rewrite to the
    sibling .html file for the built page, where that's what actually sits
    next to it."""
    return re.sub(r'href="(report(?:_phase_c)?)\.md(#[^"]*)?"', r'href="\1.html\2"', html)


def main():
    # --toc only takes effect in --standalone mode; extract the <body> inner
    # content back out so it can drop into our own page template below.
    result = subprocess.run(
        ["pandoc", str(MD_PATH), "-f", "markdown", "-t", "html5", "--standalone",
         "--syntax-highlighting=none", "--toc", "--toc-depth=2", "--section-divs=false"],
        capture_output=True, text=True, check=True,
    )
    match = re.search(r"<body>(.*)</body>", result.stdout, re.DOTALL)
    body = rewrite_cross_report_links(embed_images(match.group(1), REPORT_DIR))

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{PAGE_TITLE}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
</head>
<body>
<button class="theme-toggle">Switch to dark</button>
<div class="wrap">
{body}
</div>
<script>{TOGGLE_JS}</script>
</body>
</html>
"""
    HTML_OUT.write_text(page, encoding="utf-8")
    print(f"wrote {HTML_OUT} ({HTML_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
