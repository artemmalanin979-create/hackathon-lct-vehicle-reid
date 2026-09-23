#!/usr/bin/env python3
"""Build the complete SOLUTION.md offline: python3 06-documentation/build_pdf.py.

Tested system tools (no installation/download during generation): Python 3.13,
Markdown 3.7, WeasyPrint 64.1 (pydyf 0.11.0), Pango 1.56.4, Fontconfig 2.16.0,
DejaVu Sans / DejaVu Sans Mono 2.37. Rendering uses installed Python packages,
system libraries and fonts, not the ReID runtime wheels. Poppler 25.02.0 is
used for validation.

The default output is SOLUTION.pdf next to this script. All Markdown content,
including any appendices, is rendered; source files are never rewritten.
HTTP(S) resource fetching is forbidden. Hyperlinks are preserved, with local
repository links made relative to the PDF so the checkout can be moved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit


CSS = """
@page {
  size: A4;
  margin: 16mm 15mm 18mm;
  @bottom-left {
    content: 'Vehicle ReID · SOLUTION.md';
    font-family: 'DejaVu Sans'; font-size: 7pt; color: #596270;
  }
  @bottom-right {
    content: counter(page) ' / ' counter(pages);
    font-family: 'DejaVu Sans'; font-size: 7pt; color: #596270;
  }
}
html { font-family: 'DejaVu Sans'; font-size: 9.4pt; line-height: 1.48; color: #18212c; }
body { margin: 0; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 7mm; }
h2 { font-size: 14pt; line-height: 1.28; margin: 8mm 0 3mm; }
h3 { font-size: 11pt; line-height: 1.3; margin: 5mm 0 2mm; }
h1, h2, h3, h4, h5, h6 { break-after: avoid; }
p { margin: 0 0 3mm; orphans: 3; widows: 3; overflow-wrap: anywhere; }
a { color: #14528a; text-decoration: underline; overflow-wrap: anywhere; }
code, pre { font-family: 'DejaVu Sans Mono'; font-variant-ligatures: none; }
code { font-size: 0.87em; overflow-wrap: anywhere; }
pre {
  font-size: 7.7pt; line-height: 1.45; white-space: pre-wrap;
  overflow-wrap: anywhere; padding: 3mm; margin: 3mm 0 4mm;
  background: #f2f4f6; border-left: 2pt solid #8998a8;
}
pre code { font-size: inherit; }
table { width: 100%; table-layout: fixed; border-collapse: collapse; margin: 3mm 0 5mm; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td {
  font-size: 7.8pt; line-height: 1.42; text-align: left; vertical-align: top;
  padding: 2.1mm 1.8mm; border: 0.5pt solid #cad1d8;
  overflow-wrap: anywhere;
}
th { background: #e8edf2; font-weight: bold; }
td code { font-size: 0.92em; }
blockquote { margin: 3mm 0; padding: 2mm 3mm; border-left: 2pt solid #8998a8; }
ul, ol { padding-left: 6mm; }
li { margin-bottom: 2mm; overflow-wrap: anywhere; }
img { max-width: 100%; height: auto; }
"""


def check_freshness(source: Path, output: Path) -> dict:
    """Read the PDF's own Subject, not a sidecar that can go stale separately."""
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    info = subprocess.run(["pdfinfo", "-enc", "UTF-8", str(output)],
                          capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})
    if info.returncode:
        raise ValueError(f"Cannot read PDF metadata: {info.stderr.strip()}")
    match = re.search(r"^Subject:\s+SOLUTION\.md SHA-256: ([0-9a-f]{64})\s*$",
                      info.stdout, re.MULTILINE)
    pdf_source_hash = match.group(1) if match else None
    return {"ok": pdf_source_hash == source_hash, "source": str(source),
            "output": str(output), "source_sha256": source_hash,
            "pdf_source_sha256": pdf_source_hash}


def preserve_table_text(source: str) -> str:
    """Keep surplus literal pipes in the last cell instead of dropping text.

    SOLUTION.md has a three-column row containing ``max |delta|``. Markdown's
    table extension would silently discard the part after the extra delimiter.
    No source text is removed; only literal delimiters are escaped for HTML.
    """
    result = []
    columns = None
    fenced = False
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
        if not fenced and line.startswith('|'):
            cells = re.split(r'(?<!\\)\|', line.strip().strip('|'))
            if columns is None:
                columns = len(cells)
            if len(cells) > columns:
                cells = cells[:columns - 1] + [r'\|'.join(cells[columns - 1:])]
                line = '|' + '|'.join(cells) + '|\n'
        else:
            columns = None
        result.append(line)
    return ''.join(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--source', type=Path, default=script_dir.parent / 'SOLUTION.md')
    parser.add_argument('--output', type=Path, default=script_dir / 'SOLUTION.pdf')
    parser.add_argument('--html', type=Path, help='Optionally save the rendered HTML for inspection')
    parser.add_argument('--check', action='store_true',
                        help='Check PDF freshness without rendering (requires pdfinfo / poppler-utils)')
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    if source == output or args.html and args.html.resolve() in (source, output):
        parser.error('Source, PDF and optional HTML must have distinct paths.')
    if args.check:
        try:
            result = check_freshness(source, output)
        except (OSError, ValueError) as exc:
            parser.error(f'{exc}. Freshness check requires pdfinfo (poppler-utils).')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['ok']:
            print('PDF устарел или не содержит SHA-256 источника. Пересоберите: '
                  'python3 06-documentation/build_pdf.py (с теми же --source/--output).',
                  file=sys.stderr)
        return 0 if result['ok'] else 2
    try:
        import markdown
        import pydyf
        from weasyprint import HTML, default_url_fetcher
    except ImportError as exc:
        parser.error(f'{exc}. Required: Markdown 3.7 and WeasyPrint 64.1; see script docstring.')

    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    body = markdown.markdown(preserve_table_text(raw.decode('utf-8')),
                             extensions=['tables', 'fenced_code', 'sane_lists', 'toc'])
    document_html = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<title>Vehicle ReID — сопроводительная документация</title>'
        f'<meta name="description" content="SOLUTION.md SHA-256: {source_hash}">'
        f'<style>{CSS}</style></head><body>{body}</body></html>')

    def offline_fetch(url, *positional, **kwargs):
        if urlsplit(url).scheme not in ('file', 'data'):
            raise ValueError(f'Network resource fetching is disabled: {url}')
        return default_url_fetcher(url, *positional, **kwargs)

    local_links = 0

    def portable_links(document, pdf):
        nonlocal local_links
        for obj in pdf.objects:
            if not isinstance(obj, pydyf.Dictionary) or obj.get('Subtype') != '/Link':
                continue
            action = obj.get('A')
            if not action or action.get('S') != '/URI':
                continue
            uri = urlsplit(action['URI'].string)
            if uri.scheme == 'file' and uri.netloc in ('', 'localhost'):
                target = Path(unquote(uri.path)).resolve()
                if target.is_relative_to(source.parent):
                    relative = quote(os.path.relpath(target, output.parent), safe='/')
                    if uri.query:
                        relative += '?' + uri.query
                    if uri.fragment:
                        relative += '#' + uri.fragment
                    action['URI'] = pydyf.String(relative)
                    local_links += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    document = HTML(string=document_html, base_url=source.parent.as_uri() + '/',
                    url_fetcher=offline_fetch).render()
    # A deterministic identifier, without build timestamps or machine paths.
    document.write_pdf(output, finisher=portable_links, pdf_identifier=source_hash.encode('ascii'))
    try:
        freshness = check_freshness(source, output)
    except (OSError, ValueError) as exc:
        parser.error(f'PDF written but freshness not verified: {exc}. Install poppler-utils.')
    if not freshness['ok']:
        parser.error('Source changed during generation or PDF metadata does not match; rebuild.')
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(document_html, encoding='utf-8')
    print(json.dumps({
        'source': str(source), 'source_sha256': source_hash,
        'output': str(output), 'bytes': output.stat().st_size,
        'pages': len(document.pages), 'relative_link_annotations': local_links,
        'pdf_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'freshness_verified': True,
        'versions': {name: importlib.metadata.version(name)
                     for name in ('Markdown', 'weasyprint', 'pydyf')},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
