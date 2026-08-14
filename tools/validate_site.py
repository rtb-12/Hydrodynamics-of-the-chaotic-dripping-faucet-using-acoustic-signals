#!/usr/bin/env python3
"""Structural checks on the static site. Run: python3 tools/validate_site.py

Guards the properties the site promises: it opens from the filesystem with no
server, every link resolves, every class is styled, every element the JS reaches
for exists, and nothing is fetched from the network.

Standard library only, so CI needs no install step.
"""

import os, re, sys, pathlib
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'

VOID = {'meta', 'link', 'br', 'hr', 'img', 'input', 'use', 'path', 'circle',
        'rect', 'line', 'source', 'marker', 'ellipse', 'polygon', 'polyline', 'stop'}


class TagBalance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f'stray </{tag}>')
        elif self.stack[-1] != tag:
            self.errors.append(f'</{tag}> closes <{self.stack[-1]}>')
        else:
            self.stack.pop()


def check(page, css_classes):
    text = page.read_text()
    body = re.sub(r'<script>[\s\S]*?</script>', '', text)   # markup checks skip JS
    problems = []

    parser = TagBalance()
    parser.feed(body)
    if parser.stack:
        problems.append(f'unclosed tags: {parser.stack}')
    problems += parser.errors[:3]

    ids = re.findall(r'\bid="([^"]+)"', body)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f'duplicate ids: {dupes}')

    fragments = {a[1:] for a in re.findall(r'href="(#[^"]+)"', body)}
    dead = sorted(fragments - set(ids))
    if dead:
        problems.append(f'dead same-page anchors: {dead}')

    for href in re.findall(r'href="([^"#][^"]*)"', body):
        if href.startswith(('http', 'mailto')):
            continue
        if not (page.parent / href.split('#')[0]).resolve().exists():
            problems.append(f'dead link: {href}')

    for href in re.findall(r'href="([^"]*#[^"]+)"', body):
        if href.startswith('#'):
            continue
        target, frag = href.split('#', 1)
        tp = (page.parent / target).resolve()
        if tp.exists() and f'id="{frag}"' not in tp.read_text():
            problems.append(f'dead cross-page anchor: {href}')

    for src in re.findall(r'src="([^"]+)"', body):
        if src.startswith(('http', 'data:')):
            continue
        if not (page.parent / src).resolve().exists():
            problems.append(f'missing asset: {src}')

    # Lookbehind so data-alt= and similar do not count as a real alt attribute.
    for img in re.findall(r'<img\b[^>]*>', body):
        if not re.search(r'(?<![-\w])alt\s*=\s*"[^"]', img):
            problems.append(f'image without alt text: {img[:70]}')

    used = set()
    for attr in re.findall(r'class="([^"]+)"', body):
        used.update(attr.split())
    unstyled = sorted(used - css_classes)
    if unstyled:
        problems.append(f'classes used but never styled: {unstyled}')

    for gid in sorted(set(re.findall(r"\$\('([^']+)'\)", text))):
        if f'id="{gid}"' not in text:
            problems.append(f'JS reaches for missing element: {gid}')

    for ext in re.findall(r'(?:src|href)="(https?://[^"]+)"', text):
        problems.append(f'external request breaks offline use: {ext}')

    return problems


def main():
    if not SITE.is_dir():
        print(f'no site directory at {SITE}')
        return 1
    css_classes = set(re.findall(r'\.([a-zA-Z][\w-]*)', (SITE / 'style.css').read_text()))
    pages = sorted(SITE.glob('*.html'))
    if not pages:
        print('no pages found')
        return 1

    failed = 0
    for page in pages:
        problems = check(page, css_classes)
        if problems:
            failed += 1
            print(f'FAIL  {page.name}')
            for p in problems:
                print(f'        {p}')
        else:
            print(f'ok    {page.name}')

    print(f'\n{len(pages)} pages, {failed} with problems')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
