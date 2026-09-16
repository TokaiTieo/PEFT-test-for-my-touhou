"""Fetch public THBWiki pages for inspection; never generate dialogue or call an LLM."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        if tag in {'p', 'li', 'h2', 'h3', 'h4', 'br', 'tr'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('titles', nargs='+')
    parser.add_argument('--cache', type=Path, default=Path('.cache/canon'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for title in args.titles:
        query = urllib.parse.urlencode({'action': 'parse', 'page': title, 'prop': 'text|revid', 'format': 'json'})
        request = urllib.request.Request('https://thwiki.cc/api.php?' + query, headers={'User-Agent': 'Touhou-NPC-canon-audit/1.0'})
        data = json.load(urllib.request.urlopen(request, timeout=40))
        if 'error' in data:
            print(title, data['error']['code'], flush=True)
            continue
        page = data['parse']
        html = page['text']['*']
        plain = PlainText()
        plain.feed(html)
        text = '\n'.join(line.strip() for line in ''.join(plain.parts).splitlines() if line.strip())
        record = {'requested_title': title, 'title': page['title'], 'revision': page['revid'], 'url': 'https://thwiki.cc/index.php?' + urllib.parse.urlencode({'title': page['title'], 'oldid': page['revid']}), 'sha256': hashlib.sha256(html.encode()).hexdigest(), 'text': text}
        target = args.cache / (title.replace('/', '_') + '.json')
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(page['title'], page['revid'], len(text), flush=True)


if __name__ == '__main__':
    main()
